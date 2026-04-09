#ifndef __SUPERBBLAS_ALLOC__
#define __SUPERBBLAS_ALLOC__

#include "cache.h"
#include "performance.h"
#include "platform.h"
#include <unordered_set>

namespace superbblas {

    namespace detail {
        /// is_complex<T>::value is true if T is std::complex
        /// \tparam T: type to inspect

        template <typename T> struct is_complex {
            static const bool value = false;
        };
        template <typename T> struct is_complex<std::complex<T>> {
            static const bool value = true;
        };
        template <typename T> struct is_complex<const T> {
            static const bool value = is_complex<T>::value;
        };

        /// Return a pointer aligned or nullptr if it isn't possible
        /// \param alignment: desired alignment of the returned pointer
        /// \param size: desired allocated size
        /// \param ptr: given pointer to align
        /// \param space: storage of the given pointer

        template <typename T>
        T *align(std::size_t alignment, std::size_t size, T *ptr, std::size_t space) {
            if (alignment == 0) return ptr;
            if (ptr == nullptr) return nullptr;

            T *r = nullptr;
            // std::align isn't in old versions of gcc
#if !defined(__GNUC__) || __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9)
            void *ptr0 = (void *)ptr;
            r = (T *)std::align(alignment, size, ptr0, space);
#else
            uintptr_t new_ptr = ((uintptr_t(ptr) + (alignment - 1)) & ~(alignment - 1));
            if (new_ptr + size - uintptr_t(ptr) > space)
                r = nullptr;
            else
                r = (T *)new_ptr;
#endif

            if (r == nullptr) throw std::runtime_error("align: fail to align pointer");
            return r;
        }

        /// Set default alignment, which is alignof(T) excepting when supporting GPUs that complex
        /// types need special alignment

        template <typename T> struct default_alignment {
            constexpr static std::size_t alignment = alignof(T);
        };

        /// NOTE: thrust::complex requires sizeof(complex<T>) alignment
#ifdef SUPERBBLAS_USE_GPU
        template <typename T> struct default_alignment<std::complex<T>> {
            constexpr static std::size_t alignment = sizeof(std::complex<T>);
        };
#endif

        /// Check the given pointer has proper alignment
        /// \param v: ptr to check

        template <typename T> void check_ptr_align(const void *ptr) {
            align<T>(default_alignment<T>::alignment, sizeof(T), (T *)ptr, sizeof(T));
        }

        /// Macro SUPERBBLAS_HIP_USE_ASYNC_ALLOC controls the use of the asynchronous memory allocation API.
        /// The API is available since ROCM 5.3 but we noticed problems in 5.4 and the ROCM doc
        /// still says it is in beta in version 5.7.

#if defined(SUPERBBLAS_USE_HIP)
#    if (HIP_VERSION_MAJOR > 5) || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 8)
#        define SUPERBBLAS_HIP_USE_ASYNC_ALLOC
#    endif
#endif

        /// Allocate memory on a device
        /// \param n: number of element of type `T` to allocate
        /// \param xpu: context
        /// \param external_use: whether the allocation will be used by other libraries such as MPI
        ///
        /// NOTE: MPI calls may fail when buffers were allocated with `cudaMallocAsync`

        template <typename T, typename XPU>
        T *allocate(std::size_t n, const XPU &xpu, bool external_use = false) {
            (void)external_use;

            // Shortcut for zero allocations
            if (n == 0) return nullptr;

            tracker<XPU> _t(std::string("allocating ") + platformToStr(xpu), xpu);

            // Do the allocation
            setDevice(xpu);
            causalConnectTo(getStream(xpu), getAllocStream(xpu));
            T *r = nullptr;
            for (int attempt = 0; attempt < 2; ++attempt) {
                try {
                    if (getCustomAllocator()) {
                        r = (T *)getCustomAllocator()(sizeof(T) * n,
                                                      deviceId(xpu) == CPU_DEVICE_ID ? CPU : GPU);
                    } else if (std::is_same<Cpu, XPU>::value) {
#ifdef SUPERBBLAS_USE_MPI
                        if (external_use) {
                            MPI_check(MPI_Alloc_mem(sizeof(T) * n, MPI_INFO_NULL, &r));
                        } else
#endif
                        {
                            // Allocate the array without calling constructors, specially useful for std::complex
                            r = (T *)::operator new(sizeof(T) * n);
                        }
                    }
#ifdef SUPERBBLAS_USE_GPU
                    else if (deviceId(xpu) == CPU_DEVICE_ID) {
#    ifdef SUPERBBLAS_USE_CUDA
                        gpuCheck(cudaHostAlloc(&r, sizeof(T) * n, cudaHostAllocPortable));
#    elif defined(SUPERBBLAS_USE_HIP)
                        gpuCheck(hipHostMalloc(&r, sizeof(T) * n,
                                               hipHostMallocPortable | hipHostMallocNonCoherent));
#    endif
                    } else {
#    ifdef SUPERBBLAS_USE_CUDA
#        if CUDART_VERSION >= 11020
                        if (!external_use) {
                            gpuCheck(cudaMallocAsync(&r, sizeof(T) * n, getAllocStream(xpu)));
                        } else
#        endif
                        {
                            gpuCheck(cudaMalloc(&r, sizeof(T) * n));
                        }
#    elif defined(SUPERBBLAS_USE_HIP)
#        ifdef SUPERBBLAS_HIP_USE_ASYNC_ALLOC
                        if (!external_use) {
                            gpuCheck(hipMallocAsync(&r, sizeof(T) * n, getAllocStream(xpu)));
                        } else
#        endif
                        {
                            gpuCheck(hipMalloc(&r, sizeof(T) * n));
                        }
#    endif
                    }
#endif // SUPERBBLAS_USE_GPU
                    if (r == nullptr) throw std::runtime_error("Memory allocation failed!");
                    break;
                } catch (...) {
                    if (attempt == 0) {
                        sync(xpu);
                        sync(getAllocStream(xpu));
                        syncLegacyStream(xpu);
                        clearInternalCaches(xpu);
                    } else {
                        if (getLogLevel() > 0) {
                            std::cerr << "superbblas::detail::allocate: error allocating "
                                      << sizeof(T) * n << " bytes on device " << deviceId(xpu)
                                      << (external_use ? " for external use" : "");
                            if (getTrackingMemory()) {
                                std::size_t gpu_free = 0, gpu_total = 0, gpu_extra = 0;
#ifdef SUPERBBLAS_USE_GPU
                                gpuCheck(SUPERBBLAS_GPU_SYMBOL(MemGetInfo)(&gpu_free, &gpu_total));
                                gpu_extra = getGpuBlasMemory(xpu);
#endif
                                std::cerr << "; superbblas mem usage: cpu "
                                          << getCpuMemUsed(0) / 1024 / 1024 << " MiB  gpu "
                                          << getGpuMemUsed(0) / 1024 / 1024 << " MiB, plus "
                                          << gpu_extra / 1024 / 1024
                                          << " MiB used by gpu blas lib; there is "
                                          << gpu_free / 1024 / 1024 << " MiB free out of "
                                          << gpu_total / 1024 / 1024;
                            }
                            std::cerr << std::endl;
                        }
                        throw;
                    }
                }
            }
            causalConnectTo(getAllocStream(xpu), getStream(xpu));

            // Annotate allocation
            if (getTrackingMemory()) {
                if (getAllocations(xpu.session).count((void *)r) > 0)
                    throw std::runtime_error("Ups! Allocator returned a pointer already in use");
                getAllocations(xpu.session)[(void *)r] = sizeof(T) * n;
                if (deviceId(xpu) >= 0)
                    getGpuMemUsed(xpu.session) += double(sizeof(T) * n);
                else
                    getCpuMemUsed(xpu.session) += double(sizeof(T) * n);
            }

            return r;
        }

        /// Deallocate memory on a device
        /// \param ptr: pointer to the memory to deallocate
        /// \param xpu: context
        /// \param external_use: whether the allocation will be used by other libraries such as MPI

        template <typename T, typename XPU>
        void deallocate(T *ptr, XPU xpu, bool external_use = false) {
            (void)external_use;

            // Shortcut for zero allocations
            if (!ptr) return;

            tracker<XPU> _t(std::string("deallocating ") + platformToStr(xpu), xpu);

            // Remove annotation
            if (getTrackingMemory() && getAllocations(xpu.session).count((void *)ptr) > 0) {
                const auto &it = getAllocations(xpu.session).find((void *)ptr);
                if (deviceId(xpu) >= 0)
                    getGpuMemUsed(xpu.session) -= double(it->second);
                else
                    getCpuMemUsed(xpu.session) -= double(it->second);
                getAllocations(xpu.session).erase(it);
            }

            // Deallocate the pointer
            setDevice(xpu);
            causalConnectTo(getStream(xpu), getAllocStream(xpu));
            if (getCustomDeallocator()) {
                getCustomDeallocator()((void *)ptr, deviceId(xpu) == CPU_DEVICE_ID ? CPU : GPU);
            } else if (std::is_same<Cpu, XPU>::value) {
#ifdef SUPERBBLAS_USE_MPI
                if (external_use) {
                    MPI_check(MPI_Free_mem(ptr));
                } else
#endif
                {
                    ::operator delete(ptr);
                }
            }
#ifdef SUPERBBLAS_USE_GPU
            else if (deviceId(xpu) == CPU_DEVICE_ID) {
                sync(getAllocStream(xpu));
                gpuCheck(SUPERBBLAS_GPU_SELECT(xxx, cudaFreeHost, hipHostFree)((void *)ptr));
            } else {
#    ifdef SUPERBBLAS_USE_CUDA
#        if CUDART_VERSION >= 11020
                if (!external_use) {
                    gpuCheck(cudaFreeAsync((void *)ptr, getAllocStream(xpu)));
                } else
#        endif
                {
                    sync(getAllocStream(xpu));
                    gpuCheck(cudaFree((void *)ptr));
                }
#    elif defined(SUPERBBLAS_USE_HIP)
#        ifdef SUPERBBLAS_HIP_USE_ASYNC_ALLOC
                if (!external_use) {
                    gpuCheck(hipFreeAsync((void *)ptr, getAllocStream(xpu)));
                } else
#        endif
                {
                    sync(getAllocStream(xpu));
                    gpuCheck(hipFree((void *)ptr));
                }
#    endif
            }
#endif // SUPERBBLAS_USE_GPU
        }

#ifdef SUPERBBLAS_HIP_USE_ASYNC_ALLOC
#    undef SUPERBBLAS_HIP_USE_ASYNC_ALLOC
#endif

        /// Return a memory allocation with at least n elements of type T
        /// \param n: number of elements of the allocation
        /// \param xpu: context
        /// \param alignment: pointer alignment
        /// \param external_use: whether the allocation will be used by other libraries such as MPI

        template <typename T, typename XPU>
        std::pair<T *, std::shared_ptr<char>> allocateResouce(std::size_t n, XPU xpu,
                                                              std::size_t alignment = 0,
                                                              bool external_use = false) {
            // Shortcut for zero allocations
            if (n == 0) return {nullptr, std::shared_ptr<char>()};

            using T_no_const = typename std::remove_const<T>::type;
            if (alignment == 0) alignment = default_alignment<T_no_const>::alignment;
            /// NOTE: it is unclear, but some GPU-aware MPI libraries do not like pointers to
            ///       buffers that are not the first element of an allocation
            if (deviceId(xpu) >= 0 && external_use) alignment = 0;
            T *ptr = allocate<T_no_const>(n + (alignment + sizeof(T) - 1) / sizeof(T), xpu,
                                          external_use);
            std::size_t size = (n + (alignment + sizeof(T) - 1) / sizeof(T)) * sizeof(T);
            T *ptr_aligned = align<T>(alignment, sizeof(T) * n, ptr, size);
            return {ptr_aligned, std::shared_ptr<char>((char *)ptr, [=](char *ptr) {
                        deallocate<T_no_const>((T_no_const *)ptr, xpu, external_use);
                    })};
        }

        inline std::unordered_set<char *> &getAllocatedBuffers(const Cpu &) {
            static std::unordered_set<char *> allocs(16);
            return allocs;
        }

#ifdef SUPERBBLAS_USE_GPU
        inline std::vector<std::unordered_set<char *>> &getAllocatedBuffersGpu() {
            static std::vector<std::unordered_set<char *>> allocs(getGpuDevicesCount() + 1,
                                                                  std::unordered_set<char *>(16));
            return allocs;
        }

        inline std::unordered_set<char *> &getAllocatedBuffers(const Gpu &xpu) {
            return getAllocatedBuffersGpu().at(deviceId(xpu) + 1);
        }
#endif

        inline void clearAllocatedBuffers() {
            getAllocatedBuffers(Cpu{0}).clear();
#ifdef SUPERBBLAS_USE_GPU
            for (auto &it : getAllocatedBuffersGpu()) it.clear();
#endif
        }

        /// Tag class for all `allocateBufferResouce`
        struct allocate_buffer_t {};

        struct AllocationEntry {
            std::size_t size;          // allocation size
            std::shared_ptr<char> res; // allocation resource
            int device;                // allocStream device
            bool external_use;         // whether is going to be used for third-party library
        };

        /// Return a memory allocation with at least n elements of type T
        /// \param n: number of elements of the allocation
        /// \param xpu: context
        /// \param alignment: pointer alignment
        /// \param external_use: whether the allocation will be used by other libraries such as MPI

        template <typename T, typename XPU>
        std::pair<T *, std::shared_ptr<char>> allocateBufferResouce(std::size_t n, XPU xpu,
                                                                    std::size_t alignment = 0,
                                                                    bool external_use = false) {

            // Shortcut for zero allocations
            if (n == 0) return {nullptr, std::shared_ptr<char>()};

            tracker<Cpu> _t(std::string("allocate buffer ") + platformToStr(xpu), Cpu{});

            // Get alignment and the worst case size to adjust for alignment
            if (alignment == 0) alignment = default_alignment<T>::alignment;
            /// NOTE: it is unclear, but some GPU-aware MPI libraries do not like pointers to
            ///       buffers that are not the first element of an allocation
            if (deviceId(xpu) >= 0 && external_use) alignment = 0;
            std::size_t size = (n + (alignment + sizeof(T) - 1) / sizeof(T)) * sizeof(T);

            // Look for the smallest free allocation that can hold the requested size.
            // Also, update `getAllocatedBuffers` by removing the buffers not longer in cache.
            // We take extra care for the fake gpu allocations (the ones with device == CPU_DEVICE_ID):
            // we avoid sharing allocations for different backup devices. It should work without this hack,
            // but it avoids correlation between different devices.
            auto cache =
                getCache<char *, AllocationEntry, std::hash<char *>, allocate_buffer_t>(xpu);
            auto &all_buffers = getAllocatedBuffers(xpu);
            std::vector<char *> buffers_to_remove;
            std::size_t selected_buffer_size = std::numeric_limits<std::size_t>::max();
            std::shared_ptr<char> selected_buffer;
            for (char *buffer_ptr : all_buffers) {
                auto it = cache.find(buffer_ptr);
                if (it == cache.end()) {
                    buffers_to_remove.push_back(buffer_ptr);
                } else if (it->second.value.device == backupDeviceId(xpu) &&
                           it->second.value.external_use == external_use &&
                           it->second.value.res.use_count() == 1 && it->second.value.size >= size &&
                           it->second.value.size < selected_buffer_size) {
                    selected_buffer_size = it->second.value.size;
                    selected_buffer = it->second.value.res;
                }
            }
            for (char *buffer_ptr : buffers_to_remove) all_buffers.erase(buffer_ptr);

            // If no suitable buffer was found, create a new one and cache it
            if (!selected_buffer) {
                selected_buffer = allocateResouce<T>(n, xpu, alignment, external_use).second;
                selected_buffer_size = size;
                all_buffers.insert(selected_buffer.get());
                cache.insert(
                    selected_buffer.get(),
                    AllocationEntry{size, selected_buffer, backupDeviceId(xpu), external_use},
                    size);
            }

            // Connect the allocation stream with the current stream and make sure to connect back as soon as
            // the caller finishes using the buffer
            GpuStream stream = getStream(xpu), allocStream = getAllocStream(xpu);
            int device = backupDeviceId(xpu);
            setDevice(xpu);
            causalConnectTo(allocStream, stream);
            auto return_buffer = std::shared_ptr<char>(
                selected_buffer.get(), [stream, allocStream, selected_buffer, device](char *) {
                    if (stream != allocStream) {
                        setDevice(device);
                        causalConnectTo(stream, allocStream);
                    }
                });

            // Align and return the buffer
            T *ptr_aligned = align<T>(alignment, sizeof(T) * n, (T *)selected_buffer.get(),
                                      selected_buffer_size);
            return {ptr_aligned, return_buffer};
        }
    }

    /// Allocate memory
    /// \param n: number of element of type `T` to allocate
    /// \param ctx: context

    template <typename T> T *allocate(std::size_t n, Context ctx) {
        switch (ctx.plat) {
        case CPU: return detail::allocate<T>(n, ctx.toCpu(0));
#ifdef SUPERBBLAS_USE_GPU
        case CUDA: // Do the same as with HIP
        case HIP: return detail::allocate<T>(n, ctx.toGpu(0));
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }

    /// Deallocate memory
    /// \param ptr: pointer to the memory to deallocate
    /// \param ctx: context

    template <typename T> void deallocate(T *ptr, Context ctx) {
        switch (ctx.plat) {
        case CPU: detail::deallocate(ptr, ctx.toCpu(0)); break;
#ifdef SUPERBBLAS_USE_GPU
        case CUDA: // Do the same as with HIP
        case HIP: detail::deallocate(ptr, ctx.toGpu(0)); break;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }

    /// Allocate memory fast from the cache
    /// \param n: number of element of type `T` to allocate
    /// \param ctx: context

    template <typename T> std::shared_ptr<char> allocate_from_cache(std::size_t n, Context ctx) {
        switch (ctx.plat) {
        case CPU: return detail::allocateBufferResouce<T>(n, ctx.toCpu(0)).second;
#ifdef SUPERBBLAS_USE_GPU
        case CUDA: // Do the same as with HIP
        case HIP: return detail::allocateBufferResouce<T>(n, ctx.toGpu(0)).second;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }

    /// Clear all internal caches
    inline void clearCaches() {
        detail::destroyInternalCaches();
        detail::clearAllocatedBuffers();
    }
}

#endif // __SUPERBBLAS_ALLOC__
