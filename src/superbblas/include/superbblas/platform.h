#ifndef __SUPERBBLAS_PLATFORM__
#define __SUPERBBLAS_PLATFORM__

#include "superbblas_lib.h"
#include <complex>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef SUPERBBLAS_USE_MPI
#    include "mpi.h"
#endif

#ifdef __CUDACC__
#    define __HOST__ __host__
#    define __DEVICE__ __device__
#    ifndef SUPERBBLAS_USE_CUDA
#        define SUPERBBLAS_USE_CUDA
#    endif
#elif defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
#    define __HOST__ __host__
#    define __DEVICE__ __device__
#    ifndef SUPERBBLAS_USE_HIP
#        define SUPERBBLAS_USE_HIP
#    endif
#else
#    define __HOST__
#    define __DEVICE__
#endif

#ifdef SUPERBBLAS_CREATING_FLAGS
#    if defined(SUPERBBLAS_USE_CUDA)
EMIT_define(SUPERBBLAS_USE_CUDA)
#    elif defined(SUPERBBLAS_USE_HIP)
EMIT_define(SUPERBBLAS_USE_HIP)
#    else
EMIT_define(SUPERBBLAS_NOT_USE_GPU)
#    endif
#    ifdef SUPERBBLAS_USE_MKL
EMIT_define(SUPERBBLAS_USE_MKL)
#    endif
#endif

#ifdef SUPERBBLAS_NOT_USE_GPU
#    ifdef SUPERBBLAS_USE_CUDA
#        undef SUPERBBLAS_USE_CUDA
#    endif
#    ifdef SUPERBBLAS_USE_HIP
#        undef SUPERBBLAS_USE_HIP
#    endif
#endif

#if !defined(SUPERBBLAS_CREATING_FLAGS) && !defined(SUPERBBLAS_CREATING_LIB)
#    ifdef SUPERBBLAS_USE_CUDA
#        include <cublas_v2.h>
#        include <cuda_runtime.h>
#        include <cusolverDn.h>
#        include <cusparse.h>
#    endif

#    ifdef SUPERBBLAS_USE_HIP
#        include <hip/hip_runtime_api.h>
#        include <hipsparse/hipsparse.h>
#        include <rocblas/rocblas.h>
#        include <rocsolver/rocsolver.h>
#    endif
#endif // SUPERBBLAS_CREATING_FLAGS

#if defined(SUPERBBLAS_USE_CUDA) || defined(SUPERBBLAS_USE_HIP)
#    define SUPERBBLAS_USE_GPU
#endif

#if defined(SUPERBBLAS_USE_GPU) && !defined(SUPERBBLAS_CREATING_FLAGS) &&                          \
    !defined(SUPERBBLAS_CREATING_LIB) && !defined(SUPERBBLAS_LIB)
#    define SUPERBBLAS_GENERATE_KERNELS
#endif

#define SUPERBBLAS_CONCATX(a, b) a##b
#define SUPERBBLAS_CONCAT(a, b) SUPERBBLAS_CONCATX(a, b)

#ifdef SUPERBBLAS_USE_CUDA
#    define SUPERBBLAS_GPU_SELECT(X, Y, Z) Y
#elif defined(SUPERBBLAS_USE_HIP)
#    define SUPERBBLAS_GPU_SELECT(X, Y, Z) Z
#else
#    define SUPERBBLAS_GPU_SELECT(X, Y, Z) X
#endif

#define SUPERBBLAS_GPU_SYMBOL(X) SUPERBBLAS_CONCAT(SUPERBBLAS_GPU_SELECT(xxx, cuda, hip), X)
#define SUPERBBLAS_GPUBLAS_SYMBOL(X)                                                               \
    SUPERBBLAS_CONCAT(SUPERBBLAS_GPU_SELECT(xxx, cublas, rocblas), X)
#define SUPERBBLAS_GPUSPARSE_SYMBOL(X)                                                             \
    SUPERBBLAS_CONCAT(SUPERBBLAS_GPU_SELECT(xxx, cusparse, hipsparse), X)
#define SUPERBBLAS_GPUSOLVER_SYMBOL(X)                                                             \
    SUPERBBLAS_CONCAT(SUPERBBLAS_GPU_SELECT(xxx, cusolverDn, rocsolver), X)

#define SB_RESTRICT __restrict__

namespace superbblas {

    /// Where the data is

    enum platform {
        CPU,  ///< tradicional CPUs
        CUDA, ///< NVIDIA CUDA
        HIP   ///< AMD GPU
    };

    /// Default value in `Context`

    constexpr int CPU_DEVICE_ID = -1;

    /// Default GPU platform
    const platform GPU = SUPERBBLAS_GPU_SELECT(platform::CPU, platform::CUDA, platform::HIP);

    /// Function to allocate memory
    using Allocator = std::function<void *(std::size_t, enum platform)>;

    /// Function to deallocate memory
    using Deallocator = std::function<void(void *, enum platform)>;

    /// Cache session
    using Session = unsigned int;

    /// Return the custom allocator

    inline Allocator &getCustomAllocator() {
        static Allocator alloc{};
        return alloc;
    }

    /// Return the custom deallocator

    inline Deallocator &getCustomDeallocator() {
        static Deallocator dealloc{};
        return dealloc;
    }

    inline unsigned int getGpuDevicesCount();

    /// Platform and device information of data

    namespace detail {

        /// Datatype to represent a stream
        using GpuStream = SUPERBBLAS_GPU_SELECT(int, cudaStream_t, hipStream_t);

        /// Datatype to represent a cuda/hip runtime error
        using GpuError = SUPERBBLAS_GPU_SELECT(int, cudaError_t, hipError_t);

        /// Datatype to represent a cublas/rocblas error
        using GpuBlasError = SUPERBBLAS_GPU_SELECT(int, cublasStatus_t, rocblas_status);

        /// Datatype to represent a cusparse/hipsparse error
        using GpuSparseError = SUPERBBLAS_GPU_SELECT(int, cusparseStatus_t, hipsparseStatus_t);

        /// Datatype to represent a cusolver/rocsolver error
        using GpuSolverError = SUPERBBLAS_GPU_SELECT(int, cusolverStatus_t, rocblas_status);

        /// Datatype to represent the cublas/rocblas handle
        using GpuBlasHandle = SUPERBBLAS_GPU_SELECT(int, cublasHandle_t, rocblas_handle);

        /// Datatype to represent the cusparse/hipsparse handle
        using GpuSparseHandle = SUPERBBLAS_GPU_SELECT(int, cusparseHandle_t, hipsparseHandle_t);

        /// Datatype to represent the cusolver/rocsolver handle
        using GpuSolverHandle = SUPERBBLAS_GPU_SELECT(int, cusolverDnHandle_t, rocblas_handle);

        /// Low-level Cpu context

        struct Cpu {
            /// Cache session
            Session session;

            /// Return a CPU context with the same session
            Cpu toCpu() const { return *this; }

            /// Create a new context but with a cpu device
            Cpu toCpuPinned() const { return *this; }

            Cpu(const Session &session = 0) : session(session) {}
        };

#ifdef SUPERBBLAS_USE_GPU
        /// Low-level Gpu context

        struct SUPERBBLAS_GPU_SELECT(void, Cuda, Hip) {
            // Gpu device index, it may be CPU_DEVICE_ID
            int device;

            // Associated gpu device index if `device` is CPU_DEVICE_ID
            int backup_device;

            // Operation's stream
            GpuStream stream;

            // Allocation stream
            GpuStream alloc_stream;

            /// Cache session
            Session session;

            /// Return a CPU context with the same session
            Cpu toCpu() const { return Cpu{session}; }

            /// Create a new context but with a different stream
            SUPERBBLAS_GPU_SELECT(void, Cuda, Hip) withNewStream(GpuStream new_stream) const {
                return {device, backup_device, new_stream, alloc_stream, session};
            }

            /// Create a new context but with a cpu device
            SUPERBBLAS_GPU_SELECT(void, Cuda, Hip) toCpuPinned() const {
                return {CPU_DEVICE_ID, backup_device, stream, alloc_stream, session};
            }
        };
#endif // SUPERBBLAS_USE_GPU

        /// Type for a gpu context
        using Gpu = SUPERBBLAS_GPU_SELECT(void, Cuda, Hip);

        /// Throw exception if the given gpu runtime error isn't success
        /// \param err: gpu error

        inline void gpuCheck(GpuError err) {
#ifdef SUPERBBLAS_USE_CUDA
            if (err != cudaSuccess) {
                std::stringstream s;
                s << "CUDA error: " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err);
                throw std::runtime_error(s.str());
            }
#elif defined(SUPERBBLAS_USE_HIP)
            if (err != hipSuccess) {
                std::stringstream s;
                s << "HIP error: " << hipGetErrorName(err) << ": " << hipGetErrorString(err);
                throw std::runtime_error(s.str());
            }
#else
            // Do nothing
            (void)err;
#endif
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Return a device identification associated to the contex
        /// \param xpu: context

        inline int deviceId(Gpu xpu) { return xpu.device; }

        /// Return `deviceId` if the device index is gpu, otherwise return backup_device
        /// \param xpu: context

        inline int backupDeviceId(const Gpu &xpu) {
            return deviceId(xpu) == CPU_DEVICE_ID ? xpu.backup_device : deviceId(xpu);
        }
#endif

        inline int deviceId(const Cpu &) { return CPU_DEVICE_ID; }
        inline int backupDeviceId(const Cpu &) { return CPU_DEVICE_ID; }

        /// Set the current runtime device as the given one
        /// \param device: gpu device index to set as current
        ///
        /// NOTE: it does nothing on a cpu compilation

        inline void setDevice(int device) {
#ifdef SUPERBBLAS_USE_GPU
            if (device < 0) throw std::runtime_error("setDevice: invalid device index");
            int currentDevice;
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(GetDevice)(&currentDevice));
            if (currentDevice != device) gpuCheck(SUPERBBLAS_GPU_SYMBOL(SetDevice)(device));
#else
            // Do nothing
            (void)device;
#endif
        }

        /// Set the current runtime device as the given one
        /// \param ctx: context

#ifdef SUPERBBLAS_USE_GPU
        inline void setDevice(const Gpu &xpu) { setDevice(backupDeviceId(xpu)); }
#endif

        inline void setDevice(const Cpu &) {}

        /// Return a string identifying the platform
        /// \param xpu: context

        inline std::string platformToStr(const Cpu &) { return "cpu"; }

#ifdef SUPERBBLAS_USE_GPU
        inline std::string platformToStr(const Gpu &gpu) {
            return deviceId(gpu) == CPU_DEVICE_ID ? "host"
                                                  : SUPERBBLAS_GPU_SELECT("", "cuda", "hip");
        }
#endif

#ifdef SUPERBBLAS_USE_GPU
        /// Create a new stream
        /// \param device: device on which to stream will live

        inline GpuStream createStream(int device) {
            setDevice(device);
            GpuStream stream;
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(StreamCreate)(&stream));
            return stream;
        }

        /// Create a new stream
        /// \param xpu: device on which to stream will live

        inline GpuStream createStream(const Gpu &xpu) { return createStream(backupDeviceId(xpu)); }

        /// Destroy stream
        /// \param device: device on which to stream lives
        /// \param stream: stream to destroy

        inline void destroyStream(int device, GpuStream stream) {
            setDevice(device);
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(StreamDestroy)(stream));
        }

        /// Destroy stream
        /// \param xpu: device on which to stream lives
        /// \param stream: stream to destroy

        inline void destroyStream(const Gpu &xpu, GpuStream stream) {
            destroyStream(backupDeviceId(xpu), stream);
        }

        /// Return the associated stream
        /// \param xpu: context
        inline GpuStream getStream(const Gpu &xpu) { return xpu.stream; }

        /// Return the associated stream for allocating
        /// \param xpu: context

        inline GpuStream getAllocStream(const Gpu &xpu) { return xpu.alloc_stream; }

        /// NOTE: defined at `blas.h`

        inline void sync(GpuStream stream);

        /// Wait until everything finishes in the given context
        /// \param xpu: context

        inline void sync(const Gpu &xpu) {
            setDevice(xpu);
            sync(getStream(xpu));
        }

        /// Return the total memory available in a device
        /// \param xpu: context

        inline std::size_t totalGpuMemory(int device) {
            setDevice(device);
            std::size_t free = 0, total = 0;
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(MemGetInfo)(&free, &total));
            return total;
        }

        /// NOTE: defined at `blas.h`

        inline void syncLegacyStream(const Gpu &xpu);
#else

        inline std::size_t totalGpuMemory(int) { return 0; }

#endif // SUPERBBLAS_USE_GPU

        inline GpuStream createStream(const Cpu &) { return 0; }
        inline void destroyStream(const Cpu &, GpuStream) {}
        inline GpuStream getStream(const Cpu &) { return 0; }
        inline GpuStream getAllocStream(const Cpu &) { return 0; }
        inline void sync(const Cpu &) {}
        inline void syncLegacyStream(const Cpu &) {}

        /// Force the second stream to wait until everything finishes until now from
        /// the first stream.
        /// \param s0: first stream
        /// \param s1: second stream

        inline void causalConnectTo(GpuStream s0, GpuStream s1) {
            // Trivial case: do nothing when both are the same stream
            if (s0 == s1) return;

                // Otherwise, record an event on s0 and wait on s1
#ifdef SUPERBBLAS_USE_GPU
            SUPERBBLAS_GPU_SYMBOL(Event_t) ev;
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventCreateWithFlags)(
                &ev, SUPERBBLAS_GPU_SYMBOL(EventDisableTiming)));
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventRecord)(ev, s0));
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(StreamWaitEvent)(s1, ev, 0));
            gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventDestroy)(ev));
#else
            throw std::runtime_error("causalConnectTo: invalid operation!");
#endif
        }

        /// Force the second context to wait until everything finishes until now from
        /// the first context.
        /// \param xpu0: first context
        /// \param xpu1: second context

        template <typename XPU1> void causalConnectTo(const Cpu &, const XPU1 &) {
            // Trivial case: do noting when the first context is on cpu
        }

        template <typename XPU0,
                  typename std::enable_if<!std::is_same<XPU0, Cpu>::value, bool>::type = true>
        void causalConnectTo(const XPU0 &xpu0, const Cpu &) {
            // Trivial case: sync the first context when the second is on cpu
            sync(xpu0);
        }

#ifdef SUPERBBLAS_USE_GPU
        inline void causalConnectTo(const Gpu &xpu0, const Gpu &xpu1) {
            setDevice(xpu0);
            causalConnectTo(getStream(xpu0), getStream(xpu1));
        }
#endif // SUPERBBLAS_USE_GPU

        /// Return the device in which the pointer was allocated
        /// \param x: pointer to inspect

        inline int getPtrDevice(const void *x) {
#ifdef SUPERBBLAS_USE_CUDA
            struct cudaPointerAttributes ptr_attr;
            if (cudaPointerGetAttributes(&ptr_attr, x) != cudaSuccess) return CPU_DEVICE_ID;

#    if CUDART_VERSION >= 10000
            if (ptr_attr.type == cudaMemoryTypeUnregistered || ptr_attr.type == cudaMemoryTypeHost)
                return CPU_DEVICE_ID;
#    else
            if (!ptr_attr.isManaged && ptr_attr.memoryType == cudaMemoryTypeHost)
                return CPU_DEVICE_ID;
#    endif
            return ptr_attr.device;

#elif defined(SUPERBBLAS_USE_HIP)
            struct hipPointerAttribute_t ptr_attr;
            if (hipPointerGetAttributes(&ptr_attr, x) != hipSuccess) return CPU_DEVICE_ID;

#    if HIP_VERSION_MAJOR >= 6
            if (ptr_attr.type != hipMemoryTypeDevice) return CPU_DEVICE_ID;
#    else
            if (ptr_attr.memoryType != hipMemoryTypeDevice) return CPU_DEVICE_ID;
#    endif
            return ptr_attr.device;

#else
            (void)x;
            return CPU_DEVICE_ID;
#endif
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Return all the streams allowed to use as input/output data in superbblas calls

        inline std::vector<std::shared_ptr<GpuStream>> &getGpuAllocStreams() {
            static std::vector<std::shared_ptr<GpuStream>> h(getGpuDevicesCount());
            return h;
        }

        /// Return the gpu stream for a given device
        /// \param device: device index

        inline GpuStream getGpuAllocStream(int device) {
            auto h = getGpuAllocStreams().at(device);
            if (!h) {
                getGpuAllocStreams()[device] = h =
                    std::shared_ptr<GpuStream>(new GpuStream, [=](GpuStream *p) {
                        destroyStream(device, *p);
                        delete p;
                    });
                *h = createStream(device);
            }
            return *h;
        }
#endif

        /// Throw an error if the gpu blas status isn't success
        /// \param status: gpu blas error

        inline void gpuBlasCheck(GpuBlasError status) {
#ifdef SUPERBBLAS_USE_CUDA
            if (status != CUBLAS_STATUS_SUCCESS) {
                const char *err = "(unknown error code)";
#    if CUDART_VERSION >= 11400
                err = cublasGetStatusName(status);
#    else
                // clang-format off
                if (status == CUBLAS_STATUS_SUCCESS         ) err = "CUBLAS_STATUS_SUCCESS";
                if (status == CUBLAS_STATUS_NOT_INITIALIZED ) err = "CUBLAS_STATUS_NOT_INITIALIZED";
                if (status == CUBLAS_STATUS_ALLOC_FAILED    ) err = "CUBLAS_STATUS_ALLOC_FAILED";
                if (status == CUBLAS_STATUS_INVALID_VALUE   ) err = "CUBLAS_STATUS_INVALID_VALUE";
                if (status == CUBLAS_STATUS_ARCH_MISMATCH   ) err = "CUBLAS_STATUS_ARCH_MISMATCH";
                if (status == CUBLAS_STATUS_MAPPING_ERROR   ) err = "CUBLAS_STATUS_MAPPING_ERROR";
                if (status == CUBLAS_STATUS_EXECUTION_FAILED) err = "CUBLAS_STATUS_EXECUTION_FAILED";
                if (status == CUBLAS_STATUS_INTERNAL_ERROR  ) err = "CUBLAS_STATUS_INTERNAL_ERROR";
                if (status == CUBLAS_STATUS_NOT_SUPPORTED   ) err = "CUBLAS_STATUS_NOT_SUPPORTED";
                if (status == CUBLAS_STATUS_LICENSE_ERROR   ) err = "CUBLAS_STATUS_LICENSE_ERROR";
                    // clang-format on
#    endif
                std::stringstream s;
                s << "CUBLAS error: " << err;
                throw std::runtime_error(s.str());
            }

#elif defined(SUPERBBLAS_USE_HIP)
            if (status != rocblas_status_success) {
                const char *err = rocblas_status_to_string(status);
                std::stringstream s;
                s << "ROCBLAS error: " << err;
                throw std::runtime_error(s.str());
            }

#else
            // Do nothing
            (void)status;
#endif
        }

        /// Throw an error if the gpu sparse status isn't success
        /// \param status: gpu sparse error

        inline void gpuSparseCheck(GpuSparseError status) {
#ifdef SUPERBBLAS_USE_CUDA
            if (status != CUSPARSE_STATUS_SUCCESS) {
                std::string str = "(unknown error code)";
                // clang-format off
                if (status == CUSPARSE_STATUS_NOT_INITIALIZED          ) str = "CUSPARSE_STATUS_NOT_INITIALIZED";
                if (status == CUSPARSE_STATUS_ALLOC_FAILED             ) str = "CUSPARSE_STATUS_ALLOC_FAILED";
                if (status == CUSPARSE_STATUS_INVALID_VALUE            ) str = "CUSPARSE_STATUS_INVALID_VALUE";
                if (status == CUSPARSE_STATUS_ARCH_MISMATCH            ) str = "CUSPARSE_STATUS_ARCH_MISMATCH";
                if (status == CUSPARSE_STATUS_EXECUTION_FAILED         ) str = "CUSPARSE_STATUS_EXECUTION_FAILED";
                if (status == CUSPARSE_STATUS_INTERNAL_ERROR           ) str = "CUSPARSE_STATUS_INTERNAL_ERROR";
                if (status == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) str = "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                if (status == CUSPARSE_STATUS_NOT_SUPPORTED            ) str = "CUSPARSE_STATUS_NOT_SUPPORTED";
                if (status == CUSPARSE_STATUS_INSUFFICIENT_RESOURCES   ) str = "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
                // clang-format on

                std::stringstream ss;
                ss << "cuSparse function returned error " << str;
                throw std::runtime_error(ss.str());
            }

#elif defined(SUPERBBLAS_USE_HIP)
            if (status != HIPSPARSE_STATUS_SUCCESS) {
                std::string str = "(unknown error code)";
                // clang-format off
                if (status == HIPSPARSE_STATUS_NOT_INITIALIZED          ) str = "HIPSPARSE_STATUS_NOT_INITIALIZED";
                if (status == HIPSPARSE_STATUS_ALLOC_FAILED             ) str = "HIPSPARSE_STATUS_ALLOC_FAILED";
                if (status == HIPSPARSE_STATUS_INVALID_VALUE            ) str = "HIPSPARSE_STATUS_INVALID_VALUE";
                if (status == HIPSPARSE_STATUS_ARCH_MISMATCH            ) str = "HIPSPARSE_STATUS_ARCH_MISMATCH";
                if (status == HIPSPARSE_STATUS_MAPPING_ERROR            ) str = "HIPSPARSE_STATUS_MAPPING_ERROR";
                if (status == HIPSPARSE_STATUS_EXECUTION_FAILED         ) str = "HIPSPARSE_STATUS_EXECUTION_FAILED";
                if (status == HIPSPARSE_STATUS_INTERNAL_ERROR           ) str = "HIPSPARSE_STATUS_INTERNAL_ERROR";
                if (status == HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) str = "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                if (status == HIPSPARSE_STATUS_ZERO_PIVOT               ) str = "HIPSPARSE_STATUS_ZERO_PIVOT";
                if (status == HIPSPARSE_STATUS_NOT_SUPPORTED            ) str = "HIPSPARSE_STATUS_NOT_SUPPORTED";
                if (status == HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES   ) str = "HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES";
                // clang-format on

                std::stringstream ss;
                ss << "hipSPARSE function returned error " << str;
                throw std::runtime_error(ss.str());
            }
#else
            // Do nothing
            (void)status;
#endif
        }

        /// Throw an error if the gpu solver status isn't success
        /// \param status: gpu solver error

        inline void gpuSolverCheck(GpuSolverError status) {
#ifdef SUPERBBLAS_USE_CUDA
            if (status != CUSOLVER_STATUS_SUCCESS) {
                std::string str = "(unknown error code)";
                // clang-format off
                if (status == CUSOLVER_STATUS_NOT_INITIALIZED          ) str = "CUSOLVER_STATUS_NOT_INITIALIZED";
                if (status == CUSOLVER_STATUS_ALLOC_FAILED             ) str = "CUSOLVER_STATUS_ALLOC_FAILED";
                if (status == CUSOLVER_STATUS_INVALID_VALUE            ) str = "CUSOLVER_STATUS_INVALID_VALUE";
                if (status == CUSOLVER_STATUS_ARCH_MISMATCH            ) str = "CUSOLVER_STATUS_ARCH_MISMATCH";
                if (status == CUSOLVER_STATUS_EXECUTION_FAILED         ) str = "CUSOLVER_STATUS_EXECUTION_FAILED";
                if (status == CUSOLVER_STATUS_INTERNAL_ERROR           ) str = "CUSOLVER_STATUS_INTERNAL_ERROR";
                if (status == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) str = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                // clang-format on
                std::stringstream ss;
                ss << "cuSolver function returned error " << str;
                throw std::runtime_error(ss.str());
            }

#elif defined(SUPERBBLAS_USE_HIP)
            gpuBlasCheck(status);
#else
            // Do nothing
            (void)status;
#endif
        }

        inline std::size_t getGpuBlasMemory(const Cpu &) { return 0; }

#ifdef SUPERBBLAS_USE_GPU
        /// Return all gpu blas handles for all devices

        inline std::vector<std::shared_ptr<GpuBlasHandle>> &getGpuBlasHandles() {
            static std::vector<std::shared_ptr<GpuBlasHandle>> h(getGpuDevicesCount());
            return h;
        }

        /// Return the gpu blas handle for the given context
        /// \param xpu: context

        inline GpuBlasHandle getGpuBlasHandle(const Gpu &xpu) {
            auto h = getGpuBlasHandles().at(deviceId(xpu));
            if (!h) {
                getGpuBlasHandles()[deviceId(xpu)] = h =
                    std::shared_ptr<GpuBlasHandle>(new GpuBlasHandle, [=](GpuBlasHandle *p) {
                        setDevice(xpu);
                        gpuBlasCheck(
                            SUPERBBLAS_GPU_SELECT(XXX, cublasDestroy, rocblas_destroy_handle)(*p));
                        delete p;
                    });
                setDevice(xpu);
                gpuBlasCheck(SUPERBBLAS_GPU_SELECT(XXX, cublasCreate, rocblas_create_handle)(&*h));
            }
            setDevice(xpu);
            gpuBlasCheck(SUPERBBLAS_GPU_SELECT(XXX, cublasSetStream,
                                               rocblas_set_stream)(*h, getStream(xpu)));
            return *h;
        }

        /// Return memory allocated by the GPU BLAS library
        /// \param xpu: context

        inline std::size_t getGpuBlasMemory(const Gpu &xpu) {
            if (deviceId(xpu) < 0) return 0;
            auto h = getGpuBlasHandles().at(deviceId(xpu));
            if (!h) return 0;
            setDevice(xpu);
#    if defined(SUPERBBLAS_USE_HIP)
            std::size_t s = 0;
            gpuBlasCheck(rocblas_get_device_memory_size(*h, &s));
            return 0;
#    endif
            return 0;
        }

        /// Return all gpu sparse handles for all devices

        inline std::vector<std::shared_ptr<GpuSparseHandle>> &getGpuSparseHandles() {
            static std::vector<std::shared_ptr<GpuSparseHandle>> h(getGpuDevicesCount());
            return h;
        }

        /// Return the gpu sparse handle for the given context
        /// \param xpu: context

        inline GpuSparseHandle getGpuSparseHandle(const Gpu &xpu) {
            auto h = getGpuSparseHandles().at(deviceId(xpu));
            if (!h) {
                getGpuSparseHandles()[deviceId(xpu)] = h =
                    std::shared_ptr<GpuSparseHandle>(new GpuSparseHandle, [=](GpuSparseHandle *p) {
                        setDevice(xpu);
                        gpuSparseCheck(SUPERBBLAS_GPUSPARSE_SYMBOL(Destroy)(*p));
                        delete p;
                    });
                setDevice(xpu);
                gpuSparseCheck(SUPERBBLAS_GPUSPARSE_SYMBOL(Create)(&*h));
            }
            setDevice(xpu);
            gpuSparseCheck(SUPERBBLAS_GPUSPARSE_SYMBOL(SetStream)(*h, getStream(xpu)));
            return *h;
        }

        /// Return all gpu solver handles for all devices

        inline std::vector<std::shared_ptr<GpuSolverHandle>> &getGpuSolverHandles() {
#    ifdef SUPERBBLAS_USE_CUDA
            static std::vector<std::shared_ptr<GpuSolverHandle>> h(getGpuDevicesCount());
            return h;
#    else
            return getGpuBlasHandles();
#    endif
        }

        /// Return the gpu solver handle for the given context
        /// \param xpu: context

        inline GpuSolverHandle getGpuSolverHandle(const Gpu &xpu) {
#    ifdef SUPERBBLAS_USE_CUDA
            auto h = getGpuSolverHandles().at(deviceId(xpu));
            if (!h) {
                getGpuSolverHandles()[deviceId(xpu)] = h =
                    std::shared_ptr<GpuSolverHandle>(new GpuSolverHandle, [=](GpuSolverHandle *p) {
                        setDevice(xpu);
                        gpuSolverCheck(SUPERBBLAS_GPUSOLVER_SYMBOL(Destroy)(*p));
                        delete p;
                    });
                setDevice(xpu);
                gpuSolverCheck(SUPERBBLAS_GPUSOLVER_SYMBOL(Create)(&*h));
            }
            setDevice(xpu);
            gpuSolverCheck(SUPERBBLAS_GPUSOLVER_SYMBOL(SetStream)(*h, getStream(xpu)));
            return *h;
#    else
            return getGpuBlasHandle(xpu);
#    endif
        }
#endif // SUPERBBLAS_USE_GPU

        /// Return if `T` is a supported type
        template <typename T> struct supported_type {
            static constexpr bool value = false;
        };
        template <> struct supported_type<int> {
            static constexpr bool value = true;
        };
        template <> struct supported_type<float> {
            static constexpr bool value = true;
        };
        template <> struct supported_type<double> {
            static constexpr bool value = true;
        };
        template <> struct supported_type<std::complex<float>> {
            static constexpr bool value = true;
        };
        template <> struct supported_type<std::complex<double>> {
            static constexpr bool value = true;
        };
        template <> struct supported_type<_Complex float> {
            static constexpr bool value = true;
        };
        template <> struct supported_type<_Complex double> {
            static constexpr bool value = true;
        };
        template <typename T> struct supported_type<const T> {
            static constexpr bool value = supported_type<T>::value;
        };

#ifdef SUPERBBLAS_USE_MPI
        /// Throw exception if MPI reports an error
        /// \param error: MPI returned error

        inline void MPI_check(int error) {
            if (error == MPI_SUCCESS) return;

            char s[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(error, s, &len);

#    define CHECK_AND_THROW(ERR)                                                                   \
        if (error == ERR) {                                                                        \
            std::stringstream ss;                                                                  \
            ss << "MPI error: " #ERR ": " << std::string(&s[0], &s[0] + len);                      \
            throw std::runtime_error(ss.str());                                                    \
        }

            CHECK_AND_THROW(MPI_ERR_BUFFER);
            CHECK_AND_THROW(MPI_ERR_COUNT);
            CHECK_AND_THROW(MPI_ERR_TYPE);
            CHECK_AND_THROW(MPI_ERR_TAG);
            CHECK_AND_THROW(MPI_ERR_COMM);
            CHECK_AND_THROW(MPI_ERR_RANK);
            CHECK_AND_THROW(MPI_ERR_ROOT);
            CHECK_AND_THROW(MPI_ERR_GROUP);
            CHECK_AND_THROW(MPI_ERR_OP);
            CHECK_AND_THROW(MPI_ERR_TOPOLOGY);
            CHECK_AND_THROW(MPI_ERR_DIMS);
            CHECK_AND_THROW(MPI_ERR_ARG);
            CHECK_AND_THROW(MPI_ERR_UNKNOWN);
            CHECK_AND_THROW(MPI_ERR_TRUNCATE);
            CHECK_AND_THROW(MPI_ERR_OTHER);
            CHECK_AND_THROW(MPI_ERR_INTERN);
            CHECK_AND_THROW(MPI_ERR_IN_STATUS);
            CHECK_AND_THROW(MPI_ERR_PENDING);
            CHECK_AND_THROW(MPI_ERR_REQUEST);
            CHECK_AND_THROW(MPI_ERR_LASTCODE);
#    undef CHECK_AND_THROW
        }
#endif // SUPERBBLAS_USE_MPI
    }

    class Context {
    public:
        enum platform plat; ///< platform where the data is

        /// If `plat` is `CPU`, then `DEFAULT_DEVICE` means to use all the threads on an OpenMP
        /// fashion. If `plat` is `CUDA` and `HIP`, the value is the device identification.
        int device;

        Context(enum platform plat, int device) : plat(plat), device(device) {}

        detail::Cpu toCpu(Session session) const { return detail::Cpu{session}; }

#ifdef SUPERBBLAS_USE_GPU
        detail::Gpu toGpu(Session session) const {
            return detail::Gpu{device, device, detail::getGpuAllocStream(device),
                               detail::getGpuAllocStream(device), session};
        }

#else
        void toGpu(Session) const {
            throw std::runtime_error("Compiled without support for Cuda or HIP");
        }
#endif
    };

    /// Return a CPU context
    inline Context createCpuContext() { return Context{CPU, CPU_DEVICE_ID}; }

    /// Return a CUDA context
    /// \param device: device ID
    inline Context createCudaContext(int device = 0) {
#ifdef SUPERBBLAS_USE_CUDA
        return Context{CUDA, device};
#else
        (void)device;
        throw std::runtime_error("createGpuContext: superbblas compiled without cuda support");
#endif
    }

    /// Return a HIP context
    /// \param device: device ID
    inline Context createHipContext(int device = 0) {
#ifdef SUPERBBLAS_USE_CUDA
        return Context{HIP, device};
#else
        (void)device;
        throw std::runtime_error("createGpuContext: superbblas compiled without hip support");
#endif
    }

    /// Return a CUDA or HIP context
    /// \param device: device ID
    inline Context createGpuContext(int device = 0) {
#ifdef SUPERBBLAS_USE_GPU
        return Context{GPU, device};
#else
        (void)device;
        throw std::runtime_error("createGpuContext: superbblas compiled without gpu support");
#endif
    }

    /// Return if `T` is a supported type
    template <typename T> struct supported_type {
        static constexpr bool value = detail::supported_type<T>::value;
    };

    // Return the number of GPU devices available
    inline unsigned int getGpuDevicesCount() {
        int numDevices = 0;
#ifdef SUPERBBLAS_USE_GPU
        detail::gpuCheck(SUPERBBLAS_GPU_SYMBOL(GetDeviceCount)(&numDevices));
#endif
        return (unsigned int)numDevices;
    }

    /// Clear all internal handles to streams, cublas/rocblas, cusparse/hipsparse, and cusolver/hipsolver
    inline void clearHandles() {
#ifdef SUPERBBLAS_USE_GPU
        // Remove handles and streams
        for (auto &it : detail::getGpuBlasHandles()) it.reset();
        for (auto &it : detail::getGpuSparseHandles()) it.reset();
        for (auto &it : detail::getGpuSolverHandles()) it.reset();
        for (auto &it : detail::getGpuAllocStreams()) it.reset();
#endif
    }
}

#endif // __SUPERBBLAS_PLATFORM__
