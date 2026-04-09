#ifndef __SUPERBBLAS_COPY_N__
#define __SUPERBBLAS_COPY_N__

#include "blas.h"

#ifdef SUPERBBLAS_CREATING_LIB
/// Generate template instantiations for copy_n_lower functions with template parameters IndexType, T and Q

#    define DECL_COPY_LOWER_T_Q_EWOP(...)                                                          \
        EMIT REPLACE1(copy_n_lower, superbblas::detail::copy_n_lower<IndexType, T, Q, EWOP>)       \
            REPLACE_IndexType REPLACE_T_Q REPLACE_EWOP template __VA_ARGS__;

/// Generate template instantiations for copy_n_lower functions with template parameters IndexType, T and Q

#    define DECL_COPY_BLOCKING_LOWER_T_Q_EWOP(...)                                                 \
        EMIT REPLACE1(copy_n_blocking_lower,                                                       \
                      superbblas::detail::copy_n_blocking_lower<IndexType, T, Q, EWOP>)            \
            REPLACE_IndexType REPLACE_T_Q REPLACE_EWOP template __VA_ARGS__;

/// Generate template instantiations for zero_n functions with template parameters IndexType and T

#    define DECL_ZERO_T(...)                                                                       \
        EMIT REPLACE1(zero_n, superbblas::detail::zero_n<IndexType, T>)                            \
            REPLACE_IndexType REPLACE_T template __VA_ARGS__;

#else
#    define DECL_COPY_LOWER_T_Q_EWOP(...) __VA_ARGS__
#    define DECL_COPY_BLOCKING_LOWER_T_Q_EWOP(...) __VA_ARGS__
#    define DECL_ZERO_T(...) __VA_ARGS__
#endif

namespace superbblas {

    namespace detail {
        /// Replace std::complex by C complex
        /// \tparam T: one of float, double, std::complex<T>
        /// \return ccomplex<T>::type has the new type

        template <typename T> struct ccomplex {
            using type = T;
        };
#ifdef SUPERBBLAS_USE_FLOAT16
        template <> struct ccomplex<std::complex<_Float16>> {
            using type = _Complex _Float16;
        };
#endif
        template <> struct ccomplex<std::complex<float>> {
            using type = _Complex float;
        };
        template <> struct ccomplex<std::complex<double>> {
            using type = _Complex double;
        };

        //template <typename T> struct ccomplex<const T> {
        //    using type = const typename ccomplex<T>::type;
        //};

        /// Return whether the value is zero
        /// \param v: value to test

        template <typename T> bool is_zero(const T &v) { return std::norm(v) == 0; }
#ifdef SUPERBBLAS_USE_FLOAT16
        template <> inline bool is_zero<_Complex _Float16>(const _Complex _Float16 &v) {
            const _Float16 *v_ = (const _Float16 *)&v;
            return std::abs(v_[0]) == 0 && std::abs(v_[1]) == 0;
        }
#endif
        template <> inline bool is_zero<_Complex float>(const _Complex float &v) {
            const float *v_ = (const float *)&v;
            return std::abs(v_[0]) == 0 && std::abs(v_[1]) == 0;
        }
        template <> inline bool is_zero<_Complex double>(const _Complex double &v) {
            const double *v_ = (const double *)&v;
            return std::abs(v_[0]) == 0 && std::abs(v_[1]) == 0;
        }

        /// Throw an exception if both contexts aren't on the same device
        template <typename XPU> void check_same_device(const XPU &xpu0, const XPU &xpu1) {
            if (deviceId(xpu0) != deviceId(xpu1))
                throw std::runtime_error(
                    "check_same_device: given contexts are on different devices");
        }

        ///
        /// Non-blocking copy on CPU
        ///

        template <typename IndexType, typename T, typename Q, typename EWOP>
        void copy_n_lower(const typename elem<T>::type &alpha, const T *SB_RESTRICT v,
                          const IndexType *SB_RESTRICT indicesv, Cpu, IndexType n, Q *SB_RESTRICT w,
                          const IndexType *SB_RESTRICT indicesw, Cpu, EWOP) {

            // Shortcut for empty copy
            if (n == 0) return;

            // Shortcut for itself copy
            if (std::is_same<T, Q>::value && std::is_same<EWOP, EWOp::Copy>::value &&
                alpha == typename elem<T>::type{1} && (void *)v == (void *)w &&
                indicesv == indicesw)
                return;

            // Shortcut for zero addition
            if (std::is_same<EWOP, EWOp::Add>::value && std::norm(alpha) == 0) return;

            // Transform std::complex into C complex _Complex; std::complex exhibit some
            // performance problems with clang++
            using Tc = typename ccomplex<T>::type;
            using Qc = typename ccomplex<Q>::type;
            Tc alphac = *(Tc *)&alpha;

#ifdef _OPENMP
            if (sizeof(Q) * n > 1024u * 1024u) {
#    pragma omp parallel
                {
                    IndexType num_threads = omp_get_num_threads();
                    IndexType i = omp_get_thread_num();
                    IndexType ni = n / num_threads + (n % num_threads > i ? 1 : 0);
                    IndexType si = n / num_threads * i + std::min(n % num_threads, i);
                    Tc *vi = (Tc *)v + (indicesv == nullptr ? si : IndexType(0));
                    const IndexType *indicesvi =
                        indicesv + (indicesv != nullptr ? si : IndexType(0));
                    Qc *wi = (Qc *)w + (indicesw == nullptr ? si : IndexType(0));
                    const IndexType *indiceswi =
                        indicesw + (indicesw != nullptr ? si : IndexType(0));

                    copy_n_cpu(alphac, vi, indicesvi, Cpu{}, ni, wi, indiceswi, Cpu{}, EWOP{});
                }
            } else
#endif
            {
                copy_n_cpu(alphac, (Tc *)v, indicesv, Cpu{}, n, (Qc *)w, indicesw, Cpu{}, EWOP{});
            }
        }

#define COPY_N_VW_FOR(S)                                                                           \
    for (IndexType i = 0; i < n; ++i) {                                                            \
        IndexType vi = indicesv[i], wi = indicesw[i];                                              \
        (void)vi;                                                                                  \
        S;                                                                                         \
    }

#define COPY_N_W_FOR(S)                                                                            \
    for (IndexType i = 0; i < n; ++i) {                                                            \
        IndexType wi = indicesw[i];                                                                \
        S;                                                                                         \
    }

#define COPY_N_V_FOR(S)                                                                            \
    for (IndexType i = 0; i < n; ++i) {                                                            \
        IndexType vi = indicesv[i];                                                                \
        (void)vi;                                                                                  \
        S;                                                                                         \
    }

#define COPY_N_FOR(S)                                                                              \
    for (IndexType i = 0; i < n; ++i) { S; }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n_cpu(const T &alpha, const T *SB_RESTRICT v,
                        const IndexType *SB_RESTRICT indicesv, Cpu, IndexType n, Q *SB_RESTRICT w,
                        const IndexType *SB_RESTRICT indicesw, Cpu, EWOp::Copy) {
            // Make sure we aren't using std::complex
            static_assert(!is_complex<T>::value && !is_complex<Q>::value,
                          "don't use std::complex here; use C complex if needed");

            if (indicesv == nullptr && indicesw == nullptr) {
                /// Case: w[i] = v[i]
                if (alpha == (T)1) {
                    if (std::is_same<T, Q>::value)
                        std::memcpy(w, v, sizeof(T) * n);
                    else
                        COPY_N_FOR(w[i] = v[i]);
                } else if (is_zero(alpha)) {
                    COPY_N_FOR({ w[i] = (T)0; });
                } else {
                    COPY_N_FOR(w[i] = alpha * v[i]);
                }

            } else if (indicesv == nullptr && indicesw != nullptr) {
                /// Case: w[indicesw[i]] = v[i]
                if (alpha == (T)1) {
                    COPY_N_W_FOR(w[wi] = v[i]);
                } else if (is_zero(alpha)) {
                    COPY_N_W_FOR({ w[wi] = (T)0; });
                } else {
                    COPY_N_W_FOR(w[wi] = alpha * v[i]);
                }

            } else if (indicesv != nullptr && indicesw == nullptr) {
                /// Case: w[i] = v[indicesv[i]]
                if (alpha == (T)1) {
                    COPY_N_V_FOR(w[i] = v[vi]);
                } else if (is_zero(alpha)) {
                    COPY_N_V_FOR({ w[i] = (T)0; });
                } else {
                    COPY_N_V_FOR(w[i] = alpha * v[vi]);
                }

            } else {
                /// Case: w[indicesw[i]] = v[indicesv[i]]
                if (alpha == (T)1) {
                    COPY_N_VW_FOR(w[wi] = v[vi]);
                } else if (is_zero(alpha)) {
                    COPY_N_VW_FOR({ w[wi] = (T)0; });
                } else {
                    COPY_N_VW_FOR(w[wi] = alpha * v[vi]);
                }
            }
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n_cpu(const T &alpha, const T *SB_RESTRICT v,
                        const IndexType *SB_RESTRICT indicesv, Cpu, IndexType n, Q *SB_RESTRICT w,
                        const IndexType *SB_RESTRICT indicesw, Cpu, EWOp::Add) {
            // Make sure we aren't using std::complex
            static_assert(!is_complex<T>::value && !is_complex<Q>::value,
                          "don't use std::complex here; use C complex if needed");

            if (is_zero(alpha)) return;

            if (indicesv == nullptr && indicesw == nullptr) {
                /// Case: w[i] += v[i]
                if (alpha == (T)1) {
                    COPY_N_FOR(w[i] += v[i]);
                } else {
                    COPY_N_FOR(w[i] += alpha * v[i]);
                }

            } else if (indicesv == nullptr && indicesw != nullptr) {
                /// Case: w[indicesw[i]] += v[i]
                if (alpha == (T)1) {
                    COPY_N_W_FOR(w[wi] += v[i]);
                } else {
                    COPY_N_W_FOR(w[wi] += alpha * v[i]);
                }

            } else if (indicesv != nullptr && indicesw == nullptr) {
                /// Case: w[i] += v[indicesv[i]]
                if (alpha == (T)1) {
                    COPY_N_V_FOR(w[i] += v[vi]);
                } else {
                    COPY_N_V_FOR(w[i] += alpha * v[vi]);
                }

            } else {
                /// Case: w[indicesw[i]] += v[indicesv[i]]
                if (alpha == (T)1) {
                    COPY_N_VW_FOR(w[wi] += v[vi]);
                } else {
                    COPY_N_VW_FOR(w[wi] += alpha * v[vi]);
                }
            }
        }

#undef COPY_N_VW_FOR
#undef COPY_N_W_FOR
#undef COPY_N_V_FOR
#undef COPY_N_FOR

        ///
        /// Non-blocking copy on GPU
        ///

#ifdef SUPERBBLAS_USE_THRUST
        /// Addition of two values with different types
        template <typename T, typename Q> struct plus {
            typedef T first_argument_type;

            typedef Q second_argument_type;

            typedef Q result_type;

            __host__ __device__ result_type operator()(const T &lhs, const Q &rhs) const {
                return lhs + rhs;
            }
        };

        // Scala of a number
        template <typename T>
        struct scale : public thrust::unary_function<typename cuda_complex<T>::type,
                                                     typename cuda_complex<T>::type> {
            using cuda_T = typename cuda_complex<T>::type;
            using scalar_type = typename elem<cuda_T>::type;
            const scalar_type a;
            scale(scalar_type a) : a(a) {}
            __host__ __device__ cuda_T operator()(const cuda_T &i) const { return a * i; }
        };

        template <typename T, typename Q, typename IteratorV, typename IteratorW>
        void copy_n_same_dev_thrust(const IteratorV &itv, std::size_t n, const IteratorW &itw,
                                    EWOp::Copy, Gpu gpu) {
            thrust::copy_n(thrust_par_on(gpu), itv, n, itw);
        }

        template <typename T, typename Q, typename IteratorV, typename IteratorW>
        void copy_n_same_dev_thrust(const IteratorV &itv, std::size_t n, const IteratorW &itw,
                                    EWOp::Add, Gpu gpu) {
            thrust::transform(
                thrust_par_on(gpu), itv, itv + n, itw, itw,
                plus<typename cuda_complex<T>::type, typename cuda_complex<Q>::type>());
        }

        template <typename IndexType, typename T, typename Q, typename IteratorV, typename EWOP>
        void copy_n_same_dev_thrust(const IteratorV &itv, IndexType n, Q *w,
                                    const IndexType *indicesw, EWOP, Gpu gpu) {
            if (indicesw == nullptr) {
                copy_n_same_dev_thrust<T, Q>(itv, n, encapsulate_pointer(w), EWOP{}, gpu);
            } else {
                auto itw = thrust::make_permutation_iterator(encapsulate_pointer(w),
                                                             encapsulate_pointer(indicesw));
                copy_n_same_dev_thrust<T, Q>(itv, n, itw, EWOP{}, gpu);
            }
        }

        template <typename IndexType, typename T, typename Q, typename EWOP>
        void copy_n_same_dev_thrust(typename elem<T>::type alpha, const T *v,
                                    const IndexType *indicesv, Gpu xpuv, IndexType n, Q *w,
                                    const IndexType *indicesw, Gpu xpuw, EWOP) {
            // Shortcut for plain copy
            if (std::is_same<T, Q>::value && std::is_same<EWOP, EWOp::Copy>::value &&
                alpha == typename elem<T>::type{1} && indicesv == nullptr && indicesw == nullptr) {
                copy_n(v, xpuv, n, (T *)w, xpuw);
                return;
            }

            causalConnectTo(xpuw, xpuv);
            setDevice(xpuv);
            if (deviceId(xpuv) == CPU_DEVICE_ID) {
                launchHostKernel(
                    [=] {
                        // We call `copy_n_cpu` instead of `copy_n` with cpu contexts to avoid
                        // spawning threads inside a host kernel, they may not run on multiple cores
                        using Tc = typename ccomplex<T>::type;
                        using Qc = typename ccomplex<Q>::type;
                        copy_n_cpu(*(Tc *)&alpha, (Tc *)v, indicesv, Cpu{}, n, (Qc *)w, indicesw,
                                   Cpu{}, EWOP{});
                    },
                    xpuv);
            } else {
                if (indicesv == nullptr) {
                    auto itv = encapsulate_pointer(v);
                    if (alpha == typename elem<T>::type{1}) {
                        copy_n_same_dev_thrust<IndexType, T, Q>(itv, n, w, indicesw, EWOP{}, xpuv);
                    } else {
                        copy_n_same_dev_thrust<IndexType, T, Q>(
                            thrust::make_transform_iterator(itv, scale<T>(alpha)), n, w, indicesw,
                            EWOP{}, xpuv);
                    }
                } else {
                    auto itv = thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                                 encapsulate_pointer(indicesv));
                    if (alpha == typename elem<T>::type{1}) {
                        copy_n_same_dev_thrust<IndexType, T, Q>(itv, n, w, indicesw, EWOP{}, xpuv);
                    } else {
                        copy_n_same_dev_thrust<IndexType, T, Q>(
                            thrust::make_transform_iterator(itv, scale<T>(alpha)), n, w, indicesw,
                            EWOP{}, xpuv);
                    }
                }
                causalConnectTo(xpuv, xpuw);
            }
        }

        /// Set the first `n` elements to zero
        /// \param v: first element to set
        /// \param indices: indices of the elements to set
        /// \param n: number of elements to set
        /// \param xpu: device context

        template <typename IndexType, typename T>
        void zero_n_thrust(T *SB_RESTRICT v, const IndexType *SB_RESTRICT indices, IndexType n,
                           Gpu xpu) {
            if (indices == nullptr) {
                zero_n(v, n, xpu);
            } else if (deviceId(xpu) == CPU_DEVICE_ID) {
                launchHostKernel(
                    [=] {
                        using Tc = typename ccomplex<T>::type;
                        Tc *SB_RESTRICT vc = (Tc *)v;
                        // No openmp: we avoid spawning threads inside a host kernel, they may not run on multiple cores
                        for (IndexType i = 0; i < n; ++i) vc[indices[i]] = 0;
                    },
                    xpu);
            } else {
                setDevice(xpu);
                auto itv = thrust::make_permutation_iterator(encapsulate_pointer(v),
                                                             encapsulate_pointer(indices));
                thrust::fill_n(thrust_par_on(xpu), itv, n, T{0});
            }
        }

#endif // SUPERBBLAS_USE_THRUST

        /// Set the first `n` elements to zero
        /// \param v: first element to set
        /// \param indices: indices of the elements to set
        /// \param n: number of elements to set
        /// \param cpu: device context

        template <typename IndexType, typename T>
        void zero_n(T *SB_RESTRICT v, Cpu, const IndexType *SB_RESTRICT indices, Cpu,
                    std::size_t n) {
            if (indices == nullptr) {
                zero_n(v, n, Cpu{});
            } else {
                using Tc = typename ccomplex<T>::type;
                Tc *SB_RESTRICT vc = (Tc *)v;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                for (std::size_t i = 0; i < n; ++i) vc[indices[i]] = 0;
            }
        }

#ifdef SUPERBBLAS_USE_GPU

        /// Set the first `n` elements to zero
        /// \param v: first element to set
        /// \param indices: indices of the elements to set
        /// \param n: number of elements to set
        /// \param xpu: device context

        template <typename IndexType, typename T>
        DECL_ZERO_T(void zero_n(T *v, const Gpu &xpuv, const IndexType *indices, const Gpu &xpui,
                                IndexType n))
        IMPL({
            check_same_device(xpuv, xpui);
            causalConnectTo(xpui, xpuv);
            zero_n_thrust<IndexType, T>(v, indices, n, xpuv);
        })

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] when v and w are on device

        template <typename IndexType, typename T, typename Q, typename EWOP>
        DECL_COPY_LOWER_T_Q_EWOP(void copy_n_lower(typename elem<T>::type alpha, const T *v,
                                                   const IndexType *indicesv, const Gpu &xpuv,
                                                   IndexType n, Q *w, const IndexType *indicesw,
                                                   const Gpu &xpuw, EWOP))
        IMPL({
            assert((n == 0 || (void *)v != (void *)w || std::is_same<T, Q>::value));
            if (n == 0) return;

            // Treat zero case
            if (std::norm(alpha) == 0) {
                if (std::is_same<EWOP, EWOp::Copy>::value)
                    zero_n<IndexType>(w, xpuw, indicesw, xpuw, n);
            }

            // Actions when the v and w are on the same device
            else if (deviceId(xpuv) == deviceId(xpuw)) {
                if (indicesv == nullptr && indicesw == nullptr &&
                    alpha == typename elem<T>::type{1} && std::is_same<T, Q>::value &&
                    std::is_same<EWOP, EWOp::Copy>::value) {
                    copy_n(v, xpuv, n, (T *)w, xpuw);
                } else {
                    copy_n_same_dev_thrust(alpha, v, indicesv, xpuv, n, w, indicesw, xpuw, EWOP{});
                }
            }

            // Simple case when the v and w are NOT on the same device and no permutation is involved
            else if (indicesv == nullptr && indicesw == nullptr &&
                     alpha == typename elem<T>::type{1} && std::is_same<T, Q>::value &&
                     std::is_same<EWOP, EWOp::Copy>::value && deviceId(xpuv) != deviceId(xpuw)) {
                copy_n(v, xpuv, n, (T *)w, xpuw);
            }

            // If v is permuted, copy v[indices[i]] in a contiguous chunk, and then copy
            else if (indicesv != nullptr) {
                vector<Q, Gpu> v0(n, xpuv, doCacheAlloc);
                copy_n_lower<IndexType>(alpha, v, indicesv, xpuv, n, v0.data(), nullptr, xpuv,
                                        EWOp::Copy{});
                copy_n_lower<IndexType>(Q{1}, v0.data(), nullptr, xpuv, n, w, indicesw, xpuw,
                                        EWOP{});
            }

            // Otherwise copy v to xpuw, and then copy it to the w[indices[i]]
            else {
                vector<T, Gpu> v1(n, xpuw, doCacheAlloc);
                copy_n_lower<IndexType>(T{1}, v, nullptr, xpuv, n, v1.data(), nullptr, xpuw,
                                        EWOp::Copy{});
                copy_n_lower<IndexType>(alpha, v1.data(), nullptr, xpuw, n, w, indicesw, xpuw,
                                        EWOP{});
            }
        })

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] from device to host and vice versa

        template <typename IndexType, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP,
                  typename std::enable_if<!std::is_same<XPU0, XPU1>::value, bool>::type = true>
        void copy_n_lower(typename elem<T>::type alpha, const T *v, const IndexType *indicesv,
                          XPU0 xpu0, IndexType n, Q *w, const IndexType *indicesw, XPU1 xpu1,
                          EWOP) {
            if (n == 0) return;

            // Treat zero case
            if (std::norm(alpha) == 0) {
                if (std::is_same<EWOP, EWOp::Copy>::value) zero_n(w, xpu1, indicesw, xpu1, n);
            }

            // Base case
            else if (std::is_same<T, Q>::value && std::is_same<EWOP, EWOp::Copy>::value &&
                     indicesv == nullptr && indicesw == nullptr) {
                copy_n(v, xpu0, n, (T *)w, xpu1);
                // Scale by alpha
#    if defined(__GNUC__) && !defined(__clang__)
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wrestrict"
#    endif
                copy_n_lower<IndexType>((Q)alpha, w, nullptr, xpu1, n, w, nullptr, xpu1,
                                        EWOp::Copy{});
#    if defined(__GNUC__) && !defined(__clang__)
#        pragma GCC diagnostic pop
#    endif
            }

            // If v is permuted, copy v[indices[i]] in a contiguous chunk, and then copy
            else if (indicesv != nullptr) {
                vector<Q, XPU0> v0(n, xpu0, doCacheAlloc);
                copy_n_lower<IndexType>(alpha, v, indicesv, xpu0, n, v0.data(), nullptr, xpu0,
                                        EWOp::Copy{});
                copy_n_lower<IndexType>(Q{1}, v0.data(), nullptr, xpu0, n, w, indicesw, xpu1,
                                        EWOP{});
            }

            // Otherwise copy v to xpu1, and then copy it to the w[indices[i]]
            else {
                vector<T, XPU1> v1(n, xpu1, doCacheAlloc);
                copy_n_lower<IndexType>(T{1}, v, nullptr, xpu0, n, v1.data(), nullptr, xpu1,
                                        EWOp::Copy{});
                copy_n_lower<IndexType>(alpha, v1.data(), nullptr, xpu1, n, w, indicesw, xpu1,
                                        EWOP{});
            }
        }
#endif // SUPERBBLAS_USE_GPU

        /// Copy n values with displacements if given, w[indicesw[i]] (+)= alpha * v[indicesv[i]]
        /// \param alpha: factor applied to the read elements
        /// \param v: pointer to the first elements to read
        /// \param xpuv: context of the elements to read
        /// \param indicesv: (optional) pointer to first displacement in v
        /// \param xpuiv: context of indicesv
        /// \param n: number of elements to read
        /// \param w: pointer to the first element to write
        /// \param xpuw: context of the elements to write
        /// \param indicesw: (optional) pointer to first displacement in w
        /// \param xpuiw: context of indicesw
        /// \param EWOP: copy or add

        template <typename IndexType, typename T, typename Q, typename XPUV, typename XPUW,
                  typename EWOP>
        void copy_n(typename elem<T>::type alpha, const T *v, const XPUV &xpuv,
                    const IndexType *indicesv, const XPUV &xpuiv, IndexType n, Q *w,
                    const XPUW &xpuw, const IndexType *indicesw, const XPUW &xpuiw, EWOP) {
            // Check that the data and the indices are on the same device
            check_same_device(xpuv, xpuiv);
            check_same_device(xpuw, xpuiw);

            // If indices vectors isn't null, connect causally its stream with the input data vector,
            // which will carry on all operations
            if (indicesv != nullptr) causalConnectTo(xpuiv, xpuv);
            if (indicesw != nullptr &&
                (!std::is_same<XPUV, XPUW>::value || getStream(xpuiv) != getStream(xpuiw)))
                causalConnectTo(xpuiw, xpuv);
            copy_n_lower(alpha, v, indicesv, xpuv, n, w, indicesw, xpuw, EWOP{});
            if (indicesv != nullptr && getStream(xpuw) != getStream(xpuiv))
                causalConnectTo(xpuv, xpuiv);
            if (indicesw != nullptr && getStream(xpuw) != getStream(xpuiw) &&
                (!std::is_same<XPUV, XPUW>::value || getStream(xpuiv) != getStream(xpuiw)))
                causalConnectTo(xpuv, xpuiw);
        }

        /// Copy n values with displacements if given, w[indicesw[i]] (+)= alpha * v[indicesv[i]]
        /// \param alpha: factor applied to the read elements
        /// \param v: pointer to the first elements to read
        /// \param xpuv: context of the elements to read
        /// \param n: number of elements to read
        /// \param w: pointer to the first element to write
        /// \param xpuw: context of the elements to write
        /// \param EWOP: copy or add

        template <typename IndexType, typename T, typename Q, typename XPUV, typename XPUW,
                  typename EWOP>
        void copy_n(typename elem<T>::type alpha, const T *v, const XPUV &xpuv, IndexType n, Q *w,
                    const XPUW &xpuw, EWOP) {
            copy_n<IndexType>(alpha, v, xpuv, nullptr, xpuv, n, w, xpuw, nullptr, xpuw, EWOP{});
        }

        ///
        /// Blocking copy on CPU
        ///

        template <typename IndexType, typename T, typename Q, typename EWOP>
        void copy_n_blocking_lower(typename elem<T>::type alpha, const T *SB_RESTRICT v,
                                   IndexType blocking, const IndexType *SB_RESTRICT indicesv, Cpu,
                                   IndexType n, Q *SB_RESTRICT w,
                                   const IndexType *SB_RESTRICT indicesw, Cpu, EWOP) {

            // Shortcut for empty copy
            if (n == 0) return;

            // Shortcut for no blocking
            if (blocking == 1) {
                copy_n_lower(alpha, v, indicesv, Cpu{}, n, w, indicesw, Cpu{}, EWOP{});
                return;
            }
            if (indicesv == nullptr && indicesw == nullptr) {
                copy_n_lower(alpha, v, indicesv, Cpu{}, n * blocking, w, indicesw, Cpu{}, EWOP{});
                return;
            }

            // Transform std::complex into C complex _Complex; std::complex exhibit some
            // performance problems with clang++
            using Tc = typename ccomplex<T>::type;
            using Qc = typename ccomplex<Q>::type;
            Tc alphac = *(Tc *)&alpha;

#ifdef _OPENMP
            if (sizeof(Q) * n * blocking > 1024u * 1024u) {
#    pragma omp parallel
                {
                    IndexType num_threads = omp_get_num_threads();
                    IndexType i = omp_get_thread_num();
                    IndexType ni = n / num_threads + (n % num_threads > i ? 1 : 0);
                    IndexType si = n / num_threads * i + std::min(n % num_threads, i);
                    Tc *vi = (Tc *)v + (indicesv == nullptr ? blocking * si : IndexType(0));
                    const IndexType *indicesvi =
                        indicesv + (indicesv != nullptr ? si : IndexType(0));
                    Qc *wi = (Qc *)w + (indicesw == nullptr ? blocking * si : IndexType(0));
                    const IndexType *indiceswi =
                        indicesw + (indicesw != nullptr ? si : IndexType(0));

                    copy_n_blocking_cpu(alphac, vi, blocking, indicesvi, Cpu{}, ni, wi, indiceswi,
                                        Cpu{}, EWOP{});
                }
            } else
#endif
            {
                copy_n_blocking_cpu(alphac, (Tc *)v, blocking, indicesv, Cpu{}, n, (Qc *)w,
                                    indicesw, Cpu{}, EWOP{});
            }
        }

#define COPY_N_BLOCKING_VW_FOR(S)                                                                  \
    for (IndexType i = 0; i < n; ++i) {                                                            \
        for (IndexType j = 0; j < blocking; ++j) {                                                 \
            IndexType vj = indicesv[i] + j, wj = indicesw[i] + j;                                  \
            (void)vj;                                                                              \
            S;                                                                                     \
        }                                                                                          \
    }

#define COPY_N_BLOCKING_W_FOR(S)                                                                   \
    for (IndexType i = 0; i < n; ++i) {                                                            \
        for (IndexType j = 0; j < blocking; ++j) {                                                 \
            IndexType wj = indicesw[i] + j, idx = i * blocking + j;                                \
            (void)idx;                                                                             \
            S;                                                                                     \
        }                                                                                          \
    }

#define COPY_N_BLOCKING_V_FOR(S)                                                                   \
    for (IndexType i = 0; i < n; ++i) {                                                            \
        for (IndexType j = 0; j < blocking; ++j) {                                                 \
            IndexType vj = indicesv[i] + j, idx = i * blocking + j;                                \
            (void)vj;                                                                              \
            S;                                                                                     \
        }                                                                                          \
    }

        /// Copy n values, w[indicesw[i]] = v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n_blocking_cpu(const T &alpha, const T *SB_RESTRICT v, IndexType blocking,
                                 const IndexType *SB_RESTRICT indicesv, Cpu, IndexType n,
                                 Q *SB_RESTRICT w, const IndexType *SB_RESTRICT indicesw, Cpu,
                                 EWOp::Copy) {
            // Make sure we aren't using std::complex
            static_assert(!is_complex<T>::value && !is_complex<Q>::value,
                          "don't use std::complex here; use C complex if needed");

            if (indicesv == nullptr && indicesw != nullptr) {
                /// Case: w[indicesw[i]] = v[i]
                if (alpha == (T)1) {
                    COPY_N_BLOCKING_W_FOR(w[wj] = v[idx]);
                } else if (is_zero(alpha)) {
                    COPY_N_BLOCKING_W_FOR({ w[wj] = (T)0; });
                } else {
                    COPY_N_BLOCKING_W_FOR(w[wj] = alpha * v[idx]);
                }

            } else if (indicesv != nullptr && indicesw == nullptr) {
                /// Case: w[i] = v[indicesv[i]]
                if (alpha == (T)1) {
                    COPY_N_BLOCKING_V_FOR(w[idx] = v[vj]);
                } else if (is_zero(alpha)) {
                    COPY_N_BLOCKING_V_FOR({ w[idx] = (T)0; });
                } else {
                    COPY_N_BLOCKING_V_FOR(w[idx] = alpha * v[vj]);
                }

            } else {
                /// Case: w[indicesw[i]] = v[indicesv[i]]
                if (alpha == (T)1) {
                    COPY_N_BLOCKING_VW_FOR(w[wj] = v[vj]);
                } else if (is_zero(alpha)) {
                    COPY_N_BLOCKING_VW_FOR({ w[wj] = (T)0; });
                } else {
                    COPY_N_BLOCKING_VW_FOR(w[wj] = alpha * v[vj]);
                }
            }
        }

        /// Copy n values, w[indicesw[i]] += v[indicesv[i]]
        template <typename IndexType, typename T, typename Q>
        void copy_n_blocking_cpu(const T &alpha, const T *SB_RESTRICT v, IndexType blocking,
                                 const IndexType *SB_RESTRICT indicesv, Cpu, IndexType n,
                                 Q *SB_RESTRICT w, const IndexType *SB_RESTRICT indicesw, Cpu,
                                 EWOp::Add) {
            // Make sure we aren't using std::complex
            static_assert(!is_complex<T>::value && !is_complex<Q>::value,
                          "don't use std::complex here; use C complex if needed");

            if (is_zero(alpha)) return;

            if (indicesv == nullptr && indicesw != nullptr) {
                /// Case: w[indicesw[i]] += v[i]
                if (alpha == (T)1) {
                    COPY_N_BLOCKING_W_FOR(w[wj] += v[idx]);
                } else {
                    COPY_N_BLOCKING_W_FOR(w[wj] += alpha * v[idx]);
                }

            } else if (indicesv != nullptr && indicesw == nullptr) {
                /// Case: w[i] += v[indicesv[i]]
                if (alpha == (T)1) {
                    COPY_N_BLOCKING_V_FOR(w[idx] += v[vj]);
                } else {
                    COPY_N_BLOCKING_V_FOR(w[idx] += alpha * v[vj]);
                }

            } else {
                /// Case: w[indicesw[i]] += v[indicesv[i]]
                if (alpha == (T)1) {
                    COPY_N_BLOCKING_VW_FOR(w[wj] += v[vj]);
                } else {
                    COPY_N_BLOCKING_VW_FOR(w[wj] += alpha * v[vj]);
                }
            }
        }

#undef COPY_N_BLOCKING_VW_FOR
#undef COPY_N_BLOCKING_W_FOR
#undef COPY_N_BLOCKING_V_FOR

        ///
        /// Blocking copy on GPU
        ///

#ifdef SUPERBBLAS_USE_THRUST

        namespace copy_n_blocking_same_dev_thrust_ns {
            template <typename IndexType, typename T, typename Q, typename EWOP>
            struct copy_n_blocking_elem_v_and_w;

            /// Case: w[indicesw[i]] = v[indicesv[i]]

            template <typename IndexType, typename T, typename Q>
            struct copy_n_blocking_elem_v_and_w<IndexType, T, Q, EWOp::Copy>
                : public thrust::unary_function<IndexType, void> {
                const T alpha;
                const T *const SB_RESTRICT v;
                const IndexType blocking;
                const IndexType *const SB_RESTRICT indicesv;
                Q *const SB_RESTRICT w;
                const IndexType *const SB_RESTRICT indicesw;
                copy_n_blocking_elem_v_and_w(T alpha, const T *v, IndexType blocking,
                                             const IndexType *indicesv, Q *w,
                                             const IndexType *indicesw)
                    : alpha(alpha),
                      v(v),
                      blocking(blocking),
                      indicesv(indicesv),
                      w(w),
                      indicesw(indicesw) {}

                __HOST__ __DEVICE__ void operator()(IndexType i) {
                    IndexType d = i / blocking, r = i % blocking;
                    w[indicesw[d] + r] = alpha * v[indicesv[d] + r];
                }
            };

            /// Case: w[indicesw[i]] += v[indicesv[i]]

            template <typename IndexType, typename T, typename Q>
            struct copy_n_blocking_elem_v_and_w<IndexType, T, Q, EWOp::Add>
                : public thrust::unary_function<IndexType, void> {
                const T alpha;
                const T *const SB_RESTRICT v;
                const IndexType blocking;
                const IndexType *const SB_RESTRICT indicesv;
                Q *const SB_RESTRICT w;
                const IndexType *const SB_RESTRICT indicesw;
                copy_n_blocking_elem_v_and_w(T alpha, const T *v, IndexType blocking,
                                             const IndexType *indicesv, Q *w,
                                             const IndexType *indicesw)
                    : alpha(alpha),
                      v(v),
                      blocking(blocking),
                      indicesv(indicesv),
                      w(w),
                      indicesw(indicesw) {}

                __HOST__ __DEVICE__ void operator()(IndexType i) {
                    IndexType d = i / blocking, r = i % blocking;
                    w[indicesw[d] + r] += alpha * v[indicesv[d] + r];
                }
            };

            template <typename IndexType, typename T, typename Q, typename EWOP>
            struct copy_n_blocking_elem_w;

            /// Case: w[indicesw[i]] = v[i]

            template <typename IndexType, typename T, typename Q>
            struct copy_n_blocking_elem_w<IndexType, T, Q, EWOp::Copy>
                : public thrust::unary_function<IndexType, void> {
                const T alpha;
                const T *const SB_RESTRICT v;
                const IndexType blocking;
                Q *const SB_RESTRICT w;
                const IndexType *const SB_RESTRICT indicesw;
                copy_n_blocking_elem_w(T alpha, const T *v, IndexType blocking, Q *w,
                                       const IndexType *indicesw)
                    : alpha(alpha), v(v), blocking(blocking), w(w), indicesw(indicesw) {}

                __HOST__ __DEVICE__ void operator()(IndexType i) {
                    IndexType d = i / blocking, r = i % blocking;
                    w[indicesw[d] + r] = alpha * v[i];
                }
            };

            /// Case: w[indicesw[i]] += v[i]

            template <typename IndexType, typename T, typename Q>
            struct copy_n_blocking_elem_w<IndexType, T, Q, EWOp::Add>
                : public thrust::unary_function<IndexType, void> {
                const T alpha;
                const T *const SB_RESTRICT v;
                const IndexType blocking;
                Q *const SB_RESTRICT w;
                const IndexType *const SB_RESTRICT indicesw;
                copy_n_blocking_elem_w(T alpha, const T *v, IndexType blocking, Q *w,
                                       const IndexType *indicesw)
                    : alpha(alpha), v(v), blocking(blocking), w(w), indicesw(indicesw) {}

                __HOST__ __DEVICE__ void operator()(IndexType i) {
                    IndexType d = i / blocking, r = i % blocking;
                    w[indicesw[d] + r] += alpha * v[i];
                }
            };

            template <typename IndexType, typename T, typename Q, typename EWOP>
            struct copy_n_blocking_elem_v;

            /// Case: w[i] = v[indicesv[i]]

            template <typename IndexType, typename T, typename Q>
            struct copy_n_blocking_elem_v<IndexType, T, Q, EWOp::Copy>
                : public thrust::unary_function<IndexType, void> {
                const T alpha;
                const T *const SB_RESTRICT v;
                const IndexType blocking;
                const IndexType *const SB_RESTRICT indicesv;
                Q *const SB_RESTRICT w;
                copy_n_blocking_elem_v(T alpha, const T *v, IndexType blocking,
                                       const IndexType *indicesv, Q *w)
                    : alpha(alpha), v(v), blocking(blocking), indicesv(indicesv), w(w) {}

                __HOST__ __DEVICE__ void operator()(IndexType i) {
                    IndexType d = i / blocking, r = i % blocking;
                    w[i] = alpha * v[indicesv[d] + r];
                }
            };

            /// Case: w[i] += v[indicesv[i]]

            template <typename IndexType, typename T, typename Q>
            struct copy_n_blocking_elem_v<IndexType, T, Q, EWOp::Add>
                : public thrust::unary_function<IndexType, void> {
                const T alpha;
                const T *const SB_RESTRICT v;
                const IndexType blocking;
                const IndexType *const SB_RESTRICT indicesv;
                Q *const SB_RESTRICT w;
                copy_n_blocking_elem_v(T alpha, const T *v, IndexType blocking,
                                       const IndexType *indicesv, Q *w)
                    : alpha(alpha), v(v), blocking(blocking), indicesv(indicesv), w(w) {}

                __HOST__ __DEVICE__ void operator()(IndexType i) {
                    IndexType d = i / blocking, r = i % blocking;
                    w[i] += alpha * v[indicesv[d] + r];
                }
            };
        }

        template <typename IndexType, typename T, typename Q, typename EWOP>
        void copy_n_blocking_same_dev_thrust(typename elem<T>::type alpha, const T *v,
                                             IndexType blocking, const IndexType *indicesv,
                                             Gpu xpuv, IndexType n, Q *w, const IndexType *indicesw,
                                             Gpu xpuw, EWOP) {
            using namespace copy_n_blocking_same_dev_thrust_ns;
            if (indicesv == nullptr && indicesw == nullptr) {
                copy_n_lower<IndexType>(alpha, v, nullptr, xpuv, n * blocking, w, nullptr, xpuw,
                                        EWOP{});
                return;
            }

            causalConnectTo(xpuw, xpuv);
            setDevice(xpuv);
            if (deviceId(xpuv) == CPU_DEVICE_ID) {
                launchHostKernel(
                    [=] {
                        // We call `copy_n_blocking_cpu` instead of `copy_n_blocking` with cpu contexts to avoid
                        // spawning threads inside a host kernel, they may not run on multiple cores
                        using Tc = typename ccomplex<T>::type;
                        using Qc = typename ccomplex<Q>::type;
                        copy_n_blocking_cpu(*(Tc *)&alpha, (Tc *)v, blocking, indicesv, Cpu{}, n,
                                            (Qc *)w, indicesw, Cpu{}, EWOP{});
                    },
                    xpuv);
            } else {
                if (indicesv == nullptr && indicesw != nullptr) {
                    thrust::for_each_n(
                        thrust_par_on(xpuv), thrust::make_counting_iterator(IndexType(0)),
                        blocking * n,
                        copy_n_blocking_elem_w<IndexType, typename cuda_complex<T>::type,
                                               typename cuda_complex<Q>::type, EWOP>(
                            alpha, (typename cuda_complex<T>::type *)v, blocking,
                            (typename cuda_complex<Q>::type *)w, indicesw));
                } else if (indicesv != nullptr && indicesw == nullptr) {
                    thrust::for_each_n(
                        thrust_par_on(xpuv), thrust::make_counting_iterator(IndexType(0)),
                        blocking * n,
                        copy_n_blocking_elem_v<IndexType, typename cuda_complex<T>::type,
                                               typename cuda_complex<Q>::type, EWOP>(
                            alpha, (typename cuda_complex<T>::type *)v, blocking, indicesv,
                            (typename cuda_complex<Q>::type *)w));
                } else {
                    thrust::for_each_n(
                        thrust_par_on(xpuv), thrust::make_counting_iterator(IndexType(0)),
                        blocking * n,
                        copy_n_blocking_elem_v_and_w<IndexType, typename cuda_complex<T>::type,
                                                     typename cuda_complex<Q>::type, EWOP>(
                            alpha, (typename cuda_complex<T>::type *)v, blocking, indicesv,
                            (typename cuda_complex<Q>::type *)w, indicesw));
                }
            }
            causalConnectTo(xpuv, xpuw);
        }

#endif // SUPERBBLAS_USE_THRUST

#ifdef SUPERBBLAS_USE_GPU

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] when v and w are on device

        template <typename IndexType, typename T, typename Q, typename EWOP>
        DECL_COPY_BLOCKING_LOWER_T_Q_EWOP(void copy_n_blocking_lower(
            typename elem<T>::type alpha, const T *v, IndexType blocking, const IndexType *indicesv,
            const Gpu &xpuv, IndexType n, Q *w, const IndexType *indicesw, const Gpu &xpuw, EWOP))
        IMPL({
            if (n == 0) return;

            // Actions when the v and w are on the same device
            if (deviceId(xpuv) == deviceId(xpuw)) {
                copy_n_blocking_same_dev_thrust(alpha, v, blocking, indicesv, xpuv, n, w, indicesw,
                                                xpuw, EWOP{});

            } else if (indicesv == nullptr && indicesw == nullptr) {
                copy_n_lower<IndexType>(alpha, v, nullptr, xpuv, n * blocking, w, nullptr, xpuw,
                                        EWOP{});

            } else if (indicesv != nullptr) {
                vector<Q, Gpu> v0(n * blocking, xpuv, doCacheAlloc);
                copy_n_blocking_lower<IndexType>(alpha, v, blocking, indicesv, xpuv, n, v0.data(),
                                                 nullptr, xpuv, EWOp::Copy{});
                copy_n_blocking_lower<IndexType>(Q{1}, v0.data(), blocking, nullptr, xpuv, n, w,
                                                 indicesw, xpuw, EWOP{});
            } else {
                vector<T, Gpu> v0(n * blocking, xpuw, doCacheAlloc);
                copy_n_blocking_lower<IndexType>(T{1}, v, blocking, nullptr, xpuv, n, v0.data(),
                                                 nullptr, xpuw, EWOp::Copy{});
                copy_n_blocking_lower<IndexType>(alpha, v0.data(), blocking, nullptr, xpuw, n, w,
                                                 indicesw, xpuw, EWOP{});
            }
        })

        /// Copy n values, w[indicesw[i]] (+)= v[indicesv[i]] from device to host or vice versa

        template <typename IndexType, typename T, typename Q, typename XPUV, typename XPUW,
                  typename EWOP,
                  typename std::enable_if<!std::is_same<XPUV, XPUW>::value, bool>::type = true>
        void copy_n_blocking_lower(typename elem<T>::type alpha, const T *v, IndexType blocking,
                                   const IndexType *indicesv, const XPUV &xpuv, IndexType n, Q *w,
                                   const IndexType *indicesw, const XPUW &xpuw, EWOP) {
            if (n == 0) return;

            if (indicesv) {
                vector<Q, XPUV> v0(n * blocking, xpuv);
                copy_n_blocking_lower<IndexType>(alpha, v, blocking, indicesv, xpuv, n, v0.data(),
                                                 nullptr, xpuv, EWOp::Copy{});
                copy_n_blocking_lower<IndexType>(Q{1}, v0.data(), blocking, nullptr, xpuv, n, w,
                                                 indicesw, xpuw, EWOP{});
            } else if (indicesw) {
                vector<T, XPUW> v0(n * blocking, xpuw);
                copy_n_lower<IndexType>(T{1}, v, nullptr, xpuv, blocking * n, v0.data(), nullptr,
                                        xpuw, EWOp::Copy{});
                copy_n_blocking_lower<IndexType>(alpha, v0.data(), blocking, nullptr, xpuw, n, w,
                                                 indicesw, xpuw, EWOP{});
            } else {
                copy_n_lower<IndexType>(alpha, v, nullptr, xpuv, blocking * n, w, nullptr, xpuw,
                                        EWOP{});
            }
        }
#endif // SUPERBBLAS_USE_GPU

        /// Copy n values with displacements if given, w[indicesw[i]] (+)= alpha * v[indicesv[i]]
        /// \param alpha: factor applied to the read elements
        /// \param v: pointer to the first elements to read
        /// \param xpuv: context of the elements to read
        /// \param indicesv: (optional) pointer to first displacement in v
        /// \param xpuiv: context of indicesv
        /// \param n: number of elements to read
        /// \param w: pointer to the first element to write
        /// \param xpuw: context of the elements to write
        /// \param indicesw: (optional) pointer to first displacement in w
        /// \param xpuiw: context of indicesw
        /// \param EWOP: copy or add

        template <typename IndexType, typename T, typename Q, typename XPUV, typename XPUW,
                  typename EWOP>
        void copy_n_blocking(typename elem<T>::type alpha, const T *v, const XPUV &xpuv,
                             IndexType blocking, const IndexType *indicesv, const XPUV &xpuiv,
                             IndexType n, Q *w, const XPUW &xpuw, const IndexType *indicesw,
                             const XPUW &xpuiw, EWOP) {
            // Check that the data and the indices are on the same device
            check_same_device(xpuv, xpuiv);
            check_same_device(xpuw, xpuiw);

            // If indices vectors isn't null, connect causally its stream with the input data vector,
            // which will carry on all operations
            if (indicesv != nullptr) causalConnectTo(xpuiv, xpuv);
            if (indicesw != nullptr &&
                (!std::is_same<XPUV, XPUW>::value || getStream(xpuiv) != getStream(xpuiw)))
                causalConnectTo(xpuiw, xpuv);
            copy_n_blocking_lower(alpha, v, blocking, indicesv, xpuv, n, w, indicesw, xpuw, EWOP{});
            if (indicesv != nullptr && getStream(xpuiv) != getStream(xpuw))
                causalConnectTo(xpuv, xpuiv);
            if (indicesw != nullptr && getStream(xpuiw) != getStream(xpuw) &&
                (!std::is_same<XPUV, XPUW>::value || getStream(xpuiv) != getStream(xpuiw)))
                causalConnectTo(xpuv, xpuiw);
        }
    }
}
#endif // __SUPERBBLAS_COPY_N__
