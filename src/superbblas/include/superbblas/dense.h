#ifndef __SUPERBBLAS_DENSE__
#define __SUPERBBLAS_DENSE__

#include "dist.h"

namespace superbblas {

    namespace detail {

        /// Return an order concatenating the three given strings (or in reverse order if `co` is SlowToFast
        /// \param a,b,c: string to concatenate
        /// \return: the ordering

        template <std::size_t N, typename Va, typename Vb, typename Vc>
        Order<N> concat(const Va &a, const Vb &b, const Vc &c, CoorOrder co) {
            if (a.size() + b.size() + c.size() != N)
                throw std::runtime_error("concat: wrong string size to concat");
            if (co == FastToSlow) {
                Order<N> r;
                std::copy_n(a.begin(), a.size(), r.begin());
                std::copy_n(b.begin(), b.size(), r.begin() + a.size());
                std::copy_n(c.begin(), c.size(), r.begin() + a.size() + b.size());
                return r;
            } else {
                return concat<N>(c, b, a, FastToSlow);
            }
        }

        template <typename T0, typename T1>
        std::string concat(const T0 &t0, const T1 &t1, char t2) {
            std::string r(t0.size() + t1.size() + 1, char(0));
            std::copy(t0.begin(), t0.end(), r.begin());
            std::copy(t1.begin(), t1.end(), r.begin() + t0.size());
            r.at(t0.size() + t1.size()) = t2;
            return r;
        }

        inline void throw_or_exit(const std::string &err_msg, bool terminate = false) {
            if (terminate) {
                std::cerr << err_msg << std::endl;
                std::exit(-1);
            } else {
                throw std::runtime_error(err_msg);
            }
        }

        inline void checkLapack(int info, const std::string &name, bool terminate = false) {
            if (info == 0) return;
            if (info < 0)
                throw_or_exit(
                    std::string("Error in a lapack routine: wrong argument at position ") +
                        std::to_string(-info),
                    terminate);
            if (info > 0)
                throw_or_exit(std::string("Error in lapack routine ") + name + std::string(": ") +
                                  std::to_string(info),
                              terminate);
        }

        template <typename T> void local_cholesky(std::size_t n, std::size_t k, vector<T, Cpu> v) {

            tracker<Cpu> _t("local cholesky (Cpu)", v.ctx());
            _t.flops = (double)n * n * n / 3 * k * multiplication_cost<T>::value;
            _t.memops = (double)n * n * k * sizeof(T);

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
            int num_threads = omp_get_max_threads();
#else
            int num_threads = 1;
#endif

            T *p = v.data();
            std::vector<int> info(num_threads, 0);

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp parallel
#endif
            {
#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
                int id = omp_get_thread_num();
#else
                int id = 0;
#endif

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp for schedule(static)
#endif
                for (std::size_t i = 0; i < k; ++i)
                    if (info[id] == 0) info[id] = xpotrf('U', n, p + n * n * i, n, Cpu{});
            }

            for (int i : info) checkLapack(i, "cholesky (cpu)");
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_cholesky(std::size_t n, std::size_t k, const vector<T, Gpu> &v) {

            if (n == 0 || k == 0) return;
            if (deviceId(v.ctx()) == CPU_DEVICE_ID)
                throw std::runtime_error(
                    "superbblas::detail::local_cholesky: unsupported allocation device");

            tracker<Gpu> _t("local cholesky (GPU)", v.ctx());
            _t.flops = (double)n * n * n / 3 * k * multiplication_cost<T>::value;
            _t.memops = (double)n * n * k * sizeof(T);

            vector<int, Gpu> info(k, v.ctx(), doCacheAlloc);
            auto xpu_host = v.ctx().toCpuPinned();
#    ifdef SUPERBBLAS_USE_CUDA
            vector<T *, Gpu> v_ps_cpu(k, xpu_host, doCacheAlloc);
            auto v_ps_cpu_ptr = v_ps_cpu.data();
            auto v_ptr = v.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i) v_ps_cpu_ptr[i] = v_ptr + n * n * i;
                },
                xpu_host);
            vector<T *, Gpu> v_ps_gpu = makeSure(v_ps_cpu, v.ctx(), doCacheAlloc);
            gpuSolverCheck(SUPERBBLAS_GPUSOLVER_SYMBOL(XpotrfBatched)(
                getGpuSolverHandle(v.ctx()), CUBLAS_FILL_MODE_UPPER, n, v_ps_gpu.data(), n,
                info.data(), k));
#    else
            rocsolverXpotrfStridedBatched(rocblas_fill_upper, n, v.data(), n, n * n, info.data(), k,
                                          v.ctx());
#    endif
            vector<int, Gpu> info_cpu = makeSure(info, xpu_host, doCacheAlloc);
            auto info_cpu_ptr = info_cpu.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i)
                        checkLapack(info_cpu_ptr[i], "cholesky gpu", true /* terminate */);
                },
                xpu_host);
        }
#endif // SUPERBBLAS_USE_GPU

        /// If left_side, perform a\x -> x; and x/a -> x otherwise
        /// \param left_side: whether the inverse go to the left
        /// \param n: size of the matrix
        /// \param k: number of matrices to invert
        /// \param m: number of columns (if left_side) or rows (if !left_side) that x and y have

        template <typename T>
        void local_trsm(bool left_side, std::size_t n, std::size_t k, std::size_t m, T alpha,
                        vector<T, Cpu> a, vector<T, Cpu> x) {

            if (n == 0 || k == 0 || m == 0) return;

            tracker<Cpu> _t("local trsm (Cpu)", a.ctx());
            _t.flops = (double)n * n / 2 * m * k * multiplication_cost<T>::value;
            _t.memops = (double)(n * n / 2 + n * m * 2) * k * sizeof(T);

            const T *ap = a.data();
            T *xp = x.data();

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp parallel for schedule(static)
#endif
            for (std::size_t i = 0; i < k; ++i)
                xtrsm(left_side ? 'L' : 'R', 'U', 'N', 'N', left_side ? n : m, left_side ? m : n,
                      alpha, ap + n * n * i, n, xp + n * m * i, left_side ? n : m, a.ctx());
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_trsm(bool left_side, std::size_t n, std::size_t k, std::size_t m, T alpha,
                        const vector<T, Gpu> &a, const vector<T, Gpu> &x) {

            if (n == 0 || k == 0 || m == 0) return;
            if (deviceId(a.ctx()) == CPU_DEVICE_ID)
                throw std::runtime_error(
                    "superbblas::detail::local_trsm: unsupported allocation device");
            check_same_device(a.ctx(), x.ctx());
            causalConnectTo(x.ctx(), a.ctx());

            tracker<Gpu> _t("local trsm (GPU)", a.ctx());
            _t.flops = (double)n * n / 2 * m * k * multiplication_cost<T>::value;
            _t.memops = (double)(n * n / 2 + n * m * 2) * k * sizeof(T);

#    ifdef SUPERBBLAS_USE_CUDA
            // NOTE: cublasXtrsmBatched presents an undocumented limitation: it fails when
            // one of the dimensions of the input matrices is too large
            auto xpu_host = a.ctx().toCpuPinned();
            const std::size_t max_m = 1u << 18; // = 2^18
            for (int step = 0; step < 2; ++step) {
                std::size_t k0, m0, nk;
                if (step == 0) {
                    k0 = 0;
                    nk = m / max_m;
                    m0 = max_m;
                } else {
                    k0 = m / max_m;
                    m0 = m % max_m;
                    nk = (m0 > 0u ? 1 : 0);
                }
                if (nk == 0) continue;
                vector<T *, Gpu> a_ps(k * nk, xpu_host, doCacheAlloc);
                vector<T *, Gpu> x_ps(k * nk, xpu_host, doCacheAlloc);
                auto a_ps_ptr = a_ps.data();
                auto x_ps_ptr = x_ps.data();
                auto a_ptr = a.data();
                auto x_ptr = x.data();
                launchHostKernel(
                    [=] {
                        for (std::size_t i = 0; i < k; ++i) {
                            for (std::size_t ki = k0, kii = 0; kii < nk; ++ki, ++kii) {
                                a_ps_ptr[i * nk + kii] = a_ptr + n * n * i;
                                x_ps_ptr[i * nk + kii] =
                                    x_ptr + n * m * i + (left_side ? n : 1u) * max_m * ki;
                            }
                        }
                    },
                    xpu_host);
                vector<T *, Gpu> a_ps_gpu = makeSure(a_ps, a.ctx(), doCacheAlloc),
                                 x_ps_gpu = makeSure(x_ps, a.ctx(), doCacheAlloc);
                gpuBlasCheck(cublasXtrsmBatched(
                    getGpuBlasHandle(a.ctx()), left_side ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, left_side ? n : m0,
                    left_side ? m0 : n, alpha, a_ps_gpu.data(), n, x_ps_gpu.data(),
                    left_side ? n : m, k * nk));
            }
#    else
            gpuBlasCheck(rocblas_trsm_strided_batched_ex(
                getGpuBlasHandle(a.ctx()), left_side ? rocblas_side_left : rocblas_side_right,
                rocblas_fill_upper, rocblas_operation_none, rocblas_diagonal_non_unit,
                left_side ? n : m, left_side ? m : n, &alpha, a.data(), n, n * n, x.data(),
                left_side ? n : m, n * m, k, nullptr, 0, 0, toCudaComputeType<T>()));
#    endif
            causalConnectTo(a.ctx(), x.ctx());
        }
#endif // SUPERBBLAS_USE_GPU

        /// Perform a\x -> x
        /// \param trans: either 'n', 't', or 'c'
        /// \param n: size of the matrix
        /// \param k: number of matrices to invert
        /// \param m: number of columns (if left_side) or rows (if !left_side) that x and y have

        template <typename T>
        void local_gesm(char trans, std::size_t n, std::size_t k, std::size_t m, vector<T, Cpu> a,
                        vector<T, Cpu> x) {

            tracker<Cpu> _t("local gesm (Cpu)", a.ctx());
            // Cost approximated as the cost of LU plus multiplying two triangular matrices
            _t.flops =
                ((double)n * n * n * 2 / 3 + (double)n * n * m) * k * multiplication_cost<T>::value;
            _t.memops = ((double)n * n * 3 + n * m * 4) * k * sizeof(T);

            using BLASINT = std::int64_t;
            T *ap = a.data(), *xp = x.data();

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
            int num_threads = omp_get_max_threads();
#else
            int num_threads = 1;
#endif
            BLASINT *ipivs = new BLASINT[n * num_threads];
            std::vector<int> info(num_threads, 0);

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp parallel
#endif
            {
#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
                int id = omp_get_thread_num();
#else
                int id = 0;
#endif
                BLASINT *ipiv = ipivs + n * id;
#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp for schedule(static)
#endif
                for (std::size_t i = 0; i < k; ++i) {
                    if (info[id] == 0) info[id] = xgetrf(n, n, ap + n * n * i, n, ipiv, Cpu{});
                    if (info[id] == 0)
                        info[id] =
                            xgetrs(trans, n, m, ap + n * n * i, n, ipiv, xp + n * m * i, n, Cpu{});
                }
            }
            for (int i : info) checkLapack(i, "gesm cpu");

            delete[] ipivs;
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_gesm(char trans, std::size_t n, std::size_t k, std::size_t m,
                        const vector<T, Gpu> &a, const vector<T, Gpu> &x) {

            if (n == 0 || k == 0 || m == 0) return;
            if (deviceId(a.ctx()) == CPU_DEVICE_ID)
                throw std::runtime_error(
                    "superbblas::detail::local_gesm: unsupported allocation device");
            check_same_device(a.ctx(), x.ctx());
            causalConnectTo(x.ctx(), a.ctx());

            tracker<Gpu> _t("local gesm (GPU)", a.ctx());
            // Cost approximated as the cost of LU plus multiplying two triangular matrices
            _t.flops =
                ((double)n * n * n * 2 / 3 + (double)n * n * m) * k * multiplication_cost<T>::value;
            _t.memops = ((double)n * n * 3 + n * m * 4) * k * sizeof(T);

            vector<int, Gpu> ipivs(k * n, a.ctx(), doCacheAlloc), info(k, a.ctx(), doCacheAlloc);
            auto xpu_host = a.ctx().toCpuPinned();
#    ifdef SUPERBBLAS_USE_CUDA
            vector<T *, Gpu> a_ps(k, xpu_host, doCacheAlloc), x_ps(k, xpu_host, doCacheAlloc);
            auto a_ps_ptr = a_ps.data();
            auto x_ps_ptr = x_ps.data();
            auto a_ptr = a.data();
            auto x_ptr = x.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i) a_ps_ptr[i] = a_ptr + n * n * i;
                    for (std::size_t i = 0; i < k; ++i) x_ps_ptr[i] = x_ptr + n * m * i;
                },
                xpu_host);
            vector<T *, Gpu> a_ps_gpu = makeSure(a_ps, a.ctx(), doCacheAlloc),
                             x_ps_gpu = makeSure(x_ps, a.ctx(), doCacheAlloc);
            gpuBlasCheck(cublasXgetrfBatched(getGpuBlasHandle(a.ctx()), n, a_ps_gpu.data(), n,
                                             ipivs.data(), info.data(), k));
#    else
            rocblasXgetrfStridedBatched(n, a.data(), n, n * n, ipivs.data(), n, info.data(), k,
                                        a.ctx());
#    endif
            vector<int, Gpu> info_cpu = makeSure(info, xpu_host, doCacheAlloc);
            auto info_cpu_ptr = info_cpu.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i)
                        checkLapack(info_cpu_ptr[i], "getrf gpu", true /* terminate */);
                },
                xpu_host);
#    ifdef SUPERBBLAS_USE_CUDA
            int info_getrs;
            gpuBlasCheck(cublasXgetrsBatched(getGpuBlasHandle(a.ctx()), toCublasTrans(trans), n, m,
                                             a_ps_gpu.data(), n, ipivs.data(), x_ps_gpu.data(), n,
                                             &info_getrs, k));
            checkLapack(info_getrs, "getrs gpu");
#    else
            rocblasXgetrsStridedBatched(trans, n, m, a.data(), n, n * n, ipivs.data(), n, x.data(),
                                        n, n * m, k, a.ctx());
#    endif
            causalConnectTo(a.ctx(), x.ctx());
        }
#endif // SUPERBBLAS_USE_GPU

        template <typename T>
        void local_inversion(std::size_t n, std::size_t k, const vector<T, Cpu> &a) {

            if (n == 0 || k == 0) return;

            tracker<Cpu> _t("local inv (Cpu)", Cpu{});
            // Cost approximated as the cost of LU plus multiplying two triangular matrices
            _t.flops = (double)n * n * n * (1 + 2. / 3) * k * multiplication_cost<T>::value;
            _t.memops = (double)n * n * 7 * k * sizeof(T);

            using BLASINT = std::int64_t;
            T *ap = a.data();

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
            int num_threads = omp_get_max_threads();
#else
            int num_threads = 1;
#endif
            BLASINT *ipivs = new BLASINT[n * num_threads];
            std::vector<int> info(num_threads, 0);

            T worksize = 0;
            checkLapack(xgetri(n, ap, n, ipivs, &worksize, (BLASINT)-1, Cpu{}), "getri cpu");
            BLASINT lwork = std::real(worksize);
            std::vector<T> work(num_threads * lwork);

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp parallel
#endif
            {
#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
                int id = omp_get_thread_num();
#else
                int id = 0;
#endif
                BLASINT *ipiv = ipivs + n * id;
                T *iwork = work.data() + lwork * id;
#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp for schedule(static)
#endif
                for (std::size_t i = 0; i < k; ++i) {
                    if (info[id] == 0) info[id] = xgetrf(n, n, ap + n * n * i, n, ipiv, Cpu{});
                    if (info[id] == 0)
                        info[id] = xgetri(n, ap + n * n * i, n, ipiv, iwork, lwork, Cpu{});
                }
            }
            for (int i : info) checkLapack(i, "getri cpu");

            delete[] ipivs;
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_inversion(std::size_t n, std::size_t k, const vector<T, Gpu> &a) {

            if (n == 0 || k == 0) return;
            if (deviceId(a.ctx()) == CPU_DEVICE_ID)
                throw std::runtime_error(
                    "superbblas::detail::local_inversion: unsupported allocation device");

            tracker<Gpu> _t("local gesm (GPU)", a.ctx());
            // Cost approximated as the cost of LU plus multiplying two triangular matrices
            _t.flops = (double)n * n * n * (1 + 2. / 3) * k * multiplication_cost<T>::value;
            _t.memops = (double)n * n * 7 * k * sizeof(T);

            vector<int, Gpu> ipivs(k * n, a.ctx(), doCacheAlloc), info(k, a.ctx(), doCacheAlloc);
            auto xpu_host = a.ctx().toCpuPinned();
#    ifdef SUPERBBLAS_USE_CUDA
            vector<T, Gpu> x(a.size(), a.ctx(), doCacheAlloc);
            vector<T *, Gpu> a_ps(k, xpu_host, doCacheAlloc), x_ps(k, xpu_host, doCacheAlloc);
            auto a_ps_ptr = a_ps.data();
            auto x_ps_ptr = x_ps.data();
            auto a_ptr = a.data();
            auto x_ptr = x.data();
            launchHostKernel(
                [=] {
                    for (std::size_t i = 0; i < k; ++i) a_ps_ptr[i] = a_ptr + n * n * i;
                    for (std::size_t i = 0; i < k; ++i) x_ps_ptr[i] = x_ptr + n * n * i;
                },
                xpu_host);
            vector<T *, Gpu> a_ps_gpu = makeSure(a_ps, a.ctx(), doCacheAlloc),
                             x_ps_gpu = makeSure(x_ps, a.ctx(), doCacheAlloc);
            gpuBlasCheck(cublasXgetrfBatched(getGpuBlasHandle(a.ctx()), n, a_ps_gpu.data(), n,
                                             ipivs.data(), info.data(), k));
            {
                vector<int, Gpu> info_cpu = makeSure(info, xpu_host, doCacheAlloc);
                auto info_cpu_ptr = info_cpu.data();
                launchHostKernel(
                    [=] {
                        for (std::size_t i = 0; i < k; ++i)
                            checkLapack(info_cpu_ptr[i], "getrf gpu", true /* terminate */);
                    },
                    xpu_host);
            }
            gpuBlasCheck(cublasXgetriBatched(getGpuBlasHandle(a.ctx()), n, a_ps_gpu.data(), n,
                                             ipivs.data(), x_ps_gpu.data(), n, info.data(), k));

            // Copy the inverted matrix into `a`
            copy_n(x.data(), x.ctx(), x.size(), a.data(), a.ctx());
#    else
            rocblasXgetrfStridedBatched(n, a.data(), n, n * n, ipivs.data(), n, info.data(), k,
                                        a.ctx());
            rocblasXgetriStridedBatched(n, a.data(), n, n * n, ipivs.data(), n, info.data(), k,
                                        a.ctx());
#    endif
            {
                vector<int, Gpu> info_cpu = makeSure(info, xpu_host, doCacheAlloc);
                auto info_cpu_ptr = info_cpu.data();
                launchHostKernel(
                    [=] {
                        for (std::size_t i = 0; i < k; ++i)
                            checkLapack(info_cpu_ptr[i], "getri gpu", true /* terminate */);
                    },
                    xpu_host);
            }
        }
#endif // SUPERBBLAS_USE_GPU

        template <typename T>
        void local_svd(std::size_t k, std::size_t m, std::size_t n, vector<T, Cpu> &a,
                       vector<typename the_real<T>::type, Cpu> &s, vector<T, Cpu> &u,
                       vector<T, Cpu> &vt) {

            // Zero dimension matrix may cause problems
            if (m == 0 || n == 0 || k == 0) return;

            tracker<Cpu> _t("local svd (Cpu)", a.ctx());

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
            int num_threads = omp_get_max_threads();
#else
            int num_threads = 1;
#endif

            // Call to know the optimal workspace
            T *ap = a.data();
            typename the_real<T>::type *sp = s.data();
            T *up = u.data();
            T *vtp = vt.data();
            T work0 = 0;
            T dummyr = 0;
            auto mv = std::min(m, n);
            checkLapack(
                xgesvd('S', 'S', m, n, ap, m, sp, up, m, vtp, mv, &work0, -1, &dummyr, Cpu{}),
                "gesvd cpu");
            std::size_t lwork = std::real(work0);

            std::vector<T> work(num_threads * lwork);
            auto lrwork = is_complex<T>::value ? 3 * n : 0;
            std::vector<T> rwork(num_threads * lrwork);
            std::vector<int> info(num_threads);

#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp parallel
#endif
            {
#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
                int id = omp_get_thread_num();
#else
                int id = 0;
#endif
                T *iwork = work.data() + lwork * id;
                T *irwork = rwork.data() + lrwork * id;
#if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#    pragma omp for schedule(static)
#endif
                for (std::size_t i = 0; i < k; ++i) {
                    if (info[id] == 0)
                        info[id] =
                            xgesvd('S', 'S', m, n, ap + m * n * i, m, sp + mv * i, up + m * mv * i,
                                   m, vtp + mv * n * i, mv, iwork, lwork, irwork, Cpu{});
                }
            }
            for (int i : info) checkLapack(i, "gesvd cpu");

            // Conjugate vt
            conj(vt);
        }

#ifdef SUPERBBLAS_USE_GPU
        template <typename T>
        void local_svd(std::size_t k, std::size_t m, std::size_t n, vector<T, Gpu> &a,
                       vector<typename the_real<T>::type, Gpu> &s, vector<T, Gpu> &u,
                       vector<T, Gpu> &vt) {

            // Zero dimension matrix may cause problems
            if (m == 0 || n == 0 || k == 0) return;

            tracker<Gpu> _t("local svd (Gpu)", a.ctx());

            if (deviceId(a.ctx()) == CPU_DEVICE_ID)
                throw std::runtime_error(
                    "superbblas::detail::local_svd: unsupported allocation device");
            check_same_device(a.ctx(), u.ctx());
            check_same_device(a.ctx(), s.ctx());
            check_same_device(a.ctx(), vt.ctx());
            causalConnectTo(u.ctx(), a.ctx());
            causalConnectTo(s.ctx(), a.ctx());
            causalConnectTo(vt.ctx(), a.ctx());

            vector<int, Gpu> info(k, a.ctx(), doCacheAlloc);
            auto xpu_host = a.ctx().toCpuPinned();
            auto rank = std::min(m, n);
            bool info_checked = false;
#    ifdef SUPERBBLAS_USE_CUDA
            int lwork = 0;
            gpuSolverCheck(cusolverDnXgesvdaStridedBatched_bufferSize(
                getGpuSolverHandle(a.ctx()), CUSOLVER_EIG_MODE_VECTOR, rank, m, n, a.data(), m,
                m * n, s.data(), rank, u.data(), m, m * rank, vt.data(), rank, rank * n, &lwork,
                k));
            vector<T, Gpu> work(lwork, a.ctx(), doCacheAlloc);
            gpuSolverCheck(cusolverDnXgesvdaStridedBatched(
                getGpuSolverHandle(a.ctx()), CUSOLVER_EIG_MODE_VECTOR, rank, m, n, a.data(), m,
                m * n, s.data(), rank, u.data(), m, m * rank, vt.data(), rank, rank * n,
                work.data(), work.size(), info.data(), nullptr, k));
#    else
            int lwork = 2 * rank;
            vector<typename the_real<T>::type, Gpu> work(lwork * k, a.ctx(), doCacheAlloc);
            zero_n(work.data(), work.size(), work.ctx());
            rocsolverXgesvdStridedBatched(rocblas_svect_singular, rocblas_svect_singular, m, n,
                                          a.data(), m, m * n, s.data(), rank, u.data(), m, m * rank,
                                          vt.data(), rank, rank * n, work.data(), lwork,
                                          rocblas_outofplace, info.data(), k, a.ctx());
            work.clear();
            auto info_cpu = makeSure(info, Cpu{});
            bool all_ok = true;
            for (std::size_t i = 0; i < k; ++i) all_ok &= (info_cpu[i] == 0);
            if (all_ok) {
                info_checked = true;
            } else {
                rocsolverXgesvdStridedBatched(rocblas_svect_singular, rocblas_svect_singular, m, n,
                                              a.data(), m, m * n, s.data(), rank, u.data(), m,
                                              m * rank, vt.data(), rank, rank * n, work.data(),
                                              lwork, rocblas_inplace, info.data(), k, a.ctx());
            }
#    endif // SUPERBBLAS_USE_CUDA
            if (!info_checked) {
                vector<int, Gpu> info_cpu = makeSure(info, xpu_host, doCacheAlloc);
                auto info_cpu_ptr = info_cpu.data();
                launchHostKernel(
                    [=] {
                        for (std::size_t i = 0; i < k; ++i)
                            checkLapack(info_cpu_ptr[i], "gesvd gpu", true /* terminate */);
                    },
                    xpu_host);
            }

            // Conjugate vt
            conj(vt);

            causalConnectTo(a.ctx(), u.ctx());
            causalConnectTo(a.ctx(), s.ctx());
            causalConnectTo(a.ctx(), vt.ctx());
        }
#endif // SUPERBBLAS_USE_GPU

        /// Get the output partition
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t N>
        Proc_ranges<N> get_dense_output_partition(Proc_ranges<N> p0, const Coor<N> &dim,
                                                  const Order<N> &o0, const Order<N> &o_r,
                                                  unsigned int num_mat_dims, CoorOrder co) {
            // Find partition on cache
            using Key = std::tuple<Proc_ranges<N>, Coor<N>, PairPerms<N, N>, unsigned int>;
            struct cache_tag {};
            auto cache = getCache<Key, Proc_ranges<N>, TupleHash<Key>, cache_tag>(Cpu{});
            Key key{p0, dim, get_perms(o0, o_r), num_mat_dims};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second.value;

            // Create partition
            Coor<N> perm0 = find_permutation<N, N>(o0, o_r);
            Coor<N> dimr = reorder_coor<N, N>(dim, perm0, 1);
            Proc_ranges<N> pr(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                pr[i].resize(p0[i].size());
                for (unsigned int j = 0; j < p0[i].size(); ++j) {
                    if (volume(p0[i][j][1]) == 0) {
                        pr[i][j][0] = pr[i][j][1] = Coor<N>{};
                    } else {
                        pr[i][j][0] = reorder_coor<N, N>(p0[i][j][0], perm0);
                        pr[i][j][1] = reorder_coor<N, N>(p0[i][j][1], perm0, 1);
                        if (co == FastToSlow) {
                            for (unsigned int k = 0; k < num_mat_dims; ++k) pr[i][j][0][k] = 0;
                            for (unsigned int k = 0; k < num_mat_dims; ++k)
                                pr[i][j][1][k] = dimr[k];
                        } else {
                            for (unsigned int k = 0, k0 = N - 1; k < num_mat_dims; ++k, --k0)
                                pr[i][j][0][k0] = 0;
                            for (unsigned int k = 0, k0 = N - 1; k < num_mat_dims; ++k, --k0)
                                pr[i][j][1][k0] = dimr[k0];
                        }
                    }
                }
            }

            cache.insert(key, pr, storageSize(pr));

            return pr;
        }

        /// Return the tensor rearranged in the right ordering and distribution for doing Cholesky
        /// factorization/application
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param v0: data for the first operator
        /// \param orows: labels on the rows
        /// \param ocols: labels on the columns
        /// \param co: coordinate linearization order
        /// \param force_copy: whether to NOT avoid copy if the partition is the same
        /// \return tuple{pw,ow,vw,n}
        /// \param pw: (out) partitioning of the output tensor in consecutive ranges
        /// \param ow: (out) ordering of the output tensor
        /// \param vw: (out) data for the output operator
        /// \param n: (out) the number of rows/columns

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        std::tuple<Proc_ranges<N>, Coor<N>, Order<N>, Components_tmpl<N, T, XPU0, XPU1>,
                   std::size_t, std::size_t>
        prepare_for_cholesky(const Proc_ranges<N> &p, const Coor<N> &dim, const Order<N> &o,
                             const Components_tmpl<N, T, XPU0, XPU1> &v, const char *orows,
                             const char *ocols, Comm comm, CoorOrder co, bool force_copy = false,
                             bool check_square = true) {

            // Check the orderings

            const std::string orows_(orows), ocols_(ocols);
            for (char c : orows_) {
                if (std::find(ocols_.begin(), ocols_.end(), c) != ocols_.end())
                    throw std::runtime_error("Invalid `orows' and `ocols': they share labels");
                if (std::find(o.begin(), o.end(), c) == o.end())
                    throw std::runtime_error("Invalid `orows': invalid labels");
            }
            for (char c : ocols_)
                if (std::find(o.begin(), o.end(), c) == o.end())
                    throw std::runtime_error("Invalid `ocols': invalid labels");

            // Generate the working ordering

            std::size_t const nrows = orows_.size();
            std::size_t const ncols = ocols_.size();
            std::vector<char> ot;
            for (char c : o)
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end())
                    ot.push_back(c);
            std::size_t const nk = ot.size();
            Order<N> ow = concat<N>(orows_, ocols_, ot, co);

            // Check that number of rows and columns is the same
            Coor<N> perm0 = find_permutation<N, N>(o, ow);
            Coor<N> dimw = reorder_coor<N, N>(dim, perm0, 1);
            std::size_t m = (co == FastToSlow ? volume<N>(dimw.begin(), dimw.begin() + nrows)
                                              : volume<N>(dimw.begin() + nk + ncols,
                                                          dimw.begin() + nk + ncols + nrows));
            std::size_t n =
                (co == FastToSlow ? volume<N>(dimw.begin() + nrows, dimw.begin() + nrows + ncols)
                                  : volume<N>(dimw.begin() + nk, dimw.begin() + nk + ncols));
            if (check_square && m != n)
                throw std::runtime_error("cholesky: the matrices to factorize should be square");

            // Generate the working partition
            Proc_ranges<N> pw = get_dense_output_partition(p, dim, o, ow, nrows + ncols, co);
            Components_tmpl<N, T, XPU0, XPU1> vw = reorder_tensor(
                p, o, {{}}, dim, dim, v, pw, dimw, ow, comm, co, force_copy, doCacheAlloc);

            return {pw, dimw, ow, vw, m, n};
        }

        /// Get the output partition
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        std::pair<Proc_ranges<Ndo>, Coor<Ndo>>
        get_output_partition(const Proc_ranges<Nd0> &p0, const Coor<Nd0> &dim0,
                             const Order<Nd0> &o0, const Proc_ranges<Nd1> &p1,
                             const Coor<Nd1> &dim1, const Order<Nd1> &o1, const Order<Ndo> &o_r,
                             bool report_inconsistencies = true) {
            assert(p0.size() == p1.size());

            // Find partition on cache
            using Key = std::tuple<Proc_ranges<Nd0>, Coor<Nd0>, Proc_ranges<Nd1>, Coor<Nd1>,
                                   PairPerms<Nd0, Nd1>, PairPerms<Nd0, Ndo>, PairPerms<Nd1, Ndo>>;
            struct cache_tag {};
            auto cache =
                getCache<Key, std::pair<Proc_ranges<Ndo>, Coor<Ndo>>, TupleHash<Key>, cache_tag>(
                    Cpu{});
            Key key{p0, dim0, p1, dim1, get_perms(o0, o1), get_perms(o0, o_r), get_perms(o1, o_r)};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second.value;

            // Create partition
            Proc_ranges<Ndo> pr(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                pr[i].resize(p0[i].size());
                for (unsigned int j = 0; j < p0[i].size(); ++j) {
                    pr[i][j][0] = get_dimensions<Nd0, Nd1, Ndo>(o0, p0[i][j][0], o1, p1[i][j][0],
                                                                o_r, report_inconsistencies);
                    pr[i][j][1] = get_dimensions<Nd0, Nd1, Ndo>(o0, p0[i][j][1], o1, p1[i][j][1],
                                                                o_r, report_inconsistencies);
                    if (volume(pr[i][j][1]) == 0) pr[i][j][0] = pr[i][j][1] = Coor<Ndo>{{}};
                }
            }
            Coor<Ndo> dimr =
                get_dimensions<Nd0, Nd1, Ndo>(o0, dim0, o1, dim1, o_r, report_inconsistencies);
            cache.insert(key, {pr, dimr}, storageSize(pr));

            return {pr, dimr};
        }

        /// Compute the Cholesky factorization of several matrices
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param ncomponents0: number of consecutive components in each MPI rank
        /// \param o0: dimension labels for the first operator
        /// \param v0: data for the first operator
        /// \param orows: labels on the rows
        /// \param ocols: labels on the columns
        /// \param ctx0: context for each data pointer in v0
        /// \param session: concurrent calls should have different session

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        void cholesky(const Proc_ranges<N> &p, const Coor<N> &dim, const Order<N> &o,
                      const Components_tmpl<N, T, XPU0, XPU1> &v, const char *orows,
                      const char *ocols, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : v.first) sync(i.it.ctx());
                for (const auto &i : v.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed cholesky", Cpu{});

            // Reorder the tensor so can be processed by cholesky
            auto t = prepare_for_cholesky(p, dim, o, v, orows, ocols, comm, co);
            Proc_ranges<N> &pw = std::get<0>(t);
            const Coor<N> &dimw = std::get<1>(t);
            Order<N> &ow = std::get<2>(t);
            Components_tmpl<N, T, XPU0, XPU1> &vw = std::get<3>(t);
            std::size_t n = std::get<4>(t);

            // Do cholesky on the local pieces
            for (unsigned int i = 0; i < vw.first.size(); ++i) {
                const unsigned int componentId = vw.first[i].componentId;
                std::size_t ki = volume(pw[comm.rank][componentId][1]) / n / n;
                local_cholesky(n, ki, vw.first[i].it);
            }
            for (unsigned int i = 0; i < vw.second.size(); ++i) {
                const unsigned int componentId = vw.second[i].componentId;
                std::size_t ki = volume(pw[comm.rank][componentId][1]) / n / n;
                local_cholesky(n, ki, vw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<N, N, T>(T{1}, pw, {{}}, dimw, dimw, ow, toConst(vw), p, {{}}, dim, o, v, comm,
                          EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : v.first) sync(i.it.ctx());
                for (const auto &i : v.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void trsm(T alpha, const Proc_ranges<Nc> &pc, const Coor<Nc> &dimc, const Order<Nc> &oc,
                  const Components_tmpl<Nc, T, XPU0, XPU1> &vc, const char *orows,
                  const char *ocols, const Proc_ranges<Nx> &px, const Coor<Nx> &dimx,
                  const Order<Nx> &ox, const Components_tmpl<Nx, T, XPU0, XPU1> &vx,
                  const Proc_ranges<Ny> &py, const Coor<Ny> &dimy, const Order<Ny> &oy,
                  const Components_tmpl<Ny, T, XPU0, XPU1> &vy, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vc.first) sync(i.it.ctx());
                for (const auto &i : vc.second) sync(i.it.ctx());
                for (const auto &i : vx.first) sync(i.it.ctx());
                for (const auto &i : vx.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed trsm", Cpu{});

            // Check the compatibility of the tensors
            if (!check_dimensions(oc, dimc, ox, dimx, oy, dimy))
                throw std::runtime_error("some dimension does not match");

            // Check that v0 and v1 have the same components and on the same device
            if (!check_components_compatibility(vc, vx) || !check_components_compatibility(vx, vy))
                throw std::runtime_error(
                    "trsm: the given tensors don't have the same number of components "
                    "or they don't follow the same order on the devices");

            // Figure out whether x contracts with the rows or the columns of the cholesky factor

            const std::string orows_(orows), ocols_(ocols);
            bool contract_rows = false, contract_rows_set = false;
            bool fail = false;
            for (char c : ox) {
                if (std::find(ocols_.begin(), ocols_.end(), c) != ocols_.end()) {
                    if (!contract_rows_set) {
                        contract_rows = false;
                        contract_rows_set = true;
                    } else if (contract_rows != false) {
                        fail = true;
                        break;
                    }
                }
                if (std::find(orows_.begin(), orows_.end(), c) != orows_.end()) {
                    if (!contract_rows_set) {
                        contract_rows = true;
                        contract_rows_set = true;
                    } else if (contract_rows != true) {
                        fail = true;
                        break;
                    }
                }
            }
            if (fail || !contract_rows_set)
                throw std::runtime_error("trsm: cannot contract a mix of rows and column labels");

            // Check that all rows and column labels are in the x and y orderings

            for (char c : orows_) {
                if (contract_rows && std::find(ox.begin(), ox.end(), c) == ox.end()) fail = true;
                if (!contract_rows && std::find(oy.begin(), oy.end(), c) == oy.end()) fail = true;
            }
            for (char c : ocols_) {
                if (!contract_rows && std::find(ox.begin(), ox.end(), c) == ox.end()) fail = true;
                if (contract_rows && std::find(oy.begin(), oy.end(), c) == oy.end()) fail = true;
            }
            if (fail) throw std::runtime_error("trsm: missing labels to contract");

            // Reorder the tensor so can be processed by cholesky
            auto t = prepare_for_cholesky(pc, dimc, oc, toNonConst(vc), orows, ocols, comm, co);
            Proc_ranges<Nc> &pcw = std::get<0>(t);
            Coor<Nc> &dimcw = std::get<1>(t);
            Order<Nc> &ocw = std::get<2>(t);
            Components_tmpl<Nc, T, XPU0, XPU1> &vcw = std::get<3>(t);
            std::size_t r = std::get<4>(t); // number of rows/columns

            // Find the labels

            std::vector<char> ot, on;
            for (char c : oc)
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end())
                    ot.push_back(c);
            for (char c : ox)
                if (std::find(oc.begin(), oc.end(), c) == oc.end()) on.push_back(c);

            // Compute the ordering for tensors x and y
            // If contracting rows, then X/C -> Y => (n,r,t) x (r,c,t) -> (n,c,t).
            // Otherwise C\X -> Y => (r,c,t) x (c,n,t) -> (r,n,t)

            Order<Nx> oxw =
                contract_rows ? concat<Nx>(on, orows_, ot, co) : concat<Nx>(ocols_, on, ot, co);
            Order<Ny> oyw =
                contract_rows ? concat<Ny>(on, ocols_, ot, co) : concat<Ny>(orows_, on, ot, co);

            // Generate the working tensors

            auto tx_ = get_output_partition(pcw, dimcw, ocw, px, dimx, ox, oxw, false);
            Proc_ranges<Nx> &pxw = tx_.first;
            const Coor<Nx> &dimxw = tx_.second;
            Components_tmpl<Nx, T, XPU0, XPU1> vxw =
                reorder_tensor(px, ox, {{}}, dimx, dimx, toNonConst(vx), pxw, dimxw, oxw, comm, co,
                               true /* Force copy */, doCacheAlloc);
            auto ty_ = get_output_partition(pcw, dimcw, ocw, pxw, dimxw, oxw, oyw);
            Proc_ranges<Ny> &pyw = ty_.first;
            const Coor<Ny> &dimyw = ty_.second;
            Components_tmpl<Ny, T, XPU0, XPU1> vyw = reshape(pyw, vxw, comm);

            // Do the contraction of the local pieces

            for (unsigned int i = 0; i < vcw.first.size(); ++i) {
                const unsigned int componentId = vcw.first[i].componentId;
                std::size_t ki = volume(pcw[comm.rank][componentId][1]) / r / r;
                if (ki == 0) continue;
                std::size_t ni =
                    volume(pxw[comm.rank][componentId][1]) / r / ki; // rows/columns of x and y
                local_trsm(!contract_rows, r, ki, ni, alpha, vcw.first[i].it, vxw.first[i].it);
            }
            for (unsigned int i = 0; i < vcw.second.size(); ++i) {
                const unsigned int componentId = vcw.second[i].componentId;
                std::size_t ki = volume(pcw[comm.rank][componentId][1]) / r / r;
                if (ki == 0) continue;
                std::size_t ni =
                    volume(pxw[comm.rank][componentId][1]) / r / ki; // rows/columns of x and y
                local_trsm(!contract_rows, r, ki, ni, alpha, vcw.second[i].it, vxw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<Ny, Ny, T>(T{1}, pyw, {{}}, dimyw, dimyw, oyw, toConst(vyw), py, {{}}, dimy, oy,
                            vy, comm, EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : vy.first) sync(i.it.ctx());
                for (const auto &i : vy.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void gesm(T alpha, const Proc_ranges<Nc> &pc, const Coor<Nc> dimc, const Order<Nc> &oc,
                  const Components_tmpl<Nc, T, XPU0, XPU1> &vc, const char *orows,
                  const char *ocols, const Proc_ranges<Nx> &px, const Coor<Nx> &dimx,
                  const Order<Nx> &ox, const Components_tmpl<Nx, T, XPU0, XPU1> &vx,
                  const Proc_ranges<Ny> &py, const Coor<Ny> &dimy, const Order<Ny> &oy,
                  const Components_tmpl<Ny, T, XPU0, XPU1> &vy, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : vc.first) sync(i.it.ctx());
                for (const auto &i : vc.second) sync(i.it.ctx());
                for (const auto &i : vx.first) sync(i.it.ctx());
                for (const auto &i : vx.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed gesm", Cpu{});

            // Check the compatibility of the tensors
            if (!check_dimensions(oc, dimc, ox, dimx, oy, dimy))
                throw std::runtime_error("some dimension does not match");

            // Check that v0 and v1 have the same components and on the same device
            if (!check_components_compatibility(vc, vx) || !check_components_compatibility(vx, vy))
                throw std::runtime_error(
                    "gesm: the given tensors don't have the same number of components "
                    "or they don't follow the same order on the devices");

            // Figure out whether x contracts with the rows or the columns of the matrix to invert

            const std::string orows_(orows), ocols_(ocols);
            bool contract_rows = false, contract_rows_set = false;
            bool fail = false;
            for (char c : ox) {
                if (std::find(ocols_.begin(), ocols_.end(), c) != ocols_.end()) {
                    if (!contract_rows_set) {
                        contract_rows = false;
                        contract_rows_set = true;
                    } else if (contract_rows != false) {
                        fail = true;
                        break;
                    }
                }
                if (std::find(orows_.begin(), orows_.end(), c) != orows_.end()) {
                    if (!contract_rows_set) {
                        contract_rows = true;
                        contract_rows_set = true;
                    } else if (contract_rows != true) {
                        fail = true;
                        break;
                    }
                }
            }
            if (fail || !contract_rows_set)
                throw std::runtime_error("gesm: cannot contract a mix of rows and column labels");

            // For now, only supported to contract with columns

            if (contract_rows)
                throw std::runtime_error("gesm: unsupported to contract with row labels");

            // Check that all rows and column labels are in the x and y orderings

            for (char c : orows_) {
                if (contract_rows && std::find(ox.begin(), ox.end(), c) == ox.end()) fail = true;
                if (!contract_rows && std::find(oy.begin(), oy.end(), c) == oy.end()) fail = true;
            }
            for (char c : ocols_) {
                if (!contract_rows && std::find(ox.begin(), ox.end(), c) == ox.end()) fail = true;
                if (contract_rows && std::find(oy.begin(), oy.end(), c) == oy.end()) fail = true;
            }
            if (fail) throw std::runtime_error("gesm: missing labels to contract");

            // Reorder the tensor so can be processed by cholesky
            auto t = prepare_for_cholesky(pc, dimc, oc, toNonConst(vc), orows, ocols, comm, co,
                                          true /* Force copy */);
            Proc_ranges<Nc> &pcw = std::get<0>(t);
            Coor<Nc> &dimcw = std::get<1>(t);
            Order<Nc> &ocw = std::get<2>(t);
            Components_tmpl<Nc, T, XPU0, XPU1> &vcw = std::get<3>(t);
            std::size_t r = std::get<4>(t); // number of rows/columns

            // Find the labels

            std::vector<char> ot, on;
            for (char c : oc)
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end())
                    ot.push_back(c);
            for (char c : ox)
                if (std::find(oc.begin(), oc.end(), c) == oc.end()) on.push_back(c);

            // Compute the ordering for tensors x and y
            // If contracting rows, then X/C -> Y => (n,r,t) x (r,c,t) -> (n,c,t).
            // Otherwise C\X -> Y => (r,c,t) x (c,n,t) -> (r,n,t)

            Order<Nx> oxw =
                contract_rows ? concat<Nx>(on, orows_, ot, co) : concat<Nx>(ocols_, on, ot, co);
            Order<Ny> oyw =
                contract_rows ? concat<Ny>(on, ocols_, ot, co) : concat<Ny>(orows_, on, ot, co);

            // Generate the working tensors

            auto tx_ = get_output_partition(pcw, dimcw, ocw, px, dimx, ox, oxw, false);
            Proc_ranges<Nx> &pxw = tx_.first;
            const Coor<Nx> &dimxw = tx_.second;
            Components_tmpl<Nx, T, XPU0, XPU1> vxw =
                reorder_tensor(px, ox, {{}}, dimx, dimx, toNonConst(vx), pxw, dimxw, oxw, comm, co,
                               true /* Force copy */, doCacheAlloc);
            auto ty_ = get_output_partition(pcw, dimcw, ocw, pxw, dimxw, oxw, oyw);
            Proc_ranges<Ny> &pyw = ty_.first;
            const Coor<Ny> &dimyw = ty_.second;

            // Do the contraction of the local pieces

            for (unsigned int i = 0; i < vcw.first.size(); ++i) {
                const unsigned int componentId = vcw.first[i].componentId;
                std::size_t ki = volume(pcw[comm.rank][componentId][1]) / r / r;
                if (ki == 0) continue;
                std::size_t ni =
                    volume(pxw[comm.rank][componentId][1]) / r / ki; // rows/columns of x and y
                local_gesm('N', r, ki, ni, vcw.first[i].it, vxw.first[i].it);
            }
            for (unsigned int i = 0; i < vcw.second.size(); ++i) {
                const unsigned int componentId = vcw.second[i].componentId;
                std::size_t ki = volume(pcw[comm.rank][componentId][1]) / r / r;
                if (ki == 0) continue;
                std::size_t ni =
                    volume(pxw[comm.rank][componentId][1]) / r / ki; // rows/columns of x and y
                local_gesm('N', r, ki, ni, vcw.second[i].it, vxw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<Ny, Ny, T>(alpha, pyw, {{}}, dimyw, dimyw, oyw, toConst(vxw), py, {{}}, dimy, oy,
                            vy, comm, EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : vy.first) sync(i.it.ctx());
                for (const auto &i : vy.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        /// Compute the inversion of several matrices
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param v0: data for the first operator
        /// \param orows: labels on the rows
        /// \param ocols: labels on the columns

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        void inversion(const Proc_ranges<N> &p, const Coor<N> &dim, const Order<N> &o,
                       const Components_tmpl<N, T, XPU0, XPU1> &v, const char *orows,
                       const char *ocols, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : v.first) sync(i.it.ctx());
                for (const auto &i : v.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed cholesky", Cpu{});

            // Reorder the tensor so can be processed by local_inversion
            auto t = prepare_for_cholesky(p, dim, o, v, orows, ocols, comm, co);
            Proc_ranges<N> &pw = std::get<0>(t);
            const Coor<N> &dimw = std::get<1>(t);
            Order<N> &ow = std::get<2>(t);
            Components_tmpl<N, T, XPU0, XPU1> &vw = std::get<3>(t);
            std::size_t n = std::get<4>(t);

            // Do cholesky on the local pieces
            for (unsigned int i = 0; i < vw.first.size(); ++i) {
                const unsigned int componentId = vw.first[i].componentId;
                std::size_t ki = volume(pw[comm.rank][componentId][1]) / n / n;
                local_inversion(n, ki, vw.first[i].it);
            }
            for (unsigned int i = 0; i < vw.second.size(); ++i) {
                const unsigned int componentId = vw.second[i].componentId;
                std::size_t ki = volume(pw[comm.rank][componentId][1]) / n / n;
                local_inversion(n, ki, vw.second[i].it);
            }

            // Copy the working tensor into the given tensor
            copy<N, N, T>(T{1}, pw, {{}}, dimw, dimw, ow, toConst(vw), p, {{}}, dim, o, v, comm,
                          EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : v.first) sync(i.it.ctx());
                for (const auto &i : v.second) sync(i.it.ctx());
                barrier(comm);
            }
        }

        /// Return whether the two strings have the same labels
        //// \param x: one of the strings
        //// \param y: the other string

        template <typename T, typename Q> bool is_a_permutation(const T &x, const Q &y) {
            return x.size() == y.size() && std::is_permutation(x.begin(), x.end(), y.begin());
        }

        template <std::size_t Na, std::size_t Nx, std::size_t Ns, std::size_t Ny, typename T,
                  typename Comm, typename XPU0, typename XPU1>
        void svd(T alpha, const Proc_ranges<Na> &pa, const Coor<Na> dima, const Order<Na> &oa,
                 const Components_tmpl<Na, T, XPU0, XPU1> &va, const char *orows, const char *ocols,
                 const Proc_ranges<Nx> &px, const Coor<Nx> &dimx, const Order<Nx> &ox,
                 const Components_tmpl<Nx, T, XPU0, XPU1> &vx, const Proc_ranges<Ns> &ps,
                 const Coor<Ns> &dims, const Order<Ns> &os,
                 const Components_tmpl<Ns, typename the_real<T>::type, XPU0, XPU1> &vs,
                 const Proc_ranges<Ny> &py, const Coor<Ny> &dimy, const Order<Ny> &oy,
                 const Components_tmpl<Ny, T, XPU0, XPU1> &vy, Comm comm, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                for (const auto &i : va.first) sync(i.it.ctx());
                for (const auto &i : va.second) sync(i.it.ctx());
                for (const auto &i : vx.first) sync(i.it.ctx());
                for (const auto &i : vx.second) sync(i.it.ctx());
                for (const auto &i : vs.first) sync(i.it.ctx());
                for (const auto &i : vs.second) sync(i.it.ctx());
                for (const auto &i : vy.first) sync(i.it.ctx());
                for (const auto &i : vy.second) sync(i.it.ctx());
                barrier(comm);
            }

            tracker<Cpu> _t("distributed svd", Cpu{});

            // Check the compatibility of the tensors
            if (!check_dimensions(oa, dima, ox, dimx, os, dims) ||
                !check_dimensions(oa, dima, oy, dimy, oy, dimy))
                throw std::runtime_error("some dimension does not match");

            // Check that v0 and v1 have the same components and on the same device
            if (!check_components_compatibility(va, vx) ||
                !check_components_compatibility(va, vy) || !check_components_compatibility(va, vs))
                throw std::runtime_error(
                    "svd: the given tensors don't have the same number of components "
                    "or they don't follow the same order on the devices");

            // Get the batch labels
            const std::string orows_(orows), ocols_(ocols);
            std::string t_labels;
            for (char c : oa) {
                if (std::find(ocols_.begin(), ocols_.end(), c) == ocols_.end() &&
                    std::find(orows_.begin(), orows_.end(), c) == orows_.end()) {
                    t_labels.push_back(c);
                }
            }

            // Get the singular value index label
            char n_label = 0;
            bool first_found = true;
            for (char c : ox) {
                if (std::find(oa.begin(), oa.end(), c) == oa.end()) {
                    if (!first_found)
                        throw std::runtime_error(
                            "svd: invalid labels for the singular value index");
                    n_label = c;
                    first_found = false;
                }
            }
            if (first_found)
                throw std::runtime_error("svd: invalid labels for the singular value index");

            // Check that all labels of x are made of t_labels, orows and n
            if (!is_a_permutation(ox, concat(t_labels, orows_, n_label)))
                throw std::runtime_error("svd: invalid labels for tensor x");

            // Check that all labels of s are made of t_labels and n
            if (!is_a_permutation(os, concat(t_labels, std::string(), n_label)))
                throw std::runtime_error("svd: invalid labels for tensor s");

            // Check that all labels of y are made of t_labels, ocols and n
            if (!is_a_permutation(oy, concat(t_labels, ocols_, n_label)))
                throw std::runtime_error("svd: invalid labels for tensor y");

            // Reorder the tensor so can be processed by cholesky
            auto t =
                prepare_for_cholesky(pa, dima, oa, toNonConst(va), orows, ocols, comm, co,
                                     true /* Force copy */, false /* don't check being square */);

            Proc_ranges<Na> &paw = std::get<0>(t);
            Coor<Na> &dimaw = std::get<1>(t);
            Order<Na> &oaw = std::get<2>(t);
            Components_tmpl<Na, T, XPU0, XPU1> &vaw = std::get<3>(t);
            std::size_t rm = std::get<4>(t); // number of rows
            std::size_t rn = std::get<5>(t); // number of columns

            // Check that the singular values have the right dimensions
            if ((std::size_t)dims[std::find(os.begin(), os.end(), n_label) - os.begin()] !=
                std::min(rm, rn))
                throw std::runtime_error(
                    "svd: the given singular values tensor does not have the proper dimensions");

            // Apply alpha
            if (alpha != T{1}) {
                copy<Na, Na, T>(alpha, paw, {{}}, dimaw, dimaw, oaw, toConst(vaw), paw, {{}}, dimaw,
                                oaw, vaw, comm, EWOp::Copy{}, co);
            }

            // Compute the ordering for tensors x, s and y: (rows,n,t), (n,t), (n,rows,t)
            std::string on{n_label};
            Order<Nx> oxw = concat<Nx>(orows_, on, t_labels, co);
            Order<Ns> osw = concat<Ns>(std::string(), on, t_labels, co);
            Order<Ny> oyw = concat<Ny>(ocols_, on, t_labels, co);

            // Generate the working tensors

            auto tx_ = get_output_partition(paw, dimaw, oaw, px, dimx, ox, oxw,
                                            false /* do not report inconsistencies */);
            Proc_ranges<Nx> &pxw = tx_.first;
            const Coor<Nx> &dimxw = tx_.second;
            auto vwx = like_this_components(pxw, vaw, comm, doCacheAlloc);

            using Tr = typename the_real<T>::type;
            auto ts_ = get_output_partition(paw, dimaw, oaw, pxw, dimxw, oxw, osw);
            Proc_ranges<Ns> &psw = ts_.first;
            const Coor<Ns> &dimsw = ts_.second;
            auto vws = like_this_components_with_type<Tr>(psw, vaw, comm, doCacheAlloc);

            auto ty_ = get_output_partition(paw, dimaw, oaw, pxw, dimxw, oxw, oyw);
            Proc_ranges<Ny> &pyw = ty_.first;
            const Coor<Ny> &dimyw = ty_.second;
            auto vwy = like_this_components(pyw, vaw, comm, doCacheAlloc);

            // Do the svd of the local pieces

            for (unsigned int i = 0; i < vaw.first.size(); ++i) {
                const unsigned int componentId = vaw.first[i].componentId;
                std::size_t ki = volume(paw[comm.rank][componentId][1]) / rm / rn;
                if (ki == 0) continue;
                local_svd(ki, rm, rn, vaw.first[i].it, vws.first[i].it, vwx.first[i].it,
                          vwy.first[i].it);
            }
            for (unsigned int i = 0; i < vaw.second.size(); ++i) {
                const unsigned int componentId = vaw.second[i].componentId;
                std::size_t ki = volume(paw[comm.rank][componentId][1]) / rm / rn;
                if (ki == 0) continue;
                local_svd(ki, rm, rn, vaw.second[i].it, vws.second[i].it, vwx.second[i].it,
                          vwy.second[i].it);
            }

            // Copy the working tensors into the given tensors
            copy<Ns, Ns, Tr>(Tr{1}, psw, {{}}, dimsw, dimsw, osw, toConst(vws), ps, {{}}, dims, os,
                             vs, comm, EWOp::Copy{}, co);
            copy<Nx, Nx, T>(T{1}, pxw, {{}}, dimxw, dimxw, oxw, toConst(vwx), px, {{}}, dimx, ox,
                            vx, comm, EWOp::Copy{}, co);
            copy<Ny, Ny, T>(T{1}, pyw, {{}}, dimyw, dimyw, oyw, toConst(vwy), py, {{}}, dimy, oy,
                            vy, comm, EWOp::Copy{}, co);

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : vx.first) sync(i.it.ctx());
                for (const auto &i : vx.second) sync(i.it.ctx());
                for (const auto &i : vs.first) sync(i.it.ctx());
                for (const auto &i : vs.second) sync(i.it.ctx());
                for (const auto &i : vy.first) sync(i.it.ctx());
                for (const auto &i : vy.second) sync(i.it.ctx());
                for (const auto &i : vy.first) sync(i.it.ctx());
                for (const auto &i : vy.second) sync(i.it.ctx());
                barrier(comm);
            }
        }
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Compute the Cholesky factorization of several matrices, returning the upper triangular matrix
    /// \param p: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param o: dimension labels for the first operator
    /// \param v: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctx: context for each data pointer in v0
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t N, typename T>
    void cholesky(const PartitionItem<N> *p, const Coor<N> &dim, int ncomponents, const char *o,
                  T **v, const char *orows, const char *ocols, const Context *ctx, MPI_Comm mpicomm,
                  CoorOrder co, Session session = 0) {

        Order<N> o_ = detail::toArray<N>(o, "o");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::cholesky<N>(
            detail::get_from_size(p, ncomponents * comm.nprocs, comm, dim), dim, o_,
            detail::get_components<N>(v, nullptr, ctx, ncomponents, p, comm, session), orows, ocols,
            comm, co);
    }

    /// Solve several upper triangular linear systems
    /// \param alpha: factor on the contraction
    /// \param pc: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponentsc: number of consecutive components in each MPI rank
    /// \param oc: dimension labels for the first operator
    /// \param vc: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxc: context for each data pointer in v0
    /// \param px: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the second operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the output tensor in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output tensor
    /// \param vy: data for the output tensor
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T>
    void trsm(T alpha, const PartitionItem<Nc> *pc, const Coor<Nc> &dimc, int ncomponentsc,
              const char *oc, const T **vc, const char *orows, const char *ocols,
              const Context *ctxc, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
              int ncomponentsx, const char *ox, const T **vx, const Context *ctxx,
              const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::trsm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, comm, dimc), dimc, oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, comm, dimx), dimx,
            ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, comm, dimy), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }

    /// Solve several linear systems
    /// \param alpha: factor on the contraction
    /// \param pc: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponentsc: number of consecutive components in each MPI rank
    /// \param oc: dimension labels for the first operator
    /// \param vc: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxc: context for each data pointer in v0
    /// \param px: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the second operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the output tensor in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output tensor
    /// \param vy: data for the output tensor
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T>
    void gesm(T alpha, const PartitionItem<Nc> *pc, const Coor<Nc> &dimc, int ncomponentsc,
              const char *oc, const T **vc, const char *orows, const char *ocols,
              const Context *ctxc, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
              int ncomponentsx, const char *ox, const T **vx, const Context *ctxx,
              const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::gesm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, comm, dimc), dimc, oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, comm, dimx), dimx,
            ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, comm, dimy), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }

    /// Compute the inversion of several matrices
    /// \param p: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param o: dimension labels for the first operator
    /// \param v: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctx: context for each data pointer in v0
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t N, typename T>
    void inversion(const PartitionItem<N> *p, const Coor<N> &dim, int ncomponents, const char *o,
                   T **v, const char *orows, const char *ocols, const Context *ctx,
                   MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Order<N> o_ = detail::toArray<N>(o, "o");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::inversion<N>(
            detail::get_from_size(p, ncomponents * comm.nprocs, comm, dim), dim, o_,
            detail::get_components<N>(v, nullptr, ctx, ncomponents, p, comm, session), orows, ocols,
            comm, co);
    }

    /// Compute several singular value decompositions (SVD)
    /// \param alpha: factor on the contraction
    /// \param pa: partitioning of the origin tensor in consecutive ranges
    /// \param ncomponentsa: number of consecutive components in each MPI rank
    /// \param oa: dimension labels for the origin tensor
    /// \param va: data for the origin tensor
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxa: context for each data pointer in va
    /// \param px: partitioning of the output left singular vectors in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the output left singular vectors
    /// \param vx: data for the output left singular vectors
    /// \param ps: partitioning of the output singular values in consecutive ranges
    /// \param ncomponentss: number of consecutive components in each MPI rank
    /// \param os: dimension labels for the output singular values
    /// \param vs: data for the output singular values
    /// \param py: partitioning of the output right singular vectors in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output right singular vectors
    /// \param vy: data for the output right singular vectors
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Na, std::size_t Nx, std::size_t Ns, std::size_t Ny, typename T>
    void svd(T alpha, const PartitionItem<Na> *pa, const Coor<Na> &dima, int ncomponentsa,
             const char *oa, const T **va, const char *orows, const char *ocols,
             const Context *ctxa, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
             int ncomponentsx, const char *ox, T **vx, const Context *ctxx,
             const PartitionItem<Ns> *ps, const Coor<Ns> &dims, int ncomponentss, const char *os,
             typename detail::the_real<T>::type **vs, const Context *ctxs,
             const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
             T **vy, const Context *ctxy, MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        Order<Na> oa_ = detail::toArray<Na>(oa, "oa");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ns> os_ = detail::toArray<Ns>(os, "os");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::svd<Na, Nx, Ns, Ny>(
            alpha, detail::get_from_size(pa, ncomponentsa * comm.nprocs, comm, dima), dima, oa_,
            detail::get_components<Na>((T **)va, nullptr, ctxa, ncomponentsa, pa, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, comm, dimx), dimx,
            ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(ps, ncomponentss * comm.nprocs, comm, dims), dims, os_,
            detail::get_components<Ns>((typename detail::the_real<T>::type **)vs, nullptr, ctxs,
                                       ncomponentss, ps, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, comm, dimy), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }
#endif // SUPERBBLAS_USE_MPI

    /// Compute the Cholesky factorization of several matrices
    /// \param p: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param o: dimension labels for the first operator
    /// \param v: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctx: context for each data pointer in v0
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t N, typename T>
    void cholesky(const PartitionItem<N> *p, const Coor<N> &dim, int ncomponents, const char *o,
                  T **v, const char *orows, const char *ocols, const Context *ctx, CoorOrder co,
                  Session session = 0) {

        Order<N> o_ = detail::toArray<N>(o, "o");

        detail::SelfComm comm = detail::get_comm();

        detail::cholesky<N>(
            detail::get_from_size(p, ncomponents * comm.nprocs, comm, dim), dim, o_,
            detail::get_components<N>(v, nullptr, ctx, ncomponents, p, comm, session), orows, ocols,
            comm, co);
    }

    /// Solve several upper triangular linear systems
    /// \param alpha: factor on the contraction
    /// \param pc: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponentsc: number of consecutive components in each MPI rank
    /// \param oc: dimension labels for the first operator
    /// \param vc: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxc: context for each data pointer in v0
    /// \param px: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the second operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the output tensor in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output tensor
    /// \param vy: data for the output tensor
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T>
    void trsm(T alpha, const PartitionItem<Nc> *pc, const Coor<Nc> &dimc, int ncomponentsc,
              const char *oc, const T **vc, const char *orows, const char *ocols,
              const Context *ctxc, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
              int ncomponentsx, const char *ox, const T **vx, const Context *ctxx,
              const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::trsm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, comm, dimc), dimc, oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, comm, dimx), dimx,
            ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, comm, dimy), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }

    /// Solve several linear systems
    /// \param alpha: factor on the contraction
    /// \param pc: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponentsc: number of consecutive components in each MPI rank
    /// \param oc: dimension labels for the first operator
    /// \param vc: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxc: context for each data pointer in v0
    /// \param px: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the second operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the output tensor in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output tensor
    /// \param vy: data for the output tensor
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nc, std::size_t Nx, std::size_t Ny, typename T>
    void gesm(T alpha, const PartitionItem<Nc> *pc, const Coor<Nc> &dimc, int ncomponentsc,
              const char *oc, const T **vc, const char *orows, const char *ocols,
              const Context *ctxc, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
              int ncomponentsx, const char *ox, const T **vx, const Context *ctxx,
              const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
              T **vy, const Context *ctxy, CoorOrder co, Session session = 0) {

        Order<Nc> oc_ = detail::toArray<Nc>(oc, "oc");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::gesm<Nc, Nx, Ny>(
            alpha, detail::get_from_size(pc, ncomponentsc * comm.nprocs, comm, dimc), dimc, oc_,
            detail::get_components<Nc>((T **)vc, nullptr, ctxc, ncomponentsc, pc, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, comm, dimx), dimx,
            ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, comm, dimy), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }

    /// Compute the inversion of several matrices
    /// \param p: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param o: dimension labels for the first operator
    /// \param v: data for the first operator
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctx: context for each data pointer in v0
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t N, typename T>
    void inversion(const PartitionItem<N> *p, const Coor<N> &dim, int ncomponents, const char *o,
                   T **v, const char *orows, const char *ocols, const Context *ctx, CoorOrder co,
                   Session session = 0) {

        Order<N> o_ = detail::toArray<N>(o, "o");

        detail::SelfComm comm = detail::get_comm();

        detail::inversion<N>(
            detail::get_from_size(p, ncomponents * comm.nprocs, comm, dim), dim, o_,
            detail::get_components<N>(v, nullptr, ctx, ncomponents, p, comm, session), orows, ocols,
            comm, co);
    }

    /// Compute several singular value decompositions (SVD)
    /// \param alpha: factor on the contraction
    /// \param pa: partitioning of the origin tensor in consecutive ranges
    /// \param ncomponentsa: number of consecutive components in each MPI rank
    /// \param oa: dimension labels for the origin tensor
    /// \param va: data for the origin tensor
    /// \param orows: labels on the rows
    /// \param ocols: labels on the columns
    /// \param ctxa: context for each data pointer in va
    /// \param px: partitioning of the output left singular vectors in consecutive ranges
    /// \param ncomponentsx: number of consecutive components in each MPI rank
    /// \param ox: dimension labels for the output left singular vectors
    /// \param vx: data for the output left singular vectors
    /// \param ps: partitioning of the output singular values in consecutive ranges
    /// \param ncomponentss: number of consecutive components in each MPI rank
    /// \param os: dimension labels for the output singular values
    /// \param vs: data for the output singular values
    /// \param py: partitioning of the output right singular vectors in consecutive ranges
    /// \param ncomponentsy: number of consecutive components in each MPI rank
    /// \param oy: dimension labels for the output right singular vectors
    /// \param vy: data for the output right singular vectors
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Na, std::size_t Nx, std::size_t Ns, std::size_t Ny, typename T>
    void svd(T alpha, const PartitionItem<Na> *pa, const Coor<Na> &dima, int ncomponentsa,
             const char *oa, const T **va, const char *orows, const char *ocols,
             const Context *ctxa, const PartitionItem<Nx> *px, const Coor<Nx> &dimx,
             int ncomponentsx, const char *ox, T **vx, const Context *ctxx,
             const PartitionItem<Ns> *ps, const Coor<Ns> &dims, int ncomponentss, const char *os,
             typename detail::the_real<T>::type **vs, const Context *ctxs,
             const PartitionItem<Ny> *py, const Coor<Ny> &dimy, int ncomponentsy, const char *oy,
             T **vy, const Context *ctxy, CoorOrder co, Session session = 0) {

        Order<Na> oa_ = detail::toArray<Na>(oa, "oa");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ns> os_ = detail::toArray<Ns>(os, "os");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::svd<Na, Nx, Ns, Ny>(
            alpha, detail::get_from_size(pa, ncomponentsa * comm.nprocs, comm, dima), dima, oa_,
            detail::get_components<Na>((T **)va, nullptr, ctxa, ncomponentsa, pa, comm, session),
            orows, ocols, detail::get_from_size(px, ncomponentsx * comm.nprocs, comm, dimx), dimx,
            ox_,
            detail::get_components<Nx>((T **)vx, nullptr, ctxx, ncomponentsx, px, comm, session),
            detail::get_from_size(ps, ncomponentss * comm.nprocs, comm, dims), dims, os_,
            detail::get_components<Ns>((typename detail::the_real<T>::type **)vs, nullptr, ctxs,
                                       ncomponentss, ps, comm, session),
            detail::get_from_size(py, ncomponentsy * comm.nprocs, comm, dimy), dimy, oy_,
            detail::get_components<Ny>(vy, nullptr, ctxy, ncomponentsx, py, comm, session), comm,
            co);
    }
}

#endif // __SUPERBBLAS_DENSE__
