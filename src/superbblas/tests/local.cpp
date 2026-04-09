#include "superbblas.h"
#include <iostream>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

int main(int argc, char **argv) {
    constexpr std::size_t Nd = 7; // xyztscn
    constexpr unsigned int nS = 4, nC = 3; // length of dimension spin and color dimensions
    constexpr unsigned int X = 0, Y = 1, Z = 2, T = 3, S = 4, C = 5, N = 6;
    Coor<Nd> dim = {16, 16, 16, 32, nS, nC, 64}; // xyztscn
    const unsigned int nrep = getDebugLevel() == 0 ? 10 : 1;

    // Get options
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d %d %d %d %d", &dim[X], &dim[Y], &dim[Z], &dim[T],
                       &dim[N]) != 5) {
                std::cerr << "--dim= should follow 5 numbers, for instance -dim='2 2 2 2 2'"
                          << std::endl;
                return -1;
            }
        } else if(std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0] << " [--dim='x y z t n'] [--help]"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    std::cout << "Testing lattice dimensions xyzt= " << dim[X] << " " << dim[Y] << " " << dim[Z]
              << " " << dim[T] << "  num_vecs= " << dim[N] << std::endl;

    //using Scalar = double;
    using Scalar = std::complex<float>;
    using ScalarD = std::complex<double>;
    {
        using Tensor = std::vector<Scalar>;
        using TensorD = std::vector<ScalarD>;

        // Create tensor t0 of Nd-1 dims: a lattice color vector
        const Coor<Nd - 1> dim0 = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
        std::size_t vol0 = detail::volume(dim0);
        Tensor t0(vol0);

        // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
        const Coor<Nd> dim1 = {dim[T], dim[N], dim[S], dim[X], dim[Y], dim[Z], dim[C]}; // tnsxyzc
        std::size_t vol1 = detail::volume(dim1);
        Tensor t1(vol1);
        for (unsigned int i = 0; i < vol0; i++) t0[i] = i;

        Context ctx = createCpuContext();

        std::cout << ">>> CPU tests with " << num_threads << " threads" << std::endl;

        std::cout << "Maximum number of elements in a tested tensor: " << vol1 << " ( "
                  << vol1 * 1.0 * sizeof(Scalar) / 1024 / 1024 << " MiB)" << std::endl;

        // Copy tensor t0 into tensor 1 (for reference)
        double tref = 0.0;
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
#ifdef _OPENMP
#    pragma omp parallel for
#endif
                    for (unsigned int i = 0; i < (unsigned int)vol0; ++i)
                        t1[i + n * (unsigned int)vol0] = t0[i];
                }
            }
            t = w_time() - t;
            std::cout << "Time in dummy copying from xyzts to tnsxyzc " << t / nrep << std::endl;
            tref = t / nrep; // time in copying a whole tensor with size dim1
        }


        // Copy tensor t0 into each of the c components of tensor 1
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
                    const Coor<Nd - 1> from0 = {0};
                    const Coor<Nd> from1 = {0, n, 0};
                    local_copy(Scalar{1.0}, "xyztsc", from0, dim0, dim0, t0.data(), nullptr, ctx,
                               "tnsxyzc", from1, dim1, t1.data(), nullptr, ctx, SlowToFast, Copy);
                }
            }
            t = w_time() - t;
            std::cout << "Time in copying/permuting from xyztsc to tnsxyzc " << t / nrep
                      << " (overhead " << t / nrep / tref << " )" << std::endl;
        }

        // Copy tensor t0 into each of the n components of tensor 1 (fast)
        // {
        //     double t = w_time();
        //     for (unsigned int rep = 0; rep < nrep; ++rep) {
        //         for (int n = 0; n < dim[N]; ++n) {
        //             const Coor<Nd - 2> from0 = {0};
        //             const Coor<Nd - 1> from1 = {0, n, 0};
        //             Coor<Nd - 2> dim0a;
        //             std::copy_n(dim0.begin(), Nd - 2, dim0a.begin());
        //             Coor<Nd - 1> dim1a;
        //             std::copy_n(dim1.begin(), Nd - 1, dim1a.begin());
        //             local_copy(Scalar{1.0}, "xyzts", from0, dim0a, dim0a,
        //                        (const std::array<Scalar, nC> *)t0.data(), ctx, "tnsxyz", from1,
        //                        dim1a, (std::array<Scalar, nC> *)t1.data(), ctx, SlowToFast, Copy);
        //         }
        //     }
        //     t = w_time() - t;
        //     std::cout << "Time in copying/permuting from xyzts to tnsxyzs (fast) " << t / nrep
        //               << " (overhead " << t / nrep / tref << " )" << std::endl;
        // }

        // Shift tensor 1 on the z-direction and store it on tensor 2
        Tensor t2(vol1);
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                const Coor<Nd> from0 = {0};
                Coor<Nd> from1 = {0};
                from1[4] = 1; // Displace one on the z-direction
                local_copy(Scalar{1.0}, "tnsxyzc", from0, dim1, dim1, t1.data(), nullptr, ctx,
                           "tnsxyzc", from1, dim1, t2.data(), nullptr, ctx, SlowToFast, Copy);
            }
            t = w_time() - t;
            std::cout << "Time in shifting " << t / nrep << std::endl;
        }

        // Shift tensor 1 on the z-direction and store it on tensor 2
        {
            TensorD t2d(vol1);
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                const Coor<Nd> from0 = {0};
                Coor<Nd> from1 = {0};
                from1[4] = 1; // Displace one on the z-direction
                local_copy(Scalar{1.0}, "tnsxyzc", from0, dim1, dim1, t1.data(), nullptr, ctx,
                           "tnsxyzc", from1, dim1, t2d.data(), nullptr, ctx, SlowToFast, Copy);
            }
            t = w_time() - t;
            std::cout << "Time in shifting and converting to double " << t / nrep << std::endl;
        }

        const Coor<5> dimc = {dim[T], dim[N], dim[S], dim[N], dim[S]}; // tnsns
        std::size_t volc = detail::volume(dimc); 
        Tensor tc(volc);
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                local_contraction(Scalar{1.0}, "tnsxyzc", dim1, false, t1.data(), "tNSxyzc", dim1,
                                  false, t2.data(), Scalar{0.0}, "tNSns", dimc, tc.data(), ctx,
                                  SlowToFast);
            }
            t = w_time() - t;
            std::cout << "Time in contracting xyzc " << t / nrep << std::endl;
        }
    }
#ifdef SUPERBBLAS_USE_CUDA
    {
        using Tensor = thrust::device_vector<Scalar>;
        using TensorD = thrust::device_vector<ScalarD>;

        // Create tensor t0 of Nd-1 dims: a lattice color vector
        const Coor<Nd - 1> dim0 = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
        std::size_t vol0 = detail::volume(dim0);
        Tensor t0(vol0);

        // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
        const Coor<Nd> dim1 = {dim[T], dim[N], dim[S], dim[X], dim[Y], dim[Z], dim[C]}; // tnsxyzc
        std::size_t vol1 = detail::volume(dim1);
        Tensor t1(vol1);

        // Dummy initialization of t0
        std::vector<Scalar> t0_host(vol0);
        for (unsigned int i = 0; i < vol0; i++) t0_host[i] = i;
        t0 = t0_host;

        Context ctx = createCudaContext();

        std::cout << ">>> GPU tests" << std::endl;

        std::cout << "Maximum number of elements in a tested tensor: " << vol1 << " ( "
                  << vol1 * 1.0 * sizeof(Scalar) / 1024 / 1024 << " MiB)" << std::endl;

        // Copy tensor t0 into tensor 1 (for reference)
        double tref = 0.0;
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
                    thrust::copy_n(t0.begin(), vol0, t1.begin() + n * vol0);
                }
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            std::cout << "Time in dummy copying from xyzts to tnsxyzc " << t / nrep << std::endl;
            tref = t / nrep; // time in copying a whole tensor with size dim1
        }


        // Copy tensor t0 into each of the c components of tensor 1
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
                    const Coor<Nd - 1> from0 = {0};
                    const Coor<Nd> from1 = {0, n, 0};
                    local_copy(Scalar{1.0}, "xyztsc", from0, dim0, dim0, t0.data().get(), nullptr,
                               ctx, "tnsxyzc", from1, dim1, t1.data().get(), nullptr, ctx,
                               SlowToFast, Copy);
                }
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            std::cout << "Time in copying/permuting from xyztsc to tnsxyzc " << t / nrep
                      << " (overhead " << t / nrep / tref << " )" << std::endl;
        }

        // Copy tensor t0 into each of the c components of tensor 1 from CPU
        {
            using Tensor_cpu = std::vector<Scalar>;
            const Coor<Nd - 1> dim0 = {dim[X], dim[Y], dim[Z],
                                       dim[T], dim[S], dim[C]}; // xyztsc
            std::size_t vol0 = detail::volume(dim0);
            Tensor_cpu t0_cpu(vol0);
            for (unsigned int i = 0; i < vol0; i++) t0_cpu[i] = i;

            Context cpuctx = createCpuContext();

            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
                    const Coor<Nd - 1> from0 = {0};
                    const Coor<Nd> from1 = {0, n, 0};
                    local_copy<Nd - 1, Nd>(Scalar{1.0}, "xyztsc", from0, dim0, dim0, t0_cpu.data(),
                                           nullptr, cpuctx, "tnsxyzc", from1, dim1, t1.data().get(),
                                           nullptr, ctx, SlowToFast, Copy);
                }
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            std::cout << "Time in copying/permuting from xyztsc from cpu to "
                         "tnsxyzc on GPU "
                      << t / nrep << " (overhead " << t / nrep / tref << " )"
                      << std::endl;
        }


// #    ifndef SUPERBBLAS_LIB
//         // Copy tensor t0 into each of the c components of tensor 1 (fast?)
//         {
//             double t = w_time();
//             for (unsigned int rep = 0; rep < nrep; ++rep) {
//                 for (int n = 0; n < dim[N]; ++n) {
//                     const Coor<Nd - 2> from0 = {0};
//                     const Coor<Nd - 1> from1 = {0, n, 0};
//                     Coor<Nd - 2> dim0a;
//                     std::copy_n(dim0.begin(), Nd - 2, dim0a.begin());
//                     Coor<Nd - 1> dim1a;
//                     std::copy_n(dim1.begin(), Nd - 1, dim1a.begin());
//                     local_copy(Scalar{1.0}, "xyzts", from0, dim0a, dim0a,
//                                (const std::array<Scalar, nC> *)t0.data().get(), ctx, "tnsxyz",
//                                from1, dim1a, (std::array<Scalar, nC> *)t1.data().get(), ctx,
//                                SlowToFast, Copy);
//                 }
//             }
//             cudaDeviceSynchronize();
//             t = w_time() - t;
//             std::cout << "Time in copying/permuting from xyzts to tnsxyzs (fast?) " << t / nrep
//                       << " (overhead " << t / nrep / tref << " )" << std::endl;
//         }
// #    endif // SUPERBBLAS_LIB

        // Shift tensor 1 on the z-direction and store it on tensor 2
        Tensor t2(vol1);
         {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                const Coor<Nd> from0 = {0};
                Coor<Nd> from1 = {0};
                from1[4] = 1; // Displace one on the z-direction
                local_copy(Scalar{1.0}, "tnsxyzc", from0, dim1, dim1, t1.data().get(), nullptr, ctx,
                           "tnsxyzc", from1, dim1, t2.data().get(), nullptr, ctx, SlowToFast, Copy);
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            std::cout << "Time in shifting " << t / nrep << std::endl;
        }

        // Shift tensor 1 on the z-direction and store it on tensor 2
        {
            TensorD t2d(vol1);
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                const Coor<Nd> from0 = {0};
                Coor<Nd> from1 = {0};
                from1[4] = 1; // Displace one on the z-direction
                local_copy(Scalar{1.0}, "tnsxyzc", from0, dim1, dim1, t1.data().get(), nullptr, ctx,
                           "tnsxyzc", from1, dim1, t2d.data().get(), nullptr, ctx, SlowToFast,
                           Copy);
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            std::cout << "Time in shifting and converting to double " << t / nrep << std::endl;
        }

        const Coor<5> dimc = {dim[T], dim[N], dim[S], dim[N], dim[S]}; // tnsns
        std::size_t volc = detail::volume(dimc); 
        Tensor tc(volc);
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                local_contraction(Scalar{1.0}, "tnsxyzc", dim1, false, t1.data().get(), "tNSxyzc",
                                  dim1, false, t2.data().get(), Scalar{0.0}, "tNSns", dimc,
                                  tc.data().get(), ctx, SlowToFast);
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            std::cout << "Time in contracting " << t / nrep << std::endl;
        }

        // Copy tensor tc back to cpu
        {
            using Tensor_cpu = std::vector<Scalar>;
            Tensor_cpu tc_cpu(volc);

            Context cpuctx = createCpuContext();

            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                    const Coor<5> from0 = {0};
                    local_copy(Scalar{1.0}, "tnsNS", from0, dimc, dimc, tc.data().get(), nullptr, ctx,
                               "tnsNS", from0, dimc, tc_cpu.data(), nullptr, cpuctx, SlowToFast, Copy);
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            std::cout << "Time in copying tnsNS to the cpu " << t / nrep
                      << std::endl;
        }
    }
#endif

    reportTimings(std::cout);
    reportCacheUsage(std::cout);

    return 0;
}
