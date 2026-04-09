#include "superbblas.h"
#include <algorithm>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

constexpr std::size_t Nd = 7; // xyztscn
constexpr unsigned int X = 0, Y = 1, Z = 2, T = 3, S = 4, C = 5, N = 6;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

// Return a vector of all ones
template <typename T, typename XPU> vector<T, XPU> ones(std::size_t size, XPU xpu) {
    vector<T, Cpu> r(size, Cpu{});
    for (std::size_t i = 0; i < size; ++i) r[i] = 1.0;
    return makeSure(r, xpu);
}

// Return a vector of all ones
template <typename T, typename XPU>
vector<T, XPU> laplacian(std::size_t n, std::size_t size, XPU xpu) {
    vector<T, Cpu> r(size, Cpu{});
    if (size % (n * n) != 0)
        throw std::runtime_error("Unsupported the creation of partial square matrices");
    for (std::size_t i = 0; i < size; ++i) r[i] = 0;
    for (std::size_t k = 0, K = size / (n * n); k < K; ++k) {
        for (std::size_t i = 0; i < n; ++i) r[k * n * n + i * n + i] = 2;
        for (std::size_t i = 0; i < n - 1; ++i) r[k * n * n + (i + 1) * n + i] = -1;
        for (std::size_t i = 0; i < n - 1; ++i) r[k * n * n + i * n + (i + 1)] = -1;
    }
    return makeSure(r, xpu);
}

template <typename Q, typename XPU>
void test(Coor<Nd> dim, Coor<Nd> procs, int rank, Context ctx, XPU xpu) {

    // Set number of repetitions
    const unsigned int nrep = getDebugLevel() == 0 ? 10 : 1;

    // Create tensor t0 of Nd dims: a lattice color vector
    const Coor<Nd + 1> dim0 = {dim[X], dim[Y], dim[Z], dim[T],
                               dim[S], dim[C], dim[S], dim[C]};                       // xyztscSC
    const Coor<Nd + 1> procs0 = {procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1, 1}; // xyztscSC
    PartitionStored<Nd + 1> p0 = basic_partitioning(dim0, procs0);
    const Coor<Nd + 1> local_size0 = p0[rank][1];
    std::size_t vol0 = detail::volume(local_size0);
    vector<Q, XPU> t0 = laplacian<Q>(dim[S] * dim[C], vol0, xpu);

    const bool is_cpu = deviceId(xpu) == CPU_DEVICE_ID;
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    if (rank == 0)
        std::cout << ">>> " << (is_cpu ? "CPU" : "GPU") << " tests with " << num_threads
                  << " threads" << std::endl;

    if (rank == 0)
        std::cout << "Number of elements in tested tensor 'A' per process: " << vol0 << " ( "
                  << vol0 * 1.0 * sizeof(Q) / 1024 / 1024 << " MiB)" << std::endl;

    // Copy tensor t0 into each of the c components of tensor 1
    resetTimings();
    try {
        vector<Q, XPU> tx(vol0, xpu);
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            copy_n(t0.data(), xpu, vol0, tx.data(), xpu);
            Q *ptr0 = tx.data();
            cholesky<Nd + 1, Q>(p0.data(), dim0, 1, "xyztscSC", (Q **)&ptr0, "sc", "SC", &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                MPI_COMM_WORLD,
#endif
                                SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in cholesky " << t / nrep << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    resetTimings();

    // Create tensors tx and ty of Nd dims: a lattice color vector
    const Coor<Nd> dimx = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C], dim[N]}; // xyztscn
    const Coor<Nd> procsx = {procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1};      // xyztscn
    PartitionStored<Nd> px = basic_partitioning(dimx, procsx);
    const Coor<Nd> local_sizex = px[rank][1];
    std::size_t volx = detail::volume(local_sizex);
    vector<Q, XPU> tx = ones<Q>(volx, xpu);
    vector<Q, XPU> ty(volx, xpu);

    if (rank == 0)
        std::cout << "Number of elements in tested tensor 'X' per process: " << volx << " ( "
                  << volx * 1.0 * sizeof(Q) / 1024 / 1024 << " MiB)" << std::endl;
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            Q *ptr0 = t0.data();
            Q *ptrx = tx.data();
            Q *ptry = ty.data();
            trsm<Nd + 1, Nd, Nd, Q>(Q{1}, p0.data(), dim0, 1, "xyztscSC", (const Q **)&ptr0, "sc",
                                    "SC", &ctx, px.data(), dimx, 1, "xyztscn", (const Q **)&ptrx,
                                    &ctx, px.data(), dimx, 1, "xyztSCn", (Q **)&ptry, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                    MPI_COMM_WORLD,
#endif
                                    SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in trsm " << t / nrep << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    // Reset the A and x
    int k = 8;
    Coor<5> dim0k = {k, dim[S], dim[C], dim[S], dim[C]};
    t0 = laplacian<Q>(dim[S] * dim[C], volume(dim0k), xpu);
    PartitionStored<5> p0k = PartitionStored<5>(volume(procs), {Coor<5>{{}}, dim0k});
    const Coor<Nd + 1> dimxk = {k,      dim[X], dim[Y], dim[Z],
                                dim[T], dim[S], dim[C], dim[N]};                       // kxyztscn
    const Coor<Nd + 1> procsxk = {1, procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}; // kxyztscn
    PartitionStored<Nd + 1> pxk = basic_partitioning(dimxk, procsxk);
    tx = ones<Q>(volx * k, xpu);
    ty = vector<Q, XPU>(tx.size(), xpu);

    // Create a bunch of Cholesky factors
    {
        Q *ptr0 = t0.data();
        cholesky<5, Q>(p0k.data(), dim0k, 1, "kscSC", (Q **)&ptr0, "sc", "SC", &ctx,
#ifdef SUPERBBLAS_USE_MPI
                       MPI_COMM_WORLD,
#endif
                       SlowToFast);
    }

    resetTimings();
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            Q *ptr0 = t0.data();
            Q *ptrx = tx.data();
            Q *ptry = ty.data();
            trsm<5, Nd + 1, Nd + 1, Q>(Q{1}, p0k.data(), dim0k, 1, "kscSC", (const Q **)&ptr0, "sc",
                                       "SC", &ctx, pxk.data(), dimxk, 1, "kxyztscn",
                                       (const Q **)&ptrx, &ctx, pxk.data(), dimxk, 1, "kxyztSCn",
                                       (Q **)&ptry, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                       MPI_COMM_WORLD,
#endif
                                       SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in trsm " << t / nrep << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    // Reset the A and x
    t0 = laplacian<Q>(dim[S] * dim[C], vol0, xpu);
    tx = ones<Q>(volx, xpu);
    ty = vector<Q, XPU>(tx.size(), xpu);

    resetTimings();
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            Q *ptr0 = t0.data();
            Q *ptrx = tx.data();
            Q *ptry = ty.data();
            gesm<Nd + 1, Nd, Nd, Q>(Q{1}, p0.data(), dim0, 1, "xyztscSC", (const Q **)&ptr0, "sc",
                                    "SC", &ctx, px.data(), dimx, 1, "xyztSCn", (const Q **)&ptrx,
                                    &ctx, px.data(), dimx, 1, "xyztscn", (Q **)&ptry, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                    MPI_COMM_WORLD,
#endif
                                    SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in gesm " << t / nrep << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    tx = ones<Q>(t0.size(), xpu);
    ty = vector<Q, XPU>(0, xpu); // release memory

    resetTimings();
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            copy_n(t0.data(), t0.ctx(), t0.size(), tx.data(), tx.ctx());
            Q *ptrx = tx.data();
            inversion<Nd + 1, Q>(p0.data(), dim0, 1, "xyztscSC", (Q **)&ptrx, "sc", "SC", &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                 MPI_COMM_WORLD,
#endif
                                 SlowToFast);
        }
        sync(xpu);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in inversion " << t / nrep << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    {
        /// Compute the svd of t0, (s,cSC,xyzt)
        vector<Q, XPU> t0 = laplacian<Q>(dim[S] * dim[C], vol0, xpu);

        const int dimI = std::min(dim[S], dim[C] * dim[S] * dim[C]);
        const Coor<Nd - 1> dimx = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dimI};   // xyztsi
        const Coor<Nd - 1> procsx = {procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // xyztsi
        PartitionStored<Nd - 1> px = basic_partitioning(dimx, procsx);
        vector<Q, XPU> tx(detail::volume(px[rank][1]), xpu);

        const Coor<Nd - 2> dims = {dim[X], dim[Y], dim[Z], dim[T], dimI};        // xyzti
        const Coor<Nd - 2> procss = {procs[X], procs[Y], procs[Z], procs[T], 1}; // xyzti
        PartitionStored<Nd - 2> ps = basic_partitioning(dims, procss);
        vector<typename detail::the_real<Q>::type, XPU> ts(detail::volume(ps[rank][1]), xpu);

        const Coor<Nd + 1> dimy = {dim[X], dim[Y], dim[Z], dim[T],
                                   dim[C], dim[S], dim[C], dimI}; // xyztcSCi
        const Coor<Nd + 1> procsy = {procs[X], procs[Y], procs[Z], procs[T],
                                     1,        1,        1,        1}; // xyztcSCi
        PartitionStored<Nd + 1> py = basic_partitioning(dimy, procsy);
        vector<Q, XPU> ty(detail::volume(py[rank][1]), xpu);

        resetTimings();
        try {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                Q *ptra = t0.data();
                Q *ptrx = tx.data();
                typename superbblas::detail::the_real<Q>::type *ptrs = ts.data();
                Q *ptry = ty.data();
                svd<Nd + 1, Nd - 1, Nd - 2, Nd + 1, Q>(
                    Q{1},                                                                //
                    p0.data(), dim0, 1, "xyztscSC", (const Q **)&ptra, "s", "cSC", &ctx, //
                    px.data(), dimx, 1, "xyztsi", &ptrx, &ctx,                           //
                    ps.data(), dims, 1, "xyzti", &ptrs, &ctx,                            //
                    py.data(), dimy, 1, "xyztcSCi", &ptry, &ctx,                         //
#ifdef SUPERBBLAS_USE_MPI
                    MPI_COMM_WORLD,
#endif
                    SlowToFast);
            }
            sync(xpu);
            t = w_time() - t;
            if (rank == 0) std::cout << "Time in svd " << t / nrep << std::endl;
        } catch (const std::exception &e) {
            std::cout << "Caught error: " << e.what() << std::endl;
        }
    }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);
}

int main(int argc, char **argv) {
    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    (void)argc;
    (void)argv;
    nprocs = 1;
    rank = 0;
#endif

    Coor<Nd> dim = {16, 16, 16, 32, 4, 3, 64}; // xyztscn
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1};

    // Get options
    bool procs_was_set = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d %d %d %d %d %d", &dim[X], &dim[Y], &dim[Z], &dim[T],
                       &dim[N], &dim[C]) != 6) {
                std::cerr << "--dim= should follow 6 numbers, for instance -dim='2 2 2 2 2 2'"
                          << std::endl;
                return -1;
            }
            if (dim[C] % 4 == 0) {
                dim[S] = 4;
                dim[C] = dim[C] / 4;
            }
        } else if (std::strncmp("--procs=", argv[i], 8) == 0) {
            if (sscanf(argv[i] + 8, "%d %d %d %d", &procs[X], &procs[Y], &procs[Z], &procs[T]) !=
                4) {
                std::cerr << "--procs= should follow 4 numbers, for instance --procs='2 2 2 2'"
                          << std::endl;
                return -1;
            }
            if (detail::volume(procs) != (std::size_t)nprocs) {
                std::cerr << "The total number of processes set by the option `--procs=` should "
                             "match the number of processes"
                          << std::endl;
                return -1;
            }
            procs_was_set = true;
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--dim='x y z t n b'] [--procs='x y z t n c'] [--help]" << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // If --procs isn't set, put all processes on the first dimension
    if (!procs_was_set) procs[X] = nprocs;

    // Show lattice dimensions and processes arrangement
    if (rank == 0) {
        std::cout << "Testing lattice dimensions xyzt= " << dim[X] << " " << dim[Y] << " " << dim[Z]
                  << " " << dim[T] << " spin-color= " << dim[S] << " " << dim[C]
                  << "  num_vecs= " << dim[N] << std::endl;
        std::cout << "Processes arrangement xyzt= " << procs[X] << " " << procs[Y] << " "
                  << procs[Z] << " " << procs[T] << std::endl;
    }

    {
        Context ctx = createCpuContext();
        test<double, Cpu>(dim, procs, rank, ctx, ctx.toCpu(0));
        test<std::complex<float>, Cpu>(dim, procs, rank, ctx, ctx.toCpu(0));
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext();
        test<std::complex<float>, Gpu>(dim, procs, rank, ctx, ctx.toGpu(0));
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
