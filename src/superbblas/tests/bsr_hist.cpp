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

double timeit(const std::function<void()> &f, const std::function<void()> &ending) {
    std::size_t reps = 8, measures = 0;
    double min_time = HUGE_VAL;
    while (true) {
        double t = w_time();
        for (std::size_t i = 0; i < reps; ++i) f();
        ending();
        double dt = w_time() - t;
        if (dt < 0.001) {
            reps *= 2;
            continue;
        }
        min_time = std::min(min_time, dt / reps);
        measures++;
        int do_more_measures = (measures < 5 || measures * min_time * reps < 1) ? 1 : 0;
#ifdef SUPERBBLAS_USE_MPI
        MPI_Bcast(&do_more_measures, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        if (do_more_measures == 0) break;
    }
    return min_time;
}

// Return a vector of all ones
template <typename T, typename XPU> vector<T, XPU> ones(std::size_t size, XPU xpu) {
    vector<T, Cpu> r(size, Cpu{});
    for (std::size_t i = 0; i < size; ++i) r[i] = 1.0;
    return makeSure(r, xpu);
}

/// Extend the region one element in each direction
std::array<Coor<6>, 2> extend(std::array<Coor<6>, 2> fs, const Coor<6> &dim) {
    for (int i = 0; i < 4; ++i) {
        fs[1][i] = std::min(dim[i], fs[1][i] + 2);
        if (fs[1][i] < dim[i])
            fs[0][i]--;
        else
            fs[0][i] = 0;
    }
    fs[0] = normalize_coor(fs[0], dim);
    return fs;
}

// Return the maximum number of neighbors
unsigned int max_neighbors(const Coor<6> &op_dim) {
    unsigned int neighbors = 1;
    for (int dim = 0; dim < 4; ++dim) {
        int d = op_dim[dim];
        if (d <= 0) {
            neighbors = 0;
            break;
        }
        if (d > 1) neighbors++;
        if (d > 2) neighbors++;
    }
    return neighbors;
}

/// Create a 4D lattice with dimensions tzyxsc
template <typename T, typename XPU>
std::pair<BSR_handle *, vector<T, XPU>> create_lattice(const PartitionStored<6> &pi, int rank,
                                                       const Coor<6> op_dim, Context ctx, XPU xpu) {
    Coor<6> from = pi[rank][0]; // first nonblock dimensions of the RSB image
    Coor<6> dimi = pi[rank][1]; // nonblock dimensions of the RSB image
    dimi[4] = dimi[5] = 1;
    std::size_t voli = volume(dimi);
    vector<IndexType, Cpu> ii(voli, Cpu{});

    // Compute how many neighbors
    int neighbors = max_neighbors(op_dim);

    // Compute the domain ranges
    PartitionStored<6> pd = pi;
    for (auto &i : pd) i = extend(i, op_dim);

    // Compute the coordinates for all nonzeros
    for (auto &i : ii) i = neighbors;
    vector<Coor<6>, Cpu> jj(neighbors * voli, Cpu{});
    Coor<6, std::size_t> stride = get_strides<std::size_t>(dimi, SlowToFast);
    Coor<6, std::size_t> strided = get_strides<std::size_t>(pd[rank][1], SlowToFast);
    for (std::size_t i = 0, j = 0; i < voli; ++i) {
        std::size_t j0 = j;
        Coor<6> c = index2coor(i, dimi, stride) + from;
        jj[j++] = c;
        for (int dim = 0; dim < 4; ++dim) {
            if (op_dim[dim] == 1) continue;
            for (int dir = -1; dir < 2; dir += 2) {
                Coor<6> c0 = c;
                c0[dim] += dir;
                jj[j++] = normalize_coor(c0 - pd[rank][0], op_dim);
                if (op_dim[dim] <= 2) break;
            }
        }
        std::sort(&jj[j0], &jj[j], [&](const Coor<6> &a, const Coor<6> &b) {
            return coor2index(a, pd[rank][1], strided) < coor2index(b, pd[rank][1], strided);
        });
    }

    // Number of nonzeros
    std::size_t vol_data = voli * neighbors * op_dim[4] * op_dim[5] * op_dim[4] * op_dim[5];

    Coor<6> block{{1, 1, 1, 1, op_dim[4], op_dim[5]}};
    BSR_handle *bsrh = nullptr;
    vector<int, XPU> ii_xpu = makeSure(ii, xpu);
    vector<Coor<6>, XPU> jj_xpu = makeSure(jj, xpu);
    vector<T, XPU> data_xpu = ones<T>(vol_data, xpu);
    IndexType *iiptr = ii_xpu.data();
    Coor<6> *jjptr = jj_xpu.data();
    T *dataptr = data_xpu.data();
    create_bsr<6, 6, T>(pi.data(), op_dim, pd.data(), op_dim, 1, block, block, false, &iiptr,
                        &jjptr, (const T **)&dataptr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast, &bsrh);
    return {bsrh, data_xpu};
}

template <typename Q, typename XPU>
void test(Coor<Nd> dim, Coor<Nd> procs, int rank, int max_power, Context ctx, XPU xpu) {

    // Create a lattice operator of Nd-1 dims
    const Coor<Nd - 1> dimo = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
    const Coor<Nd - 1> procso = {procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // xyztsc
    PartitionStored<Nd - 1> po =
        basic_partitioning(dimo, procso, -1, false,
                           {{max_power - 1, max_power - 1, max_power - 1, max_power - 1, 0, 0}});
    auto op_pair = create_lattice<Q>(po, rank, dimo, ctx, xpu);
    BSR_handle *op = op_pair.first;

    // Create tensor t0 of Nd dims: an input lattice color vector
    bool preferred_layout_for_x_is_rowmajor = true;
    const char *o0 = preferred_layout_for_x_is_rowmajor ? "pXYZTSCn" : "pnXYZTSC";
    Coor<Nd + 1> dim0 =
        preferred_layout_for_x_is_rowmajor
            ? Coor<Nd + 1>{1, dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C], dim[N]}  // pxyztscn
            : Coor<Nd + 1>{1, dim[N], dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // pnxyztsc
    const Coor<Nd + 1> procs0 =
        preferred_layout_for_x_is_rowmajor
            ? Coor<Nd + 1>{1, procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}  // pxyztscn
            : Coor<Nd + 1>{1, 1, procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // pnxyztsc
    PartitionStored<Nd + 1> p0 = basic_partitioning(dim0, procs0);
    const Coor<Nd + 1> local_size0 = p0[rank][1];
    std::size_t vol0 = detail::volume(local_size0);
    vector<Q, XPU> t0 = ones<Q>(vol0, xpu);

    // Create tensor t1 of Nd+1 dims: an output lattice color vector
    bool preferred_layout_for_y_is_rowmajor = true;
    const char *o1 = preferred_layout_for_y_is_rowmajor ? "pxyztscn" : "pnxyztsc";
    Coor<Nd + 1> dim1 = preferred_layout_for_y_is_rowmajor
                            ? Coor<Nd + 1>{max_power, dim[X], dim[Y], dim[Z],
                                           dim[T],    dim[S], dim[C], dim[N]} // pxyztscn
                            : Coor<Nd + 1>{max_power, dim[N], dim[X], dim[Y],
                                           dim[Z],    dim[T], dim[S], dim[C]}; // pnxyztsc
    const Coor<Nd + 1> procs1 =
        preferred_layout_for_y_is_rowmajor
            ? Coor<Nd + 1>{1, procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}  // pxyztscn
            : Coor<Nd + 1>{1, 1, procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // pnxyztsc
    PartitionStored<Nd + 1> p1 = basic_partitioning(dim1, procs1);
    std::size_t vol1 = detail::volume(p1[rank][1]);
    vector<Q, XPU> t1 = ones<Q>(vol1, xpu);

    // Copy tensor t0 into each of the c components of tensor 1
    resetTimings();
    try {
        double t = timeit(
            [&] {
                Q *ptr0 = t0.data(), *ptr1 = t1.data();
                bsr_krylov<Nd - 1, Nd - 1, Nd + 1, Nd + 1, Q>(
#if VER <= 1
                    Q{1},
#endif
                    op, "xyztsc", "XYZTSC", p0.data(), 1, o0, {{}}, dim0, dim0, (const Q **)&ptr0,
#if VER == 0
                    Q{0},
#endif
                    p1.data(), o1, {{}}, dim1, dim1, 'p', &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                    MPI_COMM_WORLD,
#endif
                    SlowToFast);
            },
            [&] { sync(xpu); });
        if (rank == 0) std::cout << t / dim[N] << std::endl;
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    destroy_bsr(op);
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

    Coor<Nd> dim = {16, 16, 16, 32, 1, 12, 64}; // xyztscn
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1};
    int max_power = 1;

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
        } else if (std::strncmp("--power=", argv[i], 8) == 0) {
            if (sscanf(argv[i] + 8, "%d", &max_power) != 1) {
                std::cerr << "--power= should follow a number, for instance --power=3" << std::endl;
                return -1;
            }
            if (max_power < 1) {
                std::cerr << "The power should greater than zero" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--dim='x y z t n b'] [--procs='x y z t n b'] [--power=p] [--help]"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // If --procs isn't set, distributed the processes automatically
    if (!procs_was_set) {
        std::cerr << "Please set --procs`" << std::endl;
        return -1;
    }

#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext(rank % getGpuDevicesCount());
        test<std::complex<double>, Gpu>(dim, procs, rank, max_power, ctx, ctx.toGpu(0));
        clearCaches();
    }
#else
    {
        Context ctx = createCpuContext();
        test<std::complex<double>, Cpu>(dim, procs, rank, max_power, ctx, ctx.toCpu(0));
        clearCaches();
    }
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
