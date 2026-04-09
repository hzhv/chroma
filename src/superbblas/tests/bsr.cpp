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

/// Return the data pointers to a bunch of `vector`
template <typename T, typename XPU>
std::vector<T *> get_ptrs(const std::vector<vector<T, XPU>> &v) {
    std::vector<T *> ptrs;
    ptrs.reserve(v.size());
    for (const auto &i : v) ptrs.push_back(i.data());
    return ptrs;
}

/// Store a bunch of vectors and its data pointers for convenience
template <typename T, typename XPU> struct vectors {
    vectors(const std::vector<vector<T, XPU>> &v) : v(v), ptrs(get_ptrs(v)) {}
    T **data() const { return (T **)ptrs.data(); }
    const std::vector<vector<T, XPU>> &getVectors() const { return v; }

private:
    std::vector<vector<T, XPU>> v;
    std::vector<T *> ptrs;
};

// Return a vector of all zeros
template <typename T, typename XPU> vector<T, XPU> zeros(std::size_t size, XPU xpu) {
    vector<T, XPU> r(size, xpu);
    zero_n(r.data(), r.size(), xpu);
    return r;
}

// Return a vector of all ones
template <typename T, typename XPU> vector<T, XPU> ones(std::size_t size, XPU xpu) {
    vector<T, Cpu> r(size, Cpu{});
    for (std::size_t i = 0; i < size; ++i) r[i] = 1.0;
    return makeSure(r, xpu);
}

// Return a vector of vectorizing identity matrices
template <typename T, typename XPU>
vector<T, XPU> eyes(std::size_t n, std::size_t k, const T &scale, XPU xpu) {
    vector<T, Cpu> r(n * n * k, Cpu{});
    zero_n(r.data(), r.size(), r.ctx());
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < n; ++j) r[i * n * n + j * n + j] = scale;
    return makeSure(r, xpu);
}

// Return a vector of vectorizing identity matrices
template <typename T, typename XPU>
vector<T, XPU> anti_eyes(std::size_t n, std::size_t k, const T &scale, XPU xpu) {
    vector<T, Cpu> r(n * n * k, Cpu{});
    zero_n(r.data(), r.size(), r.ctx());
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < n; ++j) r[i * n * n + j * n + n - 1 - j] = scale;
    return makeSure(r, xpu);
}

// Return a vector of vectorizing identity matrices
template <typename T, typename XPU>
vector<T, XPU> random_sparse(std::size_t n, std::size_t k, const T &scale, XPU xpu) {
    vector<T, Cpu> r(n * n * k, Cpu{});
    zero_n(r.data(), r.size(), r.ctx());
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < n; ++j)
            r[i * n * n + j * n + n - 1 - j] = scale * (n % 2 ? T{1} : T{-1});
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

/// Extend the support for all regions, one element in each direction
PartitionStored<6> extend(const PartitionStored<6> &p, const Coor<6> &dim) {
    PartitionStored<6> r = p;
    for (auto &i : r) i = extend(i, dim);
    return r;
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

template <typename T> struct real_type {
    using type = T;
};
template <typename T> struct real_type<std::complex<T>> {
    using type = T;
};

template <typename T>
void get_lattice_nonzeros(const Coor<6> &row, const Coor<6> &col, unsigned int dir,
                          bool block_imaginary_fast, const Coor<6> &op_dim, T *v) {

    // ci,si,cd,sp,xi,yi,zi,ti,xd,yd,zd,td (fast index -> slow index)
    Coor<6, std::size_t> op_dim_stride = get_strides<std::size_t>(op_dim, SlowToFast);
    std::size_t vol_op = volume(op_dim);
    Coor<4> dim_blk{op_dim[5], op_dim[4], op_dim[5], op_dim[4]};
    Coor<4, std::size_t> stride_blk = get_strides<std::size_t>(dim_blk, FastToSlow);
    std::size_t disp_blk = coor2index(row, op_dim, op_dim_stride) * op_dim[4] * op_dim[5] +
                           coor2index(col, op_dim, op_dim_stride) * vol_op;
    std::size_t neighbors = max_neighbors(op_dim);
    std::size_t max_spin_component = neighbors * op_dim[4] * op_dim[4];
    using real_T = typename real_type<T>::type;
    std::size_t max_color_component =
        std::pow(std::numeric_limits<real_T>::radix, std::numeric_limits<real_T>::digits / 2) / 2 /
        neighbors / max_spin_component;
    for (int si = 0; si < op_dim[4]; ++si)
        for (int sd = 0; sd < op_dim[4]; ++sd)
            for (int ci = 0; ci < op_dim[5]; ++ci)
                for (int cd = 0; cd < op_dim[5]; ++cd)
                    v[coor2index(block_imaginary_fast ? Coor<4>{ci, si, cd, sd}
                                                      : Coor<4>{cd, sd, ci, si},
                                 dim_blk, stride_blk)] =
                        (1 + si + sd * op_dim[4] + dir * op_dim[4] * op_dim[4]) *
                        (1 + (disp_blk + ci + cd * op_dim[5]) % max_color_component);
}

template <typename T>
void get_lattice_nonzeros_block(const Coor<6> &row, const Coor<6> &col, bool block_imaginary_fast,
                                const Coor<6> &op_dim, T *v) {

    // ci,si,cd,sp,xi,yi,zi,ti,xd,yd,zd,td (fast index -> slow index)
    Coor<6, std::size_t> op_dim_stride = get_strides<std::size_t>(op_dim, SlowToFast);
    std::size_t vol_op = volume(op_dim);
    Coor<2> dim_blk{op_dim[5], op_dim[5]};
    Coor<2, std::size_t> stride_blk = get_strides<std::size_t>(dim_blk, FastToSlow);
    std::size_t disp_blk = coor2index(row, op_dim, op_dim_stride) * op_dim[4] * op_dim[5] +
                           coor2index(col, op_dim, op_dim_stride) * vol_op;
    std::size_t neighbors = max_neighbors(op_dim);
    std::size_t max_spin_component = neighbors * op_dim[4] * op_dim[4];
    using real_T = typename real_type<T>::type;
    std::size_t max_color_component =
        std::pow(std::numeric_limits<real_T>::radix, std::numeric_limits<real_T>::digits / 2) / 2 /
        neighbors / max_spin_component;
    for (int ci = 0; ci < op_dim[5]; ++ci)
        for (int cd = 0; cd < op_dim[5]; ++cd)
            v[coor2index(block_imaginary_fast ? Coor<2>{ci, cd} : Coor<2>{cd, ci}, dim_blk,
                         stride_blk)] = 1 + (disp_blk + ci + cd * op_dim[5]) % max_color_component;
}

template <typename T>
void get_lattice_nonzeros_kron(bool block_imaginary_fast, const Coor<6> &op_dim, T *v) {

    int neighbors = max_neighbors(op_dim);
    Coor<3> dim_blk{op_dim[4], op_dim[4], neighbors};
    Coor<3, std::size_t> stride_blk = get_strides<std::size_t>(dim_blk, FastToSlow);
    for (int dir = 0; dir < neighbors; ++dir)
        for (int si = 0; si < op_dim[4]; ++si)
            for (int sd = 0; sd < op_dim[4]; ++sd)
                v[coor2index(block_imaginary_fast ? Coor<3>{si, sd, dir} : Coor<3>{sd, si, dir},
                             dim_blk, stride_blk)] =
                    1 + si + sd * op_dim[4] + dir * op_dim[4] * op_dim[4];
}

/// Create a 4D lattice with dimensions tzyxsc
template <typename T, typename XPU>
std::pair<BSR_handle *, vectors<T, XPU>>
create_lattice(const PartitionStored<6> &pi, int rank, const Coor<6> op_dim,
               const std::vector<Context> &ctx, const std::vector<XPU> &xpu) {
    bool check_results = getDebugLevel() > 0;

    // Compute how many neighbors
    int neighbors = max_neighbors(op_dim);

    // Compute the domain ranges
    PartitionStored<6> pd = extend(pi, op_dim);

    const bool nonzero_blocks_imaginary_fast = false;

    std::vector<vector<IndexType, XPU>> ii_xpus;
    std::vector<vector<Coor<6>, XPU>> jj_xpus;
    std::vector<vector<T, XPU>> data_xpus;
    for (unsigned int component = 0; component < ctx.size(); ++component) {
        int rank_comp = rank * ctx.size() + component;
        Coor<6> from = pi[rank_comp][0]; // first nonblock dimensions of the RSB image
        Coor<6> dimi = pi[rank_comp][1]; // nonblock dimensions of the RSB image
        auto fromd = pd[rank_comp][0];   // first nonblock dimension of the RSB domain
        dimi[4] = dimi[5] = 1;
        std::size_t voli = volume(dimi);
        vector<IndexType, Cpu> ii(voli, Cpu{});

        // Compute the coordinates for all nonzeros
        for (auto &i : ii) i = neighbors;
        vector<Coor<6>, Cpu> jj(neighbors * voli, Cpu{});
        Coor<6, std::size_t> stride = get_strides<std::size_t>(dimi, SlowToFast);
        for (std::size_t i = 0, j = 0; i < voli; ++i) {
            Coor<6> c = index2coor(i, dimi, stride) + from;
            jj[j++] = normalize_coor(c - fromd, op_dim);
            for (int dim = 0; dim < 4; ++dim) {
                if (op_dim[dim] == 1) continue;
                for (int dir = -1; dir < 2; dir += 2) {
                    Coor<6> c0 = c;
                    c0[dim] += dir;
                    jj[j++] = normalize_coor(c0 - fromd, op_dim);
                    if (op_dim[dim] <= 2) break;
                }
            }
        }

        // Number of nonzeros
        std::size_t vol_data = voli * neighbors * op_dim[4] * op_dim[5] * op_dim[4] * op_dim[5];
        if (rank_comp == 0)
            std::cout << "Size of the sparse tensor per process: "
                      << vol_data * 1.0 * sizeof(T) / 1024 / 1024 << " MiB" << std::endl;

        vector<T, XPU> data_xpu;
        if (check_results) {
            vector<T, Cpu> data_cpu(vol_data, Cpu{});
            Coor<6, std::size_t> stride = get_strides<std::size_t>(dimi, SlowToFast);
            std::size_t vol_blk = op_dim[4] * op_dim[5] * op_dim[4] * op_dim[5];
            for (std::size_t i = 0, j = 0; i < voli; ++i) {
                Coor<6> blk_row = normalize_coor(index2coor(i, dimi, stride) + from, op_dim);
                for (int j0 = 0; j0 < neighbors; ++j, ++j0) {
                    Coor<6> blk_col = normalize_coor(jj[j] + fromd, op_dim);
                    get_lattice_nonzeros(blk_row, blk_col, j0, nonzero_blocks_imaginary_fast,
                                         op_dim, data_cpu.data() + j * vol_blk);
                }
            }
            data_xpu = makeSure(data_cpu, xpu[component]);
        } else {
            data_xpu = ones<T>(vol_data, xpu[component]);
        }

        ii_xpus.push_back(makeSure(ii, xpu[component]));
        jj_xpus.push_back(makeSure(jj, xpu[component]));
        data_xpus.push_back(data_xpu);
    }

    Coor<6> block{{1, 1, 1, 1, op_dim[4], op_dim[5]}};
    BSR_handle *bsrh = nullptr;
    vectors<IndexType, XPU> iiv(ii_xpus);
    vectors<Coor<6>, XPU> jjv(jj_xpus);
    vectors<T, XPU> datav(data_xpus);
    create_bsr<6, 6, T>(pi.data(), op_dim, pd.data(), op_dim, ctx.size(), block, block,
                        nonzero_blocks_imaginary_fast, iiv.data(), jjv.data(),
                        (const T **)datav.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast, &bsrh);
    return {bsrh, datav};
}

template <typename T>
void get_lattice_constraction_value(const Coor<6> &row, const Coor<6> &col, unsigned int dir,
                                    const Coor<6> &dim, unsigned int ncols, std::vector<T> &v) {

    // Fill the nonzeros of the sparse matrix
    std::size_t dim_sc = dim[4] * dim[5];
    std::vector<T> a(dim_sc * dim_sc);
    Coor<6, std::size_t> dim_stride = get_strides<std::size_t>(dim, SlowToFast);
    get_lattice_nonzeros(row, col, dir, true, dim, a.data());

    // Fill the values of the right-hand-size, the order is nxyztsc from slow to fast index
    using real_T = typename real_type<T>::type;
    std::size_t max_rhs_component =
        std::pow(std::numeric_limits<real_T>::radix, std::numeric_limits<real_T>::digits / 2) / 2;
    std::size_t vol_blk = dim[4] * dim[5] * ncols, vol_dim = volume(dim);
    std::vector<T> rhs(vol_blk);
    std::size_t disp_rhs = coor2index(col, dim, dim_stride);
    for (std::size_t n = 0; n < ncols; ++n)
        for (std::size_t i = 0; i < dim_sc; ++i)
            rhs[i + n * dim_sc] = 1 + (disp_rhs + i + n * vol_dim) % max_rhs_component;

    // Multiply the matrices `a` and `rhs`
    for (std::size_t n = 0; n < ncols; ++n)
        for (std::size_t i = 0; i < dim_sc; ++i)
            for (std::size_t k = 0; k < dim_sc; ++k)
                v[i + n * dim_sc] += a[i + k * dim_sc] * rhs[k + n * dim_sc];
}

/// Create a 4D lattice with dimensions tzyxsc
template <std::size_t N, typename T, typename XPU>
void test_contraction(const PartitionStored<N> &pi, int rank, const char *oy_,
                      const vectors<T, XPU> &yv, const Coor<6> &op_dim, bool do_fast_check,
                      const std::vector<Context> &ctxs) {

    for (std::size_t component = 0; component < ctxs.size(); ++component) {
        const auto y = yv.getVectors()[component];
        const auto ctx = ctxs[component];
        int component_idx = rank * ctxs.size() + component;
        if (getDebugLevel() == 0) {
            if (do_fast_check) {
                vector<T, Cpu> y_cpu = makeSure(y, Cpu{});
                T right_value = (T)(max_neighbors(op_dim) * op_dim[4] * op_dim[5]);
                for (std::size_t i = 0; i < y_cpu.size(); ++i)
                    if (std::norm(y_cpu[i] - right_value) > 1e-2)
                        throw std::runtime_error("check error");
            }
        } else {
            // Reorder y
            Order<N> oy = toArray<N>(oy_, "");
            Context cpu_ctx = createCpuContext();
            Order<7> oy_cpu = toArray<7>("xyztnsc", "");
            Coor<7> dimy_cpu = reorder_coor(pi[component_idx][1], find_permutation(oy, oy_cpu), 1);
            vector<T, Cpu> y_cpu(y.size(), Cpu{});
            PartitionStored<N> p0(1, {Coor<N>{{}}, pi[component_idx][1]});
            PartitionStored<7> p1(1, {Coor<7>{{}}, dimy_cpu});
            const T *ptr0 = y.data();
            T *ptr1 = y_cpu.data();
            copy<N, 7, T, T>(T{1}, p0.data(), 1, oy_, Coor<N>{{}}, pi[component_idx][1],
                             pi[component_idx][1], &ptr0, nullptr, &ctx, p1.data(), 1, "xyztnsc",
                             Coor<7>{{}}, dimy_cpu, &ptr1, nullptr, &cpu_ctx, SlowToFast, Copy);

            unsigned int ncols = dimy_cpu[4];

            // Compute the coordinates for all nonzeros
            Order<6> op_o = toArray<6>("xyztsc", "");
            Coor<6> from = reorder_coor(pi[component_idx][0], find_permutation(oy, op_o), 1);
            Coor<6> dimi = reorder_coor(pi[component_idx][1], find_permutation(oy, op_o), 1);
            dimi[4] = dimi[5] = 1;
            std::size_t voli = volume(dimi);
            Coor<6, std::size_t> stride = get_strides<std::size_t>(dimi, SlowToFast);
            for (std::size_t i = 0; i < voli; ++i) {
                Coor<6> c = index2coor(i, dimi, stride) + from;
                std::vector<T> right_values(op_dim[4] * op_dim[5] * ncols);
                unsigned int dir_right_values = 0;
                get_lattice_constraction_value(c, c, dir_right_values++, op_dim, ncols,
                                               right_values);
                for (int dim = 0; dim < 4; ++dim) {
                    if (op_dim[dim] == 1) continue;
                    for (int dir = -1; dir < 2; dir += 2) {
                        Coor<6> c0 = c;
                        c0[dim] += dir;
                        c0 = normalize_coor(c0, op_dim);
                        get_lattice_constraction_value(c, c0, dir_right_values++, op_dim, ncols,
                                                       right_values);
                        if (op_dim[dim] <= 2) break;
                    }
                }

                // Compare the results
                for (unsigned int k = 0, vol_blk = op_dim[4] * op_dim[5] * ncols; k < vol_blk; ++k)
                    if (std::norm(y_cpu[i * vol_blk + k] - right_values[k]) > 0.1)
                        throw std::runtime_error("check error");
            }
        }
    }
}

/// Given a range removes all elements at distance two from another range
/// \param fs: input range from which remove elements
/// \param range: reference range
/// \param dim: global lattice dimensions
///
/// The goal of the function is given two ranges with two or more dimensions,
/// the first one containing the second one, the function returns the first
/// range without the elements that are at distance two from the second range:
/// The function first create a hole removing all elements outside `range`:
///
///      input:          output:
///      * * * * *         * * *
///      * x x x *       * x x x *
///      * x x x *  ==>  * x x x *
///      * x x x *       * x x x *
///      * * * * *         * * *
///
///        - - - - - - -     fs
///            - - -         range
///    ------] - - - [------ range to remove
///    - - - - - - - - - - - dim
template <std::size_t Nd>
PartitionItem<Nd> remove_corners(const PartitionItem<Nd> &fs, const PartitionItem<Nd> &range,
                                 const Coor<Nd> &dim) {
    std::size_t hole_dims = 0; // number of dimensions of the hole
    PartitionItem<Nd> hole;
    for (std::size_t i = 0; i < Nd; ++i) {
        if (fs[1][i] == range[1][i]) {
            hole[0][i] = 0;
            hole[1][i] = dim[i];
        } else {
            hole[0][i] = range[0][i] + range[1][i];
            hole[1][i] = dim[i] - range[1][i];
            hole_dims++;
        }
    }

    // If no dimension get larger than `range`, return the same input;
    // otherwise, don't do it because there's no element a distance two
    if (hole_dims == 0) return fs;
    auto new_fs = make_hole(fs[0], fs[1], hole[0], hole[1], dim);

    // We expect that the input range will be larger than `range` on only one side,
    // so a single piece should remained after making the hole
    if (new_fs.size() != 1) throw std::runtime_error("This shouldn't happen");
    return new_fs.front();
}

/// Create a 4D lattice with dimensions tzyxsc, but the domain is split in several pieces in a way
/// that there is a piece that doesn't need communications (the core). This approach should overlap
/// most of the local sparse-dense matrix product with the halo exchanges.

template <typename T, typename XPU>
std::pair<std::vector<BSR_handle *>, std::vector<vector<T, XPU>>>
create_lattice_split(const PartitionStored<6> &pi, int rank, const Coor<6> op_dim,
                     const std::vector<Context> &ctx, const std::vector<XPU> &xpu) {
    bool check_results = getDebugLevel() > 0;

    // Compute the domain ranges
    PartitionStored<6> pd = extend(pi, op_dim);

    // Split the local part into the halo and the core
    PartitionStored<6> zero_part(pd.size());
    std::vector<PartitionStored<6>> pd_s(6 * 2, zero_part);
    for (unsigned int i = 0; i < pd.size(); ++i) {
        auto parts = make_hole(pd[i][0], pd[i][1], pi[i][0], pi[i][1], op_dim);
        if (parts.size() > pd_s.size()) throw std::runtime_error("this shouldn't happen");
        for (unsigned int j = 0; j < parts.size(); ++j)
            pd_s[j][i] = remove_corners(parts[j], pd[i], op_dim);
    }
    {
        std::vector<PartitionStored<6>> pd_s_aux;
        for (const auto &p : pd_s)
            if (p != zero_part) pd_s_aux.push_back(p);
        pd_s = pd_s_aux;
    }
    pd_s.push_back(pi);
    std::vector<PartitionStored<6>> pi_s(pd_s.size(), zero_part);
    for (unsigned int i = 0; i < pd.size(); ++i) {
        for (unsigned int p = 0; p < pd_s.size(); ++p) {
            auto fs = extend(pd_s[p][i], op_dim);
            intersection(pi[i][0], pi[i][1], fs[0], fs[1], op_dim, pi_s[p][i][0], pi_s[p][i][1]);
            if (volume(pi_s[p][i][1]) == 0 || volume(pd_s[p][i][1]) == 0)
                pi_s[p][i][0] = pi_s[p][i][1] = pd_s[p][i][0] = pd_s[p][i][1] = Coor<6>{{}};
        }
    }

    // Compute the coordinates for all nonzeros
    std::vector<BSR_handle *> bsrh_s;
    std::vector<vector<T, XPU>> data_s;
    std::size_t total_vol_data = 0;
    int global_neighbors = max_neighbors(op_dim);
    const bool nonzero_blocks_imaginary_fast = false;
    Coor<6> block{{1, 1, 1, 1, op_dim[4], op_dim[5]}};
    for (unsigned int p = 0; p < pd_s.size(); ++p) {
        std::vector<vector<IndexType, XPU>> ii_xpus;
        std::vector<vector<Coor<6>, XPU>> jj_xpus;
        std::vector<vector<T, XPU>> data_xpus;
        for (unsigned int component = 0; component < ctx.size(); ++component) {
            int rank_comp = rank * ctx.size() + component;
            Coor<6> dimi = pi_s[p][rank_comp][1]; // nonblock dimensions of the RSB image
            dimi[4] = dimi[5] = 1;
            Coor<6> dimd = pd_s[p][rank_comp][1];

            // Allocate and fill the column indices
            std::size_t voli = volume(dimi);
            vector<Coor<6>, Cpu> jj(voli * global_neighbors, Cpu{});
            Coor<6> fromi = pi_s[p][rank_comp][0]; // first nonblock element of the RSB image
            Coor<6> fromd = pd_s[p][rank_comp][0]; // first element of the domain
            Coor<6, std::size_t> stridei = get_strides<std::size_t>(dimi, SlowToFast);
            std::size_t neighbors = 0;
            std::size_t vol_data = 0;
            vector<T, Cpu> data_cpu;
            std::size_t vol_blk = op_dim[4] * op_dim[5] * op_dim[4] * op_dim[5];
            for (unsigned int k = 0; k < 2; ++k) {
                if (k == 1) {
                    // Once we know the maximum number of neighbors...
                    vol_data = voli * neighbors * op_dim[4] * op_dim[5] * op_dim[4] * op_dim[5];
                    total_vol_data += vol_data;
                    if (check_results) data_cpu = zeros<T>(vol_data, Cpu{});
                }
                for (std::size_t i = 0, j = 0; i < voli; ++i) {
                    std::size_t j0 = j;
                    unsigned int op_dir = 0;
                    Coor<6> c = normalize_coor(index2coor(i, dimi, stridei) + fromi, op_dim);
                    if (p == pd_s.size() - 1) {
                        if (k == 1 && check_results)
                            get_lattice_nonzeros(c, c, op_dir, nonzero_blocks_imaginary_fast,
                                                 op_dim, data_cpu.data() + j * vol_blk);
                        jj[j++] = normalize_coor(c - fromd, op_dim);
                    }
                    op_dir++;
                    for (int dim = 0; dim < 4; ++dim) {
                        if (op_dim[dim] == 1) continue;
                        for (int dir = -1; dir < 2; dir += 2) {
                            Coor<6> c0 = c;
                            c0[dim] += dir;
                            c0 = normalize_coor(c0, op_dim);
                            Coor<6> from0, size0;
                            intersection(c0, block, fromd, dimd, op_dim, from0, size0);
                            if (volume(size0) > 0) {
                                if (k == 1 && check_results)
                                    get_lattice_nonzeros(c, c0, op_dir,
                                                         nonzero_blocks_imaginary_fast, op_dim,
                                                         data_cpu.data() + j * vol_blk);
                                jj[j++] = normalize_coor(c0 - fromd, op_dim);
                            }
                            op_dir++;
                            if (op_dim[dim] <= 2) break;
                        }
                    }
                    neighbors = std::max(neighbors, j - j0);
                    if (k == 1) {
                        if (j == j0 && neighbors > 0) jj[j++] = Coor<6>{{}};
                        for (std::size_t j1 = j0 + neighbors; j < j1; ++j) jj[j] = jj[j - 1];
                    }
                }
            }
            jj.resize(voli * neighbors);

            // Allocate and fill the row indices
            vector<IndexType, Cpu> ii(voli, Cpu{});
            for (auto &i : ii) i = neighbors;

            ii_xpus.push_back(makeSure(ii, xpu[component]));
            jj_xpus.push_back(makeSure(jj, xpu[component]));
            data_xpus.push_back(check_results ? makeSure(data_cpu, xpu[component])
                                              : ones<T>(vol_data, xpu[component]));
            total_vol_data += vol_data;
        }
        BSR_handle *bsrh = nullptr;
        vectors<IndexType, XPU> iiv(ii_xpus);
        vectors<Coor<6>, XPU> jjv(jj_xpus);
        vectors<T, XPU> datav(data_xpus);
        create_bsr<6, 6, T>(pi_s[p].data(), op_dim, pd_s[p].data(), op_dim, ctx.size(), block,
                            block, nonzero_blocks_imaginary_fast, iiv.data(), jjv.data(),
                            (const T **)datav.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                            MPI_COMM_WORLD,
#endif
                            SlowToFast, &bsrh);
        bsrh_s.push_back(bsrh);
        for (const auto &data_xpu : data_xpus) data_s.push_back(data_xpu);
    }

    // Number of nonzeros
    if (rank == 0)
        std::cout << "Size of the sparse tensor per process: "
                  << total_vol_data * 1.0 * sizeof(T) / 1024 / 1024 << " MiB" << std::endl;

    return {bsrh_s, data_s};
}

enum KronSparsity { Dense = 0, Identity = 1, Perm = 2, PermScale = 3 };

/// Create a 4D lattice with dimensions tzyxsc and Kronecker products
template <typename T, typename XPU>
std::pair<BSR_handle *, std::array<vectors<T, XPU>, 2>>
create_lattice_kron(const PartitionStored<6> &pi, int rank, KronSparsity sparse_kron,
                    const Coor<6> op_dim, bool show_size, const std::vector<Context> &ctx,
                    const std::vector<XPU> &xpu) {

    bool check_results = getDebugLevel() > 0;

    const bool nonzero_blocks_imaginary_fast = false;

    // Compute how many neighbors
    int neighbors = max_neighbors(op_dim);

    // Compute the domain ranges
    PartitionStored<6> pd = extend(pi, op_dim);

    std::vector<vector<IndexType, XPU>> ii_xpus;
    std::vector<vector<Coor<6>, XPU>> jj_xpus;
    std::vector<vector<T, XPU>> data_xpus;
    std::vector<vector<T, XPU>> kron_xpus;
    for (unsigned int component = 0; component < ctx.size(); ++component) {
        int rank_comp = rank * ctx.size() + component;
        Coor<6> from = pi[rank_comp][0]; // first nonblock dimensions of the RSB image
        Coor<6> dimi = pi[rank_comp][1]; // nonblock dimensions of the RSB image
        dimi[4] = dimi[5] = 1;
        std::size_t voli = volume(dimi);
        vector<IndexType, Cpu> ii(voli, Cpu{});

        // Compute the coordinates for all nonzeros
        for (auto &i : ii) i = neighbors;
        vector<Coor<6>, Cpu> jj(neighbors * voli, Cpu{});
        Coor<6, std::size_t> stride = get_strides<std::size_t>(dimi, SlowToFast);
        for (std::size_t i = 0, j = 0; i < voli; ++i) {
            Coor<6> c = index2coor(i, dimi, stride) + from;
            jj[j++] = normalize_coor(c - pd[rank_comp][0], op_dim);
            for (int dim = 0; dim < 4; ++dim) {
                if (op_dim[dim] == 1) continue;
                for (int dir = -1; dir < 2; dir += 2) {
                    Coor<6> c0 = c;
                    c0[dim] += dir;
                    jj[j++] = normalize_coor(c0 - pd[rank_comp][0], op_dim);
                    if (op_dim[dim] <= 2) break;
                }
            }
        }

        // Number of nonzeros
        std::size_t vol_data = voli * neighbors * op_dim[5] * op_dim[5];
        if (show_size && rank_comp == 0)
            std::cout << "Size of the sparse tensor per process: "
                      << vol_data * 1.0 * sizeof(T) / 1024 / 1024 << " MiB" << std::endl;
        std::size_t vol_kron = neighbors * op_dim[4] * op_dim[4];

        vector<T, XPU> data_xpu, kron_xpu;
        if (check_results) {
            vector<T, Cpu> data_cpu(vol_data, Cpu{});
            Coor<6, std::size_t> stride = get_strides<std::size_t>(dimi, SlowToFast);
            std::size_t vol_blk = op_dim[5] * op_dim[5];
            for (std::size_t i = 0, j = 0; i < voli; ++i) {
                Coor<6> blk_row = normalize_coor(index2coor(i, dimi, stride) + from, op_dim);
                for (int j0 = 0; j0 < neighbors; ++j, ++j0) {
                    Coor<6> blk_col = normalize_coor(jj[j] + pd[rank_comp][0], op_dim);
                    get_lattice_nonzeros_block(blk_row, blk_col, nonzero_blocks_imaginary_fast,
                                               op_dim, data_cpu.data() + j * vol_blk);
                }
            }
            data_xpu = makeSure(data_cpu, xpu[component]);
            vector<T, Cpu> kron_cpu(vol_kron, Cpu{});
            get_lattice_nonzeros_kron(nonzero_blocks_imaginary_fast, op_dim, kron_cpu.data());
            kron_xpu = makeSure(kron_cpu, xpu[component]);
        } else {
            data_xpu = ones<T>(vol_data, xpu[component]);
            kron_xpu =
                sparse_kron == Dense
                    ? ones<T>(vol_kron, xpu[component])
                    : (sparse_kron == Identity
                           ? eyes<T>(op_dim[4], neighbors, (T)op_dim[4], xpu[component])
                           : (sparse_kron == Perm
                                  ? anti_eyes<T>(op_dim[4], neighbors, (T)op_dim[4], xpu[component])
                                  : random_sparse<T>(op_dim[4], neighbors, (T)op_dim[4],
                                                     xpu[component])));
        }

        ii_xpus.push_back(makeSure(ii, xpu[component]));
        jj_xpus.push_back(makeSure(jj, xpu[component]));
        data_xpus.push_back(data_xpu);
        kron_xpus.push_back(kron_xpu);
    }

    Coor<6> block{{1, 1, 1, 1, 1, op_dim[5]}};
    Coor<6> kron{{1, 1, 1, 1, op_dim[4], 1}};
    BSR_handle *bsrh = nullptr;
    vectors<IndexType, XPU> iiv(ii_xpus);
    vectors<Coor<6>, XPU> jjv(jj_xpus);
    vectors<T, XPU> datav(data_xpus);
    vectors<T, XPU> kronv(kron_xpus);
    create_kron_bsr<6, 6, T>(pi.data(), op_dim, pd.data(), op_dim, ctx.size(), block, block, kron,
                             kron, nonzero_blocks_imaginary_fast, iiv.data(), jjv.data(),
                             (const T **)datav.data(), (const T **)kronv.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                             MPI_COMM_WORLD,
#endif
                             SlowToFast, &bsrh);
    return {bsrh, {datav, kronv}};
}

/// Create a 7D tensor with dimensions ntzyxsc
template <typename T, std::size_t N, typename XPU>
vectors<T, XPU> create_tensor_data(const PartitionStored<N> &p, int rank, const char *o_,
                                   const Coor<6> &op_dim, unsigned int ncols,
                                   const std::vector<XPU> &xpu) {

    std::vector<vector<T, XPU>> r;
    for (unsigned int component = 0; component < xpu.size(); ++component) {
        std::size_t vol = volume(p[rank * xpu.size() + component][1]);
        if (getDebugLevel() == 0) {
            r.push_back(ones<T>(vol, xpu[component]));
        } else {
            using real_T = typename real_type<T>::type;
            std::size_t max_rhs_component = std::pow(std::numeric_limits<real_T>::radix,
                                                     std::numeric_limits<real_T>::digits / 2) /
                                            2;

            const Order<N> o = toArray<N>(o_, "");
            const Order<7> filling_order = toArray<7>("nxyztsc", "");
            vector<T, Cpu> data(vol, Cpu{});
            Coor<7> perm = find_permutation(o, filling_order);
            Coor<7> dim_filling{(int)ncols};
            std::copy_n(op_dim.begin(), 6, dim_filling.begin() + 1);
            Coor<7, std::size_t> dim_filling_stride =
                get_strides<std::size_t>(dim_filling, SlowToFast);
            Coor<N, std::size_t> stride =
                get_strides<std::size_t>(p[rank * xpu.size() + component][1], SlowToFast);
            for (std::size_t i = 0; i < vol; ++i)
                data[i] =
                    1 +
                    (coor2index(normalize_coor(
                                    reorder_coor(
                                        index2coor(i, p[rank * xpu.size() + component][1], stride) +
                                            p[rank * xpu.size() + component][0],
                                        perm),
                                    dim_filling),
                                dim_filling, dim_filling_stride)) %
                        max_rhs_component;
            r.push_back(makeSure(data, xpu[component]));
        }
    }
    return vectors<T, XPU>(r);
}

template <typename Q, typename XPU>
void test(Coor<Nd> dim, Coor<Nd> procs, int rank, int nprocs, int max_power, unsigned int nrep,
          const std::vector<Context> &ctx, const std::vector<XPU> &xpu) {

    // Create a lattice operator of Nd-1 dims
    const Coor<Nd - 1> dimo = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
    const Coor<Nd - 1> procso = {procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // xyztsc
    PartitionStored<Nd - 1> po =
        basic_partitioning("xyztsc", dimo, procso, "xyzt", nprocs, ctx.size());
    for (int i = 0; i < max_power - 1; ++i) po = extend(po, dimo);
    auto op_pair = create_lattice<Q>(po, rank, dimo, ctx, xpu);
    BSR_handle *op = op_pair.first;

    // Create tensor t0 of Nd dims: an input lattice color vector
    const Coor<Nd + 1> dim0 = {1,      dim[X], dim[Y], dim[Z],
                               dim[T], dim[S], dim[C], dim[N]};                       // pxyztscn
    const Coor<Nd + 1> procs0 = {1, procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}; // pxyztscn
    PartitionStored<Nd + 1> p0 =
        basic_partitioning("pxyztscn", dim0, procs0, "xyzt", nprocs, ctx.size());
    vectors<Q, XPU> t0 = create_tensor_data<Q>(p0, rank, "pxyztscn", dimo, dim[N], xpu);

    // Get preferred layout for the output tensor
    std::vector<MatrixLayout> preferred_layout_for_x_v(ctx.size()),
        preferred_layout_for_y_v(ctx.size());
    bsr_get_preferred_layout<Nd - 1, Nd - 1, Q>(op, ctx.size(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                                                MPI_COMM_WORLD,
#endif
                                                SlowToFast, preferred_layout_for_x_v.data(),
                                                preferred_layout_for_y_v.data());
    MatrixLayout preferred_layout_for_x = preferred_layout_for_x_v.front(),
                 preferred_layout_for_y = preferred_layout_for_y_v.front();
    (void)preferred_layout_for_x;

    // Create tensor t1 of Nd+1 dims: an output lattice color vector
    const char *o1 = preferred_layout_for_y == RowMajor ? "pxyztscn" : "pnxyztsc";
    Coor<Nd + 1> dim1 = preferred_layout_for_y == RowMajor
                            ? Coor<Nd + 1>{max_power, dim[X], dim[Y], dim[Z],
                                           dim[T],    dim[S], dim[C], dim[N]} // pxyztscn
                            : Coor<Nd + 1>{max_power, dim[N], dim[X], dim[Y],
                                           dim[Z],    dim[T], dim[S], dim[C]}; // pnxyztsc
    const Coor<Nd + 1> procs1 =
        preferred_layout_for_y == RowMajor
            ? Coor<Nd + 1>{1, procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}  // pxyztscn
            : Coor<Nd + 1>{1, 1, procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // pnxyztsc
    PartitionStored<Nd + 1> p1 = basic_partitioning(o1, dim1, procs1, "xyzt", nprocs, ctx.size());
    vectors<Q, XPU> t1 = create_tensor_data<Q>(p1, rank, o1, dimo, dim[N], xpu);

    const bool is_cpu = deviceId(xpu[0]) == CPU_DEVICE_ID;
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    if (rank == 0)
        std::cout << ">>> " << (is_cpu ? "CPU" : "GPU") << " tests with " << num_threads
                  << " threads" << std::endl;

    if (rank == 0) {
        std::size_t vol1 = detail::volume(p1[0][1]);
        std::cout << "Maximum number of elements in a tested tensor per component: " << vol1
                  << " ( " << vol1 * 1.0 * sizeof(Q) / 1024 / 1024 << " MiB)" << std::endl;
    }

    // Copy tensor t0 into each of the c components of tensor 1
    resetTimings();
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            bsr_krylov<Nd - 1, Nd - 1, Nd + 1, Nd + 1, Q>(
                Q{1}, op, "xyztsc", "XYZTSC", p0.data(), ctx.size(), "pXYZTSCn", {{}}, dim0, dim0,
                (const Q **)t0.data(), Q{0}, p1.data(), o1, {{}}, dim1, dim1, 'p', t1.data(),
                ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                MPI_COMM_WORLD,
#endif
                SlowToFast);
        }
        for (const auto &xpui : xpu) sync(xpui);
        t = w_time() - t;
        if (rank == 0) std::cout << "Time in mavec per rhs: " << t / nrep / dim[N] << std::endl;
        test_contraction(p1, rank, o1, t1, dimo, true, ctx);
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    destroy_bsr(op);

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    // Create split tensor
    auto op_pair_s = create_lattice_split<Q>(po, rank, dimo, ctx, xpu);

    // Copy tensor t0 into each of the c components of tensor 1
    resetTimings();
    try {
        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            // Set the output tensor to zero
            copy(0, p1.data(), ctx.size(), o1, {{}}, dim1, dim1, (const Q **)t0.data(), nullptr,
                 ctx.data(), p1.data(), ctx.size(), o1, {{}}, dim1, t1.data(), nullptr, ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                 MPI_COMM_WORLD,
#endif
                 SlowToFast, Copy);

            // Do the contractions on each part
            std::vector<Request> r(op_pair_s.first.size());
            for (unsigned int p = 0; p < op_pair_s.first.size(); ++p) {
                bsr_krylov<Nd - 1, Nd - 1, Nd + 1, Nd + 1, Q>(
                    Q{1}, op_pair_s.first[p], "xyztsc", "XYZTSC", p0.data(), ctx.size(), "pXYZTSCn",
                    {{}}, dim0, dim0, (const Q **)t0.data(), Q{1}, p1.data(), o1, {{}}, dim1, dim1,
                    'p', t1.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                    MPI_COMM_WORLD,
#endif
                    SlowToFast, &r[p]);
            }
            for (const auto &ri : r) wait(ri);
        }

        for (const auto &xpui : xpu) sync(xpui);
        t = w_time() - t;
        if (rank == 0)
            std::cout << "Time in mavec per rhs (split): " << t / nrep / dim[N] << std::endl;
        test_contraction(p1, rank, o1, t1, dimo, false /* Don't do quick check, it isn't correct */,
                         ctx);
    } catch (const std::exception &e) { std::cout << "Caught error: " << e.what() << std::endl; }

    for (const auto op : op_pair_s.first) destroy_bsr(op);

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);

    const Coor<Nd + 1> kdim0 = {1,      dim[X], dim[Y], dim[Z],
                                dim[T], dim[C], dim[N], dim[S]}; // pxyztcns
    PartitionStored<Nd + 1> kp0 =
        basic_partitioning("pxyztcns", kdim0, procs0, "xyzt", nprocs, ctx.size());
    t0 = create_tensor_data<Q>(kp0, rank, "pxyztcns", dimo, dim[N], xpu);
    Coor<Nd + 1> kdim1 =
        Coor<Nd + 1>{max_power, dim[X], dim[Y], dim[Z], dim[T], dim[C], dim[N], dim[S]}; // pxyztcns
    const Coor<Nd + 1> kprocs1 =
        Coor<Nd + 1>{1, procs[X], procs[Y], procs[Z], procs[T], 1, 1, 1}; // pxyztcns
    PartitionStored<Nd + 1> kp1 =
        basic_partitioning("pxyztcns", kdim1, kprocs1, "xyzt", nprocs, ctx.size());

    const std::vector<std::string> kron_sparse_str{"dense", "identity", "permutation",
                                                   "general-sparse"};
    for (int kron_sparse = 1; kron_sparse < 4; kron_sparse++) {
        // Create the Kronecker operator
        auto op_kron_s = create_lattice_kron<Q>(po, rank, (KronSparsity)kron_sparse, dimo,
                                                kron_sparse == 0 /* show size */, ctx, xpu);

        // Copy tensor t0 into each of the c components of tensor 1
        resetTimings();
        try {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                bsr_krylov<Nd - 1, Nd - 1, Nd + 1, Nd + 1, Q>(
                    Q{1}, op_kron_s.first, "xyztsc", "XYZTSC", kp0.data(), ctx.size(), "pXYZTCnS",
                    {{}}, kdim0, kdim0, (const Q **)t0.data(), Q{0}, kp1.data(), "pxyztcns", {{}},
                    kdim1, kdim1, 'p', t1.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                    MPI_COMM_WORLD,
#endif
                    SlowToFast);
            }
            for (const auto &xpui : xpu) sync(xpui);
            t = w_time() - t;
            if (rank == 0)
                std::cout << "Time in mavec per rhs (kron " << kron_sparse_str[kron_sparse]
                          << "): " << t / nrep / dim[N] << std::endl;
            test_contraction(kp1, rank, "pxyztcns", t1, dimo, true, ctx);
        } catch (const std::exception &e) {
            std::cout << "Caught error: " << e.what() << std::endl;
        }

        destroy_bsr(op_kron_s.first);

        if (rank == 0) reportTimings(std::cout);
        if (rank == 0) reportCacheUsage(std::cout);
    }
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
    int nrep = getDebugLevel() == 0 ? 10 : 1;
    int ncomponents = 0;

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
                std::cerr << "The power should be greater than zero" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--components=", argv[i], 13) == 0) {
            if (sscanf(argv[i] + 13, "%d", &ncomponents) != 1) {
                std::cerr << "--components= should follow a number, for instance --components=2"
                          << std::endl;
                return -1;
            }
            if (ncomponents < 0) {
                std::cerr << "The number of components shouldn't be negative" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--rep=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d", &nrep) != 1) {
                std::cerr << "--rep= should follow a number, for instance --rep=3" << std::endl;
                return -1;
            }
            if (nrep < 1) {
                std::cerr << "The rep should be greater than zero" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--dim='x y z t n b'] [--procs='x y z t n b'] [--power=p] "
                         "[--components=c] [--help]"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // Set the default number of components
    ncomponents = ncomponents == 0 ? 1 : ncomponents;

    // Simulate spin and color as LQCD
    if (dim[C] % 4 == 0) {
        dim[S] = 4;
        dim[C] /= 4;
    }

    // If --procs isn't set, distributed the processes automatically
    if (!procs_was_set) procs = partitioning_distributed_procs("xyztscn", dim, "tzyx", nprocs);

    // Show lattice dimensions and processes arrangement
    if (rank == 0) {
        std::cout << "Testing lattice dimensions xyzt= " << dim[X] << " " << dim[Y] << " " << dim[Z]
                  << " " << dim[T] << " spin-color= " << dim[S] << " " << dim[C]
                  << "  num_vecs= " << dim[N] << std::endl;
        std::cout << "Processes arrangement xyzt= " << procs[X] << " " << procs[Y] << " "
                  << procs[Z] << " " << procs[T] << std::endl;
        std::cout << "Number of components " << ncomponents << std::endl;
        std::cout << "Max power " << max_power << std::endl;
        std::cout << "Repetitions " << nrep << std::endl;
    }

    {
        std::vector<Context> ctx;
        for (int i = 0; i < ncomponents; ++i) ctx.push_back(createCpuContext());
        std::vector<Cpu> xpus;
        for (const auto &i : ctx) xpus.push_back(i.toCpu(0));
#ifdef SUPERBBLAS_USE_FLOAT16
        test<std::complex<_Float16>, Cpu>(dim, procs, rank, nprocs, max_power, nrep, ctx, xpus);
#endif
        test<std::complex<float>, Cpu>(dim, procs, rank, nprocs, max_power, nrep, ctx, xpus);
        test<std::complex<double>, Cpu>(dim, procs, rank, nprocs, max_power, nrep, ctx, xpus);
    }
    clearCaches();
    checkForMemoryLeaks(std::cout);
#ifdef SUPERBBLAS_USE_GPU
    {
        std::vector<Context> ctx;
        for (int i = 0; i < ncomponents; ++i)
            ctx.push_back(createGpuContext((rank * ncomponents + i) % getGpuDevicesCount()));
        std::vector<Gpu> xpus;
        for (const auto &i : ctx) xpus.push_back(i.toGpu(0));
        test<std::complex<float>, Gpu>(dim, procs, rank, nprocs, max_power, nrep, ctx, xpus);
        test<std::complex<double>, Gpu>(dim, procs, rank, nprocs, max_power, nrep, ctx, xpus);
    }
    clearCaches();
    clearHandles();
    checkForMemoryLeaks(std::cout);
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
