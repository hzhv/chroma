#include "superbblas.h"
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace superbblas;

template <std::size_t N, typename T> using Operator = std::tuple<Coor<N>, Order<N>, std::vector<T>>;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <typename T> T conj(T t) { return std::conj(t); }
template <> float conj<float>(float t) { return t; }
template <> double conj<double>(double t) { return t; }

template <typename T, typename T::value_type = 0>
T make_complex(typename T::value_type a, typename T::value_type b) {
    return T{a, b};
}
template <typename T> T make_complex(T a, T) { return a; }

template <std::size_t N> Order<N + 1> toStr(Order<N> o) {
    Order<N + 1> r{};
    std::copy(o.begin(), o.end(), r.begin());
    return r;
}

static std::size_t progress = 0;
static char progress_mark = 0;
constexpr std::size_t no_test = std::numeric_limits<std::size_t>::max();
static std::size_t test_number = 0;
static std::size_t do_test = no_test;

void initialize_test() {
    std::srand(0);
    progress = 0;
    progress_mark = 0;
    test_number = 0;
}

template <std::size_t NA, std::size_t NB, std::size_t NC, typename T>
Operator<NA + NB + NC, T> generate_tensor(char a, char b, char c, const std::map<char, int> &dims) {
    // Build the operator with A,B,C
    constexpr std::size_t N = NA + NB + NC;
    Coor<N> dim{};
    for (std::size_t i = 0; i < NA; ++i) dim[i] = dims.at(a);
    for (std::size_t i = 0; i < NB; ++i) dim[i + NA] = dims.at(b);
    for (std::size_t i = 0; i < NC; ++i) dim[i + NA + NB] = dims.at(c);
    std::size_t vol = detail::volume(dim);
    std::vector<T> v(vol);
    for (std::size_t i = 0; i < vol; ++i) v[i] = make_complex<T>(i, i);
    Order<N> o{};
    for (std::size_t i = 0; i < NA; ++i) o[i] = a + i;
    for (std::size_t i = 0; i < NB; ++i) o[i + NA] = b + i;
    for (std::size_t i = 0; i < NC; ++i) o[i + NA + NB] = c + i;
    return {dim, o, v};
}

const char sT = 'A', sA = sT + 8, sB = sA + 8, sC = sB + 8;
enum distribution { OnMaster, OnEveryone, OnEveryoneReplicated };

template <std::size_t N0, std::size_t N1, std::size_t N2, typename T, typename XPU>
void test_contraction(const T &alpha, Operator<N0, T> op0, Operator<N1, T> op1, const T &beta,
                      Operator<N2, T> op2, bool conj0, bool conj1, char dist_dir,
                      const std::vector<Context> &ctx, const std::vector<XPU> &xpu,
                      unsigned int dist_index) {
    std::array<distribution, 3> d{OnMaster, OnEveryone, OnEveryoneReplicated};
    std::array<std::array<distribution, 2>, 6> dd;
    for (unsigned int i = 0, k = 0; i < d.size(); ++i)
        for (unsigned int j = i; j < d.size(); ++j)
            dd[k++] = std::array<distribution, 2>{d[i], d[j]};
    test_contraction(alpha, op0, dd[dist_index][0], op1, dd[dist_index][1], beta, op2,
                     dd[dist_index][0], conj0, conj1, dist_dir, ctx, xpu);
}

template <std::size_t N> Coor<N> random_from() {
    Coor<N> r;
    for (std::size_t i = 0; i < N; ++i) r[i] = (std::rand() % 2 == 0 ? 0 : 1);
    return r;
}

/// Return the data pointers to a bunch of `vector`
template <typename T, typename XPU>
std::vector<T *> get_ptrs(const std::vector<superbblas::detail::vector<T, XPU>> &v) {
    std::vector<T *> ptrs;
    ptrs.reserve(v.size());
    for (const auto &i : v) ptrs.push_back(i.data());
    return ptrs;
}

/// Store a bunch of vectors and its data pointers for convenience
template <typename T, typename XPU> struct vectors {
    vectors(const std::vector<superbblas::detail::vector<T, XPU>> &v) : v(v), ptrs(get_ptrs(v)) {}
    T **data() const { return (T **)ptrs.data(); }
    const T **const_data() const { return (const T **)ptrs.data(); }
    const std::vector<superbblas::detail::vector<T, XPU>> &getVectors() const { return v; }

private:
    std::vector<superbblas::detail::vector<T, XPU>> v;
    std::vector<T *> ptrs;
};

/// Allocate tensor
template <typename T, std::size_t N, typename XPU>
vectors<T, XPU> create_tensor_data(const PartitionStored<N> &p, int rank,
                                   const std::vector<XPU> &xpu, bool init_to_ones = false) {

    std::vector<superbblas::detail::vector<T, XPU>> r;
    for (unsigned int component = 0; component < xpu.size(); ++component) {
        std::size_t vol = superbblas::detail::volume(p[rank * xpu.size() + component][1]);
        superbblas::detail::vector<T, XPU> v(vol, xpu[component]);
        r.push_back(v);
        if (init_to_ones) {
            std::vector<T> v_cpu(vol, T{1});
            detail::copy_n(v_cpu.data(), detail::Cpu{}, vol, v.data(), v.ctx());
        }
    }
    return vectors<T, XPU>(r);
}

template <std::size_t N>
PartitionStored<N> make_partinioning(const Order<N + 1> &o, const Coor<N> &dim, int nprocs,
                                     int ncomponents, distribution dist, char dist_dir) {
    switch (dist) {
    case OnMaster: {
        PartitionStored<N> r(nprocs * ncomponents);
        r[0][1] = dim;
        return r;
    }
    case OnEveryoneReplicated: {
        PartitionStored<N> r(nprocs * ncomponents, {Coor<N>{{}}, dim});
        return r;
    }
    case OnEveryone: {
        char dist_labels[2] = {dist_dir, 0};
        Coor<N> procs;
        for (std::size_t i = 0; i < N; ++i) procs[i] = (o[i] == dist_dir ? nprocs : 1);
        return basic_partitioning(o.data(), dim, procs, &dist_labels[0], nprocs, ncomponents);
    }
    default: throw std::runtime_error("wtf");
    }
}

template <std::size_t N0, std::size_t N1, std::size_t N2, typename T, typename XPU>
void test_contraction(const T &alpha, Operator<N0, T> op0, distribution d0, Operator<N1, T> op1,
                      distribution d1, const T &beta, Operator<N2, T> op2, distribution d2,
                      bool conj0, bool conj1, char dist_dir, const std::vector<Context> &ctx,
                      const std::vector<XPU> &xpu) {

    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nprocs = 1;
    rank = 0;
#endif

    const Coor<N0> from0 = random_from<N0>();
    const Coor<N1> from1 = random_from<N1>();
    const Coor<N2> from2 = random_from<N2>();
    const Coor<N0> size0 = std::get<0>(op0);
    const Coor<N1> size1 = std::get<0>(op1);
    const Coor<N2> size2 = std::get<0>(op2);
    using namespace superbblas::detail;
    const Coor<N0> dim0 = from0 + size0;
    const Coor<N1> dim1 = from1 + size1;
    const Coor<N2> dim2 = from2 + size2;
    const auto o0 = toStr(std::get<1>(op0));
    const auto o1 = toStr(std::get<1>(op1));
    const auto o2 = toStr(std::get<1>(op2));
    const std::vector<T> v0_ = std::get<2>(op0);
    const std::vector<T> v1_ = std::get<2>(op1);
    const std::vector<T> v2_ = std::get<2>(op2);

    // Skip if this is not the test number to test
    if (do_test != no_test) {
        if (do_test != test_number) {
            test_number++;
            return;
        }
    }

    Context cpu = createCpuContext();

    // Distribute op0, op1, and a zeroed op2 along the `dist_dir` direction

    PartitionStored<N0> p0_(nprocs, {{{{}}, size0}}); // tensor replicated partitioning
    T const *ptrv0_ = v0_.data();
    PartitionStored<N0> p0 = make_partinioning(o0, dim0, nprocs, ctx.size(), d0, dist_dir);
    auto v0 = create_tensor_data<T>(p0, rank, xpu);
    copy(1.0, p0_.data(), 1, o0.data(), {{}}, size0, size0, (const T **)&ptrv0_, nullptr, &cpu,
         p0.data(), ctx.size(), o0.data(), from0, dim0, v0.data(), nullptr, ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    PartitionStored<N1> p1_(nprocs, {{{{}}, size1}}); // tensor replicated partitioning
    T const *ptrv1_ = v1_.data();
    PartitionStored<N1> p1 = make_partinioning(o1, dim1, nprocs, ctx.size(), d1, dist_dir);
    auto v1 = create_tensor_data<T>(p1, rank, xpu);
    copy(1.0, p1_.data(), 1, o1.data(), {{}}, size1, size1, (const T **)&ptrv1_, nullptr, &cpu,
         p1.data(), ctx.size(), o1.data(), from1, dim1, v1.data(), nullptr, ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    PartitionStored<N2> p2 = make_partinioning(o2, dim2, nprocs, ctx.size(), d2, dist_dir);
    auto v2 = create_tensor_data<T>(p2, rank, xpu, true /* init to ones */);

    // Contract the distributed matrices

    contraction(alpha, p0.data(), from0, size0, dim0, ctx.size(), o0.data(), conj0, v0.const_data(),
                ctx.data(), p1.data(), from1, size1, dim1, ctx.size(), o1.data(), conj1,
                v1.const_data(), ctx.data(), beta, p2.data(), from2, size2, dim2, ctx.size(),
                o2.data(), v2.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                MPI_COMM_WORLD,
#endif
                SlowToFast);

    // Move the result to proc 0
    PartitionStored<N2> pr(nprocs, {{{{}}, {{}}}});
    pr[0][1] = size2; // tensor only supported on proc 0
    std::vector<T> vr(detail::volume(pr[rank][1]));
    T *ptrvr = vr.data();
    copy(1, p2.data(), ctx.size(), o2.data(), from2, size2, dim2, v2.const_data(), nullptr,
         ctx.data(), pr.data(), 1, o2.data(), {{}}, size2, &ptrvr, nullptr, &cpu,
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    // Test the resulting tensor

    int is_correct = 1;
    if (rank == 0) {
        double diff_fn = 0, fn = 0; // Frob-norm of the difference and the correct tensor
        for (std::size_t i = 0; i < v2_.size(); ++i)
            diff_fn += std::norm(beta * T{1} + alpha * v2_[i] - vr[i]), fn += std::norm(v2_[i]);
        diff_fn = std::sqrt(diff_fn);
        fn = std::sqrt(fn);
        if (diff_fn > fn * 1e-4) is_correct = 0;
        progress++;
        if (progress > 128333) {
            std::cout << std::string(1, '0' + progress_mark);
            std::cout.flush();
            progress = 0;
            progress_mark++;
        }
    }
#ifdef SUPERBBLAS_USE_MPI
    MPI_Bcast(&is_correct, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    if (!is_correct) {
        // NOTE: Put a breakpoint here to debug the cases producing wrong answers!
        contraction(alpha, p0.data(), from0, size0, dim0, ctx.size(), o0.data(), conj0,
                    v0.const_data(), ctx.data(), p1.data(), from1, size1, dim1, ctx.size(),
                    o1.data(), conj1, v1.const_data(), ctx.data(), beta, p2.data(), from2, size2,
                    dim2, ctx.size(), o2.data(), v2.data(), ctx.data(),
#ifdef SUPERBBLAS_USE_MPI
                    MPI_COMM_WORLD,
#endif
                    SlowToFast);
        std::stringstream ss;
        ss << "Result of contraction does not match with the correct answer for test "
           << test_number;
        throw std::runtime_error(ss.str());
    }

    test_number++;
}

template <std::size_t N0, std::size_t N1, std::size_t N2, typename T, typename XPU>
void test_contraction(const T &alpha, Operator<N0, T> p0, Operator<N1, T> p1, const T &beta,
                      Operator<N2, T> p2, const std::vector<Context> &ctx,
                      const std::vector<XPU> &xpu, unsigned int dist_index) {
    // Compute correct result of the contraction of p0 and p1
    const Coor<N0> dim0 = std::get<0>(p0);
    const Coor<N1> dim1 = std::get<0>(p1);
    const Coor<N2> dim2 = std::get<0>(p2);
    const Order<N0> o0 = std::get<1>(p0);
    const Order<N1> o1 = std::get<1>(p1);
    const Order<N2> o2 = std::get<1>(p2);
    const std::vector<T> v0 = std::get<2>(p0);
    const std::vector<T> v1 = std::get<2>(p1);
    std::vector<T> r0(detail::volume(dim2)); // p0 not conj, and p1 not conj
    std::vector<T> r1(detail::volume(dim2)); // p0 conj, and p1 not conj
    std::vector<T> r2(detail::volume(dim2)); // p0 not conj, and p1 conj
    std::vector<T> r3(detail::volume(dim2)); // p0 conj, and p1 conj
    Coor<N0> strides0 = detail::get_strides<IndexType>(dim0, SlowToFast);
    Coor<N1> strides1 = detail::get_strides<IndexType>(dim1, SlowToFast);
    Coor<N2> strides2 = detail::get_strides<IndexType>(dim2, SlowToFast);
    for (std::size_t i = 0, m = detail::volume(dim0); i < m; ++i) {
        std::vector<int> dim(128, -1);
        Coor<N0> c0 = detail::index2coor((IndexType)i, dim0, strides0);
        for (std::size_t d = 0; d < N0; ++d) dim[o0[d]] = c0[d];
        for (std::size_t j = 0, n = detail::volume(dim1); j < n; ++j) {
            std::vector<int> dim_ = dim;
            Coor<N1> c1 = detail::index2coor((IndexType)j, dim1, strides1);
            bool get_out = false;
            for (std::size_t d = 0; d < N1; ++d) {
                if (dim_[o1[d]] == -1)
                    dim_[o1[d]] = c1[d];
                else if (dim_[o1[d]] != c1[d]) {
                    get_out = true;
                    break;
                }
            }
            if (get_out) continue;
            Coor<N2> c2{};
            for (std::size_t d = 0; d < N2; ++d) c2[d] = dim_[o2[d]];
            IndexType k = detail::coor2index(c2, dim2, strides2);
            r0[k] += v0[i] * v1[j];
            r1[k] += conj(v0[i]) * v1[j];
            r2[k] += v0[i] * conj(v1[j]);
            r3[k] += conj(v0[i]) * conj(v1[j]);
        }
    }

    std::vector<char> labels({sT, sA, sB, sC});
    for (char c : labels) {
        // Test first operator no conj and second operator no conj
        std::get<2>(p2) = r0;
        test_contraction(alpha, p0, p1, beta, p2, false, false, c, ctx, xpu, dist_index);
        // Test first operator conj and second operator no conj
        std::get<2>(p2) = r1;
        test_contraction(alpha, p0, p1, beta, p2, true, false, c, ctx, xpu, dist_index);
        // Test first operator no conj and second operator conj
        std::get<2>(p2) = r2;
        test_contraction(alpha, p0, p1, beta, p2, false, true, c, ctx, xpu, dist_index);
        // Test first operator conj and second operator conj
        std::get<2>(p2) = r3;
        test_contraction(alpha, p0, p1, beta, p2, true, true, c, ctx, xpu, dist_index);
    }
}

template <typename T> struct normal_value {
    static constexpr T value = T{-1};
};
template <typename T> struct normal_value<std::complex<T>> {
    static constexpr std::complex<T> value = std::complex<T>{-1, 2};
};

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T, typename XPU>
void test_third_operator(const T &alpha, Operator<NT + NA + NB, T> p0, Operator<NT + NA + NC, T> p1,
                         const std::map<char, int> &dims, const std::vector<Context> &ctx,
                         const std::vector<XPU> &xpu, unsigned int dist_index) {
    const auto normv = normal_value<T>::value;
    test_contraction(alpha, p0, p1, T{0}, generate_tensor<NT, NB, NC, T>(sT, sB, sC, dims), ctx,
                     xpu, dist_index);
    test_contraction(alpha, p0, p1, T{0}, generate_tensor<NT, NC, NB, T>(sT, sC, sB, dims), ctx,
                     xpu, dist_index);
    test_contraction(alpha, p0, p1, T{1}, generate_tensor<NB, NC, NT, T>(sB, sC, sT, dims), ctx,
                     xpu, dist_index);
    test_contraction(alpha, p0, p1, T{1}, generate_tensor<NB, NT, NC, T>(sB, sT, sC, dims), ctx,
                     xpu, dist_index);
    test_contraction(alpha, p0, p1, normv, generate_tensor<NC, NB, NT, T>(sC, sB, sT, dims), ctx,
                     xpu, dist_index);
    test_contraction(alpha, p0, p1, normv, generate_tensor<NC, NT, NB, T>(sC, sT, sB, dims), ctx,
                     xpu, dist_index);
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T, typename XPU>
void test_second_operator(Operator<NT + NA + NB, T> p0, const std::map<char, int> &dims,
                          const std::vector<Context> &ctx, const std::vector<XPU> &xpu,
                          unsigned int dist_index) {
    const auto normv = normal_value<T>::value;
    test_third_operator<NT, NA, NB, NC, T>(
        T{0}, p0, generate_tensor<NT, NA, NC, T>(sT, sA, sC, dims), dims, ctx, xpu, dist_index);
    test_third_operator<NT, NA, NB, NC, T>(
        T{0}, p0, generate_tensor<NT, NC, NA, T>(sT, sC, sA, dims), dims, ctx, xpu, dist_index);
    test_third_operator<NT, NA, NB, NC, T>(
        T{1}, p0, generate_tensor<NA, NC, NT, T>(sA, sC, sT, dims), dims, ctx, xpu, dist_index);
    test_third_operator<NT, NA, NB, NC, T>(
        T{1}, p0, generate_tensor<NA, NT, NC, T>(sA, sT, sC, dims), dims, ctx, xpu, dist_index);
    test_third_operator<NT, NA, NB, NC, T>(
        normv, p0, generate_tensor<NC, NA, NT, T>(sC, sA, sT, dims), dims, ctx, xpu, dist_index);
    test_third_operator<NT, NA, NB, NC, T>(
        normv, p0, generate_tensor<NC, NT, NA, T>(sC, sT, sA, dims), dims, ctx, xpu, dist_index);
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T, typename XPU>
void test_first_operator(const std::map<char, int> &dims, const std::vector<Context> &ctx,
                         const std::vector<XPU> &xpu) {
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NT, NA, NB, T>(sT, sA, sB, dims), dims,
                                            ctx, xpu, 0);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NT, NB, NA, T>(sT, sB, sA, dims), dims,
                                            ctx, xpu, 1);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NA, NB, NT, T>(sA, sB, sT, dims), dims,
                                            ctx, xpu, 2);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NA, NT, NB, T>(sA, sT, sB, dims), dims,
                                            ctx, xpu, 3);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NB, NA, NT, T>(sB, sA, sT, dims), dims,
                                            ctx, xpu, 4);
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NB, NT, NA, T>(sB, sT, sA, dims), dims,
                                            ctx, xpu, 5);
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T, typename XPU,
          typename std::enable_if<!(NT + NA + NB == 0 || NT + NA + NC == 0 || NT + NC + NB == 0),
                                  bool>::type = true>
void test_sizes(const std::vector<Context> &ctx, const std::vector<XPU> &xpu) {
    if (NT + NA + NB == 0 || NT + NA + NC == 0 || NT + NC + NB == 0) return;
    for (int dimT = 1; dimT < 3; ++dimT)
        for (int dimA = 1; dimA < 3; ++dimA)
            for (int dimB = 1; dimB < 3; ++dimB)
                for (int dimC = 1; dimC < 3; ++dimC)
                    test_first_operator<NT, NA, NB, NC, T>(
                        {{sT, dimT}, {sA, dimA}, {sB, dimB}, {sC, dimC}}, ctx, xpu);
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T, typename XPU,
          typename std::enable_if<(NT + NA + NB == 0 || NT + NA + NC == 0 || NT + NC + NB == 0),
                                  bool>::type = true>
void test_sizes(const std::vector<Context> &, const std::vector<XPU> &) {}

template <std::size_t NT, std::size_t NA, std::size_t NB, typename T, typename XPU>
void test_for_C(const std::vector<Context> &ctx, const std::vector<XPU> &xpu) {
    test_sizes<NT, NA, NB, 0, T>(ctx, xpu);
    test_sizes<NT, NA, NB, 1, T>(ctx, xpu);
}

template <std::size_t NT, std::size_t NA, typename T, typename XPU>
void test_for_B(const std::vector<Context> &ctx, const std::vector<XPU> &xpu) {
    test_for_C<NT, NA, 0, T>(ctx, xpu);
    test_for_C<NT, NA, 1, T>(ctx, xpu);
}

template <std::size_t NT, typename T, typename XPU>
void test_for_A(const std::vector<Context> &ctx, const std::vector<XPU> &xpu) {
    test_for_B<NT, 0, T>(ctx, xpu);
    test_for_B<NT, 1, T>(ctx, xpu);
}

template <typename T, typename XPU>
void test(const std::vector<Context> &ctx, const std::vector<XPU> &xpu) {
    test_for_A<0, T>(ctx, xpu);
    test_for_A<1, T>(ctx, xpu);
}

int main(int argc, char **argv) {
    int rank = 0;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    (void)argc;
    (void)argv;
#endif

    int ncomponents = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--test=", argv[i], 7) == 0) {
            if (sscanf(argv[i] + 7, "%ld", &do_test) != 1) {
                std::cerr << "--test= should follow 1 numbers, for instance --test=42" << std::endl;
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
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0] << " [--test=#test] [--help]"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // Set the default number of components
    ncomponents = ncomponents == 0 ? 1 : ncomponents;

    {
        std::vector<Context> ctx;
        for (int i = 0; i < ncomponents; ++i) ctx.push_back(createCpuContext());
        std::vector<superbblas::detail::Cpu> xpus;
        for (const auto &i : ctx) xpus.push_back(i.toCpu(0));
        initialize_test();
        test<double>(ctx, xpus);
        test<std::complex<double>>(ctx, xpus);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        std::vector<Context> ctx;
        for (int i = 0; i < ncomponents; ++i)
            ctx.push_back(createGpuContext((rank * ncomponents + i) % getGpuDevicesCount()));
        std::vector<superbblas::detail::Gpu> xpus;
        for (const auto &i : ctx) xpus.push_back(i.toGpu(0));
        initialize_test();
        test<double>(ctx, xpus);
        test<std::complex<double>>(ctx, xpus);
        clearCaches();
        clearHandles();
        checkForMemoryLeaks(std::cout);
    }
#endif

    if (rank == 0) std::cout << " Everything went ok!" << std::endl;

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
