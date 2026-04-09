#include "superbblas.h"
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace superbblas;
using namespace superbblas::detail;

template <typename T, typename XPU> struct gen_dummy_vector {
    static vector<T, XPU> get(std::size_t size, XPU cuda) {
        vector<T, Cpu> v = gen_dummy_vector<T, Cpu>::get(size, Cpu{});
        vector<T, XPU> r(size, cuda);
        copy_n<IndexType>(T{1}, v.data(), Cpu{}, size, r.data(), cuda, EWOp::Copy{});
        return r;
    }

    static vector<T, XPU> get(std::size_t size, XPU cuda, unsigned int mod) {
        vector<T, Cpu> v = gen_dummy_vector<T, Cpu>::get(size, Cpu{}, mod);
        vector<T, XPU> r(size, cuda);
        copy_n<IndexType>(T{1}, v.data(), Cpu{}, size, r.data(), cuda, EWOp::Copy{});
        return r;
    }
};

template <typename T> struct gen_dummy_vector<T, Cpu> {
    static vector<T, Cpu> get(std::size_t size, Cpu cpu) {
        vector<T, Cpu> v(size, cpu);
        for (unsigned int i = 0; i < size; i++) v[i] = i;
        return v;
    }

    static vector<T, Cpu> get(std::size_t size, Cpu cpu, unsigned int mod) {
        vector<T, Cpu> v(size, cpu);
        for (unsigned int i = 0; i < size; i++) v[i] = i % mod;
        return v;
    }
};

template <typename XPU>
Indices<XPU> gen_dummy_perm(std::size_t size, std::size_t max_size, XPU xpu,
                            std::size_t blocking = 1) {
    if (size >= (std::size_t)std::numeric_limits<int>::max() ||
        max_size >= (std::size_t)std::numeric_limits<int>::max())
        throw std::runtime_error("Too many elements for the permutation");
    Indices<Cpu> v(size, Cpu{});
    for (std::size_t i = 0; i < size; i++) v[i] = (i % max_size) * blocking;
    //for (std::size_t i = 0; i < size; i++) v[i] = (i * 3) % max_size;
    return makeSure(v, xpu);
}

template <typename T> struct Epsilon {
    static double get(void) { return std::numeric_limits<T>::epsilon(); }
};
template <typename T> struct Epsilon<std::complex<T>> {
    static double get(void) { return std::numeric_limits<T>::epsilon(); }
};

template <typename T, typename XPU0, typename XPU1>
void check_are_equal(vector<T, XPU0> u_, vector<T, XPU1> v_) {
    if (u_.size() != v_.size()) throw std::runtime_error("Input vectors have different size!");
    vector<T, Cpu> u = superbblas::detail::makeSure(u_, Cpu{0});
    vector<T, Cpu> v = superbblas::detail::makeSure(v_, Cpu{0});
    const double bound = Epsilon<T>::get() * 100;
    for (unsigned int i = 0; i < u.size(); i++) {
        double diff = std::abs(u[i] - v[i]) / std::max(std::abs(u[i]), std::abs(v[i]));
        if (diff > bound) {
            std::stringstream ss;
            ss << "1-norm of the difference between the input vectors of size " << v.size()
               << " is " << diff << ", which is larger than the bound at " << bound
               << " epsilon=" << Epsilon<T>::get();
            throw std::runtime_error(ss.str());
        }
    }
}

template <typename T> struct toStr;

template <> struct toStr<Cpu> {
    static constexpr const char *get = "cpu ";
};
#ifdef SUPERBBLAS_USE_CUDA
template <> struct toStr<Cuda> {
    static constexpr const char *get = "cuda";
};
#elif defined(SUPERBBLAS_USE_HIP)
template <> struct toStr<Hip> {
    static constexpr const char *get = "hip ";
};
#endif
template <> struct toStr<EWOp::Add> {
    static constexpr const char *get = "add ";
};
template <> struct toStr<EWOp::Copy> {
    static constexpr const char *get = "copy";
};
template <> struct toStr<float> {
    static constexpr const char *get = "float  ";
};
template <> struct toStr<double> {
    static constexpr const char *get = "double ";
};
template <> struct toStr<std::complex<float>> {
    static constexpr const char *get = "cfloat ";
};
template <> struct toStr<std::complex<double>> {
    static constexpr const char *get = "cdouble";
};

template <typename T, typename XPU, typename EWOP>
void test_copy(std::size_t size, XPU xpu, EWOP, T a, unsigned int nrep = 10) {

    // Normalize size
    size /= (sizeof(T) / sizeof(float));
    size = size / 4 * 4;

    // Do once the operation for testing correctness
    vector<T, Cpu> t0 = gen_dummy_vector<T, Cpu>::get(size, Cpu{});
    vector<T, XPU> t0_xpu = gen_dummy_vector<T, XPU>::get(size, xpu);
    vector<T, XPU> t1_xpu = gen_dummy_vector<T, XPU>::get(size, xpu);
    // t0_xpu = (or +=) a * t0
    copy_n<IndexType>(a, t0.data(), Cpu{}, size, t0_xpu.data(), xpu, EWOP{});
    // t1_xpu = (or +=) a * t0_xpu
    copy_n<IndexType>(a, t0_xpu.data(), xpu, size, t1_xpu.data(), xpu, EWOP{});
    vector<T, Cpu> t1(size, Cpu{});
    // t1_xpu = a * t1_xpu
    copy_n<IndexType>(a, t1_xpu.data(), xpu, size, t1.data(), Cpu{}, EWOp::Copy{});

    vector<T, Cpu> r = gen_dummy_vector<T, Cpu>::get(size, Cpu{});
    copy_n<IndexType>(a, t0.data(), Cpu{}, size, r.data(), Cpu{}, EWOP{});
    copy_n<IndexType>(a, r.data(), Cpu{}, size, t0.data(), Cpu{}, EWOP{});
    copy_n<IndexType>(a, t0.data(), Cpu{}, size, r.data(), Cpu{}, EWOp::Copy{});
    check_are_equal<T>(t1, r);

    // Test with indices
    Indices<Cpu> i0 = gen_dummy_perm(size / 2, size, Cpu{});
    Indices<XPU> i0_xpu = gen_dummy_perm(size / 2, size, xpu);
    zero_n<T>(t0_xpu.data(), size, xpu);
    zero_n<T>(t1_xpu.data(), size, xpu);
    zero_n<T>(t1.data(), size, Cpu{});
    vector<T, Cpu> r0(size, Cpu{}), r1(size, Cpu{});
    zero_n<T>(r.data(), size, Cpu{});
    zero_n<T>(r0.data(), size, Cpu{});
    zero_n<T>(r1.data(), size, Cpu{});
    // t0_xpu[i] = t0[i0[i]]
    copy_n<IndexType>(T{1}, t0.data(), Cpu{}, i0.begin(), Cpu{}, i0.size(), t0_xpu.data(), xpu,
                      nullptr, xpu, EWOp::Copy{});
    copy_n<IndexType>(T{1}, t0.data(), Cpu{}, i0.begin(), Cpu{}, i0.size(), r0.data(), Cpu{},
                      nullptr, Cpu{}, EWOp::Copy{});
    check_are_equal<T>(t0_xpu, r0);
    // t0_xpu[i0_xpu[i]] = t0[i]
    copy_n<IndexType>(T{1}, t0.data(), Cpu{}, nullptr, Cpu{}, i0_xpu.size(), t0_xpu.data(), xpu,
                      i0_xpu.begin(), xpu, EWOp::Copy{});
    copy_n<IndexType>(T{1}, t0.data(), Cpu{}, nullptr, Cpu{}, i0.size(), r0.data(), Cpu{},
                      i0.begin(), Cpu{}, EWOp::Copy{});
    check_are_equal<T>(t0_xpu, r0);
    // t1_xpu[i0_xpu[i]] (+)= t0_xpu[i0_xpu[i]]
    copy_n<IndexType>(T{1}, t0_xpu.data(), xpu, i0_xpu.begin(), xpu, size / 4, t1_xpu.data(), xpu,
                      i0_xpu.begin() + size / 4, xpu, EWOP{});
    copy_n<IndexType>(T{1}, r0.data(), Cpu{}, i0.begin(), Cpu{}, size / 4, r1.data(), Cpu{},
                      i0.begin() + size / 4, Cpu{}, EWOP{});
    check_are_equal<T>(t1_xpu, r1);
    // t0[i] = t0_xpu[i0[i]]
    copy_n<IndexType>(T{1}, t1_xpu.data(), xpu, i0_xpu.begin(), xpu, size / 2, t1.data(), Cpu{},
                      nullptr, Cpu{}, EWOp::Copy{});
    copy_n<IndexType>(T{1}, r1.data(), Cpu{}, i0.begin(), Cpu{}, size / 2, r.data(), Cpu{}, nullptr,
                      Cpu{}, EWOp::Copy{});
    check_are_equal<T>(t1, r);

    // Test performance
    auto t0_host = makeSure(t0, xpu.toCpuPinned());
    sync(t0_host.ctx());

    double t;
    copy_n<IndexType, T>(a, t0.data(), Cpu{}, size, t0_xpu.data(), xpu, EWOP{});
    sync(xpu);
    t = w_time();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(a, t0.data(), Cpu{}, size, t0_xpu.data(), xpu, EWOP{});
    }
    sync(xpu);
    double t_cpu_xpu = (w_time() - t) / nrep;

    copy_n<IndexType, T>(a, t0_host.data(), t0_host.ctx(), size, t0_xpu.data(), xpu, EWOP{});
    sync(xpu);
    t = w_time();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(a, t0_host.data(), t0_host.ctx(), size, t0_xpu.data(), xpu, EWOP{});
    }
    sync(xpu);
    double t_cpup_xpu = (w_time() - t) / nrep;

    copy_n<IndexType, T>(a, t0_xpu.data(), xpu, size, t1.data(), Cpu{}, EWOP{});
    t = w_time();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(a, t0_xpu.data(), xpu, size, t1.data(), Cpu{}, EWOP{});
    }
    double t_xpu_cpu = (w_time() - t) / nrep;

    copy_n<IndexType, T>(a, t0_xpu.data(), xpu, size, t0_host.data(), t0_host.ctx(), EWOP{});
    sync(t0_host.ctx());
    t = w_time();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(a, t0_xpu.data(), xpu, size, t0_host.data(), t0_host.ctx(), EWOP{});
    }
    sync(t0_host.ctx());
    double t_xpu_cpup = (w_time() - t) / nrep;

    copy_n<IndexType, T>(a, t0_xpu.data(), xpu, size, t1_xpu.data(), xpu, EWOP{});
    sync(xpu);
    t = w_time();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType, T>(a, t0_xpu.data(), xpu, size, t1_xpu.data(), xpu, EWOP{});
    }
    sync(xpu);
    double t_xpu_xpu = (w_time() - t) / nrep;

    Indices<XPU> p = gen_dummy_perm(size, size, xpu);
    copy_n<IndexType>(a, t0_xpu.data(), xpu, p.begin(), xpu, size, t1_xpu.data(), xpu, p.begin(),
                      xpu, EWOP{});
    sync(xpu);
    t = w_time();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        copy_n<IndexType>(a, t0_xpu.data(), xpu, p.begin(), xpu, size, t1_xpu.data(), xpu,
                          p.begin(), xpu, EWOP{});
    }
    sync(xpu);
    double tp_xpu_xpu = (w_time() - t) / nrep;

    std::string var = (a == T{1} ? "     " : (a == T{0} ? "/zero" : "/mult"));
    const char *sep = "    ";
    std::cout << toStr<T>::get << " in " << toStr<EWOP>::get << var << " ("
              << sizeof(T) * size / 1024. / 1024 << " MiB)" << sep

              << toStr<Cpu>::get << " -> " << toStr<XPU>::get << " : "
              << sizeof(T) * size / t_cpu_xpu / 1024 / 1024 / 1024 << " GiB/s" << sep;

    if (deviceId(xpu) >= 0) {
        std::cout << toStr<Cpu>::get << "p-> " << toStr<XPU>::get << " : "
                  << sizeof(T) * size / t_cpup_xpu / 1024 / 1024 / 1024 << " GiB/s" << sep

                  << toStr<XPU>::get << " -> " << toStr<Cpu>::get << " : "
                  << sizeof(T) * size / t_xpu_cpu / 1024 / 1024 / 1024 << " GiB/s" << sep

                  << toStr<XPU>::get << " -> " << toStr<Cpu>::get
                  << "p: " << sizeof(T) * size / t_xpu_cpup / 1024 / 1024 / 1024 << " GiB/s" << sep

                  << toStr<XPU>::get << " -> " << toStr<XPU>::get << " : "
                  << sizeof(T) * size / t_xpu_xpu / 1024 / 1024 / 1024 << " GiB/s" << sep;
    }

    std::cout << toStr<XPU>::get << "[i] -> " << toStr<XPU>::get
              << "[i] : " << sizeof(T) * size / tp_xpu_xpu / 1024 / 1024 / 1024 << " GiB/s"
              << std::endl;
}

template <typename T, typename XPU, typename EWOP>
void test_copy(std::size_t size, XPU xpu, EWOP, unsigned int nrep = 10) {
    test_copy<T>(size, xpu, EWOP{}, T{1}, nrep);
    test_copy<T>(size, xpu, EWOP{}, T{2}, nrep);
    test_copy<T>(size, xpu, EWOP{}, T{0}, nrep);
}

template <typename T, typename XPU, typename EWOP>
void test_copy_blocking(std::size_t size, XPU xpu, EWOP, T a, unsigned int nrep = 10) {

    // Normalize size
    size /= (sizeof(T) / sizeof(float));

    // Test performance
    vector<T, Cpu> t0 = gen_dummy_vector<T, Cpu>::get(size, Cpu{});
    vector<T, Cpu> t1 = gen_dummy_vector<T, Cpu>::get(size, Cpu{});
    vector<T, XPU> t0_xpu = gen_dummy_vector<T, XPU>::get(size, xpu);
    vector<T, XPU> t1_xpu = gen_dummy_vector<T, XPU>::get(size, xpu);
    std::vector<int> blockings{1, 2, 3, 4, 8, 12, 16, 24, 32};
    for (int blocking : blockings) {
        Indices<Cpu> i0 = gen_dummy_perm(size / blocking, size / blocking, Cpu{}, blocking);
        Indices<Cpu> i1 = gen_dummy_perm(size / blocking, size / blocking, Cpu{}, blocking);
        Indices<XPU> i0_xpu = gen_dummy_perm(size / blocking, size / blocking, xpu, blocking);
        Indices<XPU> i1_xpu = gen_dummy_perm(size / blocking, size / blocking, xpu, blocking);
        auto t0_host = makeSure(t0, xpu.toCpuPinned());
        auto i0_host = makeSure(i0, xpu.toCpuPinned());
        sync(i0_host.ctx());
        sync(xpu);

        double t;
        copy_n_blocking<IndexType>(a, t0.data(), Cpu{}, blocking, nullptr, Cpu{}, size / blocking,
                                   t1_xpu.data(), xpu, nullptr, xpu, EWOP{});
        sync(xpu);
        t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            copy_n_blocking<IndexType>(a, t0.data(), Cpu{}, blocking, nullptr, Cpu{},
                                       size / blocking, t1_xpu.data(), xpu, nullptr, xpu, EWOP{});
        }
        sync(xpu);
        double t_cpu_xpu_nb = (w_time() - t) / nrep;

        copy_n_blocking<IndexType>(a, t0.data(), Cpu{}, blocking, i0.data(), Cpu{}, size / blocking,
                                   t1_xpu.data(), xpu, i1_xpu.data(), xpu, EWOP{});
        sync(xpu);
        t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            copy_n_blocking<IndexType>(a, t0.data(), Cpu{}, blocking, i0.data(), Cpu{},
                                       size / blocking, t1_xpu.data(), xpu, i1_xpu.data(), xpu,
                                       EWOP{});
        }
        sync(xpu);
        double t_cpu_xpu = (w_time() - t) / nrep;

        copy_n_blocking<IndexType>(a, t0_host.data(), t0_host.ctx(), blocking, i0_host.data(),
                                   i0_host.ctx(), size / blocking, t1_xpu.data(), xpu,
                                   i1_xpu.data(), xpu, EWOP{});
        sync(xpu);
        t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            copy_n_blocking<IndexType>(a, t0_host.data(), t0_host.ctx(), blocking, i0_host.data(),
                                       i0_host.ctx(), size / blocking, t1_xpu.data(), xpu,
                                       i1_xpu.data(), xpu, EWOP{});
        }
        sync(xpu);
        double t_cpup_xpu = (w_time() - t) / nrep;

        copy_n_blocking<IndexType>(a, t0_xpu.data(), xpu, blocking, i0_xpu.data(), xpu,
                                   size / blocking, t1.data(), Cpu{}, i1.data(), Cpu{}, EWOP{});
        t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            copy_n_blocking<IndexType>(a, t0_xpu.data(), xpu, blocking, i0_xpu.data(), xpu,
                                       size / blocking, t1.data(), Cpu{}, i1.data(), Cpu{}, EWOP{});
        }
        double t_xpu_cpu = (w_time() - t) / nrep;

        copy_n_blocking<IndexType>(a, t0_xpu.data(), xpu, blocking, i0_xpu.data(), xpu,
                                   size / blocking, t0_host.data(), t0_host.ctx(), i0_host.data(),
                                   i0_host.ctx(), EWOP{});
        sync(t0_host.ctx());
        t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            copy_n_blocking<IndexType>(a, t0_xpu.data(), xpu, blocking, i0_xpu.data(), xpu,
                                       size / blocking, t0_host.data(), t0_host.ctx(),
                                       i0_host.data(), i0_host.ctx(), EWOP{});
        }
        sync(t0_host.ctx());
        double t_xpu_cpup = (w_time() - t) / nrep;

        copy_n_blocking<IndexType>(a, t0_xpu.data(), xpu, blocking, i0_xpu.data(), xpu,
                                   size / blocking, t1_xpu.data(), xpu, i1_xpu.data(), xpu, EWOP{});
        sync(xpu);
        t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            copy_n_blocking<IndexType>(a, t0_xpu.data(), xpu, blocking, i0_xpu.data(), xpu,
                                       size / blocking, t1_xpu.data(), xpu, i1_xpu.data(), xpu,
                                       EWOP{});
        }
        sync(xpu);
        double t_xpu_xpu = (w_time() - t) / nrep;

        std::string var = (a == T{1} ? "     " : (a == T{0} ? "/zero" : "/mult"));
        const char *sep = "    ";
        std::cout << "blocking: " << blocking << "\t"

                  << toStr<T>::get << " in " << toStr<EWOP>::get << var << " ("
                  << sizeof(T) * size / 1024. / 1024 << " MiB)" << sep

                  << toStr<Cpu>::get << " -> " << toStr<XPU>::get << " : "
                  << sizeof(T) * size / t_cpu_xpu_nb / 1024 / 1024 / 1024 << " GiB/s" << sep

                  << toStr<Cpu>::get << "[i] -> " << toStr<XPU>::get
                  << "[i] : " << sizeof(T) * size / t_cpu_xpu / 1024 / 1024 / 1024 << " GiB/s"
                  << sep;

        if (deviceId(xpu) >= 0) {
            std::cout << toStr<Cpu>::get << "[i]p-> " << toStr<XPU>::get
                      << "[i] : " << sizeof(T) * size / t_cpup_xpu / 1024 / 1024 / 1024 << " GiB/s"
                      << sep

                      << toStr<XPU>::get << "[i] -> " << toStr<Cpu>::get
                      << "[i] : " << sizeof(T) * size / t_xpu_cpu / 1024 / 1024 / 1024 << " GiB/s"
                      << sep

                      << toStr<XPU>::get << "[i] -> " << toStr<Cpu>::get
                      << "[i]p: " << sizeof(T) * size / t_xpu_cpup / 1024 / 1024 / 1024 << " GiB/s"
                      << sep

                      << toStr<XPU>::get << "[i] -> " << toStr<XPU>::get
                      << "[i] : " << sizeof(T) * size / t_xpu_xpu / 1024 / 1024 / 1024 << " GiB/s";
        }

        std::cout << std::endl;
    }
}

template <typename T, typename XPU, typename EWOP>
void test_copy_blocking(std::size_t size, XPU xpu, EWOP, unsigned int nrep = 10) {
    test_copy_blocking<T>(size, xpu, EWOP{}, T{1}, nrep);
    test_copy_blocking<T>(size, xpu, EWOP{}, T{2}, nrep);
    test_copy_blocking<T>(size, xpu, EWOP{}, T{0}, nrep);
}

template <typename T, typename XPU>
void test_select(std::size_t size, XPU xpu, unsigned int nrep = 10) {

    // Normalize size
    size /= (sizeof(T) / sizeof(float));
    size = size / 4 * 4;

    // Do once the operation for testing correctness
    vector<T, Cpu> t0 = gen_dummy_vector<T, Cpu>::get(size, Cpu{}, 2);
    Indices<Cpu> t1 = gen_dummy_perm(size, size, Cpu{});
    vector<T, XPU> t0_xpu = gen_dummy_vector<T, XPU>::get(size, xpu, 2);
    Indices<XPU> t1_xpu = gen_dummy_perm(size, size, xpu);
    auto t2 = superbblas::detail::select<IndexType, T>(t1, t0, 0, t1);
    auto t2_xpu = superbblas::detail::select<IndexType, T>(t1_xpu, t0_xpu, 0, t1_xpu);
    check_are_equal(t2, t2_xpu);

    sync(xpu);
    double t = w_time();
    for (unsigned int rep = 0; rep < nrep; ++rep) {
        superbblas::detail::select<IndexType, T>(t1_xpu, t0_xpu, 0, t1_xpu);
    }
    sync(xpu);
    double t_xpu_xpu = (w_time() - t) / nrep;

    const char *sep = "    ";
    std::cout << "select: " << toStr<T>::get << " for " << toStr<XPU>::get << " in "
              << " (" << sizeof(T) * size / 1024. / 1024 << " MiB) " << sep
              << (sizeof(T) + sizeof(IndexType) / 2) * size / t_xpu_xpu / 1024 / 1024 / 1024
              << " GiB/s" << std::endl;
}

int main(int argc, char **argv) {
    int size = 1000;
    int nrep = 10;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
#endif

    // Get options
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--size=", argv[i], 7) == 0) {
            if (sscanf(argv[i] + 7, "%d", &size) != 1) {
                std::cerr << "--size= should follow 1 numbers, for instance --size=10" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--rep=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d", &nrep) != 1) {
                std::cerr << "--rep= should follow 1 numbers, for instance --rep=10" << std::endl;
                return -1;
            }
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--size=number] [--rep=number] [--help]" << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }
    std::cout << "Maximum number of elements in a tested array: " << size << std::endl;
    std::cout << "Doing " << nrep << " repetitions" << std::endl;

    std::cout << std::endl;
    std::cout << "- Non-blocking:" << std::endl;
    {
        Context ctx = createCpuContext();
        test_copy<float, Cpu>(size, ctx.toCpu(0), EWOp::Copy{}, nrep);
        test_copy<float, Cpu>(size, ctx.toCpu(0), EWOp::Add{}, nrep);
        test_copy<double, Cpu>(size, ctx.toCpu(0), EWOp::Copy{}, nrep);
        test_copy<double, Cpu>(size, ctx.toCpu(0), EWOp::Add{}, nrep);
        test_copy<std::complex<float>, Cpu>(size, ctx.toCpu(0), EWOp::Copy{}, nrep);
        test_copy<std::complex<float>, Cpu>(size, ctx.toCpu(0), EWOp::Add{}, nrep);
        test_copy<std::complex<double>, Cpu>(size, ctx.toCpu(0), EWOp::Copy{}, nrep);
        test_copy<std::complex<double>, Cpu>(size, ctx.toCpu(0), EWOp::Add{}, nrep);
        test_select<float, Cpu>(size, ctx.toCpu(0), nrep);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }

#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext();
        resetTimings();
        test_copy<float, Gpu>(size, ctx.toGpu(0), EWOp::Copy{}, nrep);
        test_copy<float, Gpu>(size, ctx.toGpu(0), EWOp::Add{}, nrep);
        test_copy<double, Gpu>(size, ctx.toGpu(0), EWOp::Copy{}, nrep);
        test_copy<double, Gpu>(size, ctx.toGpu(0), EWOp::Add{}, nrep);
        test_copy<std::complex<float>, Gpu>(size, ctx.toGpu(0), EWOp::Copy{}, nrep);
        test_copy<std::complex<float>, Gpu>(size, ctx.toGpu(0), EWOp::Add{}, nrep);
        test_copy<std::complex<double>, Gpu>(size, ctx.toGpu(0), EWOp::Copy{}, nrep);
        test_copy<std::complex<double>, Gpu>(size, ctx.toGpu(0), EWOp::Add{}, nrep);
        test_select<float, Gpu>(size, ctx.toGpu(0), nrep);
        reportTimings(std::cout);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#endif

    std::cout << std::endl;
    std::cout << "- Blocking:" << std::endl;
    {
        Context ctx = createCpuContext();
        test_copy_blocking<float, Cpu>(size, ctx.toCpu(0), EWOp::Copy{}, nrep);
        test_copy_blocking<float, Cpu>(size, ctx.toCpu(0), EWOp::Add{}, nrep);
        test_copy_blocking<double, Cpu>(size, ctx.toCpu(0), EWOp::Copy{}, nrep);
        test_copy_blocking<double, Cpu>(size, ctx.toCpu(0), EWOp::Add{}, nrep);
        test_copy_blocking<std::complex<float>, Cpu>(size, ctx.toCpu(0), EWOp::Copy{}, nrep);
        test_copy_blocking<std::complex<float>, Cpu>(size, ctx.toCpu(0), EWOp::Add{}, nrep);
        test_copy_blocking<std::complex<double>, Cpu>(size, ctx.toCpu(0), EWOp::Copy{}, nrep);
        test_copy_blocking<std::complex<double>, Cpu>(size, ctx.toCpu(0), EWOp::Add{}, nrep);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }

#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext();
        resetTimings();
        test_copy_blocking<float, Gpu>(size, ctx.toGpu(0), EWOp::Copy{}, nrep);
        test_copy_blocking<float, Gpu>(size, ctx.toGpu(0), EWOp::Add{}, nrep);
        test_copy_blocking<double, Gpu>(size, ctx.toGpu(0), EWOp::Copy{}, nrep);
        test_copy_blocking<double, Gpu>(size, ctx.toGpu(0), EWOp::Add{}, nrep);
        test_copy_blocking<std::complex<float>, Gpu>(size, ctx.toGpu(0), EWOp::Copy{}, nrep);
        test_copy_blocking<std::complex<float>, Gpu>(size, ctx.toGpu(0), EWOp::Add{}, nrep);
        test_copy_blocking<std::complex<double>, Gpu>(size, ctx.toGpu(0), EWOp::Copy{}, nrep);
        test_copy_blocking<std::complex<double>, Gpu>(size, ctx.toGpu(0), EWOp::Add{}, nrep);
        reportTimings(std::cout);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
