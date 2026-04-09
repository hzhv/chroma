#include "superbblas.h"
#include <algorithm>
#include <ccomplex>
#include <chrono>
#include <complex>
#include <iostream>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

using namespace superbblas;
using namespace superbblas::detail;

template <typename SCALAR> void test() {
    const std::unordered_map<std::type_index, std::string> type_to_string{
        {std::type_index(typeid(std::complex<float>)), "complex float"},
        {std::type_index(typeid(std::complex<double>)), "complex double"}};
    std::cout << "Testing " << type_to_string.at(std::type_index(typeid(SCALAR)))
#ifndef SUPERBBLAS_LIB
              << " with a specific implementation for a vectorization of "
              << superbblas::detail_xp::get_native_size<SCALAR>::size << " parts"
#endif
              << std::endl;
    std::vector<SCALAR> a(9);
    for (size_t i = 0; i < a.size(); ++i) a[i] = {1.f * i, .5f * i};

    for (int n = 1; n < 10; ++n) {
        std::cout << ".. for rhs= " << n << std::endl;
        std::vector<SCALAR> b(3 * n);
        for (size_t i = 0; i < b.size(); ++i) b[i] = {1.f * i, 1.f * i};

        {
            std::vector<SCALAR> c(3 * n);

            xgemm_alt_alpha1_beta1('n', 'n', 3, n, 3, a.data(), 3, b.data(), 3, c.data(), 3, Cpu{});

            std::vector<SCALAR> c0(3 * n);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < 3; ++k) c0[i + 3 * j] += a[i + 3 * k] * b[k + 3 * j];

            double r = 0;
            for (int i = 0; i < 3 * n; ++i) r += std::norm(c[i] - c0[i]);
            std::cout << "Error: " << std::sqrt(r) << std::endl;
        }
        {
            std::vector<SCALAR> c(3 * n);

            xgemm_alt_alpha1_beta1('n', 'n', n, 3, 3, b.data(), n, a.data(), 3, c.data(), n, Cpu{});

            std::vector<SCALAR> c0(3 * n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; ++k) c0[i + n * j] += b[i + n * k] * a[k + 3 * j];

            double r = 0;
            for (int i = 0; i < 3 * n; ++i) r += std::norm(c[i] - c0[i]);
            std::cout << "Error: " << std::sqrt(r) << std::endl;
        }
    }
}

#ifdef SUPERBBLAS_USE_GPU
template <typename SCALAR> void test_gpu(const Gpu &xpu) {
    const std::unordered_map<std::type_index, std::string> type_to_string{
        {std::type_index(typeid(std::complex<float>)), "complex float"},
        {std::type_index(typeid(std::complex<double>)), "complex double"}};

    const bool specialized_kernel_support =
        superbblas::detail::available_bsr_kron_3x3_4x4perm<SCALAR>(xpu);
    if (!specialized_kernel_support) {
        std::cout << "Not doing testing for " << type_to_string.at(std::type_index(typeid(SCALAR)))
                  << std::endl;
        return;
    }
    std::cout << "Testing " << type_to_string.at(std::type_index(typeid(SCALAR)))
              << " with a specialized kernel support" << std::endl;

    vector<SCALAR, Cpu> a_cpu(9, Cpu{});
    for (size_t i = 0; i < a_cpu.size(); ++i) a_cpu[i] = {i * 1.f, .5f * i};
    auto a = makeSure(a_cpu, xpu);
    vector<SCALAR, Cpu> k_cpu(4, Cpu{});
    for (size_t i = 0; i < k_cpu.size(); ++i) k_cpu[i] = {.5f * (i + 1), 1.f};
    auto k = makeSure(k_cpu, xpu);
    vector<int, Cpu> k_perm_cpu(4, Cpu{});
    for (size_t i = 0; i < k_perm_cpu.size(); ++i) k_perm_cpu[i] = 3 - i;
    auto k_perm = makeSure(k_perm_cpu, xpu);
    vector<int, Cpu> jj_cpu(1, Cpu{});
    jj_cpu[0] = 0;
    auto jj = makeSure(jj_cpu, xpu);

    for (int n = 1; n <= 10; ++n) {
        std::cout << ".. for rhs= " << n << std::endl;
        vector<SCALAR, Cpu> x_cpu(12 * n, Cpu{});
        for (size_t i = 0; i < x_cpu.size(); ++i) x_cpu[i] = {i * 1.f, .1f * i};
        auto x = makeSure(x_cpu, xpu);

        {
            vector<SCALAR, Gpu> y(12 * n, xpu);
            bsr_kron_3x3_4x4perm(a.data(), 1, 3, jj.data(), 1, 1, k.data(), k_perm.data(), x.data(),
                                 12 * n, y.data(), 12 * n, n, xpu);
            auto y_cpu = makeSure(y, Cpu{});

            std::vector<SCALAR> y0(12 * n);
            for (int j = 0; j < n; j++)
                for (int k = 0; k < 4; ++k)
                    for (int i = 0; i < 3; ++i)
                        for (int s = 0; s < 3; ++s)
                            y0[k + 4 * j + 4 * n * i] += a_cpu[i + 3 * s] *
                                                         x_cpu[k_perm_cpu[k] + 4 * j + 4 * n * s] *
                                                         k_cpu[k];

            double r = 0;
            for (int i = 0; i < 12 * n; ++i) r += std::norm(y0[i] - y_cpu[i]);
            std::cout << "Error: " << std::sqrt(r) << std::endl;
        }
    }
}

template <typename T> struct aux {
    static T normal_value() { return T{1}; }
    static T cond_conj(bool, T v) { return v; }
};
template <typename T> struct aux<std::complex<T>> {
    static std::complex<T> normal_value() { return std::complex<T>{1, .5}; }
    static std::complex<T> cond_conj(bool conj, std::complex<T> v) {
        return !conj ? v : std::conj(v);
    }
};

template <typename SCALAR>
void test_inner_prod_gpu(const Gpu &xpu) {
    const std::unordered_map<std::type_index, std::string> type_to_string{
        {std::type_index(typeid(float)), "float"},
        {std::type_index(typeid(double)), "double"},
        {std::type_index(typeid(std::complex<float>)), "complex float"},
        {std::type_index(typeid(std::complex<double>)), "complex double"}};

    std::cout << "Testing " << type_to_string.at(std::type_index(typeid(SCALAR))) << std::endl;

    const auto m = 10000, max_n = 5;
    vector<SCALAR, Cpu> a_cpu(m * max_n, Cpu{});
    const auto nval = aux<SCALAR>::normal_value();
    for (size_t i = 0; i < a_cpu.size(); ++i) a_cpu[i] = nval * SCALAR{std::log(1.f * (i + 1))};
    auto a = makeSure(a_cpu, xpu);
    vector<SCALAR, Cpu> b_cpu(m * max_n, Cpu{});
    for (size_t i = 0; i < b_cpu.size(); ++i) b_cpu[i] = nval * SCALAR{std::log(1.f * (i + 3))};
    auto b = makeSure(b_cpu, xpu);
    vector<SCALAR, Cpu> o_cpu(max_n, Cpu{});
    for (size_t i = 0; i < o_cpu.size(); ++i)
        o_cpu[i] = nval * SCALAR{1.f * i};
    const auto alpha = nval;
    const auto beta = nval * SCALAR{.5};

    auto exact_result = [=](SCALAR alpha, SCALAR beta, bool conja, bool conjb) {
        std::vector<SCALAR> r0(max_n);
        for (int j = 0; j < max_n; j++) {
            SCALAR r0j{0};
            for (int i = 0; i < m; ++i) {
                r0j += aux<SCALAR>::cond_conj(conja, a_cpu[i + m * j]) *
                       aux<SCALAR>::cond_conj(conjb, b_cpu[i + m * j]);
            }
            r0[j] = alpha * r0j + beta * o_cpu[j];
        }
        return r0;
    };

    for (int n = 1; n <= max_n; ++n) {
        std::cout << ".. for rhs= " << n << std::endl;
        vector<SCALAR, Gpu> r(n, xpu);
        for (bool conja : std::vector<bool>{false, true}) {
            for (bool conjb : std::vector<bool>{false, true}) {
                inner_prod_gpu(m, n, alpha, a.data(), 1, m, conja, b.data(), 1, m, conjb, SCALAR{0},
                               r.data(), 1, xpu);
                auto r_cpu = makeSure(r, Cpu{});

                const auto r0 = exact_result(alpha, 0, conja, conjb);

                double d = 0, nd = 0;
                for (int i = 0; i < n; ++i) d += std::norm(r0[i] - r_cpu[i]);
                for (int i = 0; i < n; ++i) nd += std::norm(r_cpu[i]);
                std::cout << "Error: " << std::sqrt(d/nd) << std::endl;
            }
        }
        for (bool conja : std::vector<bool>{false, true}) {
            for (bool conjb : std::vector<bool>{false, true}) {
                r = makeSure(o_cpu, xpu);
                inner_prod_gpu(m, n, alpha, a.data(), 1, m, conja, b.data(), 1, m, conjb, beta,
                               r.data(), 1, xpu);
                auto r_cpu = makeSure(r, Cpu{});

                const auto r0 = exact_result(alpha, beta, conja, conjb);

                double d = 0, nd = 0;
                for (int i = 0; i < n; ++i) d += std::norm(r0[i] - r_cpu[i]);
                for (int i = 0; i < n; ++i) nd += std::norm(r_cpu[i]);
                std::cout << "Error: " << std::sqrt(d/nd) << std::endl;
            }
        }
    }
}
#endif // SUPERBBLAS_USE_GPU

int main(int, char **) {
#ifdef SUPERBBLAS_USE_FLOAT16
    test<std::complex<_Float16>>();
#endif
    test<std::complex<float>>();
    test<std::complex<double>>();
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext(0);
        test_gpu<std::complex<double>>(ctx.toGpu(0));
        test_inner_prod_gpu<float>(ctx.toGpu(0));
        test_inner_prod_gpu<double>(ctx.toGpu(0));
        test_inner_prod_gpu<std::complex<float>>(ctx.toGpu(0));
        test_inner_prod_gpu<std::complex<double>>(ctx.toGpu(0));
    }
#endif

    return 0;
}
