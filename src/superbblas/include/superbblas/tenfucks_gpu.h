/// TENsor Fix User-defined Contraction K-dimensional Subroutine (TenFuCKS)

/// Proposed strategy for matrix-matrix multiplication:
/// - Use tensor cores

#ifndef __SUPERBBLAS_TENFUCKS_GPU__
#define __SUPERBBLAS_TENFUCKS_GPU__

#include "platform.h"

#if defined(SUPERBBLAS_USE_CUDA) && defined(SUPERBBLAS_GENERATE_KERNELS)
#    if __CUDA_ARCH__ >= 800
#        define SUPERBBLAS_CUDA_SUPPORTS_TENSOR_CORES_FOR_DOUBLES
#    endif
#    if __CUDA_ARCH__ >= 700
#        define SUPERBBLAS_CUDA_SUPPORTS_TENSOR_CORES
#    endif
#endif

#if defined(SUPERBBLAS_USE_HIP) && defined(SUPERBBLAS_GENERATE_KERNELS)

#    include <hip/hip_ext.h>

/// Detect architectures with tensor cores
/// NOTE: all from GFX9

#    if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) ||                       \
        defined(__gfx941__) || defined(__gfx942__) || defined(__gfx1100__) ||                      \
        defined(__gfx1101__) || defined(__gfx1102__)
#        define SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES
#    endif

/// Detect architectures with tensor cores for double precision
/// NOTE: all from GFX9 excepting GFX908

#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES) && !defined(__gfx908__)
#        define SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES
#    endif

#endif

#if defined(SUPERBBLAS_CREATING_LIB) // && defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
/// Generate template instantiations for bsr_kron_3x3_4x4perm functions with template parameter T

#    define DECL_BSR_KRON_3x3_4x4PERM_T(...)                                                       \
        EMIT REPLACE1(bsr_kron_3x3_4x4perm, superbblas::detail::bsr_kron_3x3_4x4perm<T>)           \
            REPLACE_T template __VA_ARGS__;

/// Generate template instantiations for available_bsr_kron_3x3_4x4perm functions with template parameter T

#    define DECL_AVAILABLE_BSR_KRON_3x3_4x4PERM_T(...)                                             \
        EMIT REPLACE1(available_bsr_kron_3x3_4x4perm,                                              \
                      superbblas::detail::available_bsr_kron_3x3_4x4perm<T>)                       \
            REPLACE_T template __VA_ARGS__;

#else
#    define DECL_BSR_KRON_3x3_4x4PERM_T(...) __VA_ARGS__
#    define DECL_AVAILABLE_BSR_KRON_3x3_4x4PERM_T(...) __VA_ARGS__
#endif

namespace superbblas {
    namespace detail {

#ifdef SUPERBBLAS_USE_GPU

        __host__ __device__ inline int get_a_idx(int a_ldr, int a_ldc, int num_dirs, int color_row,
                                                 int color_col, int block_row, int dir) {
            return a_ldr * color_row + a_ldc * color_col + 3 * 3 * dir +
                   3 * 3 * num_dirs * block_row;
        }

        __host__ __device__ inline int get_xy_idx(int ldr, int ncols, int color, int spin,
                                                  int block_row, int col) {
            //return color + 3*spin + 3*4*col + ldr*block_row;
            return spin + 4 * col + 4 * ncols * color + ldr * block_row;
        }

        __host__ __device__ inline int get_a_idx_complex(int a_ldr, int a_ldc, int num_dirs,
                                                         int color_row, int color_col,
                                                         int block_row, int dir) {
            return (a_ldr * color_row + a_ldc * color_col + 3 * 3 * dir +
                    3 * 3 * num_dirs * block_row) *
                   2;
        }

        __host__ __device__ inline int get_xy_idx_complex(int ldr, int ncols, int color, int spin,
                                                          int block_row, int col) {
            //return (color + 3*spin + 3*4*col + ldr*block_row)*2;
            return (spin + 4 * col + 4 * ncols * color + ldr * block_row) * 2;
        }

        __host__ __device__ inline int get_jj_idx(int num_dirs, int block_row, int dir) {
            return dir + num_dirs * block_row;
        }
#endif // SUPERBBLAS_USE_GPU

#ifdef SUPERBBLAS_USE_HIP
        template <typename T> struct bsr_kron_3x3_4x4perm_kernel;

        /// Default implementation for unsupported types

        template <typename T> struct bsr_kron_3x3_4x4perm_kernel {
            static constexpr bool type_available() { return false; }

            using ptr = T *;

            static dim3 block_size() { return dim3(0, 0, 0); }

            static dim3 grid_size(int, int) { return dim3(0, 0, 0); }

            __global__ static void available(int *flag) { *flag = 0; }

            __global__ static void fun(const T *a, int a_ldr, int a_ldc, int *jj, int block_rows,
                                       int num_dirs, const T *perm_scalars, const int *perm,
                                       const T *x, int ldx, T *y, int ldy, int ncols) {
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
            }
        };

        /// Implementation for complex double

        template <> struct bsr_kron_3x3_4x4perm_kernel<std::complex<double>> {
            static constexpr bool type_available() { return true; }

            using ptr = double *;

            static dim3 block_size() { return dim3(4, 4, 4); }

            static dim3 grid_size(int block_rows, int num_cols) {
                return dim3((num_cols + 3) / 4, block_rows, 1);
            }

            __global__ static void available(int *flag) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
                *flag = 1;
#    else
                *flag = 0;
#    endif
            }

            __global__ static void fun(const double *a, int a_ldr, int a_ldc, int *jj,
                                       int block_rows, int num_dirs, const double *perm_scalars,
                                       const int *perm, const double *x, int ldx, double *y,
                                       int ldy, int ncols) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
                (void)block_rows;
                auto col = blockIdx.x * 4 + threadIdx.y;
                auto blk_row = blockIdx.y;
                auto a_row = threadIdx.x;
                auto a_col = threadIdx.z;
                auto x_color = threadIdx.z;
                auto x_spin = threadIdx.x;
                auto y_color = threadIdx.z;
                auto y_spin = threadIdx.x;
                double y_val_r = 0.0, y_val_i = 0.0;
                for (int dir = 0; dir < num_dirs; ++dir) {
                    // read a
                    bool a_is_zero = (a_row == 3 || a_col == 3);
                    int a_idx =
                        get_a_idx_complex(a_ldr, a_ldc, num_dirs, a_row, a_col, blk_row, dir);
                    double a_val_r = 0.0, a_val_i = 0.0;
                    if (!a_is_zero) a_val_r = a[a_idx], a_val_i = a[a_idx + 1];

                    // read x
                    bool x_is_zero = (x_color == 3 || col >= ncols);
                    int x_idx = get_xy_idx_complex(ldx, ncols, x_color, perm[4 * dir + x_spin],
                                                   jj[get_jj_idx(num_dirs, blk_row, dir)], col);
                    double x_val_r = 0.0, x_val_i = 0.0;
                    int s_dir = (4 * dir + x_spin) * 2;
                    const double s_r = perm_scalars[s_dir], s_i = perm_scalars[s_dir + 1];
                    if (!x_is_zero) x_val_r = x[x_idx], x_val_i = x[x_idx + 1];
                    const double b_val_r = x_val_r * s_r - x_val_i * s_i;
                    const double b_val_i = x_val_r * s_i + x_val_i * s_r;

                    // a[real] times x[real] -> y[real]
                    y_val_r =
                        __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_r, b_val_r, y_val_r, 0, 0, 0);

                    // a[real] times x[imag] -> y[imag]
                    y_val_i =
                        __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_r, b_val_i, y_val_i, 0, 0, 0);

                    // a[imag] times x[real] -> y[imag]
                    y_val_i =
                        __builtin_amdgcn_mfma_f64_4x4x4f64(a_val_i, b_val_r, y_val_i, 0, 0, 0);

                    // a[imag] times x[imag] -> y[real]
                    y_val_r =
                        __builtin_amdgcn_mfma_f64_4x4x4f64(-a_val_i, b_val_i, y_val_r, 0, 0, 0);
                }
                bool y_is_zero = (y_color == 3 || col >= ncols);
                int y_idx = get_xy_idx_complex(ldy, ncols, y_color, y_spin, blk_row, col);
                if (!y_is_zero) y[y_idx] = y_val_r, y[y_idx + 1] = y_val_i;
#    else
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
#    endif // defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
            }
        };

        /// Implementation for complex single

        template <> struct bsr_kron_3x3_4x4perm_kernel<std::complex<float>> {
            static constexpr bool type_available() { return true; }

            using ptr = float *;

            static dim3 block_size() { return dim3(4, 16, 1); }

            static dim3 grid_size(int block_rows, int num_cols) {
                return dim3((num_cols + 15) / 16, block_rows, 1);
            }

            __global__ static void available(int *flag) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
                *flag = 1;
#    else
                *flag = 0;
#    endif
            }

            __global__ static void fun(const float *a, int a_ldr, int a_ldc, int *jj,
                                       int block_rows, int num_dirs, const float *perm_scalars,
                                       const int *perm, const float *x, int ldx, float *y, int ldy,
                                       int ncols) {
#    if defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
                (void)block_rows;
                using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
                auto col = blockIdx.x * 16 + threadIdx.y;
                auto blk_row = blockIdx.y;
                auto a_row = threadIdx.x;
                auto x_spin = threadIdx.x;
                auto y_spin = threadIdx.x;
                float4 y_val_r = {0}, y_val_i = {0};
                for (int dir = 0; dir < num_dirs; ++dir) {
                    for (int k = 0; k < 3; ++k) {
                        // read a
                        bool a_is_zero = (a_row == 3);
                        int a_idx =
                            get_a_idx_complex(a_ldr, a_ldc, num_dirs, a_row, k, blk_row, dir);
                        float a_val_r = 0.0, a_val_i = 0.0;
                        if (!a_is_zero) a_val_r = a[a_idx], a_val_i = a[a_idx + 1];

                        // read x
                        bool x_is_zero = (col >= ncols);
                        int x_idx = get_xy_idx_complex(ldx, ncols, k, perm[4 * dir + x_spin],
                                                       jj[get_jj_idx(num_dirs, blk_row, dir)], col);
                        float x_val_r = 0.0, x_val_i = 0.0;
                        int s_dir = (4 * dir + x_spin) * 2;
                        const float s_r = perm_scalars[s_dir], s_i = perm_scalars[s_dir + 1];
                        if (!x_is_zero) x_val_r = x[x_idx], x_val_i = x[x_idx + 1];
                        const float b_val_r = x_val_r * s_r - x_val_i * s_i;
                        const float b_val_i = x_val_r * s_i + x_val_i * s_r;

                        // a[real] times x[real] -> y[real]
                        y_val_r =
                            __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_r, b_val_r, y_val_r, 0, 0, 0);

                        // a[real] times x[imag] -> y[imag]
                        y_val_i =
                            __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_r, b_val_i, y_val_i, 0, 0, 0);

                        // a[imag] times x[real] -> y[imag]
                        y_val_i =
                            __builtin_amdgcn_mfma_f32_4x4x1f32(a_val_i, b_val_r, y_val_i, 0, 0, 0);

                        // a[imag] times x[imag] -> y[real]
                        y_val_r =
                            __builtin_amdgcn_mfma_f32_4x4x1f32(-a_val_i, b_val_i, y_val_r, 0, 0, 0);
                    }
                }
                bool y_is_zero = (col >= ncols);
                if (!y_is_zero) {
                    for (int k = 0; k < 3; ++k) {
                        int y_idx = get_xy_idx_complex(ldy, ncols, k, y_spin, blk_row, col);
                        y[y_idx] = y_val_r[k];
                        y[y_idx + 1] = y_val_i[k];
                    }
                }
#    else
                (void)a;
                (void)a_ldr;
                (void)a_ldc;
                (void)jj;
                (void)block_rows;
                (void)num_dirs;
                (void)perm_scalars;
                (void)perm;
                (void)x;
                (void)ldx;
                (void)y;
                (void)ldy;
                (void)ncols;
#    endif // defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES)
            }
        };

        /// Computes the BSR-kron matrix vector multiplication
        /// \param a: a[r*a_ldr+c*a_ldc+j*9] is the nonzero value at row r+3*I and column c+jj[j]
        /// \param a_ldr: jump to the element at the next row in a nonzero block
        /// \param a_ldc: jump to the element at the next column in a nonzero block
        /// \param jj: column indices for each nonzero block
        /// \param num_dirs: number of nonzero blocks per row
        /// \param perm_scalars: 4*num_dirs with the nonzero values of the CSR kron matrices
        /// \param perm: 4*num_dirs with the column indices of the CSR kron matrices
        /// \param x: right-hand-side nonzeros with ordering 4,column,3,row
        /// \param ldx: leading dimension of x
        /// \param y: output nonzeros with ordering 4,column,3,row
        /// \param ldy: leading dimension of y
        /// \param ncols: number of columns on x and y
        /// \param xpu: gpu context
        ///
        /// NOTE:
        /// The routine does:
        ///   y(0:3,0:2,I,n) = \sum_{j=0:dirs-1} [ a(0:2,I,0:2,J(I,j)) \Kron_Prod
        ///                                          k(0:4,0:4,j)  ] * x(0:3,0:2,J(I,j),n),
        /// where
        /// - J(I,j) is the block column index for block row I and direction j;
        /// - a is sparse matrix with 3x3 non-overlapping dense blocks with `dirs` nonzero block
        ///   per row;
        /// - k is a permutation 4x4 matrix with each row scaled independently;
        /// - x,y are the input/output tensors with several fields indexed by `n`.
        ///
        /// The input and output tensors are ordered as a kind of row-major: (0:3,n,0:2,I).

        template <typename T>
        DECL_BSR_KRON_3x3_4x4PERM_T(void bsr_kron_3x3_4x4perm(const T *a, int a_ldr, int a_ldc,
                                                              int *jj, int block_rows, int num_dirs,
                                                              const T *perm_scalars,
                                                              const int *perm, const T *x, int ldx,
                                                              T *y, int ldy, int ncols, Gpu xpu))
            IMPL({
                if (!bsr_kron_3x3_4x4perm_kernel<T>::type_available())
                    throw std::runtime_error("wtf!");
                using ptr = typename bsr_kron_3x3_4x4perm_kernel<T>::ptr;
                setDevice(xpu);
                hipExtLaunchKernelGGL(bsr_kron_3x3_4x4perm_kernel<T>::fun,
                                      bsr_kron_3x3_4x4perm_kernel<T>::grid_size(block_rows, ncols),
                                      bsr_kron_3x3_4x4perm_kernel<T>::block_size(),
                                      0,              // sharedMemBytes
                                      getStream(xpu), // stream
                                      0,              // Event start
                                      0,              // event stop
                                      0,              // flags
                                      (ptr)a, a_ldr, a_ldc, jj, 1, num_dirs, (ptr)perm_scalars,
                                      perm, (const ptr)x, ldx, (ptr)y, ldy, ncols);
                gpuCheck(hipGetLastError());
            })

        /// Return whether the gpu supports the specialized BSR-Kron kernel

        template <typename T>
        DECL_AVAILABLE_BSR_KRON_3x3_4x4PERM_T(bool available_bsr_kron_3x3_4x4perm(const Gpu &xpu))
            IMPL({
                if (!bsr_kron_3x3_4x4perm_kernel<T>::type_available()) return false;
                setDevice(xpu);
                int *flag;
                gpuCheck(hipMalloc(&flag, sizeof(int)));
                bsr_kron_3x3_4x4perm_kernel<T>::available<<<1, 1>>>(flag);
                gpuCheck(hipGetLastError());
                int flag_host = 0;
                gpuCheck(hipMemcpy(&flag_host, flag, sizeof(int), hipMemcpyDeviceToHost));
                gpuCheck(hipFree(flag));
                return flag_host != 0;
            })
#endif // SUPERBBLAS_USE_HIP
#ifdef SUPERBBLAS_USE_CUDA

#    ifdef SUPERBBLAS_GENERATE_KERNELS
        /// Default implementation for unsupported types

        template <typename T> struct bsr_kron_3x3_4x4perm_kernel {
            static constexpr bool type_available() { return false; }

            using ptr = typename the_real<T>::type *;

            static dim3 block_size() { return dim3(0, 0, 0); }

            static dim3 grid_size(int, int) { return dim3(0, 0, 0); }
        };

        template <typename T> __global__ void bsr_kron_3x3_4x4perm_kernel_available(int *flag) {
            *flag = 0;
        }

        template <typename T>
        __global__ void bsr_kron_3x3_4x4perm_kernel_fun(
            const typename the_real<T>::type *a, int a_ldr, int a_ldc, int *jj, int block_rows,
            int num_dirs, const typename the_real<T>::type *perm_scalars, const int *perm,
            const typename the_real<T>::type *x, int ldx, typename the_real<T>::type *y, int ldy,
            int ncols) {
            (void)a;
            (void)a_ldr;
            (void)a_ldc;
            (void)jj;
            (void)block_rows;
            (void)num_dirs;
            (void)perm_scalars;
            (void)perm;
            (void)x;
            (void)ldx;
            (void)y;
            (void)ldy;
            (void)ncols;
        }

        /// Implementation for complex double

        template <> struct bsr_kron_3x3_4x4perm_kernel<std::complex<double>> {
            static constexpr bool type_available() { return true; }

            using ptr = double *;

            static dim3 block_size() { return dim3(32, 1, 1); }

            static dim3 grid_size(int block_rows, int num_cols) {
                return dim3(block_rows, num_cols, 1);
            }
        };

        template <>
        inline __global__ void
        bsr_kron_3x3_4x4perm_kernel_available<std::complex<double>>(int *flag) {
#        if defined(SUPERBBLAS_CUDA_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
            *flag = 1;
#        else
            *flag = 0;
#        endif
        }

        /// Computes the BSR-kron matrix vector multiplication
        /// \param a: a[r*a_ldr+c*a_ldc+j*9] is the nonzero complex value at row r+3*I and column c+jj[j]
        /// \param a_ldr: jump to the element on the next row within a nonzero block
        /// \param a_ldc: jump to the element on the next column within a nonzero block
        /// \param jj: column indices for each nonzero block
        /// \param num_dirs: number of nonzero blocks per row
        /// \param perm_scalars: 4*num_dirs with the nonzero complex values of the CSR kron matrices
        /// \param perm: 4*num_dirs with the column indices of the CSR kron matrices
        /// \param x: right-hand-side complex nonzeros with ordering 4,column,3,row
        /// \param ldx: leading dimension of x
        /// \param y: output complex nonzeros with ordering 4,column,3,row
        /// \param ldy: leading dimension of y
        /// \param ncols: number of columns on x and y
	///
	/// NOTE:
	/// The intrinsic mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 performs the matrix-matrix
	/// multiplication of a 8-by-4 and a 4-by-8 matrix on a single warp of 32 threads. The layout of
	/// of the input and output matrix can be check here: 
	///  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f64
	/// The thread with index i passes and receives the following values for doing A*B + C -> D:
	/// - a(i/4,i%4)
	/// - b(i%4,i/4)
	/// - {c(i/4,(i%4)*2), c(i/4,(i%4)*2+1)}
	/// - {d(i/4,(i%4)*2), d(i/4,(i%4)*2+1)}

        template <>
        inline __global__ void bsr_kron_3x3_4x4perm_kernel_fun<std::complex<double>>(
            const double *a, int a_ldr, int a_ldc, int *jj, int block_rows, int num_dirs,
            const double *perm_scalars, const int *perm, const double *x, int ldx, double *y,
            int ldy, int ncols) {
#        if defined(SUPERBBLAS_CUDA_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
            (void)block_rows;
            double c[2] = {0, 0}; ///< accumulator
            double d[2] = {0, 0}; ///< result
            auto col = blockIdx.y;
            auto blk_row = blockIdx.x;
            auto a_row = (threadIdx.x / 4) % 4;
            auto a_col = threadIdx.x % 4;
            auto x_color = threadIdx.x % 4;
            auto x_spin = (threadIdx.x / 4) % 4;
            auto y_color = (threadIdx.x / 4) % 4;
            auto y_spin = (threadIdx.x % 2) * 2;
            int a_re = threadIdx.x / 16; ///< 0:real, 1:imag
            int x_re = threadIdx.x / 16; ///< 0:real, 1:imag
            int y_re0 = threadIdx.x / 16;
            int y_re1 = (threadIdx.x % 4) / 2;
            for (int dir = 0; dir < num_dirs; ++dir) {
                // read a
                bool a_is_zero = (a_row == 3 || a_col == 3);
                int a_idx = get_a_idx_complex(a_ldr, a_ldc, num_dirs, a_row, a_col, blk_row, dir);
                double a_val = 0.0;
                if (!a_is_zero) a_val = a[a_idx + a_re];

                // read x
                bool x_is_zero = (x_color == 3 || col >= ncols);
                int x_idx = get_xy_idx_complex(ldx, ncols, x_color, perm[4 * dir + x_spin],
                                               jj[get_jj_idx(num_dirs, blk_row, dir)], col);
                double x_val_r = 0.0, x_val_i = 0.0;
                int s_dir = (4 * dir + x_spin) * 2;
                const double s_r = perm_scalars[s_dir], s_i = perm_scalars[s_dir + 1];
                if (!x_is_zero) x_val_r = x[x_idx], x_val_i = x[x_idx + 1];
                const double b_val_r = x_val_r * s_r - x_val_i * s_i;
                const double b_val_i = x_val_r * s_i + x_val_i * s_r;
                const double b_val = x_re == 0 ? b_val_r : b_val_i;

                // Use MMA intrinsic for matrix multiplication D = A*B + C
                c[0] = d[0];
                c[1] = d[1];
                asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                             "{%0, %1}, "
                             "{%2}, "
                             "{%3}, "
                             "{%4, %5};"
                             : "=d"(d[0]), "=d"(d[1])
                             : "d"(a_val), "d"(b_val), "d"(c[0]), "d"(c[1]));
            }
            bool y_is_zero = (y_color == 3 || col >= ncols);
            int y_idx0 = get_xy_idx_complex(ldy, ncols, y_color, y_spin, blk_row, col);
            int y_idx1 = get_xy_idx_complex(ldy, ncols, y_color, y_spin + 1, blk_row, col);
            // real(y) = y_val[y_re0 == 0 && y_re1 == 0] - y_val[y_re0 == 1 && y_re1 == 1]
            // imag(y) = y_val[y_re0 == 0 && y_re1 == 1] + y_val[y_re0 == 1 && y_re1 == 0]
            // Then,
            // - threads y_re0 == 0 && y_re1 == 0 pass d[1] and get d[0] from y_re0 == 1 && y_re1 == 1
            // - threads y_re0 == 1 && y_re1 == 1 pass d[0] and get d[1] from y_re0 == 0 && y_re1 == 0
            // - threads y_re0 == 0 && y_re1 == 1 pass d[1] and get d[0] from y_re0 == 1 && y_re1 == 0
            // - threads y_re0 == 1 && y_re1 == 0 pass d[0] and get d[1] from y_re0 == 0 && y_re1 == 1
            int source_thr = (y_color + 4 * (1 - y_re0)) * 4 + y_spin / 2 + 2 * (1 - y_re1);
            int di = (y_re0 == y_re1 ? 1 - y_re0 : y_re1);
            const double d_other = __longlong_as_double(
                __shfl_sync(0xffffffff, __double_as_longlong(d[di]), source_thr));
            if (!y_is_zero && y_re0 == 0 && y_re1 == 0) y[y_idx0] = d[0] - d_other;
            if (!y_is_zero && y_re0 == 1 && y_re1 == 1) y[y_idx1] = d_other - d[1];
            if (!y_is_zero && y_re0 == 0 && y_re1 == 1) y[y_idx0 + 1] = d[0] + d_other;
            if (!y_is_zero && y_re0 == 1 && y_re1 == 0) y[y_idx1 + 1] = d[1] + d_other;
#        else
            (void)a;
            (void)a_ldr;
            (void)a_ldc;
            (void)jj;
            (void)block_rows;
            (void)num_dirs;
            (void)perm_scalars;
            (void)perm;
            (void)x;
            (void)ldx;
            (void)y;
            (void)ldy;
            (void)ncols;
#        endif // defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
        }

        /// Implementation for complex single
        /// NOTE: in progress

#        if 0

        template <> struct bsr_kron_3x3_4x4perm_kernel<std::complex<float>> {
            static constexpr bool type_available() { return true; }

            using ptr = double *;

            static dim3 block_size() { return dim3(32, 1, 1); }

            static dim3 grid_size(int block_rows, int num_cols) {
                return dim3((num_cols + 3) / 4, block_rows, 1);
            }
        };

        template <>
        __global__ void bsr_kron_3x3_4x4perm_kernel_available<std::complex<float>>(int *flag) {
#            if defined(SUPERBBLAS_CUDA_SUPPORTS_TENSOR_CORES)
            *flag = 1;
#            else
            *flag = 0;
#            endif
        }

        template <>
        __global__ void bsr_kron_3x3_4x4perm_kernel_fun<std::complex<float>>(
            const float *a, int a_ldr, int a_ldc, int *jj, int block_rows, int num_dirs,
            const float *perm_scalars, const int *perm, const float *x, int ldx, float *y, int ldy,
            int ncols) {
#            if defined(SUPERBBLAS_CUDA_SUPPORTS_TENSOR_CORES)
            (void)block_rows;
            float c[2] = {0, 0, 0, 0}; ///< accumulator
            float d[2] = {0, 0, 0, 0}; ///< result
            auto col = blockIdx.x * 4 + (threadIdx.x / 4) % 4;
            auto blk_row = blockIdx.y;
            auto a_row = (threadIdx.x / 4) % 4;
            auto a_col = threadIdx.x % 4;
            auto x_color = threadIdx.x % 4;
            auto x_spin = (threadIdx.x / 4) % 4;
            auto y_color = (threadIdx.x / 4) % 4;
            auto y_spin = (threadIdx.x % 2) * 2;
            int a_re = (threadIdx.x / 16);    ///< 0:real, 1:imag
            int x_re = (threadIdx.x % 8) / 4; ///< 0:real, 1:imag
            int y_re0 = (threadIdx.x / 4) / 4;
            int y_re1 = (threadIdx.x % 4) / 2;
            for (int dir = 0; dir < num_dirs; ++dir) {
                // read a
                bool a_is_zero = (a_row == 3 || a_col == 3);
                int a_idx = get_a_idx_complex(a_ldr, a_ldc, num_dirs, a_row, a_col, blk_row, dir);
                double a_val = 0.0;
                if (!a_is_zero) a_val = a[a_idx + a_re];

                // read x
                bool x_is_zero = (x_color == 3 || col >= ncols);
                int x_idx = get_xy_idx_complex(ldx, ncols, x_color, perm[4 * dir + x_spin],
                                               jj[get_jj_idx(num_dirs, blk_row, dir)], col);
                double x_val = 0.0, x_val_i = 0.0;
                int s_dir = (4 * dir + x_spin) * 2;
                const double s_r = perm_scalars[s_dir], s_i = perm_scalars[s_dir + 1];
                if (!x_is_zero) x_val_r = x[x_idx], x_val_i = x[x_idx + 1];
                x_val_r = x_val_r * s_r - x_val_i * s_i;
                x_val_i = x_val_r * s_i + x_val_i * s_r;
                const double x_val = x_re == 0 ? x_val_r : x_val_i;

                // Use MMA intrinsic for matrix multiplication D = A*B + C
                c = d;
                auto a_val_tf0 = __float_to_tf32(a_val0);
                auto a_val_tf1 = __float_to_tf32(a_val1);
                auto b_val_tf = __float_to_tf32(a_val);
                asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
                             "{%0, %1, %2, %3}, "
                             "{%4, %5}, "
                             "{%6}, "
                             "{%7, %8, %9, %10};"
                             : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                             : "r"(a_val_tf0), "r"(a_val_tf1), //
                               "r"(b_val_tf),                  //
                               "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
            }
            bool y_is_zero = (y_color == 3 || col >= ncols);
            int y_idx0 = get_xy_idx_complex(ldy, ncols, y_color, y_spin, blk_row, col);
            int y_idx1 = get_xy_idx_complex(ldy, ncols, y_color, y_spin + 1, blk_row, col);
            // real(y) = y_val[y_re0 == 0 && y_re1 == 0] - y_val[y_re0 == 1 && y_re1 == 1]
            // imag(y) = y_val[y_re0 == 0 && y_re1 == 1] + y_val[y_re0 == 1 && y_re1 == 0]
            // Then,
            // - threads y_re0 == 0 && y_re1 == 0 pass d[1] and get d[0] from y_re0 == 1 && y_re1 == 1
            // - threads y_re0 == 1 && y_re1 == 1 pass d[0] and get d[1] from y_re0 == 0 && y_re1 == 0
            int source_thr = 0; // get from thread
            int di = 0;         // either pass d[0] or d[1]
            if (y_re0 == 0 && y_re1 == 0)
                source_thr = (y_color + 4) * 4 + 2 + y_spin / 2, di = 1;
            else if (y_re0 == 1 && y_re1 == 1)
                source_thr = (y_color + 4) + y_spin / 2, di = 0;
            else if (y_re0 == 0 && y_re1 == 1)
                source_thr = (y_color + 4) * 4 + y_spin / 2, di = 1;
            else /* if (y_re0 == 1 && y_re1 == 0) */
                source_thr = (y_color + 4) + 2 + y_spin / 2, di = 0;
            const double d_other = __long_long_as_double(
                __shfl_sync(0xffffffff, __double_as_long_long(d[di]), source_thr));
            if (!y_is_zero && y_re0 == 0 && y_re1 == 0) y[y_idx0] = d[0] + d_other;
            if (!y_is_zero && y_re0 == 1 && y_re1 == 1) y[y_idx1] = d[1] + d_other;
            if (!y_is_zero && y_re0 == 0 && y_re1 == 1) y[y_idx0 + 1] = d[0] + d_other;
            if (!y_is_zero && y_re0 == 1 && y_re1 == 0) y[y_idx1 + 1] = d[0] + d_other;
#            else
            (void)a;
            (void)a_ldr;
            (void)a_ldc;
            (void)jj;
            (void)block_rows;
            (void)num_dirs;
            (void)perm_scalars;
            (void)perm;
            (void)x;
            (void)ldx;
            (void)y;
            (void)ldy;
            (void)ncols;
#            endif // defined(SUPERBBLAS_ROCM_SUPPORTS_TENSOR_CORES_FOR_DOUBLES)
        }
#        endif
#    endif // SUPERBBLAS_GENERATE_KERNELS

        /// Computes the BSR-kron matrix vector multiplication
        /// \param a: a[r*a_ldr+c*a_ldc+j*9] is the nonzero value at row r+3*I and column c+jj[j]
        /// \param a_ldr: jump to the element on the next row within a nonzero block
        /// \param a_ldc: jump to the element on the next column within a nonzero block
        /// \param jj: column indices for each nonzero block
        /// \param num_dirs: number of nonzero blocks per row
        /// \param perm_scalars: 4*num_dirs with the nonzero values of the CSR kron matrices
        /// \param perm: 4*num_dirs with the column indices of the CSR kron matrices
        /// \param x: right-hand-side nonzeros with ordering 4,column,3,row
        /// \param ldx: leading dimension of x
        /// \param y: output nonzeros with ordering 4,column,3,row
        /// \param ldy: leading dimension of y
        /// \param ncols: number of columns on x and y
        /// \param xpu: gpu context
        ///
        /// NOTE:
        /// The routine does:
        ///   y(0:3,0:2,I,n) = \sum_{j=0:dirs-1} [ a(0:2,I,0:2,J(I,j)) \kron
        ///                                          k(0:4,0:4,j)  ] * x(0:3,0:2,J(I,j),n),
        /// where
        /// - J(I,j) is the block column index for block row I and direction j;
        /// - a is sparse matrix with 3x3 non-overlapping dense blocks with `dirs` nonzero block
        ///   per row;
        /// - k is a permutation 4x4 matrix with each row scaled independently;
        /// - x,y are the input/output tensors with several fields indexed by `n`.
        ///
        /// The input and output tensors are ordered as a kind of row-major: (0:3,n,0:2,I).
        ///
        /// The routine performs the product of the matrix-vector products where the matrix is the Kronecker
        /// product of a 3-by-3 and a 4-by-4 matrices as two matrix-matrix products using the "vec-trick":
        /// if d = (A \kron B) * c, for m-by-m A and n-by-n B matrices, then
        ///   d = vec(B*mat(c)*A^T),
        /// where mat(x) reshapes x into a n-by-m matrix, and vec(X) returns a column vector by stacking the
        /// columns of X.

        template <typename T>
        DECL_BSR_KRON_3x3_4x4PERM_T(void bsr_kron_3x3_4x4perm(const T *a, int a_ldr, int a_ldc,
                                                              int *jj, int block_rows, int num_dirs,
                                                              const T *perm_scalars,
                                                              const int *perm, const T *x, int ldx,
                                                              T *y, int ldy, int ncols, Gpu xpu))
            IMPL({
                if (!bsr_kron_3x3_4x4perm_kernel<T>::type_available())
                    throw std::runtime_error("wtf!");
                using ptr = typename bsr_kron_3x3_4x4perm_kernel<T>::ptr;
                const auto grid = bsr_kron_3x3_4x4perm_kernel<T>::grid_size(block_rows, ncols);
                const auto blk = bsr_kron_3x3_4x4perm_kernel<T>::block_size();
                setDevice(xpu);
                bsr_kron_3x3_4x4perm_kernel_fun<T><<<grid, blk, 0, getStream(xpu)>>>(
                    (const ptr)a, a_ldr, a_ldc, jj, 1, num_dirs, (const ptr)perm_scalars, perm,
                    (const ptr)x, ldx, (ptr)y, ldy, ncols);
                gpuCheck(cudaGetLastError());
            })

        /// Return whether the gpu supports the specialized BSR-Kron kernel

        template <typename T>
        DECL_AVAILABLE_BSR_KRON_3x3_4x4PERM_T(bool available_bsr_kron_3x3_4x4perm(const Gpu &xpu))
            IMPL({
                if (!bsr_kron_3x3_4x4perm_kernel<T>::type_available()) return false;
                setDevice(xpu);
                int *flag;
                gpuCheck(cudaMalloc(&flag, sizeof(int)));
                bsr_kron_3x3_4x4perm_kernel_available<T><<<1, 1>>>(flag);
                gpuCheck(cudaDeviceSynchronize());
                gpuCheck(cudaGetLastError());
                int flag_host = 0;
                gpuCheck(cudaMemcpy(&flag_host, flag, sizeof(int), cudaMemcpyDeviceToHost));
                gpuCheck(cudaFree(flag));
                return flag_host != 0;
            })

#endif // SUPERBBLAS_USE_CUDA

    }
}
#endif // __SUPERBBLAS_TENFUCKS_GPU__
