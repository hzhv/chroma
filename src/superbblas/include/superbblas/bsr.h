#ifndef __SUPERBBLAS_BSR__
#define __SUPERBBLAS_BSR__

#include "dist.h"
#include "tenfucks_cpu.h"
#include "tenfucks_gpu.h"
#include <numeric>
#include <stdexcept>

namespace superbblas {

    namespace detail {
        enum num_type { half_t, chalf_t, float_t, cfloat_t, double_t, cdouble_t };
        template <typename T> struct num_type_v;
#ifdef SUPERBBLAS_USE_FLOAT16
        template <> struct num_type_v<_Float16> {
            static constexpr num_type value = half_t;
        };
#endif
        template <> struct num_type_v<float> {
            static constexpr num_type value = float_t;
        };
        template <> struct num_type_v<double> {
            static constexpr num_type value = double_t;
        };
#ifdef SUPERBBLAS_USE_FLOAT16
        template <> struct num_type_v<std::complex<_Float16>> {
            static constexpr num_type value = chalf_t;
        };
#endif
        template <> struct num_type_v<std::complex<float>> {
            static constexpr num_type value = cfloat_t;
        };
        template <> struct num_type_v<std::complex<double>> {
            static constexpr num_type value = cdouble_t;
        };
    }

    /// Matrix layout
    enum MatrixLayout {
        RowMajor,   // the Kronecker labels and the column index are the fastest indices
        ColumnMajor // the column index and Kronecker labels are the slowest indices
    };

    // /// Handle for a BSR operator
    struct BSR_handle {
        /// Return number of dimensions of the image space
        /// \param Nd: number of dimensions of the domain (columns) operator
        /// \param Ni: number of dimensions of the image (rows) operator
        /// \param type: nonzero values type
        /// \param ctx: contexts
        /// \param ncomponents: number of consecutive components in each MPI rank
        /// \param nprocs: number of processes supporting the matrix
        /// \param rank: current MPI rank
        /// \param co: coordinate linearization order

        virtual bool check(std::size_t, std::size_t, detail::num_type, const Context *,
                           unsigned int, unsigned int, unsigned int, CoorOrder) {
            return false;
        }

        BSR_handle() {}
        virtual ~BSR_handle() {}
    };

    namespace detail {

        template <typename XPU> struct CsrIndices {
            Indices<XPU> i;              ///< Where the columns indices (j) for the ith row start
            Indices<XPU> j;              ///< Column indices
            bool j_has_negative_indices; ///< whether j has -1 to skip these nonzeros
            int num_nnz_per_row;         ///< either the number of block nnz per row or
                                         ///< -1 if not all rows has the same number of nonzeros
            IndexType nnz;               ///< total number of nonzero blocks
        };

        /// Component of a BSR tensor
        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU> struct BSRComponent {
            Indices<Cpu> i;          ///< number of nonzero blocks on each block image
            vector<Coor<Nd>, Cpu> j; ///< domain coordinates of the nonzero blocks
            vector<T, XPU> it;       ///< nonzero values
            Coor<Nd> dimd;           ///< dimensions of the domain space
            Coor<Ni> dimi;           ///< dimensions of the image space
            Coor<Nd> blockd;         ///< dimensions of a block in the domain space
            Coor<Ni> blocki;         ///< dimensions of a block in the image space
            Coor<Nd> krond;          ///< dimensions of Kronecker in the domain space
            Coor<Ni> kroni;          ///< dimensions of Kronecker in the image space
            vector<T, XPU> kron_it;  ///< nonzero values
            bool blockImFast; ///< whether the image indices are the fastest on the dense blocks
            CoorOrder co;     ///< Coordinate order of ii and jj
            unsigned int componentId; ///< Component Id

            template <typename Q = T, typename = typename std::enable_if<std::is_same<
                                          Q, typename std::remove_const<Q>::type>::value>::type>
            operator BSRComponent<Nd, Ni, const Q, XPU>() const {
                return {i,     j,     it,      dimd,        dimi, blockd,     blocki,
                        krond, kroni, kron_it, blockImFast, co,   componentId};
            }
        };

        /// Return the fraction of nonzero values in the given vector
        /// \param v: input vector

        template <typename T> double nnz_density(const vector<T, Cpu> &v) {
            if (v.size() == 0) return 0;
            std::size_t nnz = 0;
            for (std::size_t i = 0; i < v.size(); ++i)
                if (std::norm(v[i]) > 0) nnz++;
            return (double)nnz / v.size();
        }

        ///
        /// Little sparse matrices on CPU
        ///

        /// Compress Sparse Row

        template <typename IndexType, typename T> struct CSRs {
            Indices<Cpu> ii, jj;
            vector<T, Cpu> vals;
            IndexType nrows, ncols, nmats;
            enum Type { Dense, Identity, Permutation, SparseAllOnes, Sparse };
            std::vector<Type> type;
            std::vector<T> scalar;                    ///< used by Identity, SparseAllOnes
            std::vector<std::vector<IndexType>> perm; ///< permutation
            std::vector<std::vector<T>> scalars;      ///< permutation
            MatrixLayout layout;                      ///< matrix layout for a dense type

            /// Empty constructor

            CSRs() : nrows(0), ncols(0), nmats(0), layout(ColumnMajor) {}

            /// Construct the object from a batch of dense matrices
            /// \param m: matrices storage, one after another
            /// \param nrows: number of rows for each matrix
            /// \param ncols: number of columns for each matrix
            /// \param nmats: number of matrices
            /// \param layout: column/row major

            CSRs(const vector<T, Cpu> &m, IndexType nrows, IndexType ncols, IndexType nmats,
                 MatrixLayout layout)
                : nrows(nrows), ncols(ncols), nmats(nmats), layout(layout) {

                assert(nrows * ncols * nmats == (IndexType)m.size());

                // Check the density of nonzeros
                double density = nnz_density(m);

                // If the density is larger than 0.3, treat it as dense
                if (nrows == 0 || ncols == 0 || nmats == 0 || density >= 0.3) {
                    vals = m;
                    return;
                }

                // Otherwise extract the nonzero and detect shortcuts
                ii = Indices<Cpu>(nrows * nmats + 1, Cpu{});
                ii[0] = 0;
                jj = Indices<Cpu>(std::ceil(m.size() * 0.4), Cpu{});
                vals = vector<T, Cpu>(jj.size(), Cpu{});
                type = std::vector<Type>(nmats, Dense);
                scalar = std::vector<T>(nmats, T{1});
                perm = std::vector<std::vector<IndexType>>(nmats, std::vector<IndexType>(nrows));
                scalars = std::vector<std::vector<T>>(nmats, std::vector<T>(nrows));
                for (IndexType imat = 0; imat < nmats; ++imat) {
                    T s{0};
                    for (IndexType ij = 0; ij < nrows * ncols; ++ij)
                        if (std::norm(m[ncols * nrows * imat + ij]) > std::norm(s))
                            s = m[ncols * nrows * imat + ij];
                    scalar[imat] = s;
                }
                for (IndexType imat = 0; imat < nmats; ++imat) {
                    bool is_identity = (nrows == ncols);
                    bool is_all_ones = true;
                    bool is_perm = true;
                    for (IndexType i = 0; i < nrows; ++i) {
                        IndexType idx = ii[imat * nrows + i], idx0 = idx;
                        for (IndexType j = 0; j < ncols; ++j) {
                            T mij = m[ncols * nrows * imat +
                                      (layout == ColumnMajor ? i + j * nrows : i * ncols + j)];
                            if (mij == T{0} || mij == -T{0}) continue;
                            if (mij != scalar[imat])
                                is_all_ones = is_identity = false;
                            else if (i != j)
                                is_identity = false;
                            else if (idx != idx0)
                                is_perm = false;
                            jj[idx] = j;
                            scalars[imat][i] = vals[idx] = mij;
                            perm[imat][i] = j;
                            ++idx;
                        }
                        if (idx == idx0) is_identity = is_perm = false;
                        ii[imat * nrows + i + 1] = idx;
                    }
                    if (is_identity) {
                        if (nrows != ncols) is_identity = false;
                        for (IndexType i = 0; i < nrows; ++i) {
                            T mij = m[ncols * nrows * imat +
                                      (layout == ColumnMajor ? i + i * nrows : i * ncols + i)];
                            if (mij != T{1}) is_identity = false;
                        }
                    }
                    type[imat] = (is_identity ? Identity
                                              : (is_perm ? Permutation
                                                         : (is_all_ones ? SparseAllOnes : Sparse)));
                }
            }

            /// Return whether the matrices are treat as sparse matrices

            bool is_sparse() const { return type.size() > 0; }

            /// Return whether the imat-th matrices is treated as an identity matrix
            /// \param imat: matrix index to inspect

            bool is_identity(IndexType imat) const {
                return is_sparse() ? type[imat] == Identity : false;
            }

            /// Return whether the imat-th matrices is treated as a permutation matrix
            /// \param imat: matrix index to inspect

            bool is_permutation(IndexType imat) const {
                return is_sparse() ? type[imat] == Permutation : false;
            }

            /// Return whether the imat-th matrices is treated as as sparse matrix with all nonzero elements being ones
            /// \param imat: matrix index to inspect

            bool is_all_ones(IndexType imat) const {
                return is_sparse() ? type[imat] == Identity || type[imat] == SparseAllOnes : false;
            }

            /// Return the average number of nonzeros for each matrix

            double avg_nnz_per_mat() const {
                return is_sparse() ? (double)ii[nrows * nmats] / nmats : nrows * ncols;
            }
        };

        /// CSR-dense matrix multiplication

        template <typename IndexType, typename T>
        inline void xgemm_csr_mat(T alpha, const CSRs<IndexType, T> &a, int ai,
                                  const T *SB_RESTRICT b, IndexType bm, IndexType bn,
                                  MatrixLayout blayout, IndexType ldb, T beta, T *SB_RESTRICT c,
                                  IndexType ldc) {
            assert(a.ncols == bm);
            assert(ai < a.nmats);
            if (a.nrows == 0 || bn == 0) return;

            if (!a.is_sparse()) {
                xgemm(a.layout == ColumnMajor ? 'N' : 'T', blayout == ColumnMajor ? 'N' : 'T',
                      a.nrows, bn, bm, alpha, &a.vals[a.nrows * a.ncols * ai],
                      a.layout == ColumnMajor ? a.nrows : a.ncols, b, ldb, beta, c, ldc, Cpu{});
                return;
            }

            if (beta != T{0} || blayout != ColumnMajor || alpha != T{1})
                throw std::runtime_error("wtf");

            using Tc = typename ccomplex<T>::type;
            const Tc *SB_RESTRICT bc = (const Tc *)b;
            Tc *SB_RESTRICT cc = (Tc *)c;
            const Tc alphac = *(const Tc *)&alpha;
            bool a_is_all_ones = a.is_all_ones(ai);

            if (a.is_identity(ai)) {
                for (IndexType j = 0; j < bn; ++j)
                    for (IndexType i = 0; i < bm; ++i) c[i + ldc * j] = b[i + ldb * j];
            } else if (a.is_permutation(ai)) {
                const Tc *SB_RESTRICT scalars = (const Tc *)a.scalars[ai].data();
                for (IndexType i = 0, m = a.nrows; i < m; ++i) {
                    IndexType s = a.perm[ai][i];
                    Tc a_is = (a_is_all_ones ? alphac : alphac * scalars[i]);
                    for (IndexType j = 0; j < bn; ++j) cc[i + ldc * j] = a_is * bc[s + ldb * j];
                }
            } else {
                const Tc *SB_RESTRICT valsc = (const Tc *)a.vals.data();
                for (IndexType i = 0, m = a.nrows; i < m; ++i) {
                    bool first = true;
                    for (IndexType jidx = a.ii[m * ai + i], jidx1 = a.ii[m * ai + i + 1];
                         jidx < jidx1; ++jidx) {
                        IndexType s = a.jj[jidx];
                        Tc a_is = (a_is_all_ones ? alphac : alphac * valsc[jidx]);
                        if (first) {
                            for (IndexType j = 0; j < bn; ++j)
                                cc[i + ldc * j] = a_is * bc[s + ldb * j];
                            first = false;
                        } else {
                            for (IndexType j = 0; j < bn; ++j)
                                cc[i + ldc * j] += a_is * bc[s + ldb * j];
                        }
                    }
                    if (first)
                        for (IndexType j = 0; j < bn; ++j) cc[i + ldc * j] = 0;
                }
            }
        }

        ///
        /// Implementation of operations for each platform
        ///

        /// Constrains on layout of the input and output dense tensors for a
        /// sparse-dense tensor contraction

        enum SpMMAllowedLayout {
            SameLayoutForXAndY, ///< X and Y should have the same layout, either row-major or column-major
            ColumnMajorForY,     ///< X can be either way but Y should be column-major
            ColumnMajorForXandY, ///< X and Y should be column-major
            RowMajorForXandY,    ///< X and Y should be row-major
            AnyLayoutForXAndY    ///< X and Y can be either way
        };

        /// Flavor of the jj values, used by get_bsr_indices

        enum ReturnJJ {
            ReturnJJNoBlocking, ///< return jj with no implicit blocking
            ReturnJJBlocked, ///< return jj implicitly blocked without considering the kronecker blocking
            ReturnJJBlockedWithKron, ///< return jj implicitly blocked considering the kronecker blocking
        };

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU> struct BSR;

/// NOTE: disable MKL implementation in favor of the inhouse one
#if 0 && defined(SUPERBBLAS_USE_MKL)
        inline void checkMKLSparse(sparse_status_t status) {
            static std::map<sparse_status_t, std::string> statuses = {
                {SPARSE_STATUS_NOT_INITIALIZED, "SPARSE_STATUS_NOT_INITIALIZED"},
                {SPARSE_STATUS_ALLOC_FAILED, "SPARSE_STATUS_ALLOC_FAILED"},
                {SPARSE_STATUS_INVALID_VALUE, "SPARSE_STATUS_INVALID_VALUE"},
                {SPARSE_STATUS_EXECUTION_FAILED, "SPARSE_STATUS_EXECUTION_FAILED"},
                {SPARSE_STATUS_INTERNAL_ERROR, "SPARSE_STATUS_INTERNAL_ERROR"},
                {SPARSE_STATUS_NOT_SUPPORTED, "SPARSE_STATUS_NOT_SUPPORTED"}};

            if (status != SPARSE_STATUS_SUCCESS) {
                std::stringstream ss;
                ss << "MKL sparse function returned error " << statuses[status];
                throw std::runtime_error(ss.str());
            }
        }

        template <std::size_t Nd, std::size_t Ni, typename T> struct BSR<Nd, Ni, T, Cpu> {
            BSRComponent<Nd, Ni, T, Cpu> v;     ///< BSR general information
            vector<MKL_INT, Cpu> ii, jj;        ///< BSR row and column nonzero indices
            std::shared_ptr<sparse_matrix_t> A; ///< MKL BSR descriptor
            unsigned int num_nnz_per_row;       ///< Number of nnz per row (for Kronecker BSR)

            static const SpMMAllowedLayout allowLayout = SameLayoutForXAndY;
            static const MatrixLayout preferredLayout = RowMajor;
            std::string implementation() const {
                return (volume(v.krond) > 1 || volume(v.kroni) > 1) ? "mkl_kron_bsr" : "mkl_bsr";
            }

            BSR(const BSRComponent<Nd, Ni, T, Cpu> &v) : v(v) {
                if (volume(v.dimi) == 0 || volume(v.dimd) == 0) return;
                if (volume(v.blocki) != volume(v.blockd))
                    throw std::runtime_error("MKL Sparse does not support non-square blocks");
                if (v.blockImFast)
                    throw std::runtime_error("MKL Sparse does not support column major as the "
                                             "nonzero BSR blocks layout");
                IndexType block_size = volume(v.blocki);
                IndexType ki = volume(v.kroni);
                IndexType kd = volume(v.krond);
                IndexType block_rows = volume(v.dimi) / block_size / ki;
                IndexType block_cols = volume(v.dimd) / block_size / kd;
                bool is_kron = v.kron_it.size() > 0;
                auto bsr = !is_kron ? get_bsr_indices(v, true) : get_kron_indices(v, true);
                ii = bsr.i;
                jj = bsr.j;
                num_nnz_per_row = bsr.num_nnz_per_row;
                if (bsr.j_has_negative_indices)
                    throw std::runtime_error("bsr: unsupported -1 column indices when using MKL");
                A = std::shared_ptr<sparse_matrix_t>(new sparse_matrix_t, [=](sparse_matrix_t *A) {
                    checkMKLSparse(mkl_sparse_destroy(*A));
                    delete A;
                });
                checkMKLSparse(mkl_sparse_create_bsr(
                    &*A, SPARSE_INDEX_BASE_ZERO,
                    v.blockImFast ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR,
                    block_rows, block_cols * (is_kron ? bsr.num_nnz_per_row : 1), block_size,
                    ii.data(), ii.data() + 1, jj.data(), v.it.data()));
                checkMKLSparse(mkl_sparse_set_mm_hint(
                    *A, SPARSE_OPERATION_NON_TRANSPOSE,
                    (struct matrix_descr){.type = SPARSE_MATRIX_TYPE_GENERAL,
                                          .mode = SPARSE_FILL_MODE_LOWER /* Not used */,
                                          .diag = SPARSE_DIAG_NON_UNIT},
                    SPARSE_LAYOUT_ROW_MAJOR, 100, 1000));
            }

            /// Return the number of flops for a given number of right-hand-sides
            /// \param rhs: number of vectors to multiply

            double getFlopsPerMatvec(int rhs, MatrixLayout) const {
                double b = (double)volume(v.blocki);
                double ki = (double)volume(v.kroni), kd = (double)volume(v.krond);

                // For the Kronecker variant, each operator nonzero block will involve the contraction
                // with the kronecker block (ki*kd*b*rhs flops) and with the rest (b*b*ki*rhs flops)
                if (v.kron_it.size() > 0)
                    return (ki * b * b + kd * ki * b) * jj.size() * rhs *
                           multiplication_cost<T>::value;

                // For the regular variant, each operator nonzero block will involve the contraction
                // of the block will all the rhs (b*b*rhs flops)
                return b * b * jj.size() * rhs * multiplication_cost<T>::value;
            }

            /// Return the number of memory operations for a given number of right-hand-sides
            /// \param rhs: number of vectors to multiply

            double getMemopsPerMatvec(int rhs, MatrixLayout) const {
                double b = (double)volume(v.blocki);
                double ki = (double)volume(v.kroni), kd = (double)volume(v.krond);

                // For the Kronecker variant, each operator nonzero block will involve reading the
                // input vectors and the kronecker blocks and writing all combinations, plus
                // reading the nonzero regular blocks and the input right-hand-size for each nonzero block
                // and writing the output vectors
                if (v.kron_it.size() > 0)
                    return sizeof(T) * (
                                           // reading the input vecs and kronecker contr.
                                           volume(v.dimd) * (num_nnz_per_row + 1) * rhs +
                                           // reading the kronecker elements
                                           ki * kd * num_nnz_per_row +
                                           // reading regular elements and the rhs
                                           (b * b + b * rhs * ki) * jj.size() +
                                           // writing the output
                                           volume(v.dimi) * rhs);

                // For the regular variant, each operator nonzero block will involve reading the
                // nonzero block and the input right-hand-size, plus writing the output vectors
                return (volume(v.dimi) * rhs + (b * b + b * rhs) * jj.size()) * sizeof(T);
            }

            void operator()(bool conjA, const vector<T, Cpu> &vx, IndexType ldx, MatrixLayout lx,
                            vector<T, Cpu> &vy, IndexType ldy, MatrixLayout ly,
                            IndexType ncols) const {
                const T alpha{1};
                if (lx != ly) throw std::runtime_error("Unsupported operation with MKL");
                IndexType block_size = volume(v.blocki);
                IndexType ki = volume(v.kroni);
                IndexType kd = volume(v.krond);
                IndexType block_cols = volume(v.dimd) / block_size / kd;
                IndexType block_rows = volume(v.dimi) / block_size / ki;
                const T *x = vx.data();
                T *y = vy.data();
                bool is_kron = v.kron_it.size() > 0;
                const T beta{0};
                xscal(volume(v.dimi) * ncols, beta, y, 1, Cpu{});
                if (!is_kron) {
                    checkMKLSparse(mkl_sparse_mm(
                        !conjA ? SPARSE_OPERATION_NON_TRANSPOSE
                               : SPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                        alpha, *A,
                        (struct matrix_descr){.type = SPARSE_MATRIX_TYPE_GENERAL,
                                              .mode = SPARSE_FILL_MODE_LOWER /* Not used */,
                                              .diag = SPARSE_DIAG_NON_UNIT},
                        lx == ColumnMajor ? SPARSE_LAYOUT_COLUMN_MAJOR : SPARSE_LAYOUT_ROW_MAJOR, x,
                        ncols, ldx, beta, y, ldy));
                } else if (lx == RowMajor) {
                    if (conjA) throw std::runtime_error("kron BSR: unsupported conjugation");

                    // Contract the Kronecker part: for each direction mu do:
                    //  (ki,kd)[mu] x (kd,ncols,bd,rows) -> (ki,ncols,bd,rows,mu)
                    vector<T, Cpu> aux(ki * ncols * block_size * block_cols * num_nnz_per_row,
                                       Cpu{});
                    zero_n(aux.data(), aux.size(), Cpu{});
                    const bool tb = !v.blockImFast;
                    for (unsigned int i = 0; i < num_nnz_per_row; ++i)
                        xgemm(!tb ? 'N' : 'T', 'N', ki, ncols * block_size * block_cols, kd, alpha,
                              v.kron_it.data() + ki * kd * i, !tb ? ki : kd, x, kd, T{0},
                              aux.data() + ki * ncols * block_size * block_cols * i, ki, Cpu{});

                    // Contract the block part:
                    // \sum_i (bi,bd)[i,col,mu] x (ki,ncols,bd,rows,mu)[rows=col,mu] -> (ki,ncols,bi,rows)
                    checkMKLSparse(mkl_sparse_mm(
                        SPARSE_OPERATION_NON_TRANSPOSE, T{1}, *A,
                        (struct matrix_descr){.type = SPARSE_MATRIX_TYPE_GENERAL,
                                              .mode = SPARSE_FILL_MODE_LOWER /* Not used */,
                                              .diag = SPARSE_DIAG_NON_UNIT},
                        SPARSE_LAYOUT_ROW_MAJOR, aux.data(), ki * ncols, ki * ncols, beta, y,
                        ki * ncols));
                } else {
                    if (conjA) throw std::runtime_error("kron BSR: unsupported conjugation");

                    // Contract the Kronecker part: for each direction mu do:
                    //  (bd,rows,ncols,kd) x (ki,kd)[mu] -> (bd,rows,mu,ncols,ki)
                    vector<T, Cpu> aux(block_size * block_cols * num_nnz_per_row * ncols * ki,
                                       Cpu{});
                    zero_n(aux.data(), aux.size(), Cpu{});
                    const bool tb = !v.blockImFast;
                    for (unsigned int ij = 0; ij < num_nnz_per_row * ncols; ++ij) {
                        unsigned int i = ij % num_nnz_per_row;
                        unsigned int j = ij / num_nnz_per_row;
                        xgemm('N', !tb ? 'T' : 'N', block_size * block_cols, ki, kd, alpha,
                              x + j * block_size * block_cols, block_size * block_cols * ncols,
                              v.kron_it.data() + ki * kd * i, !tb ? ki : kd, T{0},
                              aux.data() + block_size * block_cols * i +
                                  block_size * block_cols * num_nnz_per_row * j,
                              block_size * block_cols * num_nnz_per_row * ncols, Cpu{});
                    }

                    // Contract the block part:
                    // \sum_i (bi,bd)[i,col,mu] x (bd,rows,mu,ncols,ki)[rows=col,mu] -> (bi,rows,ncols,ki)
                    checkMKLSparse(mkl_sparse_mm(
                        SPARSE_OPERATION_NON_TRANSPOSE, T{1}, *A,
                        (struct matrix_descr){.type = SPARSE_MATRIX_TYPE_GENERAL,
                                              .mode = SPARSE_FILL_MODE_LOWER /* Not used */,
                                              .diag = SPARSE_DIAG_NON_UNIT},
                        SPARSE_LAYOUT_COLUMN_MAJOR, aux.data(), ncols * ki,
                        block_size * block_cols * num_nnz_per_row, beta, y,
                        block_size * block_rows));
                }
            }

            ~BSR() {}
        };
#else

        template <std::size_t Nd, std::size_t Ni, typename T> struct BSR<Nd, Ni, T, Cpu> {
            BSRComponent<Nd, Ni, T, Cpu> v; ///< BSR general information
            vector<IndexType, Cpu> ii, jj;  ///< BSR row and column nonzero indices
            static std::string implementation() { return "builtin_cpu"; }
            unsigned int num_nnz_per_row; ///< Number of nnz per row (for Kronecker BSR)
            CSRs<IndexType, T> kron;      ///< kron sparse representation

            SpMMAllowedLayout allowLayout;
            static const MatrixLayout preferredLayout = RowMajor;

            BSR(const BSRComponent<Nd, Ni, T, Cpu> &v) : v(v) {
                allowLayout = (v.kron_it.size() > 0) ? SameLayoutForXAndY : AnyLayoutForXAndY;
                if (volume(v.dimi) == 0 || volume(v.dimd) == 0) return;
                auto bsr = get_bsr_indices(v); // column indices aren't blocked
                ii = bsr.i;
                jj = bsr.j;
                num_nnz_per_row = bsr.num_nnz_per_row;
                if (v.kron_it.size() > 0)
                    kron =
                        CSRs<IndexType, T>(v.kron_it, volume(v.kroni), volume(v.krond),
                                           num_nnz_per_row, v.blockImFast ? ColumnMajor : RowMajor);
            }

            /// Return the number of flops for a given number of right-hand-sides
            /// \param rhs: number of vectors to multiply

            double getFlopsPerMatvec(int rhs, MatrixLayout layout) const {
                double bi = (double)volume(v.blocki), bd = (double)volume(v.blockd);
                double ki = (double)volume(v.kroni);

                // For the Kronecker variant, each operator nonzero block will involve the contraction
                // with the kronecker block (ki*kd*bd*rhs flops) and with the rest (bi*bd*ki*rhs flops)
                if (v.kron_it.size() > 0)
                    return multiplication_cost<T>::value * rhs *
                           (layout == RowMajor
                                ? (ki * bi * bd + kron.avg_nnz_per_mat() * bd) * jj.size()
                                : ki * volume(v.dimd) * num_nnz_per_row + ki * bi * bd * jj.size());

                // For the regular variant, each operator nonzero block will involve the contraction
                // of the block will all the rhs (bi*bd*rhs flops)
                return bi * bd * jj.size() * rhs * multiplication_cost<T>::value;
            }

            /// Return the number of memory operations for a given number of right-hand-sides
            /// \param rhs: number of vectors to multiply
            /// \param layout: input/output vector layout

            double getMemopsPerMatvec(int rhs, MatrixLayout layout) const {
                double bi = (double)volume(v.blocki), bd = (double)volume(v.blockd);
                double ki = (double)volume(v.kroni), kd = (double)volume(v.krond);

                // For the Kronecker variant, each operator nonzero block will involve reading the
                // input vectors and the kronecker blocks and the nonzero regular blocks,
                // and writing the output vectors
                if (v.kron_it.size() > 0)
                    return sizeof(T) *
                           (layout == RowMajor
                                ? (
                                      // reading the input vecs and kronecker element and regular block elements
                                      (bd * kd * rhs + kron.avg_nnz_per_mat() + bi * bd) *
                                          jj.size() +
                                      // writing the output
                                      volume(v.dimi) * rhs)
                                : (
                                      // contracting the input vectors and the kronecker elements
                                      ki * kd * num_nnz_per_row +
                                      volume(v.dimd) * (num_nnz_per_row + 1) * rhs +
                                      // contracting with the regular block elements
                                      (bi * bd + bd * rhs * ki) * jj.size() +
                                      // writing the output
                                      volume(v.dimi) * rhs));

                // For the regular variant, each operator nonzero block will involve reading the
                // nonzero block and the input right-hand-size, plus writing the output vectors
                return (volume(v.dimi) * rhs + (bi * bd + bd * rhs) * jj.size()) * sizeof(T);
            }

            void operator()(bool conjA, const vector<T, Cpu> &vx, IndexType ldx, MatrixLayout lx,
                            vector<T, Cpu> &vy, IndexType ldy, MatrixLayout ly,
                            IndexType ncols) const {
                const T alpha{1};
                if (conjA) throw std::runtime_error("Not implemented");
                if (v.kron_it.size() > 0 && lx != ly) throw std::runtime_error("Not implemented");
                IndexType bi = volume(v.blocki);
                IndexType bd = volume(v.blockd);
                IndexType ki = volume(v.kroni);
                IndexType kd = volume(v.krond);
                IndexType block_cols = volume(v.dimd) / bd / kd;
                IndexType block_rows = volume(v.dimi) / bi / ki;
                const T *x = vx.data();
                T *y = vy.data();
                const T beta{0};
                xscal(volume(v.dimi) * ncols, beta, y, 1, Cpu{});
                T *nonzeros = v.it.data();
                const bool tx = lx == RowMajor;
                const bool ty = ly == RowMajor;
                const bool tb = !v.blockImFast;
                const IndexType xs = lx == ColumnMajor ? 1 : ldx;
                if (v.kron_it.size() == 0) {
                    if (ncols > 1) {
#    if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#        pragma omp parallel for schedule(static)
#    endif
                        for (IndexType i = 0; i < block_rows; ++i) {
                            for (IndexType j = ii[i], j1 = ii[i + 1]; j < j1; ++j) {
                                if (jj[j] == -1) continue;
                                if (ly == ColumnMajor)
                                    xgemm(tb ? 'T' : 'N', tx ? 'T' : 'N', bi, ncols, bd, alpha,
                                          nonzeros + j * bi * bd, tb ? bd : bi, x + jj[j] * xs, ldx,
                                          T{1}, y + i * bi, ldy, Cpu{});
                                else
                                    xgemm(!tx ? 'T' : 'N', !tb ? 'T' : 'N', ncols, bi, bd, alpha,
                                          x + jj[j] * xs, ldx, nonzeros + j * bi * bd, tb ? bd : bi,
                                          T{1}, y + i * bi * ldy, ldy, Cpu{});
                            }
                        }
                    } else {
#    if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#        pragma omp parallel for schedule(static)
#    endif
                        for (IndexType i = 0; i < block_rows; ++i) {
                            for (IndexType j = ii[i], j1 = ii[i + 1]; j < j1; ++j) {
                                if (jj[j] == -1) continue;
                                xgemv(tb ? 'T' : 'N', tb ? bd : bi, tb ? bi : bd, alpha,
                                      nonzeros + j * bi * bd, tb ? bd : bi, x + jj[j] * xs,
                                      tx ? ldx : 1, T{1}, y + i * bi, ty ? ldy : 1, Cpu{});
                            }
                        }
                    }
                } else {
                    // With Kronecker product
                    if (lx == RowMajor) {
#    ifdef _OPENMP
#        pragma omp parallel
#    endif
                        {
                            std::vector<T> aux(ki * ncols * bd);
#    if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#        pragma omp for schedule(static)
#    endif
                            for (IndexType i = 0; i < block_rows; ++i) {
                                for (IndexType j = ii[i], j1 = ii[i + 1], j0 = 0; j < j1;
                                     ++j, ++j0) {
                                    if (jj[j] == -1) continue;
                                    // Contract with the blocking:  (ki,kd) x (kd,n,bd,rows) -> (ki,n,bd) ; note (fast,slow)
                                    xgemm_csr_mat(T{1}, kron, j0, x + jj[j] * ncols, kd, ncols * bd,
                                                  ColumnMajor, kd, T{0}, aux.data(), ki);
                                    // Contract with the Kronecker blocking: (ki,n,bd) x (bi,bd)[rows,mu] -> (ki,n,bi) ; note (fast,slow)
                                    xgemm_alt_alpha1_beta1(
                                        'N', !tb ? 'T' : 'N', ki * ncols, bi, bd, aux.data(),
                                        ki * ncols, nonzeros + j * bi * bd, !tb ? bi : bd,
                                        y + i * ki * ncols * bi, ki * ncols, Cpu{});
                                }
                            }
                        }
                    } else {
                        // Contract with the blocking: (bd,rows,n,kd) x (ki,kd)[mu] -> (bd,rows,n,ki,mu) ; note (fast,slow)
                        vector<T, Cpu> aux(bd * block_cols * ncols * ki * num_nnz_per_row, Cpu{});
                        zero_n(aux.data(), aux.size(), aux.ctx());
#    if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#        pragma omp parallel for schedule(static)
#    endif
                        for (unsigned int i = 0; i < num_nnz_per_row; ++i)
                            xgemm('N', !tb ? 'T' : 'N', bd * block_cols * ncols, ki, kd, alpha, x,
                                  bd * block_cols * ncols, v.kron_it.data() + ki * kd * i,
                                  !tb ? ki : kd, T{0},
                                  aux.data() + bd * block_cols * ncols * ki * i,
                                  bd * block_cols * ncols, Cpu{});
#    if SUPERBBLAS_USE_OPENMP_WITH_BLAS
#        pragma omp parallel for schedule(static)
#    endif
                        for (IndexType i = 0; i < block_rows; ++i) {
                            for (IndexType j = ii[i], j1 = ii[i + 1], j0 = 0; j < j1; ++j, ++j0) {
                                if (jj[j] == -1) continue;
                                // Contract with the Kronecker blocking: (bi,bd) x (bd,n,ki)[rows,mu] -> (bi,n,ki) ; note (fast,slow)
                                // Note that jj is (bd,kd,rows) but aux is (bd,rows,n,ki,mu), so jj/kd is the right shift on aux
                                xgemm(!tb ? 'N' : 'T', 'N', bi, ncols * ki, bd, T{1},
                                      nonzeros + j * bi * bd, !tb ? bi : bd,
                                      aux.data() + jj[j] / kd + bd * block_cols * ncols * ki * j0,
                                      bd * block_cols, T{1}, y + bi * i, bi * block_rows, Cpu{});
                            }
                        }
                    }
                }
            }

            ~BSR() {}
        };
#endif

#ifdef SUPERBBLAS_USE_GPU
        template <std::size_t Nd, std::size_t Ni, typename T> struct BSR<Nd, Ni, T, Gpu> {
            BSRComponent<Nd, Ni, T, Gpu> v; ///< BSR general information
            vector<IndexType, Gpu> ii, jj;  ///< BSR row and column nonzero indices
            bool
                kron_use_crafted_kernel; ///< whether to use a crafted kernel instead of cu/hipsparse
            vector<int, Gpu> kron_perm;  ///< represent the kron matrices with a permutation
            vector<T, Gpu> kron_scalars; ///< represent the kron matrices with scalars
#    ifdef SUPERBBLAS_USE_CUDA
            std::shared_ptr<cusparseMatDescr_t>
                descrA_bsr; ///< cuSparse descriptor for BSR matrices
            std::shared_ptr<cusparseSpMatDescr_t>
                descrA_other; ///< cuSparse descriptor for CSR and ELL matrices
            enum SparseFormat{FORMAT_BSR, FORMAT_CSR, FORMAT_ELL} spFormat; ///< the sparse format
#    else
            std::shared_ptr<hipsparseMatDescr_t> descrA_bsr; ///< hipSparse descriptor
#    endif
            unsigned int num_nnz_per_row;   ///< Number of nnz per row (for Kronecker BSR)
            vector<T, Cpu> kron_cpu;        ///< Host version of v.kron
            double kron_nnz_density;        ///< kron_cpu nonzero density
            std::vector<int> kron_disp;     ///< unique index for kron
            std::vector<int> kron_disp_rev; ///< kron_disp index for each unique kron matrix

            SpMMAllowedLayout allowLayout;
            MatrixLayout preferredLayout;
            std::string implementation_;
            const std::string &implementation() const { return implementation_; }

            BSR(BSRComponent<Nd, Ni, T, Gpu> v) : v(v) {
                if (deviceId(v.it.ctx()) == CPU_DEVICE_ID)
                    throw std::runtime_error("BSR: unsupported a cpu device");
                setDevice(deviceId(v.it.ctx()));
                allowLayout = ColumnMajorForY; // Default setting for empty tensor
                preferredLayout = ColumnMajor; // Default setting for empty tensor
                if (volume(v.dimi) == 0 || volume(v.dimd) == 0) return;
                if (volume(v.blocki) != volume(v.blockd))
                    throw std::runtime_error("cuSPARSE does not support non-square blocks");
                bool is_kron = v.kron_it.size() > 0;

                // Check whether to use the crafted kernels
#    ifdef SUPERBBLAS_USE_GPU
                if (is_kron) {
                    kron_use_crafted_kernel = true;
                    std::size_t ki = volume(v.kroni);
                    std::size_t kd = volume(v.krond);
                    std::size_t block_size = volume(v.blocki);
                    // Check that the kronecker block is 4 and the block is 3
                    if (ki != kd || ki != 4 || block_size != volume(v.blockd) || block_size != 3 ||
                        !available_bsr_kron_3x3_4x4perm<T>(v.it.ctx()))
                        kron_use_crafted_kernel = false;
                    // Check that the kronecker blocks can be represented with a permutation
                    auto kron_cpu = makeSure(v.kron_it, Cpu{});
                    std::size_t num_blocks = kron_cpu.size() / ki / kd;
                    vector<int, Cpu> kron_perm_cpu(kd * num_blocks, Cpu{});
                    vector<T, Cpu> kron_scalars_cpu(kd * num_blocks, Cpu{});
                    for (int i = 0; i < kd * num_blocks; ++i) kron_perm_cpu[i] = 0;
                    for (int i = 0; i < kd * num_blocks; ++i) kron_scalars_cpu[i] = T{0};
                    int ldr = (v.blockImFast ? 1 : kd);
                    int ldc = (v.blockImFast ? ki : 1);
                    for (std::size_t blk = 0; blk < num_blocks; blk++) {
                        for (std::size_t i = 0; i < ki; i++) {
                            bool is_perm = true, is_first_nnz = true;
                            for (std::size_t j = 0; j < kd; j++) {
                                T val = kron_cpu[ki * kd * blk + i * ldr + j * ldc];
                                if (std::norm(val) > 0) {
                                    if (is_first_nnz) {
                                        is_perm = true;
                                        is_first_nnz = false;
                                        kron_perm_cpu[kd * blk + i] = j;
                                        kron_scalars_cpu[kd * blk + i] = val;
                                    } else {
                                        is_perm = false;
                                    }
                                }
                            }
                            if (!is_perm) kron_use_crafted_kernel = false;
                        }
                    }
                    if (kron_use_crafted_kernel) {
                        kron_perm = makeSure(kron_perm_cpu, v.it.ctx());
                        kron_scalars = makeSure(kron_scalars_cpu, v.it.ctx());
                    }
                } else
#    endif
                {
                    kron_use_crafted_kernel = false;
                }

                // Analyze the density of kron
                if (is_kron && !kron_use_crafted_kernel) {
                    kron_cpu = makeSure(v.kron_it, Cpu{});
                    std::size_t ki = volume(v.kroni);
                    std::size_t kd = volume(v.krond);
                    std::size_t num_nnz_per_row = kron_cpu.size() / ki / kd;
                    kron_nnz_density = nnz_density(kron_cpu);

                    kron_disp = std::vector<int>(num_nnz_per_row, 0);
                    for (std::size_t i = 0; i < num_nnz_per_row; i++) {
                        kron_disp[i] = kron_disp_rev.size();
                        kron_disp_rev.push_back(i);
                        for (std::size_t j = 0; j < i; j++) {
                            bool same = true;
                            for (std::size_t k = 0; k < ki * kd; k++) {
                                if (kron_cpu[i * ki * kd + k] != kron_cpu[j * ki * kd + k]) {
                                    same = false;
                                    break;
                                }
                            }
                            if (same) {
                                kron_disp[i] = kron_disp[j];
                                kron_disp_rev.pop_back();
                                break;
                            }
                        }
                    }
                } else {
                    kron_nnz_density = 0;
                }

                // Get the nonzero pattern
                auto bsr = !is_kron ? get_bsr_indices(v, ReturnJJBlocked)
                                    : (kron_use_crafted_kernel
                                           ? get_bsr_indices(v, ReturnJJBlockedWithKron)
                                           : get_kron_indices(v, true, kron_disp));
                ii = bsr.i;
                jj = bsr.j;
                num_nnz_per_row = bsr.num_nnz_per_row;

#    ifdef SUPERBBLAS_USE_CUDA
                if (!kron_use_crafted_kernel) {
                    IndexType block_size = volume(v.blocki);
                    cudaDeviceProp prop;
                    gpuCheck(cudaGetDeviceProperties(&prop, deviceId(v.it.ctx())));
                    /// TODO: ELL format is disable, it isn't correct currently
                    if (false && bsr.num_nnz_per_row >= 0 && !is_complex<T>::value &&
                        ((std::is_same<T, float>::value && prop.major >= 8) ||
                         (std::is_same<T, double>::value && prop.major >= 8))) {
                        spFormat = FORMAT_ELL;
                    } else if (!bsr.j_has_negative_indices) {
                        spFormat = block_size == 1 ? FORMAT_CSR : FORMAT_BSR;
                    } else {
                        throw std::runtime_error("bsr: unsupported -1 column indices when using "
                                                 "cuSPARSE and not using ELL");
                    }

                    if (spFormat == FORMAT_BSR) {
                        implementation_ = "cusparse_bsr";
                        allowLayout = ColumnMajorForY;
                        preferredLayout = ColumnMajor;
                        descrA_bsr = std::shared_ptr<cusparseMatDescr_t>(
                            new cusparseMatDescr_t, [](cusparseMatDescr_t *p) {
                                cusparseDestroyMatDescr(*p);
                                delete p;
                            });
                        gpuSparseCheck(cusparseCreateMatDescr(&*descrA_bsr));
                        gpuSparseCheck(
                            cusparseSetMatIndexBase(*descrA_bsr, CUSPARSE_INDEX_BASE_ZERO));
                        gpuSparseCheck(
                            cusparseSetMatType(*descrA_bsr, CUSPARSE_MATRIX_TYPE_GENERAL));
                    } else {
                        static_assert(sizeof(IndexType) == 4,
                                      "unsupported integer other than 32 bits");
                        IndexType num_cols = volume(v.dimd);
                        IndexType num_rows = volume(v.dimi);
                        IndexType ki = volume(v.kroni);
                        IndexType kd = volume(v.krond);
                        descrA_other = std::shared_ptr<cusparseSpMatDescr_t>(
                            new cusparseSpMatDescr_t, [](cusparseSpMatDescr_t *p) {
                                cusparseDestroySpMat(*p);
                                delete p;
                            });
                        allowLayout = SameLayoutForXAndY;
                        preferredLayout = RowMajor;
                        if (spFormat == FORMAT_CSR) {
                            implementation_ = "cusparse_csr";
                            gpuSparseCheck(cusparseCreateCsr(
                                &*descrA_other, num_rows / ki,
                                num_cols / kd * (is_kron ? kron_disp_rev.size() : 1), bsr.nnz,
                                ii.data(), jj.data(), v.it.data(), CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, toCudaDataType<T>()));
                        } else {
                            implementation_ = "cusparse_ell";
                            gpuSparseCheck(cusparseCreateBlockedEll(
                                &*descrA_other, num_rows / ki,
                                num_cols / kd * (is_kron ? kron_disp_rev.size() : 1), block_size,
                                block_size * kron_disp_rev.size(), jj.data(), v.it.data(),
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, toCudaDataType<T>()));
                        }
                    }
                } else {
                    implementation_ = "cuda_mmfa";
                    allowLayout = RowMajorForXandY;
                    preferredLayout = RowMajor;
                }
#    else
                if (bsr.j_has_negative_indices)
                    throw std::runtime_error("bsr: unsupported -1 column indices when using "
                                             "hipSPARSE");

                if (!kron_use_crafted_kernel) {
                    implementation_ = "hipsparse_bsr";
                    allowLayout = ColumnMajorForY;
                    preferredLayout = !is_kron ? RowMajor : ColumnMajor;
                    descrA_bsr = std::shared_ptr<hipsparseMatDescr_t>(
                        new hipsparseMatDescr_t, [](hipsparseMatDescr_t *p) {
                            hipsparseDestroyMatDescr(*p);
                            delete p;
                        });
                    gpuSparseCheck(hipsparseCreateMatDescr(&*descrA_bsr));
                    gpuSparseCheck(
                        hipsparseSetMatIndexBase(*descrA_bsr, HIPSPARSE_INDEX_BASE_ZERO));
                    gpuSparseCheck(hipsparseSetMatType(*descrA_bsr, HIPSPARSE_MATRIX_TYPE_GENERAL));
                } else {
                    implementation_ = "rocm_mmfa";
                    allowLayout = RowMajorForXandY;
                    preferredLayout = RowMajor;
                }
#    endif
        }

        /// Return the number of flops for a given number of right-hand-sides
        /// \param rhs: number of vectors to multiply

        double
        getFlopsPerMatvec(int rhs, MatrixLayout layout) const {
            double b = (double)volume(v.blocki);
            double ki = (double)volume(v.kroni), kd = (double)volume(v.krond);

            // For the Kronecker variant, each operator nonzero block will involve the contraction
            // with the kronecker block (ki*kd*b*rhs flops) and with the rest (b*b*ki*rhs flops)
            if (v.kron_it.size() > 0)
                return (layout == RowMajor
                            ? (ki * b * b + kd * ki * b) * jj.size()
                            : kron_disp_rev.size() * ki * volume(v.dimd) + ki * b * b * jj.size()) *
                       rhs * multiplication_cost<T>::value;

            // For the regular variant, each operator nonzero block will involve the contraction
            // of the block will all the rhs (b*b*rhs flops)
            return b * b * jj.size() * rhs * multiplication_cost<T>::value;
        }

        /// Return the number of memory operations for a given number of right-hand-sides
        /// \param rhs: number of vectors to multiply

        double getMemopsPerMatvec(int rhs, MatrixLayout layout) const {
            double b = (double)volume(v.blocki);
            double ki = (double)volume(v.kroni), kd = (double)volume(v.krond);

            // For the Kronecker variant, each operator nonzero block will involve reading the
            // input vectors and the kronecker blocks and writing all combinations, plus
            // reading the nonzero regular blocks and the input right-hand-size for each nonzero block
            // and writing the output vectors
            std::size_t nnz_per_row_proccess =
                (layout == RowMajor ? num_nnz_per_row : kron_disp_rev.size());
            if (v.kron_it.size() > 0)
                return sizeof(T) * (
                                       // reading the input vecs and kronecker contr.
                                       volume(v.dimd) * (nnz_per_row_proccess + 1) * rhs +
                                       // reading the kronecker elements
                                       ki * kd * nnz_per_row_proccess +
                                       // reading regular elements and the rhs
                                       (b * b + b * rhs * ki) * jj.size() +
                                       // writing the output
                                       volume(v.dimi) * rhs);

            // For the regular variant, each operator nonzero block will involve reading the
            // nonzero block and the input right-hand-size, plus writing the output vectors
            return (volume(v.dimi) * rhs + (b * b + b * rhs) * jj.size()) * sizeof(T);
        }

    private:
        void matvec(T alpha, bool conjA, const T *x, IndexType ldx, MatrixLayout lx, T *y,
                    IndexType ldy, MatrixLayout ly, IndexType ncols, T beta = T{0}) const {
            // Check layout
            if ((allowLayout == SameLayoutForXAndY && lx != ly) ||
                ((allowLayout == ColumnMajorForY || allowLayout == ColumnMajorForXandY) &&
                 ly == RowMajor) ||
                (allowLayout == ColumnMajorForXandY && lx == RowMajor))
                throw std::runtime_error("BSR operator(): Unexpected layout");

            IndexType block_size = volume(v.blocki);
            IndexType num_cols = volume(v.dimd);
            IndexType num_rows = volume(v.dimi);
            IndexType ki = volume(v.kroni);
            IndexType kd = volume(v.krond);
            IndexType block_cols = num_cols / block_size / ki;
            IndexType block_rows = num_rows / block_size / kd;
            IndexType num_blocks = jj.size();
            bool is_kron = v.kron_it.size() > 0;

            auto gpuSparseHandle = getGpuSparseHandle(ii.ctx());
#    ifdef SUPERBBLAS_USE_CUDA
            if (spFormat == FORMAT_BSR) {
                gpuSparseCheck(cusparseXbsrmm(
                    getGpuSparseHandle(ii.ctx()),
                    v.blockImFast ? CUSPARSE_DIRECTION_COLUMN : CUSPARSE_DIRECTION_ROW,
                    !conjA ? CUSPARSE_OPERATION_NON_TRANSPOSE
                           : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                    lx == ColumnMajor ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                      : CUSPARSE_OPERATION_TRANSPOSE,
                    block_rows, ncols, block_cols * (is_kron ? kron_disp_rev.size() : 1),
                    num_blocks, alpha, *descrA_bsr, v.it.data(), ii.data(), jj.data(), block_size,
                    x, ldx, beta, y, ldy));
            } else {
                cusparseDnMatDescr_t matx, maty;
                cudaDataType cudaType = toCudaDataType<T>();
                gpuSparseCheck(cusparseCreateDnMat(
                    &matx, !conjA ? num_cols / kd * (is_kron ? kron_disp_rev.size() : 1) : num_rows,
                    ncols, ldx, (void *)x, cudaType,
                    lx == ColumnMajor ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));
                gpuSparseCheck(cusparseCreateDnMat(
                    &maty, !conjA ? num_rows / ki : num_cols, ncols, ldy, (void *)y, cudaType,
                    ly == ColumnMajor ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW));
                std::size_t bufferSize;
                gpuSparseCheck(cusparseSpMM_bufferSize(
                    gpuSparseHandle,
                    !conjA ? CUSPARSE_OPERATION_NON_TRANSPOSE
                           : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *descrA_other, matx, &beta, maty,
                    cudaType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
                vector<T, Gpu> buffer((bufferSize + sizeof(T) - 1) / sizeof(T), ii.ctx(),
                                      doCacheAlloc);
                gpuSparseCheck(cusparseSpMM(gpuSparseHandle,
                                            !conjA ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                                   : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *descrA_other,
                                            matx, &beta, maty, cudaType, CUSPARSE_SPMM_ALG_DEFAULT,
                                            buffer.data()));
                gpuSparseCheck(cusparseDestroyDnMat(matx));
                gpuSparseCheck(cusparseDestroyDnMat(maty));
            }
#    else
                gpuSparseCheck(hipsparseXbsrmm(
                    gpuSparseHandle,
                    v.blockImFast ? HIPSPARSE_DIRECTION_COLUMN : HIPSPARSE_DIRECTION_ROW,
                    !conjA ? HIPSPARSE_OPERATION_NON_TRANSPOSE
                           : HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                    lx == ColumnMajor ? HIPSPARSE_OPERATION_NON_TRANSPOSE
                                      : HIPSPARSE_OPERATION_TRANSPOSE,
                    block_rows, ncols, block_cols * (is_kron ? kron_disp_rev.size() : 1),
                    num_blocks, alpha, *descrA_bsr, v.it.data(), ii.data(), jj.data(), block_size,
                    x, ldx, beta, y, ldy));
#    endif
            }

            // Contract the Kronecker part: for each direction mu do:
            //  (bd,rows,ncols,kd) x (ki,kd)[mu] -> (bd,rows,mu,ncols,ki)

            void contract_kron_cols(T alpha, const vector<const T, Gpu> &x, int col0, int dcols,
                                    int ncols, vector<T, Gpu> &y) const {
                IndexType num_cols = volume(v.dimd);
                IndexType block_size = volume(v.blocki);
                IndexType ki = volume(v.kroni);
                IndexType kd = volume(v.krond);
                IndexType block_cols = num_cols / block_size / kd;
                const bool tb = !v.blockImFast;

                zero_n(y.data(), y.size(), y.ctx());
                // If the density of the Kronecker factors is low, treat them as sparse;
                // otherwise, perform the contraction with batched gemm
                if (kron_nnz_density < .3) {
                    for (int mu = 0; mu < (int)kron_disp_rev.size(); ++mu) {
                        for (int i = 0; i < ki; ++i) {
                            bool column_i_is_zero = true;
                            for (int j = 0; j < kd; ++j) {
                                T alpha_kron_ijmu = kron_cpu[(!tb ? i + j * ki : j + i * kd) +
                                                             kron_disp_rev[mu] * ki * kd] *
                                                    alpha;
                                if (std::norm(alpha_kron_ijmu) == 0) continue;
                                if (column_i_is_zero) {
                                    local_copy<IndexType>(
                                        alpha_kron_ijmu, Order<3>{'r', 'c', 'd'},
                                        Coor<3>{0, col0, j},
                                        Coor<3>{block_size * block_cols, dcols, 1},
                                        Coor<3>{block_size * block_cols, ncols, kd}, x, Mask<Gpu>{},
                                        Order<4>{'r', 'm', 'c', 'i'}, Coor<4>{0, mu, 0, i},
                                        Coor<4>{block_size * block_cols, (int)kron_disp_rev.size(),
                                                dcols, ki},
                                        y, Mask<Gpu>{}, EWOp::Copy{}, FastToSlow);
                                } else {
                                    local_copy<IndexType>(
                                        alpha_kron_ijmu, Order<3>{'r', 'c', 'd'},
                                        Coor<3>{0, col0, j},
                                        Coor<3>{block_size * block_cols, dcols, 1},
                                        Coor<3>{block_size * block_cols, ncols, kd}, x, Mask<Gpu>{},
                                        Order<4>{'r', 'm', 'c', 'i'}, Coor<4>{0, mu, 0, i},
                                        Coor<4>{block_size * block_cols, (int)kron_disp_rev.size(),
                                                dcols, ki},
                                        y, Mask<Gpu>{}, EWOp::Add{}, FastToSlow);
                                }
                                column_i_is_zero = false;
                            }
                        }
                    }
                } else {
                    auto kron_it_ptr = v.kron_it.data();
                    auto x_ptr = x.data();
                    auto y_ptr = y.data();
                    unsigned int kron_disp_rev_size = kron_disp_rev.size();
                    auto kron_disp_rev_ptr = kron_disp_rev.data();
                    xgemm_batch<T>('N', !tb ? 'T' : 'N', block_size * block_cols, ki, kd, alpha,
                                   block_size * block_cols * ncols, !tb ? ki : kd, T{0},
                                   block_size * block_cols * kron_disp_rev.size() * dcols,
                                   num_nnz_per_row * dcols, x.ctx(),
                                   [=](int i, T **ai, T **bi, T **ci) {
                                       unsigned int mu = i % kron_disp_rev_size;
                                       unsigned int col = i / kron_disp_rev_size;
                                       *ai = (T *)x_ptr + (col + col0) * block_size * block_cols;
                                       *bi = kron_it_ptr + ki * kd * kron_disp_rev_ptr[mu];
                                       *ci = y_ptr + block_size * block_cols * mu +
                                             block_size * block_cols * kron_disp_rev_size * col;
                                   });
                }
            }

        public:
            void operator()(bool conjA, const vector<T, Gpu> &vx_, IndexType ldx, MatrixLayout lx,
                            vector<T, Gpu> &vy_, IndexType ldy, MatrixLayout ly,
                            IndexType ncols) const {

                const T alpha{1};
                bool is_kron = v.kron_it.size() > 0;
                check_same_device(vx_.ctx(), vy_.ctx());
                check_same_device(vx_.ctx(), ii.ctx());
                causalConnectTo(vy_.ctx(), vx_.ctx());
                causalConnectTo(vx_.ctx(), ii.ctx());
                auto vx = vx_.withNewContext(ii.ctx());
                auto vy = vy_.withNewContext(ii.ctx());
                const T *x = vx.data();
                T *y = vy.data();
                const T beta{0};

                IndexType num_cols = volume(v.dimd);
                IndexType num_rows = volume(v.dimi);
                if (num_cols == 0 || num_rows == 0 || ncols == 0) return;

                if (deviceId(vx_.ctx()) == CPU_DEVICE_ID || deviceId(vy_.ctx()) == CPU_DEVICE_ID)
                    throw std::runtime_error("BSR::operator: gpu implementation does not support "
                                             "cpu input/output tensors");

                if (!is_kron) {
                    xscal(vy.size(), beta, y, 1, vy.ctx());
                    matvec(alpha, conjA, x, ldx, lx, y, ldy, ly, ncols, beta);
                    causalConnectTo(ii.ctx(), vy_.ctx());
                    return;
                }

                if (conjA)
                    throw std::runtime_error("BSR operator(): unsupported conjugate "
                                             "multiplication with BSR Kronecker");

                IndexType block_size = volume(v.blocki);
                IndexType ki = volume(v.kroni);
                IndexType kd = volume(v.krond);
                IndexType block_cols = num_cols / block_size / kd;
                IndexType block_rows = num_rows / block_size / ki;

                assert(vx.size() == (std::size_t)(block_size * block_cols * ncols * kd));
                assert(vy.size() == (std::size_t)(block_size * block_rows * ncols * ki));
                assert(v.kron_it.size() == (std::size_t)(kd * ki * num_nnz_per_row));

#    ifdef SUPERBBLAS_USE_GPU
                if (kron_use_crafted_kernel) {
                    assert(lx == RowMajor && ly == RowMajor);
                    assert(ldx >= kd * ncols && ldy >= kd * ncols);
                    bsr_kron_3x3_4x4perm(v.it.data(), v.blockImFast ? 1 : block_size,
                                         v.blockImFast ? block_size : 1, jj.data(), block_rows,
                                         num_nnz_per_row, kron_scalars.data(), kron_perm.data(), x,
                                         kd * block_size * ncols, y, ki * block_size * ncols, ncols,
                                         ii.ctx());
                    causalConnectTo(ii.ctx(), vy_.ctx());
                    return;
                }
#    endif // SUPERBBLAS_USE_GPU

                if (ly == RowMajor && lx == RowMajor) {
                    assert(ldy == ki * ncols);

                    // Limit the amount of auxiliary memory used to 50% maximum cache size
                    std::size_t vector_size =
                        sizeof(T) * ki * block_size * block_cols * num_nnz_per_row +
                        (sizeof(T) + sizeof(IndexType)) * kd * block_size * block_cols +
                        (sizeof(T) + sizeof(IndexType)) * ki * block_size * block_rows;
                    IndexType max_ncols = (int)std::min(
                        std::max((size_t)getMaxGpuCacheSize() / 2u / vector_size, (std::size_t)1),
                        (std::size_t)ncols);

                    // Pre-apply the beta if the computation is going to break in chunks
                    if (std::norm(beta) != 0 && beta != T{1} && max_ncols != ncols)
                        xscal(num_rows * ncols, beta, y, 1, ii.ctx());

                    // Process up to `max_ncols` at a time
                    for (IndexType i0 = 0, ncols0 = std::min(ncols, max_ncols); i0 < ncols;
                         i0 += ncols0, ncols0 = std::min(ncols - i0, max_ncols)) {
                        // Copy the columns [i0,i0+ncols0-1] columns of x into a continuous allocation
                        const T *x0 = x;
                        vector<T, Gpu> auxx;
                        if (ncols0 != ncols) {
                            auxx =
                                vector<T, Gpu>((std::size_t)kd * ncols0 * block_size * block_cols,
                                               ii.ctx(), doCacheAlloc);
                            x0 = auxx.data();
                            Coor<3> dimx{kd, ncols, block_size * block_cols};
                            Coor<3> dimx0{kd, ncols0, block_size * block_cols};
                            local_copy<IndexType>(
                                T{1}, trivial_order<3>(), Coor<3>{0, i0, 0}, dimx0, dimx,
                                (vector<const T, Gpu>)vx, Mask<Gpu>{}, trivial_order<3>(),
                                Coor<3>{{}}, dimx0, auxx, Mask<Gpu>{}, EWOp::Copy{}, FastToSlow);
                        }

                        // Contract the Kronecker part: for each direction mu do:
                        //  (ki,kd)[mu] x (kd,ncols,bd,rows) -> (ki,ncols,bd,rows,mu)
                        vector<T, Gpu> aux(ki * ncols0 * block_size * block_cols * num_nnz_per_row,
                                           ii.ctx(), doCacheAlloc);
                        zero_n(aux.data(), aux.size(), aux.ctx());
                        const bool tb = !v.blockImFast;
                        xgemm_batch_strided(
                            !tb ? 'N' : 'T', 'N', ki, ncols0 * block_size * block_cols, kd, alpha,
                            v.kron_it.data(), !tb ? ki : kd, ki * kd, x0, kd, 0, T{0}, aux.data(),
                            ki, ki * ncols0 * block_size * block_cols, num_nnz_per_row, aux.ctx());

                        // Contract the block part:
                        // \sum_i (bi,bd)[i,col,mu] x (ki,ncols,bd,rows,mu)[rows=col,mu] -> (ki,ncols,bi,rows)
                        T *y0 = y;
                        IndexType ldy0 = ldy;
                        vector<T, Gpu> auxy;
                        if (ncols0 != ncols) {
                            auxy = vector<T, Gpu>(ki * ncols0 * block_size * block_rows, aux.ctx(),
                                                  doCacheAlloc);
                            zero_n(auxy.data(), auxy.size(), auxy.ctx());
                            y0 = auxy.data();
                            ldy0 = ki * ncols0;
                        }
                        matvec(1.0, false, aux.data(), ki * ncols0, RowMajor, y0, ldy0, ly,
                               ncols0 * ki,
                               (std::norm(beta) == 0 || ncols0 != ncols) ? T{0} : T{1});
                        if (ncols0 != ncols) {
                            Coor<3> dimy0{ki, ncols0, block_size * block_rows};
                            Coor<3> dimy{ki, ncols, block_size * block_rows};
                            if (std::norm(beta) == 0) {
                                local_copy<IndexType>(T{1}, trivial_order<3>(), Coor<3>{{}}, dimy0,
                                                      dimy0, (vector<const T, Gpu>)auxy,
                                                      Mask<Gpu>{}, trivial_order<3>(),
                                                      Coor<3>{0, i0, 0}, dimy, vy, Mask<Gpu>{},
                                                      EWOp::Copy{}, FastToSlow);
                            } else {
                                local_copy<IndexType>(T{1}, trivial_order<3>(), Coor<3>{{}}, dimy0,
                                                      dimy0, (vector<const T, Gpu>)auxy,
                                                      Mask<Gpu>{}, trivial_order<3>(),
                                                      Coor<3>{0, i0, 0}, dimy, vy, Mask<Gpu>{},
                                                      EWOp::Add{}, FastToSlow);
                            }
                        }
                    }
                } else if (ly == ColumnMajor && lx == ColumnMajor) {
                    assert(ldy == block_size * block_rows);

                    // Limit the amount of auxiliary memory used to 50% maximum cache size
                    std::size_t vector_size =
                        sizeof(T) * block_size * block_cols * kron_disp_rev.size() * ki +
                        (sizeof(T) + sizeof(IndexType)) * block_size * block_rows * ki;
                    IndexType max_ncols =
                        std::min(std::max(getMaxGpuCacheSize() / 2u / vector_size, (std::size_t)1),
                                 (std::size_t)ncols);

                    // Pre-apply the beta if the computation is going to break in chunks
                    if (std::norm(beta) != 0 && beta != T{1} && max_ncols != ncols)
                        xscal(num_rows * ncols, beta, y, 1, ii.ctx());

                    // Process up to `max_ncols` at a time
                    for (IndexType i0 = 0, ncols0 = std::min(ncols, max_ncols); i0 < ncols;
                         i0 += ncols0, ncols0 = std::min(ncols - i0, max_ncols)) {
                        // Contract the Kronecker part: for each direction mu do:
                        //  (bd,rows,ncols,kd) x (ki,kd)[mu] -> (bd,rows,mu,ncols,ki)
                        vector<T, Gpu> aux(block_size * block_cols * kron_disp_rev.size() * ncols0 *
                                               ki,
                                           ii.ctx(), doCacheAlloc);
                        contract_kron_cols(alpha, vx, i0, ncols0, ncols, aux);

                        // Contract the block part:
                        // \sum_i (bi,bd)[i,col,mu] x (bd,rows,mu,ncols,ki)[rows=col,mu] -> (bi,rows,ncols,ki)
                        T *y0 = y;
                        IndexType ldy0 = ldy;
                        vector<T, Gpu> auxy;
                        if (ncols0 != ncols) {
                            auxy = vector<T, Gpu>(block_size * block_rows * ncols0 * ki, aux.ctx(),
                                                  doCacheAlloc);
                            y0 = auxy.data();
                            zero_n(auxy.data(), auxy.size(), auxy.ctx());
                            ldy0 = block_size * block_rows;
                        }
                        matvec(1.0, false, aux.data(),
                               block_size * block_cols * kron_disp_rev.size(), ColumnMajor, y0,
                               ldy0, ly, ncols0 * ki,
                               (std::norm(beta) == 0 || ncols0 != ncols) ? T{0} : beta);
                        if (ncols0 != ncols) {
                            Coor<3> dimy0{block_size * block_rows, ncols0, ki};
                            Coor<3> dimy{block_size * block_rows, ncols, ki};
                            if (std::norm(beta) == 0) {
                                local_copy<IndexType>(T{1}, trivial_order<3>(), Coor<3>{{}}, dimy0,
                                                      dimy0, (vector<const T, Gpu>)auxy,
                                                      Mask<Gpu>{}, trivial_order<3>(),
                                                      Coor<3>{0, i0, 0}, dimy, vy, Mask<Gpu>{},
                                                      EWOp::Copy{}, FastToSlow);
                            } else {
                                local_copy<IndexType>(T{1}, trivial_order<3>(), Coor<3>{{}}, dimy0,
                                                      dimy0, (vector<const T, Gpu>)auxy,
                                                      Mask<Gpu>{}, trivial_order<3>(),
                                                      Coor<3>{0, i0, 0}, dimy, vy, Mask<Gpu>{},
                                                      EWOp::Add{}, FastToSlow);
                            }
                        }
                    }
                } else
                    throw std::runtime_error(
                        "BSR operator(): unsupported input/output tensor layout");
                causalConnectTo(ii.ctx(), vy_.ctx());
            }

            ~BSR() {}
        };
#endif // SUPERBBLAS_USE_GPU

        /// A BSR tensor composed of several components
        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU0, typename XPU1>
        struct BSRComponents_tmpl : BSR_handle {
            /// Partition of the domain space
            Proc_ranges<Nd> pd;
            /// Dimensions of the domain space
            Coor<Nd> dimd;
            /// Partition of the image space
            Proc_ranges<Ni> pi;
            // Dimensiosn of the image space
            Coor<Ni> dimi;
            /// Components of the BSR operator
            std::pair<std::vector<BSR<Nd, Ni, T, XPU0>>, std::vector<BSR<Nd, Ni, T, XPU1>>> c;
            Coor<Nd> blockd; ///< dimensions of a block in the domain space
            Coor<Ni> blocki; ///< dimensions of a block in the image space
            Coor<Nd> krond;  ///< dimensions of Kronecker in the domain space
            Coor<Ni> kroni;  ///< dimensions of Kronecker in the image space
            CoorOrder co;    ///< Coordinate order of ii and jj

            bool check(std::size_t Nd_, std::size_t Ni_, detail::num_type type, const Context *ctx,
                       unsigned int ncomponents, unsigned int nprocs, unsigned int rank,
                       CoorOrder co) override {
                (void)rank;
                if (Nd_ != Nd || Ni_ != Ni || num_type_v<T>::value != type || nprocs != pd.size() ||
                    ncomponents != pd[rank].size())
                    return false;

                if (c.first.size() + c.second.size() != ncomponents) return false;
                for (unsigned int component = 0; component < ncomponents; ++component) {
                    for (const auto &ci : c.first)
                        if (ci.v.componentId == component && ci.v.it.size() > 0 &&
                            deviceId(ci.v.it.ctx()) != ctx[component].device)
                            return false;
                    for (const auto &ci : c.second)
                        if (ci.v.componentId == component && ci.v.it.size() > 0 &&
                            deviceId(ci.v.it.ctx()) != ctx[component].device)
                            return false;
                }

                for (const auto &ci : c.first)
                    if (ci.v.componentId >= ncomponents || ci.v.co != co) return false;
                for (const auto &ci : c.second)
                    if (ci.v.componentId >= ncomponents || ci.v.co != co) return false;

                return true;
            }
        };

#ifdef SUPERBBLAS_USE_GPU
        /// A tensor composed of several CPU and GPU elements
        template <std::size_t Nd, std::size_t Ni, typename T>
        using BSRComponents = BSRComponents_tmpl<Nd, Ni, T, Gpu, Cpu>;
#else
        /// A tensor composed of only of CPU components
        template <std::size_t Nd, std::size_t Ni, typename T>
        using BSRComponents = BSRComponents_tmpl<Nd, Ni, T, Cpu, Cpu>;
#endif // SUPERBBLAS_USE_GPU

        /// Return a components based on the nonzeros of a BSR operator
        /// \param bsr: BSR operator

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU0, typename XPU1>
        Components_tmpl<Ni, T, XPU0, XPU1>
        get_mock_components(const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> &bsr) {
            Components_tmpl<Ni, T, XPU0, XPU1> r;
            for (unsigned int i = 0; i < bsr.c.first.size(); ++i) {
                r.first.push_back(Component<Ni, T, XPU0>{
                    bsr.c.first[i].v.it, {{}}, bsr.c.first[i].v.componentId, Mask<XPU0>{}});
            }
            for (unsigned int i = 0; i < bsr.c.second.size(); ++i) {
                r.second.push_back(Component<Ni, T, XPU1>{
                    bsr.c.second[i].v.it, {{}}, bsr.c.second[i].v.componentId, Mask<XPU1>{}});
            }

            return r;
        }

        template <std::size_t Nd, std::size_t Ni, typename T, typename Comm>
        BSRComponents<Nd, Ni, T>
        get_bsr_components(T **v, const Context *ctx_v, IndexType **ii, const Context *ctx_ii,
                           Coor<Nd> **jj, const Context *ctx_jj, T **kronv,
                           const Context *ctx_kronv, unsigned int ncomponents,
                           From_size_iterator<Ni> pi, const Coor<Ni> &dimi,
                           From_size_iterator<Nd> pd, const Coor<Nd> &dimd, const Coor<Nd> &blockd,
                           const Coor<Ni> &blocki, const Coor<Nd> &krond, const Coor<Ni> &kroni,
                           bool blockImFast, Comm comm, CoorOrder co, Session session) {
            // Get components on the local process
            From_size_iterator<Nd> fsd = pd + comm.rank * ncomponents;
            From_size_iterator<Ni> fsi = pi + comm.rank * ncomponents;

            BSRComponents<Nd, Ni, T> r{};
            r.dimd = dimd;
            r.dimi = dimi;
            r.pd = detail::get_from_size(pd, ncomponents * comm.nprocs, comm, dimd);
            r.pi = detail::get_from_size(pi, ncomponents * comm.nprocs, comm, dimi);
            r.blockd = blockd;
            r.blocki = blocki;
            r.krond = krond;
            r.kroni = kroni;
            r.co = co;

            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("get_bsr_components"), dimd, dimi,
                                                  r.pd, r.pi, blockd, blocki, krond, kroni, co,
                                                  typeid(tag_type).hash_code()),
                                  comm);
            }

            for (unsigned int i = 0; i < ncomponents; ++i) {
                std::size_t nii = volume(fsi[i][1]) / volume(blocki) / volume(kroni);
                vector<IndexType, Cpu> vii;
                if (ctx_ii[i].plat == CPU) {
                    vii = to_vector(ii[i], nii, ctx_ii[i].toCpu(session));
                } else {
#ifdef SUPERBBLAS_USE_GPU
                    vii = makeSure(to_vector(ii[i], nii, ctx_ii[i].toGpu(session)), Cpu{session});
#else
                    throw std::runtime_error("superbblas was not compiled with GPU support");
#endif
                }
                const std::size_t njj = sum(vii);
                vector<Coor<Nd>, Cpu> vjj;
                if (ctx_jj[i].plat == CPU) {
                    vjj = to_vector(jj[i], njj, ctx_jj[i].toCpu(session));
                } else {
#ifdef SUPERBBLAS_USE_GPU
                    vjj = makeSure(to_vector(jj[i], njj, ctx_jj[i].toGpu(session)), Cpu{session});
#else
                    throw std::runtime_error("superbblas was not compiled with GPU support");
#endif
                }
                std::size_t nvalues = njj * volume(blockd) * volume(blocki);
                T *kronvi = kronv ? kronv[i] : (T *)nullptr;
                std::size_t num_neighbors = (nii > 0 ? njj / nii : 0);
                std::size_t nkronvalues =
                    (kronv ? volume(krond) * volume(kroni) * num_neighbors : 0);
                switch (ctx_v[i].plat) {
#ifdef SUPERBBLAS_USE_GPU
                case CPU:
                    r.c.second.push_back(BSR<Nd, Ni, T, Cpu>{BSRComponent<Nd, Ni, T, Cpu>{
                        vii, vjj, to_vector(v[i], nvalues, ctx_v[i].toCpu(session)), fsd[i][1],
                        fsi[i][1], blockd, blocki, krond, kroni,
                        to_vector(kronvi, nkronvalues, ctx_kronv[i].toCpu(session)), blockImFast,
                        co, i}});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
                case GPU:
                    r.c.first.push_back(BSR<Nd, Ni, T, Gpu>{BSRComponent<Nd, Ni, T, Gpu>{
                        vii, vjj, to_vector(v[i], nvalues, ctx_v[i].toGpu(session)), fsd[i][1],
                        fsi[i][1], blockd, blocki, krond, kroni,
                        to_vector(kronvi, nkronvalues, ctx_kronv[i].toGpu(session)), blockImFast,
                        co, i}});
                    assert(!v[i] || getPtrDevice(v[i]) == ctx_v[i].device);
                    break;
#else // SUPERBBLAS_USE_GPU
                case CPU:
                    r.c.first.push_back(BSR<Nd, Ni, T, Cpu>{BSRComponent<Nd, Ni, T, Cpu>{
                        vii, vjj, to_vector(v[i], nvalues, ctx_v[i].toCpu(session)), fsd[i][1],
                        fsi[i][1], blockd, blocki, krond, kroni,
                        to_vector(kronvi, nkronvalues, ctx_kronv[i].toCpu(session)), blockImFast,
                        co, i}});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
#endif
                default: throw std::runtime_error("Unsupported platform");
                }
            }
            return r;
        }

        template <std::size_t Nd, std::size_t Ni, typename T, typename Comm>
        BSRComponents<Nd, Ni, T> *
        get_bsr_components_from_handle(BSR_handle *bsrh, const Context *ctx, int ncomponents,
                                       const Comm &comm, CoorOrder co) {
            if (!bsrh->check(Nd, Ni, detail::num_type_v<T>::value, ctx, ncomponents, comm.nprocs,
                             comm.rank, co))
                throw std::runtime_error(
                    "Given BSR handle doesn't match the template parameters Nd, Ni, or T, does not "
                    "match contexts, or does not match MPI communicator");
            return static_cast<BSRComponents<Nd, Ni, T> *>(bsrh);
        }

        //
        // BSR implementations
        //

        template <typename T, std::size_t Na, std::size_t Nb>
        std::array<T, Na + Nb> concat(const std::array<T, Na> &a, const std::array<T, Nb> &b) {
            std::array<T, Na + Nb> r;
            std::copy_n(a.begin(), Na, r.begin());
            std::copy_n(b.begin(), Nb, r.begin() + Na);
            return r;
        }

        /// Concatenate p (if it isn't zero), a, b, and c

        template <std::size_t N, typename T, std::size_t Na, std::size_t Nb, std::size_t Nc>
        std::array<T, N> concat(char p, const std::array<T, Na> &a, std::size_t na,
                                const std::array<T, Nb> &b, std::size_t nb,
                                const std::array<T, Nc> &c, std::size_t nc) {
            std::array<T, N> r;
            int np = p == 0 ? 0 : 1;
            if (N != np + na + nb + nc) throw std::runtime_error("concat: invalid arguments");
            if (p != 0) r[0] = p;
            std::copy_n(a.begin(), na, r.begin() + np);
            std::copy_n(b.begin(), nb, r.begin() + np + na);
            std::copy_n(c.begin(), nc, r.begin() + np + na + nb);
            return r;
        }

        /// Concatenate p (if it isn't zero), a, b, c, and d

        template <std::size_t N, typename T, std::size_t Na, std::size_t Nb, std::size_t Nc,
                  std::size_t Nd>
        std::array<T, N> concat(char p, const std::array<T, Na> &a, std::size_t na,
                                const std::array<T, Nb> &b, std::size_t nb,
                                const std::array<T, Nc> &c, std::size_t nc,
                                const std::array<T, Nd> &d, std::size_t nd) {
            std::array<T, N> r;
            int np = p == 0 ? 0 : 1;
            if (N != np + na + nb + nc + nd) throw std::runtime_error("concat: invalid arguments");
            if (p != 0) r[0] = p;
            std::copy_n(a.begin(), na, r.begin() + np);
            std::copy_n(b.begin(), nb, r.begin() + np + na);
            std::copy_n(c.begin(), nc, r.begin() + np + na + nb);
            std::copy_n(d.begin(), nd, r.begin() + np + na + nb + nc);
            return r;
        }

        /// Concatenate p (if it isn't zero), a, b, c, and d

        template <std::size_t N, typename T, std::size_t Na, std::size_t Nb, std::size_t Nc,
                  std::size_t Nd, std::size_t Ne>
        std::array<T, N> concat(char p, const std::array<T, Na> &a, std::size_t na,
                                const std::array<T, Nb> &b, std::size_t nb,
                                const std::array<T, Nc> &c, std::size_t nc,
                                const std::array<T, Nd> &d, std::size_t nd,
                                const std::array<T, Ne> &e, std::size_t ne) {
            std::array<T, N> r;
            int np = p == 0 ? 0 : 1;
            if (N != np + na + nb + nc + nd + ne)
                throw std::runtime_error("concat: invalid arguments");
            if (p != 0) r[0] = p;
            std::copy_n(a.begin(), na, r.begin() + np);
            std::copy_n(b.begin(), nb, r.begin() + np + na);
            std::copy_n(c.begin(), nc, r.begin() + np + na + nb);
            std::copy_n(d.begin(), nd, r.begin() + np + na + nb + nc);
            std::copy_n(e.begin(), ne, r.begin() + np + na + nb + nc + nd);
            return r;
        }

        //
        // Auxiliary functions
        //

        /// Return BSR indices for a given tensor BSR and whether the nonzero
        /// pattern is compatible with blocked ELL, all rows has the same number
        /// of nonzeros, and unused nonzeros may be reported with a -1 on the
        /// the first domain coordinate.

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU,
                  typename std::enable_if<(Nd > 0 && Ni > 0), bool>::type = true>
        CsrIndices<XPU> get_bsr_indices(const BSRComponent<Nd, Ni, T, XPU> &v,
                                        ReturnJJ return_jj_blocked = ReturnJJNoBlocking) {
            // Check that IndexType is big enough
            if ((std::size_t)std::numeric_limits<IndexType>::max() <= volume(v.dimd))
                throw std::runtime_error("Ups! IndexType isn't big enough");

            Indices<Cpu> ii(v.i.size() + 1, Cpu{}), jj(v.j.size(), Cpu{});
            Indices<Cpu> vi = makeSure(v.i, Cpu{});
            vector<Coor<Nd>, Cpu> vj = makeSure(v.j, Cpu{});

            // Check if all rows has the same number of nonzeros
            bool same_nnz_per_row = true;
            for (std::size_t i = 1; i < vi.size(); ++i)
                if (vi[i] != vi[0]) same_nnz_per_row = false;
            int num_nnz_per_row = vi.size() > 0 && same_nnz_per_row ? vi[0] : -1;

            // Compute index for the first nonzero on the ith row
            ii[0] = 0;
            for (std::size_t i = 0; i < vi.size(); ++i) ii[i + 1] = ii[i] + vi[i];

            // Transform the domain coordinates into indices
            Coor<Nd> strided = get_strides<IndexType>(v.dimd, v.co);
            IndexType block_nnz = v.j.size();
            IndexType bd =
                (return_jj_blocked == ReturnJJNoBlocking
                     ? 1
                     : volume(v.blockd) *
                           (return_jj_blocked == ReturnJJBlockedWithKron ? volume(v.krond) : 1));
            bool there_are_minus_ones_in_columns = false;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
            for (IndexType i = 0; i < block_nnz; ++i) {
                if (vj[i][0] == -1) there_are_minus_ones_in_columns = true;
                jj[i] = (vj[i][0] == -1 ? -1 : coor2index(vj[i], v.dimd, strided) / bd);
            }

            // Unsupported -1 in the domain coordinate unless using the ELL format, which
            // requires all rows to have the same number of nonzeros
            if (there_are_minus_ones_in_columns && !same_nnz_per_row)
                throw std::runtime_error(
                    "bsr: unsupported nonzero pattern specification, some domain coordinates have "
                    "-1 but not all block rows have the same number of nonzero blocks");

            return {makeSure(ii, v.it.ctx(), doCacheAlloc, true),
                    makeSure(jj, v.it.ctx(), doCacheAlloc, true), there_are_minus_ones_in_columns,
                    num_nnz_per_row, ii[vi.size()]};
        }

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU,
                  typename std::enable_if<(Nd == 0 || Ni == 0), bool>::type = true>
        std::pair<CsrIndices<XPU>, int>
        get_bsr_indices(const BSRComponent<Nd, Ni, T, XPU> &v,
                        ReturnJJ return_jj_blocked = ReturnJJNoBlocking) {
            (void)return_jj_blocked;
            return {Indices<XPU>(0, v.it.ctx()), Indices<XPU>(0, v.it.ctx()), false, -1, 0};
        }

        /// Return the indices for a given tensor Kronecker BSR and whether the nonzero
        /// pattern is compatible with blocked ELL, all rows has the same number
        /// of nonzeros, and unused nonzeros may be reported with a -1 on the
        /// first domain coordinate.

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU,
                  typename std::enable_if<(Nd > 0 && Ni > 0), bool>::type = true>
        CsrIndices<XPU> get_kron_indices(const BSRComponent<Nd, Ni, T, XPU> &v,
                                         bool return_jj_blocked = false,
                                         std::vector<int> kron_disp = std::vector<int>()) {
            // Check that IndexType is big enough
            if ((std::size_t)std::numeric_limits<IndexType>::max() <= volume(v.dimd))
                throw std::runtime_error("Ups! IndexType isn't big enough");

            Indices<Cpu> ii(v.i.size() + 1, Cpu{}), jj(v.j.size(), Cpu{});
            Indices<Cpu> vi = makeSure(v.i, Cpu{});
            vector<Coor<Nd>, Cpu> vj = makeSure(v.j, Cpu{});

            // Check if all rows has the same number of nonzeros
            bool same_nnz_per_row = true;
            for (std::size_t i = 1; i < vi.size(); ++i)
                if (vi[i] != vi[0]) same_nnz_per_row = false;
            if (!same_nnz_per_row)
                throw std::runtime_error("get_kron_indices: unsupported having a different number "
                                         "of nonzeros in each row");
            int num_nnz_per_row = vi.size() > 0 && same_nnz_per_row ? vi[0] : -1;

            // The Kronecker BSR format is simulated by splitting the operator in several terms,
            // each of them having the nonzeros for a particular direction. In each term, only
            // one nonzero is for each row
            ii[0] = 0;
            for (std::size_t i = 0; i < vi.size(); ++i) ii[i + 1] = ii[i] + vi[i];

            // Transform the domain coordinates into indices
            Coor<Nd> strided = get_strides<IndexType>(v.dimd, v.co);
            IndexType block_nnz = v.j.size();
            IndexType bd = return_jj_blocked ? volume(v.blockd) * volume(v.krond) : 1;
            IndexType rows = volume(v.dimd);
            bool there_are_minus_ones_in_columns = false;
            if (kron_disp.size() == 0) {
                kron_disp = std::vector<int>(num_nnz_per_row);
                for (int i = 0; i < num_nnz_per_row; ++i) kron_disp[i] = i;
            }
            if (kron_disp.size() != (std::size_t)num_nnz_per_row) throw std::runtime_error("wtf!");
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
            for (IndexType i = 0; i < block_nnz; ++i) {
                if (vj[i][0] == -1) there_are_minus_ones_in_columns = true;
                jj[i] = (vj[i][0] == -1 ? -1
                                        : coor2index(vj[i], v.dimd, strided) / bd +
                                              rows / bd * kron_disp[i % num_nnz_per_row]);
            }

            // Unsupported -1 in the domain coordinate
            if (there_are_minus_ones_in_columns)
                throw std::runtime_error("get_kron_indices: unsupported nonzero pattern "
                                         "specification, some domain coordinates have -1");

            return {makeSure(ii, v.it.ctx()), makeSure(jj, v.it.ctx()),
                    there_are_minus_ones_in_columns, num_nnz_per_row, ii[vi.size()]};
        }

        template <std::size_t Nd, std::size_t Ni, typename T, typename XPU,
                  typename std::enable_if<(Nd == 0 || Ni == 0), bool>::type = true>
        std::pair<CsrIndices<XPU>, int> get_kron_indices(const BSRComponent<Nd, Ni, T, XPU> &v,
                                                         bool return_jj_blocked = false) {
            (void)return_jj_blocked;
            return {Indices<XPU>(0, v.it.ctx()), Indices<XPU>(0, v.it.ctx()), false, -1, 0};
        }

        /// Return splitting of dimension labels for the RSB operator - tensor multiplication
        /// \param dimi: dimension of the RSB operator image in consecutive ranges
        /// \param dimd: dimension of the RSB operator domain in consecutive ranges
        /// \param oi: dimension labels for the RSB operator image space
        /// \param od: dimension labels for the RSB operator domain space
        /// \param blocki: image dimensions of the block
        /// \param blockd: domain dimensions of the block
        /// \param kroni: image dimensions of the Kronecker block
        /// \param krond: domain dimensions of the Kronecker block
        /// \param is_kron: whether using the Kronecker BSR format
        /// \param dimx: dimension of the right tensor in consecutive ranges
        /// \param ox: dimension labels for the right operator
        /// \param dimy: dimension of the resulting tensor in consecutive ranges
        /// \param oy: dimension labels for the output tensor
        /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
        /// \param ncols_permissive: whether to check the number of columns
        /// \param xylayout: possible layouts for x and y
        /// \param preferred_layout: preferred layout for x and y
        /// \param co: coordinate linearization order
        /// \param transSp: (output) whether to contract with the sparse operator image space
        /// \param lx: (output) layout for the x tensor
        /// \param ly: (output) layout for the y tensor
        /// \param volC: (output) number of columns to contract
        /// \param sug_ox: (output) suggested order for the x
        /// \param sug_oy: (output) suggested order for the y
        /// \param sug_oy_trans: (output) suggested order for copying y into x
        ///
        /// For SlowToFast and no Kronecker blocking, it supports:
        ///   (I,D,i,d) x (C,D,d) -> (C,I,i)
        /// where
        /// - (D,d) are the domain RSB dimensions labels, od;
        /// - (I,i) are the image RSB dimensions labels, oi;
        /// - (C,D,d) are the dimensions labels of the right input tensor, ox
        /// - (C,I,i) are the dimensions labels of the output tensor, oy
        /// - D and I has dimensions labels that are not blocked
        /// - d and i has all blocked dimensions labels
        ///
        /// For SlowToFast and no Kronecker blocking, it supports:
        ///   (ki,kd) x (I,D,i,d) x (D,kd,C,d) -> (I,i,C,ki)
        ///

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny>
        void local_bsr_krylov_check(const Coor<Ni> &dimi, const Coor<Nd> &dimd, const Order<Ni> &oi,
                                    const Order<Nd> &od, const Coor<Ni> &blocki,
                                    const Coor<Nd> &blockd, const Coor<Ni> &kroni,
                                    const Coor<Nd> &krond, bool is_kron, const Coor<Nx> &dimx,
                                    const Order<Nx> &ox, const Coor<Ny> &dimy, const Order<Ny> &oy,
                                    char okr, bool ncols_permissive, SpMMAllowedLayout xylayout,
                                    MatrixLayout preferred_layout, CoorOrder co, bool &transSp,
                                    MatrixLayout &lx, MatrixLayout &ly, std::size_t &volC,
                                    Order<Nx> &sug_ox, Order<Ny> &sug_oy, Order<Ny> &sug_oy_trans) {

            // Detect real dimension in oi, od, x and y
            std::size_t ni = 0;
            for (int i = (int)Ni - 1; i >= 0; --i) {
                if (dimi[i] != 1 || std::find(od.begin(), od.end(), oi[i]) != od.end() ||
                    std::find(ox.begin(), ox.end(), oi[i]) != ox.end() ||
                    std::find(oy.begin(), oy.end(), oi[i]) != oy.end()) {
                    ni = i + 1;
                    break;
                }
            }
            std::size_t nd = 0;
            for (int i = (int)Nd - 1; i >= 0; --i) {
                if (dimd[i] != 1 || std::find(oi.begin(), oi.end(), od[i]) != oi.end() ||
                    std::find(ox.begin(), ox.end(), od[i]) != ox.end() ||
                    std::find(oy.begin(), oy.end(), od[i]) != oy.end()) {
                    nd = i + 1;
                    break;
                }
            }
            std::size_t nx = 0;
            for (int i = (int)Nx - 1; i >= 0; --i) {
                if (dimx[i] != 1 || std::find(od.begin(), od.end(), ox[i]) != od.end() ||
                    std::find(oi.begin(), oi.end(), ox[i]) != oi.end() ||
                    std::find(oy.begin(), oy.end(), ox[i]) != oy.end()) {
                    nx = i + 1;
                    break;
                }
            }
            std::size_t ny = 0;
            for (int i = (int)Ny - 1; i >= 0; --i) {
                if (dimy[i] != 1 || std::find(od.begin(), od.end(), oy[i]) != od.end() ||
                    std::find(oi.begin(), oi.end(), oy[i]) != oi.end() ||
                    std::find(ox.begin(), ox.end(), oy[i]) != ox.end()) {
                    ny = i + 1;
                    break;
                }
            }

            if (co == FastToSlow) {
                Order<Nx> sug_ox0;
                Order<Ny> sug_oy0;
                Order<Ny> sug_oy_trans0;
                local_bsr_krylov_check(reverse(dimi, ni), reverse(dimd, nd), reverse(oi, ni),
                                       reverse(od, nd), reverse(blocki, ni), reverse(blockd, nd),
                                       reverse(kroni, ni), reverse(krond, nd), is_kron,
                                       reverse(dimx, nx), reverse(ox, nx), reverse(dimy, ny),
                                       reverse(oy, ny), okr, ncols_permissive, xylayout,
                                       preferred_layout, SlowToFast, transSp, lx, ly, volC, sug_ox0,
                                       sug_oy0, sug_oy_trans0);
                sug_ox = reverse(sug_ox0, nx);
                sug_oy = reverse(sug_oy0, ny);
                sug_oy_trans = reverse(sug_oy_trans0, ny);
                return;
            }

            // Find all common labels in od and oi
            for (char c : od)
                if (std::find(oi.begin(), oi.end(), c) != oi.end())
                    throw std::runtime_error(
                        "Common label between the domain and image of the sparse matrix");

            // Check the dimensions
            bool failMatch = false;
            for (unsigned int i = 0; i < nx; ++i) {
                if (ox[i] == okr) continue;
                auto sd = std::find(od.begin(), od.end(), ox[i]);
                if (sd != od.end()) {
                    failMatch |= (dimd[sd - od.begin()] != dimx[i]);
                    continue;
                }
                auto si = std::find(oi.begin(), oi.end(), ox[i]);
                if (si != oi.end()) failMatch |= (dimi[si - oi.begin()] != dimx[i]);
            }
            if (failMatch)
                throw std::runtime_error("bsr_krylov: dimensions of the dense input tensor "
                                         "doesn't match the sparse tensor");
            for (unsigned int i = 0; i < ny; ++i) {
                if (oy[i] == okr) continue;
                auto sd = std::find(od.begin(), od.end(), oy[i]);
                if (sd != od.end()) {
                    failMatch |= (dimd[sd - od.begin()] != dimy[i]);
                    continue;
                }
                auto si = std::find(oi.begin(), oi.end(), oy[i]);
                if (si != oi.end()) {
                    failMatch |= (dimi[si - oi.begin()] != dimy[i]);
                    continue;
                }
                auto sx = std::find(ox.begin(), ox.end(), oy[i]);
                if (sx != ox.end()) {
                    if (ncols_permissive && (dimx[sx - ox.begin()] == 0 || dimy[i] == 0))
                        continue; // don't check rhs
                    failMatch |= (dimx[sx - ox.begin()] != dimy[i]);
                    continue;
                }
            }
            if (failMatch)
                throw std::runtime_error(
                    "bsr_krylov: dimensions of the dense output tensor "
                    "doesn't match the sparse tensor or the dense input tensor");

            // Find all common labels in od+oi, ox, and oy
            Order<Nx> oT;
            std::size_t nT = 0;
            for (char c : ox)
                if ((std::find(od.begin(), od.end(), c) != od.end() ||
                     std::find(oi.begin(), oi.end(), c) != oi.end()) &&
                    std::find(oy.begin(), oy.end(), c) != oy.end())
                    oT[nT++] = c;
            if (nT > 0)
                throw std::runtime_error(
                    "Still not supported common dimensions between the sparse input "
                    "matrix, the dense input matrix, and the dense output matrix");

            // Split the od into the non-blocked dimensions (Ds) and the blocked ones (ds) and the Kronecker ones (kds)
            Order<Nd> oDs, ods, okds;
            Coor<Nd> dimDs, dimds, dimkds;
            std::size_t nDs = 0, nds = 0, nkds = 0;
            for (std::size_t i = 0; i < nd; ++i) {
                if (blockd[i] > 1) {
                    if (blockd[i] != dimd[i])
                        throw std::runtime_error(
                            "Still not supported partially blocking a dimension");
                    if (krond[i] > 1)
                        throw std::runtime_error(
                            "Invalid simultaneous blocking and Kronecker blocking");
                    ods[nds] = od[i];
                    dimds[nds++] = dimd[i];
                } else if (krond[i] > 1) {
                    if (krond[i] != dimd[i])
                        throw std::runtime_error(
                            "Still not supported partially blocking a dimension");
                    okds[nkds] = od[i];
                    dimkds[nkds++] = dimd[i];
                } else {
                    oDs[nDs] = od[i];
                    dimDs[nDs++] = dimd[i];
                }
            }

            // Split the oi into the non-blocked dimensions (Is) and the blocked ones (is)
            Order<Ni> oIs, ois, okis;
            Coor<Ni> dimIs, dimis, dimkis;
            std::size_t nIs = 0, nis = 0, nkis = 0;
            for (std::size_t i = 0; i < ni; ++i) {
                if (blocki[i] > 1) {
                    if (dimi[i] > 0 && blocki[i] != dimi[i])
                        throw std::runtime_error(
                            "Still not supported partially blocking a dimension");
                    if (kroni[i] > 1)
                        throw std::runtime_error(
                            "Invalid simultaneous blocking and Kronecker blocking");
                    ois[nis] = oi[i];
                    dimis[nis++] = dimi[i];
                } else if (kroni[i] > 1) {
                    if (kroni[i] != dimi[i])
                        throw std::runtime_error(
                            "Still not supported partially blocking a dimension");
                    okis[nkis] = oi[i];
                    dimkis[nkis++] = dimi[i];
                } else {
                    oIs[nIs] = oi[i];
                    dimIs[nIs++] = dimi[i];
                }
            }

            // Find all common labels in ox to oy
            Order<Nx> oC;
            std::size_t nC = 0;
            volC = (volume(dimx) * volume(dimy) == 0 ? 0 : 1);
            enum { None, ContractWithDomain, ContractWithImage } kindx = None, kindy = None;
            bool powerFoundOnx = false;
            for (std::size_t i = 0; i < nx; ++i) {
                char c = ox[i];
                if (std::find(oi.begin(), oi.end(), c) != oi.end()) {
                    if (kindx == ContractWithDomain)
                        throw std::runtime_error(
                            "Unsupported to contract dense input tensor with domain and image "
                            "dimensions of the sparse tensor");
                    else
                        kindx = ContractWithImage;
                } else if (std::find(od.begin(), od.end(), c) != od.end()) {
                    if (kindx == ContractWithImage)
                        throw std::runtime_error(
                            "Unsupported to contract dense input tensor with domain and image "
                            "dimensions of the sparse tensor");
                    else
                        kindx = ContractWithDomain;
                } else if (okr != 0 && c == okr) {
                    powerFoundOnx = true;
                    if (dimx[i] > 1)
                        throw std::runtime_error(
                            "The power dimension on the input vector has a size larger than one");
                } else if (std::find(oy.begin(), oy.end(), c) != oy.end()) {
                    oC[nC++] = c;
                    volC *= dimx[i];
                } else if (dimx[i] > 1) {
                    throw std::runtime_error(
                        "Dimension label for the dense input vector doesn't match the "
                        "input sparse dimensions nor the dense output dimensions");
                }
            }
            Order<Nx> oxr;
            std::size_t nxr = Nx - nx;
            for (std::size_t i = 0; i < nxr; ++i) oxr[i] = ox[nx + i];

            // Find all common labels in ox to oy
            bool powerFoundOny = false;
            int power = 1;
            for (std::size_t i = 0; i < ny; ++i) {
                char c = oy[i];
                if (std::find(oi.begin(), oi.end(), c) != oi.end()) {
                    if (kindy == ContractWithDomain)
                        throw std::runtime_error(
                            "Unsupported to an output tensor with dimensions both from the domain "
                            "and the image on the sparse matrix");
                    else
                        kindy = ContractWithImage;
                } else if (std::find(od.begin(), od.end(), c) != od.end()) {
                    if (kindy == ContractWithImage)
                        throw std::runtime_error(
                            "Unsupported to an output tensor with dimensions both from the domain "
                            "and the image on the sparse matrix");
                    else
                        kindy = ContractWithDomain;
                } else if (okr != 0 && c == okr) {
                    powerFoundOny = true;
                    power = dimy[i];
                } else if (std::find(ox.begin(), ox.end(), c) != ox.end()) {
                    // Do nothing
                } else if (dimy[i] > 1) {
                    throw std::runtime_error(
                        "Dimension label for the dense input vector doesn't match the "
                        "input sparse dimensions nor the dense output dimensions");
                }
            }
            Order<Nx> oyr;
            std::size_t nyr = Ny - ny;
            for (std::size_t i = 0; i < nyr; ++i) oyr[i] = oy[ny + i];

            // Check okr: either zero or a label on oy
            if (okr != 0 && (!powerFoundOnx || !powerFoundOny))
                throw std::runtime_error(
                    "The power dimension isn't on the input or output dense tensor");

            // If power, dimi should be equal to dimd
            if (okr != 0) {
                if (Nd != Ni)
                    throw std::runtime_error(
                        "Unsupported power for operators that have different number of dimensions "
                        "for the domain and image spaces");
                if (power > 1)
                    for (unsigned int i = 0, i1 = std::min(Nd, Ni); i < i1; ++i)
                        if (dimd[i] != dimi[i])
                            throw std::runtime_error("When using powers the domain and the image "
                                                     "of the sparse operator should be the same");
            }

            // Check that kindx and kindy aren't None and they are distinct
            if (kindx == None)
                throw std::runtime_error(
                    "Unsupported to contract sparse and dense tensors without a common dimension");
            if (kindy == None)
                throw std::runtime_error(
                    "Unsupported to the resulting dense tensor of contracting sparse and dense "
                    "tensors has no common dimension with the sparse tensor");
            if (kindx == kindy) throw std::runtime_error("Invalid contraction");

            if (!is_kron) {
                // Contraction with the blocking:
                // Check that ox should one of (okr,C,D,d) or (okr,D,d,C) or (okr,C,I,i) or (okr,I,i,C)

                if (kindx == ContractWithDomain) {
                    auto sCx = std::search(ox.begin(), ox.end(), oC.begin(), oC.begin() + nC);
                    auto sDx = std::search(ox.begin(), ox.end(), oDs.begin(), oDs.begin() + nDs);
                    auto sdx = std::search(ox.begin(), ox.end(), ods.begin(), ods.begin() + nds);
                    lx = (nC == 0 || ((nDs == 0 || sCx < sDx) && (nds == 0 || sCx < sdx)))
                             ? ColumnMajor
                             : RowMajor;
                    if ((okr != 0 && ox[0] != okr) || (nC > 0 && sCx == ox.end()) ||
                        (nDs > 0 && sDx == ox.end()) || (nds > 0 && sdx == ox.end()) ||
                        (nDs > 0 && nds > 0 && sDx > sdx) ||
                        (nC > 0 && nDs > 0 && nds > 0 && sDx < sCx && sCx < sdx) ||
                        (volC > 1 && lx == RowMajor && xylayout == ColumnMajorForXandY)) {
                        lx = preferred_layout;
                        sug_ox = (lx == ColumnMajor
                                      ? concat<Nx>(okr, oC, nC, oDs, nDs, ods, nds, oxr, nxr)
                                      : concat<Nx>(okr, oDs, nDs, ods, nds, oC, nC, oxr, nxr));
                    } else {
                        sug_ox = ox;
                    }

                    auto sCy = std::search(oy.begin(), oy.end(), oC.begin(), oC.begin() + nC);
                    auto sIy = std::search(oy.begin(), oy.end(), oIs.begin(), oIs.begin() + nIs);
                    auto siy = std::search(oy.begin(), oy.end(), ois.begin(), ois.begin() + nis);
                    ly = (nC == 0 || ((nIs == 0 || sCy < sIy) && (nds == 0 || sCy < siy)))
                             ? ColumnMajor
                             : RowMajor;
                    if ((okr != 0 && oy[0] != okr) || (nC > 0 && sCy == oy.end()) ||
                        (nIs > 0 && sIy == oy.end()) || (nis > 0 && siy == oy.end()) ||
                        (nIs > 0 && nis > 0 && sIy > siy) ||
                        (nC > 0 && nIs > 0 && nis > 0 && sIy < sCy && sCy < siy) ||
                        (lx != ly && xylayout == SameLayoutForXAndY) ||
                        (ly == RowMajor && xylayout == ColumnMajorForY) ||
                        (lx == RowMajor && xylayout == ColumnMajorForXandY)) {
                        ly = (xylayout == SameLayoutForXAndY
                                  ? lx
                                  : (xylayout == ColumnMajorForY ? ColumnMajor : preferred_layout));
                        sug_oy = (ly == ColumnMajor
                                      ? concat<Ny>(okr, oC, nC, oIs, nIs, ois, nis, oyr, nyr)
                                      : concat<Ny>(okr, oIs, nIs, ois, nis, oC, nC, oyr, nyr));
                    } else {
                        sug_oy = oy;
                    }
                } else {
                    auto sCx = std::search(ox.begin(), ox.end(), oC.begin(), oC.begin() + nC);
                    auto sIx = std::search(ox.begin(), ox.end(), oIs.begin(), oIs.begin() + nIs);
                    auto six = std::search(ox.begin(), ox.end(), ois.begin(), ois.begin() + nis);
                    lx = (nC == 0 || ((nIs == 0 || sCx < sIx) && (nis == 0 || sCx < six)))
                             ? ColumnMajor
                             : RowMajor;
                    if ((okr != 0 && ox[0] != okr) || (nC > 0 && sCx == ox.end()) ||
                        (nIs > 0 && sIx == ox.end()) || (nis > 0 && six == ox.end()) ||
                        (nIs > 0 && nis > 0 && sIx > six) ||
                        (nC > 0 && nIs > 0 && nis > 0 && sIx < sCx && sCx < six) ||
                        (lx == RowMajor && xylayout == ColumnMajorForXandY)) {
                        lx = preferred_layout;
                        sug_ox = (lx == ColumnMajor
                                      ? concat<Nx>(okr, oC, nC, oIs, nIs, ois, nis, oxr, nxr)
                                      : concat<Nx>(okr, oIs, nIs, ois, nis, oC, nC, oxr, nxr));
                    } else {
                        sug_ox = ox;
                    }

                    auto sCy = std::search(oy.begin(), oy.end(), oC.begin(), oC.begin() + nC);
                    auto sDy = std::search(oy.begin(), oy.end(), oDs.begin(), oDs.begin() + nDs);
                    auto sdy = std::search(oy.begin(), oy.end(), ods.begin(), ods.begin() + nds);
                    ly = (nC == 0 || ((nDs == 0 || sCy < sDy) && (nds == 0 || sCy < sdy)))
                             ? ColumnMajor
                             : RowMajor;
                    if ((okr != 0 && oy[0] != okr) || (nC > 0 && sCy == oy.end()) ||
                        (nDs > 0 && sDy == oy.end()) || (nds > 0 && sdy == oy.end()) ||
                        (nDs > 0 && nds > 0 && sDy > sdy) ||
                        (nC > 0 && nDs > 0 && nds > 0 && sDy < sCy && sCy < sdy) ||
                        (lx != ly && xylayout == SameLayoutForXAndY) ||
                        (ly == RowMajor && xylayout == ColumnMajorForY) ||
                        (ly == RowMajor && xylayout == ColumnMajorForXandY)) {

                    } else {
                        ly = (xylayout == SameLayoutForXAndY
                                  ? lx
                                  : (xylayout == ColumnMajorForY ? ColumnMajor : preferred_layout));
                        sug_oy = (ly == ColumnMajor
                                      ? concat<Ny>(okr, oC, nC, oDs, nDs, ods, nds, oyr, nyr)
                                      : concat<Ny>(okr, oDs, nDs, ods, nds, oC, nC, oyr, nyr));
                    }
                }
            } else { // !is_kron
                // Contraction with the blocking and the Kronecker blocking
                // Check that ox should (okr,D,d,C,kd) or (okr,I,i,C,ki) for row major,
                // and (okr,kd,C,D,d) or (okr,ki,C,I,i) for column major.

                Order<Nx> sug_ox_row_major, sug_ox_col_major;
                Order<Ny> sug_oy_row_major, sug_oy_col_major;
                if (kindx == ContractWithDomain) {
                    sug_ox_row_major =
                        concat<Nx>(okr, oDs, nDs, ods, nds, oC, nC, okds, nkds, oxr, nxr);
                    sug_ox_col_major =
                        concat<Nx>(okr, okds, nkds, oC, nC, oDs, nDs, ods, nds, oxr, nxr);
                    sug_oy_row_major =
                        concat<Ny>(okr, oIs, nIs, ois, nis, oC, nC, okis, nkis, oyr, nyr);
                    sug_oy_col_major =
                        concat<Ny>(okr, okis, nkis, oC, nC, oIs, nIs, ois, nis, oyr, nyr);
                } else {
                    sug_ox_row_major =
                        concat<Nx>(okr, oIs, nIs, ois, nis, oC, nC, okis, nkis, oxr, nxr);
                    sug_ox_col_major =
                        concat<Nx>(okr, okis, nkis, oC, nC, oIs, nIs, ois, nis, oxr, nxr);
                    sug_oy_row_major =
                        concat<Ny>(okr, oDs, nDs, ods, nds, oC, nC, okds, nkds, oyr, nyr);
                    sug_oy_col_major =
                        concat<Ny>(okr, okds, nkds, oC, nC, oDs, nDs, ods, nds, oyr, nyr);
                }
                if (ox == sug_ox_row_major && oy == sug_oy_row_major) {
                    ly = lx = RowMajor;
                } else if (ox == sug_ox_col_major && oy == sug_oy_col_major) {
                    ly = lx = ColumnMajor;
                } else {
                    ly = lx = preferred_layout;
                }
                if ((xylayout == ColumnMajorForY && ly != ColumnMajor) ||
                    (xylayout == ColumnMajorForXandY && lx != ColumnMajor)) {
                    lx = ly = ColumnMajor;
                }
                if (xylayout == RowMajorForXandY && lx != RowMajor) { lx = ly = RowMajor; }
                ly = lx = preferred_layout;
                sug_ox = lx == RowMajor ? sug_ox_row_major : sug_ox_col_major;
                sug_oy = ly == RowMajor ? sug_oy_row_major : sug_oy_col_major;
            }

            if (okr != 0 && power > 1) {
                sug_oy_trans = sug_oy;
                for (unsigned int i = 0; i < Ny; ++i) {
                    if (kindx == ContractWithDomain) {
                        auto s = std::find(oi.begin(), oi.end(), sug_oy_trans[i]);
                        if (s != oi.end()) sug_oy_trans[i] = od[s - oi.begin()];
                    } else {
                        auto s = std::find(od.begin(), od.end(), sug_oy_trans[i]);
                        if (s != od.end()) sug_oy_trans[i] = oi[s - od.begin()];
                    }
                }
            }

            transSp = kindx == ContractWithImage;
        }

        /// RSB operator - tensor multiplication
        /// \param pim: partitioning of the RSB operator image in consecutive ranges
        /// \param pdm: pseudo-partitioning of the RSB operator domain in consecutive ranges
        /// \param oim: dimension labels for the RSB operator image space
        /// \param odm: dimension labels for the RSB operator domain space
        /// \param blockim: image dimensions of the block
        /// \param blockdm: domain dimensions of the block
        /// \param vm: BSR tensor components
        /// \param px: partitioning of the right tensor in consecutive ranges
        /// \param ox: dimension labels for the right operator
        /// \param vx: right input tensor components
        /// \param py: partitioning of the resulting tensor in consecutive ranges
        /// \param oy: dimension labels for the output tensor
        /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
        /// \param vy: output tensor components
        /// \param co: coordinate linearization order
        /// \param copyadd: either copy or add the multiplication result into the output tensor y

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T,
                  typename XPU>
        void local_bsr_krylov(const BSR<Nd, Ni, T, XPU> &bsr, const Order<Ni> &oim,
                              const Order<Nd> &odm, const Coor<Nx> &dimx, const Order<Nx> &ox,
                              vector<T, XPU> vx, const Coor<Ny> &dimy, const Order<Ny> &oy,
                              char okr, vector<T, XPU> vy) {

            tracker<XPU> _t(std::string("local BSR matvec (") + bsr.implementation() +
                                std::string(")"),
                            vx.ctx());

            // Quick exit
            if (volume(dimy) == 0) return;

            // Check inputs and get the common dimensions
            bool transSp;
            MatrixLayout lx, ly;
            std::size_t volC;
            Order<Nx> sug_ox;
            Order<Ny> sug_oy;
            Order<Ny> sug_oy_trans;
            bool is_kron =
                (volume(bsr.v.krond) > 1 || volume(bsr.v.kroni) > 1 || bsr.v.kron_it.size() > 0);
            local_bsr_krylov_check(bsr.v.dimi, bsr.v.dimd, oim, odm, bsr.v.blocki, bsr.v.blockd,
                                   bsr.v.kroni, bsr.v.krond, is_kron, dimx, ox, dimy, oy, okr,
                                   true /* permissive */, bsr.allowLayout, bsr.preferredLayout,
                                   bsr.v.co, transSp, lx, ly, volC, sug_ox, sug_oy, sug_oy_trans);
            if (sug_ox != ox || sug_oy != oy)
                throw std::runtime_error(
                    "Unsupported layout for the input and output dense tensors");

            std::size_t vold = volume(bsr.v.dimd), voli = volume(bsr.v.dimi);
            IndexType ki = volume(bsr.v.kroni);
            IndexType kd = volume(bsr.v.krond);
            if (vold == 0 || voli == 0) return;
            // Layout for row major: (kd,n,bd,rows)
            // Layout for column major: (bd,rows,n,kd)
            IndexType ldx = lx == ColumnMajor ? (!transSp ? vold / kd : voli / ki)
                                              : (!transSp ? kd : ki) * volC;
            IndexType ldy = ly == ColumnMajor ? (!transSp ? voli / ki : vold / kd)
                                              : (!transSp ? ki : kd) * volC;

            // Do the contraction
            _t.flops = bsr.getFlopsPerMatvec(volC, lx);
            _t.memops = bsr.getMemopsPerMatvec(volC, lx);
            _t.arity = volC;
            bsr(transSp, vx, ldx, lx, vy, ldy, ly, volC);
        }

        /// Get the partitions for the dense input and output tensors
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param sizex: number of elements to operate in each dimension
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename Comm>
        std::pair<Proc_ranges<Nx>, Proc_ranges<Ny>>
        get_output_partition(Proc_ranges<Nd> pd, const Order<Nd> &od, Proc_ranges<Ni> pi,
                             const Order<Ni> &oi, Proc_ranges<Nx> px, const Order<Nx> &ox,
                             const Order<Nx> &sug_ox, const Coor<Nx> &sizex, const Order<Ny> &oy,
                             const Order<Ny> &sug_oy, char okr, const Comm &comm,
                             bool just_local = false) {
            assert(pd.size() == pi.size() && pi.size() == px.size());

            // Find partition on cache
            Order<Nd + Ni> om = concat(od, oi);
            using Key =
                std::tuple<Proc_ranges<Nd>, Proc_ranges<Ni>, Proc_ranges<Nx>, Coor<Nx>,
                           PairPerms<Nd + Ni, Nx>, PairPerms<Nx, Ny>, PairPerms<Nd + Ni, Ny>,
                           PairPerms<Nx, Nx>, PairPerms<Ny, Ny>, char, bool, int>;
            struct cache_tag {};
            auto cache = getCache<Key, std::pair<Proc_ranges<Nx>, Proc_ranges<Ny>>, TupleHash<Key>,
                                  cache_tag>(Cpu{});
            Key key{pd,
                    pi,
                    px,
                    sizex,
                    get_perms(om, ox),
                    get_perms(ox, oy),
                    get_perms(om, oy),
                    get_perms(ox, sug_ox),
                    get_perms(oy, sug_oy),
                    okr,
                    just_local,
                    just_local ? comm.rank : 0};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second.value;

            // Find position of the power label
            int power_pos = 0;
            if (okr != 0) {
                auto it_oy = std::find(oy.begin(), oy.end(), okr);
                if (it_oy == oy.end())
                    throw std::runtime_error("The dimension label `okr` wasn't found on `oy`");
                power_pos = it_oy - oy.begin();
            }

            // Create partition
            Proc_ranges<Nx> pxr(px.size());
            Proc_ranges<Ny> pyr(px.size());
            for (unsigned int i = 0; i < pi.size(); ++i) {
                if (just_local && i != comm.rank) continue;
                pxr[i].resize(pi[i].size());
                pyr[i].resize(pi[i].size());
                for (unsigned int j = 0; j < pi[i].size(); ++j) {
                    pxr[i][j][0] = get_dimensions(om, concat(pd[i][j][0], pi[i][j][0]), ox, {{}},
                                                  sug_ox, false, 0);
                    pxr[i][j][1] = get_dimensions(om, concat(pd[i][j][1], pi[i][j][1]), ox, sizex,
                                                  sug_ox, false, 1);
                    pyr[i][j][0] = get_dimensions(om, concat(pd[i][j][0], pi[i][j][0]), ox, {{}},
                                                  sug_oy, false, 0);
                    pyr[i][j][1] = get_dimensions(om, concat(pd[i][j][1], pi[i][j][1]), ox, sizex,
                                                  sug_oy, false, 1);
                    if (okr != 0) {
                        pyr[i][j][0][power_pos] = 0;
                        pyr[i][j][1][power_pos] = 1;
                    }

                    // Normalize range
                    if (volume(pxr[i][j][1]) == 0) pxr[i][j][0] = pxr[i][j][1] = Coor<Nx>{{}};
                    if (volume(pyr[i][j][1]) == 0) pyr[i][j][0] = pyr[i][j][1] = Coor<Ny>{{}};
                }
            }
            cache.insert(key, {pxr, pyr}, storageSize(pxr) + storageSize(pyr));

            return {pxr, pyr};
        }

        /// RSB operator - tensor multiplication
        /// \param bsr: BSR tensor components
        /// \param oim: dimension labels for the RSB operator image space
        /// \param odm: dimension labels for the RSB operator domain space
        /// \param px: partitioning of the right tensor in consecutive ranges
        /// \param ox: dimension labels for the right operator
        /// \param vx: right input tensor components
        /// \param py: partitioning of the resulting tensor in consecutive ranges
        /// \param oy: dimension labels for the output tensor
        /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
        /// \param vy: output tensor components
        /// \param co: coordinate linearization order
        /// \param copyadd: either copy or add the multiplication result into the output tensor y

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T,
                  typename Comm, typename XPU0, typename XPU1>
        Request bsr_krylov(T alpha, const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> bsr,
                           const Order<Ni> oim, const Order<Nd> odm, const Proc_ranges<Nx> &px,
                           const Order<Nx> &ox, const Coor<Nx> &fromx, const Coor<Nx> &sizex,
                           const Coor<Nx> &dimx, const Components_tmpl<Nx, T, XPU0, XPU1> &vx,
                           T beta, const Proc_ranges<Ny> py, const Order<Ny> oy,
                           const Coor<Ny> fromy, const Coor<Ny> sizey, const Coor<Ny> dimy,
                           char okr, const Components_tmpl<Ny, T, XPU0, XPU1> vy, Comm comm,
                           CoorOrder co, bool just_local = false) {

            // Check that common arguments have the same value in all processes

            if (getDebugLevel() >= 1) {
                if (!just_local) {
                    struct tag_type {}; // For hashing template arguments
                    check_consistency(std::make_tuple(std::string("bsr_krylov"), alpha, oim, odm,
                                                      px, ox, fromx, sizex, dimx, beta, py, oy,
                                                      fromy, sizey, dimy, okr, comm.nprocs, co,
                                                      typeid(tag_type).hash_code()),
                                      comm);
                }
                for (const auto &i : bsr.c.first) sync(i.v.it.ctx());
                for (const auto &i : bsr.c.second) sync(i.v.it.ctx());
            }

            tracker<Cpu> _t("distributed BSR matvec", Cpu{0});

            // Check that co is that same as BSR
            if (co != bsr.co)
                throw std::runtime_error("Unsupported to use a different coordinate ordering that "
                                         "one used to create the matrix");

            /// Get power and bring pieces of BSR operator to do powers without extra communications
            unsigned int power = 1;
            unsigned int power_pos = 0;
            if (okr != 0) {
                auto it_oy = std::find(oy.begin(), oy.end(), okr);
                if (it_oy == oy.end())
                    throw std::runtime_error("The dimension label `okr` wasn't found on `oy`");
                power_pos = it_oy - oy.begin();
                power = sizey[power_pos];
            }

            // Generate the partitioning and the storage for the dense matrix input and output tensor
            Order<Nx> sug_ox = ox;
            Order<Ny> sug_oy = oy;
            Order<Ny> sug_oy_trans;
            if (bsr.c.first.size() > 0) {
                bool transSp;
                MatrixLayout lx, ly;
                std::size_t volC;
                bool is_kron =
                    (volume(bsr.c.first[0].v.krond) > 1 || volume(bsr.c.first[0].v.kroni) > 1 ||
                     bsr.c.first[0].v.kron_it.size() > 0);
                local_bsr_krylov_check(bsr.dimi, bsr.dimd, oim, odm, bsr.blocki, bsr.blockd,
                                       bsr.kroni, bsr.krond, is_kron, sizex, ox, sizey, oy, okr,
                                       false /* not permissive */, bsr.c.first[0].allowLayout,
                                       bsr.c.first[0].preferredLayout, co, transSp, lx, ly, volC,
                                       sug_ox, sug_oy, sug_oy_trans);
            } else if (bsr.c.second.size() > 0) {
                bool transSp;
                MatrixLayout lx, ly;
                std::size_t volC;
                bool is_kron =
                    (volume(bsr.c.second[0].v.krond) > 1 || volume(bsr.c.second[0].v.kroni) > 1 ||
                     bsr.c.second[0].v.kron_it.size() > 0);
                local_bsr_krylov_check(bsr.dimi, bsr.dimd, oim, odm, bsr.blocki, bsr.blockd,
                                       bsr.kroni, bsr.krond, is_kron, sizex, ox, sizey, oy, okr,
                                       false /* not permissive */, bsr.c.second[0].allowLayout,
                                       bsr.c.second[0].preferredLayout, co, transSp, lx, ly, volC,
                                       sug_ox, sug_oy, sug_oy_trans);
            }
            Coor<Nx> sug_dimx = reorder_coor(dimx, find_permutation(ox, sug_ox));
            Coor<Ny> sizey0 = sizey;
            if (power > 1) sizey0[power_pos] = 1;
            Coor<Ny> sug_sizey = reorder_coor(sizey0, find_permutation(oy, sug_oy));

            auto pxy_ = get_output_partition(bsr.pd, odm, bsr.pi, oim, px, ox, sug_ox, sizex, oy,
                                             sug_oy, okr, comm, just_local);

            // Copy the input dense tensor to a compatible layout to the sparse tensor
            Proc_ranges<Nx> px_ = pxy_.first;
            ForceLocal force_local = (just_local ? doForceLocal : dontForceLocal);
            auto vx_and_req = reorder_tensor_request(
                px, ox, fromx, sizex, dimx, vx, px_, sug_dimx, sug_ox, get_mock_components(bsr),
                comm, co, power > 1 /* force copy when power > 1 */, doCacheAlloc, force_local);
            Components_tmpl<Nx, T, XPU0, XPU1> vx_ = vx_and_req.first;

            // Scale the output vector
            copy<Ny, Ny, T>(beta, py, fromy, sizey, dimy, oy, toConst(vy), py, fromy, dimy, oy, vy,
                            comm, EWOp::Copy{}, co, force_local);

            Request bsr_req = [=] {
                tracker<Cpu> _t("distributed BSR matvec", Cpu{0});

                // Wait for the data to be ready
                wait(vx_and_req.second);

                // Allocate the output tensor
                Proc_ranges<Ny> py_ = pxy_.second;
                Components_tmpl<Ny, T, XPU0, XPU1> vy_ =
                    like_this_components(py_, vx_, comm, doCacheAlloc);

                // Do contraction
                for (unsigned int p = 0; p < power; ++p) {
                    for (unsigned int i = 0; i < bsr.c.first.size(); ++i) {
                        const unsigned int componentId = bsr.c.first[i].v.componentId;
                        local_bsr_krylov<Nd, Ni, Nx, Ny, T>(
                            bsr.c.first[i], oim, odm, px_[comm.rank][componentId][1], sug_ox,
                            vx_.first[i].it, py_[comm.rank][componentId][1], sug_oy, okr,
                            vy_.first[i].it);
                    }
                    for (unsigned int i = 0; i < bsr.c.second.size(); ++i) {
                        const unsigned int componentId = bsr.c.second[i].v.componentId;
                        local_bsr_krylov<Nd, Ni, Nx, Ny, T>(
                            bsr.c.second[i], oim, odm, px_[comm.rank][componentId][1], sug_ox,
                            vx_.second[i].it, py_[comm.rank][componentId][1], sug_oy, okr,
                            vy_.second[i].it);
                    }

                    // Copy the result to final tensor
                    Coor<Ny> fromyi = fromy;
                    if (p > 0) fromyi[power_pos] += p;
                    if (does_proc_ranges_self_intersect(bsr.pd, bsr.dimd)) {
                        Proc_ranges<Ny> py0_ = remove_repetitions(py_, sug_sizey);
                        Components_tmpl<Ny, T, XPU0, XPU1> vy0_ = reorder_tensor(
                            py_, sug_oy, {{}}, sug_sizey, sug_sizey, vy_, py0_, sug_sizey, sug_oy,
                            comm, co, false /* don't force copy */, doCacheAlloc);
                        copy<Ny, Ny, T>(T{1}, py_, {{}}, sug_sizey, sug_sizey, sug_oy, toConst(vy_),
                                        py0_, {{}}, sug_sizey, sug_oy, vy0_, comm, EWOp::Copy{}, co,
                                        force_local);
                        copy<Ny, Ny, T>(p == 0 ? alpha : T{1}, py0_, {{}}, sug_sizey, sug_sizey,
                                        sug_oy, toConst(vy0_), py, fromyi, dimy, oy, vy, comm,
                                        EWOp::Add{}, co, force_local);
                    } else {
                        copy<Ny, Ny, T>(p == 0 ? alpha : T{1}, py_, {{}}, sug_sizey, sug_sizey,
                                        sug_oy, toConst(vy_), py, fromyi, dimy, oy, vy, comm,
                                        EWOp::Add{}, co, force_local);
                    }

                    // Copy the result into x for doing the next power
                    if (p == power - 1) break;
                    copy<Nx, Nx, T>(T{0}, px_, {{}}, sug_dimx, sug_dimx, sug_ox, toConst(vx_), px_,
                                    {{}}, sug_dimx, sug_ox, vx_, comm, EWOp::Copy{}, co,
                                    force_local);
                    copy<Ny, Nx, T>(T{1}, py_, {{}}, sug_sizey, sug_sizey, sug_oy_trans,
                                    toConst(vy_), px_, {{}}, sug_dimx, sug_ox, vx_, comm,
                                    EWOp::Copy{}, co, force_local);
                }
            };

            _t.stop();

            // Do the contraction now if we have all the data ready; otherwise, postpone
            Request r;
            if (vx_and_req.second)
                r = bsr_req;
            else
                wait(bsr_req);

            if (getDebugLevel() >= 1) {
                for (const auto &i : bsr.c.first) sync(i.v.it.ctx());
                for (const auto &i : bsr.c.second) sync(i.v.it.ctx());
                if (!just_local) barrier(comm);
            }

            return r;
        }

        /// Return optimal label orderings for input/output tensors in RSB operator - tensor multiplication
        /// \param bsr: BSR tensor components
        /// \param oim: dimension labels for the RSB operator image space
        /// \param odm: dimension labels for the RSB operator domain space
        /// \param ox: dimension labels for the right operator
        /// \param sizex: dimensions for the right tensor
        /// \param py: partitioning of the resulting tensor in consecutive ranges
        /// \param oy: dimension labels for the output tensor
        /// \param sizey: dimensions for the output tensor
        /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
        /// \param co: coordinate linearization order
        /// \param sug_ox: (out) suggested dimension labels for the right tensor
        /// \param sug_oy: (out) suggested dimension labels for the output tensor

        template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T,
                  typename XPU0, typename XPU1>
        void suggested_orders_for_bsr_krylov(const BSRComponents_tmpl<Nd, Ni, T, XPU0, XPU1> bsr,
                                             const Order<Ni> oim, const Order<Nd> odm,
                                             const Order<Nx> &ox, const Coor<Nx> &sizex,
                                             const Order<Ny> oy, const Coor<Ny> sizey, char okr,
                                             CoorOrder co, Order<Nx> &sug_ox, Order<Ny> &sug_oy) {

            // Check that co is that same as BSR
            if (co != bsr.co)
                throw std::runtime_error("Unsupported to use a different coordinate ordering that "
                                         "one used to create the matrix");

            /// Get power and bring pieces of BSR operator to do powers without extra communications
            if (okr != 0) {
                auto it_oy = std::find(oy.begin(), oy.end(), okr);
                if (it_oy == oy.end())
                    throw std::runtime_error("The dimension label `okr` wasn't found on `oy`");
            }

            // Generate the partitioning and the storage for the dense matrix input and output tensor
            if (bsr.c.first.size() > 0) {
                bool transSp;
                MatrixLayout lx, ly;
                std::size_t volC;
                Order<Ny> sug_oy_trans;
                bool is_kron =
                    (volume(bsr.c.first[0].v.krond) > 1 || volume(bsr.c.first[0].v.kroni) > 1 ||
                     bsr.c.first[0].v.kron_it.size() > 0);
                local_bsr_krylov_check(bsr.dimi, bsr.dimd, oim, odm, bsr.blocki, bsr.blockd,
                                       bsr.kroni, bsr.krond, is_kron, sizex, ox, sizey, oy, okr,
                                       false /* not permissive */, bsr.c.first[0].allowLayout,
                                       bsr.c.first[0].preferredLayout, co, transSp, lx, ly, volC,
                                       sug_ox, sug_oy, sug_oy_trans);
            } else if (bsr.c.second.size() > 0) {
                bool transSp;
                MatrixLayout lx, ly;
                std::size_t volC;
                Order<Ny> sug_oy_trans;
                bool is_kron =
                    (volume(bsr.c.second[0].v.krond) > 1 || volume(bsr.c.second[0].v.kroni) > 1 ||
                     bsr.c.second[0].v.kron_it.size() > 0);
                local_bsr_krylov_check(bsr.dimi, bsr.dimd, oim, odm, bsr.blocki, bsr.blockd,
                                       bsr.kroni, bsr.krond, is_kron, sizex, ox, sizey, oy, okr,
                                       false /* not permissive */, bsr.c.second[0].allowLayout,
                                       bsr.c.second[0].preferredLayout, co, transSp, lx, ly, volC,
                                       sug_ox, sug_oy, sug_oy_trans);
            }
        }
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Create BSR sparse operator
    /// \param pim: partitioning of the RSB operator image in consecutive ranges
    /// \param pdm: pseudo-partitioning of the RSB operator domain in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param blockim: image dimensions of the block
    /// \param blockdm: domain dimensions of the block
    /// \param blockImFast: whether the blocks are stored with the image indices the fastest
    /// \param ii: ii[i] is the number of nonzero blocks on the i-th blocked image operator element
    /// \param ctx: context for ii
    /// \param jj: domain coordinates of the nonzero blocks of RSB operator
    /// \param ctx: context for jj
    /// \param v: nonzero values
    /// \param ctx: context for v
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param bsrh (out) handle to BSR nonzero pattern
    ///
    /// NOTE: keep allocated the space pointed out by ii, jj, and v until calling `destroy_bsr`.

    template <std::size_t Nd, std::size_t Ni, typename T>
    void create_bsr(const PartitionItem<Ni> *pim, const Coor<Ni> &dimi,
                    const PartitionItem<Nd> *pdm, const Coor<Nd> &dimd, int ncomponents,
                    const Coor<Ni> &blockim, const Coor<Nd> &blockdm, bool blockImFast,
                    IndexType **ii, const Context *ctx_ii, Coor<Nd> **jj, const Context *ctx_jj,
                    const T **v, const Context *ctx_v, MPI_Comm mpicomm, CoorOrder co,
                    BSR_handle **bsrh, Session session = 0) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *r =
            new detail::BSRComponents<Nd, Ni, T>{detail::get_bsr_components<Nd, Ni, T>(
                (T **)v, ctx_v, ii, ctx_ii, jj, ctx_jj, nullptr, ctx_v, ncomponents, pim, dimi, pdm,
                dimd, blockdm, blockim, detail::ones<Nd>(), detail::ones<Ni>(), blockImFast, comm,
                co, session)};
        *bsrh = r;
    }

    template <std::size_t Nd, std::size_t Ni, typename T>
    void create_bsr(const PartitionItem<Ni> *pim, const Coor<Ni> &dimi,
                    const PartitionItem<Nd> *pdm, const Coor<Nd> &dimd, int ncomponents,
                    const Coor<Ni> &blockim, const Coor<Nd> &blockdm, bool blockImFast,
                    IndexType **ii, Coor<Nd> **jj, const T **v, const Context *ctx,
                    MPI_Comm mpicomm, CoorOrder co, BSR_handle **bsrh, Session session = 0) {

        create_bsr(pim, dimi, pdm, dimd, ncomponents, blockim, blockdm, blockImFast, ii, ctx, jj,
                   ctx, v, ctx, mpicomm, co, bsrh, session);
    }

    /// Create Kronecker BSR sparse operator
    /// \param pim: partitioning of the RSB operator image in consecutive ranges
    /// \param pdm: pseudo-partitioning of the RSB operator domain in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param blockim: image dimensions of the block
    /// \param blockdm: domain dimensions of the block
    /// \param kronim: image dimensions of the Kronecker block
    /// \param krondm: domain dimensions of the Kronecker block
    /// \param blockImFast: whether the blocks and Kronecker blocks are stored with the image indices the fastest
    /// \param ii: ii[i] is the number of nonzero blocks on the i-th blocked image operator element
    /// \param jj: domain coordinates of the nonzero blocks of RSB operator
    /// \param v: nonzero values for the blocks
    /// \param kronv: nonzero values for the Kronecker blocks
    /// \param ctx: context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param bsrh (out) handle to BSR nonzero pattern
    ///
    /// NOTE: keep allocated the space pointed out by ii, jj, and v until calling `destroy_bsr`.

    template <std::size_t Nd, std::size_t Ni, typename T>
    void create_kron_bsr(const PartitionItem<Ni> *pim, const Coor<Ni> &dimi,
                         const PartitionItem<Nd> *pdm, const Coor<Nd> &dimd, int ncomponents,
                         const Coor<Ni> &blockim, const Coor<Nd> &blockdm, const Coor<Ni> &kronim,
                         const Coor<Nd> &krondm, bool blockImFast, IndexType **ii, Coor<Nd> **jj,
                         const T **v, const T **kronv, const Context *ctx, MPI_Comm mpicomm,
                         CoorOrder co, BSR_handle **bsrh, Session session = 0) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *r =
            new detail::BSRComponents<Nd, Ni, T>{detail::get_bsr_components<Nd, Ni, T>(
                (T **)v, ctx, ii, ctx, jj, ctx, (T **)kronv, ctx, ncomponents, pim, dimi, pdm, dimd,
                blockdm, blockim, krondm, kronim, blockImFast, comm, co, session)};
        *bsrh = r;
    }

    /// BSR sparse operator - tensor multiplication
    /// \param bsrh: BSR handle
    /// \param oim: dimension labels for the RSB operator image space
    /// \param odm: dimension labels for the RSB operator domain space
    /// \param px: partitioning of the right tensor in consecutive ranges
    /// \param ox: dimension labels for the right operator
    /// \param vx: data for the second operator
    /// \param py: partitioning of the resulting tensor in consecutive ranges
    /// \param oy: dimension labels for the output tensor
    /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
    /// \param vy: data for the output tensor
    /// \param ctxy: context for each data pointer in vy
    /// \param just_local: compute the local part of the product only

    template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T>
    void bsr_krylov(T alpha, BSR_handle *bsrh, const char *oim, const char *odm,
                    const PartitionItem<Nx> *px, int ncomponents, const char *ox,
                    const Coor<Nx> &fromx, const Coor<Nx> &sizex, const Coor<Nx> &dimx,
                    const T **vx, T beta, const PartitionItem<Ny> *py, const char *oy,
                    const Coor<Ny> &fromy, const Coor<Ny> &sizey, const Coor<Ny> &dimy, char okr,
                    T **vy, const Context *ctx, MPI_Comm mpicomm, CoorOrder co,
                    Request *request = nullptr, bool just_local = false, Session session = 0) {

        Order<Ni> oim_ = detail::toArray<Ni>(oim, "oim");
        Order<Nd> odm_ = detail::toArray<Nd>(odm, "odm");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        Request r = detail::bsr_krylov<Nd, Ni, Nx, Ny, T>(
            alpha, *bsr, oim_, odm_,
            detail::get_from_size(px, ncomponents * comm.nprocs, comm, dimx), ox_, fromx, sizex,
            dimx,
            detail::get_components<Nx>((T **)vx, nullptr, ctx, ncomponents, px, comm, session),
            beta, detail::get_from_size(py, ncomponents * comm.nprocs, comm, dimy), oy_, fromy,
            sizey, dimy, okr,
            detail::get_components<Ny>(vy, nullptr, ctx, ncomponents, py, comm, session), comm, co,
            just_local);

        if (request)
            *request = r;
        else
            wait(r);
    }

    /// Return optimal label orderings for input/output tensors in RSB operator - tensor multiplication
    /// \param bsrh: BSR handle
    /// \param ncomponents: number of components in the BSR handle
    /// \param ctx: context for each data pointer in the BSR handle
    /// \param oim: dimension labels for the RSB operator image space
    /// \param odm: dimension labels for the RSB operator domain space
    /// \param ox: dimension labels for the input tensor
    /// \param sizex: dimensions for the input tensor
    /// \param oy: dimension labels for the output tensor
    /// \param sizey: dimensions for the output tensor
    /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param sug_ox: (out) suggested dimension labels for the input tensor
    /// \param sug_oy: (out) suggested dimension labels for the output tensor

    template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T>
    void suggested_orders_for_bsr_krylov(BSR_handle *bsrh, int ncomponents, const Context *ctx,
                                         const char *oim, const char *odm, const char *ox,
                                         const Coor<Nx> &sizex, const char *oy,
                                         const Coor<Ny> &sizey, char okr, MPI_Comm mpicomm,
                                         CoorOrder co, char *sug_ox, char *sug_oy) {

        Order<Ni> oim_ = detail::toArray<Ni>(oim, "oim");
        Order<Nd> odm_ = detail::toArray<Nd>(odm, "odm");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        Order<Nx> sug_ox_;
        Order<Ny> sug_oy_;
        detail::suggested_orders_for_bsr_krylov<Nd, Ni, Nx, Ny, T>(
            *bsr, oim_, odm_, ox_, sizex, oy_, sizey, okr, co, sug_ox_, sug_oy_);
        if (sug_ox) std::copy_n(sug_ox_.data(), Nx, sug_ox);
        if (sug_oy) std::copy_n(sug_oy_.data(), Ny, sug_oy);
    }

    /// Return the preferred layout for the input and output tensor in `bsr_krylov`
    /// \param bsrh: BSR handle
    /// \param ncomponents: number of components in the BSR handle
    /// \param ctx: context for each data pointer in the BSR handle
    /// \param comm: MPI communicator
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param preferred_layout_for_x: (out) preferred layout for the input tensor for each component
    /// \param preferred_layout_for_y: (out) preferred layout for the output tensor for each component

    template <std::size_t Nd, std::size_t Ni, typename T>
    void bsr_get_preferred_layout(BSR_handle *bsrh, int ncomponents, const Context *ctx,
                                  MPI_Comm mpicomm, CoorOrder co,
                                  MatrixLayout *preferred_layout_for_x,
                                  MatrixLayout *preferred_layout_for_y) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        for (unsigned int i = 0; i < bsr->c.first.size(); ++i) {
            const unsigned int componentId = bsr->c.first[i].v.componentId;
            preferred_layout_for_x[componentId] = bsr->c.first[i].preferredLayout;
            preferred_layout_for_y[componentId] =
                bsr->c.first[i].allowLayout == detail::ColumnMajorForY
                    ? ColumnMajor
                    : bsr->c.first[i].preferredLayout;
        }
        for (unsigned int i = 0; i < bsr->c.second.size(); ++i) {
            const unsigned int componentId = bsr->c.second[i].v.componentId;
            preferred_layout_for_x[componentId] = bsr->c.second[i].preferredLayout;
            preferred_layout_for_y[componentId] =
                bsr->c.second[i].allowLayout == detail::ColumnMajorForY
                    ? ColumnMajor
                    : bsr->c.second[i].preferredLayout;
        }
    }
#endif // SUPERBBLAS_USE_MPI

    /// Create BSR sparse operator
    /// \param pim: partitioning of the RSB operator image in consecutive ranges
    /// \param pdm: pseudo-partitioning of the RSB operator domain in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param blockim: image dimensions of the block
    /// \param blockdm: domain dimensions of the block
    /// \param blockImFast: whether the blocks are stored with the image indices the fastest
    /// \param ii: ii[i] is the number of nonzero blocks on the i-th blocked image operator element
    /// \param ctx_ii: context of ii
    /// \param jj: domain coordinates of the nonzero blocks of RSB operator
    /// \param ctx_jj: context of jj
    /// \param v: nonzero values
    /// \param ctx_v: context for v
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param bsrh (out) handle to BSR nonzero pattern
    ///
    /// NOTE: keep allocated the space pointed out by ii, jj, and v until calling `destroy_bsr`.

    template <std::size_t Nd, std::size_t Ni, typename T>
    void create_bsr(const PartitionItem<Ni> *pim, const Coor<Ni> &dimi,
                    const PartitionItem<Nd> *pdm, const Coor<Nd> &dimd, int ncomponents,
                    const Coor<Ni> &blockim, const Coor<Nd> &blockdm, bool blockImFast,
                    IndexType **ii, const Context *ctx_ii, Coor<Nd> **jj, const Context *ctx_jj,
                    const T **v, const Context *ctx_v, CoorOrder co, BSR_handle **bsrh,
                    Session session = 0) {

        detail::SelfComm comm = detail::get_comm();

        detail::BSRComponents<Nd, Ni, T> *r =
            new detail::BSRComponents<Nd, Ni, T>{detail::get_bsr_components<Nd, Ni, T>(
                (T **)v, ctx_v, ii, ctx_ii, jj, ctx_jj, nullptr, ctx_v, ncomponents, pim, dimi, pdm,
                dimd, blockdm, blockim, detail::ones<Nd>(), detail::ones<Ni>(), blockImFast, comm,
                co, session)};
        *bsrh = r;
    }

    template <std::size_t Nd, std::size_t Ni, typename T>
    void create_bsr(const PartitionItem<Ni> *pim, const Coor<Ni> &dimi,
                    const PartitionItem<Nd> *pdm, const Coor<Nd> &dimd, int ncomponents,
                    const Coor<Ni> &blockim, const Coor<Nd> &blockdm, bool blockImFast,
                    IndexType **ii, Coor<Nd> **jj, const T **v, const Context *ctx, CoorOrder co,
                    BSR_handle **bsrh, Session session = 0) {

        create_bsr(pim, dimi, pdm, dimd, ncomponents, blockim, blockdm, blockImFast, ii, ctx, jj,
                   ctx, v, ctx, co, bsrh, session);
    }

    /// Create Kronecker BSR sparse operator
    /// \param pim: partitioning of the RSB operator image in consecutive ranges
    /// \param pdm: pseudo-partitioning of the RSB operator domain in consecutive ranges
    /// \param ncomponents: number of consecutive components in each MPI rank
    /// \param blockim: image dimensions of the block
    /// \param blockdm: domain dimensions of the block
    /// \param kronim: image dimensions of the Kronecker block
    /// \param krondm: domain dimensions of the Kronecker block
    /// \param blockImFast: whether the blocks are stored with the image indices the fastest
    /// \param ii: ii[i] is the number of nonzero blocks on the i-th blocked image operator element
    /// \param jj: local domain coordinates of the nonzero blocks of the RSB operator
    /// \param v: nonzero values
    /// \param kronv: nonzero values for the Kronecker blocks
    /// \param ctx: context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param bsrh (out) handle to BSR nonzero pattern
    ///
    /// NOTE: keep allocated the space pointed out by ii, jj, and v until calling `destroy_bsr`.

    template <std::size_t Nd, std::size_t Ni, typename T>
    void create_kron_bsr(const PartitionItem<Ni> *pim, const Coor<Ni> &dimi,
                         const PartitionItem<Nd> *pdm, const Coor<Nd> &dimd, int ncomponents,
                         const Coor<Ni> &blockim, const Coor<Nd> &blockdm, const Coor<Ni> &kronim,
                         const Coor<Nd> &krondm, bool blockImFast, IndexType **ii, Coor<Nd> **jj,
                         const T **v, const T **kronv, const Context *ctx, CoorOrder co,
                         BSR_handle **bsrh, Session session = 0) {

        detail::SelfComm comm = detail::get_comm();

        detail::BSRComponents<Nd, Ni, T> *r =
            new detail::BSRComponents<Nd, Ni, T>{detail::get_bsr_components<Nd, Ni, T>(
                (T **)v, ctx, ii, ctx, jj, ctx, (T **)kronv, ctx, ncomponents, pim, dimi, pdm, dimd,
                blockdm, blockim, krondm, kronim, blockImFast, comm, co, session)};
        *bsrh = r;
    }

    /// Destroy RSB sparse operator
    /// \param bsrh: origin RSB handle to copy from

    inline void destroy_bsr(BSR_handle *bsrh) { delete bsrh; }

    /// BSR sparse operator - tensor multiplication
    /// \param bsrh: BSR handle
    /// \param oim: dimension labels for the RSB operator image space
    /// \param odm: dimension labels for the RSB operator domain space
    /// \param px: partitioning of the right tensor in consecutive ranges
    /// \param ox: dimension labels for the right operator
    /// \param fromx: first coordinate to operate from the origin tensor
    /// \param sizex: number of elements to operate in each dimension
    /// \param dimx: dimension size for the origin tensor
    /// \param vx: data for the second operator
    /// \param py: partitioning of the resulting tensor in consecutive ranges
    /// \param oy: dimension labels for the output tensor
    /// \param fromy: first coordinate to copy the result from the destination tensor
    /// \param sizey: number of elements to copy in each dimension
    /// \param dimy: dimension size for the destination tensor
    /// \param okr: dimension label for the RSB operator powers (or zero for a single power)
    /// \param vy: data for the output tensor
    /// \param ctxy: context for each data pointer in vy

    template <std::size_t Nd, std::size_t Ni, std::size_t Nx, std::size_t Ny, typename T>
    void bsr_krylov(T alpha, BSR_handle *bsrh, const char *oim, const char *odm,
                    const PartitionItem<Nx> *px, int ncomponents, const char *ox,
                    const Coor<Nx> &fromx, const Coor<Nx> &sizex, const Coor<Nx> &dimx,
                    const T **vx, T beta, const PartitionItem<Ny> *py, const char *oy,
                    const Coor<Ny> &fromy, const Coor<Ny> &sizey, const Coor<Ny> &dimy, char okr,
                    T **vy, const Context *ctx, CoorOrder co, Request *request = nullptr,
                    Session session = 0) {

        Order<Ni> oim_ = detail::toArray<Ni>(oim, "oim");
        Order<Nd> odm_ = detail::toArray<Nd>(odm, "odm");
        Order<Nx> ox_ = detail::toArray<Nx>(ox, "ox");
        Order<Ny> oy_ = detail::toArray<Ny>(oy, "oy");

        detail::SelfComm comm = detail::get_comm();

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        wait(detail::bsr_krylov<Nd, Ni, Nx, Ny, T>(
            alpha, *bsr, oim_, odm_,
            detail::get_from_size(px, ncomponents * comm.nprocs, comm, dimx), ox_, fromx, sizex,
            dimx,
            detail::get_components<Nx>((T **)vx, nullptr, ctx, ncomponents, px, comm, session),
            beta, detail::get_from_size(py, ncomponents * comm.nprocs, comm, dimy), oy_, fromy,
            sizey, dimy, okr,
            detail::get_components<Ny>(vy, nullptr, ctx, ncomponents, py, comm, session), comm,
            co));
        if (request) *request = Request{};
    }

    /// Return the preferred layout for the input and output tensor in `bsr_krylov`
    /// \param bsrh: BSR handle
    /// \param ncomponents: number of components in the BSR handle
    /// \param ctx: context for each data pointer in the BSR handle
    /// \param comm: MPI communicator
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param preferred_layout_for_x: (out) preferred layout for the input tensor for each component
    /// \param preferred_layout_for_y: (out) preferred layout for the output tensor for each component

    template <std::size_t Nd, std::size_t Ni, typename T>
    void bsr_get_preferred_layout(BSR_handle *bsrh, int ncomponents, const Context *ctx,
                                  CoorOrder co, MatrixLayout *preferred_layout_for_x,
                                  MatrixLayout *preferred_layout_for_y) {

        detail::SelfComm comm = detail::get_comm();

        detail::BSRComponents<Nd, Ni, T> *bsr =
            detail::get_bsr_components_from_handle<Nd, Ni, T>(bsrh, ctx, ncomponents, comm, co);

        for (unsigned int i = 0; i < bsr->c.first.size(); ++i) {
            const unsigned int componentId = bsr->c.first[i].v.componentId;
            preferred_layout_for_x[componentId] = bsr->c.first[i].preferredLayout;
            preferred_layout_for_y[componentId] =
                bsr->c.first[i].allowLayout == detail::ColumnMajorForY
                    ? ColumnMajor
                    : bsr->c.first[i].preferredLayout;
        }
        for (unsigned int i = 0; i < bsr->c.second.size(); ++i) {
            const unsigned int componentId = bsr->c.second[i].v.componentId;
            preferred_layout_for_x[componentId] = bsr->c.second[i].preferredLayout;
            preferred_layout_for_y[componentId] =
                bsr->c.second[i].allowLayout == detail::ColumnMajorForY
                    ? ColumnMajor
                    : bsr->c.second[i].preferredLayout;
        }
    }
}
#endif // __SUPERBBLAS_BSR__
