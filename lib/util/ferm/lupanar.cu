// lupanar.cu -- block ILU(0) preconditioner on the GPU, see lupanar.h
//
// The GPU part (GpuTri, pack_faces, tri_kernel, dmma884, cmma, tri_kernel_tc) is the LUPANAR benchmark's
// (lupanar_cuda_1.cpp)
// unchanged, except that GpuTri lost the fields only the host used. The host part is new.
#include "lupanar.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/complex.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace cg = cooperative_groups;

using scalar  = std::complex<double>;
using dscalar = thrust::complex<double>;          // layout-compatible with std::complex<double>
#define MPI_SCALAR MPI_CXX_DOUBLE_COMPLEX

// ============================ GPU part ============================
namespace lupanar_gpu {

constexpr int TC_MP      = 32;    // tensor-core kernel: column-panel width (multiple of 8)
constexpr int TC_THREADS = 128;   // tensor-core kernel: threads per block (4 warps)

// Everything the kernel needs. All kernel-visible pointers are device pointers.
struct GpuTri {
  int nloc, k, m, bs, nlev, face_total;
  const int* ptr;  const int* col;                 // BSR structure (off-diagonal blocks)
  const dscalar* val;  const dscalar* dinv;        // off-diagonal blocks, inverse diagonal blocks
  int *level_ptr, *level_rows;                     // rows sorted by level
  int *face_idx;                                   // concatenated send faces (one per downstream rank)
  dscalar* sbuf;                                   // pack buffer, face_total*bs
};

// Pack outgoing faces into one contiguous buffer (already ordered by destination).
// Grid-stride loop, so it can be called by every thread of a cooperative kernel.
__device__ __forceinline__ void pack_faces(const GpuTri& T, const dscalar* ext) {
  const int bs = T.bs;
  const size_t total = (size_t)T.face_total * bs;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total;
       e += (size_t)gridDim.x * blockDim.x) {
    const size_t row = e / bs, w = e - row * bs;
    T.sbuf[e] = ext[(size_t)T.face_idx[row] * bs + w];
  }
}

// ---------- SIMT kernel: all levels (gemm_sub + gemm per row), then face packing ----------
__global__ void tri_kernel(GpuTri T, const dscalar* __restrict__ rhs, dscalar* ext) {
  extern __shared__ unsigned char smem_raw[];
  dscalar* s = reinterpret_cast<dscalar*>(smem_raw);       // k x m accumulator tile
  cg::grid_group grid = cg::this_grid();
  const int k = T.k, m = T.m, bs = T.bs, kk = k * k;

  for (int lev = 0; lev < T.nlev; ++lev) {
    const int beg = T.level_ptr[lev], end = T.level_ptr[lev + 1];
    for (int i = beg + blockIdx.x; i < end; i += gridDim.x) {
      const int r = T.level_rows[i];
      const int p0 = T.ptr[r], p1 = T.ptr[r + 1];

      // s = rhs_r - sum_p B_p * X_p          (the "gemm_sub" part)
      for (int idx = threadIdx.x; idx < bs; idx += blockDim.x) {
        const int a = idx / m, j = idx - a * m;
        dscalar acc = rhs[(size_t)r * bs + idx];
        for (int p = p0; p < p1; ++p) {
          const dscalar* B = T.val + (size_t)p * kk + (size_t)a * k;
          const dscalar* X = ext + (size_t)T.col[p] * bs + j;
          for (int b = 0; b < k; ++b) acc -= B[b] * X[(size_t)b * m];
        }
        s[idx] = acc;
      }
      __syncthreads();

      // ext_r = Dinv_r * s                   (the "gemm" part)
      for (int idx = threadIdx.x; idx < bs; idx += blockDim.x) {
        const int a = idx / m, j = idx - a * m;
        const dscalar* Dv = T.dinv + (size_t)r * kk + (size_t)a * k;
        dscalar acc(0.0, 0.0);
        for (int b = 0; b < k; ++b) acc += Dv[b] * s[b * m + j];
        ext[(size_t)r * bs + idx] = acc;
      }
      __syncthreads();                                      // s is reused by the next row
    }
    grid.sync();       // all rows of this level are visible before the next level reads them
  }

  pack_faces(T, ext);
}

// ---------- Tensor-core kernel (FP64 DMMA, m8n8k4) ----------
// D = A*B + C with A 8x4 (row), B 4x8 (col), C/D 8x8, all double. Per-lane fragments
// (g = lane/4, t = lane%4):  A: a = A[g][t]   B: b = B[t][g]   C/D: c0 = C[g][2t], c1 = C[g][2t+1].
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void dmma884(double& d0, double& d1, double a, double b,
                                        double c0, double c1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
               : "=d"(d0), "=d"(d1) : "d"(a), "d"(b), "d"(c0), "d"(c1));
}
// Complex accumulate (cr,ci) += (ar+i*ai)*(br+i*bi) using 4 real DMMAs:
//   cr += ar*br + nai*bi   (nai = -ai) ;   ci += ar*bi + ai*br
// To compute (cr,ci) -= A*B, the caller simply passes the negated A (-ar, -ai).
__device__ __forceinline__ void cmma(double* cr, double* ci, double ar, double ai, double nai,
                                     double br, double bi) {
  dmma884(cr[0], cr[1], ar,  br, cr[0], cr[1]);
  dmma884(cr[0], cr[1], nai, bi, cr[0], cr[1]);
  dmma884(ci[0], ci[1], ar,  bi, ci[0], ci[1]);
  dmma884(ci[0], ci[1], ai,  br, ci[0], ci[1]);
}
#endif

// Work split: one thread block per row of the current level. The k x m tile of the row is
// processed in column panels of width TC_MP. Per panel, the warps share the (k/8)*(TC_MP/8)
// output tiles: phase 1 computes S = rhs_r - sum_p B_p X_p into shared memory, phase 2 computes
// ext_r = Dinv_r * S. Operands are read directly from the interleaved complex arrays with
// zero-padding for k, m that are not multiples of 4/8, so any k and m work.
__global__ void tri_kernel_tc(GpuTri T, const dscalar* rhs_c, dscalar* ext_c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ unsigned char smem_raw[];
  double* S = reinterpret_cast<double*>(smem_raw);         // k x TC_MP complex, interleaved
  cg::grid_group grid = cg::this_grid();

  const int k = T.k, m = T.m, kk = k * k;
  const size_t bs = (size_t)T.bs;
  const double* rhs  = reinterpret_cast<const double*>(rhs_c);
  double*       ext  = reinterpret_cast<double*>(ext_c);
  const double* val  = reinterpret_cast<const double*>(T.val);
  const double* dinv = reinterpret_cast<const double*>(T.dinv);

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rtiles = (k + 7) / 8;

  for (int lev = 0; lev < T.nlev; ++lev) {
    const int beg = T.level_ptr[lev], end = T.level_ptr[lev + 1];
    for (int i = beg + blockIdx.x; i < end; i += gridDim.x) {
      const int r = T.level_rows[i];
      const int p0 = T.ptr[r], p1 = T.ptr[r + 1];

      for (int j0 = 0; j0 < m; j0 += TC_MP) {
        const int mw = min(TC_MP, m - j0);                  // panel width
        const int ntiles_n = (mw + 7) / 8;
        const int ntiles = rtiles * ntiles_n;

        // ---- phase 1: S = rhs_r - sum_p B_p * X_p ----
        for (int tile = warp; tile < ntiles; tile += nwarps) {
          const int rt = tile / ntiles_n, nt = tile - rt * ntiles_n;
          const int row  = rt * 8 + g;                      // A row / C row
          const int ncol = nt * 8 + g;                      // B column (panel-local)
          const int ccol = nt * 8 + 2 * t;                  // first C column (panel-local)
          double cr[2] = {0.0, 0.0}, ci[2] = {0.0, 0.0};
          if (row < k) {
            for (int u = 0; u < 2; ++u)
              if (ccol + u < mw) {
                const size_t idx = ((size_t)r * bs + (size_t)row * m + j0 + ccol + u) * 2;
                cr[u] = rhs[idx]; ci[u] = rhs[idx + 1];
              }
          }
          for (int p = p0; p < p1; ++p) {
            const double* B = val + (size_t)p * kk * 2;
            const double* X = ext + (size_t)T.col[p] * bs * 2;
            for (int kb = 0; kb < k; kb += 4) {
              const int ka = kb + t;
              double ar = 0, ai = 0, br = 0, bi = 0;
              if (row < k && ka < k) { ar = -B[((size_t)row * k + ka) * 2];      // negated: we subtract
                                       ai = -B[((size_t)row * k + ka) * 2 + 1]; }
              if (ka < k && ncol < mw) {
                const size_t idx = ((size_t)ka * m + j0 + ncol) * 2;
                br = X[idx]; bi = X[idx + 1];
              }
              cmma(cr, ci, ar, ai, -ai, br, bi);
            }
          }
          if (row < k)
            for (int u = 0; u < 2; ++u)
              if (ccol + u < mw) {
                S[((size_t)row * TC_MP + ccol + u) * 2]     = cr[u];
                S[((size_t)row * TC_MP + ccol + u) * 2 + 1] = ci[u];
              }
        }
        __syncthreads();                                    // S complete for this panel

        // ---- phase 2: ext_r = Dinv_r * S ----
        const double* Dv = dinv + (size_t)r * kk * 2;
        for (int tile = warp; tile < ntiles; tile += nwarps) {
          const int rt = tile / ntiles_n, nt = tile - rt * ntiles_n;
          const int row  = rt * 8 + g;
          const int ncol = nt * 8 + g;
          const int ccol = nt * 8 + 2 * t;
          double cr[2] = {0.0, 0.0}, ci[2] = {0.0, 0.0};
          for (int kb = 0; kb < k; kb += 4) {
            const int ka = kb + t;
            double ar = 0, ai = 0, br = 0, bi = 0;
            if (row < k && ka < k) { ar = Dv[((size_t)row * k + ka) * 2];
                                     ai = Dv[((size_t)row * k + ka) * 2 + 1]; }
            if (ka < k && ncol < mw) {
              br = S[((size_t)ka * TC_MP + ncol) * 2];
              bi = S[((size_t)ka * TC_MP + ncol) * 2 + 1];
            }
            cmma(cr, ci, ar, ai, -ai, br, bi);
          }
          if (row < k)
            for (int u = 0; u < 2; ++u)
              if (ccol + u < mw) {
                const size_t idx = ((size_t)r * bs + (size_t)row * m + j0 + ccol + u) * 2;
                ext[idx] = cr[u]; ext[idx + 1] = ci[u];
              }
        }
        __syncthreads();                                    // S is reused by the next panel / row
      }
    }
    grid.sync();                                            // level done, visible to the next one
  }
  pack_faces(T, ext_c);
#else
  __trap();                                                 // built for an architecture without FP64 DMMA
#endif
}

} // namespace lupanar_gpu

// ============================ host part ============================
extern "C" {
  void zgemm_(const char* ta, const char* tb, const int* m, const int* n, const int* k,
	      const scalar* alpha, const scalar* a, const int* lda, const scalar* b, const int* ldb,
	      const scalar* beta, scalar* c, const int* ldc);
  void zgetrf_(const int* m, const int* n, scalar* a, const int* lda, int* ipiv, int* info);
  void zgetri_(const int* n, scalar* a, const int* lda, const int* ipiv, scalar* work,
	       const int* lwork, int* info);
}

namespace lupanar {
  using namespace lupanar_gpu;

  namespace {
    constexpr int TAG_FACTOR = 50, TAG_LOWER = 100, TAG_UPPER = 200;
    constexpr int MAX_WORK = 4; // workspaces kept (one per number of right-hand sides)

    [[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("lupanar: " + msg); }

    void cu(cudaError_t e, const char* what) {
      if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
    }

    int checked_int(size_t n, const char* what) {
      if (n > (size_t)2147483647) fail(std::string(what) + " does not fit in an int");
      return (int)n;
    }

    template <class T> T* dev_alloc(size_t n) {
      void* p = nullptr;
      cu(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
      return static_cast<T*>(p);
    }

    template <class T, class S> T* dev_upload(const std::vector<S>& v) {
      static_assert(sizeof(T) == sizeof(S), "same layout expected");
      T* p = dev_alloc<T>(v.size());
      if (!v.empty())
	cu(cudaMemcpy(p, v.data(), v.size() * sizeof(S), cudaMemcpyHostToDevice), "cudaMemcpy");
      return p;
    }

    template <class T> T* pin_alloc(size_t n) {
      void* p = nullptr;
      cu(cudaHostAlloc(&p, std::max<size_t>(n, 1) * sizeof(T), cudaHostAllocDefault), "cudaHostAlloc");
      return static_cast<T*>(p);
    }

    /// Rows exchanged with one rank: n rows from `off` on, counted in the ghost rows (receive) or
    /// in face_idx (send)
    struct Seg {
      int rank, off, n;
    };

    /// One triangular solve, ext_r = dinv_r (rhs_r - sum_p val_p ext_col(p)). ext has the local rows
    /// followed by the ghost rows, the latter grouped by the rank they come from.
    struct Tri {
      int next = 0; ///< local rows + ghost rows
      int nlev = 0, width = 0, face_total = 0;
      std::vector<Seg> up, down; ///< ranks to wait for, ranks to send to
      int *ptr = nullptr, *col = nullptr, *level_ptr = nullptr, *level_rows = nullptr,
	  *face_idx = nullptr;	 ///< device
      dscalar *val = nullptr, *dinv = nullptr; ///< device
    };

    /// Buffers and launch configuration for one number of right-hand sides
    struct Work {
      int m = 0;
      long stamp = 0;
      bool use_tc = false;
      int block = 0, grid_l = 0, grid_u = 0;
      size_t smem = 0;
      dscalar *x = nullptr, *w = nullptr, *z = nullptr, *sbuf_l = nullptr, *sbuf_u = nullptr; ///< device
      scalar *h_x = nullptr, *h_y = nullptr, *h_sbuf = nullptr, *h_rbuf = nullptr; ///< pinned host
    };

    /// Block pattern of the whole matrix, the same on every rank. Sites are renumbered 0..n-1 in
    /// ascending gid, which is the elimination order.
    struct Pattern {
      int n = 0;
      std::vector<long long> gid; ///< ascending
      std::vector<int> owner;
      std::vector<size_t> ptr;
      std::vector<int> col; ///< ascending within each row

      int find(long long g) const {
	auto it = std::lower_bound(gid.begin(), gid.end(), g);
	return it != gid.end() && *it == g ? (int)(it - gid.begin()) : -1;
      }
      bool has(int i, int j) const {
	return std::binary_search(col.begin() + ptr[i], col.begin() + ptr[i + 1], j);
      }
    };

    /// Collect the pattern of every rank. Local rows: ptr/colg, columns ascending and unique.
    /// Every rank sees the same data, so every rank throws on the same errors.
    Pattern gather_pattern(MPI_Comm comm, const std::vector<long long>& row_gid,
			   const std::vector<int>& ptr, const std::vector<long long>& colg) {
      int np = 1;
      MPI_Comm_size(comm, &np);
      std::vector<long long> mine;
      mine.reserve(2 * row_gid.size() + colg.size());
      for (size_t r = 0; r < row_gid.size(); ++r) {
	mine.push_back(row_gid[r]);
	mine.push_back(ptr[r + 1] - ptr[r]);
	mine.insert(mine.end(), colg.begin() + ptr[r], colg.begin() + ptr[r + 1]);
      }
      int cnt = checked_int(mine.size(), "the pattern of a rank");
      std::vector<int> cnts(np), displs(np);
      MPI_Allgather(&cnt, 1, MPI_INT, cnts.data(), 1, MPI_INT, comm);
      size_t total = 0;
      for (int q = 0; q < np; ++q) {
	displs[q] = checked_int(total, "the pattern of the matrix");
	total += cnts[q];
      }
      checked_int(total, "the pattern of the matrix");
      std::vector<long long> all(total);
      MPI_Allgatherv(mine.data(), cnt, MPI_LONG_LONG, all.data(), cnts.data(), displs.data(),
		     MPI_LONG_LONG, comm);

      struct Row {
	long long gid;
	int owner;
	size_t off;
	int ncol;
      };
      std::vector<Row> rows;
      for (int q = 0; q < np; ++q)
	for (size_t o = displs[q], e = o + cnts[q]; o < e; o += 2 + all[o + 1])
	  rows.push_back(Row{all[o], q, o + 2, (int)all[o + 1]});
      std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
	return a.gid < b.gid || (a.gid == b.gid && a.owner < b.owner);
      });
      for (size_t i = 1; i < rows.size(); ++i)
	if (rows[i].gid == rows[i - 1].gid)
	  fail("site " + std::to_string(rows[i].gid) + " is a row of rank " +
	       std::to_string(rows[i - 1].owner) + " and of rank " + std::to_string(rows[i].owner));

      Pattern P;
      P.n = checked_int(rows.size(), "the number of rows");
      P.gid.resize(P.n);
      P.owner.resize(P.n);
      P.ptr.assign(P.n + 1, 0);
      for (int i = 0; i < P.n; ++i) {
	P.gid[i] = rows[i].gid;
	P.owner[i] = rows[i].owner;
	P.ptr[i + 1] = P.ptr[i] + rows[i].ncol;
      }
      P.col.resize(P.ptr[P.n]);
      for (int i = 0; i < P.n; ++i) {
	bool diag = false;
	for (int c = 0; c < rows[i].ncol; ++c) {
	  const int j = P.find(all[rows[i].off + c]);
	  if (j < 0)
	    fail("column " + std::to_string(all[rows[i].off + c]) + " of row " +
		 std::to_string(P.gid[i]) + " is not a row of any rank");
	  P.col[P.ptr[i] + c] = j;
	  diag = diag || j == i;
	}
	if (!diag) fail("row " + std::to_string(P.gid[i]) + " has no diagonal block");
      }
      return P;
    }

    /// ILU(0) changes block (i, j) when a_il, a_lj and a_ij are all there for some l < i, j. Only
    /// i == j is implemented: throw if there is such an l for i != j.
    void check_no_triangle(const Pattern& P) {
      std::vector<size_t> tptr(P.n + 1, 0);
      for (int c : P.col) ++tptr[c + 1];
      for (int i = 0; i < P.n; ++i) tptr[i + 1] += tptr[i];
      std::vector<int> trow(P.col.size());
      {
	std::vector<size_t> pos(tptr.begin(), tptr.end() - 1);
	for (int i = 0; i < P.n; ++i)
	  for (size_t p = P.ptr[i]; p < P.ptr[i + 1]; ++p) trow[pos[P.col[p]]++] = i;
      }
      for (int l = 0; l < P.n; ++l)
	for (size_t q = tptr[l]; q < tptr[l + 1]; ++q) {
	  const int i = trow[q];
	  if (i <= l) continue;
	  for (size_t p = P.ptr[l]; p < P.ptr[l + 1]; ++p) {
	    const int j = P.col[p];
	    if (j <= l || j == i || !P.has(i, j)) continue;
	    fail("sites " + std::to_string(P.gid[l]) + ", " + std::to_string(P.gid[i]) + " and " +
		 std::to_string(P.gid[j]) +
		 " are all coupled to each other; ILU(0) is only implemented for patterns without "
		 "such triangles");
	  }
	}
    }

    /// Who sends what to whom in one triangular solve
    struct Plan {
      std::vector<int> ghost; ///< sites received, grouped by owner, ascending within each
      std::vector<Seg> up;
      std::vector<int> face; ///< sites sent, grouped by destination, ascending within each
      std::vector<Seg> down;
      int stages = 1; ///< longest chain of ranks waiting for each other
    };

    /// lower: row i needs its columns c < i; else its columns c > i. Sender and receiver derive
    /// the same lists, in the same order, from the global pattern.
    Plan make_plan(const Pattern& P, int me, int np, bool lower) {
      typedef std::pair<int, int> Pair;
      std::vector<Pair> g, f, e; // (owner, site), (destination, site), (from rank, to rank)
      for (int i = 0; i < P.n; ++i) {
	const int oi = P.owner[i];
	for (size_t p = P.ptr[i]; p < P.ptr[i + 1]; ++p) {
	  const int c = P.col[p], oc = P.owner[c];
	  if (oc == oi || (c < i) != lower) continue;
	  if (e.empty() || e.back() != Pair(oc, oi)) e.push_back(Pair(oc, oi));
	  if (oi == me) g.push_back(Pair(oc, c));
	  if (oc == me) f.push_back(Pair(oi, c));
	}
      }
      auto uniq = [](std::vector<Pair>& v) {
	std::sort(v.begin(), v.end());
	v.erase(std::unique(v.begin(), v.end()), v.end());
      };
      uniq(g);
      uniq(f);
      uniq(e);

      Plan pl;
      auto segs = [](const std::vector<Pair>& v, std::vector<int>& sites, std::vector<Seg>& s) {
	for (size_t i = 0; i < v.size(); ++i) {
	  if (s.empty() || s.back().rank != v[i].first) s.push_back(Seg{v[i].first, (int)i, 0});
	  ++s.back().n;
	  sites.push_back(v[i].second);
	}
      };
      segs(g, pl.ghost, pl.up);
      segs(f, pl.face, pl.down);

      // Longest path of the rank graph; a cycle would be a deadlock
      std::vector<int> head(np + 1, 0), indeg(np, 0), depth(np, 1), queue;
      for (const Pair& x : e) {
	++head[x.first + 1];
	++indeg[x.second];
      }
      for (int q = 0; q < np; ++q) head[q + 1] += head[q];
      for (int q = 0; q < np; ++q)
	if (indeg[q] == 0) queue.push_back(q);
      for (size_t h = 0; h < queue.size(); ++h) {
	const int a = queue[h];
	for (int j = head[a]; j < head[a + 1]; ++j) {
	  const int b = e[j].second;
	  depth[b] = std::max(depth[b], depth[a] + 1);
	  if (--indeg[b] == 0) queue.push_back(b);
	}
      }
      if ((int)queue.size() != np)
	fail("the ranks wait for each other in a cycle: with the sites in ascending order, the "
	     "rows of two ranks must not depend on each other in both directions");
      pl.stages = *std::max_element(depth.begin(), depth.end());
      return pl;
    }

    /// Levels of the local rows: a row is one level after the last local row it needs. `order`
    /// lists the rows so that every row comes after the ones it needs. col >= nloc is a ghost.
    void make_levels(int nloc, const std::vector<int>& order, const std::vector<int>& ptr,
		     const std::vector<int>& col, std::vector<int>& level_ptr,
		     std::vector<int>& level_rows, int& width) {
      std::vector<int> lev(nloc, 0);
      int nlev = 0;
      for (int r : order) {
	int l = 0;
	for (int p = ptr[r]; p < ptr[r + 1]; ++p)
	  if (col[p] < nloc) l = std::max(l, lev[col[p]] + 1);
	lev[r] = l;
	nlev = std::max(nlev, l + 1);
      }
      level_ptr.assign(nlev + 1, 0);
      for (int r = 0; r < nloc; ++r) ++level_ptr[lev[r] + 1];
      for (int l = 0; l < nlev; ++l) level_ptr[l + 1] += level_ptr[l];
      level_rows.resize(nloc);
      std::vector<int> pos(level_ptr.begin(), level_ptr.end() - 1);
      for (int r : order) level_rows[pos[lev[r]]++] = r;
      width = 0;
      for (int l = 0; l < nlev; ++l) width = std::max(width, level_ptr[l + 1] - level_ptr[l]);
    }

    void free_tri(Tri& T) {
      cudaFree(T.ptr);
      cudaFree(T.col);
      cudaFree(T.level_ptr);
      cudaFree(T.level_rows);
      cudaFree(T.face_idx);
      cudaFree(T.val);
      cudaFree(T.dinv);
      T = Tri();
    }

    void free_work(Work& w) {
      cudaFree(w.x);
      cudaFree(w.w);
      cudaFree(w.z);
      cudaFree(w.sbuf_l);
      cudaFree(w.sbuf_u);
      if (w.h_x) cudaFreeHost(w.h_x);
      if (w.h_y) cudaFreeHost(w.h_y);
      if (w.h_sbuf) cudaFreeHost(w.h_sbuf);
      if (w.h_rbuf) cudaFreeHost(w.h_rbuf);
      w = Work();
    }
  }

  struct Solver {
    MPI_Comm comm = MPI_COMM_NULL;
    int me = 0, np = 1;
    int device = -1;
    cudaStream_t stream = nullptr;
    Options ops;
    Info info;
    int k = 0, nloc = 0;
    Tri L, U;
    int cc_major = 0, nsm = 0;
    size_t smem_max = 0; ///< largest shared memory a thread block can ask for
    std::vector<Work> work;
    long clock = 0;
    std::vector<MPI_Request> req;
  };

  Solver* create(const Matrix& A, const Options& ops) {
    const int k = A.k, nloc = (int)A.row_gid.size();
    const size_t kk = (size_t)k * k;
    if (k <= 0) fail("the block size has to be positive");
    if (A.ptr.size() != A.row_gid.size() + 1 || A.ptr[0] != 0 ||
	A.col_gid.size() != (size_t)A.ptr[nloc] || A.val.size() != A.col_gid.size() * kk)
      fail("the sizes of the arrays of the matrix do not match");
    if (ops.kernel < 0 || ops.kernel > 2 || ops.grid < 0 || ops.grid > 1) fail("invalid options");

    Solver* s = new Solver;
    struct Guard {
      Solver* s;
      ~Guard() {
	if (s) destroy(s);
      }
    } guard{s};
    s->ops = ops;
    s->k = k;
    s->nloc = nloc;
    MPI_Comm_dup(A.comm, &s->comm);
    MPI_Comm_rank(s->comm, &s->me);
    MPI_Comm_size(s->comm, &s->np);
    const int me = s->me, np = s->np;

    // The GPU: whatever is current on this thread
    int coop = 0, smem = 0;
    cu(cudaGetDevice(&s->device), "cudaGetDevice");
    cu(cudaDeviceGetAttribute(&s->nsm, cudaDevAttrMultiProcessorCount, s->device), "cudaDeviceGetAttribute");
    cu(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, s->device), "cudaDeviceGetAttribute");
    cu(cudaDeviceGetAttribute(&s->cc_major, cudaDevAttrComputeCapabilityMajor, s->device), "cudaDeviceGetAttribute");
    cu(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, s->device), "cudaDeviceGetAttribute");
    s->smem_max = smem;
    if (!coop) fail("the GPU cannot launch cooperative kernels");
    cu(cudaStreamCreate(&s->stream), "cudaStreamCreate");

    // Local rows with the columns in ascending order and repeated blocks added up
    std::vector<int> ptr(nloc + 1, 0);
    std::vector<long long> colg;
    std::vector<scalar> val;
    colg.reserve(A.col_gid.size());
    val.reserve(A.val.size());
    {
      std::vector<int> o;
      for (int r = 0; r < nloc; ++r) {
	o.resize(A.ptr[r + 1] - A.ptr[r]);
	for (size_t q = 0; q < o.size(); ++q) o[q] = A.ptr[r] + (int)q;
	std::stable_sort(o.begin(), o.end(),
			 [&](int a, int b) { return A.col_gid[a] < A.col_gid[b]; });
	for (int p : o) {
	  const scalar* v = A.val.data() + (size_t)p * kk;
	  if ((int)colg.size() > ptr[r] && colg.back() == A.col_gid[p]) {
	    scalar* w = val.data() + val.size() - kk;
	    for (size_t e = 0; e < kk; ++e) w[e] += v[e];
	  } else {
	    colg.push_back(A.col_gid[p]);
	    val.insert(val.end(), v, v + kk);
	  }
	}
	ptr[r + 1] = (int)colg.size();
      }
    }

    // The pattern of all ranks, and what it implies: who waits for whom
    const Pattern P = gather_pattern(s->comm, A.row_gid, ptr, colg);
    check_no_triangle(P);
    const Plan pl = make_plan(P, me, np, true), pu = make_plan(P, me, np, false);

    // Local rows and blocks in the numbering of P. Each row: lower blocks, diagonal, upper blocks
    std::vector<int> gi(nloc), ci(colg.size()), diag(nloc, -1), lrow(P.n, -1), lord(nloc);
    for (int r = 0; r < nloc; ++r) {
      gi[r] = P.find(A.row_gid[r]);
      lrow[gi[r]] = r;
      lord[r] = r;
    }
    std::sort(lord.begin(), lord.end(), [&](int a, int b) { return gi[a] < gi[b]; });
    std::vector<int> loff(nloc + 1, 0), uoff(nloc + 1, 0);
    for (int r = 0; r < nloc; ++r) {
      for (int p = ptr[r]; p < ptr[r + 1]; ++p) {
	ci[p] = P.find(colg[p]);
	if (ci[p] == gi[r]) diag[r] = p;
      }
      loff[r + 1] = loff[r] + diag[r] - ptr[r];
      uoff[r + 1] = uoff[r] + ptr[r + 1] - diag[r] - 1;
    }
    // Block of d_c^{-1} a_ci in uval, or -1 if the matrix has no a_ci (c local)
    auto upper_of = [&](int c, int i) {
      const int lc = lrow[c];
      auto b = ci.begin() + diag[lc] + 1, e = ci.begin() + ptr[lc + 1];
      auto it = std::lower_bound(b, e, i);
      return it != e && *it == i ? uoff[lc] + (int)(it - b) : -1;
    };

    // ILU(0), rank after rank in the order of the L solve:
    //   d_i = a_ii - sum_{c < i} a_ic (d_c^{-1} a_ci),    dinv_i = d_i^{-1},    uval_ij = d_i^{-1} a_ij
    const double t_fact = MPI_Wtime();
    std::vector<scalar> dinv((size_t)nloc * kk), uval((size_t)uoff[nloc] * kk);
    {
      // From the ranks upstream: d_c^{-1} a_ci for every a_ic with c < i on another rank, sorted
      // by (rank, c, i); zeros where the matrix has no a_ci
      typedef std::tuple<int, int, int, int> In; // rank, c, i, local block
      std::vector<In> in;
      for (int r = 0; r < nloc; ++r)
	for (int p = ptr[r]; p < diag[r]; ++p)
	  if (P.owner[ci[p]] != me) in.push_back(In(P.owner[ci[p]], ci[p], gi[r], p));
      std::sort(in.begin(), in.end());
      std::vector<int> src(colg.size(), -1);
      for (size_t q = 0; q < in.size(); ++q) src[std::get<3>(in[q])] = (int)q;
      std::vector<scalar> rbuf(in.size() * kk);
      std::vector<MPI_Request> req;
      for (size_t q = 0, e = 0; q < in.size(); q = e) {
	for (e = q; e < in.size() && std::get<0>(in[e]) == std::get<0>(in[q]); ++e);
	req.push_back(MPI_REQUEST_NULL);
	MPI_Irecv(rbuf.data() + q * kk, checked_int((e - q) * kk, "a message of the factorization"),
		  MPI_SCALAR, std::get<0>(in[q]), TAG_FACTOR, s->comm, &req.back());
      }
      MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);

      // Row-major blocks given to column-major BLAS: C = A B is zgemm(B, A)
      const int lwork = 64 * k;
      const scalar one(1.0), zero(0.0), minus(-1.0);
      std::vector<scalar> D(kk), wk(lwork);
      std::vector<int> ipiv(k);
      for (int r : lord) {
	std::copy_n(val.begin() + (size_t)diag[r] * kk, kk, D.begin());
	for (int p = ptr[r]; p < diag[r]; ++p) {
	  const scalar* G = nullptr; // d_c^{-1} a_ci
	  if (P.owner[ci[p]] != me) {
	    G = rbuf.data() + (size_t)src[p] * kk;
	  } else {
	    const int u = upper_of(ci[p], gi[r]);
	    if (u >= 0) G = uval.data() + (size_t)u * kk;
	  }
	  if (G)
	    zgemm_("N", "N", &k, &k, &k, &minus, G, &k, val.data() + (size_t)p * kk, &k, &one,
		   D.data(), &k);
	}
	int info = 0;
	zgetrf_(&k, &k, D.data(), &k, ipiv.data(), &info);
	if (info == 0) zgetri_(&k, D.data(), &k, ipiv.data(), wk.data(), &lwork, &info);
	if (info != 0)
	  fail("the diagonal block of site " + std::to_string(A.row_gid[r]) +
	       " is singular after the elimination");
	std::copy_n(D.begin(), kk, dinv.begin() + (size_t)r * kk);
	for (int p = diag[r] + 1; p < ptr[r + 1]; ++p)
	  zgemm_("N", "N", &k, &k, &k, &one, val.data() + (size_t)p * kk, &k, D.data(), &k, &zero,
		 uval.data() + (size_t)(uoff[r] + p - diag[r] - 1) * kk, &k);
      }

      // To the ranks downstream, the same lists seen from the other side
      typedef std::tuple<int, int, int> Out; // rank, c, i
      std::vector<Out> out;
      for (int i = 0; i < P.n; ++i) {
	if (P.owner[i] == me) continue;
	for (size_t p = P.ptr[i]; p < P.ptr[i + 1] && P.col[p] < i; ++p)
	  if (P.owner[P.col[p]] == me) out.push_back(Out(P.owner[i], P.col[p], i));
      }
      std::sort(out.begin(), out.end());
      std::vector<scalar> sbuf(out.size() * kk);
      for (size_t q = 0; q < out.size(); ++q) {
	const int u = upper_of(std::get<1>(out[q]), std::get<2>(out[q]));
	if (u >= 0) std::copy_n(uval.begin() + (size_t)u * kk, kk, sbuf.begin() + q * kk);
      }
      req.clear();
      for (size_t q = 0, e = 0; q < out.size(); q = e) {
	for (e = q; e < out.size() && std::get<0>(out[e]) == std::get<0>(out[q]); ++e);
	req.push_back(MPI_REQUEST_NULL);
	MPI_Isend(sbuf.data() + q * kk, checked_int((e - q) * kk, "a message of the factorization"),
		  MPI_SCALAR, std::get<0>(out[q]), TAG_FACTOR, s->comm, &req.back());
      }
      MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
    }
    s->info.t_factor = MPI_Wtime() - t_fact;

    // The two triangular solves on the GPU:
    //   L: w_i = d_i^{-1} (x_i - sum_{c < i} a_ic w_c)         val = a_ic,           dinv = d_i^{-1}
    //   U: z_i = w_i - sum_{j > i} (d_i^{-1} a_ij) z_j         val = d_i^{-1} a_ij,  dinv = 1
    auto build = [&](bool lower, const Plan& plan, Tri& T) {
      std::vector<int> ext(P.n, -1); // where the kernel finds each site
      for (int r = 0; r < nloc; ++r) ext[gi[r]] = r;
      for (size_t g = 0; g < plan.ghost.size(); ++g) ext[plan.ghost[g]] = nloc + (int)g;
      std::vector<int> tptr(nloc + 1, 0), tcol, face(plan.face.size()), level_ptr, level_rows;
      for (int r = 0; r < nloc; ++r) {
	for (int p = lower ? ptr[r] : diag[r] + 1, e = lower ? diag[r] : ptr[r + 1]; p < e; ++p)
	  tcol.push_back(ext[ci[p]]);
	tptr[r + 1] = (int)tcol.size();
      }
      for (size_t f = 0; f < face.size(); ++f) face[f] = lrow[plan.face[f]];
      make_levels(nloc, lower ? lord : std::vector<int>(lord.rbegin(), lord.rend()), tptr, tcol,
		  level_ptr, level_rows, T.width);
      T.next = nloc + (int)plan.ghost.size();
      T.nlev = (int)level_ptr.size() - 1;
      T.face_total = (int)face.size();
      T.up = plan.up;
      T.down = plan.down;
      T.ptr = dev_upload<int>(tptr);
      T.col = dev_upload<int>(tcol);
      T.level_ptr = dev_upload<int>(level_ptr);
      T.level_rows = dev_upload<int>(level_rows);
      T.face_idx = dev_upload<int>(face);
    };
    build(true, pl, s->L);
    build(false, pu, s->U);
    s->U.val = dev_upload<dscalar>(uval);
    std::vector<scalar>().swap(uval);
    s->L.dinv = dev_upload<dscalar>(dinv);
    std::fill(dinv.begin(), dinv.end(), scalar(0.0));
    for (int r = 0; r < nloc; ++r)
      for (int a = 0; a < k; ++a) dinv[(size_t)r * kk + (size_t)a * k + a] = 1.0;
    s->U.dinv = dev_upload<dscalar>(dinv);
    std::vector<scalar>().swap(dinv);
    {
      std::vector<scalar> lval((size_t)loff[nloc] * kk);
      for (int r = 0; r < nloc; ++r)
	std::copy_n(val.begin() + (size_t)ptr[r] * kk, (size_t)(diag[r] - ptr[r]) * kk,
		    lval.begin() + (size_t)loff[r] * kk);
      s->L.val = dev_upload<dscalar>(lval);
    }

    Info& I = s->info;
    I.k = k;
    I.nloc = nloc;
    I.nglobal = P.n;
    I.ranks = np;
    I.stages = std::max(pl.stages, pu.stages);
    I.levels_l = s->L.nlev;
    I.levels_u = s->U.nlev;
    I.width_l = s->L.width;
    I.width_u = s->U.width;
    I.ghosts_l = s->L.next - nloc;
    I.ghosts_u = s->U.next - nloc;
    I.sends_l = s->L.face_total;
    I.sends_u = s->U.face_total;
    guard.s = nullptr;
    return s;
  }

  namespace {
    /// Buffers and launch configuration for m right-hand sides, made on first use
    Work& get_work(Solver& s, int m) {
      if (m <= 0) fail("the number of right-hand sides has to be positive");
      for (Work& w : s.work)
	if (w.m == m) {
	  w.stamp = ++s.clock;
	  return w;
	}
      if ((int)s.work.size() >= MAX_WORK) {
	auto old = std::min_element(s.work.begin(), s.work.end(),
				    [](const Work& a, const Work& b) { return a.stamp < b.stamp; });
	free_work(*old);
	s.work.erase(old);
      }
      Work w;
      const int k = s.k;
      const size_t bs = (size_t)k * m;
      checked_int(bs, "k * m");

      // Which kernel and how to launch it. The benchmark takes the tensor cores when k > 16 on sm_80+;
      // with a single right-hand side seven of their eight columns are padding and SIMT is 4-8%
      // faster (A100, k = 48 and 64), with 8 it is 2.5-3.5 times slower.
      const bool tc_ok = s.cc_major >= 8;
      const size_t smem_simt = bs * sizeof(dscalar), smem_tc = (size_t)k * TC_MP * sizeof(dscalar);
      int kernel = s.ops.kernel;
      if (kernel == 0) kernel = k > 16 && m > 1 && tc_ok ? 2 : 1;
      if (kernel == 1 && smem_simt > s.smem_max && tc_ok) kernel = 2;
      if (kernel == 2 && !tc_ok) fail("the tensor-core kernel needs an sm_80+ GPU (FP64 DMMA)");
      w.use_tc = kernel == 2;
      w.block = w.use_tc ? TC_THREADS : 128;
      w.smem = w.use_tc ? smem_tc : smem_simt;
      if (w.smem > s.smem_max) fail("k * m is too large for the shared memory of a thread block");
      // Several solvers share the kernels: never lower the limit another one asked for
      static size_t smem_asked[2] = {0, 0};
      size_t& asked = smem_asked[w.use_tc ? 1 : 0];
      asked = std::max(asked, w.smem);
      int bpm = 0;
      if (w.use_tc) {
	cu(cudaFuncSetAttribute(tri_kernel_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)asked),
	   "cudaFuncSetAttribute");
	cu(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, tri_kernel_tc, w.block, w.smem),
	   "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
      } else {
	cu(cudaFuncSetAttribute(tri_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)asked),
	   "cudaFuncSetAttribute");
	cu(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, tri_kernel, w.block, w.smem),
	   "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
      }
      if (bpm < 1) fail("the kernel does not fit on the GPU");
      // A level uses one thread block per row and every block takes part in every grid.sync()
      const int grid_max = s.nsm * bpm;
      auto grid = [&](const Tri& T) {
	if (s.ops.grid == 1) return grid_max;
	const size_t need =
	  std::max<size_t>(T.width, ((size_t)T.face_total * bs + w.block - 1) / w.block);
	return (int)std::min<size_t>(grid_max, std::max<size_t>(need, 1));
      };
      w.grid_l = grid(s.L);
      w.grid_u = grid(s.U);

      s.work.push_back(w);
      Work& v = s.work.back(); // from here on destroy() frees what was allocated
      const size_t n = (size_t)s.nloc * bs;
      v.x = dev_alloc<dscalar>(n);
      v.w = dev_alloc<dscalar>((size_t)s.L.next * bs);
      v.z = dev_alloc<dscalar>((size_t)s.U.next * bs);
      v.sbuf_l = dev_alloc<dscalar>((size_t)s.L.face_total * bs);
      v.sbuf_u = dev_alloc<dscalar>((size_t)s.U.face_total * bs);
      v.h_x = pin_alloc<scalar>(n);
      v.h_y = pin_alloc<scalar>(n);
      v.h_sbuf = pin_alloc<scalar>((size_t)std::max(s.L.face_total, s.U.face_total) * bs);
      v.h_rbuf = pin_alloc<scalar>((size_t)(std::max(s.L.next, s.U.next) - s.nloc) * bs);
      v.m = m;
      v.stamp = ++s.clock;
      return v;
    }

    /// Receive the rows of the ranks upstream, run all levels in one kernel, send the rows the
    /// ranks downstream wait for. Host-staged MPI (pinned buffers).
    void tri_solve(Solver& s, const Tri& T, const Work& w, bool lower, Stats& st) {
      const int m = w.m, tag = lower ? TAG_LOWER : TAG_UPPER;
      const size_t bs = (size_t)s.k * m;
      const dscalar* rhs = lower ? w.x : w.w;
      dscalar* ext = lower ? w.w : w.z;
      dscalar* sbuf = lower ? w.sbuf_l : w.sbuf_u;

      if (!T.up.empty()) {
	const double t = MPI_Wtime();
	s.req.resize(T.up.size());
	for (size_t i = 0; i < T.up.size(); ++i)
	  MPI_Irecv(w.h_rbuf + (size_t)T.up[i].off * bs,
		    checked_int((size_t)T.up[i].n * bs, "a message of the solve"), MPI_SCALAR,
		    T.up[i].rank, tag, s.comm, &s.req[i]);
	MPI_Waitall((int)s.req.size(), s.req.data(), MPI_STATUSES_IGNORE);
	st.mpi_recv += MPI_Wtime() - t;
	cu(cudaMemcpyAsync(ext + (size_t)s.nloc * bs, w.h_rbuf,
			   (size_t)(T.next - s.nloc) * bs * sizeof(dscalar), cudaMemcpyHostToDevice,
			   s.stream),
	   "cudaMemcpyAsync");
      }

      const double t = MPI_Wtime();
      if (s.nloc > 0) {
	GpuTri G;
	G.nloc = s.nloc;
	G.k = s.k;
	G.m = m;
	G.bs = (int)bs;
	G.nlev = T.nlev;
	G.face_total = T.face_total;
	G.ptr = T.ptr;
	G.col = T.col;
	G.val = T.val;
	G.dinv = T.dinv;
	G.level_ptr = T.level_ptr;
	G.level_rows = T.level_rows;
	G.face_idx = T.face_idx;
	G.sbuf = sbuf;
	void* args[] = {(void*)&G, (void*)&rhs, (void*)&ext};
	cu(cudaLaunchCooperativeKernel(w.use_tc ? (void*)tri_kernel_tc : (void*)tri_kernel,
				       dim3(lower ? w.grid_l : w.grid_u), dim3(w.block), args, w.smem,
				       s.stream),
	   "cudaLaunchCooperativeKernel");
	if (T.face_total > 0)
	  cu(cudaMemcpyAsync(w.h_sbuf, sbuf, (size_t)T.face_total * bs * sizeof(dscalar),
			     cudaMemcpyDeviceToHost, s.stream),
	     "cudaMemcpyAsync");
      }
      // h_sbuf has to be complete before it is sent, and h_rbuf is reused by the next solve
      cu(cudaStreamSynchronize(s.stream), "cudaStreamSynchronize");
      st.kernel += MPI_Wtime() - t;

      if (!T.down.empty()) {
	const double t1 = MPI_Wtime();
	s.req.resize(T.down.size());
	for (size_t i = 0; i < T.down.size(); ++i)
	  MPI_Isend(w.h_sbuf + (size_t)T.down[i].off * bs,
		    checked_int((size_t)T.down[i].n * bs, "a message of the solve"), MPI_SCALAR,
		    T.down[i].rank, tag, s.comm, &s.req[i]);
	MPI_Waitall((int)s.req.size(), s.req.data(), MPI_STATUSES_IGNORE);
	st.mpi_send += MPI_Wtime() - t1;
      }
    }
  }

  void solve(Solver* s, int m, const scalar* x, scalar* y, Stats* st) {
    const double t0 = MPI_Wtime();
    int dev = -1;
    cu(cudaGetDevice(&dev), "cudaGetDevice");
    if (dev != s->device)
      fail("solve() called with CUDA device " + std::to_string(dev) +
	   " current, but the solver was created on device " + std::to_string(s->device));
    Work& w = get_work(*s, m);
    const size_t n = (size_t)s->nloc * s->k * m * sizeof(dscalar);
    Stats z;

    const double t1 = MPI_Wtime();
    if (s->nloc > 0) cu(cudaMemcpyAsync(w.x, x, n, cudaMemcpyHostToDevice, s->stream), "cudaMemcpyAsync");
    tri_solve(*s, s->L, w, true, z);
    const double t2 = MPI_Wtime();
    tri_solve(*s, s->U, w, false, z);
    const double t3 = MPI_Wtime();
    if (s->nloc > 0) {
      cu(cudaMemcpyAsync(y, w.z, n, cudaMemcpyDeviceToHost, s->stream), "cudaMemcpyAsync");
      cu(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize");
    }

    s->info.kernel = w.use_tc ? 2 : 1;
    s->info.grid_l = w.grid_l;
    s->info.grid_u = w.grid_u;
    if (st) {
      z.lower = t2 - t1;
      z.upper = t3 - t2;
      z.total = MPI_Wtime() - t0;
      *st = z;
    }
  }

  void buffers(Solver* s, int m, scalar** x, scalar** y) {
    Work& w = get_work(*s, m);
    if (x) *x = w.h_x;
    if (y) *y = w.h_y;
  }

  const Info& info(const Solver* s) { return s->info; }

  void destroy(Solver* s) {
    if (!s) return;
    for (Work& w : s->work) free_work(w);
    free_tri(s->L);
    free_tri(s->U);
    if (s->stream) cudaStreamDestroy(s->stream);
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (s->comm != MPI_COMM_NULL && !finalized) MPI_Comm_free(&s->comm);
    delete s;
  }
}
