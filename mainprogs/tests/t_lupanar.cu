// t_lupanar.cu -- standalone test of util/ferm/lupanar.h on a 4D lattice with nearest-neighbour couplings
//
//   srun -n P ./t_lupanar N0 N1 N2 N3 P0 P1 P2 P3 k m[,m...] [options]
//
//   N0..N3   lattice; site (x,y,z,t) has gid x + N0*(y + N1*(z + N2*t)), so x runs fastest
//   P0..P3   ranks per direction (P0*P1*P2*P3 = P, N_d divisible by P_d); every rank gets a box
//   k        block size
//   m        numbers of right-hand sides, solved one after the other with the same solver
//   --bc pppp      p = periodic, o = open, per direction (default pppp). Periodic with N_d = 2
//                  makes two blocks for the same neighbour (they add up); N_d = 1 has no neighbour
//   --kernel auto|simt|tc     --grid cap|full
//   --iters n      timed solves per m (default 20)
//   --cold-gpu MB  before every timed solve, overwrite MB megabytes of device memory (the solver's matrix
//                  leaves the GPU's caches, as it does inside an application that uses the GPU in between)
//   --cold-cpu MB  the same for the host: MB megabytes of host memory are rewritten before every timed solve
//                  (with either option read total=, not wall=: wall includes the overwriting)
//   --shuffle      rows and blocks of every rank in random order
//   --noref        no serial reference on rank 0 (timing only)
//   --exact        ILU(0) is exact for this lattice (a chain): also require |A y - x| / |x| < 1e-12
//   --expect-throw create() has to throw (e.g. periodic N_d = 3 has triangles)
//
// The blocks are random numbers that depend only on (row, column, direction, a, b), so rank 0 can
// build the whole matrix and compare with a serial ILU(0) written without BLAS.
#include "util/ferm/lupanar.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

typedef std::complex<double> scalar;

// ---------- the matrix ----------
static inline uint64_t mix(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static inline double unif(uint64_t h) { return (double)(mix(h) >> 11) * (2.0 / 9007199254740992.0) - 1.0; }

// Block of row site `row` and column site `col`; dir = 2*d + (forward ? 0 : 1), 8 = diagonal
static void block(long long row, long long col, int dir, int k, scalar* v) {
  const uint64_t h0 = mix(mix(mix((uint64_t)row) ^ (uint64_t)col) ^ (uint64_t)dir);
  const double scale = 0.5 / std::sqrt((double)k);
  for (int a = 0; a < k; ++a)
    for (int b = 0; b < k; ++b) {
      const uint64_t h = mix(h0 ^ (((uint64_t)a << 32) | (uint64_t)b));
      v[a * k + b] = scale * scalar(unif(h), unif(h ^ 0x5851f42d4c957f2dULL));
    }
  if (dir == 8)
    for (int a = 0; a < k; ++a) v[a * k + a] += 4.0;
}

static scalar rhs_entry(long long gid, int a, int j) {
  const uint64_t h = mix(mix((uint64_t)gid ^ 0xabcdef12345ULL) ^ (((uint64_t)a << 32) | (uint64_t)j));
  return scalar(unif(h), unif(h ^ 0x2545f4914f6cdd1dULL));
}

struct Lattice {
  int N[4], P[4];
  bool periodic[4];
  long long sites() const { return (long long)N[0] * N[1] * N[2] * N[3]; }
  long long gid(const int* c) const { return c[0] + (long long)N[0] * (c[1] + (long long)N[1] * (c[2] + (long long)N[2] * c[3])); }
  void coor(long long g, int* c) const {
    for (int d = 0; d < 4; ++d) {
      c[d] = (int)(g % N[d]);
      g /= N[d];
    }
  }
  int owner(const int* c) const {
    int r = 0;
    for (int d = 3; d >= 0; --d) r = r * P[d] + c[d] / (N[d] / P[d]);
    return r;
  }
  // (column site, dir) of every block of row site g, the diagonal first
  std::vector<std::pair<long long, int>> row(long long g) const {
    std::vector<std::pair<long long, int>> v;
    v.push_back(std::make_pair(g, 8));
    int c[4];
    coor(g, c);
    for (int d = 0; d < 4; ++d)
      for (int s = 0; s < 2; ++s) {
	int n[4] = {c[0], c[1], c[2], c[3]};
	n[d] += s == 0 ? 1 : -1;
	if (n[d] < 0 || n[d] >= N[d]) {
	  if (!periodic[d] || N[d] == 1) continue;
	  n[d] = (n[d] + N[d]) % N[d];
	}
	v.push_back(std::make_pair(gid(n), 2 * d + s));
      }
    return v;
  }
};

// ---------- serial reference (rank 0), no BLAS ----------
// Y = B X (acc = 0), Y += B X (acc = 1), Y -= B X (acc = -1); B k x k, X and Y k x m, row-major
static void gemm_ref(int k, int m, const scalar* B, const scalar* X, scalar* Y, int acc) {
  const double* b = reinterpret_cast<const double*>(B);
  const double* x = reinterpret_cast<const double*>(X);
  double* y = reinterpret_cast<double*>(Y);
  const double sg = acc < 0 ? -1.0 : 1.0;
  if (acc == 0) std::fill(y, y + 2 * (size_t)k * m, 0.0);
  for (int a = 0; a < k; ++a)
    for (int c = 0; c < k; ++c) {
      const double br = sg * b[2 * (a * k + c)], bi = sg * b[2 * (a * k + c) + 1];
      const double* xr = x + 2 * (size_t)c * m;
      double* yr = y + 2 * (size_t)a * m;
      for (int j = 0; j < m; ++j) {
	yr[2 * j] += br * xr[2 * j] - bi * xr[2 * j + 1];
	yr[2 * j + 1] += br * xr[2 * j + 1] + bi * xr[2 * j];
      }
    }
}

// Gauss-Jordan with partial pivoting, in place, row-major
static void invert_ref(int k, scalar* A) {
  std::vector<scalar> I((size_t)k * k, 0.0);
  for (int a = 0; a < k; ++a) I[a * k + a] = 1.0;
  for (int c = 0; c < k; ++c) {
    int piv = c;
    for (int r = c + 1; r < k; ++r)
      if (std::abs(A[r * k + c]) > std::abs(A[piv * k + c])) piv = r;
    if (std::abs(A[piv * k + c]) == 0.0) throw std::runtime_error("reference: singular block");
    if (piv != c)
      for (int b = 0; b < k; ++b) {
	std::swap(A[c * k + b], A[piv * k + b]);
	std::swap(I[c * k + b], I[piv * k + b]);
      }
    const scalar inv = 1.0 / A[c * k + c];
    for (int b = 0; b < k; ++b) {
      A[c * k + b] *= inv;
      I[c * k + b] *= inv;
    }
    for (int r = 0; r < k; ++r) {
      if (r == c) continue;
      const scalar f = A[r * k + c];
      if (f == 0.0) continue;
      for (int b = 0; b < k; ++b) {
	A[r * k + b] -= f * A[c * k + b];
	I[r * k + b] -= f * I[c * k + b];
      }
    }
  }
  std::copy(I.begin(), I.end(), A);
}

struct Reference {
  int k = 0;
  long long n = 0;
  std::vector<size_t> ptr;	 // per site
  std::vector<long long> col;	 // ascending within a row, merged
  std::vector<scalar> val, dinv; // blocks, d_i^{-1}

  const scalar* find(long long i, long long j) const {
    auto b = col.begin() + ptr[i], e = col.begin() + ptr[i + 1];
    auto it = std::lower_bound(b, e, j);
    return it != e && *it == j ? val.data() + (size_t)(it - col.begin()) * k * k : nullptr;
  }

  void build(const Lattice& lat, int k_) {
    k = k_;
    n = lat.sites();
    const size_t kk = (size_t)k * k;
    std::vector<scalar> v(kk);
    ptr.assign(n + 1, 0);
    for (long long i = 0; i < n; ++i) {
      auto row = lat.row(i);
      std::stable_sort(row.begin(), row.end(),
		       [](const std::pair<long long, int>& a, const std::pair<long long, int>& b) { return a.first < b.first; });
      for (auto& e : row) {
	block(i, e.first, e.second, k, v.data());
	if (col.size() > ptr[i] && col.back() == e.first) {
	  for (size_t q = 0; q < kk; ++q) val[val.size() - kk + q] += v[q];
	} else {
	  col.push_back(e.first);
	  val.insert(val.end(), v.begin(), v.end());
	}
      }
      ptr[i + 1] = col.size();
    }
    // d_i = a_ii - sum_{c < i} a_ic d_c^{-1} a_ci
    dinv.resize((size_t)n * kk);
    std::vector<scalar> t(kk);
    for (long long i = 0; i < n; ++i) {
      scalar* D = dinv.data() + (size_t)i * kk;
      std::copy_n(find(i, i), kk, D);
      for (size_t p = ptr[i]; p < ptr[i + 1] && col[p] < i; ++p) {
	const scalar* aci = find(col[p], i);
	if (!aci) continue;
	gemm_ref(k, k, dinv.data() + (size_t)col[p] * kk, aci, t.data(), 0);
	gemm_ref(k, k, val.data() + p * kk, t.data(), D, -1);
      }
      invert_ref(k, D);
    }
  }

  // z = (D + U)^{-1} D (D + L)^{-1} x; x, z: n * k * m
  void solve(int m, const std::vector<scalar>& x, std::vector<scalar>& z) const {
    const size_t kk = (size_t)k * k, bs = (size_t)k * m;
    std::vector<scalar> w(x.size()), s(bs);
    z.resize(x.size());
    for (long long i = 0; i < n; ++i) {
      std::copy_n(x.begin() + i * bs, bs, s.begin());
      for (size_t p = ptr[i]; p < ptr[i + 1] && col[p] < i; ++p)
	gemm_ref(k, m, val.data() + p * kk, w.data() + col[p] * bs, s.data(), -1);
      gemm_ref(k, m, dinv.data() + i * kk, s.data(), w.data() + i * bs, 0);
    }
    for (long long i = n - 1; i >= 0; --i) {
      std::fill(s.begin(), s.end(), scalar(0.0));
      for (size_t p = ptr[i]; p < ptr[i + 1]; ++p)
	if (col[p] > i) gemm_ref(k, m, val.data() + p * kk, z.data() + col[p] * bs, s.data(), 1);
      std::copy_n(w.begin() + i * bs, bs, z.begin() + i * bs);
      gemm_ref(k, m, dinv.data() + i * kk, s.data(), z.data() + i * bs, -1);
    }
  }

  // |A y - x| / |x|
  double residual(int m, const std::vector<scalar>& x, const std::vector<scalar>& y) const {
    const size_t kk = (size_t)k * k, bs = (size_t)k * m;
    std::vector<scalar> s(bs);
    double nr = 0, nx = 0;
    for (long long i = 0; i < n; ++i) {
      for (size_t e = 0; e < bs; ++e) s[e] = -x[i * bs + e];
      for (size_t p = ptr[i]; p < ptr[i + 1]; ++p)
	gemm_ref(k, m, val.data() + p * kk, y.data() + col[p] * bs, s.data(), 1);
      for (size_t e = 0; e < bs; ++e) {
	nr += std::norm(s[e]);
	nx += std::norm(x[i * bs + e]);
      }
    }
    return std::sqrt(nr / nx);
  }
};

// ---------- driver ----------
static std::vector<int> int_list(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) {
    v.push_back(std::atoi(p));
    while (*p && *p != ',') ++p;
    if (*p == ',') ++p;
  }
  return v;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, np = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  if (argc < 11) {
    if (rank == 0) std::fprintf(stderr, "usage: %s N0 N1 N2 N3 P0 P1 P2 P3 k m[,m...] [--bc pppp] [--kernel auto|simt|tc] [--grid cap|full] [--iters n] [--cold-gpu MB] [--cold-cpu MB] [--shuffle] [--noref] [--exact] [--expect-throw]\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  Lattice lat;
  for (int d = 0; d < 4; ++d) {
    lat.N[d] = std::atoi(argv[1 + d]);
    lat.P[d] = std::atoi(argv[5 + d]);
    lat.periodic[d] = true;
  }
  const int k = std::atoi(argv[9]);
  const std::vector<int> ms = int_list(argv[10]);
  lupanar::Options ops;
  int iters = 20, cold_gpu = 0, cold_cpu = 0;
  bool shuffle = false, noref = false, exact = false, expect_throw = false;
  std::string bc = "pppp", kernel = "auto", grid = "cap";
  for (int a = 11; a < argc; ++a) {
    const std::string o = argv[a];
    if (o == "--bc" && a + 1 < argc) bc = argv[++a];
    else if (o == "--kernel" && a + 1 < argc) kernel = argv[++a];
    else if (o == "--grid" && a + 1 < argc) grid = argv[++a];
    else if (o == "--iters" && a + 1 < argc) iters = std::atoi(argv[++a]);
    else if (o == "--cold-gpu" && a + 1 < argc) cold_gpu = std::atoi(argv[++a]);
    else if (o == "--cold-cpu" && a + 1 < argc) cold_cpu = std::atoi(argv[++a]);
    else if (o == "--shuffle") shuffle = true;
    else if (o == "--noref") noref = true;
    else if (o == "--exact") exact = true;
    else if (o == "--expect-throw") expect_throw = true;
    else {
      if (rank == 0) std::fprintf(stderr, "unknown option %s\n", o.c_str());
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }
  ops.kernel = kernel == "simt" ? 1 : kernel == "tc" ? 2 : 0;
  ops.grid = grid == "full" ? 1 : 0;
  bool ok = bc.size() == 4 && lat.P[0] * lat.P[1] * lat.P[2] * lat.P[3] == np && k > 0 && !ms.empty();
  for (int d = 0; d < 4 && ok; ++d) {
    lat.periodic[d] = bc[d] == 'p';
    ok = lat.N[d] > 0 && lat.P[d] > 0 && lat.N[d] % lat.P[d] == 0;
  }
  if (!ok) {
    if (rank == 0) std::fprintf(stderr, "invalid arguments (P0*P1*P2*P3 has to be the number of ranks, N_d divisible by P_d)\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  // One GPU per rank of the node, round robin
  {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int lr = 0, ng = 0;
    MPI_Comm_rank(node, &lr);
    MPI_Comm_free(&node);
    if (cudaGetDeviceCount(&ng) != cudaSuccess || ng < 1 || cudaSetDevice(lr % ng) != cudaSuccess) {
      std::fprintf(stderr, "rank %d: no GPU\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  int failed = 0;
  try {
    // The rows of this rank
    const size_t kk = (size_t)k * k;
    const long long n = lat.sites();
    lupanar::Matrix A;
    A.comm = MPI_COMM_WORLD;
    A.k = k;
    for (long long g = 0; g < n; ++g) {
      int c[4];
      lat.coor(g, c);
      if (lat.owner(c) == rank) A.row_gid.push_back(g);
    }
    std::mt19937 rng(12345 + rank);
    if (shuffle) std::shuffle(A.row_gid.begin(), A.row_gid.end(), rng);
    const int nloc = (int)A.row_gid.size();
    A.ptr.assign(1, 0);
    for (int r = 0; r < nloc; ++r) {
      auto row = lat.row(A.row_gid[r]);
      if (shuffle) std::shuffle(row.begin(), row.end(), rng);
      for (auto& e : row) {
	A.col_gid.push_back(e.first);
	A.val.resize(A.val.size() + kk);
	block(A.row_gid[r], e.first, e.second, k, A.val.data() + A.val.size() - kk);
      }
      A.ptr.push_back((int)A.col_gid.size());
    }

    lupanar::Solver* s = nullptr;
    try {
      s = lupanar::create(A, ops);
    } catch (const std::exception& e) {
      if (!expect_throw) throw;
      if (rank == 0) std::printf("create() threw as expected: %s\nPASS\n", e.what());
      MPI_Finalize();
      return 0;
    }
    if (expect_throw) throw std::runtime_error("create() did not throw");
    std::vector<scalar>().swap(A.val);

    Reference ref;
    if (rank == 0 && !noref) ref.build(lat, k);

    // Where the rows of each rank go in the global vector (rank 0)
    std::vector<int> cnt(np), dis(np);
    MPI_Gather(&nloc, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<long long> gids;
    if (rank == 0) {
      for (int q = 0, o = 0; q < np; o += cnt[q++]) dis[q] = o;
      gids.resize(n);
    }
    MPI_Gatherv(A.row_gid.data(), nloc, MPI_LONG_LONG, gids.data(), cnt.data(), dis.data(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    for (int m : ms) {
      const size_t bs = (size_t)k * m;
      std::vector<scalar> x((size_t)nloc * bs), y((size_t)nloc * bs);
      for (int r = 0; r < nloc; ++r)
	for (int a = 0; a < k; ++a)
	  for (int j = 0; j < m; ++j) x[((size_t)r * k + a) * m + j] = rhs_entry(A.row_gid[r], a, j);

      // Correctness, with ordinary (pageable) arrays
      lupanar::solve(s, m, x.data(), y.data());
      double err = -1, res = -1;
      if (!noref) {
	std::vector<scalar> yall;
	std::vector<int> c2(np), d2(np);
	if (rank == 0) {
	  yall.resize((size_t)n * bs);
	  for (int q = 0; q < np; ++q) {
	    c2[q] = cnt[q] * (int)bs;
	    d2[q] = dis[q] * (int)bs;
	  }
	}
	MPI_Gatherv(y.data(), nloc * (int)bs, MPI_CXX_DOUBLE_COMPLEX, yall.data(), c2.data(), d2.data(), MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	if (rank == 0) {
	  std::vector<scalar> xg((size_t)n * bs), yg((size_t)n * bs), zr;
	  for (long long g = 0; g < n; ++g)
	    for (int a = 0; a < k; ++a)
	      for (int j = 0; j < m; ++j) xg[((size_t)g * k + a) * m + j] = rhs_entry(g, a, j);
	  for (long long r = 0; r < n; ++r) std::copy_n(yall.begin() + r * bs, bs, yg.begin() + gids[r] * bs);
	  ref.solve(m, xg, zr);
	  double d = 0, z = 0;
	  for (size_t e = 0; e < zr.size(); ++e) {
	    d = std::max(d, std::abs(yg[e] - zr[e]));
	    z = std::max(z, std::abs(zr[e]));
	  }
	  err = d / z;
	  res = ref.residual(m, xg, yg);
	}
      }

      // Timing, with the pinned buffers
      scalar *px = nullptr, *py = nullptr;
      lupanar::buffers(s, m, &px, &py);
      std::copy(x.begin(), x.end(), px);
      lupanar::Stats st, sum;
      lupanar::solve(s, m, px, py);
      void* gpu_scratch = nullptr;
      if (cold_gpu > 0 && cudaMalloc(&gpu_scratch, (size_t)cold_gpu << 20) != cudaSuccess)
	throw std::runtime_error("--cold-gpu: cudaMalloc failed");
      std::vector<char> cpu_scratch((size_t)cold_cpu << 20);
      MPI_Barrier(MPI_COMM_WORLD);
      const double t0 = MPI_Wtime();
      for (int it = 0; it < iters; ++it) {
	if (cold_gpu > 0 || cold_cpu > 0) {
	  if (gpu_scratch) {
	    cudaMemset(gpu_scratch, it & 255, (size_t)cold_gpu << 20);
	    cudaDeviceSynchronize();
	  }
	  std::fill(cpu_scratch.begin(), cpu_scratch.end(), (char)it);
	  MPI_Barrier(MPI_COMM_WORLD); // the ranks enter the solve together, as in an application
	}
	lupanar::solve(s, m, px, py, &st);
	sum.total += st.total;
	sum.lower += st.lower;
	sum.upper += st.upper;
	sum.mpi_recv += st.mpi_recv;
	sum.mpi_send += st.mpi_send;
	sum.kernel += st.kernel;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      const double wall = (MPI_Wtime() - t0) / std::max(iters, 1);
      if (gpu_scratch) cudaFree(gpu_scratch);
      double pinned_diff = 0;
      for (size_t e = 0; e < y.size(); ++e) pinned_diff = std::max(pinned_diff, std::abs(py[e] - y[e]));
      double loc[7] = {sum.total, sum.lower, sum.upper, sum.mpi_recv, sum.mpi_send, sum.kernel, pinned_diff}, mx[7];
      MPI_Reduce(loc, mx, 7, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      const lupanar::Info I = lupanar::info(s);
      int iloc[8] = {I.levels_l, I.levels_u, I.width_l, I.width_u, I.ghosts_l, I.sends_l, I.grid_l, I.grid_u}, imx[8];
      MPI_Reduce(iloc, imx, 8, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
      double tf = I.t_factor, tfmx = 0;
      MPI_Reduce(&tf, &tfmx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if (rank == 0) {
	const double ms_ = 1e3 / std::max(iters, 1);
	const bool pass = (noref || (err >= 0 && err < 1e-12)) && (!exact || noref || res < 1e-12) && mx[6] < 1e-12;
	if (!pass) failed = 1;
	std::printf("lupanar test N=%dx%dx%dx%d bc=%s P=%dx%dx%dx%d k=%d m=%d kernel=%s grid=%s(%d,%d)%s"
		    " stages=%d levels=%d,%d width=%d,%d ghosts=%d sends=%d t_factor=%.3fs"
		    " | err=%.2e resid=%.2e pinned_diff=%.1e"
		    " | wall=%.3fms max: total=%.3fms lower=%.3fms upper=%.3fms mpi_recv=%.3fms mpi_send=%.3fms kernel=%.3fms"
		    " | %s\n",
		    lat.N[0], lat.N[1], lat.N[2], lat.N[3], bc.c_str(), lat.P[0], lat.P[1], lat.P[2], lat.P[3], k, m,
		    I.kernel == 2 ? "tc" : "simt", grid.c_str(), imx[6], imx[7],
		    (std::string(shuffle ? " shuffle" : "") + (cold_gpu ? " cold_gpu=" + std::to_string(cold_gpu) : "") +
		     (cold_cpu ? " cold_cpu=" + std::to_string(cold_cpu) : "")).c_str(),
		    I.stages, imx[0], imx[1], imx[2], imx[3], imx[4], imx[5], tfmx, err, res, mx[6],
		    wall * 1e3, mx[0] * ms_, mx[1] * ms_, mx[2] * ms_, mx[3] * ms_, mx[4] * ms_, mx[5] * ms_,
		    pass ? "PASS" : "FAIL");
	std::fflush(stdout);
      }
    }
    lupanar::destroy(s);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "rank %d: %s\n", rank, e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return failed;
}
