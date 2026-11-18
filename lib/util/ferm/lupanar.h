// lupanar.h -- block ILU(0) preconditioner on the GPU: y = U \ (L \ x)
//
// Library form of the standalone LUPANAR benchmark of a GPU block triangular solve (lupanar_cuda_1.cpp).
// The two cooperative CUDA kernels (SIMT and FP64 tensor cores) are the benchmark's; this adds what the
// benchmark did not have:
//   - the ILU(0) factorization of a user matrix (the benchmark used made-up L and U),
//   - any block-sparse pattern and any distribution of the rows over the ranks (periodic lattices,
//     MGProton's partitions), instead of a non-periodic 4D Laplacian on an MPI Cartesian grid,
//   - a plain C++ interface without CUDA types.
//
// The matrix is block-sparse with dense k x k blocks; one block row per lattice site. `gid` numbers
// the sites globally and IS the elimination order (ascending). ILU(0) is the incomplete LU without
// fill-in; on a pattern without triangles (i ~ j, j ~ l, i ~ l) it changes only the diagonal blocks,
//     d_i = a_ii - sum_{l < i} a_il d_l^{-1} a_li,        M = (D + L_A) D^{-1} (D + U_A),
// the textbook ILU(0) in the order of `gid`. (SuperLU_DIST's ILU(0) with one dense supernode per site
// in its NATURAL order is the same except that it adds part of the dropped fill to the next block.)
// create() throws if the pattern has a triangle.
#ifndef LUPANAR_H
#define LUPANAR_H

#include <mpi.h>
#include <complex>
#include <vector>

namespace lupanar {

  /// Rows of the matrix held by this rank, block CSR.
  struct Matrix {
    MPI_Comm comm = MPI_COMM_SELF;	///< ranks sharing the matrix; every row is on exactly one of them
    int k = 0;				///< block size
    std::vector<long long> row_gid;	///< global site of every local row, in any order
    std::vector<int> ptr;		///< row_gid.size() + 1
    std::vector<long long> col_gid;	///< global site of every block (the diagonal block included)
    std::vector<std::complex<double>> val; ///< k*k per block, row-major; blocks repeated in a row add up
  };

  struct Options {
    int kernel = 0; ///< 0 auto (tensor cores when k > 16 and m > 1 on sm_80+), 1 SIMT, 2 FP64 tensor cores
    int grid = 0;   ///< thread blocks of the cooperative kernel: 0 = no more than the widest level
		    ///< needs, 1 = fill the GPU (the benchmark's choice)
  };

  /// What create() found; the same on every rank except the local counts
  struct Info {
    int k = 0;
    int nloc = 0;	      ///< local rows
    long long nglobal = 0;    ///< rows on all ranks
    int ranks = 1;
    int stages = 1;	      ///< longest chain of ranks that wait for each other in a triangular solve
    int levels_l = 0, levels_u = 0; ///< levels of the local rows (one grid.sync() per level)
    int width_l = 0, width_u = 0;   ///< rows in the widest level
    int ghosts_l = 0, ghosts_u = 0; ///< rows received from other ranks per solve
    int sends_l = 0, sends_u = 0;   ///< rows sent to other ranks per solve
    int kernel = 0;	      ///< 1 SIMT, 2 tensor cores (as launched by the last solve; 0 before)
    int grid_l = 0, grid_u = 0;	    ///< thread blocks launched by the last solve
    double t_factor = 0;      ///< seconds spent in the factorization (host)
  };

  /// Wall-clock seconds of one solve() on this rank
  struct Stats {
    double total = 0;	 ///< the whole call
    double lower = 0;	 ///< L solve: receive, kernel, send
    double upper = 0;	 ///< U solve
    double mpi_recv = 0; ///< waiting for the upstream ranks (includes their kernels)
    double mpi_send = 0;
    double kernel = 0;	 ///< kernel launches to the end of the stream (both solves)
  };

  struct Solver;

  /// Factorize (host, at once) and set up the GPU. Collective over m.comm. Uses the CUDA device that
  /// is current on the calling thread and never changes it; solve() must be called with the same one.
  Solver* create(const Matrix& m, const Options& ops = Options());

  /// y = U \ (L \ x) for m right-hand sides. x and y are host arrays of nloc * k * m values:
  /// value (r, a, j) of local row r (in the order of Matrix::row_gid), block component a and
  /// right-hand side j is at (r*k + a)*m + j. Collective over the matrix' communicator.
  void solve(Solver* s, int m, const std::complex<double>* x, std::complex<double>* y,
	     Stats* st = nullptr);

  /// Pinned host buffers of nloc * k * m values that can be given to solve() as x and y (faster
  /// than ordinary memory). They stay valid while m is among the last four values used.
  void buffers(Solver* s, int m, std::complex<double>** x, std::complex<double>** y);

  const Info& info(const Solver* s);

  void destroy(Solver* s);
}

#endif
