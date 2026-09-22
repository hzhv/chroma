// -*- C++ -*-
/*! \file
 * \brief Wall-clock residual tracing for MGProton solvers.  Default off: SB_MG_TRACE=<path.csv>.
 *
 * One CSV row per solver begin/iteration/end, carrying two cumulative clocks:
 *   elapsed_run_s    from markRunStart(), i.e. MGPROTON setup; spans setup and all inversions.
 *   elapsed_solve_s  from the start of the current inversion; empty during setup.
 * Both from steady_clock.  detail::w_time() is deliberately not reused -- it is system_clock
 * (performance.h:248), which is not monotonic.
 *
 * Callers sample right after `max_tol = max(div(normr, normr0))`, where the residual is already
 * host-resident: norm<1>() contracts with an explicit OnHost and max() opens with
 * make_sure(none, OnHost, ...).  So this adds no cudaDeviceSynchronize, no MPI call, and computes
 * no residual of its own.
 *
 * Rank 0 is the sole writer and nothing here is collective -- mr() serves both the outer solver
 * and the per-rank inner solver, and a collective in the latter deadlocks (see mgproton.cc:879).
 * Residuals on Local operators (the DD inner solvers) are per-rank, not global; callers pass that
 * through to the residual_scope column.
 */

#ifndef __INCLUDE_MGPROTON_TRACE__
#define __INCLUDE_MGPROTON_TRACE__

#include "chromabase.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#ifdef BUILD_SB
namespace Chroma
{
  namespace SB
  {
    /// Wall-clock residual tracing.  Enabled by SB_MG_TRACE=<path> (unset = off).
    namespace mgtrace
    {
      namespace detail
      {
	/// Accept 1/true/TRUE/yes/YES/on/ON and any nonzero integer; reject 0/false/no/off.
	/// Mirrors parseSuperLUEnvFlag (mgproton.cc:4622) rather than the bare getenv()
	/// truthiness used by SB_PREC_PROBE, under which SB_MG_TRACE=0 would switch tracing ON.
	inline bool parseFlag(const char* env, bool dflt)
	{
	  if (!env)
	    return dflt;
	  if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 ||
	      std::strcmp(env, "FALSE") == 0 || std::strcmp(env, "no") == 0 ||
	      std::strcmp(env, "NO") == 0 || std::strcmp(env, "off") == 0 ||
	      std::strcmp(env, "OFF") == 0)
	    return false;
	  return true;
	}

	/// Trace destination, from SB_MG_TRACE (or CHROMA_MG_TRACE).  Empty means disabled.
	/// A bare truthy value with no path is accepted and mapped to a default filename.
	inline const std::string& tracePath()
	{
	  static const std::string p = []() -> std::string {
	    const char* e = std::getenv("SB_MG_TRACE");
	    if (!e)
	      e = std::getenv("CHROMA_MG_TRACE");
	    if (!e || e[0] == '\0')
	      return std::string();
	    if (!parseFlag(e, true))
	      return std::string();
	    // "1"/"on"/... with no path -> default name
	    if (std::strchr(e, '/') == nullptr && std::strchr(e, '.') == nullptr)
	      return std::string("mgproton_trace.csv");
	    return std::string(e);
	  }();
	  return p;
	}

	struct Event {
	  long long solve_id;
	  const char* phase;
	  int rank;
	  const char* solver;
	  std::string prefix;
	  long long call_id;
	  long long parent_call_id;
	  long long outer_iteration;
	  long long local_iteration;
	  const char* event;
	  double residual;	   ///< NaN when not applicable (begin, or an aborted call)
	  const char* scope;
	  double elapsed_run_s;
	  double elapsed_solve_s;  ///< NaN during setup -> written as an empty field
	};

	struct State {
	  bool initialised = false;
	  bool writer = false;	 ///< rank 0 and SB_MG_TRACE set
	  bool run_started = false;
	  bool in_solve = false;
	  int rank = -1;
	  long long solve_id = 0;
	  long long next_call_id = 0;
	  std::chrono::steady_clock::time_point t_run0;
	  std::chrono::steady_clock::time_point t_solve0;
	  std::vector<long long> call_stack;	///< call_id of each open solver
	  std::vector<long long> iter_stack;	///< latest local_iteration of each open solver
	  std::vector<Event> buf;
	  std::string run_id;
	  bool header_written = false;
	};

	inline State& st()
	{
	  static State s;
	  return s;
	}

	/// Deferred one-time init.  Must not run before QDP is up, because it calls
	/// Layout::nodeNumber(); the first call comes from markRunStart(), long after init.
	inline void ensureInit()
	{
	  State& s = st();
	  if (s.initialised)
	    return;
	  s.initialised = true;
	  s.rank = (int)Layout::nodeNumber();
	  s.writer = (!tracePath().empty() && s.rank == 0);
	  if (!s.writer)
	    return;
	  const char* job = std::getenv("SLURM_JOB_ID");
	  s.run_id = job ? std::string(job) : std::string("nojob");
	  s.run_id += "-";
	  s.run_id += std::to_string((long long)std::chrono::system_clock::now()
				       .time_since_epoch()
				       .count() /
				     1000000000LL);
	  s.buf.reserve(8192);
	}

	inline void writeField(std::ofstream& f, double v)
	{
	  if (std::isnan(v))
	    f << "";  // empty cell: "not applicable", never a fabricated 0
	  else
	    f << v;
	}

	/// Append the buffer to the CSV and clear it.  Open/append/close per flush so an abrupt
	/// exit keeps everything already flushed.
	inline void flush()
	{
	  State& s = st();
	  if (!s.writer || s.buf.empty())
	    return;
	  std::ofstream f(tracePath().c_str(), std::ios::app);
	  if (!f)
	  {
	    std::fprintf(stderr, "MGTRACE: cannot open '%s' for append; tracing disabled\n",
			 tracePath().c_str());
	    std::fflush(stderr);
	    s.writer = false;
	    s.buf.clear();
	    return;
	  }
	  if (!s.header_written)
	  {
	    f << "run_id,solve_id,phase,rank,solver,prefix,call_id,parent_call_id,"
		 "outer_iteration,local_iteration,event,residual,residual_scope,"
		 "elapsed_run_s,elapsed_solve_s\n";
	    s.header_written = true;
	  }
	  f.setf(std::ios::scientific);
	  for (const Event& e : s.buf)
	  {
	    f.precision(9);
	    f << s.run_id << ',' << e.solve_id << ',' << e.phase << ',' << e.rank << ','
	      << e.solver << ',' << e.prefix << ',' << e.call_id << ',' << e.parent_call_id << ','
	      << e.outer_iteration << ',' << e.local_iteration << ',' << e.event << ',';
	    writeField(f, e.residual);
	    f << ',' << e.scope << ',';
	    writeField(f, e.elapsed_run_s);
	    f << ',';
	    writeField(f, e.elapsed_solve_s);
	    f << '\n';
	  }
	  s.buf.clear();
	}

	/// Backstop flush for the case where a run ends without closing the outermost solver.
	struct FinalFlush {
	  ~FinalFlush()
	  {
	    flush();
	  }
	};
	inline FinalFlush& finalFlush()
	{
	  static FinalFlush ff;
	  return ff;
	}

	inline void record(const char* solver, const std::string& prefix, long long call_id,
			   long long local_it, const char* event, double residual, bool local_scope)
	{
	  State& s = st();
	  if (!s.writer)
	    return;
	  // Timestamp taken HERE, at the event, not when the buffer is written out.
	  const auto now = std::chrono::steady_clock::now();
	  Event e;
	  e.solve_id = s.solve_id;
	  e.phase = s.in_solve ? "solve" : "setup";
	  e.rank = s.rank;
	  e.solver = solver;
	  e.prefix = prefix;
	  e.call_id = call_id;
	  e.parent_call_id = s.call_stack.size() >= 2
			       ? s.call_stack[s.call_stack.size() - 2]
			       : -1;
	  e.outer_iteration = s.iter_stack.empty() ? -1 : s.iter_stack.front();
	  e.local_iteration = local_it;
	  e.event = event;
	  e.residual = residual;
	  e.scope = std::isnan(residual) ? "" : (local_scope ? "local_rank" : "global");
	  e.elapsed_run_s =
	    s.run_started ? std::chrono::duration<double>(now - s.t_run0).count() : 0.0;
	  e.elapsed_solve_s =
	    s.in_solve ? std::chrono::duration<double>(now - s.t_solve0).count()
		       : std::numeric_limits<double>::quiet_NaN();
	  s.buf.push_back(std::move(e));
	}
      }

      /// True when tracing is on for this process.  Cheap after the first call.
      inline bool enabled()
      {
	detail::ensureInit();
	return detail::st().writer;
      }

      /// Start of the run clock.  Call at the top of MGPROTON setup.
      inline void markRunStart()
      {
	detail::ensureInit();
	detail::State& s = detail::st();
	if (s.run_started)
	  return;
	s.run_started = true;
	s.t_run0 = std::chrono::steady_clock::now();
	if (s.writer)
	  detail::finalFlush();	 // register the backstop now that we know we are writing
      }

      /// Setup is done; everything after this belongs to an inversion.  Call right after the
      /// "MGPROTON invertor ready" print.
      inline void setPhaseSolve()
      {
	detail::ensureInit();
	detail::st().in_solve = true;
	detail::flush();  // get the setup rows on disk before the first inversion
      }

      /// RAII marker for one solver call.  Emits `begin` on construction and `end` on destruction,
      /// so early returns (num_cols == 0, max(normr0) == 0) still close the record.
      struct Scope {
	long long id = -1;
	const char* solver;
	std::string prefix;
	long long last_it = -1;
	double last_res = std::numeric_limits<double>::quiet_NaN();
	bool last_local = false;
	bool ended = false;
	bool on = false;

	Scope(const char* solver_, const std::string& prefix_) : solver(solver_), prefix(prefix_)
	{
	  on = enabled();
	  if (!on)
	    return;
	  detail::State& s = detail::st();
	  // A solver opening at depth 0 during the solve phase is a new inversion: l0, and every
	  // level nested under it, then share one solve origin.  Detecting it by depth rather than
	  // by hooking one call site also covers the MultipleLatticeFermions path for free.
	  if (s.call_stack.empty() && s.in_solve)
	  {
	    ++s.solve_id;
	    s.t_solve0 = std::chrono::steady_clock::now();
	  }
	  id = ++s.next_call_id;
	  s.call_stack.push_back(id);
	  s.iter_stack.push_back(-1);
	  detail::record(solver, prefix, id, -1, "begin",
			 std::numeric_limits<double>::quiet_NaN(), false);
	}

	/// One iteration's residual, already host-resident.
	void iter(long long it, double residual, bool local_scope)
	{
	  if (!on)
	    return;
	  detail::State& s = detail::st();
	  if (!s.iter_stack.empty())
	    s.iter_stack.back() = it;
	  last_it = it;
	  last_res = residual;
	  last_local = local_scope;
	  detail::record(solver, prefix, id, it, "iteration", residual, local_scope);
	}

	/// The solver's summary line.
	void finish(long long it, double residual, bool local_scope)
	{
	  if (!on || ended)
	    return;
	  ended = true;
	  last_it = it;
	  last_res = residual;
	  last_local = local_scope;
	  detail::record(solver, prefix, id, it, "end", residual, local_scope);
	  pop();
	}

	~Scope()
	{
	  if (!on)
	    return;
	  if (!ended)
	  {
	    // Early return: close the call with whatever the last residual was.
	    detail::record(solver, prefix, id, last_it, "end", last_res, last_local);
	    pop();
	  }
	}

      private:
	void pop()
	{
	  detail::State& s = detail::st();
	  if (!s.call_stack.empty())
	    s.call_stack.pop_back();
	  if (!s.iter_stack.empty())
	    s.iter_stack.pop_back();
	  // Outermost solver closed: one inversion (or the setup-phase solve) is done.  Flush so a
	  // crash or a walltime kill still leaves a usable trace.
	  if (s.call_stack.empty())
	    detail::flush();
	}
      };
    }
  }
}
#endif // BUILD_SB
#endif // __INCLUDE_MGPROTON_TRACE__
