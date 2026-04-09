#ifndef __SUPERBBLAS_RUNTIME_FEATURES__
#define __SUPERBBLAS_RUNTIME_FEATURES__

#include <algorithm>
#include <cstdlib>

namespace superbblas {

    /// Return the log level that may have been set by the environment variable SB_LOG
    /// \return int: log level
    /// The current log levels are:
    ///   * 0: no log (default)
    ///   * 1: some log

    inline int getLogLevel() {
        static int log_level = []() {
            const char *l = std::getenv("SB_LOG");
            if (l) return std::max(0, std::atoi(l));
            return 0;
        }();
        return log_level;
    }

    /// Return the debug level that may have been set by the environment variable SB_DEBUG
    /// \return int: debug level
    /// The current log levels are:
    ///   * 0: no extra checking (default)
    ///   * >= 1: GPU sync and MPI barriers before and after `copy` and `contraction`
    ///   * >= 2: verify all `copy` calls (expensive)

    inline int getDebugLevel() {
        static int debug_level = []() {
            const char *l = std::getenv("SB_DEBUG");
            if (l) return std::max(0, std::atoi(l));
            return 0;
        }();
        return debug_level;
    }

    /// Return whether to track memory consumption, which may have been set by the environment variable SB_TRACK_MEM
    /// \return bool: whether to track memory consumption
    /// The accepted value in the environment variable SB_TRACK_MEM are:
    ///   * 0: no tracking memory consumption (default)
    ///   * != 0: tracking memory consumption

    inline bool &getTrackingMemory() {
        static bool track_mem = []() {
            const char *l = std::getenv("SB_TRACK_MEM");
            if (l) return (0 != std::atoi(l));
            return false;
        }();
        return track_mem;
    }

    /// Return whether to track timings, which may have been set by the environment variable SB_TRACK_TIME
    /// \return bool: whether to track the time that critical functions take
    /// The accepted value in the environment variable SB_TRACK_TIME are:
    ///   * 0: no tracking time (default)
    ///   * != 0: tracking time

    inline bool &getTrackingTime() {
        static bool track_time = []() {
            const char *l = std::getenv("SB_TRACK_TIME");
            if (l) return (0 != std::atoi(l));
            return false;
        }();
        return track_time;
    }

    /// Return whether to sync before taking timings, which may have been set by the environment variable SB_TRACK_TIME_SYNC
    /// \return bool: whether to sync before taking the time that critical functions take
    /// The accepted value in the environment variable SB_TRACK_TIME_SYNC are:
    ///   * 0: no synchronization (default)
    ///   * != 0: do synchronization

    inline bool &getTrackingTimeSync() {
        static bool track_time_sync = []() {
            const char *l = std::getenv("SB_TRACK_TIME_SYNC");
            if (l) return (0 != std::atoi(l));
            return false;
        }();
        return track_time_sync;
    }

    /// Return whether to use immediate MPI calls instead of blocking
    /// \return bool: whether to use immediate MPI calls instead of blocking
    /// The accepted value in the environment variable SB_MPI_NONBLOCK are:
    ///   * 0: use blocking MPI calls
    ///   * != 0: use the immediate MPI calls (default)

    inline bool getUseMPINonBlock() {
        static bool mpi_nonblock = []() {
            const char *l = std::getenv("SB_MPI_NONBLOCK");
            if (l) return (0 != std::atoi(l));
            return true;
        }();
        return mpi_nonblock;
    }

    /// Return whether to use MPI_Alltoall instead of MPI_Send/Recv, which may have been set by the environment variable SB_USE_ALLTOALL
    /// \return bool: whether to use MPI_Alltoall instead of MPI_Send/Recv
    /// The accepted value in the environment variable SB_USE_ALLTOALL are:
    ///   * 0: use MPI_Send/Recv
    ///   * != 0: use MPI_Alltoallv (default)

    inline bool getUseAlltoall() {
        static bool use_alltoall = []() {
            const char *l = std::getenv("SB_USE_ALLTOALL");
            if (l) return (0 != std::atoi(l));
            return true;
        }();
        return use_alltoall;
    }

    /// Return whether to allow passing GPU pointers to MPI calls
    /// \return int: unspecified by the user when zero and greater than zero when allowing passing GPU pointers to MPI calls
    /// The accepted value in the environment variable SB_MPI_GPU are:
    ///   * 0: don't allow passing GPU pointers to MPI calls
    ///   * != 0: allow passing GPU pointers to MPI calls

    inline int getUseMPIGpu() {
        static int use_mpi_gpu = []() {
            const char *l = std::getenv("SB_MPI_GPU");
            if (l) return (0 != std::atoi(l) ? 1 : -1);
            return 0;
        }();
        return use_mpi_gpu;
    }

    /// Return the maximum size of the cache permutation for CPU in GiB
    /// \return int: value
    /// The accepted value in the environment variable SB_CACHEGB_CPU are:
    ///   * < 0: use the 10% of the total memory (default)
    ///   * >= 0: use that amount of GiB for cache

    inline double getMaxCacheGiBCpu() {
        static double size = []() {
            const char *l = std::getenv("SB_CACHEGB_CPU");
            if (l) return std::atof(l);
            return -1.0;
        }();
        return size;
    }

    /// Return the maximum size of the cache permutation for GPU in GiB
    /// \return int: value
    /// The accepted value in the environment variable SB_CACHEGB_GPU are:
    ///   * < 0: use the 10% of the total memory of the device (default)
    ///   * >= 0: use that amount of GiB for cache

    inline double getMaxCacheGiBGpu() {
        static double size = []() {
            const char *l = std::getenv("SB_CACHEGB_GPU");
            if (l) return std::atof(l);
            return -1.0;
        }();
        return size;
    }
}

#endif // __SUPERBBLAS_RUNTIME_FEATURES__
