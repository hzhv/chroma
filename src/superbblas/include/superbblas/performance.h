#ifndef __SUPERBBLAS_PERFORMANCE__
#define __SUPERBBLAS_PERFORMANCE__

#include "platform.h"
#include "runtime_features.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/// If SUPERBBLAS_USE_NVTX macro is defined, then the tracker reports to the NVIDIA profiler
/// the tracker name and duration

#ifdef SUPERBBLAS_USE_NVTX
#    include <nvToolsExt.h>
#endif

namespace superbblas {

    namespace detail {
        /// Return the relative cost of the multiplication with respect to real floating point
        /// \tparam T: type to consider
        template <typename T>
        struct multiplication_cost; // { constexpr static double value = 0.0; };
#ifdef SUPERBBLAS_USE_FLOAT16
        template <> struct multiplication_cost<_Float16> {
            constexpr static double value = 0.5;
        };
#endif
        template <> struct multiplication_cost<float> {
            constexpr static double value = 1.0;
        };
        template <> struct multiplication_cost<double> {
            constexpr static double value = 2.0;
        };
#ifdef SUPERBBLAS_USE_FLOAT16
        template <> struct multiplication_cost<std::complex<_Float16>> {
            constexpr static double value = 2.0;
        };
#endif
        template <> struct multiplication_cost<std::complex<float>> {
            constexpr static double value = 4.0;
        };
        template <> struct multiplication_cost<std::complex<double>> {
            constexpr static double value = 8.0;
        };
#ifdef SUPERBBLAS_USE_FLOAT16
        template <> struct multiplication_cost<_Complex _Float16> {
            constexpr static double value = 2.0;
        };
#endif
        template <> struct multiplication_cost<_Complex float> {
            constexpr static double value = 4.0;
        };
        template <> struct multiplication_cost<_Complex double> {
            constexpr static double value = 8.0;
        };
        template <> struct multiplication_cost<int> {
            constexpr static double value = 1.0;
        };
        template <> struct multiplication_cost<std::size_t> {
            constexpr static double value = 2.0;
        };
    }

    /// Get total memory allocated on the host/cpu if tracking memory consumption (see `getTrackingMemory`)

    inline double &getCpuMemUsed(Session session) {
        static std::array<double, 256> mem{{}};
        return mem[session];
    }

    /// Get total memory allocated on devices if tracking memory consumption (see `getTrackingMemory`)

    inline double &getGpuMemUsed(Session session) {
        static std::array<double, 256> mem{{}};
        return mem[session];
    }

#ifdef SUPERBBLAS_USE_GPU
    using GpuEvent = SUPERBBLAS_GPU_SYMBOL(Event_t);
    using TimingGpuEvent = std::array<GpuEvent, 2>;

    /// A list of pairs of starting and ending timing events
    using TimingGpuEvents = std::vector<TimingGpuEvent>;
#endif

    /// Performance metrics, time, memory usage, etc
    struct Metric {
        double cpu_time;   ///< wall-clock time for the cpu
        double gpu_time;   ///< wall-clock time for the gpu
        double flops;      ///< single precision multiplications
        double memops;     ///< bytes read and write from memory
        double arity;      ///< entities processed (eg, rhs for matvecs)
        double max_mem;    ///< memory usage in bytes
        std::size_t calls; ///< number of times the function was called
#ifdef SUPERBBLAS_USE_GPU
        /// List of start-end gpu events for calls of this function in course
        TimingGpuEvents timing_events;
#endif
        /// Name of the parent call
        std::string parent;
        /// Whether the parent has been set
        bool is_parent_set;
        Metric()
            : cpu_time(0),
              gpu_time(0),
              flops(0),
              memops(0),
              arity(0),
              max_mem(0),
              calls(0),
              is_parent_set(false) {}
    };

    /// Type for storing the timings
    using Timings = std::unordered_map<std::string, Metric>;

    /// Return the performance timings
    inline Timings &getTimings(Session session) {
        static std::vector<Timings> timings(256, Timings{16});
        return timings[session];
    }

    /// Type for storing the memory usage
    using CacheUsage = std::unordered_map<std::string, double>;

    /// Return the performance timings
    inline CacheUsage &getCacheUsage(Session session) {
        static std::vector<CacheUsage> cacheUsage(256, CacheUsage{16});
        return cacheUsage[session];
    }

    namespace detail {

        /// Template namespace for managing the gpu timings
        template <typename XPU> struct TimingEvents;

#ifdef SUPERBBLAS_USE_GPU
        template <> struct TimingEvents<Gpu> {
            /// Gpu timing event
            using TimingEvent = TimingGpuEvent;

            /// Extract the timings from the recorded events just finished, remove them from the vector,
            /// and return the accumulated time
            /// \param events: (in/out) vector of events to inspect

            static double processEvents(TimingGpuEvents &events) {
                double new_time = 0;
                events.erase(std::remove_if(events.begin(), events.end(),
                                            [&](const TimingGpuEvent &ev) {
                                                // Try to get the elapsed time between the two events
                                                float ms = 0;
                                                auto err = SUPERBBLAS_GPU_SYMBOL(EventElapsedTime)(
                                                    &ms, ev[0], ev[1]);
                                                if (err == SUPERBBLAS_GPU_SYMBOL(Success)) {
                                                    // If successful, register the time and erase the entry in the vector
                                                    new_time += ms / 1000.0;
                                                    return true;
                                                } else {
                                                    // Otherwise, do nothing
                                                    return false;
                                                }
                                            }),
                             events.end());
                return new_time;
            }

            /// Return a gpu timing event and start counting
            /// \param xpu: context

            static TimingEvent startRecordingEvent(const Gpu &xpu) {
                TimingEvent tev;
                setDevice(xpu);
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventCreate)(&tev[0]));
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventCreate)(&tev[1]));
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventRecord)(tev[0], getStream(xpu)));
                return tev;
            }

            /// Mark the end of a recording
            /// \param tev: gpu timing event
            /// \param xpu: context

            static void endRecordingEvent(const TimingEvent &tev, const Gpu &xpu) {
                setDevice(xpu);
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(EventRecord)(tev[1], getStream(xpu)));
            }

            static void updateGpuTimingEvents(Metric &metric) {
                metric.gpu_time += processEvents(metric.timing_events);
            }

            static void updateGpuTimingEvents(Metric &metric, const TimingEvent &tev) {
                updateGpuTimingEvents(metric);
                metric.timing_events.push_back(tev);
            }
        };
#endif

        /// Dummy implementation of `TimingEvents` for cpu

        template <> struct TimingEvents<Cpu> {
            using TimingEvent = char;
            static TimingEvent startRecordingEvent(const Cpu &) { return 0; }
            static void endRecordingEvent(const TimingEvent &, const Cpu &) {}
            static void updateGpuTimingEvents(Metric &) {}
            static void updateGpuTimingEvents(Metric &, const TimingEvent &) {}
        };

        /// Stack of function calls being tracked
        using CallStack = std::vector<std::string>;

        /// Return the current function call stack begin tracked
        inline CallStack &getCallStackWithPath(Session session) {
            static std::vector<CallStack> callStack(256, CallStack{});
            return callStack[session];
        }

        /// Push function call to be tracked
        inline void pushCall(std::string funcName, Session session) {
            if (getCallStackWithPath(session).empty()) {
                // If the stack is empty, just append the function name
                getCallStackWithPath(session).push_back(funcName);
            } else {
                // Otherwise, push the previous one appending "/`funcName`"
                getCallStackWithPath(session).push_back(getCallStackWithPath(session).back() + "/" +
                                                        funcName);
            }
        }

        /// Pop function call from the stack
        inline std::array<std::string, 2> popCall(Session session) {
            assert(getCallStackWithPath(session).size() > 0);
            std::string back = getCallStackWithPath(session).back();
            getCallStackWithPath(session).pop_back();
            return {back, getCallStackWithPath(session).size() > 0
                              ? getCallStackWithPath(session).back()
                              : std::string{}};
        }

        /// Return the number of seconds from some start
        inline double w_time() {
            return std::chrono::duration<double>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        }

        /// Track time between creation and destruction of the object
        template <typename XPU> struct tracker {
            /// Whether to track time
            bool track_time;
            /// Whether to track memory usage
            bool track_mem;
            /// Whether the tacker has been stopped
            bool stopped;
#ifdef SUPERBBLAS_USE_NVTX
            /// Whether the tracker has reported the end of the task
            bool reported;
#endif
            /// Name of the function being tracked
            const std::string funcName;
            /// Memory usage at that point
            const double mem_cpu, mem_gpu;
            /// Instant of the construction
            const std::chrono::time_point<std::chrono::system_clock> start;
            /// Context
            const XPU xpu;
            /// Cpu elapsed time
            double elapsedTime;
            /// Gpu starting and ending events
            typename TimingEvents<XPU>::TimingEvent timingEvent;
            /// Single precision multiplications
            double flops;
            /// Bytes read and write from memory
            double memops;
            /// Entities processed (eg, rhs for matvecs)
            double arity;

            /// Start a tracker
            tracker(const std::string &funcName, XPU xpu, bool timeAnyway = false)
                : track_time(timeAnyway || getTrackingTime()),
                  track_mem(getTrackingMemory()),
                  stopped(!(track_time || track_mem)),
#ifdef SUPERBBLAS_USE_NVTX
                  reported(false),
#endif
                  funcName(funcName),
                  mem_cpu(track_mem ? getCpuMemUsed(xpu.session) : 0),
                  mem_gpu(track_mem ? getGpuMemUsed(xpu.session) : 0),
                  start(track_time ? std::chrono::system_clock::now()
                                   : std::chrono::time_point<std::chrono::system_clock>{}),
                  xpu(xpu),
                  elapsedTime(0),
                  flops(0),
                  memops(0),
                  arity(0) {

                if (!stopped) {
                    pushCall(funcName, xpu.session);
                    if (track_time) timingEvent = TimingEvents<XPU>::startRecordingEvent(xpu);
                }
#ifdef SUPERBBLAS_USE_NVTX
                // Register this scope of time starting
                nvtxRangePushA(this->funcName.c_str());
#endif
            }

            ~tracker() { stop(); }

            /// Stop the tracker and store the timing
            void stop() {
#ifdef SUPERBBLAS_USE_NVTX
                if (!reported) {
                    // Register this scope of time finishing
                    nvtxRangePop();
                    reported = true;
                }
#endif

                if (stopped) return;
                stopped = true;

                if (track_time) {
                    // Record gpu ending event
                    TimingEvents<XPU>::endRecordingEvent(timingEvent, xpu);

                    // Enforce a synchronization
                    if (getTrackingTimeSync()) sync(xpu);

                    // Count elapsed time since the creation of the object
                    elapsedTime =
                        std::chrono::duration<double>(std::chrono::system_clock::now() - start)
                            .count();
                }

                // Pop out this call and get a string representing the current call stack
                auto funcNameWithStackAndParent = popCall(xpu.session);
                const std::string &funcNameWithStack = funcNameWithStackAndParent[0];
                const std::string &parent = funcNameWithStackAndParent[1];

                if (track_time) {
                    // Record the time
                    auto &timing = getTimings(xpu.session)[funcNameWithStack];
                    timing.cpu_time += elapsedTime;
                    timing.flops += flops;
                    timing.memops += memops;
                    timing.arity += arity;
                    timing.calls++;
                    if (!timing.is_parent_set) {
                        timing.parent = parent;
                        timing.is_parent_set = true;
                    }
                    TimingEvents<XPU>::updateGpuTimingEvents(timing, timingEvent);

                    // Add flops and memops to parent call
                    if (parent.size() > 0) {
                        auto &parent_timing = getTimings(xpu.session)[parent];
                        parent_timing.flops += flops;
                        parent_timing.memops += memops;
                    }
                }

                // Record memory not released
                if (track_mem) {
                    getCacheUsage(xpu.session)[funcNameWithStack] +=
                        getCpuMemUsed(xpu.session) - mem_cpu + getGpuMemUsed(xpu.session) - mem_gpu;
                }
            }

            /// Stop the tracker and return timing
            double stopAndGetElapsedTime() {
                stop();
                return elapsedTime;
            }

            // Forbid copy constructor and assignment operator
            tracker(const tracker &) = delete;
            tracker &operator=(tracker const &) = delete;
        };
    }

    /// Reset all tracked timings
    inline void resetTimings() {
        for (Session session = 0; session < 256; ++session) getTimings(session).clear();
    }

    /// Report all tracked timings
    /// \param s: stream to write the report

    template <typename OStream> void reportTimings(OStream &s) {
        if (!getTrackingTime()) return;

        // Save stream state
        std::ios old_state(nullptr);
        old_state.copyfmt(s);

        // Print the timings alphabetically
        s << "Timing of superbblas kernels:" << std::endl;
        s << "-----------------------------" << std::endl;
        std::vector<std::string> names;
        for (Session session = 0; session < 256; ++session)
            for (const auto &it : getTimings(session)) names.push_back(it.first);
        std::sort(names.begin(), names.end());

        // Update the gpu timings
#ifdef SUPERBBLAS_USE_GPU
        // Aggregate all gpu time of the called functions; that's used as an approximation of the
        // total gpu time spent by the function if no gpu time has been actually recorded.
        // NOTE: visiting the functions in reverse lexicographic order is a way to guarantee visiting
        // all children before visiting the parent node
        std::unordered_map<std::string, double> children_gpu_time(16);
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto &name = names[names.size() - i - 1];
            for (Session session = 0; session < 256; ++session) {
                auto it = getTimings(session).find(name);
                if (it != getTimings(session).end()) {
                    detail::TimingEvents<detail::Gpu>::updateGpuTimingEvents(it->second);
                    children_gpu_time[it->second.parent] +=
                        (it->second.gpu_time > 0 ? it->second.gpu_time
                                                 : children_gpu_time[it->first]);
                }
            }
        }
#endif

        for (const auto &name : names) {
            // Gather the metrics for a given function on all sessions
            double cpu_time = 0, gpu_time = 0, flops = 0, memops = 0, calls = 0;
            for (Session session = 0; session < 256; ++session) {
                auto it = getTimings(session).find(name);
                if (it != getTimings(session).end()) {
                    cpu_time += it->second.cpu_time;
                    gpu_time += it->second.gpu_time;
                    flops += it->second.flops;
                    memops += it->second.memops;
                    calls += it->second.calls;
                }
            }
#ifdef SUPERBBLAS_USE_GPU
            // Use the aggregate gpu time of the called functions if no gpu time was recorded
            if (gpu_time == 0) gpu_time = children_gpu_time[name];
#endif
            // For computing flops and memory bandwidth, use gpu time if given
            double time = (gpu_time > 0 ? gpu_time : cpu_time);
            double gflops_per_sec = (time > 0 ? flops / time : 0) / 1000.0 / 1000.0 / 1000.0;
            double gmemops_per_sec = (time > 0 ? memops / time : 0) / 1024.0 / 1024.0 / 1024.0;
            double intensity = (memops > 0 ? flops / (memops / sizeof(float)) : 0.0);
            s << name << " : " << std::fixed << std::setprecision(3) << cpu_time << " s ("
#ifdef SUPERBBLAS_USE_GPU
              << "gpu_time: " << gpu_time << " "
#endif
              << "calls: " << std::setprecision(0) << calls                                      //
              << " flops: " << flops                                                             //
              << " bytes: " << memops                                                            //
              << " GFLOPs_single: " << std::scientific << std::setprecision(3) << gflops_per_sec //
              << " GBYTES/s: " << gmemops_per_sec                                                //
              << " intensity: " << std::fixed << std::setprecision(1) << intensity               //
              << " )" << std::endl;
        }

	// Restore stream state
        s.copyfmt(old_state);
    }

    /// Report all tracked cache memory usage
    /// \param s: stream to write the report

    template <typename OStream> void reportCacheUsage(OStream &s) {
        if (!getTrackingMemory()) return;

        // Print the timings alphabetically
        s << "Cache usage of superbblas kernels:" << std::endl;
        s << "-----------------------------" << std::endl;
        std::vector<std::string> names;
        for (Session session = 0; session < 256; ++session)
            for (const auto &it : getCacheUsage(session)) names.push_back(it.first);
        std::sort(names.begin(), names.end());
        for (const auto &name : names) {
            double total = 0;
            for (Session session = 0; session < 256; ++session) {
                auto it = getCacheUsage(session).find(name);
                if (it != getCacheUsage(session).end()) total += it->second;
            }
            s << name << " : " << total / 1024 / 1024 / 1024 << " GiB" << std::endl;
        }
    }

    namespace detail {
        /// Structure to store the memory allocations
        /// NOTE: the only instance is expected to be in `getAllocations`.

        struct Allocations : public std::unordered_map<void *, std::size_t> {
            Allocations(std::size_t num_backets)
                : std::unordered_map<void *, std::size_t>{num_backets} {}
        };

        /// Return all current allocations

        inline Allocations &getAllocations(Session session) {
            static std::vector<Allocations> allocs(256, Allocations{16});
            return allocs[session];
        }
    }

    /// Report current memory allocations
    /// \param s: stream to write the report

    template <typename OStream> void reportCurrentMemoryAllocations(OStream &s) {
        if (!getTrackingMemory()) return;

        // Check if there is some memory allocation
        bool some_alloc = false;
        for (Session i = 0; i < 256; ++i)
            if (detail::getAllocations(i).size() > 0) some_alloc = true;
        if (!some_alloc) return;

        // Print current allocations
        s << "Current memory allocation from superbblas:" << std::endl;
        s << "-----------------------------" << std::endl;
        for (Session i = 0; i < 256; ++i)
            for (const auto &it : detail::getAllocations(i))
                s << it.first << ": " << (double)it.second / 1024 / 1024 / 1024 << " GiB"
                  << std::endl;
    }

    /// Throw an exception after reporting the current memory allocations if there is any
    /// \param s: stream to write the report

    template <typename OStream> void checkForMemoryLeaks(OStream &s) {
        if (!getTrackingMemory()) return;

        // Check if there is some memory allocation
        bool some_alloc = false;
        for (Session i = 0; i < 256; ++i)
            if (detail::getAllocations(i).size() > 0) some_alloc = true;

        // Check if the counters are also zero
        double total_cpu_used = 0, total_gpu_used = 0;
        for (Session s = 0; s < 256; s++) total_cpu_used += getCpuMemUsed(s);
        for (Session s = 0; s < 256; s++) total_gpu_used += getGpuMemUsed(s);
        if (!some_alloc && (total_cpu_used > 0 || total_gpu_used > 0))
            throw std::runtime_error("checkForMemoryLeaks: memory counters are not consistent");

        if (!some_alloc) return;

        // Print the allocations
        reportCurrentMemoryAllocations(s);

        throw std::runtime_error("checkForMemoryLeaks: some allocations are still around");
    }
}

#endif // __SUPERBBLAS_PERFORMANCE__
