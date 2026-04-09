#ifndef __SUPERBBLAS_DIST__
#define __SUPERBBLAS_DIST__

#include "tensor.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef SUPERBBLAS_USE_MPI
#    include "mpi.h"

// If using Open-MPI check if supporting GPU aware API
#    if OMPI_MAJOR_VERSION >= 4 || MPICH_NUMVERSION >= 40000000
#        include "mpi-ext.h"
#    endif // OMPI_MAJOR_VERSION >= 4
#    if defined(SUPERBBLAS_USE_CUDA) &&                                                            \
        (defined(MPIX_CUDA_AWARE_SUPPORT) ||                                                       \
         (defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA))
#        define SUPERBBLAS_TEST_MPI_GPU
#    endif
#    if defined(SUPERBBLAS_USE_HIP) &&                                                             \
        (defined(MPIX_ROCM_AWARE_SUPPORT) ||                                                       \
         (defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM))
#        define SUPERBBLAS_TEST_MPI_GPU
#    endif
#endif // SUPERBBLAS_USE_MPI

namespace superbblas {

    /// First coordinate and size of a range of coordinates supported on a process/component.
    /// See ConstPartition.

    template <std::size_t N> using PartitionItem = std::array<Coor<N>, 2>;

    /// Distribution of the elements of a N-dimension tensor among the processes/components
    ///
    /// This structure is a three-dimensional array with dimensions [P][2][N], where P is the number of
    /// processes/components and the tensors have N dimensions. The values of the array indicates that
    /// the i-th process/component stores all elements with coordinates c such that for all j dimensions:
    ///    (*this)[i][0][j] <= c[j] <= (*this)[i][0][j] + (*this)[i][1][j],      (mod dim[j]),
    /// where dim are tensor dimensions. In other words, each process/components stores a continuum range
    /// of coordinates such that the first coordinate at the j-dimension is [i][0][j] and
    /// stores up to [i][1][j] elements in that j dimension.

    template <std::size_t N> using ConstPartition = const PartitionItem<N> *;

    /// Callback to execute to finish an operation
    using Request = std::function<void(void)>;

    /// Wait until the operation is finished
    /// \param request: operation to finish

    inline void wait(const Request &request) {
        if (request) request();
    }

    template <std::size_t N>
    std::vector<std::array<Coor<N>, 2>> make_hole(const Coor<N> &from, const Coor<N> &size,
                                                  const Coor<N> &hole_from,
                                                  const Coor<N> &hole_size, const Coor<N> &dim);

    namespace detail {

        /// Type use in MPI calls to indicate cardinality and displacements
        using MpiInt = int;

        /// Set a user-defined MPI type that allows to send up 256 GiB in a single calls
        /// MPI type with this size; the maximum package size is
        constexpr std::size_t MpiTypeSize = 64;

        /// First coordinate and size of a range
        template <std::size_t N> using From_size_item = PartitionItem<N>;
        /// List of ranges
        template <std::size_t N> using From_size = std::vector<From_size_item<N>>;
        /// From_size iterator
        template <std::size_t N> using From_size_iterator = const From_size_item<N> *;
        template <std::size_t N> using Proc_ranges = std::vector<From_size<N>>;

        /// Self[source range index on this process][dest process][range index]: list of ranges, or
        /// self[dest range index on this process][source process][range index]: list of ranges
        template <std::size_t N>
        using Range_proc_range_ranges = std::vector<std::vector<Proc_ranges<N>>>;

        template <std::size_t Nd0, std::size_t Nd1>
        using PairPerms = std::tuple<Coor<Nd0>, Coor<Nd1>>;

        // Supported types for contractions: the ones supported by superbblas excepting int
        template <typename T> struct supported_type_for_contractions {
            static constexpr bool value = supported_type<T>::value;
        };
        template <> struct supported_type_for_contractions<int> {
            static constexpr bool value = false;
        };

        enum ForceLocal { dontForceLocal, doForceLocal };

        //
        // Auxiliary functions
        //

        /// Return the given number if it is multiple of the second number or the next multiple
        /// \param n: number to try
        /// \param base: multiple to try
        /// \return: ceil(n/base)*base

        template <typename T> constexpr T multiple_of(T n, T base) {
            return (n + base - 1) / base * base;
        }

        /// Return the permutations associated to two order

        template <std::size_t Nd0, std::size_t Nd1>
        PairPerms<Nd0, Nd1> get_perms(const Order<Nd0> &o0, const Order<Nd1> &o1) {
            return PairPerms<Nd0, Nd1>{find_permutation<Nd1, Nd0>(o1, o0),
                                       find_permutation<Nd0, Nd1>(o0, o1)};
        }

#ifdef SUPERBBLAS_USE_MPI
        /// Communicator
        struct MpiComm {
            unsigned int nprocs; ///< Number of processes
            unsigned int rank;   ///< Process id
            MPI_Comm comm;       ///< MPI communicator
        };

        /// Return a communicator for a MPI_Comm
        inline MpiComm get_comm(MPI_Comm comm) {
            int nprocs, rank;
            MPI_check(MPI_Comm_size(comm, &nprocs));
            MPI_check(MPI_Comm_rank(comm, &rank));
            return MpiComm{(unsigned int)nprocs, (unsigned int)rank, comm};
        }

#endif // SUPERBBLAS_USE_MPI

        /// Communicator
        struct SelfComm {
            unsigned int nprocs; ///< Number of processes
            unsigned int rank;   ///< Process id
        };

        /// Return a communicator for a MPI_Comm
        inline SelfComm get_comm() { return SelfComm{1u, 0u}; }

#ifdef SUPERBBLAS_USE_MPI
        /// Return the MPI_datatype for a type returned by `NativeMpiDatatype`
        inline MPI_Datatype get_mpi_datatype() {
            if (MpiTypeSize == sizeof(char)) return MPI_CHAR;
            if (MpiTypeSize == sizeof(float)) return MPI_FLOAT;
            if (MpiTypeSize == sizeof(double)) return MPI_DOUBLE;
            MPI_Datatype t;
            MPI_check(MPI_Type_contiguous(MpiTypeSize, MPI_CHAR, &t));
            MPI_check(MPI_Type_commit(&t));
            return t;
        }
#endif // SUPERBBLAS_USE_MPI

        /// Component of a tensor
        template <std::size_t Nd, typename T, typename XPU> struct Component {
            vector<T, XPU> it;        ///< data
            Coor<Nd> dim;             ///< dimension of the tensor
            unsigned int componentId; ///< Component Id
            Mask<XPU> mask_it;        ///< Mask

            template <
                typename Q = T,
                typename std::enable_if<std::is_same<Q, typename std::remove_const<Q>::type>::value,
                                        bool>::type = true>
            operator Component<Nd, const Q, XPU>() const {
                return {it, dim, componentId, mask_it};
            }

            template <typename Q = T,
                      typename std::enable_if<
                          !std::is_same<Q, typename std::remove_const<Q>::type>::value,
                          bool>::type = true>
            operator Component<Nd, typename std::remove_const<Q>::type, XPU>() const {
                return {it, dim, componentId, mask_it};
            }

            Component withNewContext(const XPU &xpu) const {
                return {it.withNewContext(xpu), dim, componentId, mask_it.withNewContext(xpu)};
            }
        };

        /// A tensor composed of several components
        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        using Components_tmpl =
            std::pair<std::vector<Component<Nd, T, XPU0>>, std::vector<Component<Nd, T, XPU1>>>;

#ifdef SUPERBBLAS_USE_GPU
        /// A tensor composed of several CPU and GPU elements
        template <std::size_t Nd, typename T> using Components = Components_tmpl<Nd, T, Gpu, Cpu>;
#else
        /// A tensor composed of only of CPU components
        template <std::size_t Nd, typename T> using Components = Components_tmpl<Nd, T, Cpu, Cpu>;
#endif // SUPERBBLAS_USE_GPU

        template <std::size_t Nd, typename T, typename Comm>
        Components<Nd, T> get_components(T **v, const MaskType **mask, const Context *ctx,
                                         unsigned int ncomponents, From_size_iterator<Nd> p,
                                         const Comm &comm, Session session) {
            // Get components on the local process
            From_size_iterator<Nd> fs = p + comm.rank * ncomponents;

            Components<Nd, T> r;
            for (unsigned int i = 0; i < ncomponents; ++i) {
                MaskType *maski = mask ? (MaskType *)mask[i] : (MaskType *)nullptr;
                const auto vol_fsi = volume(fs[i][1]);
                const auto &fsi1 = vol_fsi > 0 ? fs[i][1] : Coor<Nd>{{}};
                switch (ctx[i].plat) {
#ifdef SUPERBBLAS_USE_GPU
                case CPU:
                    r.second.push_back(
                        Component<Nd, T, Cpu>{to_vector(v[i], vol_fsi, ctx[i].toCpu(session)), fsi1,
                                              i, to_vector(maski, vol_fsi, ctx[i].toCpu(session))});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
                case GPU:
                    r.first.push_back(
                        Component<Nd, T, Gpu>{to_vector(v[i], vol_fsi, ctx[i].toGpu(session)), fsi1,
                                              i, to_vector(maski, vol_fsi, ctx[i].toGpu(session))});
                    assert(!v[i] || getPtrDevice(v[i]) == ctx[i].device);
                    break;
#else // SUPERBBLAS_USE_GPU
                case CPU:
                    r.first.push_back(
                        Component<Nd, T, Cpu>{to_vector(v[i], vol_fsi, ctx[i].toCpu(session)), fsi1,
                                              i, to_vector(maski, vol_fsi, ctx[i].toCpu(session))});
                    assert(!v[i] || getPtrDevice(v[i]) == CPU_DEVICE_ID);
                    break;
#endif
                default: throw std::runtime_error("Unsupported platform");
                }
            }
            return r;
        }

        /// Return a const version of `Component_tmpl`

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        std::size_t num_components(const Components_tmpl<Nd, T, XPU0, XPU1> &c) {
            return c.first.size() + c.second.size();
        }

        /// Return a const version of `Component_tmpl`

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        Components_tmpl<Nd, const T, XPU0, XPU1>
        toConst(const Components_tmpl<Nd, T, XPU0, XPU1> &c) {
            return {std::vector<Component<Nd, const T, XPU0>>(c.first.begin(), c.first.end()),
                    std::vector<Component<Nd, const T, XPU1>>(c.second.begin(), c.second.end())};
        }

        /// Return a non-const version of `Component_tmpl`

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        Components_tmpl<Nd, typename std::remove_const<T>::type, XPU0, XPU1>
        toNonConst(const Components_tmpl<Nd, T, XPU0, XPU1> &c) {
            return {std::vector<Component<Nd, typename std::remove_const<T>::type, XPU0>>(
                        c.first.begin(), c.first.end()),
                    std::vector<Component<Nd, typename std::remove_const<T>::type, XPU1>>(
                        c.second.begin(), c.second.end())};
        }

        /// Print a message in the standard error
        /// \param comm: a communicator
        /// \param msg: thing to print

        template <typename Comm, typename Msg> void print(const Comm &comm, const Msg msg) {
            std::cerr << "[" << comm.rank << "] " << msg << std::endl;
            std::cerr.flush();
        }

        template <typename Ostream, typename T, std::size_t N>
        Ostream &operator<<(Ostream &s, const std::array<T, N> &v) {
            s << "{";
            for (const auto &i : v) s << " " << i;
            s << "}";
            return s;
        }

        template <typename Ostream, typename T>
        Ostream &operator<<(Ostream &s, const vector<T, Cpu> &v) {
            s << "{";
            for (const auto &i : v) s << " " << i;
            s << "}";
            return s;
        }

        /// Print a vector in the standard error
        /// \param comm: a communicator
        /// \param v: vector print
        /// \param name: name to prefix the print

        template <typename Comm, typename Vector>
        void print(const Comm &comm, const Vector &v, std::string name) {
            std::cerr << "[" << comm.rank << "] "
                      << " " << name << ":";
            for (const auto &i : v) std::cerr << " " << i;
            std::cerr << std::endl;
            std::cerr.flush();
        }

        /// Return an order with values 0, 1, 2, ..., N-1

        template <std::size_t N> Order<N> trivial_order() {
            Order<N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = (char)(i + 1);
            return r;
        }

        /// Total volume of a list of ranges
        /// \param fs: vector of first coordinate and size of the ranges to translate

        template <std::size_t Nd> std::size_t volume(const From_size<Nd> &fs) {
            std::size_t vol = 0;
            for (const auto &fsi : fs) vol += volume(fsi[1]);
            return vol;
        }

        /// Return coor % dim
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions

        inline IndexType normalize_coor(IndexType coor, IndexType dim) {
            return (dim == 0 ? 0 : (coor + dim * (coor < 0 ? -coor / dim + 1 : 0)) % dim);
        }

        /// Return coor[i] % dim[i]
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions

        template <std::size_t Nd>
        Coor<Nd> normalize_coor(const Coor<Nd> &coor, const Coor<Nd> &dim) {
            Coor<Nd> r;
            for (std::size_t j = 0; j < Nd; j++) r[j] = normalize_coor(coor[j], dim[j]);
            return r;
        }

        /// Return the intersection of two 1D ranges for a NOT toroidal lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param fromr: first coordinate of the resulting range
        /// \param sizer: size of the resulting range

        inline void intersection(IndexType from0, IndexType size0, IndexType from1, IndexType size1,
                                 IndexType dim, IndexType &fromr, IndexType &sizer) {
            fromr = from0 + std::min(std::max(from1 - from0, IndexType{0}), size0);
            sizer = from0 + std::min(std::max(from1 + size1 - from0, IndexType{0}), size0) - fromr;
            fromr = (fromr + dim) % dim;
            if (sizer == dim) fromr = from0;
        }

        /// When intersecting full support intervals, indicates which interval to return

        enum IntersectionDominant { FirstIntervalIsDominant, SecondIntervalIsDominant };

        /// Return the intersection between two ranges in a periodic lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice
        /// \param intersect_dominant: which interval to return when both have full support

        template <std::size_t Nd>
        std::pair<std::array<From_size_item<Nd>, 2>, Coor<Nd>>
        intersection_aux(const Coor<Nd> &from0, const Coor<Nd> &size0, const Coor<Nd> &from1,
                         const Coor<Nd> &size1, const Coor<Nd> &dim,
                         IntersectionDominant intersect_dominant = FirstIntervalIsDominant) {

            std::array<From_size_item<Nd>, 2> grid;
            Coor<Nd> grid_n{};
            for (std::size_t i = 0; i < Nd; ++i) {
                if (size0[i] > dim[i] || size1[i] > dim[i])
                    throw std::runtime_error("intersection_aux: invalid input arguments");

                //
                // Compute the subintervals for the dimension ith
                //
                IndexType fromr0 = 0, sizer0 = 0, fromr1 = 0, sizer1 = 0, fromr2 = 0, sizer2 = 0;

                // Proceed with easy cases: if one of the ranges in the whole lattice
                if (size0[i] == dim[i] && size1[i] == dim[i]) {
                    fromr0 = intersect_dominant == FirstIntervalIsDominant ? from0[i] : from1[i];
                    sizer0 = intersect_dominant == FirstIntervalIsDominant ? size0[i] : size1[i];
                } else if (size1[i] == dim[i]) {
                    fromr0 = from0[i];
                    sizer0 = size0[i];
                } else if (size0[i] == dim[i]) {
                    fromr0 = from1[i];
                    sizer0 = size1[i];
                }
                // Proceed with the general case
                else {
                    intersection(from0[i], size0[i], from1[i], size1[i], dim[i], fromr0, sizer0);
                    intersection(from0[i], size0[i], from1[i] + dim[i], size1[i], dim[i], fromr1,
                                 sizer1);
                    intersection(from0[i] + dim[i], size0[i], from1[i], size1[i], dim[i], fromr2,
                                 sizer2);
                }
                if (sizer0 > 0) {
                    grid[grid_n[i]][0][i] = fromr0;
                    grid[grid_n[i]++][1][i] = sizer0;
                }
                if (sizer1 > 0) {
                    grid[grid_n[i]][0][i] = fromr1;
                    grid[grid_n[i]++][1][i] = sizer1;
                }
                if (sizer2 > 0) {
                    grid[grid_n[i]][0][i] = fromr2;
                    grid[grid_n[i]++][1][i] = sizer2;
                }
            }
            return {grid, grid_n};
        }

        /// Return the intersection between two ranges in a periodic lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice
        /// \param fromr: first coordinate of the first resulting range
        /// \param sizer: size of the first resulting range
        /// \param intersect_dominant: which interval to return when both have full support

        template <std::size_t Nd>
        void intersection(const Coor<Nd> &from0, const Coor<Nd> &size0, const Coor<Nd> &from1,
                          const Coor<Nd> &size1, const Coor<Nd> &dim, Coor<Nd> &fromr,
                          Coor<Nd> &sizer,
                          IntersectionDominant intersect_dominant = FirstIntervalIsDominant) {
            auto p = intersection_aux<Nd>(from0, size0, from1, size1, dim, intersect_dominant);
            std::size_t vol = volume(p.second);
            if (vol == 0) {
                fromr = Coor<Nd>{{}};
                sizer = Coor<Nd>{{}};
            } else if (vol == 1) {
                fromr = p.first[0][0];
                sizer = p.first[0][1];
            } else {
                throw std::runtime_error("Not supported complex overlap of intervals");
            }
        }

        /// Return all ranges resulting from intersecting the two given ranges in a periodic lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice
        /// \param intersect_dominant: which interval to return when both have full support

        template <std::size_t Nd>
        From_size<Nd>
        intersection(const Coor<Nd> &from0, const Coor<Nd> &size0, const Coor<Nd> &from1,
                     const Coor<Nd> &size1, const Coor<Nd> &dim,
                     IntersectionDominant intersect_dominant = FirstIntervalIsDominant) {
            auto p = intersection_aux<Nd>(from0, size0, from1, size1, dim, intersect_dominant);
            IndexType vol = volume(p.second);
            if (vol == 0) {
                return {};
            } else if (vol == 1) {
                From_size<Nd> r(1);
                r[0] = p.first[0];
                return r;
            } else {
                From_size<Nd> r(vol);
                Coor<Nd> stride = get_strides<IndexType>(p.second, FastToSlow);
                for (IndexType i = 0; i < vol; ++i) {
                    Coor<Nd> c = index2coor(i, p.second, stride);
                    for (std::size_t j = 0; j < Nd; ++j) {
                        r[i][0][j] = p.first[c[j]][0][j];
                        r[i][1][j] = p.first[c[j]][1][j];
                    }
                }
                return r;
            }
        }

        /// Return all ranges resulting from intersecting the two given ranges in a periodic lattice
        /// \param fs0: vector of first coordinate and size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice
        /// \param intersect_dominant: which interval to return when both have full support

        template <std::size_t Nd>
        From_size<Nd>
        intersection(const From_size<Nd> &fs0, const Coor<Nd> &from1, const Coor<Nd> &size1,
                     const Coor<Nd> &dim,
                     IntersectionDominant intersect_dominant = FirstIntervalIsDominant) {
            vector<std::pair<std::array<From_size_item<Nd>, 2>, Coor<Nd>>, Cpu> p(fs0.size(),
                                                                                  Cpu{});
            std::size_t vol = 0;
            for (std::size_t i = 0; i < fs0.size(); ++i) {
                p[i] = intersection_aux<Nd>(fs0[i][0], fs0[i][1], from1, size1, dim,
                                            intersect_dominant);
                vol += volume(p[i].second);
            }
            From_size<Nd> r(vol);
            std::size_t ri = 0;
            for (std::size_t i = 0; i < fs0.size(); ++i) {
                Coor<Nd> stride = get_strides<IndexType>(p[i].second, FastToSlow);
                for (IndexType j = 0, j1 = volume(p[i].second); j < j1; ++j) {
                    Coor<Nd> c = index2coor(j, p[i].second, stride);
                    for (std::size_t k = 0; k < Nd; ++k) {
                        r[ri][0][k] = p[i].first[c[k]][0][k];
                        r[ri][1][k] = p[i].first[c[k]][1][k];
                    }
                    ++ri;
                }
            }
            return r;
        }

        /// Return all ranges resulting from intersecting ranges from each list in a periodic lattice
        /// \param fs0: first vector of ranges (first coordinate and size)
        /// \param fs1: second vector of ranges (first coordinate and size)
        /// \param dim: lattice size
        /// \param intersect_dominant: which interval to return when both have full support

        template <std::size_t Nd>
        From_size<Nd>
        intersection(const From_size<Nd> &fs0, const From_size<Nd> fs1, const Coor<Nd> &dim,
                     IntersectionDominant intersect_dominant = FirstIntervalIsDominant) {
            vector<std::pair<std::array<From_size_item<Nd>, 2>, Coor<Nd>>, Cpu> p(
                fs0.size() * fs1.size(), Cpu{});
            std::size_t vol = 0;
            for (std::size_t i = 0; i < fs0.size(); ++i) {
                for (std::size_t j = 0; j < fs1.size(); ++j) {
                    p[j + i * fs1.size()] = intersection_aux<Nd>(
                        fs0[i][0], fs0[i][1], fs1[j][0], fs1[j][1], dim, intersect_dominant);
                    vol += volume(p[j + i * fs1.size()].second);
                }
            }
            From_size<Nd> r(vol);
            std::size_t ri = 0;
            for (std::size_t i = 0; i < p.size(); ++i) {
                Coor<Nd> stride = get_strides<IndexType>(p[i].second, FastToSlow);
                for (IndexType j = 0, j1 = volume(p[i].second); j < j1; ++j) {
                    Coor<Nd> c = index2coor(j, p[i].second, stride);
                    for (std::size_t k = 0; k < Nd; ++k) {
                        r[ri][0][k] = p[i].first[c[k]][0][k];
                        r[ri][1][k] = p[i].first[c[k]][1][k];
                    }
                    ++ri;
                }
            }
            return r;
        }

        /// Shift a list of ranges
        /// \param fs0: vector of first coordinate and size of the ranges to translate
        /// \param from0: origin coordinate on the origin lattice
        /// \param dim0: dimensions of the origin lattice
        /// \param from1: origin coordinate on the destination lattice
        /// \param dim1: dimensions of the destination lattice
        /// \param perm: permutation of the coordinates

        template <std::size_t Nd>
        From_size<Nd> shift_ranges(const From_size<Nd> &fs, const Coor<Nd> &from,
                                   const Coor<Nd> &to, const Coor<Nd> &dim) {
            From_size<Nd> r(fs.size());
            for (std::size_t i = 0; i < fs.size(); ++i) {
                r[i][0] = normalize_coor(fs[i][0] - from + to, dim);
                r[i][1] = fs[i][1];
            }
            return r;
        }

        /// Sort a list of ranges based on the first coordinate
        /// \param fs: vector of first coordinate and size of the ranges to order
        /// \param dim: dimensions of the tensor where the ranges belong
        /// \param stride: strides for those dimensions

        template <std::size_t Nd, typename SIdx>
        From_size<Nd> sort_ranges(const From_size<Nd> &fs, const Coor<Nd> &dim,
                                  const Coor<Nd, SIdx> &stride) {
            From_size<Nd> r(fs.size());
            for (std::size_t i = 0; i < fs.size(); ++i) r[i] = fs[i];
            std::sort(r.begin(), r.end(),
                      [&](const From_size_item<Nd> &a, const From_size_item<Nd> &b) {
                          return coor2index(a[0], dim, stride) < coor2index(b[0], dim, stride);
                      });
            return r;
        }

        /// Translate a range from one coordinate lattice to another
        /// \param fs0: vector of first coordinate and size of the ranges to translate
        /// \param from0: origin coordinate on the origin lattice
        /// \param dim0: dimensions of the origin lattice
        /// \param from1: origin coordinate on the destination lattice
        /// \param dim1: dimensions of the destination lattice

        template <std::size_t Nd>
        From_size<Nd> reorder_coor(const From_size<Nd> &fs, const Coor<Nd> &perm) {
            From_size<Nd> r(fs.size());
            for (std::size_t i = 0; i < fs.size(); ++i) {
                r[i] = {reorder_coor(fs[i][0], perm), reorder_coor(fs[i][1], perm)};
            }
            return r;
        }

        /// Translate a range from one coordinate lattice to another
        /// \param rfrom0: first coordinate of the range to translate
        /// \param rsize0: size of the range to translate
        /// \param from0: origin coordinate on the origin lattice
        /// \param dim0: dimensions of the origin lattice
        /// \param from1: origin coordinate on the destination lattice
        /// \param dim1: dimensions of the destination lattice
        /// \param perm: permutation of the coordinates
        /// \param fromr: first coordinate of input range into the destination lattice
        /// \param sizer: size of the input range on the destination lattice

        template <std::size_t Nd0, std::size_t Nd1>
        void translate_range(const Coor<Nd0> &rfrom0, const Coor<Nd0> &rsize0,
                             const Coor<Nd0> &from0, const Coor<Nd0> &dim0, const Coor<Nd1> &from1,
                             const Coor<Nd1> &dim1, const Coor<Nd1> perm, Coor<Nd1> &fromr,
                             Coor<Nd1> &sizer) {
            fromr = normalize_coor<Nd1>(
                reorder_coor<Nd0, Nd1>(normalize_coor<Nd0>(rfrom0 - from0 + dim0, dim0), perm) +
                    from1,
                dim1);
            sizer = reorder_coor<Nd0, Nd1>(rsize0, perm, 1);
            if (volume(sizer) == 0) sizer = Coor<Nd1>{{}};
        }

        /// Translate a range from one coordinate lattice to another
        /// \param fs0: vector of first coordinate and size of the ranges to translate
        /// \param from0: origin coordinate on the origin lattice
        /// \param dim0: dimensions of the origin lattice
        /// \param from1: origin coordinate on the destination lattice
        /// \param dim1: dimensions of the destination lattice
        /// \param perm: permutation of the coordinates

        template <std::size_t Nd0, std::size_t Nd1>
        From_size<Nd1> translate_range(const From_size<Nd0> &fs0, const Coor<Nd0> &from0,
                                       const Coor<Nd0> &dim0, const Coor<Nd1> &from1,
                                       const Coor<Nd1> &dim1, const Coor<Nd1> perm) {
            From_size<Nd1> r(fs0.size());
            for (std::size_t i = 0; i < fs0.size(); ++i)
                translate_range<Nd0, Nd1>(fs0[i][0], fs0[i][1], from0, dim0, from1, dim1, perm,
                                          r[i][0], r[i][1]);
            return r;
        }

        /// Return whether p1 - p0 is empty
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param dim0: dimension size for the origin tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param o1: dimension labels for the destination tensor

        template <std::size_t Nd0, std::size_t Nd1>
        bool has_full_support(const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
                              const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                              const Proc_ranges<Nd1> &p1, const Coor<Nd1> &from1,
                              const Coor<Nd1> &dim1, const Order<Nd1> &o1) {

            // Compute r0 = (from, size) - p0
            From_size<Nd0> r0(1, {from0, size0});
            From_size<Nd0> aux;
            for (const auto &pi : p0) {
                for (const auto fs_p : pi) {
                    aux.resize(0);
                    for (const auto &fs_r : r0) {
                        auto left = make_hole(fs_r[0], fs_r[1], fs_p[0], fs_p[1], dim0);
                        aux.insert(aux.end(), left.begin(), left.end());
                    }
                    std::swap(r0, aux);
                }
            }

            // Shortcut when the remaining list is empty
            if (volume(r0) == 0) return true;

            // Translate the restricted range to the destination lattice
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            From_size<Nd1> r1 = translate_range(r0, from0, dim0, from1, dim1, perm0);

            for (const auto &pi : p1) {
                for (const auto &fs_p : pi) {
                    if (volume(intersection(r1, fs_p[0], fs_p[1], dim1)) > 0) return false;
                }
            }

            return true;
        }

        /// Throw an error if not all processes give the same value
        /// \param t: value to test
        /// \param comm: communicator
        ///
        /// NOTE: the no MPI version does nothing

        template <typename T, typename H = Hash<T>>
        void check_consistency(const T &, const SelfComm &) {}

        template <std::size_t N, std::size_t Nv, typename T, typename Comm, typename XPU0,
                  typename XPU1>
        void check_components(const Proc_ranges<N> &p, const Components_tmpl<Nv, T, XPU0, XPU1> &v,
                              const Comm &comm) {
            if (p.size() != comm.nprocs || p[comm.rank].size() != v.first.size() + v.second.size())
                throw std::runtime_error("wtf");
        }

        template <std::size_t N, typename Comm>
        void check_components(const Proc_ranges<N> &p, const Comm &comm) {
            if (p.size() != comm.nprocs) throw std::runtime_error("wtf");
        }

#ifdef SUPERBBLAS_USE_MPI
        /// Communication barrier

        inline void barrier(MpiComm comm) { MPI_check(MPI_Barrier(comm.comm)); }

        template <typename T, typename H = Hash<T>>
        void check_consistency(const T &t, const MpiComm &comm) {
            if (getDebugLevel() == 0 || comm.nprocs == 1) return;
            const std::size_t h0 = H::hash(t) + (std::size_t)comm.nprocs;
            std::size_t h = h0;
            MPI_check(MPI_Bcast(&h, sizeof(h) / sizeof(int), MPI_INT, 0, comm.comm));
            if (h0 != h) throw std::runtime_error("check_consistency failed!");
        }

        /// Vectors used in MPI communications
        template <typename T, typename XPUbuff> struct PackedValues {
            vector<T, XPUbuff> buf;     ///< pointer to data
            vector<MpiInt, Cpu> counts; ///< number of items send/receive for rank i
            vector<MpiInt, Cpu> displ;  ///< index of the first element to send/receive for rank i
        };

        /// Allocate buffers and prepare arrays from a list of ranges to be used in a MPI communication
        /// \param toSend: iterator over a list of tensor ranges to be packed
        /// \param comm: communicator

        template <typename T, typename XPUbuff, std::size_t Nd>
        PackedValues<T, XPUbuff> prepare_pack(const Range_proc_range_ranges<Nd> &toSend,
                                              const MpiComm &comm, const XPUbuff &xpu) {

            // Allocate PackedValues
            static_assert(MpiTypeSize % sizeof(T) == 0,
                          "Please change MpiTypeSize to be a power of two!");

            // Prepare counts and displ
            vector<MpiInt, Cpu> counts(comm.nprocs, Cpu{});
            vector<MpiInt, Cpu> displ(comm.nprocs, Cpu{});
            std::size_t n = 0; // accumulate total number of T elements
            int d = 0;         // accumulate total number of MpiT elements
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                std::size_t n_rank = 0;  // total number of T elements in rank
                if (rank != comm.rank) { // Skip the communications of the local rank
                    // Compute the total number of T elements for rank i
                    for (unsigned int irange = 0; irange < toSend.size(); ++irange)
                        for (const auto &ranges : toSend[irange][rank]) n_rank += volume(ranges);
                }
                std::size_t new_size = multiple_of(n_rank * sizeof(T), MpiTypeSize);
                n += new_size / sizeof(T);
                counts[rank] = new_size / MpiTypeSize;
                displ[rank] = d;
                d += counts[rank];
            }
            if (d * MpiTypeSize != n * sizeof(T))
                throw std::runtime_error(
                    "Exceeded the maximum package size: increase `MpiTypeSize`");

            // NOTE: MPI calls may have problems passing null pointers as buffers
            if (n == 0) n = MpiTypeSize / sizeof(T);

            vector<T, XPUbuff> buf(n, xpu, doCacheAllocExternal, MpiTypeSize);

            return PackedValues<T, XPUbuff>{buf, counts, displ};
        }

        /// Return the common blocksize given a list of ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param size0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param ncomponents1: number of components
        /// \param comm: communicator
        /// \param co: coordinate linearization order
        /// \param nblock: (out) the first `nblock` dimensions are equivalent to a trivial permutation
        /// \param blocksize: (out) the volume of the first `nblock` dimensions or one at least

        template <std::size_t Nd0, std::size_t Nd1>
        void get_block_size_for_copy_normalize(
            const Order<Nd0> &o0, const typename Range_proc_range_ranges<Nd0>::value_type &toSend,
            const Coor<Nd0> &size0, const Order<Nd1> &o1, const MpiComm &comm, CoorOrder co,
            // Output
            std::size_t &nblock, std::size_t &blocksize) {

            assert(toSend.size() == comm.nprocs);

            // Quick exit for zero volume
            nblock = 0;
            blocksize = 1;
            if (volume(size0) == 0) return;

            Coor<Nd1> perm0 = find_permutation(o0, o1);
            nblock = std::min(Nd0, Nd1);
            if (co == FastToSlow) {
                for (unsigned int rank = 0; rank < toSend.size(); ++rank) {
                    // Skip the communications of the local rank
                    if (rank == comm.rank) continue;
                    for (auto const &r : toSend[rank]) {
                        for (const auto &fs : r) {
                            if (volume(fs[1]) == 0) continue;
                            std::size_t i = 0;
                            for (std::size_t i1 = 0; i1 < Nd1; ++i1) {
                                superbblas::IndexType i0 = perm0[i1];
                                if (i0 < 0) continue;
                                if ((std::size_t)i0 != i) break;
                                if (i >= nblock) break;
                                if (fs[0][i0] != 0 || fs[1][i0] != size0[i0]) break;
                                ++i;
                            }
                            nblock = i;
                        }
                    }
                }
                for (std::size_t i = 0; i < nblock; ++i) blocksize *= size0[i];
                std::size_t compress_nblock = 0;
                for (std::size_t i = 0; i < nblock; ++i)
                    if (size0[i] > 1) ++compress_nblock;
                nblock = compress_nblock;
            } else {
                for (unsigned int rank = 0; rank < toSend.size(); ++rank) {
                    // Skip the communications of the local rank
                    if (rank == comm.rank) continue;
                    for (auto const &r : toSend[rank]) {
                        for (const auto &fs : r) {

                            if (volume(fs[1]) == 0) continue;
                            std::size_t i = 0;
                            for (int i1 = (int)Nd1 - 1; i1 >= 0; --i1) {
                                superbblas::IndexType i0 = perm0[i1];
                                if (i0 < 0) continue;
                                if ((std::size_t)i0 != Nd0 - i - 1) break;
                                if (i >= nblock) break;
                                if (fs[0][i0] != 0 || fs[1][i0] != size0[i0]) break;
                                ++i;
                            }
                            nblock = i;
                        }
                    }
                }
                for (std::size_t i = 0; i < nblock; ++i) blocksize *= size0[Nd0 - i - 1];
                std::size_t compress_nblock = 0;
                for (std::size_t i = 0; i < nblock; ++i)
                    if (size0[Nd0 - i - 1] > 1) ++compress_nblock;
                nblock = compress_nblock;
            }
        }

        /// Pack a list of subtensors contiguously in memory
        /// \param o0: dimension labels for the origin tensor
        /// \param fs: a From_size iterator
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param comm: communicator
        /// \param co: coordinate linearization order

        template <typename IndexType, std::size_t Nd0, std::size_t Nd1, typename T, typename Q,
                  typename XPU0, typename XPUbuff>
        void pack_component(const Order<Nd0> &o0,
                            const typename Range_proc_range_ranges<Nd0>::value_type &fs,
                            const Coor<Nd0> &dim0, vector<const T, XPU0> v0, Mask<XPU0> mask0,
                            const Order<Nd1> &o1, Indices<Cpu> &disp1, vector<Q, XPUbuff> &v1,
                            MpiComm comm, CoorOrder co) {

            assert(fs.size() == comm.nprocs);

            // Find indices on cache
            using Key = std::tuple<typename Range_proc_range_ranges<Nd0>::value_type, Coor<Nd0>,
                                   PairPerms<Nd0, Nd1>, Indices<Cpu>, int, int, int, CoorOrder>;
            using Value = std::tuple<IndicesT<IndexType, XPU0>, IndicesT<IndexType, XPUbuff>,
                                     size_t, Indices<Cpu>>;
            struct cache_tag {};
            auto cache = getCache<Key, Value, TupleHash<Key>, cache_tag>(v0.ctx());
            Key key{fs,
                    dim0,
                    get_perms(o0, o1),
                    clone(disp1),
                    comm.rank,
                    deviceId(v0.ctx()),
                    deviceId(v1.ctx()),
                    co};
            auto it = mask0.size() == 0 ? cache.find(key) : cache.end();

            // If they are not, compute the permutation vectors
            IndicesT<IndexType, XPU0> indices0_xpu;
            IndicesT<IndexType, XPUbuff> indices1;
            std::size_t blocksize = 1;
            if (it == cache.end()) {
                tracker<XPU0> _t("comp. pack permutation", v0.ctx());

                // Figure out the common blocksize
                std::size_t nblock = 0;
                if (mask0.size() == 0)
                    get_block_size_for_copy_normalize(o0, fs, dim0, o1, comm, co, nblock,
                                                      blocksize);

                // Get the maximum volume of communicated data without the local part
                std::size_t vol = 0;
                for (unsigned int rank = 0; rank < fs.size(); ++rank)
                    if (rank != comm.rank)
                        for (const auto &lranges : fs[rank]) vol += volume(lranges) / blocksize;

                Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
                IndicesT<IndexType, Cpu> indices0{vol, Cpu{}};
                IndicesT<IndexType, Cpu> indices1_cpu{vol, Cpu{}};
                Mask<Cpu> mask0_cpu = makeSure(mask0, Cpu{});
                std::size_t n = 0;
                for (std::size_t rank = 0; rank < fs.size(); ++rank) {
                    // Skip the communications of the local rank
                    if (rank == comm.rank) continue;

                    for (const auto &ranges : fs[rank]) {
                        for (const auto &fsi : ranges) {
                            // Compute the permutation so that the subtensors are packed on the natural
                            // order on the destination; in other words, apply the permutation before
                            // doing the MPI call
                            Coor<Nd0> fromi = fsi[0], sizei = fsi[1];
                            Coor<Nd1> sizei1 = reorder_coor<Nd0, Nd1>(sizei, perm0, 1);
                            auto indices0i = get_permutation_origin<IndexType>(
                                o0, fromi, sizei, dim0, o1, {{}}, sizei1,
                                DontAllowImplicitPermutation, Cpu{}, co, nblock);
                            assert(indices0i.first.size() + n <= vol);
                            IndicesT<IndexType, Cpu> indices0i_mask = indices0i.first;
                            IndexType indices0i_disp = indices0i.second;
                            if (mask0_cpu.size() > 0)
                                indices0i_mask = select(indices0i.first, mask0_cpu, indices0i_disp,
                                                        indices0i_mask);
                            std::transform(indices0i_mask.begin(), indices0i_mask.end(),
                                           indices0.begin() + n,
                                           [=](IndexType d) { return d + indices0i_disp; });

                            auto indices1i = get_permutation_destination<IndexType>(
                                o0, fromi, sizei, dim0, o1, {{}}, sizei1,
                                DontAllowImplicitPermutation, Cpu{}, co, nblock);
                            assert(indices0i.first.size() == indices1i.first.size());
                            IndicesT<IndexType, Cpu> indices1i_mask = indices1i.first;
                            IndexType indices1i_disp = indices1i.second;
                            if (mask0_cpu.size() > 0)
                                indices1i_mask = select(indices0i.first, mask0_cpu, indices0i_disp,
                                                        indices1i_mask);
                            IndexType dispi = disp1[rank] + indices1i_disp;
                            std::transform(indices1i_mask.begin(), indices1i_mask.end(),
                                           indices1_cpu.begin() + n,
                                           [=](IndexType d) { return d + dispi; });

                            disp1[rank] += indices1i_mask.size() * blocksize;
                            n += indices1i_mask.size();
                            assert(n <= vol);
                        }
                    }
                }
                indices0.resize(n);
                indices1_cpu.resize(n);
                indices0_xpu = makeSure(indices0, v0.ctx());
                indices1 = makeSure(indices1_cpu, v1.ctx());

                // The cache trackers consider that all cache entries are on the same device; so just track the
                // indices0_xpu when using gpus
                if (mask0.size() == 0) {
                    std::size_t size =
                        storageSize(indices0_xpu) +
                        (deviceId(v0.ctx()) == deviceId(v1.ctx()) ? storageSize(indices1) : 0ul);
                    cache.insert(key,
                                 Value{archive(indices0_xpu), archive(indices1), blocksize,
                                       archive(clone(disp1))},
                                 size);
                }
            } else {
                indices0_xpu = std::get<0>(it->second.value);
                indices1 = std::get<1>(it->second.value);
                blocksize = std::get<2>(it->second.value);
                const auto new_disp1 = std::get<3>(it->second.value);
                std::copy_n(new_disp1.data(), new_disp1.size(), disp1.data());
            }

            // Do the copy
            tracker<XPUbuff> _t(std::string("local copy from ") + platformToStr(v0.ctx()) +
                                    std::string(" to ") + platformToStr(v1.ctx()),
                                v1.ctx());
            _t.memops = (double)(sizeof(T) + sizeof(Q)) * indices0_xpu.size() * blocksize;
            copy_n_blocking<IndexType, T, Q>(1.0, v0.data(), v0.ctx(), blocksize,
                                             indices0_xpu.begin(), indices0_xpu.ctx(),
                                             indices0_xpu.size(), v1.data(), v1.ctx(),
                                             indices1.begin(), indices1.ctx(), EWOp::Copy{});
        }

        /// Pack a list of ranges to be used in a MPI communication
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param ncomponents0: number of elements in toSend and v
        /// \param v: vector containing the values to send
        /// \param o0: dimension labels for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param comm: communicator
        /// \param co: coordinate linearization order

        template <typename IndexType, typename Q, std::size_t Nd0, std::size_t Nd1, typename T,
                  typename XPU0, typename XPU1, typename XPUbuff>
        PackedValues<Q, XPUbuff> pack(const Range_proc_range_ranges<Nd0> &toSend,
                                      const Components_tmpl<Nd0, const T, XPU0, XPU1> &v,
                                      const Order<Nd0> &o0, const Order<Nd1> &o1, MpiComm comm,
                                      XPUbuff xpu, CoorOrder co) {

            assert(num_components(v) == toSend.size());

            tracker<Cpu> _t("prepare and pack", Cpu{});

            PackedValues<Q, XPUbuff> r = prepare_pack<Q>(toSend, comm, xpu);

            Indices<Cpu> buf_disp(comm.nprocs, Cpu{});
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank)
                buf_disp[rank] = r.displ[rank] * (MpiTypeSize / sizeof(Q));

            for (unsigned int componentId0 = 0; componentId0 < toSend.size(); ++componentId0) {
                for (const Component<Nd0, const T, XPU0> &c : v.first)
                    if (c.componentId == componentId0)
                        pack_component<IndexType>(o0, toSend[componentId0], c.dim, c.it, c.mask_it,
                                                  o1, buf_disp, r.buf, comm, co);
                for (const Component<Nd0, const T, XPU1> &c : v.second)
                    if (c.componentId == componentId0)
                        pack_component<IndexType>(o0, toSend[componentId0], c.dim, c.it, c.mask_it,
                                                  o1, buf_disp, r.buf, comm, co);
            }

            // Update the counts when using mask
            if (v.first.size() > 0 && v.first[0].mask_it.size() > 0)
                for (unsigned int rank = 0; rank < comm.nprocs; ++rank)
                    r.counts[rank] = (buf_disp[rank] * sizeof(Q) + MpiTypeSize - 1) / MpiTypeSize -
                                     r.displ[rank];
            return r;
        }

        ///
        template <typename IndexType, typename XPU> struct IndicesT_tmpl {
            unsigned int componentId;
            IndicesT<IndexType, XPU> it;
        };

        template <typename IndexType, typename XPU0, typename XPU1>
        using Range_IndicesT_tmpl = std::pair<std::vector<IndicesT_tmpl<IndexType, XPU0>>,
                                              std::vector<IndicesT_tmpl<IndexType, XPU1>>>;

        /// Vectors used in MPI communications
        template <typename IndexType, typename T, typename XPUbuff, typename XPU0, typename XPU1>
        struct UnpackedValues : public PackedValues<T, XPUbuff> {
            /// indices of the buffer
            std::vector<IndicesT<IndexType, XPUbuff>> indices_buf;
            /// indices of the destination elements
            Range_IndicesT_tmpl<IndexType, XPU0, XPU1> indices;
            /// number of indices to process at once
            std::vector<IndicesT<IndexType, Cpu>> indices_groups;
            /// blocksize for block copying
            std::vector<std::size_t> blocksize;
            UnpackedValues(const vector<T, XPUbuff> &buf, const vector<MpiInt, Cpu> &counts,
                           const vector<MpiInt, Cpu> &displ,
                           const std::vector<IndicesT<IndexType, XPUbuff>> &indices_buf,
                           const Range_IndicesT_tmpl<IndexType, XPU0, XPU1> &indices,
                           const std::vector<IndicesT<IndexType, Cpu>> &indices_groups,
                           const std::vector<std::size_t> &blocksize)
                : PackedValues<T, XPUbuff>{buf, counts, displ},
                  indices_buf(indices_buf),
                  indices(indices),
                  indices_groups(indices_groups),
                  blocksize(blocksize) {}
        };

        /// Return whether some ranges to receive overlaps
        /// \param toReceive: list of tensor ranges to receive
        /// \param dim: dimensions of the destination tensor
        /// \param comm: communication

        template <std::size_t Nd>
        bool does_self_intersect(const typename Range_proc_range_ranges<Nd>::value_type &toReceive,
                                 const Coor<Nd> &dim, std::size_t myrank) {

            for (std::size_t irank = 0; irank < toReceive.size(); ++irank) {
                if (irank == myrank) continue;
                for (unsigned int icomp = 0; icomp < toReceive[irank].size(); ++icomp) {
                    for (unsigned int irange = 0; irange < toReceive[irank][icomp].size();
                         ++irange) {
                        for (std::size_t jrank = 0; jrank <= irank; ++jrank) {
                            for (unsigned int jcomp = 0, jcomp1 = jrank < irank
                                                                      ? toReceive[jrank].size()
                                                                      : icomp + 1;
                                 jcomp < jcomp1; jcomp++) {
                                for (unsigned int jrange = 0,
                                                  jrange1 = (jrank < irank || jcomp < icomp)
                                                                ? toReceive[jrank][jcomp].size()
                                                                : irange;
                                     jrange < jrange1; jrange++) {
                                    if (volume(intersection(toReceive[irank][icomp][irange][0],
                                                            toReceive[irank][icomp][irange][1],
                                                            toReceive[jrank][jcomp][jrange][0],
                                                            toReceive[jrank][jcomp][jrange][1],
                                                            dim)) > 0)
                                        return true;
                                }
                            }
                        }
                    }
                }
            }
            return false;
        }

        /// Return a copy of the given tensor with an allocation stream suitable to be stored
        /// in cache
        /// \param v: vector to store
        ///
        /// NOTE: the allocation streams are the ones that live forever, while the regular
        /// streams can come from coflow and be destroy anytime.
        /// NOTE: the following implementations support doing `archive(indices)` in `prepare_unpack`

        template <typename T, typename Q> std::pair<T, Q> archive(const std::pair<T, Q> &v) {
            return {archive(v.first), archive(v.second)};
        }

        template <typename IndexType, typename XPU>
        IndicesT_tmpl<IndexType, XPU> archive(const IndicesT_tmpl<IndexType, XPU> &v) {
            return {v.componentId, archive(v.it)};
        }

        /// Allocate buffers for the receiving tensor pieces from a MPI communication
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param xpu: context for the buffer
        /// \param comm: communication
        /// \param co: coordinate linearization order

        template <typename IndexType, std::size_t Nd, typename T, typename XPU0, typename XPU1,
                  typename XPUbuff, typename EWOP>
        UnpackedValues<IndexType, T, XPUbuff, XPU0, XPU1>
        prepare_unpack(const Range_proc_range_ranges<Nd> &toReceive,
                       const Components_tmpl<Nd, T, XPU0, XPU1> &v, XPUbuff xpu,
                       const MpiComm &comm, CoorOrder co, EWOP) {

            assert(toReceive.size() == num_components(v));

            tracker<Cpu> _t("prepare unpack", Cpu{});

            // Find indices on cache
            using Key = std::tuple<Range_proc_range_ranges<Nd>, std::vector<Coor<Nd>>, int, int,
                                   int, std::vector<int>, CoorOrder>;
            using Value =
                std::tuple<vector<MpiInt, Cpu>,                        // counts
                           vector<MpiInt, Cpu>,                        // displ
                           std::vector<IndicesT<IndexType, XPUbuff>>,  // indices for the buffer
                           Range_IndicesT_tmpl<IndexType, XPU0, XPU1>, // indices
                           std::vector<IndicesT<IndexType, Cpu>>, // number of indices to process
                           std::vector<std::size_t>>;             // blocksize
            struct cache_tag {};
            auto cache = getCache<Key, Value, TupleHash<Key>, cache_tag>(xpu);

            std::vector<int> deviceIds(num_components(v));
            for (const auto &it : v.first) deviceIds[it.componentId] = deviceId(it.it.ctx());
            for (const auto &it : v.second) deviceIds[it.componentId] = deviceId(it.it.ctx());
            std::vector<Coor<Nd>> component_dims(toReceive.size());
            for (const auto &it : v.first) component_dims[it.componentId] = it.dim;
            for (const auto &it : v.second) component_dims[it.componentId] = it.dim;
            Key key{toReceive,     component_dims, comm.nprocs, comm.rank,
                    deviceId(xpu), deviceIds,      co};

            bool using_mask = false;
            for (const auto &it : v.first)
                if (it.mask_it.size() > 0) using_mask = true;
            for (const auto &it : v.second)
                if (it.mask_it.size() > 0) using_mask = true;
            auto it = !using_mask ? cache.find(key) : cache.end();

            // If they are not, compute the permutation vectors
            vector<MpiInt, Cpu> counts;
            vector<MpiInt, Cpu> displ;
            std::vector<IndicesT<IndexType, XPUbuff>> indices_buf;
            Range_IndicesT_tmpl<IndexType, XPU0, XPU1> indices;
            std::vector<IndicesT<IndexType, Cpu>> indices_groups;
            std::vector<std::size_t> blocksize;
            if (it == cache.end()) {
                counts = vector<MpiInt, Cpu>(comm.nprocs, Cpu{});
                displ = vector<MpiInt, Cpu>(comm.nprocs, Cpu{});
                std::vector<Mask<Cpu>> masks(toReceive.size());
                for (const auto &it : v.first) masks[it.componentId] = makeSure(it.mask_it, Cpu{});
                for (const auto &it : v.second) masks[it.componentId] = makeSure(it.mask_it, Cpu{});

                // Figure out the common blocksize
                std::vector<std::size_t> nblock(toReceive.size(), 0);
                blocksize = std::vector<std::size_t>(toReceive.size(), 1);
                Order<Nd> o = trivial_order<Nd>();
                for (std::size_t i = 0; i < toReceive.size(); ++i) {
                    if (masks[i].size() == 0)
                        get_block_size_for_copy_normalize(o, toReceive[i], component_dims[i], o,
                                                          comm, co, nblock[i], blocksize[i]);
                }

                // Compute the destination indices and the total number of elements received from each process
                std::vector<std::size_t> num_elems(toReceive.size(), 0);
                for (std::size_t i = 0; i < comm.nprocs; ++i) counts[i] = 0;
                std::vector<std::size_t> n(comm.nprocs);
                std::vector<std::vector<IndicesT<IndexType, Cpu>>> indices0_groups(
                    toReceive.size());
                std::vector<std::vector<IndexType>> disp_bufs(toReceive.size());
                std::size_t disp_buf = 0;
                const std::size_t num_T = MpiTypeSize / sizeof(T);
                for (std::size_t rank = 0; rank < comm.nprocs; ++rank) {
                    if (rank == comm.rank) continue;

                    const std::size_t srcrange1 =
                        (toReceive.size() > 0 ? toReceive[0][rank].size() : 0);
                    for (std::size_t srcrange = 0; srcrange < srcrange1; ++srcrange) {
                        for (std::size_t dstrange = 0; dstrange < toReceive.size(); ++dstrange) {
                            for (const auto &fsi : toReceive[dstrange][rank][srcrange]) {
                                Coor<Nd> fromi = fsi[0], sizei = fsi[1];
                                auto indices1_pair = get_permutation_destination<IndexType>(
                                    o, {{}}, sizei, sizei, o, fromi, component_dims[dstrange],
                                    DontAllowImplicitPermutation, Cpu{}, co, nblock[dstrange]);
                                IndicesT<IndexType, Cpu> indices1 = indices1_pair.first;
                                IndexType disp = indices1_pair.second;

                                // Apply the masks
                                if (masks[dstrange].size() > 0)
                                    indices1 = select(indices1, masks[dstrange], disp, indices1);
                                else
                                    indices1 = clone(indices1);

                                // Apply the displacement
                                std::for_each(indices1.begin(), indices1.end(),
                                              [=](IndexType &d) { d += disp; });

                                // Store the number of permutation and the number of elements
                                n[rank] += indices1.size() * blocksize[dstrange];
                                num_elems[dstrange] += indices1.size();
                                indices0_groups[dstrange].push_back(indices1);
                                disp_bufs[dstrange].push_back(disp_buf);
                                disp_buf += indices1.size() * blocksize[dstrange];
                            }
                        }
                    }

                    disp_buf = multiple_of(disp_buf, num_T);
                }

                // Compute the counts
                for (std::size_t i = 0; i < comm.nprocs; ++i)
                    counts[i] = (n[i] * sizeof(T) + MpiTypeSize - 1) / MpiTypeSize;

                // Compute the displacements
                displ[0] = 0;
                for (std::size_t i = 1; i < comm.nprocs; ++i)
                    displ[i] = displ[i - 1] + counts[i - 1];

                // Create the permutation for the buffer
                indices_buf = std::vector<IndicesT<IndexType, XPUbuff>>(toReceive.size());
                for (unsigned int irange = 0; irange < toReceive.size(); ++irange) {
                    IndicesT<IndexType, Cpu> indices_buf_cpu(num_elems[irange], Cpu{});
                    for (std::size_t i = 0, i_buf = 0; i < indices0_groups[irange].size(); ++i) {
                        std::size_t num_blocks = 0;
                        const IndexType disp_buf = disp_bufs[irange][i];
                        for (IndexType j = 0, j1 = indices0_groups[irange][i].size(); j < j1; ++j)
                            indices_buf_cpu[i_buf++] =
                                disp_buf + (num_blocks++) * blocksize[irange];
                    }
                    indices_buf[irange] = makeSure(indices_buf_cpu, xpu);
                }

                // Concatenate all indices into a single permutation vector
                indices = Range_IndicesT_tmpl<IndexType, XPU0, XPU1>();
                indices.first.resize(v.first.size());
                indices.second.resize(v.second.size());
                for (unsigned int irange = 0; irange < toReceive.size(); ++irange) {
                    IndicesT<IndexType, Cpu> indices_cpu(num_elems[irange], Cpu{});
                    std::size_t i0 = 0;
                    for (const auto &indices : indices0_groups[irange]) {
                        copy_n<IndexType>(indices.data(), Cpu{}, indices.size(),
                                          indices_cpu.data() + i0, Cpu{});
                        i0 += indices.size();
                    }
                    for (unsigned int i = 0; i < v.first.size(); ++i)
                        if (v.first[i].componentId == irange)
                            indices.first[i] = {irange, makeSure(indices_cpu, v.first[i].it.ctx())};
                    for (unsigned int i = 0; i < v.second.size(); ++i)
                        if (v.second[i].componentId == irange)
                            indices.second[i] = {irange,
                                                 makeSure(indices_cpu, v.second[i].it.ctx())};
                }

                // If EWOP is addition and the toReceive ranges intersect, then copy_n may result
                // in undefined behaviour as several threads may add on the same destination element
                indices_groups.resize(toReceive.size());
                for (unsigned int irange = 0; irange < toReceive.size(); ++irange) {
                    if (std::is_same<EWOP, EWOp::Add>::value &&
                        does_self_intersect(toReceive[irange], component_dims[irange], comm.rank)) {
                        IndicesT<IndexType, Cpu> indices_groups_irange(
                            indices0_groups[irange].size(), Cpu{});
                        std::size_t i0 = 0;
                        for (const auto &indices : indices0_groups[irange])
                            indices_groups_irange[i0++] = indices.size();
                        indices_groups[irange] = indices_groups_irange;
                    } else {
                        indices_groups[irange] = IndicesT<IndexType, Cpu>(1, Cpu{});
                        indices_groups[irange][0] = indices_buf[irange].size();
                    }
                }

                if (!using_mask) {
                    std::size_t size = storageSize(indices_buf) + storageSize(indices);
                    cache.insert(key,
                                 Value{archive(counts), archive(displ), archive(indices_buf),
                                       archive(indices), archive(indices_groups), blocksize},
                                 size);
                }
            } else {
                counts = std::get<0>(it->second.value);
                displ = std::get<1>(it->second.value);
                indices_buf = std::get<2>(it->second.value);
                indices = std::get<3>(it->second.value);
                indices_groups = std::get<4>(it->second.value);
                blocksize = std::get<5>(it->second.value);
            }

            std::size_t buf_count = (displ.back() + counts.back()) * (MpiTypeSize / sizeof(T));

            // NOTE: MPI calls may have problems passing null pointers as buffers
            if (buf_count == 0) buf_count = MpiTypeSize / sizeof(T);

            // Allocate the buffer
            vector<T, XPUbuff> buf(buf_count, xpu, doCacheAllocExternal, MpiTypeSize);

            return UnpackedValues<IndexType, T, XPUbuff, XPU0, XPU1>{
                buf, counts, displ, indices_buf, indices, indices_groups, blocksize};
        }

        /// Unpack and copy packed tensors from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to packed tensors

        template <typename IndexType, std::size_t Nd, typename T, typename XPUbuff, typename XPU0,
                  typename XPU1, typename EWOP>
        void unpack(const UnpackedValues<IndexType, T, XPUbuff, XPU0, XPU1> &r,
                    const Components_tmpl<Nd, T, XPU0, XPU1> &v, EWOP,
                    typename elem<T>::type alpha) {

            tracker<XPUbuff> _t(std::string("unpack from ") + platformToStr(r.buf.ctx()),
                                r.buf.ctx());

            // Transfer the buffer to the destination device
            for (unsigned int irange = 0; irange < r.indices_buf.size(); ++irange) {
                for (unsigned int j = 0; j < r.indices.first.size(); ++j) {
                    if (r.indices.first[j].componentId != irange) continue;
                    IndexType disp = 0;
                    for (unsigned int i = 0, i1 = r.indices_groups[irange].size(); i < i1; ++i) {
                        copy_n_blocking<IndexType, T, T>(
                            alpha, r.buf.data(), r.buf.ctx(), r.blocksize[irange],
                            r.indices_buf[irange].data() + disp, r.indices_buf[irange].ctx(),
                            r.indices_groups[irange][i], v.first[j].it.data(), v.first[j].it.ctx(),
                            r.indices.first[j].it.data() + disp, r.indices.first[j].it.ctx(),
                            EWOP{});
                        disp += r.indices_groups[irange][i];
                    }
                    _t.memops += (double)sizeof(T) * 2.0 * r.indices.first[j].it.size() *
                                 r.blocksize[irange];
                }
                for (unsigned int j = 0; j < r.indices.second.size(); ++j) {
                    if (r.indices.second[j].componentId != irange) continue;
                    IndexType disp = 0;
                    for (unsigned int i = 0, i1 = r.indices_groups[irange].size(); i < i1; ++i) {
                        copy_n_blocking<IndexType, T, T>(
                            alpha, r.buf.data(), r.buf.ctx(), r.blocksize[irange],
                            r.indices_buf[irange].data() + disp, r.indices_buf[irange].ctx(),
                            r.indices_groups[irange][i], v.second[j].it.data(),
                            v.second[j].it.ctx(), r.indices.second[j].it.data() + disp,
                            r.indices.second[j].it.ctx(), EWOP{});
                        disp += r.indices_groups[irange][i];
                    }
                    _t.memops += (double)sizeof(T) * 2.0 * r.indices.second[j].it.size() *
                                 r.blocksize[irange];
                }
            }
        }

        /// Return a counter, used by `send_receive`

        inline std::size_t &getSendReceiveCallNumer() {
            static std::size_t call_number = 0;
            return call_number;
        }

        /// Asynchronous sending and receiving
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param xpubuff1: context to hold the mpi receiver buffer
        /// \param v1: destination data
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors

        template <typename IndexType, typename XPUbuff0, typename XPUbuff1, std::size_t Nd0,
                  std::size_t Nd1, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP>
        Request send_receive(const Order<Nd0> &o0, const Range_proc_range_ranges<Nd0> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, XPUbuff0 xpubuff0,
                             const Order<Nd1> &o1, const Range_proc_range_ranges<Nd1> &toReceive,
                             const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, XPUbuff1 xpubuff1,
                             MpiComm comm, EWOP, CoorOrder co, typename elem<T>::type alpha) {

            if (comm.nprocs <= 1) return [] {};

            // Annotate the calls so that the returned lambda can be paired with this call
            std::size_t call_number = ++getSendReceiveCallNumer();

            struct tag_type {}; // For hashing template arguments
            if (getDebugLevel() > 0) {
                check_consistency(std::make_tuple(std::string("send_receive"), call_number, o0, o1,
                                                  co, alpha, typeid(tag_type).hash_code()),
                                  comm);
            }

            tracker<Cpu> _t("packing", Cpu{});

            // Pack v0 and prepare for receiving data from other processes
            PackedValues<Q, XPUbuff0> v0ToSend =
                pack<IndexType, Q>(toSend, v0, o0, o1, comm, xpubuff0, co);
            UnpackedValues<IndexType, Q, XPUbuff1, XPU0, XPU1> v1ToReceive =
                prepare_unpack<IndexType>(toReceive, v1, xpubuff1, comm, co, EWOP{});

            // Do a ton of checking
            static MPI_Datatype dtype = get_mpi_datatype();
            assert(v0ToSend.counts.size() == comm.nprocs);
            assert(v0ToSend.displ.size() == comm.nprocs);
            assert(v1ToReceive.counts.size() == comm.nprocs);
            assert(v1ToReceive.displ.size() == comm.nprocs);
            int dtype_size = 0;
            MPI_check(MPI_Type_size(dtype, &dtype_size));
            (void)dtype_size;
            assert((std::size_t)dtype_size == MpiTypeSize);
            assert((v0ToSend.displ.back() + v0ToSend.counts.back()) * MpiTypeSize <=
                   v0ToSend.buf.size() * sizeof(Q));
            assert((v1ToReceive.displ.back() + v1ToReceive.counts.back()) * MpiTypeSize <=
                   v1ToReceive.buf.size() * sizeof(Q));
            assert(v0ToSend.counts[comm.rank] == 0);
            assert(v1ToReceive.counts[comm.rank] == 0);
            if (getDebugLevel() > 0) {
                // Check that all processes agree in the amount of data to send/receive
                std::vector<int> send_counts(comm.rank == 0 ? comm.nprocs * comm.nprocs : 0);
                MPI_check(MPI_Gather(v0ToSend.counts.data(), comm.nprocs, MPI_INT,
                                     send_counts.data(), comm.nprocs, MPI_INT, 0, comm.comm));
                std::vector<int> recv_counts(comm.rank == 0 ? comm.nprocs * comm.nprocs : 0);
                MPI_check(MPI_Gather(v1ToReceive.counts.data(), comm.nprocs, MPI_INT,
                                     recv_counts.data(), comm.nprocs, MPI_INT, 0, comm.comm));
                if (comm.rank == 0)
                    for (unsigned int i = 0; i < comm.nprocs; ++i)
                        for (unsigned int j = 0; j < comm.nprocs; ++j)
                            if (send_counts[i * comm.nprocs + j] !=
                                recv_counts[j * comm.nprocs + i])
                                throw std::runtime_error(
                                    "send_receive: inconsistent communication pattern");
            }

            // Do the MPI communication
            std::vector<MPI_Request> r;
            const int tag = 0;
            const unsigned int T_num = dtype_size / sizeof(T);
            causalConnectTo(v1ToReceive.buf.ctx(), v0ToSend.buf.ctx());
            sync(v0ToSend.buf.ctx());
            if (deviceId(v0ToSend.buf.ctx()) != deviceId(v1ToReceive.buf.ctx()) ||
                getStream(v0ToSend.buf.ctx()) != getStream(v1ToReceive.buf.ctx()))
                sync(v1ToReceive.buf.ctx());
            _t.stop();
            if (getUseMPINonBlock()) {
                if (getUseAlltoall()) {
                    tracker<Cpu> _t("MPI ialltoall", Cpu{});
                    r.resize(1);
                    MPI_check(MPI_Ialltoallv(v0ToSend.buf.data(), v0ToSend.counts.data(),
                                             v0ToSend.displ.data(), dtype, v1ToReceive.buf.data(),
                                             v1ToReceive.counts.data(), v1ToReceive.displ.data(),
                                             dtype, comm.comm, &r.front()));
                } else {
                    tracker<Cpu> _t("MPI isend_recv", Cpu{});
                    r.reserve(comm.nprocs * 2);
                    for (unsigned int p = 0; p < comm.nprocs; ++p) {
                        if (v1ToReceive.counts[p] == 0) continue;
                        r.push_back(MPI_REQUEST_NULL);
                        MPI_check(MPI_Irecv(v1ToReceive.buf.data() + v1ToReceive.displ[p] * T_num,
                                            v1ToReceive.counts[p], dtype, p, tag, comm.comm,
                                            &r.back()));
                    }
                    for (unsigned int p = 0; p < comm.nprocs; ++p) {
                        if (v0ToSend.counts[p] == 0) continue;
                        r.push_back(MPI_REQUEST_NULL);
                        MPI_check(MPI_Isend(v0ToSend.buf.data() + v0ToSend.displ[p] * T_num,
                                            v0ToSend.counts[p], dtype, p, tag, comm.comm,
                                            &r.back()));
                    }
                }
            } else {
                if (getUseAlltoall()) {
                    tracker<Cpu> _t("MPI alltoall", Cpu{});
                    MPI_check(MPI_Alltoallv(v0ToSend.buf.data(), v0ToSend.counts.data(),
                                            v0ToSend.displ.data(), dtype, v1ToReceive.buf.data(),
                                            v1ToReceive.counts.data(), v1ToReceive.displ.data(),
                                            dtype, comm.comm));
                } else {
                    tracker<Cpu> _t("MPI send_recv", Cpu{});
                    for (unsigned int p = 0; p < comm.nprocs; ++p) {
                        if (v0ToSend.counts[p] == 0 && v1ToReceive.counts[p] == 0) continue;
                        MPI_check(MPI_Sendrecv(
                            v0ToSend.buf.data() + v0ToSend.displ[p] * T_num, v0ToSend.counts[p],
                            dtype, p, tag, v1ToReceive.buf.data() + v1ToReceive.displ[p] * T_num,
                            v1ToReceive.counts[p], dtype, p, tag, comm.comm, MPI_STATUS_IGNORE));
                    }
                }
                if (deviceId(v1ToReceive.buf.ctx()) >= 0) syncLegacyStream(v1ToReceive.buf.ctx());
                unpack(v1ToReceive, v1, EWOP{}, Q(alpha));
                return {};
            }

            // Do this later
            // NOTE: keep `v0ToSend` and `v1ToReceive` around until `MPI_Ialltoallv` is finished
            return [=]() mutable {
                // Make sure that all processes wait for the copy operations in the same order
                if (getDebugLevel() > 0) {
                    check_consistency(std::make_tuple(std::string("wait for send_receive"),
                                                      call_number, typeid(tag_type).hash_code()),
                                      comm);
                }

                // Wait for the MPI communication to finish
                {
                    tracker<Cpu> _t("MPI wait", Cpu{});
                    MPI_check(MPI_Waitall((int)r.size(), r.data(), MPI_STATUS_IGNORE));
                }

                // Clear origin buffer
                v0ToSend.buf.clear();

                // Copy back to v1
                if (deviceId(v1ToReceive.buf.ctx()) >= 0) syncLegacyStream(v1ToReceive.buf.ctx());
                unpack(v1ToReceive, v1, EWOP{}, Q(alpha));
            };
        }
#endif // SUPERBBLAS_USE_MPI

        inline void barrier(SelfComm) {}

        /// Asynchronous sending and receiving; do nothing for `SelfComm` communicator
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors

        template <typename IndexType, typename XPUbuff0, typename XPUbuff1, std::size_t Nd0,
                  std::size_t Nd1, typename T, typename Q, typename XPU0, typename XPU1,
                  typename EWOP>
        Request send_receive(const Order<Nd0> &o0, const Range_proc_range_ranges<Nd0> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, XPUbuff0 xpubuff0,
                             const Order<Nd1> &o1, const Range_proc_range_ranges<Nd1> &toReceive,
                             const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, XPUbuff1 xpubuff1,
                             SelfComm comm, EWOP, CoorOrder co, typename elem<T>::type alpha) {
            (void)o0;
            (void)toSend;
            (void)v0;
            (void)xpubuff0;
            (void)o1;
            (void)toReceive;
            (void)v1;
            (void)xpubuff1;
            (void)co;
            (void)alpha;
            if (comm.nprocs <= 1) return [] {};
            throw std::runtime_error("Unsupported SelfComm with nprocs > 1");
        }

        /// Return the total volume of the ranges
        /// \param ranges: list of tensor ranges

        template <std::size_t Nd> std::size_t volume(const Range_proc_range_ranges<Nd> &ranges) {
            std::size_t vol = 0;
            for (auto const &llranges : ranges)
                for (auto const &lranges : llranges)
                    for (auto const &r : lranges) vol += volume(r);
            return vol;
        }

        /// Return whether some MPI calls support GPU pointers

        inline bool test_support_for_mpi_gpu() {
#ifdef SUPERBBLAS_TEST_MPI_GPU
            static const bool test_mpi_gpu = [] {
#    ifdef SUPERBBLAS_USE_CUDA
                return (bool)MPIX_Query_cuda_support();
#    elif defined(SUPERBBLAS_USE_HIP)
                return (bool)MPIX_Query_rocm_support();
#    else
                return false;
#    endif
            }();
            return test_mpi_gpu;
#else
            return false;
#endif // SUPERBBLAS_TEST_MPI_GPU
        }

        /// Asynchronous sending and receiving; do nothing for `SelfComm` communicator
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param xpubuff0: context to hold the mpi sender buffer
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors
        ///
        /// NOTE: choose size_t as the IndexType in case the local volume is too large

        template <std::size_t Nd0, typename XPUbuff0, typename XPUbuff1, std::size_t Nd1,
                  typename T, typename Q, typename XPU0, typename XPU1, typename Comm,
                  typename EWOp>
        Request
        send_receive_choose_size(const Order<Nd0> &o0, const Range_proc_range_ranges<Nd0> &toSend,
                                 const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                                 XPUbuff0 xpubuff0, const Order<Nd1> &o1,
                                 const Range_proc_range_ranges<Nd1> &toReceive,
                                 const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, XPUbuff1 xpubuff1,
                                 Comm comm, EWOp, CoorOrder co, typename elem<T>::type alpha) {

            bool use_size_t = false;
            const std::size_t max_IndexType = (std::size_t)std::numeric_limits<IndexType>::max();
            for (const auto &c : v0.first)
                if (volume(c.dim) >= max_IndexType) use_size_t = true;
            for (const auto &c : v0.second)
                if (volume(c.dim) >= max_IndexType) use_size_t = true;
            for (const auto &c : v1.first)
                if (volume(c.dim) >= max_IndexType) use_size_t = true;
            for (const auto &c : v1.second)
                if (volume(c.dim) >= max_IndexType) use_size_t = true;
            if (volume(toSend) >= max_IndexType || volume(toReceive) >= max_IndexType)
                use_size_t = true;

            if (!use_size_t) {
                return send_receive<IndexType>(o0, toSend, v0, xpubuff0, o1, toReceive, v1,
                                               xpubuff1, comm, EWOp{}, co, alpha);
            } else {
                return send_receive<std::size_t>(o0, toSend, v0, xpubuff0, o1, toReceive, v1,
                                                 xpubuff1, comm, EWOp{}, co, alpha);
            }
        }

        /// Asynchronous sending and receiving; do nothing for `SelfComm` communicator
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param comm: communication
        /// \param co: coordinate linearization order
        /// \param alpha: factor applied to sending tensors

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1, typename Comm, typename EWOp>
        Request send_receive(const Order<Nd0> &o0, const Range_proc_range_ranges<Nd0> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                             const Order<Nd1> &o1, const Range_proc_range_ranges<Nd1> &toReceive,
                             const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, const Comm &comm, EWOp,
                             CoorOrder co, typename elem<T>::type alpha) {

            // Whether to allow the use of gpu buffers for the sender/receiver buffers
            static const bool use_mpi_gpu = [] {
                if (getUseMPIGpu() == 0) return test_support_for_mpi_gpu();
                return getUseMPIGpu() > 0;
            }();

            bool really_use_mpi_gpu = false;
            if (use_mpi_gpu) {
                // Check if there are gpu components
                for (const auto &it : v0.first)
                    if (deviceId(it.it.ctx()) >= 0 && it.it.size() > 0) really_use_mpi_gpu = true;
                for (const auto &it : v1.first)
                    if (deviceId(it.it.ctx()) >= 0 && it.it.size() > 0) really_use_mpi_gpu = true;
            }

            // Use mpi send/receive buffers on cpu memory
            if (!really_use_mpi_gpu) {
#ifdef SUPERBBLAS_USE_GPU
                if (volume(toSend) * sizeof(T) + volume(toReceive) * sizeof(Q) <=
                    getMaxGpuCacheSize()) {
                    // Make the sender/receiver buffers on host pinned memory to improve the transfer rates copying
                    // data from/to the gpus
                    if (v0.first.size() > 0 && v0.first.front().it.size() > 0) {
                        Gpu gpu0 = v0.first.front().it.ctx().toCpuPinned();
                        if (v1.first.size() > 0 && v1.first.front().it.size() > 0) {
                            return send_receive_choose_size(o0, toSend, v0, gpu0, o1, toReceive, v1,
                                                            v1.first.front().it.ctx().toCpuPinned(),
                                                            comm, EWOp{}, co, alpha);
                        } else {
                            return send_receive_choose_size(o0, toSend, v0, gpu0, o1, toReceive, v1,
                                                            Cpu{}, comm, EWOp{}, co, alpha);
                        }
                    } else if (v1.first.size() > 0 && v1.first.front().it.size() > 0) {
                        return send_receive_choose_size(o0, toSend, v0, Cpu{}, o1, toReceive, v1,
                                                        v1.first.front().it.ctx().toCpuPinned(),
                                                        comm, EWOp{}, co, alpha);
                    }
                }
#endif // SUPERBBLAS_USE_GPU
                return send_receive_choose_size(o0, toSend, v0, Cpu{}, o1, toReceive, v1, Cpu{},
                                                comm, EWOp{}, co, alpha);
            }

            // Use mpi send/receive buffers on gpu memory
            // NOTE: both buffers should be on the same device
            XPU0 gpu;
            bool found_it = false;
            for (const auto &it : v0.first) {
                if (deviceId(it.it.ctx()) >= 0 && it.it.size() > 0) {
                    gpu = it.it.ctx();
                    found_it = true;
                    break;
                }
            }
            if (!found_it) {
                for (const auto &it : v1.first) {
                    if (deviceId(it.it.ctx()) >= 0 && it.it.size() > 0) {
                        gpu = it.it.ctx();
                        found_it = true;
                        break;
                    }
                }
            }
            if (!found_it) throw std::runtime_error("wtf");
            return send_receive_choose_size(o0, toSend, v0, gpu, o1, toReceive, v1, gpu, comm,
                                            EWOp{}, co, alpha);
        }

        /// Return a list of ranges after subtracting a list of holes
        /// \param fs: input ranges
        /// \param holes: input list of holes to subtract
        /// \param dim: space dimension where the ranges are embedded

        template <std::size_t N>
        From_size<N> make_hole(const From_size<N> &fs, const From_size<N> &holes,
                               const Coor<N> &dim) {
            // Shortcut when N == 0
            if (N == 0) return {};

            // Compute r0 = (from, size) - p0
            From_size<N> r = fs;
            From_size<N> aux;
            for (const auto fsi : holes) {
                aux.resize(0);
                for (const auto &fs_r : r) {
                    auto left = superbblas::make_hole(fs_r[0], fs_r[1], fsi[0], fsi[1], dim);
                    aux.insert(aux.end(), left.begin(), left.end());
                }
                std::swap(r, aux);
            }

            return r;
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: dimension labels for the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <std::size_t Nd0, std::size_t Nd1, typename Comm, typename EWOP>
        Range_proc_range_ranges<Nd0>
        get_indices_to_send(Proc_ranges<Nd0> p0, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                            const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                            const Proc_ranges<Nd1> &p1, const Order<Nd1> &o1,
                            const Coor<Nd1> &from1, const Coor<Nd1> &dim1, const Comm &comm, EWOP) {

            tracker<Cpu> _t("comp. tensor overlaps", Cpu{});

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            Coor<Nd1> perm0 = find_permutation(o0, o1);
            Coor<Nd0> perm1 = find_permutation(o1, o0);
            Coor<Nd1> size1 = reorder_coor(size0, perm0, 1); // size in the destination

            Range_proc_range_ranges<Nd0> rr(p0[comm.rank].size());
            for (unsigned int irange = 0; irange < p0[comm.rank].size(); ++irange) {
                // Restrict the local source range to the range from0, size0
                Coor<Nd0> local_from0 = p0[comm.rank][irange][0];
                Coor<Nd0> local_size0 = p0[comm.rank][irange][1];
                From_size<Nd0> rlocal0 = intersection(from0, size0, local_from0, local_size0, dim0,
                                                      FirstIntervalIsDominant);

                // Compute the indices
                Coor<Nd1, std::size_t> stride1 = get_strides<std::size_t>(dim1, FastToSlow);
                std::vector<Proc_ranges<Nd0>> r(p1.size());
                for (unsigned int i = 0; i < p1.size(); ++i) {
                    r[i].resize(p1[i].size());
                    for (unsigned int j = 0; j < p1[i].size(); ++j) {
                        const Coor<Nd1> &local_from1 = p1[i][j][0];
                        const Coor<Nd1> &local_size1 = p1[i][j][1];
                        From_size<Nd1> rlocal1 = intersection(
                            from1, size1, local_from1, local_size1, dim1, FirstIntervalIsDominant);

                        From_size<Nd0> rfs0 =
                            translate_range(rlocal1, from1, dim1, from0, dim0, perm1);

                        // Remove ranges that can be locally copied
                        if (std::is_same<EWOP, EWOp::Copy>::value && i != comm.rank)
                            rfs0 = make_hole(rfs0, p0[i], dim0);

                        r[i][j] = shift_ranges(
                            translate_range(
                                sort_ranges(translate_range(intersection(rfs0, rlocal0, dim0,
                                                                         FirstIntervalIsDominant),
                                                            from0, dim0, from1, dim1, perm0),
                                            dim1, stride1),
                                from1, dim1, from0, dim0, perm1),
                            local_from0, {{}}, dim0);
                    }
                }
                rr[irange] = r;
            }

            return rr;
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: dimension labels for the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <std::size_t Nd0, std::size_t Nd1, typename Comm, typename EWOP>
        Range_proc_range_ranges<Nd1>
        get_indices_to_receive(const Proc_ranges<Nd0> &p0, const Order<Nd0> &o0,
                               const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                               const Coor<Nd0> &dim0, const Proc_ranges<Nd1> &p1,
                               const Order<Nd1> &o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                               const Comm &comm, EWOP) {

            tracker<Cpu> _t("comp. tensor overlaps", Cpu{});

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            Coor<Nd1> perm0 = find_permutation(o0, o1);
            Coor<Nd1> size1 = reorder_coor(size0, perm0, 1); // size in the destination

            Range_proc_range_ranges<Nd1> rr(p1[comm.rank].size());
            for (unsigned int irange = 0; irange < p1[comm.rank].size(); ++irange) {
                // Restrict the local range in v1 to the range from1, size1
                Coor<Nd1> local_from1 = p1[comm.rank][irange][0];
                Coor<Nd1> local_size1 = p1[comm.rank][irange][1];
                From_size<Nd1> rlocal1 = intersection(from1, size1, local_from1, local_size1, dim1,
                                                      FirstIntervalIsDominant);

                // Translate the restricted range to the origin lattice
                Coor<Nd0> perm1 = find_permutation(o1, o0);
                From_size<Nd0> rfs0 = translate_range(rlocal1, from1, dim1, from0, dim0, perm1);

                // Remove ranges that can be locally copied
                const From_size<Nd0> &rfs0_with_holes =
                    (std::is_same<EWOP, EWOp::Copy>::value ? make_hole(rfs0, p0[comm.rank], dim0)
                                                           : rfs0);

                // Compute the indices
                Coor<Nd1, std::size_t> stride1 = get_strides<std::size_t>(dim1, FastToSlow);
                std::vector<Proc_ranges<Nd1>> r(p0.size());
                for (unsigned int i = 0; i < p0.size(); ++i) {
                    r[i].resize(p0[i].size());
                    for (unsigned int j = 0; j < p0[i].size(); ++j) {
                        const Coor<Nd0> &local_from0 = p0[i][j][0];
                        const Coor<Nd0> &local_size0 = p0[i][j][1];
                        From_size<Nd0> rlocal0 = intersection(
                            from0, size0, local_from0, local_size0, dim0, FirstIntervalIsDominant);

                        // Remove ranges that can be locally copied
                        const From_size<Nd0> &this_rfs0 =
                            (std::is_same<EWOP, EWOp::Copy>::value && i != comm.rank
                                 ? rfs0_with_holes
                                 : rfs0);

                        r[i][j] = shift_ranges(
                            sort_ranges(translate_range(intersection(this_rfs0, rlocal0, dim0,
                                                                     FirstIntervalIsDominant),
                                                        from0, dim0, from1, dim1, perm0),
                                        dim1, stride1),
                            local_from1, {{}}, dim1);
                    }
                }
                rr[irange] = r;
            }

            return rr;
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: dimension labels for the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <std::size_t Nd0, std::size_t Nd1, typename Comm, typename EWOP>
        void check_indices_to_receive(const Proc_ranges<Nd0> &p0, const Order<Nd0> &o0,
                                      const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                                      const Coor<Nd0> &dim0, const Proc_ranges<Nd1> &p1,
                                      const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                      const Coor<Nd1> &dim1, const Comm &comm, EWOP,
                                      const Range_proc_range_ranges<Nd1> &r) {

            Coor<Nd1> perm0 = find_permutation(o0, o1);
            Range_proc_range_ranges<Nd1> rr(p1[comm.rank].size());
            for (unsigned int proc = 0; proc < comm.nprocs; ++proc) {
                auto comm0 = comm;
                comm0.rank = proc;
                const auto &rp = get_indices_to_send(p0, o0, from0, size0, dim0, p1, o1, from1,
                                                     dim1, comm0, EWOP{});
                for (unsigned int irange0 = 0; irange0 < p0[proc].size(); ++irange0) {
                    for (unsigned int irange1 = 0; irange1 < p1[comm.rank].size(); ++irange1) {
                        const auto &fs01 =
                            shift_ranges<Nd0>(rp.at(irange0).at(comm.rank).at(irange1), {{}},
                                              p0.at(proc).at(irange0).at(0), dim0);
                        const auto &r01 = translate_range(fs01, from0, dim0, from1, dim1, perm0);
                        bool failed = false;
                        if (r.at(irange1).at(proc).size() == 0) {
                            failed = (volume(r01) > 0);
                        } else {
                            const auto &r01_given =
                                shift_ranges(r.at(irange1).at(proc).at(irange0), {{}},
                                             p1.at(comm.rank).at(irange1).at(0), dim1);
                            failed = (r01 != r01_given);
                        }
                        if (failed) {
                            throw std::runtime_error("failed consistency of get_indices_to_send "
                                                     "and get_indices_to_receive");
                        }
                    }
                }
            }
        }

        /// Check that dim0 and dim1 have the same dimensions
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: first coordinate not to copy from the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor

        template <std::size_t Nd0, std::size_t Nd1>
        bool check_equivalence(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                               const Coor<Nd1> dim1) {

            if (volume(dim0) == 0 && volume(dim1) == 0) return true;
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> new_dim1 = reorder_coor<Nd0, Nd1>(dim0, perm0, 1);
            return new_dim1 == dim1;
        }

        namespace ns_copy_test {
            enum MockFilling { FillWithIndices, FillWithZeros };

            /// Return a vector with the global indices of the elements that contains
            /// \param from: first coordinate of the component
            /// \param size: number of elements to copy in each dimension
            /// \param dim: global dimension size of the tensor
            /// \param co: coordinate linearization order
            /// \param mf: either fill with indices or zeros

            template <std::size_t Nd>
            vector<std::size_t, Cpu> get_mock_components(const Coor<Nd> &from, const Coor<Nd> &size,
                                                         const Coor<Nd> &dim, Cpu cpu, CoorOrder co,
                                                         MockFilling mf) {
                std::size_t vol = volume(size);
                vector<std::size_t, Cpu> r(vol, cpu);

                if (mf == FillWithIndices) {
                    Coor<Nd, std::size_t> local_stride = get_strides<std::size_t>(size, co);
                    Coor<Nd, std::size_t> stride = get_strides<std::size_t>(dim, co);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                    for (std::size_t i = 0; i < vol; ++i)
                        r[i] = coor2index(
                            normalize_coor(index2coor<Nd>(i, size, local_stride) + from, dim), dim,
                            stride);
                } else {
                    zero_n(r.data(), vol, r.ctx());
                }

                return r;
            }

            /// Return a vector with the global indices of the elements that contains
            /// \param from: first coordinate of the component
            /// \param size: number of elements to copy in each dimension
            /// \param dim: global dimension size of the tensor
            /// \param co: coordinate linearization order
            /// \param mf: either fill with indices or zeros

            template <std::size_t Nd, typename XPU,
                      typename std::enable_if<!std::is_same<Cpu, XPU>::value, bool>::type = true>
            vector<std::size_t, XPU> get_mock_components(const Coor<Nd> &from, const Coor<Nd> &size,
                                                         const Coor<Nd> &dim, XPU xpu, CoorOrder co,
                                                         MockFilling mf) {
                std::size_t vol = volume(size);
                vector<std::size_t, XPU> r(vol, xpu);
                vector<std::size_t, Cpu> r_host =
                    get_mock_components(from, size, dim, Cpu{}, co, mf);
                copy_n<std::size_t>(1, r_host.data(), r_host.ctx(), vol, r.data(), r.ctx(),
                                    EWOp::Copy{});
                return r;
            }

            template <typename T>
            using mockIndexType = typename std::conditional<std::is_const<T>::value,
                                                            const std::size_t, std::size_t>::type;

            /// Return a tensor with the same shape as the given one but where each element has its index
            /// \param p0: partitioning of the origin tensor in consecutive ranges
            /// \param o0: dimension labels for the origin tensor
            /// \param from0: first coordinate to copy from the origin tensor
            /// \param size0: number of elements to copy in each dimension
            /// \param v0: data for the origin tensor
            /// \param p1: partitioning of the destination tensor in consecutive ranges
            /// \param o1: dimension labels for the destination tensor
            /// \param dim1: dimension size for the destination tensor
            /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
            /// \param v1: data for the destination tensor

            template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
            Components_tmpl<Nd, mockIndexType<T>, XPU0, XPU1>
            get_mock_components(const From_size<Nd> &p, const Coor<Nd> &dim,
                                const Components_tmpl<Nd, T, XPU0, XPU1> &v, CoorOrder co,
                                MockFilling mf) {
                Components_tmpl<Nd, mockIndexType<T>, XPU0, XPU1> r;
                for (const Component<Nd, T, XPU0> &c : v.first) {
                    r.first.push_back(Component<Nd, std::size_t, XPU0>{
                        get_mock_components(p[c.componentId][0], c.dim, dim, c.it.ctx(), co, mf),
                        c.dim, c.componentId, c.mask_it});
                }
                for (const Component<Nd, T, XPU1> &c : v.second) {
                    r.second.push_back(Component<Nd, std::size_t, XPU1>{
                        get_mock_components(p[c.componentId][0], c.dim, dim, c.it.ctx(), co, mf),
                        c.dim, c.componentId, c.mask_it});
                }
                return r;
            }

            /// Test to copy the content of plural tensor v0 into v1
            /// \param p0: partitioning of the origin tensor in consecutive ranges
            /// \param from0: first coordinate to copy from the origin tensor
            /// \param size0: number of elements to copy in each dimension
            /// \param dim0: number of elements on the origin tensor on each dimension
            /// \param o0: dimension labels for the origin tensor
            /// \param o1: dimension labels for the destination tensor
            /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
            /// \param dim1: dimension size for the destination tensor
            /// \param v: data to check
            /// \param local_from1: first coordinate of the destination tensor
            /// \param co: coordinate linearization order

            template <std::size_t Nd0, std::size_t Nd1, typename XPU, typename EWOP>
            void test_copy_check(const Proc_ranges<Nd0> &p, const Coor<Nd0> &from0,
                                 const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                 const Order<Nd0> &o0, const Coor<Nd1> &from1,
                                 const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                                 const Component<Nd1, std::size_t, XPU> &v,
                                 const Coor<Nd1> &local_from1, EWOP, CoorOrder co) {

                Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
                Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
                Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
                std::size_t vol = volume(v.dim);
                Coor<Nd1, std::size_t> local_stride1 = get_strides<std::size_t>(v.dim, co);
                Coor<Nd0, std::size_t> stride0 = get_strides<std::size_t>(dim0, co);
                vector<std::size_t, Cpu> v_host = makeSure(v.it, Cpu{});
                vector<MaskType, Cpu> m_host = makeSure(v.mask_it, Cpu{});

#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
                for (std::size_t i = 0; i < vol; ++i) {
                    Coor<Nd1> c1 =
                        normalize_coor(index2coor(i, v.dim, local_stride1) + local_from1, dim1);
                    std::size_t true_val = 0;
                    if (is_in_interval(from1, size1, dim1, c1)) {
                        Coor<Nd0> c0 =
                            normalize_coor(reorder_coor(c1 - from1, perm1) + from0, dim0);
                        true_val = coor2index(c0, dim0, stride0);
                        int rep = 0;
                        for (const auto &ranges : p)
                            for (const auto &fs : ranges)
                                if (is_in_interval(fs[0], fs[1], dim0, c0)) ++rep;
                        if (std::is_same<EWOp::Add, EWOP>::value)
                            true_val *= rep;
                        else if (rep == 0)
                            true_val = 0;
                        if (m_host.size() > 0 && m_host[i] == 0) true_val = 0;
                    }
                    if (v_host[i] != true_val)
                        throw std::runtime_error("test_copy_check does not pass!");
                }
            }

            /// Test to copy the content of plural tensor v0 into v1
            /// \param alpha: factor applied to the input tensors
            /// \param p0: partitioning of the origin tensor in consecutive ranges
            /// \param o0: dimension labels for the origin tensor
            /// \param from0: first coordinate to copy from the origin tensor
            /// \param size0: number of elements to copy in each dimension
            /// \param v0: data for the origin tensor
            /// \param p1: partitioning of the destination tensor in consecutive ranges
            /// \param o1: dimension labels for the destination tensor
            /// \param dim1: dimension size for the destination tensor
            /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
            /// \param v1: data for the destination tensor
            /// \param comm: communicator context
            /// \param ewop: either to copy or to add the origin values into the destination values
            /// \param co: coordinate linearization order

            template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                      typename XPU0, typename XPU1, typename EWOP>
            void
            test_copy(typename elem<T>::type, const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
                      const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                      const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                      const Proc_ranges<Nd1> &p1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                      const Order<Nd1> &o1, const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1,
                      Comm comm, EWOP, CoorOrder co) {

                bool trackingTime = getTrackingTime();
                getTrackingTime() = false;

                // Fill the mock input and output tensors
                const Components_tmpl<Nd0, const std::size_t, XPU0, XPU1> v0_ =
                    get_mock_components(p0[comm.rank], dim0, v0, co, FillWithIndices);
                const Components_tmpl<Nd1, std::size_t, XPU0, XPU1> v1_ =
                    get_mock_components(p1[comm.rank], dim1, v1, co, FillWithZeros);

                // Copy the indices
                copy(1, p0, from0, size0, dim0, o0, v0_, p1, from1, dim1, o1, v1_, comm, EWOP{}, co,
                     dontForceLocal, false /* don't do test */);

                // Check that the modified elements on v1_ are what they should be
                for (const Component<Nd1, std::size_t, XPU0> &c : v1_.first) {
                    test_copy_check<Nd0, Nd1>(p0, from0, size0, dim0, o0, from1, dim1, o1, c,
                                              p1[comm.rank][c.componentId][0], EWOP{}, co);
                }
                for (const Component<Nd1, std::size_t, XPU1> &c : v1_.second) {
                    test_copy_check<Nd0, Nd1>(p0, from0, size0, dim0, o0, from1, dim1, o1, c,
                                              p1[comm.rank][c.componentId][0], EWOP{}, co);
                }

                getTrackingTime() = trackingTime;
            }
        }

        /// Return whether the distribution has overlaps with itself
        /// \param p: partitioning of the origin tensor in consecutive ranges
        /// \param from: first coordinate to consider
        /// \param size: number of elements to consider in each dimension

        template <std::size_t Nd>
        bool are_there_repetitions(const From_size<Nd> &p, const Coor<Nd> &from,
                                   const Coor<Nd> &size, const Coor<Nd> &dim) {

            tracker<Cpu> _t("are there repetitions", p.ctx());

            unsigned int nprocs = p.size();
            for (unsigned int i0 = 0; i0 < nprocs; ++i0) {
                // Restrict (from, size) to the p[i0] range
                Coor<Nd> fromi0, sizei0;
                intersection(from, size, p[i0][0], p[i0][1], dim, fromi0, sizei0);
                if (volume(sizei0) == 0) continue;

                // Intersect the range with p[i1] range and return if an overlap exists
                for (unsigned int i1 = i0 + 1; i1 < nprocs; ++i1)
                    if (intersection(p[i1][0], p[i1][1], fromi0, sizei0, dim).size() > 0)
                        return true;
            }

            return false;
        }

        /// Return whether the copy operation may need communications
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param dim0: dimension size for the origin tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param o1: dimension labels for the destination tensor

        template <std::size_t Nd0, std::size_t Nd1, typename EWOP>
        bool may_need_communications(const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
                                     const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                     const Order<Nd0> &o0, const Proc_ranges<Nd1> &p1,
                                     const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                                     const Order<Nd1> &o1, EWOP) {

            assert(p0.size() == p1.size());
            tracker<Cpu> _t("avoid communications", Cpu{});
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1); // size in the destination
            Proc_ranges<Nd1> p1_(p1.size());
            for (unsigned int irank = 0; irank < p1.size(); ++irank)
                p1_[irank] = intersection(p1[irank], from1, size1, dim1);
            if (std::is_same<EWOP, EWOp::Add>::value) {
                for (unsigned int irank = 0; irank < p0.size(); ++irank) {
                    auto fs01 = translate_range(intersection(p0[irank], from0, size0, dim0), from0,
                                                dim0, from1, dim1, perm0);
                    for (unsigned int jrank = 0; jrank < p0.size(); ++jrank) {
                        if (irank == jrank) continue;
                        if (volume(intersection(p1_[jrank], fs01, dim1)) > 0) return true;
                    }
                }
            } else {
                Proc_ranges<Nd0> p0_(p0.size());
                for (unsigned int irank = 0; irank < p0.size(); ++irank)
                    p0_[irank] = translate_range(intersection(p0[irank], from0, size0, dim0), from0,
                                                 dim0, from1, dim1, perm0);
                for (unsigned int jrank = 0; jrank < p1.size(); ++jrank) {
                    // Local components to copy in rank `jrank`
                    auto fsj = intersection(p1_[jrank], p0_[jrank], dim1);
                    for (unsigned int irank = 0; irank < p0.size(); ++irank) {
                        if (irank == jrank) continue;
                        // Ranges copied from irank to jrank
                        auto fsij = intersection(p1_[jrank], p0_[irank], dim1);
                        // Return true if the elements to copy are more than locally copied
                        if (volume(fsij) > volume(intersection(fsij, fsj, dim1))) return true;
                    }
                }
            }
            return false;
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Return the gpu components with a parallel context
        /// \param v: components

        template <std::size_t Nd, typename T>
        Components_tmpl<Nd, T, Gpu, Cpu>
        anabranch_begin(const Components_tmpl<Nd, T, Gpu, Cpu> &v) {
            // Trivial case: do nothing if v have zero or one gpu components
            if (v.first.size() <= 1) return v;

            // Recreate v but with new gpu contexts
            Components_tmpl<Nd, T, Gpu, Cpu> r;
            for (const auto &c : v.first)
                r.first.push_back(c.withNewContext(anabranch_begin(c.it.ctx())));
            r.second = v.second;

            // Return the new v
            return r;
        }
#endif // SUPERBBLAS_USE_GPU

        template <std::size_t Nd, typename T>
        Components_tmpl<Nd, T, Cpu, Cpu>
        anabranch_begin(const Components_tmpl<Nd, T, Cpu, Cpu> &v) {
            // Trivial case: do nothing if v doesn't have gpu contexts
            return v;
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Merge back all operations executed asynchronously on the new contexts
        /// \param v: components

        template <std::size_t Nd, typename T>
        void anabranch_end(const Components_tmpl<Nd, T, Gpu, Cpu> &v) {
            // Trivial case: do nothing if v have zero or one components
            if (v.first.size() <= 1) return;

            // Join back all gpu contexts on v
            for (const auto &c : v.first) anabranch_end(c.it.ctx());
        }
#endif // SUPERBBLAS_USE_GPU

        template <std::size_t Nd, typename T>
        void anabranch_end(const Components_tmpl<Nd, T, Cpu, Cpu> &) {
            // Trivial case: do nothing if v have no gpu components
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param alpha: factor applied to the input tensors
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <std::size_t Nd, typename T, typename Q, typename Comm, typename XPU0,
                  typename XPU1, typename EWOP>
        Request copy_request(typename elem<T>::type alpha, const Proc_ranges<Nd> &p0,
                             const Coor<Nd> &from0, const Coor<Nd> &size0, const Coor<Nd> &dim0,
                             const Order<Nd> &o0,
                             const Components_tmpl<Nd, const T, XPU0, XPU1> &v0,
                             const Proc_ranges<Nd> &p1, const Coor<Nd> &from1, const Coor<Nd> &dim1,
                             const Order<Nd> &o1, const Components_tmpl<Nd, Q, XPU0, XPU1> &v1,
                             Comm comm, EWOP ewop, CoorOrder co, bool do_test) {
            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("copy_request"), alpha, p0, from0,
                                                  size0, dim0, o0, p1, from1, dim1, o1, co, do_test,
                                                  typeid(tag_type).hash_code()),
                                  comm);
            }

            if (getDebugLevel() >= 2 && do_test) {
                ns_copy_test::test_copy(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1, o1,
                                        v1, comm, EWOP{}, co);
            }

            tracker<Cpu> _t("distributed copy", Cpu{});

            check_components(p0, v0, comm);
            check_components(p1, v1, comm);

            // Check the compatibility of the tensors
            if (!check_isomorphic<Nd, Nd>(o0, size0, dim0, o1, dim1))
                throw std::runtime_error("Invalid copy operation");

            constexpr std::size_t Nd0 = Nd;
            constexpr std::size_t Nd1 = Nd;

            Range_proc_range_ranges<Nd0> toSend;
            Range_proc_range_ranges<Nd1> toReceive;
            bool need_comms, zeroout_v1;

            if (std::norm(alpha) != 0) {
                // Find precomputed pieces on cache
                using Key =
                    std::tuple<Proc_ranges<Nd0>, Coor<Nd0>, Coor<Nd0>, Coor<Nd0>, Proc_ranges<Nd1>,
                               Coor<Nd1>, Coor<Nd1>, PairPerms<Nd0, Nd1>, int>;
                struct Value {
                    Range_proc_range_ranges<Nd0> toSend;
                    Range_proc_range_ranges<Nd1> toReceive;
                    bool need_comms;
                    bool zeroout_v1;
                };
                struct cache_tag {};
                auto cache = getCache<Key, Value, TupleHash<Key>, cache_tag>(Cpu{});
                Key key{p0, from0, size0, dim0, p1, from1, dim1, get_perms(o0, o1), comm.rank};
                auto it = cache.find(key);

                // Generate the list of subranges to send and receive
                if (it == cache.end()) {
                    toSend = get_indices_to_send(p0, o0, from0, size0, dim0, p1, o1, from1, dim1,
                                                 comm, EWOP{});
                    toReceive = get_indices_to_receive(p0, o0, from0, size0, dim0, p1, o1, from1,
                                                       dim1, comm, EWOP{});
                    if (getDebugLevel() > 1) {
                        check_indices_to_receive(p0, o0, from0, size0, dim0, p1, o1, from1, dim1,
                                                 comm, EWOP{}, toReceive);
                    }

                    // Check whether communications can be avoided
                    // NOTE: when doing copy, avoid doing copy if the destination pieces can be get from
                    //       the local origin
                    need_comms =
                        (comm.nprocs <= 1 ? false
                                          : may_need_communications(p0, from0, size0, dim0, o0, p1,
                                                                    from1, dim1, o1, EWOP{}));

                    // Check whether the destination tensor should be zero out because the origin
                    // tensor hasn't full support and some elements aren't going to be _touched_ on the
                    // destination tensor
                    zeroout_v1 =
                        (std::is_same<EWOP, EWOp::Copy>::value &&
                         !has_full_support(p0, from0, size0, dim0, o0, p1, from1, dim1, o1));

                    // Save the results
                    cache.insert(key, {toSend, toReceive, need_comms, zeroout_v1}, 0);
                } else {
                    toSend = it->second.value.toSend;
                    toReceive = it->second.value.toReceive;
                    need_comms = it->second.value.need_comms;
                    zeroout_v1 = it->second.value.zeroout_v1;
                }
            } else {
                need_comms = false;
                zeroout_v1 = std::is_same<EWOP, EWOp::Copy>::value;
            }

            // Zero out v1 if needed
            if (zeroout_v1) {
                Coor<Nd1> size1 = reorder_coor(size0, find_permutation(o0, o1), 1);
                for (const Component<Nd1, Q, XPU0> &c1 : v1.first) {
                    const auto &fsi = p1[comm.rank][c1.componentId];
                    const auto tozero =
                        shift_ranges(intersection(fsi[0], fsi[1], from1, size1, dim1), fsi[0],
                                     Coor<Nd1>{{}}, fsi[1]);
                    for (unsigned int i = 0, i1 = tozero.size(); i < i1; ++i) {
                        local_copy<Nd1, Nd1, Q, Q>(Q{0}, o1, tozero[i][0], tozero[i][1], c1.dim,
                                                   vector<const Q, XPU0>(c1.it), c1.mask_it, o1,
                                                   tozero[i][0], c1.dim, c1.it, c1.mask_it,
                                                   EWOp::Copy{}, co);
                    }
                }
                for (const Component<Nd1, Q, XPU1> &c1 : v1.second) {
                    const auto &fsi = p1[comm.rank][c1.componentId];
                    const auto tozero =
                        shift_ranges(intersection(fsi[0], fsi[1], from1, size1, dim1), fsi[0],
                                     Coor<Nd1>{{}}, fsi[1]);
                    for (unsigned int i = 0, i1 = tozero.size(); i < i1; ++i) {
                        local_copy<Nd1, Nd1, Q, Q>(Q{0}, o1, tozero[i][0], tozero[i][1], c1.dim,
                                                   vector<const Q, XPU1>(c1.it), c1.mask_it, o1,
                                                   tozero[i][0], c1.dim, c1.it, c1.mask_it,
                                                   EWOp::Copy{}, co);
                    }
                }
            }
            if (std::norm(alpha) == 0) return Request{};

            // Do the sending and receiving
            Request mpi_req;
            if (need_comms)
                mpi_req = send_receive<Nd0, Nd1>(o0, toSend, v0, o1, toReceive, v1, comm, ewop, co,
                                                 alpha);

            // Do the local copies
            for (const Component<Nd0, const T, XPU0> &c0 : v0.first) {
                for (const Component<Nd1, Q, XPU0> &c1 : v1.first) {
                    const auto &toSend0 = toSend[c0.componentId][comm.rank][c1.componentId];
                    const auto &toReceive0 = toReceive[c1.componentId][comm.rank][c0.componentId];
                    assert(toSend0.size() == toReceive0.size());
                    for (unsigned int i = 0, i1 = toSend0.size(); i < i1; ++i) {
                        local_copy<Nd0, Nd1, T, Q>(alpha, o0, toSend0[i][0], toSend0[i][1], c0.dim,
                                                   c0.it, c0.mask_it, o1, toReceive0[i][0], c1.dim,
                                                   c1.it, c1.mask_it, ewop, co);
                    }
                }
                for (const Component<Nd1, Q, XPU1> &c1 : v1.second) {
                    const auto &toSend0 = toSend[c0.componentId][comm.rank][c1.componentId];
                    const auto &toReceive0 = toReceive[c1.componentId][comm.rank][c0.componentId];
                    assert(toSend0.size() == toReceive0.size());
                    for (unsigned int i = 0, i1 = toSend0.size(); i < i1; ++i) {
                        local_copy<Nd0, Nd1, T, Q>(alpha, o0, toSend0[i][0], toSend0[i][1], c0.dim,
                                                   c0.it, c0.mask_it, o1, toReceive0[i][0], c1.dim,
                                                   c1.it, c1.mask_it, ewop, co);
                    }
                }
            }
            for (const Component<Nd0, const T, XPU1> &c0 : v0.second) {
                for (const Component<Nd1, Q, XPU0> &c1 : v1.first) {
                    const auto &toSend0 = toSend[c0.componentId][comm.rank][c1.componentId];
                    const auto &toReceive0 = toReceive[c1.componentId][comm.rank][c0.componentId];
                    assert(toSend0.size() == toReceive0.size());
                    for (unsigned int i = 0, i1 = toSend0.size(); i < i1; ++i) {
                        local_copy<Nd0, Nd1, T, Q>(alpha, o0, toSend0[i][0], toSend0[i][1], c0.dim,
                                                   c0.it, c0.mask_it, o1, toReceive0[i][0], c1.dim,
                                                   c1.it, c1.mask_it, ewop, co);
                    }
                }
                for (const Component<Nd1, Q, XPU1> &c1 : v1.second) {
                    const auto &toSend0 = toSend[c0.componentId][comm.rank][c1.componentId];
                    const auto &toReceive0 = toReceive[c1.componentId][comm.rank][c0.componentId];
                    assert(toSend0.size() == toReceive0.size());
                    for (unsigned int i = 0, i1 = toSend0.size(); i < i1; ++i) {
                        local_copy<Nd0, Nd1, T, Q>(alpha, o0, toSend0[i][0], toSend0[i][1], c0.dim,
                                                   c0.it, c0.mask_it, o1, toReceive0[i][0], c1.dim,
                                                   c1.it, c1.mask_it, ewop, co);
                    }
                }
            }

            return mpi_req;
        }

        /// Return an empty mask, all levels are free to be used

        inline std::vector<bool> get_labels_mask() {
            return std::vector<bool>((int)std::numeric_limits<char>::max() -
                                     (int)std::numeric_limits<char>::min());
        }

        /// Mark the given labels as used
        /// \param o: labels
        /// \param m: mask

        template <std::size_t Nd> void update_label_mask(const Order<Nd> &o, std::vector<bool> &m) {
            for (char c : o) m[(int)c - (int)std::numeric_limits<char>::min()] = true;
        }

        /// Auxiliary struct used by `dummy_normalize_copy`

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        struct tensor_description {
            Proc_ranges<Nd> p;
            Coor<Nd> from, size, dim;
            Order<Nd> o;
            Components_tmpl<Nd, T, XPU0, XPU1> v;
        };

        /// Return an equivalent tensor but the given `Nd` dimensions
        /// \param p0: partitioning of the tensor in consecutive ranges
        /// \param o0: dimension labels for the tensor
        /// \param from0: first coordinate to copy from the tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the tensor

        template <std::size_t Nd, std::size_t Nd0, typename T, typename XPU0, typename XPU1,
                  typename std::enable_if<(Nd0 < Nd), bool>::type = true>
        tensor_description<Nd, T, XPU0, XPU1>
        dummy_normalize_copy(const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
                             const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                             const Components_tmpl<Nd0, T, XPU0, XPU1> &v0, std::vector<bool> &m) {
            Proc_ranges<Nd> new_p(p0.size());
            for (std::size_t i = 0; i < p0.size(); ++i) {
                new_p[i].resize(p0[i].size());
                for (std::size_t j = 0; j < p0[i].size(); ++j) {
                    std::copy_n(p0[i][j][0].begin(), Nd0, new_p[i][j][0].begin());
                    std::copy_n(p0[i][j][1].begin(), Nd0, new_p[i][j][1].begin());
                    for (std::size_t k = Nd0; k < Nd; ++k) new_p[i][j][0][k] = 0;
                    for (std::size_t k = Nd0; k < Nd; ++k) new_p[i][j][1][k] = 1;
                }
            }
            Coor<Nd> new_from, new_size, new_dim;
            std::copy_n(from0.begin(), Nd0, new_from.begin());
            std::copy_n(size0.begin(), Nd0, new_size.begin());
            std::copy_n(dim0.begin(), Nd0, new_dim.begin());
            for (std::size_t j = Nd0; j < Nd; ++j) new_from[j] = 0;
            for (std::size_t j = Nd0; j < Nd; ++j) new_size[j] = new_dim[j] = 1;
            Order<Nd> new_o;
            std::copy_n(o0.begin(), Nd0, new_o.begin());
            std::size_t j = Nd0;
            for (unsigned int c = (unsigned int)(-std::numeric_limits<char>::min()) + 1u;
                 c < m.size() && j < Nd; ++c) {
                if (!m[c]) {
                    new_o[j++] = (char)((int)c + (int)std::numeric_limits<char>::min());
                    m[c] = true;
                }
            }
            if (j != Nd) throw std::runtime_error("dummy_normalize_copy: run out of labels");

            Components_tmpl<Nd, T, XPU0, XPU1> new_v;
            for (const auto &c0 : v0.first) {
                Coor<Nd> new_dim;
                std::copy_n(c0.dim.begin(), Nd0, new_dim.begin());
                for (std::size_t j = Nd0; j < Nd; ++j) new_dim[j] = 1;
                new_v.first.push_back(
                    Component<Nd, T, XPU0>{c0.it, new_dim, c0.componentId, c0.mask_it});
            }
            for (const auto &c0 : v0.second) {
                Coor<Nd> new_dim;
                std::copy_n(c0.dim.begin(), Nd0, new_dim.begin());
                for (std::size_t j = Nd0; j < Nd; ++j) new_dim[j] = 1;
                new_v.second.push_back(
                    Component<Nd, T, XPU1>{c0.it, new_dim, c0.componentId, c0.mask_it});
            }

            return {new_p, new_from, new_size, new_dim, new_o, new_v};
        }

        template <std::size_t Nd, typename T, typename XPU0, typename XPU1>
        tensor_description<Nd, T, XPU0, XPU1>
        dummy_normalize_copy(const Proc_ranges<Nd> &p0, const Coor<Nd> &from0,
                             const Coor<Nd> &size0, const Coor<Nd> &dim0, const Order<Nd> &o0,
                             const Components_tmpl<Nd, T, XPU0, XPU1> &v0, std::vector<bool> &) {
            return {p0, from0, size0, dim0, o0, v0};
        };

        /// Copy the content of plural tensor v0 into v1
        /// \param alpha: factor applied to the input tensors
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order
        /// \param force_local: whether to avoid communications
        ///
        /// NOTE: this function makes the origin and the destination tensor of the same number of dimensions
        /// to reduce the compilation time

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1, typename EWOP>
        Request copy_request_normalized(
            typename elem<T>::type alpha, const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
            const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
            const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const Proc_ranges<Nd1> &p1,
            const Coor<Nd1> &from1, const Coor<Nd1> &dim1, const Order<Nd1> &o1,
            const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm, EWOP ewop, CoorOrder co,
            ForceLocal force_local = dontForceLocal, bool do_test = true) {

            check_components(p0, v0, comm);
            check_components(p1, v1, comm);

            Proc_ranges<Nd0> new_p0 =
                (force_local == dontForceLocal
                     ? p0
                     : Proc_ranges<Nd0>(p0.begin() + comm.rank, p0.begin() + comm.rank + 1));
            Proc_ranges<Nd1> new_p1 =
                (force_local == dontForceLocal
                     ? p1
                     : Proc_ranges<Nd1>(p1.begin() + comm.rank, p1.begin() + comm.rank + 1));
            auto m = get_labels_mask();
            update_label_mask(o0, m);
            update_label_mask(o1, m);
            constexpr std::size_t Nd = multiple_of(std::max(Nd0, Nd1), (std::size_t)4);
            auto t0 = dummy_normalize_copy<Nd>(new_p0, from0, size0, dim0, o0, v0, m);
            auto t1 = dummy_normalize_copy<Nd>(new_p1, from1, Coor<Nd1>{{}}, dim1, o1, v1, m);
            return force_local == dontForceLocal
                       ? copy_request(alpha, t0.p, t0.from, t0.size, t0.dim, t0.o, t0.v, t1.p,
                                      t1.from, t1.dim, t1.o, t1.v, comm, ewop, co, do_test)
                       : copy_request(alpha, t0.p, t0.from, t0.size, t0.dim, t0.o, t0.v, t1.p,
                                      t1.from, t1.dim, t1.o, t1.v, detail::get_comm(), ewop, co,
                                      do_test);
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param alpha: factor applied to the input tensors
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order
        /// \param force_local: whether to avoid communications

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1, typename EWOp>
        void copy(typename elem<T>::type alpha, const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
                  const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                  const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, const Proc_ranges<Nd1> &p1,
                  const Coor<Nd1> &from1, const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                  const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm, EWOp ewop, CoorOrder co,
                  ForceLocal force_local = dontForceLocal, bool do_test = true) {

            wait(copy_request_normalized(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1, o1,
                                         v1, comm, ewop, co, force_local, do_test));
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param alpha: factor applied to the input tensors
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1>
        Request copy(typename elem<T>::type alpha, const Proc_ranges<Nd0> &p0,
                     const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                     const Order<Nd0> &o0, const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                     const Proc_ranges<Nd1> &p1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                     const Order<Nd1> &o1, const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, Comm comm,
                     CopyAdd copyadd, CoorOrder co) {

            if (getDebugLevel() >= 1) {
                barrier(comm);
                for (const auto &i : v1.first) sync(i.it.ctx());
                for (const auto &i : v1.second) sync(i.it.ctx());
            }

            Request r;
            switch (copyadd) {
            case Copy:
                r = copy_request_normalized(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1,
                                            o1, v1, comm, EWOp::Copy{}, co);
                break;
            case Add:
                r = copy_request_normalized(alpha, p0, from0, size0, dim0, o0, v0, p1, from1, dim1,
                                            o1, v1, comm, EWOp::Add{}, co);
                break;
            }

            if (getDebugLevel() >= 1) {
                for (const auto &i : v1.first) sync(i.it.ctx());
                for (const auto &i : v1.second) sync(i.it.ctx());
                barrier(comm);
            }

            return r;
        }

        /// Return value for the dimensions in o_r matching the given for o0 and o1

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        Coor<Ndo> get_dimensions(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                                 const Coor<Nd1> &dim1, const Order<Ndo> &o_r,
                                 bool report_inconsistencies = true, IndexType missing = 0) {
            std::map<char, IndexType> m;
            for (std::size_t i = 0; i < Nd0; ++i) m[o0[i]] = dim0[i];
            for (std::size_t i = 0; i < Nd1; ++i) {
                auto it = m.find(o1[i]);
                if (it == m.end())
                    m[o1[i]] = dim1[i];
                else if (report_inconsistencies && it->second != dim1[i])
                    throw std::runtime_error("Incompatible distributions for contraction");
            }
            Coor<Ndo> r;
            for (std::size_t i = 0; i < Ndo; ++i)
                r[i] = (report_inconsistencies || m.count(o_r[i]) == 1 ? m.at(o_r[i]) : missing);
            return r;
        }

        /// Return value for the dimensions in o_r matching the given for o0 and o1 and o2

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Nd2, std::size_t Ndo>
        Coor<Ndo> get_dimensions(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                                 const Coor<Nd1> &dim1, const Order<Nd2> &o2, const Coor<Nd2> &dim2,
                                 const Order<Ndo> &o_r) {
            std::map<char, IndexType> m;
            for (std::size_t i = 0; i < Nd2; ++i) m[o2[i]] = dim2[i];
            for (std::size_t i = 0; i < Nd1; ++i) m[o1[i]] = dim1[i];
            for (std::size_t i = 0; i < Nd0; ++i) m[o0[i]] = dim0[i];
            Coor<Ndo> r;
            for (std::size_t i = 0; i < Ndo; ++i) r[i] = m[o_r[i]];
            return r;
        }

        /// Return value for the dimensions in o_r matching the given for o0 and o1

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        From_size_item<Ndo> get_dimensions(const Order<Nd0> &o0, From_size_item<Nd0> fs0,
                                           const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                                           From_size_item<Nd1> fs1, const Order<Ndo> &o_r) {

            for (std::size_t i0 = 0; i0 < Nd0; ++i0) {
                auto s1 = std::find(o1.begin(), o1.end(), o0[i0]);
                if (s1 != o1.end()) {
                    unsigned int i1 = s1 - o1.begin();
                    intersection(fs0[0][i0], fs0[1][i0], fs1[0][i1], fs1[1][i1], dim0[i0],
                                 fs0[0][i0], fs0[1][i0]);
                    fs1[0][i1] = fs0[0][i0];
                    fs1[1][i1] = fs0[1][i0];
                }
            }

            From_size_item<Ndo> fsr;

            for (std::size_t i0 = 0; i0 < Nd0; ++i0) {
                auto sr = std::find(o_r.begin(), o_r.end(), o0[i0]);
                if (sr != o_r.end()) {
                    unsigned int ir = sr - o_r.begin();
                    fsr[0][ir] = fs0[0][i0];
                    fsr[1][ir] = fs0[1][i0];
                }
            }

            for (std::size_t i1 = 0; i1 < Nd1; ++i1) {
                auto sr = std::find(o_r.begin(), o_r.end(), o1[i1]);
                if (sr != o_r.end()) {
                    unsigned int ir = sr - o_r.begin();
                    fsr[0][ir] = fs1[0][i1];
                    fsr[1][ir] = fs1[1][i1];
                }
            }

            return fsr;
        }

        enum ZeroInit { dontZeroInit, doZeroInit };

        /// Return a new components based on a partition taking the contexts from given components
        /// \param p: partitioning
        /// \param v: tensor components
        /// \param comm: communicator
        /// \param cacheAlloc: whether to use cache the allocation
        /// \param zero_init: whether to zeroed the new allocation

        template <typename Q, std::size_t N, std::size_t Nv, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        Components_tmpl<N, Q, XPU0, XPU1> like_this_components_with_type(
            const Proc_ranges<N> &p, const Components_tmpl<Nv, T, XPU0, XPU1> &v, Comm comm,
            CacheAlloc cacheAlloc = dontCacheAlloc, ZeroInit zero_init = dontZeroInit) {

            check_components(p, v, comm);

            // Allocate the tensor
            Components_tmpl<N, Q, XPU0, XPU1> v1;
            for (unsigned int i = 0; i < v.first.size(); ++i) {
                const Coor<N> &dimi = p[comm.rank][v.first[i].componentId][1];
                vector<Q, XPU0> v1i(volume(dimi), v.first[i].it.ctx(), cacheAlloc);
                if (zero_init == doZeroInit) zero_n(v1i.data(), v1i.size(), v1i.ctx());
                v1.first.push_back(
                    Component<N, Q, XPU0>{v1i, dimi, v.first[i].componentId, Mask<XPU0>{}});
            }
            for (unsigned int i = 0; i < v.second.size(); ++i) {
                const Coor<N> &dimi = p[comm.rank][v.second[i].componentId][1];
                vector<Q, XPU1> v1i(volume(dimi), v.second[i].it.ctx(), cacheAlloc);
                if (zero_init == doZeroInit) zero_n(v1i.data(), v1i.size(), v1i.ctx());
                v1.second.push_back(
                    Component<N, Q, XPU1>{v1i, dimi, v.second[i].componentId, Mask<XPU1>{}});
            }

            return v1;
        }

        template <std::size_t N, std::size_t Nv, typename T, typename Comm, typename XPU0,
                  typename XPU1>
        Components_tmpl<N, T, XPU0, XPU1>
        like_this_components(const Proc_ranges<N> &p, const Components_tmpl<Nv, T, XPU0, XPU1> &v,
                             Comm comm, CacheAlloc cacheAlloc = dontCacheAlloc,
                             ZeroInit zero_init = dontZeroInit) {
            return like_this_components_with_type<T>(p, v, comm, cacheAlloc, zero_init);
        }

        /// Return a new components based on a partition selecting the context from the component
        /// with more overlap over the given components
        /// \param p: partitioning
        /// \param from: first element to consider
        /// \param dim: dimensions of the tensor
        /// \param v: tensor components
        /// \param p1: new partitioning
        /// \param comm: communicator
        /// \param cacheAlloc: whether to use cache the allocation
        /// \param zero_init: whether to zeroed the new allocation

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        Components_tmpl<N, T, XPU0, XPU1>
        like_this_components(const Proc_ranges<N> &p, const Order<N> &o0, const Coor<N> &from,
                             const Coor<N> &dim, const Components_tmpl<N, T, XPU0, XPU1> &v,
                             const Proc_ranges<N> &p1, const Order<N> &o1, Comm comm,
                             CacheAlloc cacheAlloc = dontCacheAlloc,
                             ZeroInit zero_init = dontZeroInit) {

            check_components(p, v, comm);
            check_components(p1, comm);

            // Deciding the device for each new component
            // NOTE: maximize the overlap with the original devices
            Coor<N> perm1 = find_permutation(o1, o0);
            std::vector<unsigned int> device(p1[comm.rank].size());
            for (unsigned int i = 0; i < p1[comm.rank].size(); ++i) {
                std::size_t max_vol = 0;
                unsigned int max_idx = 0;
                for (unsigned int j = 0; j < p[comm.rank].size(); ++j) {
                    std::size_t vol = volume(
                        intersection(normalize_coor(p[comm.rank][j][0] + from, dim),
                                     p[comm.rank][j][1], reorder_coor(p1[comm.rank][i][0], perm1),
                                     reorder_coor(p1[comm.rank][i][1], perm1), dim));
                    if (vol > max_vol) {
                        max_vol = vol;
                        max_idx = j;
                    }
                }
                device[i] = max_idx;
            }

            // Allocate the tensor
            Components_tmpl<N, T, XPU0, XPU1> v1;
            for (unsigned int i = 0; i < p1[comm.rank].size(); ++i) {
                for (unsigned int j = 0; j < v.first.size(); ++j) {
                    if (v.first[j].componentId != device[i]) continue;
                    const Coor<N> &dimi = p1[comm.rank][i][1];
                    vector<T, XPU0> v1i(volume(dimi), v.first[j].it.ctx(), cacheAlloc);
                    if (zero_init == doZeroInit) zero_n(v1i.data(), v1i.size(), v1i.ctx());
                    v1.first.push_back(Component<N, T, XPU0>{v1i, dimi, i, Mask<XPU0>{}});
                }
                for (unsigned int j = 0; j < v.second.size(); ++j) {
                    if (v.second[j].componentId != device[i]) continue;
                    const Coor<N> &dimi = p1[comm.rank][i][1];
                    vector<T, XPU1> v1i(volume(dimi), v.second[i].it.ctx(), cacheAlloc);
                    if (zero_init == doZeroInit) zero_n(v1i.data(), v1i.size(), v1i.ctx());
                    v1.second.push_back(Component<N, T, XPU1>{v1i, dimi, i, Mask<XPU1>{}});
                }
            }

            return v1;
        }

        /// Return a tensor with a given partitioning and ordering
        /// \param p0: partitioning of the input tensor
        /// \param o0: dimension labels for the input tensor
        /// \param v0: input tensor components
        /// \param p1: partitioning of the output tensor in consecutive ranges
        /// \param o1: dimension labels for the output tensor
        /// \param co: coordinate linearization order
        /// \param force_copy: whether to NOT avoid copy if the partition is the same
        /// \param cacheAlloc: whether to cache the allocation
        /// \param zero_init: whether to zeroed the new allocation

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1,
                  std::size_t N1, typename Q>
        std::pair<Components_tmpl<N, T, XPU0, XPU1>, Request> reorder_tensor_request(
            const Proc_ranges<N> &p0, const Order<N> &o0, const Coor<N> &from0,
            const Coor<N> &size0, const Coor<N> &dim0, const Components_tmpl<N, T, XPU0, XPU1> &v0,
            const Proc_ranges<N> &p1, const Coor<N> &dim1, const Order<N> &o1,
            const Components_tmpl<N1, Q, XPU0, XPU1> &v1_sample, Comm comm, CoorOrder co,
            bool force_copy = false, CacheAlloc cacheAlloc = dontCacheAlloc,
            ForceLocal force_local = dontForceLocal, ZeroInit zero_init = dontZeroInit) {

            // If the two orderings and partitions are equal, return the tensor
            if (!force_copy && from0 == Coor<N>{{}} && o0 == o1 && p0 == p1 &&
                check_components_compatibility(v0, v1_sample))
                return {v0, Request{}};

            // Allocate the tensor
            auto v1 = like_this_components(p1, v1_sample, comm, cacheAlloc, zero_init);

            // Copy the content of v0 into v1
            return {v1, copy_request_normalized<N, N, T>(T{1}, p0, from0, size0, dim0, o0,
                                                         toConst(v0), p1, {{}}, dim1, o1, v1, comm,
                                                         EWOp::Copy{}, co, force_local)};
        }

        /// Return a tensor with a given partitioning and ordering
        /// \param p0: partitioning of the input tensor
        /// \param o0: dimension labels for the input tensor
        /// \param v0: input tensor components
        /// \param p1: partitioning of the output tensor in consecutive ranges
        /// \param o1: dimension labels for the output tensor
        /// \param co: coordinate linearization order
        /// \param force_copy: whether to NOT avoid copy if the partition is the same
        /// \param cacheAlloc: whether to cache the allocation
        /// \param zero_init: whether to zeroed the new allocation

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1,
                  std::size_t N1, typename Q>
        Components_tmpl<N, T, XPU0, XPU1>
        reorder_tensor(const Proc_ranges<N> &p0, const Order<N> &o0, const Coor<N> &from0,
                       const Coor<N> &size0, const Coor<N> &dim0,
                       const Components_tmpl<N, T, XPU0, XPU1> &v0, const Proc_ranges<N> &p1,
                       const Coor<N> &dim1, const Order<N> &o1,
                       const Components_tmpl<N1, Q, XPU0, XPU1> &v1_sample, Comm comm, CoorOrder co,
                       bool force_copy = false, CacheAlloc cacheAlloc = dontCacheAlloc,
                       ForceLocal force_local = dontForceLocal, ZeroInit zero_init = dontZeroInit) {

            const auto t =
                reorder_tensor_request(p0, o0, from0, size0, dim0, v0, p1, dim1, o1, v1_sample,
                                       comm, co, force_copy, cacheAlloc, force_local, zero_init);
            wait(t.second);
            return t.first;
        }

        /// Return a tensor with a given partitioning and ordering
        /// \param p0: partitioning of the input tensor
        /// \param o0: dimension labels for the input tensor
        /// \param v0: input tensor components
        /// \param p1: partitioning of the output tensor in consecutive ranges
        /// \param o1: dimension labels for the output tensor
        /// \param co: coordinate linearization order
        /// \param force_copy: whether to NOT avoid copy if the partition is the same
        /// \param cacheAlloc: whether to cache the allocation
        /// \param force_local: whether to avoid communications
        /// \param zero_init: whether to zeroed the new allocation

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        std::pair<Components_tmpl<N, T, XPU0, XPU1>, Request> reorder_tensor_request(
            const Proc_ranges<N> &p0, const Order<N> &o0, const Coor<N> &from0,
            const Coor<N> &size0, const Coor<N> &dim0, const Components_tmpl<N, T, XPU0, XPU1> &v0,
            const Proc_ranges<N> &p1, const Coor<N> &dim1, const Order<N> &o1, Comm comm,
            CoorOrder co, bool force_copy = false, CacheAlloc cacheAlloc = dontCacheAlloc,
            ForceLocal force_local = dontForceLocal, ZeroInit zero_init = dontZeroInit) {

            // If the two orderings and partitions are equal, return the tensor
            if (!force_copy && from0 == Coor<N>{{}} && o0 == o1 && p0 == p1) return {v0, Request()};

            // Allocate the tensor
            auto v1 =
                like_this_components(p0, o0, from0, dim0, v0, p1, o1, comm, cacheAlloc, zero_init);

            // Copy the content of v0 into v1
            return {v1, copy_request_normalized<N, N, T>(T{1}, p0, from0, size0, dim0, o0,
                                                         toConst(v0), p1, {{}}, dim1, o1, v1, comm,
                                                         EWOp::Copy{}, co, force_local)};
        }

        /// Return a tensor with a given partitioning and ordering
        /// \param p0: partitioning of the input tensor
        /// \param o0: dimension labels for the input tensor
        /// \param v0: input tensor components
        /// \param p1: partitioning of the output tensor in consecutive ranges
        /// \param o1: dimension labels for the output tensor
        /// \param co: coordinate linearization order
        /// \param force_copy: whether to NOT avoid copy if the partition is the same
        /// \param cacheAlloc: whether to cache the allocation
        /// \param force_local: whether to avoid communications
        /// \param zero_init: whether to zeroed the new allocation

        template <std::size_t N, typename T, typename Comm, typename XPU0, typename XPU1>
        Components_tmpl<N, T, XPU0, XPU1>
        reorder_tensor(const Proc_ranges<N> &p0, const Order<N> &o0, const Coor<N> &from0,
                       const Coor<N> &size0, const Coor<N> &dim0,
                       const Components_tmpl<N, T, XPU0, XPU1> &v0, const Proc_ranges<N> &p1,
                       const Coor<N> &dim1, const Order<N> &o1, Comm comm, CoorOrder co,
                       bool force_copy = false, CacheAlloc cacheAlloc = dontCacheAlloc,
                       ForceLocal force_local = dontForceLocal, ZeroInit zero_init = dontZeroInit) {

            const auto t =
                reorder_tensor_request(p0, o0, from0, size0, dim0, v0, p1, dim1, o1, comm, co,
                                       force_copy, cacheAlloc, force_local, zero_init);
            wait(t.second);
            return t.first;
        }

        /// Return the component for a different partitioning
        /// \param p: partitioning of the input tensor
        /// \param v: input tensor components
        /// \param comm: communications

        template <std::size_t Nr, typename T, typename Comm, typename XPU0, typename XPU1,
                  std::size_t N>
        Components_tmpl<Nr, T, XPU0, XPU1>
        reshape(const Proc_ranges<Nr> &p, const Components_tmpl<N, T, XPU0, XPU1> &v, Comm comm) {
            Components_tmpl<Nr, T, XPU0, XPU1> r;
            r.first.reserve(v.first.size());
            r.second.reserve(v.second.size());
            for (unsigned int i = 0; i < v.first.size(); ++i) {
                const unsigned int componentId = v.first[i].componentId;
                if (volume(p[comm.rank][componentId][1]) != volume(v.first[i].dim))
                    throw std::runtime_error("wtf");
                r.first.push_back(Component<Nr, T, XPU0>{
                    v.first[i].it, p[comm.rank][componentId][1], componentId, v.first[i].mask_it});
            }
            for (unsigned int i = 0; i < v.second.size(); ++i) {
                const unsigned int componentId = v.second[i].componentId;
                if (volume(p[comm.rank][componentId][1]) != volume(v.second[i].dim))
                    throw std::runtime_error("wtf");
                r.second.push_back(Component<Nr, T, XPU1>{v.second[i].it,
                                                          p[comm.rank][componentId][1], componentId,
                                                          v.second[i].mask_it});
            }

            return r;
        }

        /// Check that the given components are compatible
        /// \param v0: components to test
        /// \param v1: components to test

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1>
        bool check_components_compatibility(const Components_tmpl<Nd0, T, XPU0, XPU1> &v0,
                                            const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1) {

            // Check that v0 and v1 have the same components and on the same device
            if (v0.first.size() != v1.first.size() || v0.second.size() != v1.second.size())
                return false;
            bool unmatch_dev = false;
            for (unsigned int i = 0; i < v0.first.size(); ++i)
                if (deviceId(v0.first[i].it.ctx()) != deviceId(v1.first[i].it.ctx()))
                    unmatch_dev = true;
            for (unsigned int i = 0; i < v0.second.size(); ++i)
                if (deviceId(v0.second[i].it.ctx()) != deviceId(v1.second[i].it.ctx()))
                    unmatch_dev = true;
            if (unmatch_dev) return false;

            return true;
        }

        // Remove self-intersections
        /// \param p: partitioning
        /// \param dim: dimensions

        template <std::size_t Nd>
        Proc_ranges<Nd> remove_repetitions(const Proc_ranges<Nd> &p, const Coor<Nd> &dim) {
            Proc_ranges<Nd> r(p.size());
            for (unsigned int i = 0; i < p.size(); ++i) {
                for (unsigned int i0 = 0; i0 < p[i].size(); ++i0) {
                    From_size<Nd> fs(1, p[i][i0]);
                    for (unsigned int j = 0; j <= i; ++j) {
                        for (unsigned int j0 = 0, j1 = (j < i ? p[j].size() : i0); j0 < j1; ++j0) {
                            From_size<Nd> fsr;
                            for (unsigned int fsi = 0; fsi < fs.size(); ++fsi) {
                                if (volume(intersection(fs[fsi][0], fs[fsi][1], p[j][j0][0],
                                                        p[j][j0][1], dim)) == 0) {
                                    fsr.push_back(fs[fsi]);
                                } else {
                                    // make hole
                                    auto new_fs = superbblas::make_hole(
                                        fs[fsi][0], fs[fsi][1], p[j][j0][0], p[j][j0][1], dim);
                                    for (const auto &fsi_ : new_fs) fsr.push_back(fsi_);
                                }
                            }
                            fs = fsr;
                        }
                    }
                    for (const auto &fsi : fs) r[i].push_back(fsi);
                }
            }

            return r;
        }

        /// Return partitions for the input tensors that are compatible for contraction
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param sug_o0: suggested dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param sug_o1: suggested dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Nd2, std::size_t Ndo>
        std::tuple<Proc_ranges<Nd0>, Proc_ranges<Nd1>, Proc_ranges<Ndo>>
        get_partitions_for_contraction(const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
                                       const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                       const Order<Nd0> &o0, const Order<Nd0> &sug_o0,
                                       const Proc_ranges<Nd1> &p1, const Coor<Nd1> &from1,
                                       const Coor<Nd1> &size1, const Coor<Nd1> &dim1,
                                       const Order<Nd1> &o1, const Order<Nd1> &sug_o1,
                                       const Coor<Nd2> &dim2, const Order<Nd2> &o2,
                                       const Order<Ndo> &o_r) {

            // Normalize the first tensor as the larger of the two in volume
            if (volume(size0) < volume(size1)) {
                const auto &p10 =
                    get_partitions_for_contraction(p1, from1, size1, dim1, o1, sug_o1, p0, from0,
                                                   size0, dim0, o0, sug_o0, dim2, o2, o_r);
                return {std::get<1>(p10), std::get<0>(p10), std::get<2>(p10)};
            }

            using Key = std::tuple<Proc_ranges<Nd0>, Coor<Nd0>, Coor<Nd0>, Coor<Nd0>, Coor<Nd0>, //
                                   Proc_ranges<Nd1>, Coor<Nd1>, Coor<Nd1>, Coor<Nd1>, Coor<Nd1>, //
                                   Coor<Nd2>, PairPerms<Nd2, Ndo>, PairPerms<Nd0, Nd1>,
                                   PairPerms<Nd1, Nd2>, PairPerms<Nd0, Nd2>>;
            using Value = std::tuple<Proc_ranges<Nd0>, Proc_ranges<Nd1>, Proc_ranges<Ndo>>;
            struct cache_tag {};
            auto cache = getCache<Key, Value, TupleHash<Key>, cache_tag>(Cpu{});
            Key key{p0,
                    from0,
                    size0,
                    dim0,
                    find_permutation(o0, sug_o0), //
                    p1,
                    from1,
                    size1,
                    dim1,
                    find_permutation(o1, sug_o1), //
                    dim2,
                    get_perms(o2, o_r), //
                    get_perms(o0, o1),
                    get_perms(o1, o2),
                    get_perms(o0, o2)};
            auto it = cache.find(key);
            if (it != cache.end()) { return it->second.value; }

            // Reorder the first tensor if needed
            Proc_ranges<Nd0> p0_ = remove_repetitions(p0, dim0);
            Proc_ranges<Nd0> p0r(p0.size());
            Coor<Nd0> perm0 = find_permutation(o0, sug_o0);
            for (unsigned int i = 0; i < p0_.size(); ++i) {
                p0r[i] = reorder_coor(
                    shift_ranges(intersection(p0_[i], from0, size0, dim0), from0, {{}}, dim0),
                    perm0);
            }

            // Change the second partition by using the same distribution as the first tensor
            // for the shared labels and replicated for the remaining labels
            Proc_ranges<Nd1> p1r(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                p1r[i].resize(p0r[i].size());
                for (unsigned int j = 0; j < p0r[i].size(); ++j) {
                    p1r[i][j][0] = get_dimensions(sug_o0, p0r[i][j][0], o1, {{}}, sug_o1, false);
                    p1r[i][j][1] = get_dimensions(sug_o0, p0r[i][j][1], o1, size1, sug_o1, false);
                }
            }

            Proc_ranges<Ndo> pr(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                pr[i].resize(p0r[i].size());
                for (unsigned int j = 0; j < p0r[i].size(); ++j) {
                    pr[i][j][0] = get_dimensions(sug_o0, p0r[i][j][0], o1, {{}}, o2, {{}}, o_r);
                    pr[i][j][1] = get_dimensions(sug_o0, p0r[i][j][1], o1, size1, o2, dim2, o_r);
                }
            }

            // Return
            auto r = std::make_tuple(p0r, p1r, pr);
            cache.insert(key, r, 0);
            return r;
        }

        /// Return whether some ranges to receive overlaps
        /// \param p: partition
        /// \param dim: dimensions of the destination tensor

        template <std::size_t Nd>
        bool does_proc_ranges_self_intersect(const Proc_ranges<Nd> &p, const Coor<Nd> &dim) {

            for (unsigned int pi = 0; pi < p.size(); ++pi) {
                for (unsigned int fsi = 0; fsi < p[pi].size(); ++fsi) {
                    for (unsigned int pj = pi; pj < p.size(); ++pj) {
                        for (unsigned int fsj = pi == pj ? fsi + 1 : 0; fsj < p[pj].size(); ++fsj) {
                            if (volume(intersection(p[pi][fsi][0], p[pi][fsi][1], //
                                                    p[pj][fsj][0], p[pj][fsj][1], dim)) > 0)
                                return true;
                        }
                    }
                }
            }

            return false;
        }

        template <std::size_t Nd, typename T, typename Comm, typename XPU0, typename XPU1>
        Request
        contraction_normalized(T alpha, const Proc_ranges<Nd> &p0, const Coor<Nd> &from0,
                               const Coor<Nd> &size0, const Coor<Nd> &dim0, const Order<Nd> &o0,
                               bool conj0, const Components_tmpl<Nd, T, XPU0, XPU1> &v0,
                               const std::size_t &Nd0, const Proc_ranges<Nd> &p1,
                               const Coor<Nd> &from1, const Coor<Nd> &size1, const Coor<Nd> &dim1,
                               const Order<Nd> &o1, bool conj1,
                               const Components_tmpl<Nd, T, XPU0, XPU1> &v1, const std::size_t Nd1,
                               T beta, const Proc_ranges<Nd> &pr, const Coor<Nd> &fromr,
                               const Coor<Nd> &sizer, const Coor<Nd> &dimr, const Order<Nd> &o_r,
                               const Components_tmpl<Nd, T, XPU0, XPU1> &vr, const std::size_t Ndo,
                               const Comm &comm, CoorOrder co) {
            if (getDebugLevel() >= 1) {
                for (const auto &i : vr.first) sync(i.it.ctx());
                for (const auto &i : vr.second) sync(i.it.ctx());
                barrier(comm);
            }

            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("contraction"), alpha, p0, from0,
                                                  size0, dim0, o0, conj0, p1, from1, size1, dim1,
                                                  o1, conj1, beta, fromr, sizer, dimr, o_r, co,
                                                  typeid(tag_type).hash_code()),
                                  comm);
            }

            // Check the compatibility of the tensors
            if (!check_dimensions(o0, size0, o1, size1, o_r, sizer))
                throw std::runtime_error("some dimension does not match");

            // Get the optimal ordering for the output tensor pr_
            Order<Nd> sug_o0;
            Order<Nd> sug_o1;
            Order<Nd> sug_or;
            bool swap_operands;
            suggested_orders_for_contraction(Nd0, o0, size0, conj0, Nd1, o1, size1, conj1, Ndo, o_r,
                                             sizer, sug_o0, sug_o1, sug_or, swap_operands, co);
            if (swap_operands) {
                return contraction_normalized(alpha, p1, from1, size1, dim1, o1, conj1, v1, Nd1, p0,
                                              from0, size0, dim0, o0, conj0, v0, Nd0, beta, pr,
                                              fromr, sizer, dimr, o_r, vr, Ndo, comm, co);
            }

            tracker<Cpu> _t("distributed contraction", Cpu{});

            Coor<Nd> sug_size0 = reorder_coor(size0, find_permutation(o0, sug_o0));
            Coor<Nd> sug_size1 = reorder_coor(size1, find_permutation(o1, sug_o1));
            Coor<Nd> sug_sizer = reorder_coor(sizer, find_permutation(o_r, sug_or));

            // Change the partition of the input tensors so that the local portions to contract
            // are local
            const auto &p01 =
                get_partitions_for_contraction(p0, from0, size0, dim0, o0, sug_o0, p1, from1, size1,
                                               dim1, o1, sug_o1, dimr, o_r, sug_or);
            const auto &p0_ = std::get<0>(p01);
            const auto &p1_ = std::get<1>(p01);
            Components_tmpl<Nd, T, XPU0, XPU1> v0_ =
                reorder_tensor(p0, o0, from0, size0, dim0, v0, p0_, sug_size0, sug_o0, comm, co,
                               false /* don't force copy */, doCacheAlloc);
            Components_tmpl<Nd, T, XPU0, XPU1> v1_ =
                reorder_tensor(p1, o1, from1, size1, dim1, v1, p1_, sug_size1, sug_o1, v0_, comm,
                               co, false /* don't force copy */, doCacheAlloc);

            // Try to avoid the extra allocation
            const auto &pr_ = std::get<2>(p01);
            bool avoid_r_alloc =
                (std::norm(beta) == 0 && fromr == Coor<Nd>{{}} && dimr == sizer && sug_or == o_r &&
                 pr == pr_ && !does_proc_ranges_self_intersect(pr, dimr) &&
                 check_components_compatibility(vr, v0_));

            // Scale the output tensor by beta
            if (!avoid_r_alloc) {
                copy<Nd, Nd, T>(beta, pr, fromr, sizer, dimr, o_r, toConst(vr), pr, fromr, dimr,
                                o_r, vr, comm, EWOp::Copy{}, co);
            }

            // Generate the partitioning and the storage for the output tensor
            Components_tmpl<Nd, T, XPU0, XPU1> vr_ =
                avoid_r_alloc ? vr : like_this_components(pr_, v0_, comm, doCacheAlloc);

            for (unsigned int i = 0; i < v0_.first.size(); ++i) {
                const unsigned int componentId = v0_.first[i].componentId;
                local_contraction_normalized(
                    alpha, sug_o0, p0_[comm.rank][componentId][1], conj0,
                    vector<const T, XPU0>(v0_.first[i].it), Nd0, sug_o1,
                    p1_[comm.rank][componentId][1], conj1, vector<const T, XPU0>(v1_.first[i].it),
                    Nd1, avoid_r_alloc ? beta : T{0}, sug_or, pr_[comm.rank][componentId][1],
                    vr_.first[i].it, Ndo, co);
            }
            for (unsigned int i = 0; i < v0_.second.size(); ++i) {
                const unsigned int componentId = v0_.second[i].componentId;
                local_contraction_normalized(
                    alpha, sug_o0, p0_[comm.rank][componentId][1], conj0,
                    vector<const T, XPU1>(v0_.second[i].it), Nd0, sug_o1,
                    p1_[comm.rank][componentId][1], conj1, vector<const T, XPU1>(v1_.second[i].it),
                    Nd1, avoid_r_alloc ? beta : T{0}, sug_or, pr_[comm.rank][componentId][1],
                    vr_.second[i].it, Ndo, co);
            }

            // Reduce all the subtensors to the final tensor
            Request req;
            if (!avoid_r_alloc) {
                req = copy_request_normalized<Nd, Nd, T>(1, pr_, {{}}, sug_sizer, sug_sizer, sug_or,
                                                         toConst(vr_), pr, fromr, dimr, o_r, vr,
                                                         comm, EWOp::Add{}, co);
            }

            _t.stop();
            if (getDebugLevel() >= 1) {
                for (const auto &i : vr.first) sync(i.it.ctx());
                for (const auto &i : vr.second) sync(i.it.ctx());
                barrier(comm);
            }

            return req;
        }

        /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
        /// \param alpha: factor on the contraction
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param ncomponents0: number of consecutive components in each MPI rank
        /// \param o0: dimension labels for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param v0: data for the first operator
        /// \param ctx0: context for each data pointer in v0
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param ncomponents1: number of consecutive components in each MPI rank
        /// \param o1: dimension labels for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param v1: data for the second operator
        /// \param ctx1: context for each data pointer in v1
        /// \param beta: factor on the destination tensor
        /// \param pr: partitioning of the resulting tensor in consecutive ranges
        /// \param ncomponentsr: number of consecutive components in each MPI rank
        /// \param o_r: dimension labels for the output operator
        /// \param vr: data for the second operator
        /// \param ctxr: context for each data pointer in vr
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        Request
        contraction(T alpha, const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
                    const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0, bool conj0,
                    const Components_tmpl<Nd0, T, XPU0, XPU1> &v0, const Proc_ranges<Nd1> &p1,
                    const Coor<Nd1> &from1, const Coor<Nd1> &size1, const Coor<Nd1> &dim1,
                    const Order<Nd1> &o1, bool conj1, const Components_tmpl<Nd1, T, XPU0, XPU1> &v1,
                    T beta, const Proc_ranges<Ndo> &pr, const Coor<Ndo> &fromr,
                    const Coor<Ndo> &sizer, const Coor<Ndo> &dimr, const Order<Ndo> &o_r,
                    const Components_tmpl<Ndo, T, XPU0, XPU1> &vr, const Comm &comm, CoorOrder co) {

            auto m = get_labels_mask();
            update_label_mask(o0, m);
            update_label_mask(o1, m);
            update_label_mask(o_r, m);
            constexpr std::size_t Nd =
                multiple_of(std::max(std::max(Nd0, Nd1), Ndo), (std::size_t)4);
            auto t0 = dummy_normalize_copy<Nd>(p0, from0, size0, dim0, o0, v0, m);
            auto t1 = dummy_normalize_copy<Nd>(p1, from1, size1, dim1, o1, v1, m);
            auto tr = dummy_normalize_copy<Nd>(pr, fromr, sizer, dimr, o_r, vr, m);
            std::size_t Nd0c, Nd1c, Ndoc;
            for (Nd0c = std::min(Nd0, 1ul); Nd0c < Nd0; ++Nd0c) {
                if (std::find(o1.begin(), o1.end(), o0[Nd0c]) != o1.end()) continue;
                if (std::find(o_r.begin(), o_r.end(), o0[Nd0c]) != o_r.end()) continue;
                break;
            }
            for (Nd1c = std::min(Nd1, 1ul); Nd1c < Nd1; ++Nd1c) {
                if (std::find(o0.begin(), o0.end(), o1[Nd1c]) != o0.end()) continue;
                if (std::find(o_r.begin(), o_r.end(), o1[Nd1c]) != o_r.end()) continue;
                break;
            }
            for (Ndoc = std::min(Ndo, 1ul); Ndoc < Ndo; ++Ndoc) {
                if (std::find(o0.begin(), o0.end(), o_r[Ndoc]) != o0.end()) continue;
                if (std::find(o1.begin(), o1.end(), o_r[Ndoc]) != o1.end()) continue;
                break;
            }
            return contraction_normalized(
                alpha, t0.p, t0.from, t0.size, t0.dim, t0.o, conj0, t0.v, Nd0c, //
                t1.p, t1.from, t1.size, t1.dim, t1.o, conj1, t1.v, Nd1c,        //
                beta, tr.p, tr.from, tr.size, tr.dim, tr.o, tr.v, Ndoc, comm, co);
        }

        /// Check that the size of the ranges are inside the tensor dimensions
        template <std::size_t Nd>
        void check_from_size(const Proc_ranges<Nd> &p, const Coor<Nd> &dim) {
            for (const auto &pi : p)
                for (const auto &it : pi)
                    if (!all_less_or_equal(it.at(1), dim))
                        throw std::runtime_error("invalid partition");
        }

        /// Return a From_size from a partition that can be hashed and stored
        /// \param p: partitioning
        /// \return: From_size

        template <std::size_t Nd, typename Comm>
        Proc_ranges<Nd> get_from_size(const PartitionItem<Nd> *p, std::size_t n, const Comm &comm,
                                      const Coor<Nd> &dim) {
            if (Nd == 0) return {};
            if (n % comm.nprocs != 0)
                throw std::runtime_error("partition is incompatible with MPI communicator");
            Proc_ranges<Nd> r(comm.nprocs);
            unsigned int ncomponents = n / comm.nprocs;
            for (unsigned int i = 0; i < comm.nprocs; ++i) r[i].resize(ncomponents);
            for (unsigned int i = 0; i < n; ++i)
                if (volume(p[i][1]) > 0) r[i / ncomponents][i % ncomponents] = p[i];
            check_from_size(r, dim);
            return r;
        }
    }

    namespace detail {
        /// Approximate factorization of a number with factors of 2 and 3.
        /// The returning value is largest than 0.75 times the original value.

        struct factors_2_3 {
            unsigned int two;   ///< powers of two
            unsigned int three; ///< powers of three
            unsigned int value; ///< 2^two * 3^three

            /// Empty construction; initialize to 1
            factors_2_3() : two(0), three(0), value(1) {}

            /// Constructor
            /// \param number: value to factorize

            factors_2_3(unsigned int number) {
                if (number == 0) throw std::runtime_error("unsupported value");

                // a) Try to exactly factorize the number with powers of two and three
                two = three = 0;
                value = 1;
                unsigned int remaining = number;
                for (; remaining % 2 == 0; ++two, remaining /= 2, value *= 2)
                    ;
                for (; remaining % 3 == 0; ++three, remaining /= 3, value *= 3)
                    ;

                // b) Find as many powers as possible of tree and then two
                for (; remaining >= 3; ++three, remaining /= 3, value *= 3)
                    ;
                if (remaining >= 2) ++two, remaining /= 2, value *= 2;

                // c) Try to exchange factors of 3 by 4
                for (; three > 0 && value * 4 / 3 <= number;
                     --three, two += 2, value = value * 4 / 3)
                    ;
            }

            /// Internal constructor
            factors_2_3(unsigned int two, unsigned int three, unsigned int value)
                : two(two), three(three), value(value) {}

            factors_2_3 operator*(const factors_2_3 &v) const {
                return {two + v.two, three + v.three, value * v.value};
            }
        };
    }

    /// Return the number of processes in each direction to partition the tensor
    /// \param order: dimension labels
    /// \param dim: dimension size for the tensor
    /// \param dist_labels: labels to distribute
    /// \param nprocs: number of precesses

    template <std::size_t Nd, typename std::enable_if<(Nd > 0), bool>::type = true>
    Coor<Nd> partitioning_distributed_procs(const char *order, const Coor<Nd> &dim,
                                            const char *dist_labels, unsigned int nprocs) {

        Coor<Nd> p; // returning value

        // The default is no distribution, which is one proc in each direction
        for (std::size_t i = 0; i < Nd; ++i) p[i] = 1;

        // Get the labels that are going to be distributed
        Order<Nd> order_ = detail::toArray<Nd>(order, "order");
        Coor<Nd> dist_perm;
        unsigned int dist_n = 0;
        for (unsigned int i = 0, n = std::strlen(dist_labels); i < n; ++i) {
            const auto &it = std::find(order_.begin(), order_.end(), dist_labels[i]);
            if (it != order_.end() && dim[it - order_.begin()] > 1)
                dist_perm[dist_n++] = it - order_.begin();
        }

        // Return the default distribution If no dimension is going to be distributed or the tensor is empty
        if (dist_n == 0 || detail::volume(dim) == 0 || nprocs <= 1) return p;

        std::array<detail::factors_2_3, Nd> p_f23;
        for (unsigned int i = 0; i < dist_n; ++i) p_f23[i] = detail::factors_2_3(1);
        detail::factors_2_3 vol_p(1);

        // Iteratively put factors 2 and 3 on the coordinates with largest size per process
        detail::factors_2_3 nprocs_f23(nprocs);
        std::array<detail::factors_2_3, 2> factors{3u, 2u};
        while (true) {
            // Sort the dimensions by local size from largest to smalles
            Coor<Nd> perm;
            for (unsigned int j = 0; j < dist_n; ++j) perm[j] = j;
            for (unsigned int j = 0; j < dist_n; ++j) {
                unsigned int large_i = j;
                std::size_t large_val = dim[dist_perm[perm[j]]] / p_f23[perm[j]].value;
                for (unsigned int i = j + 1; i < dist_n; ++i) {
                    std::size_t val = dim[dist_perm[perm[i]]] / p_f23[perm[i]].value;
                    if (large_val < val) large_i = i, large_val = val;
                }
                std::swap(perm[j], perm[large_i]);
            }

            // Try to put a factor of three or two in that direction
            bool factor_applied = false;
            for (unsigned int j = 0; j < dist_n; ++j) {
                for (const auto &factor : factors) {
                    if (nprocs_f23.value % (vol_p.value * factor.value) == 0) {
                        p_f23[perm[j]] = p_f23[perm[j]] * factor;
                        vol_p = vol_p * factor;
                        factor_applied = true;
                        break;
                    }
                }
                if (factor_applied) break;
            }
            if (factor_applied) continue;

            // Get out if we cannot put more factors
            break;
        }

        for (unsigned int i = 0; i < dist_n; ++i) p[dist_perm[i]] = p_f23[i].value;
        assert(detail::volume(p) <= nprocs && detail::volume(p) >= nprocs * 3 / 4);
        return p;
    }

    /// Return a partitioning for a tensor of `dim` dimension onto a grid of processes
    /// \param order: (can be null) dimension labels
    /// \param dim: dimension size for the tensor
    /// \param procs: number of processes in each direction
    /// \param dist_labels: (can be null) order use to assign the processes to each subtensor
    /// \param nprocs: (optional) number of precesses
    /// \param ncomponents: (optional) number of components

    template <std::size_t Nd>
    std::vector<PartitionItem<Nd>> basic_partitioning(const char *order, Coor<Nd> dim,
                                                      Coor<Nd> procs, const char *dist_labels,
                                                      int nprocs = -1, int ncomponents = 1) {

        // Check other arguments
        int vol_procs = (int)detail::volume<Nd>(procs);
        if (nprocs >= 0 && vol_procs > nprocs)
            throw std::runtime_error(
                "The total number of processes from `procs` is greater than `nprocs`");

        // Reorder the labels starting with dist_labels
        Coor<Nd> perm;
        if (order != nullptr && dist_labels != nullptr) {
            if (std::strlen(order) != Nd)
                throw std::runtime_error("basic_partitioning: invalid `order`, its length doesn't "
                                         "match the template parameter");
            const unsigned int n = std::strlen(dist_labels);
            unsigned int dist_n = 0;
            for (unsigned int i = 0; i < n; ++i) {
                const auto &it = std::find(order, order + Nd, dist_labels[i]);
                if (it != order + Nd) perm[dist_n++] = it - order;
            }
            for (unsigned int i = 0; i < Nd; ++i) {
                const auto &it = std::find(dist_labels, dist_labels + n, order[i]);
                if (it == dist_labels + n) perm[dist_n++] = i;
            }
            if (dist_n != Nd) throw std::runtime_error("wtf");
        } else {
            for (unsigned int i = 0; i < Nd; ++i) perm[i] = i;
        }

        std::vector<PartitionItem<Nd>> fs((nprocs < 0 ? vol_procs : nprocs) * ncomponents);
        Coor<Nd> procs_perm = detail::reorder_coor(procs, perm);
        Coor<Nd> stride_perm = detail::get_strides<IndexType>(procs_perm, SlowToFast);
        for (int rank = 0; rank < vol_procs; ++rank) {
            Coor<Nd> cproc = detail::index2coor(rank, procs_perm, stride_perm);
            PartitionItem<Nd> fsi;
            for (std::size_t i = 0; i < Nd; ++i) {
                // Number of elements in process with rank 'cproc[i]' on dimension 'i'
                fsi[1][perm[i]] = dim[perm[i]] / procs_perm[i] +
                                  (dim[perm[i]] % procs_perm[i] > cproc[i] ? 1 : 0);

                // First coordinate in process with rank 'rank' on dimension 'i'
                fsi[0][perm[i]] = fsi[1][perm[i]] == dim[perm[i]]
                                      ? 0
                                      : dim[perm[i]] / procs_perm[i] * cproc[i] +
                                            std::min(cproc[i], dim[perm[i]] % procs_perm[i]);
            }

            // Normalize empty ranges
            if (detail::volume(fsi[1]) == 0) fsi[0] = fsi[1] = Coor<Nd>{{}};

            if (ncomponents == 1) {
                fs[rank] = fsi;
            } else {
                auto fsi_components = basic_partitioning(
                    order, fsi[1],
                    partitioning_distributed_procs(order, fsi[1], dist_labels, ncomponents),
                    dist_labels, ncomponents);
                for (int c = 0; c < ncomponents; ++c) {
                    using detail::operator+;
                    fs[rank * ncomponents + c] = {fsi_components[c][0] + fsi[0],
                                                  fsi_components[c][1]};
                    if (detail::volume(fs[rank * ncomponents + c][1]) == 0)
                        fs[rank * ncomponents + c][0] = fs[rank * ncomponents + c][1] =
                            Coor<Nd>{{}};
                }
            }
        }

        return fs;
    }

    /// Return a partitioning for a tensor of `dim` dimension onto a grid of processes
    /// \param dim1: dimension size for the tensor
    /// \param procs: number of processes on each dimension
    /// \param nprocs: (optional) total number of processes; if not given or it is less than the zero,
    ///                it will be the product of all elements in `procs`
    /// \param replicate: (optional) if true and the total processes of `procs` is one, then replicate
    ///                   the support of the tensor on every process
    /// \param ext_power: (optional) extend the support that many units in the positive and negative
    ///                   direction for each dimension

    template <std::size_t Nd>
    std::vector<PartitionItem<Nd>> basic_partitioning(Coor<Nd> dim, Coor<Nd> procs, int nprocs = -1,
                                                      bool replicate = false,
                                                      Coor<Nd> ext_power = {{}}) {
        int vol_procs = (int)detail::volume<Nd>(procs);
        if (nprocs >= 0 && vol_procs > nprocs)
            throw std::runtime_error(
                "The total number of processes from `procs` is greater than `nprocs`");
        for (std::size_t i = 0; i < Nd; ++i)
            if (ext_power[i] < 0) throw std::runtime_error("Unsupported value for `power`");

        std::vector<PartitionItem<Nd>> fs(nprocs < 0 ? vol_procs : nprocs);
        Coor<Nd> stride = detail::get_strides<IndexType>(procs, SlowToFast);
        for (int rank = 0; rank < vol_procs; ++rank) {
            Coor<Nd> cproc = detail::index2coor(rank, procs, stride);
            for (std::size_t i = 0; i < Nd; ++i) {
                // Number of elements in process with rank 'cproc[i]' on dimension 'i'
                fs[rank][1][i] = std::min(
                    dim[i] / procs[i] + (dim[i] % procs[i] > cproc[i] ? 1 : 0) + ext_power[i] * 2,
                    dim[i]);

                // First coordinate in process with rank 'rank' on dimension 'i'
                fs[rank][0][i] = fs[rank][1][i] == dim[i] ? 0
                                                          : (dim[i] / procs[i] * cproc[i] +
                                                             std::min(cproc[i], dim[i] % procs[i]) -
                                                             ext_power[i] + dim[i]) %
                                                                dim[i];
            }
        }
        if (replicate && vol_procs == 1)
            for (auto &fsi : fs) fsi = fs[0];
        return fs;
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Copy the content of plural tensor v0 into v1
    /// \param alpha: factor applied to v0
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param mpicomm: MPI communicator context
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param mask0: vector of mask pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param mask1: vector of mask pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param request: (optional) return a callback to finish the operation later with `wait`
    /// \param session: (optional) concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void copy(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
              const T **v0, const MaskType **mask0, const Context *ctx0,
              const PartitionItem<Nd1> *p1, int ncomponents1, const char *o1,
              const Coor<Nd1> &from1, const Coor<Nd1> &dim1, Q **v1, const MaskType **mask1,
              const Context *ctx1, MPI_Comm mpicomm, CoorOrder co, CopyAdd copyadd,
              Request *request = nullptr, Session session = 0) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        Request r = detail::copy<Nd0, Nd1>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, comm, dim0), from0, size0,
            dim0, detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, mask0, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, comm, dim1), from1, dim1,
            detail::toArray<Nd1>(o1, "o1"),
            detail::get_components<Nd1>(v1, mask1, ctx1, ncomponents1, p1, comm, session), comm,
            copyadd, co);

        if (request)
            *request = r;
        else
            wait(r);
    }
#endif // SUPERBBLAS_USE_MPI

    /// Copy the content of plural tensor v0 into v1
    /// \param alpha: factor applied to v0
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param dim0: dimension size for the origin tensor
    /// \param v0: vector of data pointers for the origin tensor
    /// \param data0: vector of mask pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param dim1: dimension size for the destination tensor
    /// \param v1: vector of data pointers for the origin tensor
    /// \param mask1: vector of mask pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param request: (optional) return a callback to finish the operation later with `wait`
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void copy(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> from0, const Coor<Nd0> size0, const Coor<Nd0> dim0,
              const T **v0, const MaskType **mask0, const Context *ctx0,
              const PartitionItem<Nd1> *p1, int ncomponents1, const char *o1, const Coor<Nd1> from1,
              const Coor<Nd1> dim1, Q **v1, const MaskType **mask1, const Context *ctx1,
              CoorOrder co, CopyAdd copyadd, Request *request = nullptr, Session session = 0) {

        detail::SelfComm comm = detail::get_comm();

        wait(detail::copy<Nd0, Nd1>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, comm, dim0), from0, size0,
            dim0, detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, mask0, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, comm, dim1), from1, dim1,
            detail::toArray<Nd1>(o1, "o1"),
            detail::get_components<Nd1>(v1, mask1, ctx1, ncomponents1, p1, comm, session), comm,
            copyadd, co));
        if (request) *request = Request{};
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
    /// \param alpha: factor on the contraction
    /// \param p0: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponents1: number of consecutive components in each MPI rank
    /// \param o1: dimension labels for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param ctx1: context for each data pointer in v1
    /// \param beta: factor on the destination tensor
    /// \param pr: partitioning of the resulting tensor in consecutive ranges
    /// \param ncomponentsr: number of consecutive components in each MPI rank
    /// \param o_r: dimension labels for the output operator
    /// \param vr: data for the second operator
    /// \param ctxr: context for each data pointer in vr
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T alpha, const PartitionItem<Nd0> *p0, const Coor<Nd0> &from0,
                     const Coor<Nd0> &size0, const Coor<Nd0> &dim0, int ncomponents0,
                     const char *o0, bool conj0, const T **v0, const Context *ctx0,
                     const PartitionItem<Nd1> *p1, const Coor<Nd1> &from1, const Coor<Nd1> &size1,
                     const Coor<Nd1> &dim1, int ncomponents1, const char *o1, bool conj1,
                     const T **v1, const Context *ctx1, T beta, const PartitionItem<Ndo> *pr,
                     const Coor<Ndo> &fromr, const Coor<Ndo> &sizer, const Coor<Ndo> &dimr,
                     int ncomponentsr, const char *o_r, T **vr, const Context *ctxr,
                     MPI_Comm mpicomm, CoorOrder co, Request *request = nullptr,
                     Session session = 0) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        Request r = detail::contraction<Nd0, Nd1, Ndo>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, comm, dim0), from0, size0,
            dim0, o0_, conj0,
            detail::get_components<Nd0>((T **)v0, nullptr, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, comm, dim1), from1, size1, dim1,
            o1_, conj1,
            detail::get_components<Nd1>((T **)v1, nullptr, ctx1, ncomponents1, p1, comm, session),
            beta, detail::get_from_size(pr, ncomponentsr * comm.nprocs, comm, dimr), fromr, sizer,
            dimr, o_r_,
            detail::get_components<Ndo>(vr, nullptr, ctxr, ncomponentsr, pr, comm, session), comm,
            co);
        if (request)
            *request = r;
        else
            wait(r);
    }

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<!detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T, const PartitionItem<Nd0> *, const Coor<Nd0> &, const Coor<Nd0> &,
                     const Coor<Nd0> &, int, const char *, bool, const T **, const Context *,
                     const PartitionItem<Nd1> *, const Coor<Nd1> &, const Coor<Nd1> &,
                     const Coor<Nd1> &, int, const char *, bool, const T **, const Context *, T,
                     const PartitionItem<Ndo> *, const Coor<Ndo> &, const Coor<Ndo> &,
                     const Coor<Ndo> &, int, const char, T **, const Context *, MPI_Comm, CoorOrder,
                     Request * = nullptr, Session = 0) {
        throw std::runtime_error("contraction: unsupported type");
    }
#endif // SUPERBBLAS_USE_MPI

    /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
    /// \param alpha: factor on the contraction
    /// \param p0: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponents1: number of consecutive components in each MPI rank
    /// \param o1: dimension labels for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param ctx1: context for each data pointer in v1
    /// \param beta: factor on the destination tensor
    /// \param pr: partitioning of the resulting tensor in consecutive ranges
    /// \param ncomponentsr: number of consecutive components in each MPI rank
    /// \param o_r: dimension labels for the output operator
    /// \param vr: data for the second operator
    /// \param ctxr: context for each data pointer in vr
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T alpha, const PartitionItem<Nd0> *p0, const Coor<Nd0> from0,
                     const Coor<Nd0> size0, const Coor<Nd0> &dim0, int ncomponents0, const char *o0,
                     bool conj0, const T **v0, const Context *ctx0, const PartitionItem<Nd1> *p1,
                     const Coor<Nd1> &from1, const Coor<Nd1> &size1, const Coor<Nd1> &dim1,
                     int ncomponents1, const char *o1, bool conj1, const T **v1,
                     const Context *ctx1, T beta, const PartitionItem<Ndo> *pr,
                     const Coor<Ndo> &fromr, const Coor<Ndo> &sizer, const Coor<Ndo> &dimr,
                     int ncomponentsr, const char *o_r, T **vr, const Context *ctxr, CoorOrder co,
                     Request *request = nullptr, Session session = 0) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        detail::SelfComm comm = detail::get_comm();

        wait(detail::contraction<Nd0, Nd1, Ndo>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, comm, dim0), from0, size0,
            dim0, o0_, conj0,
            detail::get_components<Nd0>((T **)v0, nullptr, ctx0, ncomponents0, p0, comm, session),
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, comm, dim1), from1, size1, dim1,
            o1_, conj1,
            detail::get_components<Nd1>((T **)v1, nullptr, ctx1, ncomponents1, p1, comm, session),
            beta, detail::get_from_size(pr, ncomponentsr * comm.nprocs, comm, dimr), fromr, sizer,
            dimr, o_r_,
            detail::get_components<Ndo>(vr, nullptr, ctxr, ncomponentsr, pr, comm, session), comm,
            co));
        if (request) *request = Request{};
    }

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T,
              typename std::enable_if<!detail::supported_type_for_contractions<T>::value,
                                      bool>::type = true>
    void contraction(T, const PartitionItem<Nd0> *, const Coor<Nd0> &, int, const char *, bool,
                     const T **, const Context *, const PartitionItem<Nd1> *, const Coor<Nd1> &,
                     int, const char *, bool, const T **, const Context *, T,
                     const PartitionItem<Ndo> *, const Coor<Ndo> &, int, const char, T **,
                     const Context *, CoorOrder, Request * = nullptr, Session = 0) {
        throw std::runtime_error("contraction: unsupported type");
    }

    namespace detail {
        /// Return the subranges resulting from subtracting a range, that is, making a hole
        /// \param from: first element of the range to subtract
        /// \param size: number of elements in each direction of the range to subtract
        /// \param dim: total number of elements in each direction

        template <std::size_t N>
        std::vector<std::array<Coor<N>, 2>> make_hole(const Coor<N> &from, const Coor<N> &size,
                                                      const Coor<N> &dim) {
            /// Shortcut when N == 0
            if (N == 0) return {};

            /// Shortcut when subtracting an empty range
            if (detail::volume(size) == 0) return {std::array<Coor<N>, 2>{Coor<N>{{}}, dim}};

            // In the general case, return as many subranges as dimensions, each of the subranges
            // follows the pattern
            //  returned |  Coor 0  |  Coor 1  |  Coor 2  |
            //  subrange | subrange | subrange | subrange | ...
            //  --------------------------------------------
            //      0    | antihole |   full   |  full    | ...
            //      1    |   hole   | antihole |  full    | ...
            //      2    |   hole   |   hole   | antihole | ...
            //    ...

            std::vector<std::array<Coor<N>, 2>> r(N); // subranges to return
            for (std::size_t i = 0; i < N; ++i) {
                Coor<N> nfrom, nsize;
                // Fill with hole
                for (std::size_t j = 0; j < i; j++) {
                    nfrom[j] = from[j];
                    nsize[j] = size[j];
                }

                // Fill with the antihole
                nfrom[i] = detail::normalize_coor(from[i] + size[i], dim[i]);
                nsize[i] = dim[i] - size[i];

                // Fill with full
                for (std::size_t j = i + 1; j < N; j++) {
                    nfrom[j] = 0;
                    nsize[j] = dim[j];
                }

                r[i] = std::array<Coor<N>, 2>{nfrom, nsize};
            }

            return r;
        }
    }

    /// Return the subranges resulting from subtracting a range from another range, that is, making a hole
    /// \param from: first element of the range to subtract from
    /// \param size: number of elements in each direction of the range to subtract from
    /// \param hole_from: first element of the range to subtract
    /// \param hole_size: number of elements in each direction of the range to subtract
    /// \param dim: total number of elements in each direction

    template <std::size_t N>
    std::vector<std::array<Coor<N>, 2>> make_hole(const Coor<N> &from, const Coor<N> &size,
                                                  const Coor<N> &hole_from,
                                                  const Coor<N> &hole_size, const Coor<N> &dim) {
        /// Shortcut when N == 0
        if (N == 0) return {};

        /// Shortcut when subtracting an empty range
        if (detail::volume(hole_size) == 0)
            return std::vector<std::array<Coor<N>, 2>>(1, std::array<Coor<N>, 2>{from, size});

        // Make a hole on the whole tensor
        auto parts = detail::make_hole(hole_from, hole_size, dim);

        // Intersect the parts with the range
        auto final_parts = detail::intersection(parts, from, size, dim);

        // Filter out empty subregions
        std::vector<std::array<Coor<N>, 2>> r;
        r.reserve(final_parts.size());
        for (const auto &fs : final_parts)
            if (detail::volume(fs[1]) > 0) r.push_back(fs);
        return r;
    }
}

#endif //  __SUPERBBLAS_DIST__
