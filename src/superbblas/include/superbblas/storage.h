#ifndef __SUPERBBLAS_STORAGE__
#define __SUPERBBLAS_STORAGE__

#include "crc32.h"
#include "dist.h"
#include "tensor.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <list>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#if defined(SUPERBBLAS_USE_ANARCHOFS) && defined(SUPERBBLAS_USE_MPI)
#    include "anarchofs_lib.h"
#endif

/// Specification for simple, sparse, streamed tensor (S3T) format
/// magic_number <i32>: 314
/// version <i32>: version of S3T format (currently 0)
/// values_datatype <i32>: datatype used for the values; currently supported values:
///  - 0: float
///  - 1: double
///  - 2: complex float
///  - 3: complex double
/// checksum <i32>: checksum level
///  - 0: no checksums
///  - 1: global checksum on the entire file
///  - 2: checksum every block
/// dimensions <i32>: number of dimensions
/// metadata_size <i32>: length of the metadata in <char>s
/// metadata_content <char*metadata_size>: content of the metadata
/// padding <char*((8 - metadata_size % 8) % 8)>: zero
/// size <double*dimensions>: size of the tensor in each dimension (coordinates in SlowToFast)
/// checksum_block <double>: largest contiguous data length in bytes to compute the checksum;
///  data blocks larger than that will report the checksum of the checksums
/// num_chunks <double>: number of chunks that follows
/// chunk: repeat as many times as needed
///  -  number_of_blocks <double>: number of blocks
///  -  from_size <{from <double*dimensions>, size <double*dimensions>}*number_of_blocks>: the i-th
///     pair of coordinates indicates the first coordinate present in the i-th block and the size
///     of the block in each dimension;
///  -  values <type indicated by values_type>
///  -  (if checksum is 2) values_checksum <double*number_of_blocks>: checksum of the values.
/// (if checksum is 1 or 2) global_checksum <double>: if checksum is 1, this is the checksum
/// of the entire content of the file up to this position; if checksum is 2, this is the
/// checksum of the entire content of the file up this position excepting `num_chunks`, `values`
/// and `values_checksum`.
///
/// NOTES:
/// - The restrictions in the metadata's length and the padding are to make all subsequent fields
///   properly aligned in case an implementation accesses the file by mapping it into memory.
/// - A slightly simpler implementation is to restrict each chunk to a single block; allowing
///   multiple blocks in a chunks gives a mechanism to increase the locality when reading all
///   "from_size" in the file.
/// - The type of the coordinates for from_size is double instead of the obvious better type i64
///   just because the latter type is not supported by MPI.

namespace superbblas {

    /// Type of the values
    enum values_datatype { FLOAT = 0, DOUBLE = 1, CFLOAT = 2, CDOUBLE = 3, CHAR = 4, INT = 5 };

    /// Type of checksum
    enum checksum_type {
        NoChecksum = 0,     ///< Do not do any checksum
        GlobalChecksum = 1, ///< Checksum the entire file
        BlockChecksum = 2   ///< Checksum every block individually
    };

    namespace detail {
        /// Magic number
        const int magic_number = 314;

        /// Open file modes
        enum Mode { CreateForReadWrite, ReadWrite, OnlyRead };

        /// Return the values_datatype of a type
        template <typename T> values_datatype get_values_datatype();
        template <> inline values_datatype get_values_datatype<float>() { return FLOAT; }
        template <> inline values_datatype get_values_datatype<double>() { return DOUBLE; }
        template <> inline values_datatype get_values_datatype<std::complex<float>>() {
            return CFLOAT;
        }
        template <> inline values_datatype get_values_datatype<std::complex<double>>() {
            return CDOUBLE;
        }
        template <> inline values_datatype get_values_datatype<int>() { return INT; }
        template <> inline values_datatype get_values_datatype<char>() { return CHAR; }

        /// File descriptor
        template <typename Comm> struct File;

        /// Call used for open the storage
        enum CommType {
            SEQ, ///< without MPI
            MPI  ///< with MPI
        };

        //
        // Low layer implementation without MPI
        //

        // File descriptor specialization for SelfComm
        template <> struct File<SelfComm> {
            using type = std::FILE *;
            static constexpr CommType value = SEQ;
        };

        template <typename Str> void gen_error(const Str &error_msg) {
            std::stringstream ss;
            ss << error_msg << ": " << strerror(errno);
            throw std::runtime_error(ss.str());
        }

        inline std::FILE *file_open_local(SelfComm, const char *filename, Mode mode) {
            std::FILE *f = nullptr;
            switch (mode) {
            case CreateForReadWrite: f = std::fopen(filename, "wb+"); break;
            case ReadWrite: f = std::fopen(filename, "rb+"); break;
            case OnlyRead: f = std::fopen(filename, "rb"); break;
            }
            if (f == nullptr) {
                std::stringstream ss;
                ss << "Error opening file `" << filename << "'";
                gen_error(ss.str());
            }
            return f;
        }

        inline void seek(std::FILE *f, std::size_t offset) {
            if (offset >= (std::size_t)std::numeric_limits<long>::max())
                gen_error("Too small type to represent the displacement");
            if (std::fseek(f, offset, SEEK_SET) != 0) gen_error("Error setting file position");
        }

        template <typename T> void write(std::FILE *f, const T *v, std::size_t n) {
            if (std::fwrite(v, sizeof(T), n, f) != n) gen_error("Error writing in a file");
        }

        template <typename T> void iwrite(std::FILE *f, const T *v, std::size_t n) {
            write(f, v, n);
        }

        template <typename T> void iwrite(std::FILE *f, const T *v, std::size_t n, vector<T, Cpu>) {
            write(f, v, n);
        }

        template <typename T> void read(std::FILE *f, T *v, std::size_t n) {
            if (std::fread(v, sizeof(T), n, f) != n) gen_error("Error reading from a file");
        }

        inline void preallocate(std::FILE *f, std::size_t n) {
            off_t old_offset = 0, end_of_file;
            if (n >= (std::size_t)std::numeric_limits<long>::max())
                throw std::runtime_error("Too small type to represent the displacement");

            // Save the current position on the file
            if ((old_offset = std::ftell(f)) == -1) gen_error("Error getting file position");

            // Get the current size of the file
            if (std::fseek(f, -1, SEEK_END) != 0) gen_error("Error setting file position");
            if ((end_of_file = std::ftell(f) + 1) == 0) gen_error("Error getting file position");
            std::size_t curr_size = end_of_file;

            if (curr_size < n) {
                std::vector<char> v(std::min(n - curr_size, (std::size_t)256 * 1024 * 1024));

                if (std::fseek(f, 0, SEEK_END) != 0) gen_error("Error setting file position");
                while (curr_size < n) {
                    std::size_t d = std::min(n - curr_size, v.size());
                    write(f, v.data(), d);
                    curr_size += d;
                }
            }

            // Restore position on the file
            if (std::fseek(f, old_offset, SEEK_SET) != 0) gen_error("Error setting file position");
        }

        inline void truncate(std::FILE *f, std::size_t n) {
            if (ftruncate(fileno(f), n) != 0) gen_error("Error truncating file");
        }

        inline void flush(std::FILE *f) {
            if (std::fflush(f) != 0) gen_error("Error flushing file");
        }

        inline void check_pending_requests(std::FILE *) {}

        inline void close(std::FILE *f) {
            if (std::fclose(f) != 0) gen_error("Error closing file");
        }

        //
        // Low layer implementation with MPI
        //

#ifdef SUPERBBLAS_USE_MPI
#    ifdef SUPERBBLAS_USE_MPIIO
        /// Use MPI IO

        /// Return the MPI_Datatype of a type
        template <typename T> MPI_Datatype mpi_datatype_basic_from_type();
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<float>() { return MPI_FLOAT; }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<double>() {
            return MPI_DOUBLE;
        }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<std::complex<float>>() {
            return MPI_FLOAT;
        }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<std::complex<double>>() {
            return MPI_DOUBLE;
        }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<int>() { return MPI_INT; }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<char>() { return MPI_CHAR; }
        template <> inline MPI_Datatype mpi_datatype_basic_from_type<unsigned char>() {
            return MPI_CHAR;
        }

        /// Return how many items of the basic type are in T type
        template <typename T> unsigned int get_count_from_type();
        template <> inline unsigned int get_count_from_type<float>() { return 1; }
        template <> inline unsigned int get_count_from_type<double>() { return 1; }
        template <> inline unsigned int get_count_from_type<std::complex<float>>() { return 2; }
        template <> inline unsigned int get_count_from_type<std::complex<double>>() { return 2; }
        template <> inline unsigned int get_count_from_type<int>() { return 1; }
        template <> inline unsigned int get_count_from_type<char>() { return 1; }
        template <> inline unsigned int get_count_from_type<unsigned char>() { return 1; }

        struct AllocAbstract {
            MPI_Request req;
            AllocAbstract(MPI_Request req) : req{req} {}
            virtual ~AllocAbstract() {}
        };

        template <typename T> struct Alloc : public AllocAbstract {
            vector<T, Cpu> v;
            Alloc(MPI_Request req, vector<T, Cpu> v) : AllocAbstract{req}, v{v} {}
            ~Alloc() override {}
        };

        // MPI_File replacement
        struct File_Requests {
            MPI_File f;
            std::list<AllocAbstract *> reqs;
            File_Requests() : f{}, reqs{} {}
            File_Requests(MPI_File f) : f(f), reqs{} {}
        };

        // File descriptor specialization for MpiComm
        template <> struct File<MpiComm> {
            using type = File_Requests;
            static constexpr CommType value = MPI;
        };

        inline File_Requests file_open_local(MpiComm comm, const char *filename, Mode mode) {
            MPI_File fh;
            barrier(comm);
            switch (mode) {
            case CreateForReadWrite:
                // Delete file if it exists
                MPI_File_delete(filename, MPI_INFO_NULL);
                MPI_check(MPI_File_open(comm.comm, filename, MPI_MODE_CREATE | MPI_MODE_RDWR,
                                        MPI_INFO_NULL, &fh));
                break;
            case ReadWrite:
                MPI_check(MPI_File_open(comm.comm, filename, MPI_MODE_RDWR, MPI_INFO_NULL, &fh));
                break;
            case OnlyRead:
                MPI_check(MPI_File_open(comm.comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh));
                break;
            }
            return {fh};
        }

        inline void seek(File_Requests &f, std::size_t offset) {
            MPI_check(MPI_File_seek(f.f, offset, MPI_SEEK_SET));
        }

        template <typename T> void write(File_Requests &f, const T *v, std::size_t n) {
            MPI_Status status;
            MPI_check(MPI_File_write(f.f, v, n * get_count_from_type<T>(),
                                     mpi_datatype_basic_from_type<T>(), &status));
        }

        template <typename T> void read(File_Requests &f, T *v, std::size_t n) {
            MPI_Status status;
            MPI_check(MPI_File_read(f.f, v, n * get_count_from_type<T>(),
                                    mpi_datatype_basic_from_type<T>(), &status));
        }

        template <typename T>
        void iwrite(File_Requests &f, const T *v, std::size_t n, vector<T, Cpu> w) {
            MPI_Request req;
            MPI_check(MPI_File_iwrite(f.f, v, n * get_count_from_type<T>(),
                                      mpi_datatype_basic_from_type<T>(), &req));
            f.reqs.push_back(new Alloc<T>{req, w});
        }

        template <typename T> void iwrite(File_Requests &f, const T *v, std::size_t n) {
            vector<T, Cpu> w(n, Cpu{0});
            std::copy_n(v, n, w.data());
            iwrite(f, w.data(), n, w);
        }

        inline void flush(File_Requests &f) {
            for (AllocAbstract *r : f.reqs) {
                MPI_check(MPI_Wait(&r->req, MPI_STATUS_IGNORE));
                delete r;
            }
            f.reqs.clear();
            MPI_check(MPI_File_sync(f.f));
        }

        inline void check_pending_requests(File_Requests &f) {
            f.reqs.remove_if([](AllocAbstract *r) {
                int finished = 0;
                MPI_check(MPI_Test(&r->req, &finished, MPI_STATUS_IGNORE));
                if (finished) {
                    delete r;
                    return true;
                }
                return false;
            });
        }

        inline void preallocate(File_Requests &f, std::size_t n) {
            flush(f);
            MPI_check(MPI_File_preallocate(f.f, n));
        }

        inline void truncate(File_Requests &f, std::size_t n) {
            MPI_check(MPI_File_set_size(f.f, n));
        }

        inline void close(File_Requests &f) {
            flush(f);
            MPI_check(MPI_File_close(&f.f));
        }

#    else  // SUPERBBLAS_USE_MPIIO
        /// Don't use MPI IO

        // MPI_File replacement
        struct File_Comm {
            std::FILE *f;
            MpiComm comm;
        };

        // File descriptor specialization for MpiComm
        template <> struct File<MpiComm> {
            using type = File_Comm;
            static constexpr CommType value = MPI;
        };

        inline File_Comm file_open_local(MpiComm comm, const char *filename, Mode mode) {
            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                check_consistency(std::make_tuple(std::string("file_open_local"),
                                                  std::string(filename), (int)mode),
                                  comm);
            }

            std::FILE *f = nullptr;
            // Avoid all processes to create the file at the same time; so root process create the file, and the rest open it
            if (comm.rank == 0) {
                f = file_open_local(detail::get_comm(), filename, mode);
                barrier(comm);
            } else {
                barrier(comm);
                f = file_open_local(detail::get_comm(), filename,
                                    mode == OnlyRead ? OnlyRead : ReadWrite);
            }
            return {f, comm};
        }

        inline void preallocate(File_Comm f, std::size_t n) {
            if (getDebugLevel() > 0) {
                check_consistency(std::make_tuple(std::string("file_preallocate"), n), f.comm);
            }

            if (f.comm.rank == 0) preallocate(f.f, n);
            barrier(f.comm);
        }

        inline void truncate(File_Comm f, std::size_t n) {
            if (getDebugLevel() > 0) {
                check_consistency(std::make_tuple(std::string("file_truncate"), n), f.comm);
            }

            if (f.comm.rank == 0) truncate(f.f, n);
            barrier(f.comm);
        }

        inline void seek(File_Comm f, std::size_t offset) { seek(f.f, offset); }

        template <typename T> void write(File_Comm f, const T *v, std::size_t n) {
            write(f.f, v, n);
        }

        template <typename T> void iwrite(File_Comm f, const T *v, std::size_t n) {
            write(f.f, v, n);
        }

        template <typename T> void iwrite(File_Comm f, const T *v, std::size_t n, vector<T, Cpu>) {
            write(f.f, v, n);
        }

        template <typename T> void read(File_Comm f, T *v, std::size_t n) { read(f.f, v, n); }

        inline void flush(File_Comm f) {
            if (getDebugLevel() > 0) {
                check_consistency(std::make_tuple(std::string("file_flush")), f.comm);
            }

            flush(f.f);
            barrier(f.comm);
        }

        inline void check_pending_requests(File_Comm) {}

        inline void close(File_Comm &f) {
            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                check_consistency(std::make_tuple(std::string("close_file")), f.comm);
            }

            close(f.f);
            barrier(f.comm);
        }
#    endif // SUPERBBLAS_USE_MPIIO

#endif // SUPERBBLAS_USE_MPI

#if defined(SUPERBBLAS_USE_ANARCHOFS) && defined(SUPERBBLAS_USE_MPI)
        // Anarchofs and local file handler
        template <typename Comm> struct FileAfs {
            anarchofs::client::File *f_afs;
            typename File<Comm>::type f_local;
            static constexpr CommType value = File<Comm>::value;
        };

        /// General class
        template <typename Comm> using FileHandler = FileAfs<Comm>;

        template <typename Comm>
        inline FileAfs<Comm> file_open(const Comm &comm, const char *filename, Mode mode) {

            // Check if the file starts with afs:
            bool is_remote = (std::strncmp("afs:", filename, 4) == 0);

            // If not is remote, open as a local file
            if (!is_remote) return FileAfs<Comm>{nullptr, file_open_local(comm, filename, mode)};

            // Skip "afs:"
            const char *filename_core = filename + 4;

            // Not supported writing mode
            if (mode != OnlyRead)
                throw std::runtime_error("unsupported write operations for anarchofs");

            // Open the file with anarchofs
            auto f = anarchofs::client::open(filename_core);
            if (f == nullptr) {
                std::stringstream ss;
                ss << "Error opening file `" << filename << "'";
                throw std::runtime_error(ss.str());
            }

            return FileAfs<Comm>{f, {}};
        }

        template <typename Comm> inline void preallocate(FileAfs<Comm> &f, std::size_t n) {
            if (f.f_afs != nullptr)
                throw std::runtime_error("preallocation: unsupported operation for anarchofs");
            preallocate(f.f_local, n);
        }

        template <typename Comm> inline void truncate(FileAfs<Comm> &f, std::size_t n) {
            if (f.f_afs != nullptr)
                throw std::runtime_error("truncate: unsupported operation for anarchofs");
            truncate(f.f_local, n);
        }

        template <typename Comm> inline void seek(FileAfs<Comm> &f, std::size_t offset) {
            if (f.f_afs != nullptr)
                anarchofs::client::seek(f.f_afs, offset);
            else
                seek(f.f_local, offset);
        }

        template <typename Comm, typename T>
        void write(FileAfs<Comm> &f, const T *v, std::size_t n) {
            if (f.f_afs != nullptr)
                throw std::runtime_error("write: unsupported operation for anarchofs");
            write(f.f_local, v, n);
        }

        template <typename Comm, typename T>
        void iwrite(FileAfs<Comm> &f, const T *v, std::size_t n) {
            if (f.f_afs != nullptr)
                throw std::runtime_error("write: unsupported operation for anarchofs");
            iwrite(f.f_local, v, n);
        }

        template <typename Comm, typename T>
        void iwrite(FileAfs<Comm> &f, const T *v, std::size_t n, vector<T, Cpu>) {
            if (f.f_afs != nullptr)
                throw std::runtime_error("write: unsupported operation for anarchofs");
            write(f.f_local, v, n);
        }

        template <typename Comm, typename T> void read(FileAfs<Comm> &f, T *v, std::size_t n) {
            if (f.f_afs != nullptr) {
                if (anarchofs::client::read(f.f_afs, (char *)v, n * sizeof(T)) !=
                    (std::int64_t)(n * sizeof(T)))
                    throw std::runtime_error("read: error reading from anarchofs");
            } else {
                read(f.f_local, v, n);
            }
        }

        template <typename Comm> inline void flush(FileAfs<Comm> &f) {
            if (f.f_afs == nullptr) flush(f.f_local);
        }

        template <typename Comm> inline void check_pending_requests(FileAfs<Comm> &f) {
            if (f.f_afs == nullptr) check_pending_requests(f.f_local);
        }

        template <typename Comm> inline void close(FileAfs<Comm> &f) {
            if (f.f_afs != nullptr) {
                if (!anarchofs::client::close(f.f_afs))
                    throw std::runtime_error("error closing file");
                f.f_afs = nullptr;
            } else {
                close(f.f_local);
            }
        }
#else
        /// General class
        template <typename Comm> using FileHandler = typename File<Comm>::type;

        template <typename Comm>
        inline FileHandler<Comm> file_open(const Comm &comm, const char *filename, Mode mode) {
            return file_open_local(comm, filename, mode);
        }
#endif // defined(SUPERBBLAS_USE_ANARCHOFS) && defined(SUPERBBLAS_USE_MPI)

        /// Data-structure to accelerate the intersection of sparse tensors
        template <std::size_t N, typename Key = void> struct GridHash {
            /// Index element in `blocks` and `values`
            using BlockIndex = std::size_t;
            /// Sparse tensor dimensions
            Coor<N> dim;
            /// Nonzero blocks of the sparse tensors
            std::vector<From_size_item<N>> blocks;
            /// Values associated to each block
            std::vector<Key> values;

            /// List of hyperplanes forming a non-regular grid
            std::array<From_size<1>, N> grid;
            /// From grid coordinate index (SlowToFast) to `blocks` and `values` indices
            std::unordered_multimap<Coor<N>, BlockIndex, TupleHash<Coor<N>>> gridToBlocks;

            GridHash(Coor<N> dim) : dim{dim}, grid{{}}, gridToBlocks(16) {
                assert(check_positive(dim));
            }

            /// Insert a key associated to a range
            /// \param from: coordinates of the first element in the range
            /// \param size: dimensions of the range
            /// \param key: key associate to the range

            void append_block(Coor<N> from, const Coor<N> &size, const Key &key) {
                // Shortcut for empty ranges
                std::size_t vol = volume(size);
                if (vol == 0) return;

                // Normalize from when being the whole dimension
                from = normalize_from(from, size);

                // Add the new block
                blocks.push_back({from, size});
                values.push_back(key);

                // Append the new intervals
                for (unsigned int i = 0; i < N; ++i) {
                    // fs = {from, size} - \sum_j grid[i]_j
                    From_size<1> fs(1, {Coor<1>{from[i]}, Coor<1>{size[i]}});
                    for (const auto &j : grid[i])
                        fs = detail::intersection(
                            fs, Coor<1>{normalize_coor(j[0][0] + j[1][0], dim[i])},
                            Coor<1>{dim[i] - j[1][0]}, Coor<1>{dim[i]});

                    // Insert the remaining at the end of the list of ranges in that direction
                    grid[i].insert(grid[i].end(), fs.begin(), fs.end());
                }

                // Inserting the key into the grid
                for (const auto &gi : grid_intersection(from, size))
                    gridToBlocks.insert({gi, blocks.size() - 1});
            }

            /// Return a list of the blocks, the intersection relative to the blocks, and the
            /// associated keys of blocks with non-empty overlap with the given range.
            /// \param from: coordinates of the first element in the range
            /// \param size: dimensions of the range

            std::vector<std::pair<std::array<From_size_item<N>, 2>, Key>>
            intersection(Coor<N> from, const Coor<N> &size) const {
                // Shortcut for empty ranges
                if (volume(size) == 0) return {};

                // Normalize from when being the whole dimension
                from = normalize_from(from, size);

                // Compute the return
                std::vector<std::pair<std::array<From_size_item<N>, 2>, Key>> r;
                std::unordered_set<BlockIndex> visited(16);
                for (const auto &gi : grid_intersection(from, size)) {
                    const auto ranges = gridToBlocks.equal_range(gi);
                    for (auto it = ranges.first; it != ranges.second; ++it) {
                        BlockIndex bidx = it->second;

                        // Skip if already visited
                        if (visited.count(bidx) > 0) continue;

                        // Do intersection between the block and the given range
                        auto ranges =
                            detail::intersection(blocks[bidx][0], blocks[bidx][1], from, size, dim);
                        for (const auto &fs : ranges) {
                            if (volume(fs[1]) == 0) continue;
                            r.push_back({{blocks[bidx],
                                          {normalize_coor(fs[0] - blocks[bidx][0], dim), fs[1]}},
                                         values[bidx]});
                        }

                        // Note the visited block
                        visited.insert(bidx);
                    }
                }

                return r;
            }

        private:
            /// Return the subgrid with overlaps with the given range
            std::vector<Coor<N>> grid_intersection(Coor<N> from, Coor<N> size) const {
                // Find all the intersections with the intervals in each direction
                std::array<std::vector<IndexType>, N> intervals{{}};
                for (std::size_t i = 0; i < N; ++i)
                    for (unsigned int fs_index = 0; fs_index < grid[i].size(); ++fs_index)
                        if (volume(detail::intersection(grid[i][fs_index][0], grid[i][fs_index][1],
                                                        Coor<1>{from[i]}, Coor<1>{size[i]},
                                                        Coor<1>{dim[i]})) > 0)
                            intervals[i].push_back(fs_index);

                // Produce all the Cartesian combinations
                Coor<N> subgrid_dim;
                for (std::size_t i = 0; i < N; ++i) subgrid_dim[i] = intervals[i].size();
                Coor<N> subgrid_stride = get_strides<IndexType>(subgrid_dim, FastToSlow);
                std::vector<Coor<N>> r;
                IndexType vol = volume(subgrid_dim);
                r.reserve(vol);
                for (IndexType subgrid_index = 0; subgrid_index < vol; ++subgrid_index) {
                    auto c = index2coor(subgrid_index, subgrid_dim, subgrid_stride);
                    Coor<N> gi;
                    for (std::size_t i = 0; i < N; ++i) gi[i] = intervals[i][c[i]];
                    r.push_back(gi);
                }

                return r;
            }

            // Normalize from when being the whole dimension
            Coor<N> normalize_from(Coor<N> from, const Coor<N> &size) const {
                for (std::size_t i = 0; i < N; ++i)
                    if (size[i] == dim[i]) from[i] = 0;
                return normalize_coor(from, dim);
            }
        };

        ///
        /// Checksums
        ///

        /// Do checksum of blocks up to this size
        const std::size_t default_checksum_blocksize = 64 * 1024 * 1024; // 64 MiB

        /// Checksum value type
        using checksum_t = uint32_t;

        /// Compute the checksum of a data block
        /// \param str: pointer to the given data
        /// \param size: size of the data in bytes
        /// \param checksum_blocksize: if greater than zero, compute the checksum of chunks up to
        ///        this size and then return the checksum of the checksums

        template <typename T>
        checksum_t do_checksum(const T *str, std::size_t size = 1,
                               std::size_t checksum_blocksize = 0, checksum_t prev_checksum = 0) {
            Cpu cpu{0};
            tracker<Cpu> _t("do checksums", cpu);

            // Update size to bytes
            size *= sizeof(T);

            // Return the CRC of the string if not using blocking
            if (checksum_blocksize == 0) return crc32(prev_checksum, (unsigned char *)str, size);

            // Do not allow a previous checksum when blocking checksums
            if (prev_checksum != 0) throw std::runtime_error("Ups! This should not happen");

            // Get number of blocks
            std::size_t num_blocks = (size + checksum_blocksize - 1) / checksum_blocksize;

            std::vector<checksum_t> block_checksums(num_blocks);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
            for (std::size_t i = 0; i < num_blocks; ++i) {
                std::size_t first_element = i * checksum_blocksize;
                std::size_t num_elements = std::min(checksum_blocksize, size - first_element);
                block_checksums[i] = crc32(0, (unsigned char *)str + first_element, num_elements);
            }

            return crc32(0, (unsigned char *)block_checksums.data(),
                         num_blocks * sizeof(checksum_t));
        }

        ///
        /// Other auxiliary functions
        ///

        template <typename T,
                  typename std::enable_if<std::is_arithmetic<T>::value, bool>::type = true>
        void change_endianness(T *v, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                char *c = (char *)&v[i];
                for (std::size_t j = 0; j < sizeof(T) / 2; ++j)
                    std::swap(c[j], c[sizeof(T) - 1 - j]);
            }
        }

        template <typename T> void change_endianness(std::complex<T> *v, std::size_t n) {
            change_endianness((T *)v, n * 2u);
        }

        struct Storage_context_abstract {
            virtual std::size_t getNdim() { throw std::runtime_error("Not implemented"); }
            virtual CommType getCommType() { throw std::runtime_error("Not implemented"); }
            virtual void flush() {}
            virtual void preallocate(std::size_t) {}
            virtual ~Storage_context_abstract() {}
        };

        template <std::size_t N, typename Comm> struct Storage_context : Storage_context_abstract {
            values_datatype values_type;  ///< type of the nonzero values
            std::size_t header_size;      ///< number of bytes before the field num_chunks
            std::size_t disp;             ///< number of bytes before the current chunk
            FileHandler<Comm> fh;         ///< file descriptor
            const Coor<N> dim;            ///< global tensor dimensions
            const bool change_endianness; ///< whether to change endianness
            bool modified_for_flush;      ///< whether the storage content changed since last flush
            bool modified_for_checksum; ///< whether the storage content changed since last checksum
            const checksum_type checksum;         ///< What kind of checksum to perform
            const std::size_t checksum_blocksize; ///< blocksize for computing checksums
            checksum_t checksum_val;              ///< checksum of the file excepting the values
                                                  ///< (when checksum is BlockChecksum)
            std::size_t num_chunks;               ///< number of chunks written
            bool allow_writing;                   ///< whether to allow writing

            /// displacement in the file of the values of a block
            std::vector<std::size_t> disp_values;
            /// displacement in the file of the checksum of the values of a block
            std::vector<std::size_t> disp_checksum;
            /// whether the checksum of the values of a block is done
            std::vector<char> is_checksum_done;
            GridHash<N, std::size_t> blocks; ///< list of blocks already written

            Storage_context(values_datatype values_type, std::size_t header_size,
                            FileHandler<Comm> fh, Coor<N> dim, bool change_endianness,
                            bool is_new_storage, checksum_type checksum,
                            std::size_t checksum_blocksize, checksum_t checksum_val,
                            bool allow_writing)
                : values_type(values_type),
                  header_size(header_size),
                  disp(header_size + sizeof(double)), // hop over num_chunks
                  fh(fh),
                  dim(dim),
                  change_endianness(change_endianness),
                  modified_for_flush(is_new_storage),
                  modified_for_checksum(is_new_storage),
                  checksum(checksum),
                  checksum_blocksize(checksum_blocksize),
                  checksum_val(checksum_val),
                  num_chunks(0),
                  allow_writing(allow_writing),
                  blocks(dim) {}

            std::size_t getNdim() override { return N; }
            CommType getCommType() override { return File<Comm>::value; }
            void flush() override {
                if (modified_for_flush) detail::flush(fh);
            }
            void preallocate(std::size_t size) override { detail::preallocate(fh, size); }
            ~Storage_context() override {
                if (allow_writing) {
                    if (modified_for_flush) detail::flush(fh);
                    std::size_t filesize = disp + (checksum == NoChecksum ? 0 : sizeof(double));
                    truncate(fh, filesize);
                }
                close(fh);
            }
        };

        template <std::size_t N, typename T, typename Comm>
        Storage_context<N, Comm> *get_storage_context(Storage_context_abstract *stoh) {
            if (stoh == nullptr) throw std::runtime_error("The given storage handle is null");
            if (stoh->getNdim() != N || stoh->getCommType() != File<Comm>::value)
                throw std::runtime_error("Invalid storage handle");
            Storage_context<N, Comm> *sto_ctx = (Storage_context<N, Comm> *)stoh;
            if (sto_ctx->values_type != get_values_datatype<T>())
                throw std::runtime_error("Invalid storage handle");
            return sto_ctx;
        }

        template <std::size_t Nd0, std::size_t Nd1> struct Op {
            From_size_item<Nd0> first_tensor;
            From_size_item<Nd0> first_subtensor;
            From_size_item<Nd1> second_tensor;
            From_size_item<Nd1> second_subtensor;
            std::size_t blockIndex;
        };

        /// Return the ranges involved in the source and destination tensors
        /// \param dim0: dimension size for the origin tensor
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param grid: sparse tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied

        template <std::size_t Nd0, std::size_t Nd1, typename From_size0>
        std::vector<std::vector<Op<Nd0, Nd1>>>
        get_overlap_ranges(Coor<Nd0> dim0, const From_size0 &p0, const Order<Nd0> &o0,
                           const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                           const GridHash<Nd1, std::size_t> &grid, const Order<Nd1> &o1,
                           const Coor<Nd1> &from1) {

            Cpu cpu{0};
            tracker<Cpu> _t("comp. tensor overlaps on storage", cpu);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, grid.dim)));

            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            std::vector<std::vector<Op<Nd0, Nd1>>> r(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                // Restrict the local range in v0 to the range from0, size0
                Coor<Nd0> rlocal_from0, rlocal_size0;
                intersection<Nd0>(from0, size0, p0[i][0], p0[i][1], dim0, rlocal_from0,
                                  rlocal_size0);

                // Translate the restricted range to the destination lattice
                Coor<Nd1> rfrom1, rsize1;
                translate_range(rlocal_from0, rlocal_size0, from0, dim0, from1, grid.dim, perm0,
                                rfrom1, rsize1);

                // Compute the range to receive
                auto overlaps = grid.intersection(rfrom1, rsize1);

                for (const auto &o : overlaps) {
                    // Compute the range to left tensor
                    Coor<Nd0> rfrom0, rsize0;
                    translate_range(o.first[0][0] + o.first[1][0], o.first[1][1], from1, grid.dim,
                                    from0, dim0, perm1, rfrom0, rsize0);
                    rfrom0 = normalize_coor(rfrom0 - p0[i][0], dim0);

                    r[i].push_back({p0[i], {rfrom0, rsize0}, o.first[0], o.first[1], o.second});
                }
            }

            return r;
        }

        /// Return the number of dimensions that are contiguous
        /// \param from: first coordinate to copy
        /// \param size: number of coordinates to copy in each direction
        /// \param dim: dimension size for the tensor
        /// \param co: coordinate linearization order
        /// \return: {displacement, blocking, permutation}

        template <typename IndexType, std::size_t Nd>
        std::tuple<std::size_t, IndexType, IndicesT<IndexType, Cpu>>
        get_normalize_permutation(Coor<Nd> from, Coor<Nd> size, Coor<Nd> dim, CoorOrder co,
                                  Cpu cpu) {

            // Work with FastToSlow because `get_permutation` works in that way
            if (co != FastToSlow) {
                from = reverse(from);
                size = reverse(size);
                dim = reverse(dim);
                co = FastToSlow;
            }

            // Going from least to most significant, the last contiguous dimension is the one
            // that the range's size is smaller than the dimension or the range does not start
            // at the origin.
            std::size_t nblock = 0, blocking = 1;
            for (std::size_t i = 0; i < Nd; ++i) {
                if (from[i] + size[i] > dim[i]) break;
                nblock++;
                blocking *= size[i];
                if (from[i] != 0 || size[i] < dim[i]) break;
            }

            // Normalize from
            auto new_from = from;
            IndexType disp = 0;
            auto strides = get_strides<IndexType>(dim, co);
            for (std::size_t i = 0; i < Nd; ++i) {
                if (from[i] + size[i] <= dim[i]) {
                    new_from[i] = 0;
                    disp += from[i] * strides[i];
                }
            }

            // Normalize size to use blocking
            Coor<Nd> new_size = size;
            for (std::size_t i = 0; i < nblock; ++i) new_size[i] = 1;

            // Get permutation
            auto v =
                get_permutation(new_from, new_size, dim, strides, AllowImplicitPermutation, cpu);

            return {disp, blocking, v};
        }

        /// Remove intersections of a list of ranges against another list of ranges
        /// \param fs: list of given ranges
        /// \param p: pointer to the first range to remove
        /// \param num_blocks: number of ranges to remove
        /// \param dim: dimensions

        template <std::size_t Nd>
        std::vector<PartitionItem<Nd>>
        remove_repetitions(const std::vector<PartitionItem<Nd>> &fs, const PartitionItem<Nd> *p,
                           std::size_t num_blocks, const Coor<Nd> &dim) {
            std::vector<PartitionItem<Nd>> r;
            r.reserve(fs.size());
            for (const auto fsi : fs)
                if (volume(fsi[1]) > 0) r.push_back(fsi);
            for (unsigned int i = 0; i < num_blocks; ++i) {
                std::vector<PartitionItem<Nd>> r0;
                r0.reserve(r.size());
                for (const auto &ri : r) {
                    auto new_ri = superbblas::make_hole<Nd>(ri[0], ri[1], p[i][0], p[i][1], dim);
                    for (const auto new_rii : new_ri)
                        if (volume(new_rii[1]) > 0) r0.push_back(new_rii);
                }
                std::swap(r, r0);
            }
            return r;
        }

        /// Remove intersections between a given range and another list of ranges
        /// \param from: first element of the given range
        /// \param size: number of elements of the given range in each direction
        /// \param p: pointer to the first range to remove
        /// \param num_blocks: number of ranges to remove
        /// \param dim: dimensions

        template <std::size_t Nd>
        std::vector<PartitionItem<Nd>>
        remove_repetitions(const Coor<Nd> &from, const Coor<Nd> &size, const PartitionItem<Nd> *p,
                           std::size_t num_blocks, const Coor<Nd> &dim) {
            return remove_repetitions(
                std::vector<PartitionItem<Nd>>(1, PartitionItem<Nd>{from, size}), p, num_blocks,
                dim);
        }

        /// Copy the content of tensor v0 into the storage
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param fh: MPI file handler
        /// \param disp: number of bytes from the beginning of the file before the coordinate zero of this block
        /// \param co: coordinate linearization order
        ///
        /// NOTE: the current file position should be at the beginning of the "values" section
        /// where v0 is going to be written.

        template <typename IndexType, std::size_t Nd0, std::size_t Nd1, typename T, typename Q,
                  typename XPU0, typename Comm>
        void local_save(typename elem<T>::type alpha, Order<Nd0> o0, Coor<Nd0> from0,
                        Coor<Nd0> size0, Coor<Nd0> dim0, vector<const T, XPU0> v0,
                        const Order<Nd1> &o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                        Storage_context<Nd1, Comm> &sto, std::size_t blockIndex, CoorOrder co,
                        bool do_change_endianness) {

            tracker<XPU0> _t("local save", v0.ctx());

            // Shortcut for an empty range
            std::size_t vol = volume(size0);
            if (vol == 0) return;

            // Make agree in ordering source and destination
            if (co != SlowToFast) {
                o0 = reverse(o0);
                from0 = reverse(from0);
                size0 = reverse(size0);
                dim0 = reverse(dim0);
                co = SlowToFast;
            }

            // Write the values of v0 contiguously
            Cpu cpu = v0.ctx().toCpu();
            vector<Q, Cpu> v0_host(vol, cpu);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, find_permutation<Nd0, Nd1>(o0, o1), 1);
            local_copy<Nd0, Nd1, T, Q>(alpha, o0, from0, size0, dim0, v0, {}, o1, {{}}, size1,
                                       v0_host, {}, EWOp::Copy{}, co);

            // Change endianness
            if (do_change_endianness) change_endianness(v0_host.data(), v0_host.size());

            // Do the saving
            _t.memops = (double)vol * sizeof(Q);
            std::size_t disp = sto.disp_values[blockIndex];
            auto t = get_normalize_permutation<IndexType>(from1, size1, dim1, co, cpu);
            auto disp1 = std::get<0>(t);
            auto blk1 = std::get<1>(t);
            IndicesT<IndexType, Cpu> indices1 = std::get<2>(t);
            for (std::size_t i = 0; i < indices1.size(); ++i) {
                seek(sto.fh,
                     disp + (disp1 + (indices1.data() == nullptr ? i : indices1[i])) * sizeof(Q));
                iwrite(sto.fh, v0_host.data() + i * blk1, blk1, v0_host);
            }

            // Compute the checksum if the block is going to be completely overwritten
            if (sto.checksum == BlockChecksum && from1 == Coor<Nd1>{{}} && size1 == dim1) {
                double checksum =
                    do_checksum(v0_host.data(), v0_host.size(), sto.checksum_blocksize);
                if (do_change_endianness) change_endianness(&checksum, 1);
                seek(sto.fh, sto.disp_checksum[blockIndex]);
                iwrite(sto.fh, &checksum, 1);
            }
        }

        /// Copy the content of tensor v0 into the storage
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param fh: MPI file handler
        /// \param disp: number of bytes from the beginning of the file before the coordinate zero of this block
        /// \param co: coordinate linearization order
        ///
        /// NOTE: the current file position should be at the beginning of the "values" section
        /// where v0 is going to be written.

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename Comm>
        void local_save(typename elem<T>::type alpha, Order<Nd0> o0, Coor<Nd0> from0,
                        Coor<Nd0> size0, Coor<Nd0> dim0, vector<const T, XPU0> v0,
                        const Order<Nd1> &o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
                        Storage_context<Nd1, Comm> &sto, std::size_t blockIndex, CoorOrder co,
                        bool do_change_endianness) {

            if (std::max(volume(dim0), volume(dim1)) >=
                (std::size_t)std::numeric_limits<IndexType>::max()) {
                local_save<std::size_t, Nd0, Nd1, T, Q>(alpha, o0, from0, size0, dim0, v0, o1,
                                                        from1, dim1, sto, blockIndex, co,
                                                        do_change_endianness);
            } else {
                local_save<IndexType, Nd0, Nd1, T, Q>(alpha, o0, from0, size0, dim0, v0, o1, from1,
                                                      dim1, sto, blockIndex, co,
                                                      do_change_endianness);
            }
        }

        /// Copy from a storage into the tensor v1
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param fh: file handler
        /// \param disp: number of bytes from the beginning of the file before the coordinate zero of this block
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <typename IndexType, std::size_t Nd0, std::size_t Nd1, typename T, typename Q,
                  typename XPU1, typename EWOP, typename FileT>
        void local_load(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, FileT &fh, std::size_t disp,
                        Order<Nd1> o1, Coor<Nd1> from1, Coor<Nd1> dim1, vector<Q, XPU1> v1, EWOP,
                        CoorOrder co, bool do_change_endianness) {

            tracker<XPU1> _t("local load", v1.ctx());

            // Shortcut for an empty range
            std::size_t vol = volume(size0);
            if (vol == 0) return;

            // Make agree in ordering source and destination
            if (co != SlowToFast) {
                o1 = reverse(o1);
                from1 = reverse(from1);
                dim1 = reverse(dim1);
                co = SlowToFast;
            }

            // Do the reading
            _t.memops = (double)vol * sizeof(T);
            Cpu cpu = v1.ctx().toCpu();
            vector<T, Cpu> v0(vol, cpu);
            auto t = get_normalize_permutation<IndexType>(from0, size0, dim0, co, cpu);
            auto disp0 = std::get<0>(t);
            auto blk0 = std::get<1>(t);
            IndicesT<IndexType, Cpu> indices0 = std::get<2>(t);
            for (std::size_t i = 0; i < indices0.size(); ++i) {
                seek(fh,
                     disp + (disp0 + (indices0.data() == nullptr ? i : indices0[i])) * sizeof(T));
                read(fh, v0.data() + i * blk0, blk0);
            }

            // Change endianness
            if (do_change_endianness) change_endianness(v0.data(), v0.size());

            // Write the values of v0 into v1
            local_copy<Nd0, Nd1, T, Q>(alpha, o0, {{}}, size0, size0, (vector<const T, Cpu>)v0, {},
                                       o1, from1, dim1, v1, {}, EWOp::Copy{}, co);
        }

        /// Copy from a storage into the tensor v1
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param fh: file handler
        /// \param disp: number of bytes from the beginning of the file before the coordinate zero of this block
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU1,
                  typename EWOP, typename FileT>
        void local_load(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, FileT &fh, std::size_t disp,
                        Order<Nd1> o1, Coor<Nd1> from1, Coor<Nd1> dim1, vector<Q, XPU1> v1, EWOP,
                        CoorOrder co, bool do_change_endianness) {

            if (std::max(volume(dim0), volume(dim1)) >=
                (std::size_t)std::numeric_limits<IndexType>::max()) {
                local_load<std::size_t, Nd0, Nd1, T, Q>(alpha, o0, from0, size0, dim0, fh, disp, o1,
                                                        from1, dim1, v1, EWOP{}, co,
                                                        do_change_endianness);
            } else {

                local_load<IndexType, Nd0, Nd1, T, Q>(alpha, o0, from0, size0, dim0, fh, disp, o1,
                                                      from1, dim1, v1, EWOP{}, co,
                                                      do_change_endianness);
            }
        }

        /// Copy the content of plural tensor v0 into a storage
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param o1: dimension labels for the storage
        /// \param sto: storage context
        /// \param comm: communicator context
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1>
        void save(typename elem<T>::type alpha, const Proc_ranges<Nd0> &p0, const Coor<Nd0> &from0,
                  const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const Order<Nd0> &o0,
                  const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0, Order<Nd1> o1,
                  Storage_context<Nd1, Comm> &sto, Coor<Nd1> from1, CoorOrder co,
                  const Comm &comm) {

            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                for (const auto &i : v0.first) sync(i.it.ctx());
                for (const auto &i : v0.second) sync(i.it.ctx());
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("save"), alpha, p0, from0, size0,
                                                  dim0, o0, o1, from1, co,
                                                  typeid(tag_type).hash_code()),
                                  comm);
            }

            tracker<XPU1> _t("save", Cpu{});

            // Turn o1 and from1 into SlowToFast
            if (co == FastToSlow) {
                o1 = reverse(o1);
                from1 = reverse(from1);
            }

            // Compute the local ranges to save and all the ranges that going to be modified (for
            // tracking checksums). Also, avoid independent processes to write on the same chunk
            // by removing overlaps over the given ranges to write
            unsigned int num_components = p0[comm.rank].size();
            std::vector<std::vector<Op<Nd0, Nd1>>> overlaps(num_components); ///< [componentId][ops]
            From_size<Nd0> ranges_to_save;
            ranges_to_save.reserve(num_components);
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                // We visit all the ranks only when block checksum; otherwise we visit comm.rank
                if (sto.checksum != BlockChecksum && rank != comm.rank) continue;

                for (unsigned int componentId = 0, num_components = p0[rank].size();
                     componentId < num_components; ++componentId) {
                    if (componentId == 0) ranges_to_save.resize(0);

                    // Remove overlaps with ranges on the same process
                    From_size<Nd0> ranges =
                        remove_repetitions(p0[rank][componentId][0], p0[rank][componentId][1],
                                           ranges_to_save.data(), ranges_to_save.size(), dim0);

                    // Remove overlaps with ranges in smaller ranks
                    for (unsigned int r = 0; r < rank; ++r)
                        ranges = remove_repetitions(ranges, p0[r].data(), p0[r].size(), dim0);

                    // Keep all the ranges to write on the same process
                    ranges_to_save.insert(ranges_to_save.end(), ranges.begin(), ranges.end());

                    // Generate the list of subranges to send from each component from v0 to v1
                    auto ranges_overlaps =
                        get_overlap_ranges(dim0, ranges, o0, from0, size0, sto.blocks, o1, from1);

                    // Mark whether the chunks are going to be completely overwritten, to
                    // track what checksums are going to be computed on the fly by `local_save`
                    if (sto.checksum == BlockChecksum) {
                        for (auto &ranges_overlaps_it : ranges_overlaps)
                            for (auto &op : ranges_overlaps_it)
                                sto.is_checksum_done[op.blockIndex] =
                                    (op.second_subtensor[0] == Coor<Nd1>{{}} &&
                                             op.second_tensor[1] == op.second_subtensor[1]
                                         ? 1
                                         : 0);
                    }

                    if (rank == comm.rank) {
                        // Translate the given ranges to their origin
                        for (auto &ranges_overlaps_it : ranges_overlaps)
                            for (auto &op : ranges_overlaps_it)
                                op.first_subtensor[0] =
                                    normalize_coor(op.first_subtensor[0] + op.first_tensor[0] -
                                                       p0[comm.rank][componentId][0],
                                                   dim0);

                        // Annotate the calls to `local_save`
                        for (const auto &ranges_overlaps_it : ranges_overlaps)
                            overlaps[componentId].insert(overlaps[componentId].end(),
                                                         ranges_overlaps_it.begin(),
                                                         ranges_overlaps_it.end());
                    }
                }
            }

            // Do the local file modifications
            for (const Component<Nd0, const T, XPU0> &c0 : v0.first) {
                for (const auto &o : overlaps[c0.componentId]) {
                    assert(check_equivalence(o0, o.first_subtensor[1], o1, o.second_subtensor[1]));
                    local_save<Nd0, Nd1, T, Q>(alpha, o0, o.first_subtensor[0],
                                               o.first_subtensor[1], c0.dim, c0.it, o1,
                                               o.second_subtensor[0], o.second_tensor[1], sto,
                                               o.blockIndex, co, sto.change_endianness);
                }
            }
            for (const Component<Nd0, const T, XPU1> &c0 : v0.second) {
                for (const auto &o : overlaps[c0.componentId]) {
                    assert(check_equivalence(o0, o.first_subtensor[1], o1, o.second_subtensor[1]));
                    local_save<Nd0, Nd1, T, Q>(alpha, o0, o.first_subtensor[0],
                                               o.first_subtensor[1], c0.dim, c0.it, o1,
                                               o.second_subtensor[0], o.second_tensor[1], sto,
                                               o.blockIndex, co, sto.change_endianness);
                }
            }

            // Mark the storage as modified
            sto.modified_for_flush = sto.modified_for_checksum = true;

            // Release resources on finished requests
            check_pending_requests(sto.fh);
        }

        /// Copy a range from a storage into a plural tensor v0
        /// \param alpha: factor on the copy
        /// \param sto: storage context
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o0: dimension labels for the origin tensor
        /// \param p1: partitioning of the origin tensor in consecutive ranges
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param o1: dimension labels for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename Comm,
                  typename XPU0, typename XPU1, typename EWOP>
        void load(typename elem<T>::type alpha, Storage_context<Nd0, Comm> &sto, Coor<Nd0> from0,
                  Coor<Nd0> size0, Order<Nd0> o0, const From_size<Nd1> &p1, const Coor<Nd1> &from1,
                  const Coor<Nd1> &dim1, const Order<Nd1> &o1,
                  const Components_tmpl<Nd1, Q, XPU0, XPU1> &v1, EWOP, CoorOrder co,
                  const Comm &comm) {

            tracker<XPU1> _t("load", Cpu{});

            // Turn o0, from0, and size0 into SlowToFast
            if (co == FastToSlow) {
                o0 = reverse(o0);
                from0 = reverse(from0);
                size0 = reverse(size0);
            }

            // Generate the list of subranges to send from each component from v0 to v1
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            auto overlaps = get_overlap_ranges(dim1, p1, o1, from1, size1, sto.blocks, o0, from0);

            // Synchronize the content of the storage before reading from it
            if (sto.modified_for_flush) {
                flush(sto.fh);
                sto.modified_for_flush = false;
            }

            // Do the local file modifications
            for (const Component<Nd1, Q, XPU0> &c1 : v1.first) {
                for (const auto &o : overlaps[c1.componentId]) {
                    assert(check_equivalence(o0, o.second_subtensor[1], o1, o.first_subtensor[1]));
                    local_load<Nd0, Nd1, T, Q>(
                        alpha, o0, o.second_subtensor[0], o.second_subtensor[1], o.second_tensor[1],
                        sto.fh, sto.disp_values[o.blockIndex], o1, o.first_subtensor[0], c1.dim,
                        c1.it, EWOP{}, co, sto.change_endianness);
                }
            }
            for (const Component<Nd1, Q, XPU1> &c1 : v1.second) {
                for (const auto &o : overlaps[c1.componentId]) {
                    local_load<Nd0, Nd1, T, Q>(
                        alpha, o0, o.second_subtensor[0], o.second_subtensor[1], o.second_tensor[1],
                        sto.fh, sto.disp_values[o.blockIndex], o1, o.first_subtensor[0], c1.dim,
                        c1.it, EWOP{}, co, sto.change_endianness);
                }
            }
        }

        /// Return the nonzero blocks stored
        /// \param stoh: handle to a tensor storage
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param size1: size in destination tensor
        /// \param blocks: (output) vector of from-size pairs with the nonzero blocks
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Comm>
        void get_blocks(Storage_context<Nd0, Comm> &sto, Order<Nd0> o0, Order<Nd1> o1,
                        const Coor<Nd1> &from1, const Coor<Nd1> &size1,
                        std::vector<PartitionItem<Nd1>> &blocks, CoorOrder co) {

            tracker<Cpu> _t("get_blocks", Cpu{0});

            // Turn o0 into SlowToFast
            if (co == FastToSlow) o0 = reverse(o0);

            // Generate the list of subranges to send from each component from v0 to v1
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> dim1 = reorder_coor<Nd0, Nd1>(sto.dim, perm0, 1);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            Coor<Nd0> from0 = reorder_coor<Nd1, Nd0>(from1, perm1, 0);
            PartitionItem<Nd1> p1{from1, size1};
            auto overlaps = get_overlap_ranges(dim1, to_vector(&p1, 1, Cpu{}), o1, from1, size1,
                                               sto.blocks, o0, from0);

            for (const auto &o : overlaps[0]) {
                assert(check_equivalence(o0, o.second_subtensor[1], o1, o.first_subtensor[1]));
                blocks.push_back(o.first_subtensor);
            }
        }

        /// Create a file where to store a tensor
        /// \param dim: tensor dimensions
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
        /// \param filename: path and name of the file
        /// \param metadata: metadata content
        /// \param metadata_length: number of characters in the metadata
        /// \param checksum: checksum level
        /// \param stoh (out) handle to a tensor storage
        ///
        /// If the file exists, its content will be lost

        template <std::size_t Nd, typename T, typename Comm>
        Storage_context<Nd, Comm> *create_storage(Coor<Nd> dim, CoorOrder co, const char *filename,
                                                  const char *metadata, int metadata_length,
                                                  checksum_type checksum, Comm comm) {

            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("create_storage"), dim, co,
                                                  std::string(filename),
                                                  std::string(metadata, metadata_length),
                                                  (int)checksum, typeid(tag_type).hash_code()),
                                  comm);
            }

            if (co == FastToSlow) dim = detail::reverse(dim);

            // Check that int has a size of 4
            if (sizeof(int) != 4) throw std::runtime_error("Expected int to have size 4");

            // Create file
            FileHandler<Comm> fh = file_open(comm, filename, CreateForReadWrite);

            // Root process writes down header
            std::size_t padding_size = (8 - metadata_length % 8) % 8;
            std::size_t header_size =
                sizeof(int) * 6 + metadata_length + padding_size + sizeof(double) * (Nd + 1);
            checksum_t checksum_val = 0;

            if (comm.rank == 0) {
                // Write magic_number
                int i32 = magic_number;
                write(fh, &i32, 1);
                checksum_val = do_checksum(&i32, 1, 0, checksum_val);

                // Write version
                i32 = 0;
                write(fh, &i32, 1);
                checksum_val = do_checksum(&i32, 1, 0, checksum_val);

                // Write values_datatype
                i32 = get_values_datatype<T>();
                write(fh, &i32, 1);
                checksum_val = do_checksum(&i32, 1, 0, checksum_val);

                // Write checksum
                i32 = checksum;
                write(fh, &i32, 1);
                checksum_val = do_checksum(&i32, 1, 0, checksum_val);

                // Write number of dimensions
                i32 = Nd;
                write(fh, &i32, 1);
                checksum_val = do_checksum(&i32, 1, 0, checksum_val);

                // Write metadata
                write(fh, &metadata_length, 1);
                checksum_val = do_checksum(&metadata_length, 1, 0, checksum_val);
                write(fh, metadata, metadata_length);
                checksum_val = do_checksum(metadata, metadata_length, 0, checksum_val);

                // Write padding
                std::vector<char> padding(padding_size);
                write(fh, padding.data(), padding.size());
                checksum_val = do_checksum(padding.data(), padding_size, 0, checksum_val);

                // Write tensor size
                std::array<double, Nd> dimd;
                std::copy_n(dim.begin(), Nd, dimd.begin());
                write(fh, &dimd[0], Nd);
                checksum_val = do_checksum(&dimd, 1, 0, checksum_val);

                // Write checksum blocksize
                double d = default_checksum_blocksize;
                write(fh, &d, 1);
                checksum_val = do_checksum(&d, 1, 0, checksum_val);

                // Write num_chunks
                d = 0;
                write(fh, &d, 1);
            }

            // Create the handler
            return new Storage_context<Nd, Comm>{get_values_datatype<T>(),
                                                 header_size,
                                                 fh,
                                                 dim,
                                                 false /* don't change endianness */,
                                                 true /* new storage */,
                                                 checksum,
                                                 default_checksum_blocksize,
                                                 checksum_val,
                                                 true /* allow writing */};
        }

        /// Read fields in the header of a storage
        /// \param filename: path and name of the file
        /// \param allow_writing: whether to allow writing
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
        /// \param values_dtype: (out) type of the values
        /// \param metadata: (out) metadata content
        /// \param size: (out)tensor dimensions
        /// \param header_size: (out) number of bytes before the first chunk
        /// \param do_change_endianness: (out) whether to change endianness
        /// \param checksum: (out) checksum type
        /// \param checksum_blocksize: (out) blocksize use by compute_checksum
        /// \param checksum_val: (out) checksum of the data up to num_chunks
        /// \param fh: (out) file handler

        template <typename Comm>
        void open_storage(const char *filename, bool allow_writing, CoorOrder co,
                          values_datatype &values_dtype, std::vector<char> &metadata,
                          std::vector<IndexType> &size, std::size_t &header_size,
                          bool &do_change_endianness, checksum_type &checksum,
                          std::size_t &checksum_blocksize, checksum_t &checksum_val, Comm comm,
                          FileHandler<Comm> &fh) {

            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("open_storage"), allow_writing, co,
                                                  typeid(tag_type).hash_code()),
                                  comm);
            }

            // Check that int has a size of 4
            if (sizeof(int) != 4) throw std::runtime_error("Expected int to have size 4");

            // Open the existing file for reading and writing
            fh = file_open(comm, filename, allow_writing ? ReadWrite : OnlyRead);

            // Read magic_number and check Endianness
            do_change_endianness = false;
            int i32;
            read(fh, &i32, 1);
            if (i32 != magic_number) {
                change_endianness(&i32, 1);
                if (i32 != magic_number) {
                    throw std::runtime_error("Unexpected value for the magic number; the file may "
                                             "not be a tensor storage");
                }
                do_change_endianness = true;
            }

            // Read version
            read(fh, &i32, 1);
            if (do_change_endianness) change_endianness(&i32, 1);
            if (i32 != 0)
                throw std::runtime_error(
                    "Unsupported version of the tensor format; try a newer version of supperbblas");

            // Read values_datatype
            read(fh, &i32, 1);
            if (do_change_endianness) change_endianness(&i32, 1);
            values_dtype = (values_datatype)i32;

            // Read checksum_type
            read(fh, &i32, 1);
            if (do_change_endianness) change_endianness(&i32, 1);
            if (i32 < 0 || i32 > 2) throw std::runtime_error("Unsupported checksum type");
            checksum = (checksum_type)i32;

            // Read the number of dimensions
            int Nd = 0;
            if (do_change_endianness) change_endianness(&Nd, 1);
            read(fh, &Nd, 1);

            // Read metadata
            int metadata_length = 0;
            read(fh, &metadata_length, 1);
            if (do_change_endianness) change_endianness(&metadata_length, 1);
            metadata.resize(metadata_length);
            read(fh, metadata.data(), metadata_length);

            // Read padding
            std::vector<char> padding((8 - metadata_length % 8) % 8);
            read(fh, padding.data(), padding.size());

            // Read tensor size
            std::vector<double> dimd(Nd);
            read(fh, &dimd[0], Nd);
            if (do_change_endianness) change_endianness(&dimd[0], Nd);
            size.resize(Nd);
            std::copy_n(dimd.begin(), Nd, size.begin());
            if (co == FastToSlow) std::reverse(size.begin(), size.end());

            // Read checksum_block
            double d;
            read(fh, &d, 1);
            if (do_change_endianness) change_endianness(&d, 1);
            checksum_blocksize = d;

            // Compute total header size
            header_size =
                sizeof(int) * 6 + metadata_length + padding.size() + sizeof(double) * (Nd + 1);

            // Re-read again the header and compute the checksum
            if (checksum == BlockChecksum) {
                std::vector<char> header(header_size);
                seek(fh, 0);
                read(fh, header.data(), header_size);
                checksum_val = do_checksum(header.data(), header_size);
            } else {
                checksum_val = 0;
            }
        }

        /// Return the ranges involved in the source and destination tensors
        /// \param dim0: dimension size for the origin tensor
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param dim1: dimension size for the destination tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied

        template <std::size_t Nd0, std::size_t Nd1, typename From_size0>
        std::vector<From_size_item<Nd1>>
        translate_ranges(Coor<Nd0> dim0, const From_size0 &p0, const Order<Nd0> &o0,
                         const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd1> dim1,
                         const Order<Nd1> &o1, const Coor<Nd1> &from1) {

            Cpu cpu{0};
            tracker<Cpu> _t("comp. tensor overlaps on storage", cpu);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            std::vector<From_size_item<Nd1>> r(p0.size());
            for (unsigned int i = 0; i < p0.size(); ++i) {
                // Restrict the local range in v0 to the range from0, size0
                Coor<Nd0> rlocal_from0, rlocal_size0;
                intersection<Nd0>(from0, size0, p0[i][0], p0[i][1], dim0, rlocal_from0,
                                  rlocal_size0);

                // Translate the restricted range to the destination lattice
                translate_range(rlocal_from0, rlocal_size0, from0, dim0, from1, dim1, perm0,
                                r[i][0], r[i][1]);
            }

            return r;
        }

        /// Add blocks to storage after restricted the range indicated by from0, size0, and from1
        /// \param p0: blocks to add
        /// \param num_blocks: number of items in p0
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate of the range to consider on p0
        /// \param size0: number of elements to consider in each dimension
        /// \param o1: dimension labels for the storage
        /// \param sto: storage context
        /// \param from1: first coordinate of the ranges to add
        /// \param comm: communicator context
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename Q, typename Comm>
        void append_blocks(const PartitionItem<Nd0> *p0, std::size_t num_blocks,
                           const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                           const Order<Nd0> &o0, Order<Nd1> o1, Storage_context<Nd1, Comm> &sto,
                           Coor<Nd1> from1, Comm comm, CoorOrder co) {

            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                auto p0_ = to_vector(p0, num_blocks, Cpu{0});
                check_consistency(std::make_tuple(std::string("append_blocks"), p0_, from0, size0,
                                                  dim0, o0, o1, from1, co,
                                                  typeid(tag_type).hash_code()),
                                  comm);
            }

            tracker<Cpu> _t("append blocks", Cpu{0});

            // Generate the list of subranges to add
            auto p0_ = to_vector(p0, num_blocks, Cpu{0});
            if (co == FastToSlow) {
                o1 = reverse(o1);
                from1 = reverse(from1);
            }
            auto p = translate_ranges(dim0, p0_, o0, from0, size0, sto.dim, o1, from1);

            // Write the coordinates for all non-empty blocks
            std::vector<From_size_item<Nd1>> new_blocks;
            new_blocks.reserve(num_blocks);

            std::vector<std::size_t> num_values; ///< number of values for each block
            num_values.reserve(num_blocks);
            std::vector<double> chunk_header(1); ///< header of the chunk
            chunk_header.reserve(1 + num_blocks * (Nd1 * 2 + 1));
            for (std::size_t i = 0; i < num_blocks; ++i) {
                // Remove the overlaps with ranges already stored and on the same block
                std::vector<PartitionItem<Nd1>> sto_overlaps;
                for (const auto &pair_fs_key : sto.blocks.intersection(p[i][0], p[i][1]))
                    sto_overlaps.insert(sto_overlaps.end(), pair_fs_key.first.begin(),
                                        pair_fs_key.first.end());
                auto fs0 = remove_repetitions(p[i][0], p[i][1], sto_overlaps.data(),
                                              sto_overlaps.size(), sto.dim);
                auto fs1 = remove_repetitions(fs0, new_blocks.data(), new_blocks.size(), sto.dim);

                for (const auto &fs : fs1) {
                    num_values.push_back(volume(fs[1]));
                    new_blocks.push_back(fs);
                }

                // Root process writes the "from" and "size" for each block
                if (comm.rank == 0) {
                    for (const auto &fs : fs1) {
                        chunk_header.insert(chunk_header.end(), fs[0].begin(), fs[0].end());
                        chunk_header.insert(chunk_header.end(), fs[1].begin(), fs[1].end());
                    }
                }
            }

            // If no new block, get out
            if (new_blocks.size() == 0) return;

            // Annotate where the nonzero values start for the new blocks
            std::size_t values_start =
                sto.disp + sizeof(double) + new_blocks.size() * Nd1 * sizeof(double) * 2;
            for (std::size_t i = 0; i < new_blocks.size(); ++i) {
                sto.blocks.append_block(new_blocks[i][0], new_blocks[i][1], sto.disp_values.size());
                sto.disp_values.push_back(values_start);
                values_start += num_values[i] * sizeof(Q);
            }

            // Write the number of blocks in this chunk and preallocate for the values
            if (comm.rank == 0) {
                chunk_header[0] = new_blocks.size();
                // Change endianness if needed
                if (sto.change_endianness)
                    change_endianness(chunk_header.data(), chunk_header.size());

                // Compute checksum
                if (sto.checksum == BlockChecksum)
                    sto.checksum_val =
                        do_checksum(chunk_header.data(), chunk_header.size(), 0, sto.checksum_val);

                // Write all the blocks of this chunk
                seek(sto.fh, sto.disp);
                iwrite(sto.fh, chunk_header.data(), chunk_header.size());
            }

            // If using checksum at the level of blocks, add the space for the checksums
            if (sto.checksum == BlockChecksum) {
                for (std::size_t i = 0; i < new_blocks.size(); ++i) {
                    sto.disp_checksum.push_back(values_start);
                    sto.is_checksum_done.push_back(0);
                    values_start += sizeof(double);
                }
            }

            // Update disp
            sto.disp = values_start;

            // Update num_chunks
            sto.num_chunks++;
            if (comm.rank == 0) {
                double num_chunks = sto.num_chunks;
                if (sto.change_endianness) change_endianness(&num_chunks, 1);
                seek(sto.fh, sto.header_size);
                iwrite(sto.fh, &num_chunks, 1);
            }

            // Mark the storage as modified
            sto.modified_for_flush = sto.modified_for_checksum = true;
        }

        /// Read all blocks from storage
        /// \param stoh: handle to a tensor storage

        template <std::size_t Nd1, typename Q, typename Comm>
        void read_all_blocks(Storage_context<Nd1, Comm> &sto) {

            // Read num_chunks
            double num_chunks = 0;
            std::size_t cur = sto.header_size;
            seek(sto.fh, cur);
            read(sto.fh, &num_chunks, 1);
            cur += sizeof(double);
            if (sto.change_endianness) change_endianness(&num_chunks, 1);
            sto.num_chunks = num_chunks;

            // Read chunks
            for (std::size_t chunk = 0; chunk < sto.num_chunks; chunk++) {
                // Read the number of blocks in this chunk
                double d;
                read(sto.fh, &d, 1);
                if (sto.checksum == BlockChecksum)
                    sto.checksum_val = do_checksum(&d, 1, 0, sto.checksum_val);
                if (sto.change_endianness) change_endianness(&d, 1);
                std::size_t num_blocks = d;

                // Read blocks
                std::vector<std::size_t> num_values;     ///< number of values for each block
                std::vector<From_size_item<Nd1>> blocks; ///< ranges of the blocks
                num_values.reserve(num_blocks);
                blocks.reserve(num_blocks);
                for (std::size_t i = 0; i < num_blocks; ++i) {
                    // Read from and size
                    std::array<double, Nd1> fromd, sized;
                    read(sto.fh, &fromd[0], Nd1);
                    read(sto.fh, &sized[0], Nd1);
                    if (sto.checksum == BlockChecksum)
                        sto.checksum_val = do_checksum(&fromd, 1, 0, sto.checksum_val);
                    if (sto.checksum == BlockChecksum)
                        sto.checksum_val = do_checksum(&sized, 1, 0, sto.checksum_val);
                    if (sto.change_endianness) change_endianness(&fromd[0], Nd1);
                    if (sto.change_endianness) change_endianness(&sized[0], Nd1);
                    Coor<Nd1> from, size;
                    std::copy_n(fromd.begin(), Nd1, from.begin());
                    std::copy_n(sized.begin(), Nd1, size.begin());
                    num_values.push_back(volume(size));
                    blocks.push_back(From_size_item<Nd1>{from, size});
                }

                // Annotate where the nonzero values start for the block
                cur += sizeof(double) + num_blocks * Nd1 * sizeof(double) * 2;
                for (std::size_t i = 0; i < num_blocks; ++i) {
                    sto.blocks.append_block(blocks[i][0], blocks[i][1], sto.disp_values.size());
                    sto.disp_values.push_back(cur);
                    cur += num_values[i] * sizeof(Q);
                }
                if (sto.checksum == BlockChecksum) {
                    for (std::size_t i = 0; i < num_blocks; ++i) {
                        sto.disp_checksum.push_back(cur);
                        sto.is_checksum_done.push_back(1); // mark them as done
                        cur += sizeof(double);
                    }
                }

                // Set the beginning for a new block
                seek(sto.fh, cur);
            }

            // Update disp
            sto.disp = cur;

            // Check the checksum on the headers
            if (sto.checksum != NoChecksum) {
                // Read checksum
                double d;
                read(sto.fh, &d, 1);
                if (sto.change_endianness) change_endianness(&d, 1);

                // Compare the checksum with the one stored on the file
                if (sto.checksum == BlockChecksum && sto.checksum_val != d)
                    throw std::runtime_error("Checksum failed!");

                // Store checksum on sto
                if (sto.checksum == GlobalChecksum) sto.checksum_val = d;
            }
        }

        /// Open a storage for reading and writing
        /// \param filename: path and name of the file
        /// \param allow_writing: whether to allow writing
        /// \param comm: communicator
        ///
        /// NOTE: If the file does not exist, an exception will raise

        template <std::size_t Nd, typename T, typename Comm>
        Storage_context<Nd, Comm> *open_storage_template(const char *filename, bool allow_writing,
                                                         Comm comm) {

            // Open storage and check template parameters
            FileHandler<Comm> fh;
            values_datatype values_dtype;
            std::vector<char> metadata;
            std::vector<IndexType> size;
            std::size_t header_size;
            bool do_change_endianness;
            checksum_type checksum;
            std::size_t checksum_blocksize;
            checksum_t checksum_header;
            open_storage(filename, allow_writing, SlowToFast, values_dtype, metadata, size,
                         header_size, do_change_endianness, checksum, checksum_blocksize,
                         checksum_header, comm, fh);

            if (values_dtype != get_values_datatype<T>())
                throw std::runtime_error(
                    "The template parameter T does not match with the datatype of the storage");
            if (Nd != size.size())
                throw std::runtime_error(
                    "The template parameter Nd does not match with the number of "
                    "dimensions of the storage");
            Coor<Nd> dim;
            std::copy_n(size.begin(), Nd, dim.begin());

            // Create the handler
            Storage_context<Nd, Comm> *sto = new Storage_context<Nd, Comm>{
                values_dtype, header_size,          fh,
                dim,          do_change_endianness, false /* not new storage */,
                checksum,     checksum_blocksize,   checksum_header,
                allow_writing};

            // Read the nonzero blocks
            read_all_blocks<Nd, T, Comm>(*sto);

            // Return handler
            return sto;
        }

#ifdef SUPERBBLAS_USE_MPI
        template <typename T>
        inline void gatherv(const T *sendbuf, std::size_t sendcount, int *counts, T *recvbuf,
                            MpiComm comm) {
            std::size_t recvcount = 0;
            for (unsigned int i = 0; i < comm.nprocs; ++i) recvcount += counts[i];
            if (recvcount * sizeof(T) > (std::size_t)std::numeric_limits<int>::max())
                throw std::runtime_error("Too many elements to gather");
            std::vector<int> recvcounts(comm.nprocs);
            std::vector<int> displs(comm.nprocs);
            for (unsigned int i = 0; i < comm.nprocs; ++i) {
                recvcounts[i] = counts[i] * sizeof(T);
                displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]) * sizeof(T);
            }
            MPI_check(MPI_Gatherv(sendbuf, sendcount * sizeof(T), MPI_CHAR, recvbuf,
                                  recvcounts.data(), displs.data(), MPI_CHAR, 0, comm.comm));
        }
#endif // SUPERBBLAS_USE_MPI

        template <typename T>
        inline void gatherv(const T *sendbuf, std::size_t sendcount, int *counts, T *recvbuf,
                            SelfComm) {
            if (sendcount != (std::size_t)counts[0])
                throw std::runtime_error("gather: Invalid arguments");
            std::copy_n(sendbuf, counts[0], recvbuf);
        }

        /// Compute all checksums in a storage
        /// \param sto: storage

        template <std::size_t Nd, typename T, typename Comm>
        void check_or_write_checksums(Storage_context<Nd, Comm> &sto, Comm comm, bool do_write) {
            // Check that common arguments have the same value in all processes
            if (getDebugLevel() > 0) {
                struct tag_type {}; // For hashing template arguments
                check_consistency(std::make_tuple(std::string("check_or_write_checksums"), do_write,
                                                  typeid(tag_type).hash_code()),
                                  comm);
            }

            // Quick exit
            if (do_write && sto.modified_for_checksum == false) return;

            Cpu cpu{0};
            tracker<Cpu> _t("checksums (closing/checking)", cpu);

            // Synchronize the content of the storage before reading from it
            if (sto.modified_for_flush) {
                flush(sto.fh);
                sto.modified_for_flush = false;
            }

            switch (sto.checksum) {
            case NoChecksum: {
                // Do nothing
                break;
            }

            case GlobalChecksum: {
                // Number of blocks of size sto.checksum_blocksize
                IndexType num_blocks =
                    (sto.disp + sto.checksum_blocksize - 1) / sto.checksum_blocksize;

                // Divide the blocks over all processes
                std::vector<PartitionItem<1>> p =
                    basic_partitioning(Coor<1>{num_blocks}, Coor<1>{IndexType(comm.nprocs)});
                std::size_t first_block_to_process = p[comm.rank][0][0];
                std::size_t num_blocks_to_process = p[comm.rank][1][0];
                std::vector<int> num_blocks_to_process_v;
                num_blocks_to_process_v.reserve(comm.nprocs);
                for (const auto &pi : p) num_blocks_to_process_v.push_back(pi[1][0]);

                // Compute the checksum for each block
                std::vector<unsigned char> buffer(sto.checksum_blocksize);
                std::vector<uint32_t> checksums(num_blocks_to_process);
                for (std::size_t b = 0; b < num_blocks_to_process; ++b) {
                    std::size_t first_byte_to_process =
                        (first_block_to_process + b) * sto.checksum_blocksize;
                    std::size_t num_bytes_to_process =
                        std::min(sto.disp - first_byte_to_process, sto.checksum_blocksize);
                    seek(sto.fh, first_byte_to_process);
                    read(sto.fh, buffer.data(), num_bytes_to_process);
                    checksums[b] = do_checksum(buffer.data(), num_bytes_to_process);
                }

                // Change endianness
                if (sto.change_endianness) change_endianness(checksums.data(), checksums.size());

                // Compute the checksum of the checksums
                std::vector<checksum_t> all_checksums(comm.rank == 0 ? num_blocks : 0);
                gatherv(checksums.data(), checksums.size(), num_blocks_to_process_v.data(),
                        all_checksums.data(), comm);

                if (comm.rank == 0) {
                    checksum_t checksum = do_checksum(all_checksums.data(), all_checksums.size());
                    if (do_write) {
                        // Write the checksum
                        sto.checksum_val = checksum;
                        double global_checksum = checksum;
                        if (sto.change_endianness) change_endianness(&global_checksum, 1);
                        seek(sto.fh, sto.disp);
                        iwrite(sto.fh, &global_checksum, 1);
                    } else {
                        // Check the checksum
                        if (checksum != sto.checksum_val)
                            throw std::runtime_error("Checksum failed");
                    }
                }

                if (do_write) sto.modified_for_flush = true;
                break;
            }

            case BlockChecksum: {
                // Divide the blocks among the processes
                std::size_t num_blocks = sto.disp_values.size();
                std::size_t first_block_to_process =
                    num_blocks / comm.nprocs * comm.rank +
                    std::min((std::size_t)comm.rank, num_blocks % (std::size_t)comm.nprocs);
                std::size_t num_blocks_to_process =
                    num_blocks / comm.nprocs +
                    ((num_blocks % (std::size_t)comm.nprocs) > (std::size_t)comm.rank ? 1u : 0);

                // Compute the checksum for the blocks that haven't done yet if do_write, or
                // compute the checksum for every block otherwise
                std::vector<T> buffer;
                for (std::size_t b = 0, blockIndex = first_block_to_process;
                     b < num_blocks_to_process; ++b, ++blockIndex) {

                    // Skip if the checksum is already computed and we want to write the checksums
                    if (do_write && sto.is_checksum_done[blockIndex] != 0) continue;

                    // Compute the checksum of the block
                    std::size_t vol = volume(sto.blocks.blocks[blockIndex][1]);
                    if (vol == 0) continue;
                    buffer.resize(vol);
                    seek(sto.fh, sto.disp_values[blockIndex]);
                    read(sto.fh, buffer.data(), vol);
                    double checksum =
                        do_checksum(buffer.data(), buffer.size(), sto.checksum_blocksize);

                    if (do_write) {
                        // Write the checksum for the block
                        if (sto.change_endianness) change_endianness(&checksum, 1);
                        seek(sto.fh, sto.disp_checksum[blockIndex]);
                        iwrite(sto.fh, &checksum, 1);
                    } else {
                        // Read the checksum for the block
                        double checksum_on_disk = -1;
                        seek(sto.fh, sto.disp_checksum[blockIndex]);
                        read(sto.fh, &checksum_on_disk, 1);
                        if (sto.change_endianness) change_endianness(&checksum_on_disk, 1);

                        // Compare checksums
                        if (checksum != checksum_on_disk)
                            throw std::runtime_error(
                                std::string("Checksum failed: block checksum failed on block ") +
                                std::to_string(blockIndex) + std::string(" : checksum ") +
                                std::to_string(checksum_on_disk) + std::string(" expected ") +
                                std::to_string(checksum));
                    }
                }

                if (comm.rank == 0) {
                    if (do_write) {
                        // Write the checksum of headers
                        double checksum_headers = sto.checksum_val;
                        if (sto.change_endianness) change_endianness(&checksum_headers, 1);
                        seek(sto.fh, sto.disp);
                        iwrite(sto.fh, &checksum_headers, 1);
                    } else {
                        // Check the checksum of headers
                        double checksum_headers;
                        seek(sto.fh, sto.disp);
                        read(sto.fh, &checksum_headers, 1);
                        if (sto.change_endianness) change_endianness(&checksum_headers, 1);
                        if (checksum_headers != sto.checksum_val)
                            throw std::runtime_error(
                                "Checksum failed: header checksum failed (postcheck)");
                    }
                }

                if (do_write) sto.modified_for_flush = true;
                break;
            }
            }

            // Note that all checksums have been stored
            if (do_write) sto.modified_for_checksum = false;
        }
    }

    using Storage_handle = detail::Storage_context_abstract *;

#ifdef SUPERBBLAS_USE_MPI
    /// Create a file where to store a tensor
    /// \param dim: tensor dimensions
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param filename: path and name of the file
    /// \param metadata: metadata content
    /// \param metadata_length: number of characters in the metadata
    /// \param checksum: checksum level (NoChecksum: no checksum; GlobalChecksum: checksum of the entire file;
    ///                  BlockChecksum: checksum on each data block)
    /// \param stoh (out) handle to a tensor storage
    ///
    /// If the file exists, its content will be lost

    template <std::size_t Nd, typename T>
    void create_storage(const Coor<Nd> &dim, CoorOrder co, const char *filename,
                        const char *metadata, int metadata_length, checksum_type checksum,
                        MPI_Comm mpicomm, Storage_handle *stoh) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        *stoh = detail::create_storage<Nd, T>(dim, co, filename, metadata, metadata_length,
                                              checksum, comm);
    }

    /// Read fields in the header of a storage
    /// \param filename: path and name of the file
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param values_type: (out) type of the values, 0 is float, 1 is double, 2 is complex float,
    ///        3: complex double
    /// \param metadata: (out) metadata content
    /// \param dim: (out) tensor dimensions

    inline void read_storage_header(const char *filename, CoorOrder co,
                                    values_datatype &values_dtype, std::vector<char> &metadata,
                                    std::vector<IndexType> &size, MPI_Comm mpicomm) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::FileHandler<detail::MpiComm> fh;
        std::size_t header_size;
        bool do_change_endianness;
        checksum_type checksum;
        std::size_t checksum_blocksize;
        detail::checksum_t checksum_header;
        detail::open_storage(filename, false, co, values_dtype, metadata, size, header_size,
                             do_change_endianness, checksum, checksum_blocksize, checksum_header,
                             comm, fh);
        detail::close(fh);
    }

    /// Open a storage for reading and writing
    /// \param filename: path and name of the file
    /// \param allow_writing: whether to allow writing
    /// \param stoh (out) handle to a tensor storage
    ///
    /// NOTE: If the file does not exist, an exception will raise

    template <std::size_t Nd, typename T>
    void open_storage(const char *filename, bool allow_writing, MPI_Comm mpicomm,
                      Storage_handle *stoh) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        // Open storage and check template parameters
        *stoh = detail::open_storage_template<Nd, T>(filename, allow_writing, comm);
    }

    /// Add blocks to storage
    /// \param p: partitioning of the origin tensor in consecutive ranges
    /// \param num_blocks: number of items in p
    /// \param stoh: handle to a tensor storage
    /// \param mpicomm: MPI communicator context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd1> *p, int num_blocks, const Coor<Nd1> &dim,
                       Storage_handle stoh, MPI_Comm mpicomm, CoorOrder co) {

        detail::Storage_context<Nd1, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::append_blocks<Nd1, Nd1, Q>(p, num_blocks, {{}}, dim, dim,
                                           detail::trivial_order<Nd1>(),
                                           detail::trivial_order<Nd1>(), sto, {{}}, comm, co);
    }

    /// Add blocks to storage
    /// \param p0: blocks to add
    /// \param num_blocks: number of items in p0
    /// \param o0: dimension labels for the blocks to add
    /// \param from0: first coordinate of the range to consider on p0
    /// \param size0: number of elements to copy in each dimension
    /// \param o1: dimension labels for the storage
    /// \param from1: first coordinate of the ranges to add
    /// \param stoh: handle to a tensor storage
    /// \param mpicomm: MPI communicator context
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd0> *p0, int num_blocks, const char *o0,
                       const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                       const char *o1, const Coor<Nd1> &from1, Storage_handle stoh,
                       MPI_Comm mpicomm, CoorOrder co) {

        detail::Storage_context<Nd1, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::append_blocks<Nd0, Nd1, Q>(p0, num_blocks, from0, size0, dim0,
                                           detail::toArray<Nd0>(o0, "o0"),
                                           detail::toArray<Nd1>(o1, "o1"), sto, from1, comm, co);
    }

    /// Copy the content of plural tensor v0 into a storage
    /// \param alpha: factor applied to v0
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param mpicomm: MPI communicator context
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param o1: dimension labels for the storage
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param stoh: handle to a tensor storage
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void save(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
              const T **v0, const Context *ctx0, const char *o1, const Coor<Nd1> &from1,
              Storage_handle stoh, MPI_Comm mpicomm, CoorOrder co, Session session = 0) {

        detail::Storage_context<Nd1, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::save<Nd0, Nd1, T, Q>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, comm, dim0), from0, size0,
            dim0, detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, nullptr, ctx0, ncomponents0, p0, comm, session),
            detail::toArray<Nd1>(o1, "o1"), sto, from1, co, comm);
    }

    /// Copy from a storage into a plural tensor v1
    /// \param alpha: factor applied to v0
    /// \param stoh: handle to a tensor storage
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void load(typename elem<T>::type alpha, Storage_handle stoh, const char *o0,
              const Coor<Nd0> &from0, const Coor<Nd0> &size0, const PartitionItem<Nd1> *p1,
              int ncomponents1, const char *o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
              Q **v1, const Context *ctx1, MPI_Comm mpicomm, CoorOrder co, CopyAdd copyadd,
              Session session = 0) {

        detail::Storage_context<Nd0, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd0, T, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);
        const auto &o0_array = detail::toArray<Nd0>(o0, "o0");
        const auto &o1_array = detail::toArray<Nd1>(o1, "o1");
        const auto &p1_proc_ranges =
            detail::get_from_size(p1, ncomponents1 * comm.nprocs, comm, dim1);
        const auto &v1_c =
            detail::get_components<Nd1>(v1, nullptr, ctx1, ncomponents1, p1, comm, session);

        // Check that common arguments have the same value in all processes
        if (getDebugLevel() > 0) {
            for (const auto &i : v1_c.first) detail::sync(i.it.ctx());
            for (const auto &i : v1_c.second) detail::sync(i.it.ctx());
            struct tag_type {}; // For hashing template arguments
            detail::check_consistency(std::make_tuple(std::string("load"), alpha, from0, size0,
                                                      o0_array, p1_proc_ranges, from1, dim1,
                                                      o1_array, copyadd, co,
                                                      typeid(tag_type).hash_code()),
                                      comm);
        }

        if (copyadd == Copy)
            detail::load<Nd0, Nd1, T, Q>(alpha, sto, from0, size0, o0_array,
                                         p1_proc_ranges.at(comm.rank), from1, dim1, o1_array, v1_c,
                                         detail::EWOp::Copy{}, co, comm);
        else
            detail::load<Nd0, Nd1, T, Q>(alpha, sto, from0, size0, o0_array,
                                         p1_proc_ranges.at(comm.rank), from1, dim1, o1_array, v1_c,
                                         detail::EWOp::Add{}, co, comm);
    }

    /// Return the nonzero blocks stored
    /// \param stoh: handle to a tensor storage
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param o0: dimension labels for the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param o1: dimension labels for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param size1: size in destination tensor
    /// \param blocks: (output) vector of from-size pairs with the nonzero blocks
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd0, std::size_t Nd1, typename T>
    void get_blocks(Storage_handle stoh, const char *o0, const char *o1, const Coor<Nd1> from1,
                    const Coor<Nd1> size1, std::vector<PartitionItem<Nd1>> &blocks,
                    MPI_Comm mpicomm, CoorOrder co) {

        (void)mpicomm;
        detail::Storage_context<Nd0, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd0, T, detail::MpiComm>(stoh);

        detail::get_blocks<Nd0, Nd1, T>(sto, detail::toArray<Nd0>(o0, "o0"),
                                        detail::toArray<Nd1>(o1, "o1"), from1, size1, blocks, co);
    }

    /// Check the checksums in storage
    /// \param stoh: handle to a tensor storage
    /// \param mpicomm: MPI communicator context

    template <std::size_t Nd1, typename Q>
    void check_storage(Storage_handle stoh, MPI_Comm mpicomm) {

        detail::Storage_context<Nd1, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::check_or_write_checksums<Nd1, Q>(sto, comm, false /* don't write checksums */);
    }

    /// Close storage
    /// \param stoh: handle to a tensor storage
    /// \param mpicomm: MPI communicator context

    template <std::size_t Nd1, typename Q>
    void close_storage(Storage_handle stoh, MPI_Comm mpicomm) {

        detail::Storage_context<Nd1, detail::MpiComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::MpiComm>(stoh);
        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::check_or_write_checksums<Nd1, Q>(sto, comm, true /* do write checksums */);

        delete stoh;
    }

#endif // SUPERBBLAS_USE_MPI

    /// Create a file where to store a tensor
    /// \param dim: tensor dimensions
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param filename: path and name of the file
    /// \param metadata: metadata content
    /// \param metadata_length: number of characters in the metadata
    /// \param checksum: checksum level (NoChecksum: no checksum; GlobalChecksum: checksum of the entire file;
    /// BlockChecksum: checksum on each data block)
    /// \param stoh (out) handle to a tensor storage
    ///
    /// If the file exists, its content will be lost

    template <std::size_t Nd, typename T>
    void create_storage(const Coor<Nd> &dim, CoorOrder co, const char *filename,
                        const char *metadata, int metadata_length, checksum_type checksum,
                        Storage_handle *stoh) {

        detail::SelfComm comm = detail::get_comm();

        *stoh = detail::create_storage<Nd, T>(dim, co, filename, metadata, metadata_length,
                                              checksum, comm);
    }

    /// Read fields in the header of a storage
    /// \param filename: path and name of the file
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param values_type: (out) type of the values, 0 is float, 1 is double, 2 is complex float,
    ///        3: complex double
    /// \param metadata: (out) metadata content
    /// \param dim: (out) tensor dimensions

    inline void read_storage_header(const char *filename, CoorOrder co,
                                    values_datatype &values_dtype, std::vector<char> &metadata,
                                    std::vector<IndexType> &size) {

        detail::SelfComm comm = detail::get_comm();

        detail::FileHandler<detail::SelfComm> fh;
        std::size_t header_size;
        bool do_change_endianness;
        checksum_type checksum;
        std::size_t checksum_blocksize;
        detail::checksum_t checksum_header;
        detail::open_storage(filename, false, co, values_dtype, metadata, size, header_size,
                             do_change_endianness, checksum, checksum_blocksize, checksum_header,
                             comm, fh);
        detail::close(fh);
    }

    /// Extend the size of the file
    /// \param stoh:  handle to a tensor storage
    /// \param size: expected file size in bytes

    inline void preallocate_storage(Storage_handle stoh, std::size_t size) {
        stoh->preallocate(size);
    }

    /// Force pending writing to make them visible to other processes
    /// \param stoh:  handle to a tensor storage

    inline void flush_storage(Storage_handle stoh) { stoh->flush(); }

    /// Check the checksums in storage
    /// \param stoh: handle to a tensor storage

    template <std::size_t Nd1, typename Q> void check_storage(Storage_handle stoh) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::check_or_write_checksums<Nd1, Q>(sto, comm, false /* don't write checksums */);
    }

    /// Close storage
    /// \param stoh: handle to a tensor storage

    template <std::size_t Nd1, typename Q> void close_storage(Storage_handle stoh) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::check_or_write_checksums<Nd1, Q>(sto, comm, true /* do write checksums */);

        delete stoh;
    }

    /// Open a storage for reading and writing
    /// \param filename: path and name of the file
    /// \param allow_writing: whether to allow writing
    /// \param stoh (out) handle to a tensor storage
    ///
    /// NOTE: If the file does not exist, an exception will raise

    template <std::size_t Nd, typename T>
    void open_storage(const char *filename, bool allow_writing, Storage_handle *stoh) {

        detail::SelfComm comm = detail::get_comm();

        // Open storage and check template parameters
        *stoh = detail::open_storage_template<Nd, T>(filename, allow_writing, comm);
    }

    /// Add blocks to storage
    /// \param p: partitioning of the origin tensor in consecutive ranges
    /// \param num_blocks: number of items in p
    /// \param stoh: handle to a tensor storage
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd1> *p, int num_blocks, const Coor<Nd1> &dim,
                       Storage_handle stoh, CoorOrder co) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::append_blocks<Nd1, Nd1, Q>(
            p, num_blocks, {{}}, dim, dim, detail::trivial_order<Nd1>(),
            detail::trivial_order<Nd1>(), sto, Coor<Nd1>{{}}, comm, co);
    }

    /// Add blocks to storage
    /// \param p0: blocks to add
    /// \param num_blocks: number of items in p0
    /// \param o0: dimension labels for the blocks to add
    /// \param from0: first coordinate of the range to consider on p0
    /// \param size0: number of elements to copy in each dimension
    /// \param o1: dimension labels for the storage
    /// \param from1: first coordinate of the ranges to add
    /// \param stoh: handle to a tensor storage
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename Q>
    void append_blocks(const PartitionItem<Nd0> *p0, int num_blocks, const char *o0,
                       const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> dim0,
                       const char *o1, const Coor<Nd1> &from1, Storage_handle stoh, CoorOrder co) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::append_blocks<Nd0, Nd1, Q>(p0, num_blocks, from0, size0, dim0,
                                           detail::toArray<Nd0>(o0, "o0"),
                                           detail::toArray<Nd1>(o1, "o1"), sto, from1, comm, co);
    }

    /// Copy the content of plural tensor v0 into a storage
    /// \param alpha: factor applied to v0
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param mpicomm: MPI communicator context
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param o1: dimension labels for the storage
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param stoh: handle to a tensor storage
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void save(typename elem<T>::type alpha, const PartitionItem<Nd0> *p0, int ncomponents0,
              const char *o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
              const T **v0, const Context *ctx0, const char *o1, const Coor<Nd1> &from1,
              Storage_handle stoh, CoorOrder co, Session session = 0) {

        detail::Storage_context<Nd1, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd1, Q, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        detail::save<Nd0, Nd1, T, Q>(
            alpha, detail::get_from_size(p0, ncomponents0 * comm.nprocs, comm, dim0), from0, size0,
            dim0, detail::toArray<Nd0>(o0, "o0"),
            detail::get_components<Nd0>(v0, nullptr, ctx0, ncomponents0, p0, comm, session),
            detail::toArray<Nd1>(o1, "o1"), sto, from1, co, comm);
    }

    /// Copy from a storage into a plural tensor v1
    /// \param alpha: factor applied to v0
    /// \param stoh: handle to a tensor storage
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void load(typename elem<T>::type alpha, Storage_handle stoh, const char *o0,
              const Coor<Nd0> from0, const Coor<Nd0> size0, const PartitionItem<Nd1> *p1,
              int ncomponents1, const char *o1, const Coor<Nd1> &from1, const Coor<Nd1> &dim1,
              Q **v1, const Context *ctx1, CoorOrder co, CopyAdd copyadd, Session session = 0) {

        detail::Storage_context<Nd0, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd0, T, detail::SelfComm>(stoh);
        detail::SelfComm comm = detail::get_comm();

        if (copyadd == Copy)
            detail::load<Nd0, Nd1, T, Q>(
                alpha, sto, from0, size0, detail::toArray<Nd0>(o0, "o0"),
                detail::get_from_size(p1, ncomponents1 * comm.nprocs, comm, dim1)[comm.rank], from1,
                dim1, detail::toArray<Nd1>(o1, "o1"),
                detail::get_components<Nd1>(v1, nullptr, ctx1, ncomponents1, p1, comm, session),
                detail::EWOp::Copy{}, co, comm);
        else
            detail::load<Nd0, Nd1, T, Q>(
                alpha, sto, from0, size0, detail::toArray<Nd0>(o0, "o0"),
                detail::get_from_size(p1, ncomponents1 * comm.nprocs, comm, dim1)[comm.rank], from1,
                dim1, detail::toArray<Nd1>(o1, "o1"),
                detail::get_components<Nd1>(v1, nullptr, ctx1, ncomponents1, p1, comm, session),
                detail::EWOp::Add{}, co, comm);
    }

    /// Return the nonzero blocks stored
    /// \param stoh: handle to a tensor storage
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param o0: dimension labels for the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param o1: dimension labels for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param size1: size in destination tensor
    /// \param blocks: (output) vector of from-size pairs with the nonzero blocks
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

    template <std::size_t Nd0, std::size_t Nd1, typename T>
    void get_blocks(Storage_handle stoh, const char *o0, const char *o1, const Coor<Nd1> from1,
                    const Coor<Nd1> size1, std::vector<PartitionItem<Nd1>> &blocks, CoorOrder co) {

        detail::Storage_context<Nd0, detail::SelfComm> &sto =
            *detail::get_storage_context<Nd0, T, detail::SelfComm>(stoh);

        detail::get_blocks<Nd0, Nd1, T>(sto, detail::toArray<Nd0>(o0, "o0"),
                                        detail::toArray<Nd1>(o1, "o1"), from1, size1, blocks, co);
    }
}

#endif // __SUPERBBLAS_STORAGE__
