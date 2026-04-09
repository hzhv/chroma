#include "superbblas.h"
#include <climits>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>
#ifdef _OPENMP
#    include <omp.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <std::size_t Nd> PartitionStored<Nd> dist_tensor_on_root(Coor<Nd> dim, int nprocs) {
    PartitionStored<Nd> fs(nprocs);
    if (1 <= nprocs) fs[0][1] = dim;
    return fs;
}

/// Extend the region one element in each direction
template <std::size_t Nd>
std::vector<std::array<Coor<Nd>, 2>> extend(const std::vector<std::array<Coor<Nd>, 2>> &fs,
                                            std::size_t i, IndexType size, const Coor<Nd> &dim) {
    auto r = fs;
    for (auto &ri : r) {
        ri[1][i] = std::min(dim[i], ri[1][i] + size);
        if (ri[1][i] < dim[i])
            ri[0][i] -= size / 2;
        else
            ri[0][i] = 0;
        ri[0] = normalize_coor(ri[0], dim);
    }
    return r;
}

void test_checksum() {
    const char *data = "Quixote was a great guy";
    const int n = std::strlen(data);
    checksum_t checksum_val0 = do_checksum(data, n);
    checksum_t checksum_val1 = 0;
    for (int i = 0; i < n; ++i) checksum_val1 = do_checksum(&data[i], 1, 0, checksum_val1);
    checksum_t checksum_val2 = 0;
    for (int i = 0; i < n; i += 2)
        checksum_val2 = do_checksum(&data[i], std::min(n - i, 2), 0, checksum_val2);
    if (checksum_val0 != checksum_val1 || checksum_val0 != checksum_val2)
        throw std::runtime_error("Checksum isn't associative");
}

constexpr std::size_t Nd = 8;           // mdtgsSnN
constexpr unsigned int nS = 4, nG = 16; // length of dimension spin and number of gammas
constexpr unsigned int M = 0, D = 1, T = 2, G = 3, S0 = 4, S1 = 5, N0 = 6, N1 = 7;

template <typename Scalar, typename XPU>
void test(Coor<Nd> dim, checksum_type checksum, Coor<Nd> procs, int nprocs, int rank, Context ctx,
          XPU xpu, unsigned int nrep, bool use_anarchofs = false) {

    std::string metadata = "S3T format!";
    const char *filename_local = "tensor.s3t";
    std::vector<char> wd(std::size_t(1) + PATH_MAX);
    if (getcwd(wd.data(), wd.size()) == NULL) throw std::runtime_error("error on getwcd");
    std::string filename_remote_str =
        std::string("afs:") + std::string(wd.data()) + std::string("/tensor.s3t");
    const char *filename_remote = !use_anarchofs ? filename_local : filename_remote_str.c_str();

    const unsigned int num_reqs = 1000;

    // Samples of different S to request
    std::vector<int> nn(1, dim[N0]);
    while (nn.back() > 16) nn.push_back(nn.back() / 2);
    std::reverse(nn.begin(), nn.end());

    // Create tensor t0 of Nd dims: a genprop
    const Coor<Nd - 1> dim0{dim[D], dim[T], dim[G], dim[S0], dim[S1], dim[N0], dim[N1]}; // dtgsSnN
    const Coor<Nd - 1> procs0 = {procs[D],  procs[T],  procs[G], procs[S0],
                                 procs[S1], procs[N0], procs[N1]}; // dtgsSnN
    PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
    const Coor<Nd - 1> local_size0 = p0[rank][1];

    // Generate random requests
    std::size_t vol = detail::volume(dim);
    std::vector<std::size_t> reqs(num_reqs);
    {
        std::size_t hash = 5831;
        for (std::size_t c = 0; c < reqs.size(); ++c) {
            hash = hash * 33 + c;
            reqs[c] = hash % (vol / dim[N0] / dim[N1]);
        }
    }

    if (rank == 0)
        std::cout << "Testing "
                  << (checksum == NoChecksum
                          ? "without checksum"
                          : (checksum == BlockChecksum ? "block checksum" : "global checksum"))
                  << std::endl
                  << "Maximum number of elements in a tested tensor per process: "
                  << detail::volume(local_size0) << " ( "
                  << detail::volume(local_size0) * 1.0 * sizeof(Scalar) / 1024 / 1024
                  << " MiB)   Expected file size: " << vol * 1.0 * sizeof(Scalar) / 1024 / 1024
                  << " MiB" << std::endl;

    resetTimings();

    // Create a file copying the content from a buffer; this should be the fastest way
    // to populate the file
    double trefw = 0.0;
    const bool dowrite = true;
    std::vector<double> trefr(nn.size(), 0.0);
    if (rank == 0) {
        std::FILE *f = std::fopen(filename_local, "w+");
        if (f == nullptr) superbblas::detail::gen_error("Error opening file for writing");

        // Dummy initialization of t0
        std::size_t vol0 = detail::volume(dim) / dim[M];
        std::vector<Scalar> t0(vol0);
        for (unsigned int i = 0; i < vol0; i++) t0[i] = i;

        double t = w_time();
        for (unsigned int rep = 0; rep < nrep; ++rep) {
            for (int m = 0; m < dim[M]; ++m) {
                if (std::fwrite(t0.data(), sizeof(Scalar), vol0, f) != vol0)
                    superbblas::detail::gen_error("Error writing in a file");
            }
        }
        std::fflush(f);
        t = w_time() - t;
        std::cout << "Time in dummy writing the tensor " << t / nrep << " s  "
                  << dim[M] * vol0 * sizeof(Scalar) * nrep / t / 1024 / 1024 << " MiB/s"
                  << std::endl;
        trefw = t / nrep; // time in copying a whole tensor with size dim1

        std::fclose(f);
        f = std::fopen(filename_local, "rb");
        if (f == nullptr) superbblas::detail::gen_error("Error opening file for reading");

        for (std::size_t nni = 0; nni < nn.size(); ++nni) {
            std::vector<Scalar> t1(nn[nni] * nn[nni]);
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (std::size_t r : reqs) {
                    if (std::fseek(f, sizeof(Scalar) * dim[N0] * dim[N1] * r, SEEK_SET) != 0)
                        throw std::runtime_error("Error setting file position");
                    std::size_t fread_out =
                        std::fread(t1.data(), sizeof(Scalar), nn[nni] * nn[nni], f);
                    if (fread_out != (std::size_t)nn[nni] * nn[nni])
                        superbblas::detail::gen_error("Error reading in a file");
                }
            }
            t = w_time() - t;
            std::cout << "Time in dummy reading the tensor with " << nn[nni] << "^2 elements "
                      << t / nrep << " s  "
                      << nn[nni] * nn[nni] * sizeof(Scalar) * nrep * reqs.size() / t / 1024 / 1024
                      << " MiB/s" << std::endl;
            trefr[nni] = t / nrep; // time in copying a whole tensor with size dim1
        }

        std::fclose(f);
    }

#ifdef SUPERBBLAS_USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Save tensor t0
    if (dowrite) {
        std::size_t vol0 = detail::volume(local_size0);
        vector<Scalar, Cpu> t0_cpu(vol0, Cpu{});
        for (unsigned int i = 0; i < vol0; i++) t0_cpu[i] = i;
        vector<Scalar, XPU> t0 = makeSure(t0_cpu, xpu);

        for (int with_trans = 0; with_trans < 2; ++with_trans) {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                Storage_handle stoh;
                create_storage<Nd, Scalar>(dim, SlowToFast, filename_local, metadata.c_str(),
                                           metadata.size(), checksum,
#ifdef SUPERBBLAS_USE_MPI
                                           MPI_COMM_WORLD,
#endif
                                           &stoh);
                std::array<Coor<Nd>, 2> fs{Coor<Nd>{{}}, dim};
                append_blocks<Nd, Scalar>(&fs, 1, dim, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                          MPI_COMM_WORLD,
#endif
                                          SlowToFast);
                for (int m = 0; m < dim[M]; ++m) {
                    const Coor<Nd - 1> from0{{}};
                    const Coor<Nd> from1{m};
                    Scalar *ptr0 = t0.data();
                    save<Nd - 1, Nd, Scalar, Scalar>(
                        1.0, p0.data(), 1, "dtgsSnN", from0, dim0, dim0, (const Scalar **)&ptr0,
                        &ctx, with_trans == 0 ? "mdtgsSnN" : "mdtgsSNn", from1, stoh,
#ifdef SUPERBBLAS_USE_MPI
                        MPI_COMM_WORLD,
#endif
                        SlowToFast);
                }
                close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                          ,
                                          MPI_COMM_WORLD
#endif
                );
            }
            t = w_time() - t;
            if (rank == 0)
                std::cout << "Time in writing " << (with_trans == 0 ? "" : "[with transposition] ")
                          << t / nrep << " s (overhead " << t / nrep / trefw << " )" << std::endl;
        }
    }

    Storage_handle stoh;
    open_storage<Nd, Scalar>(filename_remote, false /* don't allow writing */,
#ifdef SUPERBBLAS_USE_MPI
                             MPI_COMM_WORLD,
#endif
                             &stoh);

    // Check storage
    check_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                              ,
                              MPI_COMM_WORLD
#endif
    );

    // Load into tensor t1
    if (dowrite) {
        const Coor<Nd - 2> dimr{dim[M], dim[D], dim[T], dim[G], dim[S0], dim[S1]}; // mdtgsS
        Coor<Nd - 2, std::size_t> stride = detail::get_strides<std::size_t>(dimr, SlowToFast);

        for (std::size_t nni = 0; nni < nn.size(); ++nni) {
            int n = nn[nni];
            // Create tensor t1 for reading the genprop on root process
            Coor<2> dim1{{n, n}};
            PartitionStored<2> p1(nprocs);
            p1[0][1] = dim1;
            std::size_t vol1 = detail::volume(p1[rank][1]);
            vector<Scalar, XPU> t1(vol1, xpu);

            for (int with_trans = 0; with_trans < 2; ++with_trans) {
                double t = w_time();
                for (unsigned int rep = 0; rep < nrep; ++rep) {
                    for (auto req : reqs) {
                        Coor<Nd> from0{{}};
                        std::copy_n(detail::index2coor(req, dimr, stride).begin(), Nd - 2,
                                    from0.begin());
                        Coor<Nd> size0{{}};
                        for (auto &c : size0) c = 1;
                        size0[Nd - 2] = size0[Nd - 1] = n;
                        const Coor<2> from1{{}};
                        Scalar *ptr1 = t1.data();
                        load<Nd, 2, Scalar, Scalar>(1.0, stoh, "mdtgsSnN", from0, size0, p1.data(),
                                                    1, with_trans == 0 ? "nN" : "Nn", from1, dim1,
                                                    &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                                    MPI_COMM_WORLD,
#endif
                                                    SlowToFast, Copy);
                    }
                }
                t = w_time() - t;
                if (rank == 0)
                    std::cout << "Time in reading the tensor with " << n << "^2 elements "
                              << (with_trans == 0 ? "" : "[with transposition] ") << t / nrep
                              << " s  "
                              << " (overhead " << t / nrep / trefr[nni] << " )" << std::endl;
            }
        }
    }

    close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                              ,
                              MPI_COMM_WORLD
#endif
    );

    for (CoorOrder co : std::array<CoorOrder, 2>{SlowToFast, FastToSlow}) {
        Storage_handle stoh;
        create_storage<Nd, Scalar>(dim, co, filename_local, metadata.c_str(), metadata.size(),
                                   checksum,
#ifdef SUPERBBLAS_USE_MPI
                                   MPI_COMM_WORLD,
#endif
                                   &stoh);
        std::array<Coor<Nd>, 2> fs{Coor<Nd>{{}}, dim};
        append_blocks<Nd, Scalar>(&fs, 1, dim, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                  MPI_COMM_WORLD,
#endif
                                  co);

        // Store proper values to test the storage
        {
            PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
            const Coor<Nd - 1> local_size0 = p0[rank][1];
            std::size_t vol0 = detail::volume(local_size0);
            vector<Scalar, Cpu> t0_cpu(vol0, Cpu{});

            Coor<Nd - 1, std::size_t> local_strides0 =
                detail::get_strides<std::size_t>(local_size0, co);
            Coor<Nd, std::size_t> strides1 = detail::get_strides<std::size_t>(dim, co);
            for (int m = 0; m < dim[M]; ++m) {
                const Coor<Nd - 1> from0{{}};
                const Coor<Nd> from1{m};
                for (std::size_t i = 0; i < vol0; ++i) {
                    Coor<Nd - 1> c0 = index2coor(i, local_size0, local_strides0) + p0[rank][0];
                    Coor<Nd> c{m};
                    std::copy_n(c0.begin(), Nd - 1, c.begin() + 1);
                    t0_cpu[i] = coor2index(c, dim, strides1);
                }
                vector<Scalar, XPU> t0 = makeSure(t0_cpu, xpu);
                Scalar *ptr0 = t0.data();
                save<Nd - 1, Nd, Scalar, Scalar>(1.0, p0.data(), 1, "dtgsSnN", from0, dim0, dim0,
                                                 (const Scalar **)&ptr0, &ctx, "mdtgsSnN", from1,
                                                 stoh,
#ifdef SUPERBBLAS_USE_MPI
                                                 MPI_COMM_WORLD,
#endif
                                                 co);
            }

            flush_storage(stoh);

            if (rank == 0) {
                // The data of the only block should contain the numbers from zero to vol
                std::size_t padding_size = (8 - metadata.size() % 8) % 8;
                std::size_t header_size =
                    sizeof(int) * 6 + metadata.size() + padding_size + sizeof(double) * (Nd + 1);
                std::size_t disp = header_size + sizeof(double) * (2 + Nd * 2);
                std::ifstream f(filename_local, std::ios::binary);
                f.seekg(disp);
                Scalar s;
                for (std::size_t i = 0; i < vol; ++i) {
                    f.read((char *)&s, sizeof(s));
                    if (i != std::real(s)) throw std::runtime_error("Failing reading from storage");
                }
                f.close();
            }
        }

        // Check metadata
        {
            values_datatype dtype;
            std::vector<char> metadata0;
            std::vector<IndexType> dim0;
            read_storage_header(filename_remote, co, dtype, metadata0, dim0);

            if (std::string(metadata0.begin(), metadata0.end()) != metadata)
                throw std::runtime_error("Error recovering metadata");

            if (std::vector<IndexType>(dim.begin(), dim.end()) != dim0)
                throw std::runtime_error("Error recovering tensor dimensions");

            if (dtype != get_values_datatype<Scalar>())
                throw std::runtime_error("Error recovering the tensor datatype");
        }

        // Test the readings
        {
            const Coor<Nd - 2> dimr{dim[M], dim[D], dim[T], dim[G], dim[S0], dim[S1]}; // mdtgsS
            Coor<Nd - 2, std::size_t> stridesr = detail::get_strides<std::size_t>(dimr, co);
            Coor<Nd, std::size_t> strides = detail::get_strides<std::size_t>(dim, co);

            for (auto n : nn) {
                Coor<2> dimnn{n, n};
                Coor<2, std::size_t> stridesnn = detail::get_strides<std::size_t>(dimnn, co);

                // Create tensor t1 for reading the genprop on root process
                PartitionStored<2> p1(nprocs);
                p1[0][1] = dimnn;
                std::size_t vol1 = detail::volume(p1[rank][1]);
                vector<Scalar, Cpu> t1_m1(vol1, Cpu{});
                for (std::size_t i = 0; i < vol1; ++i) t1_m1[i] = -1;
                vector<Scalar, XPU> t1(vol1, xpu);

                for (auto req : reqs) {
                    Coor<Nd> from0{{}};
                    std::copy_n(detail::index2coor(req, dimr, stridesr).begin(), Nd - 2,
                                from0.begin());
                    Coor<Nd> size0{{}};
                    for (auto &c : size0) c = 1;
                    size0[Nd - 2] = size0[Nd - 1] = n;
                    const Coor<2> from1{{}};
                    copy_n(t1_m1.data(), Cpu{}, vol1, t1.data(), xpu);
                    Scalar *ptr1 = t1.data();
                    for (int with_trans = 0; with_trans < 2; ++with_trans) {
                        load<Nd, 2, Scalar, Scalar>(1.0, stoh, "mdtgsSnN", from0, size0, p1.data(),
                                                    1, with_trans == 0 ? "nN" : "Nn", from1, dimnn,
                                                    &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                                    MPI_COMM_WORLD,
#endif
                                                    co, Copy);
                        vector<Scalar, Cpu> t1_cpu = makeSure(t1, Cpu{});
                        for (std::size_t i = 0; i < vol1; ++i) {
                            Coor<2> cnn = index2coor(i, dimnn, stridesnn) + p1[rank][0];
                            Coor<Nd> c{{}};
                            c[Nd - 2] = cnn[with_trans == 0 ? 0 : 1];
                            c[Nd - 1] = cnn[with_trans == 0 ? 1 : 0];
                            if (std::real(t1_cpu[i]) != coor2index(from0 + c, dim, strides))
                                throw std::runtime_error("Storage failed!");
                        }
                    }
                }
            }
        }

        close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#endif
        );

        open_storage<Nd, Scalar>(filename_remote, false /* don't allow writing */,
#ifdef SUPERBBLAS_USE_MPI
                                 MPI_COMM_WORLD,
#endif
                                 &stoh);

        // Check storage
        check_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#endif
        );

        close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#endif
        );

        create_storage<Nd, Scalar>(dim, co, filename_local, metadata.c_str(), metadata.size(),
                                   checksum,
#ifdef SUPERBBLAS_USE_MPI
                                   MPI_COMM_WORLD,
#endif
                                   &stoh);

        // Store proper values to test the sparse storage
        {
            PartitionStored<Nd - 1> p0 = extend(basic_partitioning(dim0, procs0), 1, 4, dim0);
            const Coor<Nd - 1> local_size0 = p0[rank][1];
            std::size_t vol0 = detail::volume(local_size0);
            vector<Scalar, Cpu> t0_cpu(vol0, Cpu{});

            Coor<Nd - 1, std::size_t> local_strides0 =
                detail::get_strides<std::size_t>(local_size0, co);
            Coor<Nd, std::size_t> strides1 = detail::get_strides<std::size_t>(dim, co);
            for (int m = 0; m < dim[M]; ++m) {
                const Coor<Nd - 1> from0{{}};
                const Coor<Nd> from1{m};

                append_blocks<Nd - 1, Nd, Scalar>(p0.data(), nprocs, "dtgsSnN", Coor<Nd - 1>{{}},
                                                  dim0, dim0, "mdtgsSnN", from1, stoh,
#ifdef SUPERBBLAS_USE_MPI
                                                  MPI_COMM_WORLD,
#endif
                                                  co);

                for (std::size_t i = 0; i < vol0; ++i) {
                    Coor<Nd - 1> c0 = index2coor(i, local_size0, local_strides0) + p0[rank][0];
                    Coor<Nd> c{m};
                    std::copy_n(c0.begin(), Nd - 1, c.begin() + 1);
                    t0_cpu[i] = coor2index(c, dim, strides1);
                }
                vector<Scalar, XPU> t0 = makeSure(t0_cpu, xpu);
                Scalar *ptr0 = t0.data();
                save<Nd - 1, Nd, Scalar, Scalar>(1.0, p0.data(), 1, "dtgsSnN", from0, dim0, dim0,
                                                 (const Scalar **)&ptr0, &ctx, "mdtgsSnN", from1,
                                                 stoh,
#ifdef SUPERBBLAS_USE_MPI
                                                 MPI_COMM_WORLD,
#endif
                                                 co);
            }
        }

        // Test the readings
        {
            const Coor<Nd - 2> dimr{dim[M], dim[D], dim[T], dim[G], dim[S0], dim[S1]}; // mdtgsS
            Coor<Nd - 2, std::size_t> stridesr = detail::get_strides<std::size_t>(dimr, co);
            Coor<Nd, std::size_t> strides = detail::get_strides<std::size_t>(dim, co);

            for (auto n : nn) {
                Coor<2> dimnn{n, n};
                Coor<2, std::size_t> stridesnn = detail::get_strides<std::size_t>(dimnn, co);

                // Create tensor t1 for reading the genprop on root process
                PartitionStored<2> p1(nprocs);
                p1[0][1] = dimnn;
                std::size_t vol1 = detail::volume(p1[rank][1]);
                vector<Scalar, Cpu> t1_m1(vol1, Cpu{});
                for (std::size_t i = 0; i < vol1; ++i) t1_m1[i] = -1;
                vector<Scalar, XPU> t1(vol1, xpu);

                for (auto req : reqs) {
                    Coor<Nd> from0{{}};
                    std::copy_n(detail::index2coor(req, dimr, stridesr).begin(), Nd - 2,
                                from0.begin());
                    Coor<Nd> size0{{}};
                    for (auto &c : size0) c = 1;
                    size0[Nd - 2] = size0[Nd - 1] = n;
                    const Coor<2> from1{{}};
                    copy_n(t1_m1.data(), Cpu{}, vol1, t1.data(), xpu);
                    Scalar *ptr1 = t1.data();
                    load<Nd, 2, Scalar, Scalar>(1.0, stoh, "mdtgsSnN", from0, size0, p1.data(), 1,
                                                "nN", from1, dimnn, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                                                MPI_COMM_WORLD,
#endif
                                                co, Copy);
                    vector<Scalar, Cpu> t1_cpu = makeSure(t1, Cpu{});
                    for (std::size_t i = 0; i < vol1; ++i) {
                        Coor<2> cnn = index2coor(i, dimnn, stridesnn) + p1[rank][0];
                        Coor<Nd> c{{}};
                        c[Nd - 2] = cnn[0];
                        c[Nd - 1] = cnn[1];
                        if (std::real(t1_cpu[i]) != coor2index(from0 + c, dim, strides))
                            throw std::runtime_error("Storage failed!");
                    }
                }
            }
        }

        close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#endif
        );

        open_storage<Nd, Scalar>(filename_remote, false /* don't allow writing */,
#ifdef SUPERBBLAS_USE_MPI
                                 MPI_COMM_WORLD,
#endif
                                 &stoh);

        // Check storage
        check_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#endif
        );

        close_storage<Nd, Scalar>(stoh
#ifdef SUPERBBLAS_USE_MPI
                                  ,
                                  MPI_COMM_WORLD
#endif
        );
    }

    if (rank == 0) reportTimings(std::cout);
    if (rank == 0) reportCacheUsage(std::cout);
}

int main(int argc, char **argv) {
    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    (void)argc;
    (void)argv;
    nprocs = 1;
    rank = 0;
#endif

    test_checksum();

    Coor<Nd> dim = {2, 3, 5, nG, nS, nS, 4, 4}; // mdtgsSnN
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1, 1};
    const unsigned int nrep = getDebugLevel() == 0 ? 10 : 1;

    // Get options
    bool procs_was_set = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6,
                       std::is_same<superbblas::IndexType, int>::value ? "%d %d %d %d %d"
                                                                       : "%ld %ld %ld %ld %ld",
                       &dim[M], &dim[D], &dim[T], &dim[G], &dim[N0]) != 5) {
                std::cerr << "--dim= should follow 5 numbers, for instance -dim='2 2 2 2 2'"
                          << std::endl;
                return -1;
            }
            dim[N1] = dim[N0];
        } else if (std::strncmp("--procs=", argv[i], 8) == 0) {
            if (sscanf(argv[i] + 8, std::is_same<superbblas::IndexType, int>::value ? "%d" : "%ld",
                       &procs[T]) != 1) {
                std::cerr << "--procs= should follow one number, for instance --procs=2"
                          << std::endl;
                return -1;
            }
            if (detail::volume(procs) != (std::size_t)nprocs) {
                std::cerr << "The total number of processes set by the option `--procs=` should "
                             "match the number of processes"
                          << std::endl;
                return -1;
            }
            procs_was_set = true;
        } else if (std::strncmp("--help", argv[i], 6) == 0) {
            std::cout << "Commandline option:\n  " << argv[0]
                      << " [--dim='m d t g n'] [--procs=t] [--help]" << std::endl;
            return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // If --procs isn't set, put all processes on the first dimension
    if (!procs_was_set) procs[T] = nprocs;

    // Show lattice dimensions and processes arrangement
    if (rank == 0) {
        std::cout << "Testing lattice dimensions mdtgsn= " << dim[M] << " " << dim[D] << " "
                  << dim[T] << " " << dim[G] << " " << dim[S0] << " " << dim[N0] << std::endl;
        std::cout << "Processes arrangement t= " << procs[T] << std::endl;
    }

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    {
        if (rank == 0) std::cout << ">>> CPU tests with " << num_threads << " threads" << std::endl;
        if (rank == 0) std::cout << ">>> test for float" << std::endl;
        Context ctx = createCpuContext();
        test<float>(dim, NoChecksum, procs, nprocs, rank, ctx, ctx.toCpu(0), nrep);
        test<float>(dim, BlockChecksum, procs, nprocs, rank, ctx, ctx.toCpu(0), nrep);
        test<float>(dim, GlobalChecksum, procs, nprocs, rank, ctx, ctx.toCpu(0), nrep);
        if (rank == 0) std::cout << ">>> test for complex double" << std::endl;
        test<std::complex<double>>(dim, NoChecksum, procs, nprocs, rank, ctx, ctx.toCpu(0), nrep);
        test<std::complex<double>>(dim, BlockChecksum, procs, nprocs, rank, ctx, ctx.toCpu(0),
                                   nrep);
        test<std::complex<double>>(dim, GlobalChecksum, procs, nprocs, rank, ctx, ctx.toCpu(0),
                                   nrep);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        if (rank == 0) std::cout << ">>> GPU tests with " << num_threads << " threads" << std::endl;
        if (rank == 0) std::cout << ">>> test for complex double" << std::endl;
        Context ctx = createGpuContext();
        test<std::complex<double>>(dim, BlockChecksum, procs, nprocs, rank, ctx, ctx.toGpu(0),
                                   nrep);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#endif
#ifdef SUPERBBLAS_USE_ANARCHOFS
    {
        if (rank == 0) std::cout << ">>> CPU tests with " << num_threads << " threads" << std::endl;
        if (rank == 0) std::cout << ">>> test for float with anarchofs" << std::endl;
        Context ctx = createCpuContext();
        test<float>(dim, BlockChecksum, procs, nprocs, rank, ctx, ctx.toCpu(0), nrep,
                    true /* use anarchofs */);
        clearCaches();
        checkForMemoryLeaks(std::cout);
    }
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
