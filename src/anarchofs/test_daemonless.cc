#include "anarchofs_lib.h"
#include <cstdio>

/// Replace "@NPROC" by the process id in the given string; used for debugging
/// \param path: given string

inline std::string replace_hack(const char *path, int id) {
    std::string::size_type n = 0;
    std::string path_s(path);
    std::string re("@NPROC");
    std::string this_proc_s = std::to_string(id);
    while ((n = path_s.find(re)) != std::string::npos) {
        path_s.replace(n, re.size(), this_proc_s);
    }
    return path_s;
}

const std::size_t file_size = 10 * 1024 * 1024;

void create_file(const char *base_path, int id) {
    auto filename = replace_hack(base_path, id) + std::string("/f") + std::to_string(id);
    auto f = std::fopen(filename.c_str(), "w");
    if (f == nullptr) throw std::runtime_error("error creating the file");
    std::vector<int> data(file_size / sizeof(int));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = id + i;
    std::fwrite(data.data(), data.size(), sizeof(int), f);
    std::fclose(f);
}

void check_file(const char *base_path, int id, int max_loops = 10) {
    auto filename = std::string(base_path) + std::string("/f") + std::to_string(id);
    auto f = anarchofs::client::open(filename.c_str());
    if (f == nullptr) throw std::runtime_error("error remote reading the file");
    std::vector<int> data(file_size / sizeof(int));

    for (int loops = 0; loops < max_loops; ++loops) {
        anarchofs::client::seek(f, 0);
        anarchofs::client::read(f, (char *)data.data(), data.size() * sizeof(int));
        for (std::size_t i = 0; i < data.size(); ++i)
            if ((std::size_t)data[i] != id + i) throw std::runtime_error("check didn't pass");

        anarchofs::client::seek(f, 0);
        for (std::size_t i = 0; i < data.size(); ++i) {
            int d;
            anarchofs::client::read(f, (char *)&d, sizeof(int));
            if ((std::size_t)d != id + i) throw std::runtime_error("check didn't pass");
        }
    }

    anarchofs::client::close(f);
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
        throw std::runtime_error("MPI does not support the required thread level");

    int nprocs = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Start anarchofs daemon
    anarchofs::start_mpi_loop(true, &argc, &argv);

    // Read the path of the based filesystem
    const char *base_path = "/tmp";
    if (argc > 1) base_path = argv[1];

    if (rank == 0) create_file(base_path, 0);
    MPI_Barrier(MPI_COMM_WORLD);

    check_file(base_path, 0, rank == 0 ? 1 : 10);

    anarchofs::stop_mpi_loop();

    MPI_Finalize();
    return 0;
}
