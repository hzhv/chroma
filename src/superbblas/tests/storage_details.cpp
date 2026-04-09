#include "superbblas.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace superbblas;
using namespace superbblas::detail;

template <std::size_t N, typename T> void check_storage(const char *filename) {
    Storage_handle stoh;
    open_storage<N, T>(filename, false /* don't allow writing */, &stoh);
    check_storage<N, T>(stoh);
    close_storage<N, T>(stoh);
}

template <std::size_t N = 16>
void check_storage(const char *filename, values_datatype dtype, int num_dims) {
    if (num_dims != N) {
        check_storage<N - 1>(filename, dtype, num_dims);
    } else {
        switch (dtype) {
        case FLOAT: check_storage<N, float>(filename); break;
        case DOUBLE: check_storage<N, double>(filename); break;
        case CFLOAT: check_storage<N, std::complex<float>>(filename); break;
        case CDOUBLE: check_storage<N, std::complex<double>>(filename); break;
        case CHAR: check_storage<N, char>(filename); break;
        case INT: check_storage<N, int>(filename); break;
        }
    }
}

template <> void check_storage<0>(const char *filename, values_datatype dtype, int num_dims) {
    (void)filename;
    (void)dtype;
    (void)num_dims;
}

using Blocks = std::vector<std::vector<std::vector<int>>>;

template <std::size_t N, typename T>
Blocks get_blocks(const char *filename, const std::vector<IndexType> &dim) {
    Storage_handle stoh;
    open_storage<N, T>(filename, false /* don't allow writing */, &stoh);
    std::vector<char> o(N + 1);
    for (unsigned int i = 0; i < N; ++i) o[i] = (char)(i + 1);
    Coor<N> dimc{};
    std::copy_n(dim.begin(), N, dimc.begin());
    std::vector<PartitionItem<N>> blocks;
    get_blocks<N, N, T>(stoh, o.data(), o.data(), {}, dimc, blocks, FastToSlow);
    close_storage<N, T>(stoh);
    Blocks blocks_out;
    for (const auto &b : blocks)
        blocks_out.push_back({std::vector<int>(b[0].begin(), b[0].end()),
                              std::vector<int>(b[1].begin(), b[1].end())});
    return blocks_out;
}

template <std::size_t N = 16>
Blocks get_blocks(const char *filename, values_datatype dtype, const std::vector<IndexType> &dim) {
    if (dim.size() != N) {
        return get_blocks<N - 1>(filename, dtype, dim);
    } else {
        switch (dtype) {
        case FLOAT: return get_blocks<N, float>(filename, dim); break;
        case DOUBLE: return get_blocks<N, double>(filename, dim); break;
        case CFLOAT: return get_blocks<N, std::complex<float>>(filename, dim); break;
        case CDOUBLE: return get_blocks<N, std::complex<double>>(filename, dim); break;
        case CHAR: return get_blocks<N, char>(filename, dim); break;
        case INT: return get_blocks<N, int>(filename, dim); break;
        default: return {};
        }
    }
}

template <>
Blocks get_blocks<0>(const char *filename, values_datatype dtype,
                     const std::vector<IndexType> &dim) {
    (void)filename;
    (void)dtype;
    (void)dim;
    return {};
}

template <typename T> std::string to_string(const T &c) {
    std::stringstream ss;
    if (c.size() > 0) ss << c[0];
    for (std::size_t i = 1; i < c.size(); ++i) ss << " " << c[i];
    return ss.str();
}

bool show(const char *filename, bool list_blocks, bool only_metadata) {
    values_datatype dtype;
    std::vector<char> metadata;
    std::vector<IndexType> dim;
    try {
        read_storage_header(filename, FastToSlow, dtype, metadata, dim);
    } catch (const std::exception &e) {
        std::cerr << "Ops! " << e.what() << std::endl;
        return false;
    }

    std::string dtypeS = "unknown";
    switch (dtype) {
    case FLOAT: dtypeS = "float"; break;
    case DOUBLE: dtypeS = "double"; break;
    case CFLOAT: dtypeS = "complex float"; break;
    case CDOUBLE: dtypeS = "complex double"; break;
    case CHAR: dtypeS = "char"; break;
    case INT: dtypeS = "int"; break;
    }

    std::string metadataS(metadata.begin(), metadata.end());

    if (only_metadata) {
        std::cout << metadataS << std::endl;
        return true;
    }

    std::cout << "datatype: " << dtypeS << std::endl                 //
              << "number of dimensions: " << dim.size() << std::endl //
              << "dimensions:" << to_string(dim) << std::endl        //
              << "metadata: (begin)" << std::endl
              << metadataS << std::endl
              << "(end)" << std::endl;

    // Check the checksums
    try {
        check_storage(filename, dtype, (int)dim.size());
    } catch (const std::exception &e) {
        std::cerr << "Ops! " << e.what() << std::endl;
        return false;
    }
    std::cout << "checksums: ok!" << std::endl;

    // Show blocks
    if (list_blocks) {
        std::cout << "blocks:" << std::endl;
        for (const auto &range : get_blocks(filename, dtype, dim))
            std::cout << "   from: " << to_string(range[0]) << "   size: " << to_string(range[1])
                      << std::endl;
    }

    return true;
}

bool print(const char *filename, std::vector<int> fromCoor, std::vector<int> sizeCoor) {
    (void)filename;
    (void)fromCoor;
    (void)sizeCoor;
    std::cerr << "Ops! Functionality still not implemented!" << std::endl;
    return false;
}

int main(int argc, char **argv) {

    const std::string help =
        "Application for inspecting files in S3T format. Command line:        \n"
        "                                                                     \n"
        "  storage_details <file > <action> [arg1] [arg2] ...                 \n"
        "                                                                     \n"
        "Actions:                                                             \n"
        "- storage_details <file> [show] [--list-blocks] [--only-metadata]    \n"
        "  Show information about the file.                                   \n"
        "  --list-blocks: show the coordinate ranges of all blocks in the     \n"
        "    file.                                                            \n"
        "  --only-metadata: don't show any information about the storage      \n"
        "    excepting its metadata.                                          \n"
        "                                                                     \n"
        "- storage_details <file> print [from <c0> ... [size <s0> ...]]       \n"
        "  Print the values of the subtensor starting at c0 ... and           \n"
        "  extending s0 ... coordinates in each dimension.                    \n"
        "                                                                     \n"
        "- storage_details [--help|-h]                                        \n"
        "  Show this help                                                     \n"
        "                                                                     \n";

    // Process options

    char *filename = nullptr;
    bool list_blocks = false;
    bool only_metadata = false;
    std::vector<int> fromCoor, sizeCoor;
    enum Action { Show, Print, Help } action = Help;
    enum Parsestate {
        FilenameOrHelp,
        Action,
        Arguments,
        From,
        FromCoor,
        SizeCoor
    } state = FilenameOrHelp;

    for (int i = 1; i < argc; ++i) {
        switch (state) {
            // Process filename
        case FilenameOrHelp:
            if (std::strncmp("--help", argv[i], 10) == 0 || std::strncmp("-h", argv[i], 10) == 0) {
                action = Help;
                // Don't process more arguments
                i = argc;
            } else {
                filename = argv[i];
                state = Action;
                action = Show;
            }
            break;

            // Process action
        case Action:
            if (std::strncmp("show", argv[i], 10) == 0) {
                action = Show;
                state = Arguments;
            } else if (std::strncmp("print", argv[i], 10) == 0) {
                action = Print;
                state = From;
            } else {
                action = Show;
                state = Arguments;
                --i; // reprocess this argument
            }
            break;

            // Process arguments for action `ahow`
        case Arguments:
            if (std::strncmp("--list-blocks", argv[i], 15) == 0) {
                list_blocks = true;
            } else if (std::strncmp("--only-metadata", argv[i], 18) == 0) {
                only_metadata = true;
            } else {
                std::cout << "Unknown argument: " << argv[i];
                return -1;
            }
            break;

            // Process token `from`
        case From:
            if (std::strncmp("from", argv[i], 15) == 0) {
                state = FromCoor;
            } else {
                std::cout << "Expected `from` as argument but got: " << argv[i];
                return -1;
            }
            break;

            // Process the coordinates for `from` and the token `size`
        case FromCoor: {
            int c = 0;
            if (std::strncmp("size", argv[i], 15) == 0) {
                state = SizeCoor;
            } else if (std::sscanf("%d", argv[i], &c) == 1) {
                fromCoor.push_back(c);
            } else {
                std::cout << "Expected a number or `size` as argument but got: " << argv[i];
                return -1;
            }
            break;
        }

            // Process the coordinates for `size`
        case SizeCoor: {
            int c = 0;
            if (std::sscanf("%d", argv[i], &c) == 1) {
                sizeCoor.push_back(c);
            } else {
                std::cout << "Expected a number as argument but got: " << argv[i];
                return -1;
            }
            break;
        }
        }
    }

    // Do the thing

    bool success = true;
    switch (action) {
    case Show: success = show(filename, list_blocks, only_metadata); break;
    case Print: success = print(filename, fromCoor, sizeCoor); break;
    case Help: std::cout << help << std::endl; break;
    }

    return success ? 0 : -1;
}
