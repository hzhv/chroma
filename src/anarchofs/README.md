# AnarchoFS

AnarchoFS is a virtual filesystem that unionizes the content of several directories at different machines.
It uses MPI to communicate between processes and transfer content.
Interface:
- C++ API to `open` (read-only) files, change cursor (`seek`), `read` content, and `close` handler.
- virtual filesystem with using FUSE (optional)

The initial motivation of this project is to overcome the IO bottleneck of the shared filesystems in advanced computer facilities.

## Installation

Dependencies:

- C++ compiler
- MPI library, for instance [OpenMPI](https://www.open-mpi.org/)
- [libfuse](https://github.com/libfuse/libfuse) (optional)

Execute the default `make` action to compile the binary with `CXX` being the MPI compiler wrapper. For instance:
```
make CXX=mpicxx                   # no fuse
make CXX=mpicxx AFS_WITH_FUSE=yes # with fuse
```

Otherwise, compile directly the file `anarchofs.cc` with the MPI (and fuse) includes and dependencies. For instance:
```
# Without fuse
mpicxx anarchofs.cc -o anarchofs
# With fuse
mpicxx anarchofs.cc `pkg-config fuse3 --cflags --libs` -DAFS_DAEMON_USE_FUSE -o anarchofs
```

## Usage with C++

Interface description:
```
/// Open a file (for read-only for now)
/// \param path: path of the file to open
/// \return: file handler or (null if failed)
anarchofs::client::File *anarchofs::client::open(const char *filename);

/// Write the content of the file into a given buffer
/// \param f: file handler
/// \param v: buffer where to write the content
/// \param n: number of characters to read
/// \return: number of characters written into the buffer if positive; error code otherwise
std::int64_t anarchofs::client::read(anarchofs::client::File *f, char *v, std::size_t n);

/// Change the current cursor of the file handler
/// \param f: file handler
/// \param offset: absolute offset of the first element to be read
void anarchofs::client::seek(anarchofs::client::File *f, std::size_t offset);

/// Close a file hander
/// \param f: file handler
/// \return: whether the operation was successful
bool anarchofs::client::close(anarchofs::client::File *f);
```

Example:
```
#include "anarchofs_lib.h"
int main(int argc, char **argv) {
    // Open a file '/tmp/remote_file', which may be in another computer
    auto f = anarchofs::client::open("/tmp/remote_file");
    if (f == nullptr) throw std::runtime_error("error remote reading the file");

    // Read ten integers starting from the second
    std::vector<int> data(10);
    anarchofs::client::seek(f, sizeof(int) * 2);
    anarchofs::client::read(f, (char *)data.data(), data.size() * sizeof(int));

    // Close the file handler
    anarchofs::client::close(f);

    return 0;
}
```

## Usage with FUSE

The more simple way is to 

```
mpirun -np <nprocs> ./anarchofs -s -f -o max_threads=1 -o modules=subdir -o subdir=<basedir> <mountpoint>
```

For instance, the following unionizes the local directory `/tmp` from the machines `hostname{0,1}`:
```
mpirun -H hostname0,hostname1 -np 2 mkdir -p ~/tmp_shared
mpirun -H hostname0,hostname1 -np 2 ./anarchofs -s -f -o max_threads=1 -o modules=subdir -o subdir=/tmp ~/tmp_shared &
```

To unmount the virtual filesystem, kill the `mpirun ... anarchofs` process or invoke `fusermount` (or `fusermount3`) on each machine:
```
mpirun -np <nprocs> fusermount3 -u <mountpoint>
```
