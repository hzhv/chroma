/// AnarchoFS daemon

#define BUILD_AFS_DAEMON
#include "anarchofs_lib.h"

#ifdef AFS_DAEMON_USE_FUSE
/// AnarchoFS implementation with FUSE.
///
/// This code is based on example/passthrough.c in libfuse, https://github.com/libfuse/libfuse.
/// Copyrights from project FUSE: Filesystem in Userspace
///   Copyright (C) 2001-2007  Miklos Szeredi <miklos@szeredi.hu>
///   Copyright (C) 2011       Sebastian Pipping <sebastian@pipping.org>

#    include <dirent.h>
#    include <errno.h>
#    include <fcntl.h>
#    include <fuse.h>
#    include <stdio.h>
#    include <string.h>
#    include <sys/stat.h>
#    include <unistd.h>
#    ifdef __FreeBSD__
#        include <sys/socket.h>
#        include <sys/un.h>
#    endif
#    include <sys/time.h>

static int fill_dir_plus = 0;

static void *xmp_init(struct fuse_conn_info *conn, struct fuse_config *cfg) {
    (void)conn;
    cfg->use_ino = 0;

    /* Pick up changes from lower filesystem right away. This is
	   also necessary for better hardlink support. When the kernel
	   calls the unlink() handler, it does not know the inode of
	   the to-be-removed entry and can therefore not invalidate
	   the cache of the associated inode - resulting in an
	   incorrect st_nlink value being reported for any remaining
	   hardlinks to this inode. */
    cfg->entry_timeout = 0;
    cfg->attr_timeout = 0;
    cfg->negative_timeout = 0;

    return NULL;
}

static void xmp_destroy(void *private_data) {
    (void)private_data;
    if (!anarchofs::stop_mpi_loop()) { printf("something when wrong stopping mpi!\n"); }
}

static int xmp_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi) {
    (void)fi;

    anarchofs::FileType file_type;
    anarchofs::Offset file_size;
    if (!anarchofs::get_status(path, &file_type, &file_size)) return -ENOENT;
    memset(stbuf, 0, sizeof(struct stat));
    stbuf->st_mode = file_type == anarchofs::FileType::Directory
                         ? S_IFDIR
                         : (file_type == anarchofs::FileType::Link
                                ? S_IFLNK
                                : (file_type == anarchofs::FileType::RegularFile ? S_IFREG : 0));
    stbuf->st_size = file_size;
    return 0;
}

static int xmp_access(const char *path, int mask) {
    anarchofs::FileType file_type;
    anarchofs::Offset file_size;
    if (!anarchofs::get_status(path, &file_type, &file_size)) return -ENOENT;
    if (mask & W_OK) return -1;
    return 0;
}

static int xmp_readlink(const char *path, char *buf, size_t size) {
    (void)path;
    (void)buf;
    (void)size;
    assert(false);
    return -EPERM;
}

static int xmp_readdir(const char *path, void *buf, fuse_fill_dir_t filler, off_t offset,
                       struct fuse_file_info *fi, enum fuse_readdir_flags flags) {
    //DIR *dp;
    //struct dirent *de;

    (void)offset;
    (void)fi;
    (void)flags;

    auto entries = anarchofs::get_directory_list(path);
    std::map<anarchofs::FileType, mode_t> to_st_mode = {{anarchofs::FileType::RegularFile, S_IFREG},
                                                        {anarchofs::FileType::Link, S_IFLNK},
                                                        {anarchofs::FileType::Directory, S_IFDIR},
                                                        {anarchofs::FileType::Other, 0}};
    for (const auto &it : entries) {
        struct stat st;
        memset(&st, 0, sizeof(st));
        st.st_ino = 0;
        st.st_mode = to_st_mode.at(it.type);
        if (filler(buf, it.filename.c_str(), &st, 0, (fuse_fill_dir_flags)fill_dir_plus)) break;
    }
    return 0;
}

static int xmp_open(const char *path, struct fuse_file_info *fi) {
    //if (fi->flags != O_RDONLY) return -ENOTSUP;
    auto file_id = anarchofs::get_open_file(path);
    if (file_id == anarchofs::no_file_id) return -ENOENT;

    fi->fh = file_id;
    return 0;
}

static int xmp_read(const char *path, char *buf, size_t size, off_t offset,
                    struct fuse_file_info *fi) {
    anarchofs::FileId file_id;
    int res;

    if (fi == NULL)
        file_id = anarchofs::get_open_file(path);
    else
        file_id = fi->fh;

    if (file_id == anarchofs::no_file_id) return -ENOENT;

    res = anarchofs::read(file_id, offset, size, buf);
    if (res == -1) res = -EIO;

    if (fi == NULL) anarchofs::close(file_id);
    return res;
}

static int xmp_statfs(const char *path, struct statvfs *stbuf) {
    int res;

    res = statvfs(path, stbuf);
    if (res == -1) return -errno;

    stbuf->f_flag = ST_NOATIME | ST_NODEV | ST_NODIRATIME | ST_NOEXEC | ST_NOSUID | ST_RDONLY;

    return 0;
}

static int xmp_release(const char *path, struct fuse_file_info *fi) {
    (void)path;
    anarchofs::close(fi->fh);
    return 0;
}

static int xmp_fsync(const char *path, int isdatasync, struct fuse_file_info *fi) {
    /* Just a stub.	 This method is optional and can safely be left
	   unimplemented */

    (void)path;
    (void)isdatasync;
    (void)fi;
    return 0;
}

static const struct fuse_operations xmp_oper = {
    .getattr = xmp_getattr,
    .readlink = xmp_readlink,
    .open = xmp_open,
    .read = xmp_read,
    .statfs = xmp_statfs,
    .release = xmp_release,
    .fsync = xmp_fsync,
    .readdir = xmp_readdir,
    .init = xmp_init,
    .destroy = xmp_destroy,
    .access = xmp_access,
};
#endif // AFS_DAEMON_USE_FUSE

int main(int argc, char *argv[]) {
#ifdef AFS_DAEMON_USE_FUSE
    umask(0);
#endif
    if (!anarchofs::start_mpi_loop(&argc, &argv)) {
        printf("something when wrong!\n");
        return -1;
    }
#ifdef AFS_DAEMON_USE_FUSE
    anarchofs::server::start_socket_loop();
    int r = fuse_main(argc, argv, &xmp_oper, NULL);
    return r;
#else
    anarchofs::server::start_socket_loop(false /* launch another thread */);
    return 0;
#endif
}
