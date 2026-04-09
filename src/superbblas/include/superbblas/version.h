#ifndef __SUPERBBLAS_VERSION__
#define __SUPERBBLAS_VERSION__

#define SUPERBBLAS_VERSION_MAJOR 0
#define SUPERBBLAS_VERSION_MINOR 2

/// Superbblas version number: xxyy, where xx is the major version number and yy is the minor version number
#define SUPERBBLAS_VERSION (SUPERBBLAS_VERSION_MAJOR * 100 + SUPERBBLAS_VERSION_MINOR)

namespace superbblas {

    constexpr unsigned int version_major = SUPERBBLAS_VERSION_MAJOR;
    constexpr unsigned int version_minor = SUPERBBLAS_VERSION_MAJOR;
    constexpr unsigned int version = SUPERBBLAS_VERSION;

}

#endif
