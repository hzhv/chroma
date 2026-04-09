#ifndef __SUPERBBLAS_SUPERBBLAS_LIB__
#define __SUPERBBLAS_SUPERBBLAS_LIB__

#include "superbblas_flags.h"

/// Generates EMIT # define X; use this macro to inject macros in the automatically generated file
/// `superbblas_flags.h`.
///
/// NOTE: REPLACE(" ",) is going to remove all double quotes `"` and this is a workaround to the
///       limitation of macros to produce `#` and to avoid the macro argument to be expanded

#define EMIT_define(X) EMIT REPLACE(" ", ) "#" define #X

#ifdef SUPERBBLAS_CREATING_FLAGS
/// When the macro SUPERBBLAS_LIB is defined, the library is not being used as a header-only
/// library, and the definition of the already compiled functions should be hided (see macro
/// `IMPL`)

EMIT_define(SUPERBBLAS_LIB)
#endif

#ifdef SUPERBBLAS_CREATING_LIB
#    define DECL(...) __VA_ARGS__
#else
#    define DECL(...)
#endif

/// On header-only mode, the macro just returns the arguments; otherwise it consumes them and
/// replaced them by `;`

#ifdef SUPERBBLAS_LIB
#    define IMPL(...) ;
#else
#    define IMPL(...) __VA_ARGS__
#endif

#endif // __SUPERBBLAS_SUPERBBLAS_LIB__
