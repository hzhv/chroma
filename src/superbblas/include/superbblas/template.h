#ifndef __SUPERBBLAS_TEMPLATE__
#define __SUPERBBLAS_TEMPLATE__

#include <cfloat>
#include <climits>
#include <cstdint>

#include "template_types.h"

// Macro __SUPERBBLAS_USE_REAL - only defined for a non-complex type
// Macro __SUPERBBLAS_USE_COMPLEX - only defined for a complex type.

#if defined(__SUPERBBLAS_USE_HALF) || defined(__SUPERBBLAS_USE_FLOAT) ||                           \
    defined(__SUPERBBLAS_USE_DOUBLE)
#    define __SUPERBBLAS_USE_REAL
#elif defined(__SUPERBBLAS_USE_HALFCOMPLEX) || defined(__SUPERBBLAS_USE_FLOATCOMPLEX) ||           \
    defined(__SUPERBBLAS_USE_DOUBLECOMPLEX)
#    define __SUPERBBLAS_USE_COMPLEX
#else
#    error No template macro defined
#endif

// Macro ARITH(H,K,S,C,D,Z,Q,W) - Return an argument depending on the __SUPERBBLAS_USE_* defined

#if defined(__SUPERBBLAS_USE_DOUBLE)
#    define ARITH(H, K, S, C, D, Z, Q, W) D
#    define REAL_ARITH(H, K, S, C, D, Z, Q, W) D
#elif defined(__SUPERBBLAS_USE_DOUBLECOMPLEX)
#    define ARITH(H, K, S, C, D, Z, Q, W) Z
#    define REAL_ARITH(H, K, S, C, D, Z, Q, W) D
#elif defined(__SUPERBBLAS_USE_FLOAT)
#    define ARITH(H, K, S, C, D, Z, Q, W) S
#    define REAL_ARITH(H, K, S, C, D, Z, Q, W) S
#elif defined(__SUPERBBLAS_USE_FLOATCOMPLEX)
#    define ARITH(H, K, S, C, D, Z, Q, W) C
#    define REAL_ARITH(H, K, S, C, D, Z, Q, W) S
#elif defined(__SUPERBBLAS_USE_HALF)
#    define ARITH(H, K, S, C, D, Z, Q, W) H
#    define REAL_ARITH(H, K, S, C, D, Z, Q, W) H
#elif defined(__SUPERBBLAS_USE_HALFCOMPLEX)
#    define ARITH(H, K, S, C, D, Z, Q, W) K
#    define REAL_ARITH(H, K, S, C, D, Z, Q, W) H
#elif defined(__SUPERBBLAS_USE_QUAD)
#    define ARITH(H, K, S, C, D, Z, Q, W) Q
#    define REAL_ARITH(H, K, S, C, D, Z, Q, W) Q
#elif defined(__SUPERBBLAS_USE_QUADCOMPLEX)
#    define ARITH(H, K, S, C, D, Z, Q, W) W
#    define REAL_ARITH(H, K, S, C, D, Z, Q, W) Q
#else
#    error No template macro defined
#endif

#ifndef __SUPERBBLAS_TEMPLATE_PRIVATE__
#    define __SUPERBBLAS_TEMPLATE_PRIVATE__
#    define CONCAT(a, b) CONCATX(a, b)
#    define CONCATX(a, b) a##b
#    define STR(X) STR0(X)
#    define STR0(X) #X

#    define MACHINE_EPSILON                                                                        \
        ARITH(0.000977, 0.000977, FLT_EPSILON, FLT_EPSILON, DBL_EPSILON, DBL_EPSILON, 1.92593e-34, \
              1.92593e-34)

#    define MACHINE_MAX                                                                            \
        ARITH(65504.0, 65504.0, FLT_MAX, FLT_MAX, DBL_MAX, DBL_MAX, FLT128_MAX, FLT128_MAX)

#    ifdef F77NOUNDERSCORE
#        define FORTRAN_FUNCTION(X) X
#    else
#        define FORTRAN_FUNCTION(X) CONCAT(X, _)
#    endif

#endif //  __SUPERBBLAS_TEMPLATE_PRIVATE__

#endif // __SUPERBBLAS_TEMPLATE__
