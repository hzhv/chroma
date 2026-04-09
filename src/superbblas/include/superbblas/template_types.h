#ifndef __SUPERBBLAS_TEMPLATE_TYPES__
#define __SUPERBBLAS_TEMPLATE_TYPES__

// If THIS_FILE is defined, then THIS_FILE is going to be self-included several times, each time
// with one of the following macros defined:
// - __SUPERBBLAS_USE_HALF
// - __SUPERBBLAS_USE_HALFCOMPLEX
// - __SUPERBBLAS_USE_FLOAT
// - __SUPERBBLAS_USE_FLOATCOMPLEX
// - __SUPERBBLAS_USE_DOUBLE
// - __SUPERBBLAS_USE_DOUBLECOMPLEX

#ifndef THIS_FILE
#    error "Please define THIS_FILE"
#endif
#ifdef THIS_FILE

// #define SHOW_TYPE

#    include "template_undef.h"

#    ifdef SHOW_TYPE
#        warning compiling double
#    endif
#    define __SUPERBBLAS_USE_DOUBLE
#    include THIS_FILE
#    include "template_undef.h"
#    undef __SUPERBBLAS_USE_DOUBLE

#    ifdef SHOW_TYPE
#        warning compiling half
#    endif
#    define __SUPERBBLAS_USE_HALF
#    include THIS_FILE
#    include "template_undef.h"
#    undef __SUPERBBLAS_USE_HALF

#    ifdef SHOW_TYPE
#        warning compiling half complex
#    endif
#    define __SUPERBBLAS_USE_HALFCOMPLEX
#    include THIS_FILE
#    include "template_undef.h"
#    undef __SUPERBBLAS_USE_HALFCOMPLEX

#    ifdef SHOW_TYPE
#        warning compiling float
#    endif
#    define __SUPERBBLAS_USE_FLOAT
#    include THIS_FILE
#    include "template_undef.h"
#    undef __SUPERBBLAS_USE_FLOAT

#    ifdef SHOW_TYPE
#        warning compiling float complex
#    endif
#    define __SUPERBBLAS_USE_FLOATCOMPLEX
#    include THIS_FILE
#    include "template_undef.h"
#    undef __SUPERBBLAS_USE_FLOATCOMPLEX

#    ifdef SHOW_TYPE
#        warning compiling double complex
#    endif
#    define __SUPERBBLAS_USE_DOUBLECOMPLEX
#    include "template.h"
// #include THIS_FILE
// #include "template_undef.h"
// #undef __SUPERBBLAS_USE_DOUBLECOMPLEX

// #define __SUPERBBLAS_USE_QUAD
// #include THIS_FILE
// #include "template_undef.h"
// #undef __SUPERBBLAS_USE_QUAD
// #define __SUPERBBLAS_USE_QUADCOMPLEX
// #include THIS_FILE
// #include "template_undef.h"
// #undef __SUPERBBLAS_USE_QUADCOMPLEX

#    undef THIS_FILE
#endif // THIS_FILE

#endif // __SUPERBBLAS_TEMPLATE_TYPES__
