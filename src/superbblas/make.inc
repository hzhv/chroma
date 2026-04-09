
#
# If SUPERBBLAS_WITH_MPI is yes, then include support for MPI
#
SUPERBBLAS_WITH_MPI ?= $(if $(findstring mpi,${CXX}),yes,no)
ifeq ($(SUPERBBLAS_WITH_MPI), yes)
    MPISBFLAG ?= -DSUPERBBLAS_USE_MPI -DSUPERBBLAS_USE_MPIIO
endif

#
# If SUPERBBLAS_WITH_ANARCHOFS is yes, then include support for it
#
SUPERBBLAS_WITH_ANARCHOFS ?= $(if $(findstring undefined,$(origin ANARCHOFS_HOME)),no,yes)
ifeq ($(SUPERBBLAS_WITH_ANARCHOFS), yes)
    AFSFLAGS ?= -DSUPERBBLAS_USE_ANARCHOFS -I$(ANARCHOFS_HOME)
endif

#
# Compiler for cpu target, CXX, compiler flags when compiling for cpu target and for NVCC -Xcompiler
# when compiling for cuda, and general linking flags for all targets
#
CXX ?= c++
ifndef MKLROOT
  BLAS_LDFLAGS ?= -lopenblas -DSUPERBBLAS_USE_CBLAS
else
  BLAS_FLAGS ?= -I${MKLROOT}/include -DSUPERBBLAS_USE_MKL
  BLAS_LDFLAGS ?= -L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -lm -ldl
endif


#CXXFLAGS ?= -g -O0 -Wall -Wextra -fopenmp $(MPISBFLAG) $(BLAS_FLAGS) $(AFSFLAGS)
CXXFLAGS ?= -Ofast -march=native -DNDEBUG -Wall -Wextra -fopenmp $(MPISBFLAG) $(BLAS_FLAGS) $(AFSFLAGS)
LDFLAGS ?= $(BLAS_LDFLAGS)

#
# Compiler for cuda target, compiler flags, and linking flags
#
NVCC ?= $(shell which nvcc)
CUDADIR ?= $(shell dirname $(NVCC))/..
#NVCCFLAGS ?= -x cu -g -G -arch=sm_75
NVCCFLAGS ?= -x cu -O3 -arch=sm_75
NVCCLDFLAGS ?= -lcusolver -lcusparse -lcublas -lcudart

#
# Compiler for rocm target, compiler flags, and linking flags
#
PRIM ?= 
THRUST ?= 
HIP ?= $(shell which hipcc)
ROCMDIR ?= $(shell hipconfig -R)
HIPFLAGS ?= $(shell hipconfig -C) -I$(ROCMDIR)/include -I$(PRIM) -I$(THRUST) -std=c++14
HIPLDFLAGS ?= -L$(ROCMDIR)/lib -lrocsolver -lhipsparse -lhipblas -lrocblas -lamdhip64

#
# Building directory
#
BUILDDIR ?= $(PWD)/build

#
# Installation
#
PREFIX ?= $(BUILDDIR)
includedir ?= $(PREFIX)/include
libdir ?= $(PREFIX)/lib
LIBRARY = libsuperbblas.a

#
# Operating system
#
UNAME ?= $(shell uname)

#
# Other tools
#
RANLIB ?= ranlib
PYTHON ?= $(if $(findstring 1984, $(shell python -c 'print(1984)')), \
  python, \
  $(if $(findstring 1984, $(shell python2 -c 'print(1984)')),python2,python3))

