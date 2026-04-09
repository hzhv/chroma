###
# For each of the source files in ${SOURCES}, this makefile defines the
# actions %_cpu and %_cuda to compile with CXX and NVCC respectively.
# If ${CXX} contains mpi, then the macro SUPERBBLAS_USE_MPI is defined to
# activate the MPI portion of the tests and the library.
#
# Examples of actions:
#
#  make local_cpu             # builds local.cpp with ${CXX} compiler
#  make local_cuda dist_cuda  # builds both sources with ${NVCC} compiler
#  make dist_cuda CXX=mpicxx  # builds dist.cpp with ${NVCC} adding the flags
#                             # and linking information for MPI

SHELL = /bin/bash
BUILDDIR ?= ..
SB_LDFLAGS ?=

include ../make.inc

SOURCES := blas.cpp dist.cpp contract.cpp storage.cpp storage_details.cpp bsr.cpp dense.cpp bsr_hist.cpp tenfucks.cpp

CPU_TARGETS := $(patsubst %.cpp,%_cpu,$(SOURCES))
CUDA_TARGETS := $(patsubst %.cpp,%_cuda,$(SOURCES))
HIP_TARGETS := $(patsubst %.cpp,%_hip,$(SOURCES))
CPU_LIB_TARGETS := $(patsubst %.cpp,%_cpu_lib,$(SOURCES))
CUDA_LIB_TARGETS := $(patsubst %.cpp,%_cuda_lib,$(SOURCES))
HIP_LIB_TARGETS := $(patsubst %.cpp,%_hip_lib,$(SOURCES))

SB_INCLUDE ?= -I$(BUILDDIR)/include -I$(BUILDDIR)/include/xsimd/include
MPISBFLAG ?= $(if $(findstring mpi,${CXX}),-DSUPERBBLAS_USE_MPI,)

comma := ,
MPIINCFLAGS ?= $(if $(findstring mpi,${CXX}),$(patsubst -Wl$(comma)%,,$(shell ${CXX} -showme:compile 2>/dev/null || ${CXX} -show -CC=)),)
MPILDFLAGS ?= $(if $(findstring mpi,${CXX}),\
        $(filter-out -pthread -fexceptions, \
            $(subst -Wl$(comma),-Xlinker ,$(shell ${CXX} -showme:link 2>/dev/null || ${CXX} -show -CC=))),)
SB_LDFLAGS := -L$(BUILDDIR)/lib -lsuperbblas
CUDAINCLUDE ?= -I$(CUDADIR)/include
CUDALDFLAGS ?= -L$(CUDADIR)/lib64
NVCCSTDLANG ?= -std c++14

all_cpu: $(CPU_TARGETS)
all_cuda: $(CUDA_TARGETS)
all_hip: $(HIP_TARGETS)
all_cpu_lib: $(CPU_LIB_TARGETS)
all_cuda_lib: $(CUDA_LIB_TARGETS)
all_hip_lib: $(HIP_LIB_TARGETS)

$(CPU_TARGETS): %_cpu: %.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${LDFLAGS}

$(CPU_LIB_TARGETS): %_cpu_lib: %.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${SB_LDFLAGS} ${LDFLAGS}

storage_details: storage_details.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${LDFLAGS}

$(CUDA_TARGETS): %_cuda: %.cpp
	${NVCC} ${SB_INCLUDE} ${MPISBFLAG} ${NVCCFLAGS} ${NVCCSTDLANG} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' \
	        ${NVCCLDFLAGS} ${MPILDFLAGS} ${LDFLAGS} ${CUDA_EXTRA_FLAGS} $< -o $@

$(CUDA_LIB_TARGETS): %_cuda_lib: %.cpp
	${CXX} ${SB_INCLUDE} ${CUDAINCLUDE} ${MPISBFLAG} ${CXXFLAGS} $< -o $@ ${SB_LDFLAGS} ${CUDALDFLAGS} ${NVCCLDFLAGS} ${LDFLAGS}

$(HIP_TARGETS): %_hip: %.cpp
	${HIP} ${SB_INCLUDE} ${MPISBFLAG} ${HIPFLAGS} ${HIPSTDLANG} ${CXXFLAGS} ${MPIINCFLAGS} \
	       ${HIPLDFLAGS} ${MPILDFLAGS} ${LDFLAGS} $< -o $@

$(HIP_LIB_TARGETS): %_hip_lib: %.cpp
	${CXX} ${SB_INCLUDE} ${MPISBFLAG} ${HIPFLAGS} ${HIPSTDLANG} ${CXXFLAGS} ${MPIINCFLAGS} \
	       ${SB_LDFLAGS} ${HIPLDFLAGS} ${MPILDFLAGS} ${LDFLAGS} $< -o $@

all_cpu all_cpu_lib all_cuda all_cuda_lib all_hip all_hip_lib: all_%:
	SB_TRACK_MEM=1 ./blas_$*
	SB_TRACK_MEM=1 ./storage_$*
	SB_TRACK_MEM=1 SB_DEBUG=5 ./bsr_$* --dim='2 2 2 2 2 2'
ifeq ($(SUPERBBLAS_WITH_MPI), yes)
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_TRACK_MEM=1 SB_DEBUG=5 mpirun -np 6 --oversubscribe ./dist_$*  --procs='1 1 2 3' --dim='4 4 4 4 64'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_TRACK_MEM=1 SB_DEBUG=5 mpirun -np 3 --oversubscribe ./dist_$*  --procs='1 1 1 3' --dim='4 4 4 2 64'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_TRACK_MEM=1 SB_DEBUG=5 mpirun -np 4 --oversubscribe ./dense_$*  --procs='1 1 2 2' --dim='4 4 4 4 4 4'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_TRACK_MEM=1 SB_DEBUG=5 mpirun -np 4 --oversubscribe ./bsr_$*  --procs='1 1 2 2' --dim='4 4 4 4 4 4'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_TRACK_MEM=1 SB_DEBUG=5 mpirun -np 4 --oversubscribe ./bsr_$*  --procs='1 1 2 2' --dim='4 4 4 4 4 4' --components=2
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_TRACK_MEM=1 SB_DEBUG=0 mpirun -np 6 --oversubscribe ./contract_$*
endif

test_dist_cpu test_dist_cuda test_dist_hip test_dist_cpu_lib test_dist_cuda_lib test_dist_hip_lib: test_dist_%:
	for proc_geom in "1 1 1 1 1" "2 1 1 1 1" \
			 "1 2 1 1 1" "2 2 1 1 1" "4 2 1 1 1" \
			 "2 2 2 2 2" "3 2 2 2 2" \
			 "2 8 2 2 2" "16 8 2 2 2" \
			 "3 16 8 2 2" "16 16 8 2 2" \
			 "3 8 8 8 8" "16 8 8 8 8"; \
	do \
		echo $$proc_geom | while read p geom; do \
			echo OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np $$p --oversubscribe ./dist_$*  --dim="$$geom 2"; \
			OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np $$p --oversubscribe ./dist_$*  --dim="$$geom 2"; \
		done; \
	done

hist_cpu hist_cuda hist_hip: hist_%:
	@rm -rf repo
	@git clone https://github.com/eromero-vlc/superbblas repo &> /dev/null
	@echo "#" commit mpi_1rhs mpi_12rhs mpi_48rhs thr_1rhs thr_12rhs thr_48rhs
	@for commit in _local_ `git log --pretty='%h' 411a5e0fd172fc2e12ddc17907cbb820e3e8576b..HEAD`; do \
		( cd repo && git checkout $$commit &> /dev/null ); \
		for ver in 0 1 2; do \
			if [ $$commit == _local_ ]; then \
				make bsr_hist_$* BUILDDIR=.. CXX=${CXX} CXXFLAGS="${CXXFLAGS} -DVER=$$ver" &> /dev/null || continue; \
			else \
				make bsr_hist_$* BUILDDIR=repo CXX=${CXX} CXXFLAGS="${CXXFLAGS} -DVER=$$ver" &> /dev/null || continue; \
			fi; \
			r1="`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   mpirun -np 8 ./bsr_hist_$* --dim='8 8 8 8 1 12' --procs='1 2 2 2'`"; \
			r12="`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  mpirun -np 8 ./bsr_hist_$* --dim='8 8 8 8 12 12' --procs='1 2 2 2'`"; \
			r48="`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  mpirun -np 8 ./bsr_hist_$* --dim='8 8 8 8 48 12' --procs='1 2 2 2'`"; \
			r1t="`OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1  mpirun --bind-to none -np 1 ./bsr_hist_$* --dim='8 8 8 8 1 12' --procs='1 1 1 1'`"; \
			r12t="`OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 mpirun --bind-to none -np 1 ./bsr_hist_$* --dim='8 8 8 8 12 12' --procs='1 1 1 1'`"; \
			r48t="`OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 mpirun --bind-to none -np 1 ./bsr_hist_$* --dim='8 8 8 8 48 12' --procs='1 1 1 1'`"; \
			echo $$commit $$r1 $$r12 $$r48 $$r1t $$r12t $$r48t; \
			break; \
		done; \
	done

clean:
	rm -f ${CPU_TARGETS} ${CPU_LIB_TARGETS} ${CUDA_TARGETS} ${CUDA_LIB_TARGETS} ${HIP_TARGETS} ${HIP_LIB_TARGETS} storage_details

known_issues: dist_cpu dense_cuda
	OPENBLAS_NUM_THREADS=1 ./dense_cuda --dim='200000 1 1 1 4 4'
	# Hangs for Open MPI 4.0.3 since commit 3530391ab167425aacee9e42b3abcacd9c85b117 (already fixed)
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -np 3 ./dist_cpu  --procs='1 1 1 3' --dim='1 1 1 2 4'
	# MPI_Ialltoallv hungs (already fixed)
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np 4 ./dist_cpu --procs='2 1 1 2' --dim='2 1 1 2 1'
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SB_DEBUG=5 mpirun -np 6 ./dist_cpu --procs='1 1 2 3' --dim='4 4 4 4 64'

bsr_cuda_nsys: export CUDA_EXTRA_FLAGS := -DSUPERBBLAS_USE_NVTX -lnvToolsExt
bsr_cuda_nsys: bsr_cuda
	rm -f bsr_cuda_nsys.nsys-rep
	OPENBLAS_NUM_THREADS=1 SB_TRACK_TIME=0 SB_MPI_GPU=0 nsys profile -t nvtx,cuda -o bsr_cuda_nsys ./bsr_cuda --dim="8 16 16 16 16 12" --rep=2 --components=4 

.PHONY: clean ${CPU_TARGETS} ${CPU_LIB_TARGETS} ${CUDA_TARGETS} ${CUDA_LIB_TARGETS} ${HIP_TARGETS} ${HIP_LIB_TARGETS} storage_details
