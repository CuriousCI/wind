#
# Wind-tunnel
#
# Parallel computing (Degree in Computer Engineering)
# 2020/2021
#
# (c) 2021 Arturo Gonzalez-Escribano
# Grupo Trasgo, Universidad de Valladolid (Spain)
#

# Compilers
CC=gcc
OMPFLAG=-fopenmp
MPICC=mpicc
CUDACC=nvcc
ARCH=-arch=sm_61

# Flags for optimization and libs
FLAGS=-O3 -Wall
LIBS=-lm

# Targets to build
OBJS=wind_seq # wind_omp wind_mpi wind_cuda wind_mpi_omp wind_mpi_cuda wind_omp_cuda

# Rules. By default show help
help:
	@echo
	@echo "Wind tunnel"
	@echo
	@echo "Group Trasgo, Universidad de Valladolid (Spain)"
	@echo
	@echo "make wind_seq	Build only the reference sequential version"
	@echo "make wind_omp	Build only the OpenMP version"
	@echo "make wind_mpi	Build only the MPI version"
	@echo "make wind_cuda	Build only the CUDA version"
	@echo
	@echo "make all	Build all versions (Sequential, OpenMP)"
	@echo "make debug	Build all version with demo output for small surfaces"
	@echo "make clean	Remove targets"
	@echo

all: $(OBJS)

wind_seq: wind.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

wind_omp: wind_omp.c
	$(CC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

wind_mpi: wind_mpi.c
	$(MPICC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

wind_cuda: wind_cuda.cu
	$(CUDACC) $(ARCH) -O3 $(DEBUG) $< $(LIBS) -o $@
	
wind_mpi_omp: wind_mpi_omp.c
	$(MPICC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

wind_omp_cuda: wind_omp_cuda.cu
	$(CUDACC) $(ARCH) -O3 $(DEBUG) -Xcompiler -fopenmp $< $(LIBS) -o $@
	
wind_mpi_cuda: wind_mpi_cuda.cu
	$(CUDACC) $(ARCH) $< -o $@ -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

# Remove the target files
clean:
	rm -rf $(OBJS) *.txt

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" FLAGS= all

