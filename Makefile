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

# Flags for optimization and libs
FLAGS=-O3 -Wall
LIBS=-lm

# Targets to build
OBJS=wind_seq wind_omp wind_mpi wind_cuda wind_pthread wind_pthread_2

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
	$(CUDACC) $(DEBUG) $< $(LIBS) -o $@

wind_pthread: wind_pthread.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -lpthread -o $@
 
wind_pthread_2: wind_pthread_2.c
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -lpthread -o $@

# DEBUGGING

wind_seq_debug: wind.c
	$(CC) -Wall -pedantic -ggdb3 $< $(LIBS) -o $@

wind_mpi_debug: wind_mpi.c
	$(MPICC) -Wall -pedantic -ggdb3 $< $(LIBS) -o $@

wind_omp_debug: wind_omp.c
	$(CC) -Wall -pedantic -ggdb3 $(OMPFLAG) $< $(LIBS) -o $@
	
wind_pthread_debug: wind_pthread.c
	$(CC) -Wall -pedantic -ggdb3 -lpthread $< $(LIBS) -o $@

wind_pthread_2_debug: wind_pthread_2.c
	$(CC) -Wall -pedantic -ggdb3 -lpthread $< $(LIBS) -o $@

# Remove the target files
clean:
	rm -rf $(OBJS) wind_pthread wind_pthread_2 wind_seq_debug wind_omp_debug wind_pthread_debug wind_pthread_2_debug wind_mpi_debug *.txt

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" FLAGS= all
