#set text(font: "New Computer Modern", lang: "en", weight: "light", size: 11pt)
#set page(margin: 1.75in)
#set par(leading: 0.55em, spacing: 0.85em, first-line-indent: 1.8em, justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#show figure: set block(breakable: true)
#show heading: set block(above: 1.4em, below: 1em, sticky: false)
#show sym.emptyset : sym.diameter 
#show raw.where(block: true): block.with(inset: 1em, stroke: (y: (thickness: .1pt, dash: "dashed")))

#let note(body) = block(inset: 1em, stroke: (thickness: .1pt, dash: "dashed"), [#body])

#show outline.entry.where(level: 1): it => {
  show repeat : none
  v(1.1em, weak: true)
  text(size: 1em, strong(it))
}

#show raw: r => {
  show regex("t(\d)"): it => {
    box(baseline: .2em, circle(stroke: .5pt, inset: .1em, align(center + horizon, 
          text(size: .8em, strong(repr(it).slice(2, -1)))
    )))
  }

  r
}

#let reft(reft) = box(width: 9pt, height: 9pt, clip: true, radius: 100%, stroke: .5pt, baseline: 1pt,
  align(center + horizon,
    text(font: "CaskaydiaCove NF", size: 6pt, strong(str(reft)))
  )
)


#page(align(center + horizon)[
    #heading(outlined: false, numbering: none, text(size: 1.5em)[Wind Tunnel Simulation]) 
    #text(size: 1.3em)[Cicio Ionuț]
    #align(bottom, datetime.today().display("[day]/[month]/[year]"))
  ]
)

#page(outline(indent: auto))

- performance difference between rows and columns, and why
- performance when adding moving particles
- performance when adding fixed particles
- performance inlet


TODO:
- data vs task parallelism
- MPI_init
- MPI_finalize
- mpirun -n ...
- MPI_Comm_rank
- MPI_Comm_size
- MPI_Send
  - non overtaking (for the same process)
- MPI_Recv
- MPI tags
- Custom MPI communicators?
- MPI_ANY_SOURCE
- MPI_ANY_TAG
- MPI_Status
- MPI_Get_cout
- multiple mpi hosts
- Non blocking
- MPI_Wait
- MPI_Test
- MPI_Reduce
- MPI_BCast
- MPI_Allreduce
- https://engineering.fb.com/2021/07/15/open-source/fsdp
- NCCL (Nvidia, hard to try, I have only 1 gpu)
- MPI_Scatter
- MPI_Gather
- MPI_Barrier
- MPI_Allgather
- MPI_Reducescatter
- MPI_Alltoall
- MPI_Wtime (not my problem)
- distribution of timings
- PMC7 - slide 15
- comm_sz vs order of matrix (table)
  - $T_"serial" (n)$
  - $T_"parallel" (n, p)$
  - Speedup $S(n, p) = (T_"serial" (n) ) / (T_"parallel" (n, p))$
  - Ideally $S(n, p) = p$ (linear speedup)
  - speedup better when increasing problem size (PMC7 - slide 23)
  - Scalability $S(n, p) = (T_"parallel" (n, 1)) / (T_"parallel" (n, p))$
  - Efficiency $E(n, p) = S(n, p) / p = T_"serial"(n) / (p dot.c T_"parallel" (n, p))$
  - Ideally $E(n, p) = 1$ (worst with smaller examples)
  - strong vs weak scaling
    - strong: fixed size, increase the number of processes
    - weak: increase both processes and size at same scale
  - Amdahl's law
    - $T_"parallel" (p) = (1 - alpha) T_"serial" + alpha T_"serial" / p$
    - $0 <= alpha <= 1$ is the fraction that can be parallelized
    - Speedup Amdahl $S(p) = T_"serial" / ((1 - alpha) T_"serial" + alpha (T_"serial" / p))$
    - $lim_(p -> inf) S(p) = 1 / (1 - alpha)$
    - PDF7 - slide 37
  - Gustafson's law
    - $S(n, p) = (1 - alpha) + alpha p$
- MPI_Sendrecv
- In place
- MPI_Type
- MPI_Type_free

Pthread
- pthread_create
- pthread_join
- pthraad_self
- pthread_equal
- thread_handles (name for array)
- -lpthread
- pthread_mutex_t
- pthread_mutex_init
- pthread_mutex_destroy
- pthread_mutex_lock
- pthread_mutex_unlock
- pthread_mutex_trylock
- problems
  - starvation
  - deadlocks
  - produce consumer
- sem_init
- sem_destroy
- sem_post
- sem_wait
- lscup PMC 10 - slide 9
- pthread_barrier (to sync)
- pthread_cond_t
- pthread_cond_init
- pthread_cond_wait
- pthread_cond_broadcast PMC 10 - slide 58
- pthread_cond_destroy
- TODO: attributes?
- phtread_rwlock_init
- pthread_rwlock_destroy
- pthread_rwlock_rdlock
- pthread_rwlock_wrlock
- pthread_rwlock_unlock

OpenMP
- ```c #pragma omp parallel```
- X export OMP_NUM_THREADS=4
- omp_get_num_threads()
- omp_get_thread_num()
- ```c #pragma omp parallel num_threads(thread_count)```
- number of threads isn't guaranteed
- implicit barrier after block is completed
- names
  - team of threads (that execute a block in parallel)
  - master: original thread of execution
  - parent: thread that started a team of threads
  - child: each thread started by a parent
- ```c #pragma omp end parallel```
- ```c #ifndef _OPENMP #else #endif```
- ```c #pragma omp critical``` on critical section
  - it can have a name ```cpp critical(name)``` : different names can be executed at the same time
- ```c #pragma omp atomic``` only for 1 instruction, iff parallelizable
- scope: set of threads that can access the variable
  - shared: accessed by all the threads in the team (default for otuside variables)
  - private: accessed by a single thread (default for variables in scope)
- reduction operator: binary operator
- reduction: computation that repeatedly applies the same reduction operator to a sequence of operands in order to get a single result
- ```c #pragma omp parallel for reduction(<operator>: <variable list>)```
- ```c default(none)``` for scope
- ```c shared()```
- ```c private(x)``` x is not initialized
- ```c firstprivate(x)``` same as private, but value is initialized to outside value
- ```c lastprivate(x)``` same as private, but the thread with the *last iteration* sets it value outside
- ```c threadprivate(x)``` thread specific persistent storage for global data (the variable must be global or static)
- ```c copyin``` used with threadprivate to initialize threadprivate copies from the master thread's variables
- ```c single copyprivate(private_var)```
- ```c #pragma omp parallel for```
- nested:
  - invert loops... (not a bad idea)
  - collapse in one loop
  - ```c #pragma omp parallel for collapse(2)```
  - nested parallelism is disabled by default
- data dependencies PMC 14 - slide 29:
  - loop-carried dependence
  - flow dependence (RAW)
  - anti-flow dependence (WAR)
  - output dependence (WAW)
  - input dependence (RAR)
- flow dependence removal (TODO)
  1. reduction/induction variable fix
  2. loop skewing
  3. partial parallelization
  4. refactoring
  5. fissioning  
  6. algorithm change
- scheduling
  - default (0: first 10, 1: second 10 etc...)
  - cyclic (0: 1, 11, 21, etc..., 1: 2, 12, 22 etc...)
    - ```c schedule(static, 1)```, assigned before the loop is executed
    - dynamic or guided (iterations assigned while the loop is executing)
    - auto (compiler / runtime system decides)
    - runtime (runtime system decides)
- omp_lock_t write_lock;
- omp_init_lock
- omp_set_lock
- omp_unset_lock
- omp_destroy_lock
- master/single directives:
  - only 1 thread executes it
  - with "single" a barrier is put at the end of the block
- ```c #pragma omp barrier```
- ```c #pragma omp parallel sections```
  - ```c #pragma omp section```
- ```c #pragma omp for ordered```
  - ```c #pragma omp ordered```
- PMC15 - slide 54 MPI + omp/pthread

CUDA
- GPU and CPU memories are disjoint
- not the same floating point representation
- typical program
  - allocate GPU memory
  - transfer from host to GPU
  - run CUDA kernel
  - copy results from GPU to host
- 6D structure
  - each thread has a (x, y, z) in a block
  - each block has a (x, y, z) in a grid
  - PMC16 - slide 26 (different capability, different sizes possible)
  - TODO: get capability of my CPU (6.1)
- kernel (function)
  - dim3 block(3, 2)
    - max 1024 threads per block
  - dim3 grid(4, 3, 2)
  - always void
  - types:
    - `__global__`: called from both device and host; from CC 3.5 can be executed on host too
    - `__device__`: runs on the GPU and can be called only by a kernel
    - `__host__`: used in combination with `__device__` to generate 2 codes
  - `threadIdx`: position of thread within block
  - `blockIdx`: position of thread's block within grid
  - `int myId = stuff`
- threads are executed in *warps* (size 32)
  - threads in a *block* are split in *warps*
  - multiple warps can be executed together, but at different execution paths
  - warp divergence

- CUDA device (8 blocks per SM, 1024 threads per SM, 512 threads per block)
  - $8 times 8$ blocks
    - 64 threads per block
    - 1024 / 64 blocks needed (which is 16, and not all are used)
  - $16 times 16$
    - 256 threads per block
    - 1024 / 256 = 4 which is a good amount of blocks: 4
  - $32 times 32$
    - 1024 threads per block
    - more than the 512 allowed
- cudaMalloc()
- cudaFree()
- cudaMemcpy()
  - cudaMemcpyHostToHost
  - cudaMemcpyHostToDevice
  - cudaMemcpyDeviceToHost
  - cudaMemcpyDeviceToDevice
  - cudaMemcpyDefault (when Unified Virtual Address space capable)
  - PMC 16 - slide 64
- Memory Types
  - registers: hold local variables
    - store local variables
    - split among resident threads
    - maxNumRegisters determined by CC // TODO: determin maxNumRegisters
    - if number is exceeded, variables are stored in global
    - decided by the compiler
    - nvcc --Xptxas
    - *occupancy* = $"resident_warps" / "maximum_warps"$
      - close to 1 is desirable (to hide latencies)
      - analyzed via profiler
  - shared memory: fast on-chip, used to hold frequently used data (used to exchange data between cores of the same SM)
    - used like L1 "user-managed" cache
    - `__shared__` specifier used to indicate data that must go on shared memory
    - void `__syncthreads()` used to avoid WAR, RAW, WAW hazards
  - L1/L2 cache: transparent to the programmer
  - other
    - global memory: off-chip, high capacity, relatively slow (only part accessible by the host)
    - texture and surface memory: permits fast implementation of filtering/interpolation operator
    - constant memory: cached, non modifiable, broadcast constants to all threads in a wrap
  - variable declaration:
    - automatic variables (no arrays): *registers* (scope thread) - lifetime kernel
    - automatic arrays: *global memory* (scope thread) - lifetime kernel
    - `__device__` `__shared__` int sharedVar; *shared memory* (scope block) - lifetime kernel
    - `__device__` int globalVar; *global memory* (scope grid) - lifetime application
    - `__device__` `__constant__` int constVar; *constant memory* (scope grid) - lifetime application
  - Roofline model
    - https://docs.nersc.gov/tools/performance/roofline/
    - https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
    - $beta$ = 0.5GB/s
    - PMC20
    - PMC18
  - matrix multiplication
  - atomicAdd
  - MPI+GPU
    - PMC20j


- cudaDeviceSynchronize(); // barrier
- nvcc --arch=sm_20 (cc 2.0) // cuda capability
  

Other
- message matching
  - recv type = send type
  - recv buff size > send buff size
- P2P
  - buffered
  - synchronous
  - ready

#page(bibliography("bibl.bib"))

// rows=250 
// columns=250 
// max_iter=2000 
// var_threshold=0.5
// inlet_pos=0 
// inlet_size=$rows
// particles_f_band_pos=1 
// particles_f_band_size=$((rows-1)) 
// particles_f_density=0.5
// particles_m_band_pos=1 
// particles_m_band_size=$((rows-1)) 
// particles_m_density=0.5
// short_rnd1=$((RANDOM % 10000 + 1)) short_rnd2=$((RANDOM % 10000 + 1)) short_rnd3=$((RANDOM % 10000 + 1))

// = Introduction
//
// You are provided with a sequential code that simulates in a simplified way, how the air pressure spreads in a Wind Tunnel, with fixed obstacles and flying particles. 
//
// - the simulation represents a zenithal view of the tunnel, by a 2-dimensional cut of the tunnel along the main air-flow axis
// - the space is modelled as a set of discrete and evenly spaced points or positions 
// - a *bidimensional array* is used to represent the value of *air pressure at each point* in a given *time step*
// - another *unidimensional array* stores the information of a collection of *flying particles* (they can be pushed and moved by the air flow), or *fixed structures* that cannot move
// - all particles have a *mass* and a *resistance* to the flow of air, partially _blocking_ and _deviating_ it
//
//
// / Fan inlet:
// - there is an air inlet at the start of the tunnel with a big fan
// - the values of air pressure introduced by the fan are represented in the row 0 of the array
// - after a given number of simulation steps the fan *turning-phase* is slightly changed recomputing the values of air pressure introduced, also adding some random noise 
// - the fan phase is cyclically repeated
// - every time step, the values of air pressure in the array are updated with the old values in upper rows, propagating the air flow to the lower rows
// - on each iteration only some rows are updated
// - they represent wave-fronts, separated by a given number of rows.
//
// / Fixed and flying particles:
// - the information of a particle includes:
//   - position coordinates
//   - a two dimensional velocity vector
//   - mass
//   - and air flow resistance
// - all of them are numbers with decimals represented in fixed position 
// - the particles let some of the air flow to advance depending on their resistance 
// - the air flow that is blocked by the particle is deviated creating more pressure in the front of the particle
// - the flying particles are also pushed by the air when the pressure in their front is enough to move their mass
// - the updates of deviated pressure and flying particles movements are done after a given number of time steps (the same as the distance between propagation wave fronts)
//
// / End of simulation:
// - the simulation ends:
//   - after a fixed number of iterations (input argument) 
//   - or if the maximum variation of air pressure in any cell of the array is less than a given threshold
// - at the end, the program outputs a results line with:
//   - the number of iterations done 
//   - the value of the maximum variability of an update in the last iteration
//   - and the air pressure value of several chosen array positions in three rows: 
//     - near the inlet 
//     - in the middle of the tunnel
//     - in the last row of the tunnel; 
//     - or in the last row updated by the first wave front in case that the simulation stops before the air flow arrives at the last row
//
//
// Symbols:
//   - `[ ]` One or more particles/obstacles in the position
//   - `.` Air pressu in the range $[0, 0.5)$
//   - `n` A number $n$ represents a pressure between $[n, n + 1)$
//   - `*` Air pressure equal or greater than 10
//
// #align(center, {
//   set text(font: "CaskaydiaCove NF")
//   image("wind.svg")
// })
//
// = Scalabilità
//
// - rispetto alla dimensione della griglia
// - rispetto alla dimensione dell'inlet
// - rispetto al numero di particelle e al tipo
//
// Il tutto testato con diversi numeri casuali (usando numeri primi)
//
// = Correttezza
// - valgrind e roba cicli CPU etc...
// - perf stat
// - gprof
//
// = Architetture
//
// - statistiche varie architetture e roba simile 
//   - getconf LEVEL3_CACHE_ASSOC
//   - lscpu| grep -E '^Thread|^Core|^Socket|^CPU\('
//
// = Alcune note 
//
// #note[
// / Task parallelism:
// - Partition various tasks among the cores
// - The “temporal parallelism” we have seen in circuits is a specific type of task parallelism
//
// / Data parallelism:
// - Partition the data used in solving the problem among the cores
// - Each core carries out similar operations on it’s part of the data
// - Similar to “spatial parallelism” in circuits
// ]
//
//
// Which one can i use here? Maybe both?
// Tasks:
// - update the inlet
// - move the particles  
// - update the flow
// - propagate
//
// To propagate I need all the above... I can propagate multiple times for one of the above (a.k.a 8 times)
//
// If I want to parallelize the data, I have to work on distributing the data of the matrices
// The problem with distributing the data is that some data depends on other data... maybe? Not as much as the tasks.
//
// #note[
// In practice, cores need to coordinate, for different reasons:
// / Communication: e.g., one core sends its partial sum to another core
// / Load balancing: share the work evenly so that one is not heavily loaded (what if some file to compress is much bigger than the others?)
// - If not, p-1 one cores could wait for the slowest (wasted resources & power!)
// / Synchronization: each core works at its own pace, make sure some core does not get too far ahead
// - E.g., one core fills the list of files to compress. If the other ones start too early they might miss some files
// ]
//
// Load balancing is secondary here... can be done at the start probably. Communication and Synchronization are the important ones, but how?
//
// #note[
// / Foster’s methodology:
//
// / Partitioning: divide the computation to be performed and the data operated on by the computation into small tasks. The focus here should be on identifying tasks that can be executed in parallel.
//
// / Communication: determine what communication needs to be carried out among the tasks identified in the previous step
//
// / Agglomeration or aggregation: combine tasks and communications identified in the first step into larger tasks. For example, if task A must be executed before task B can be executed, it may make sense to aggregate them into a single composite task. 
//
// / Mapping: assign the composite tasks identified in the previous step to processes/threads. This should be done so that communication is minimized, and each process/thread gets roughly the same amount of work.
// ]
// PMC7
// PMC8 datatypes


