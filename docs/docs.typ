#set text(font: "New Computer Modern", lang: "en", weight: "light", size: 11pt)
#set page(margin: 1.75in)
#set par(leading: 0.55em, spacing: 0.85em, first-line-indent: 1.8em, justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#show figure: set block(breakable: true)
#show figure.caption: set align(center)
#show heading: set block(above: 1.4em, below: 1em)
#show outline.entry.where(level: 1): it => { show repeat : none; v(1.1em, weak: true); text(size: 1em, strong(it)) }
#show raw.where(block: true): block.with(inset: 1em, width: 100%, fill: luma(254), stroke: (left: 5pt + luma(245), rest: 1pt + luma(245)))
#show figure.where(kind: raw): it => { set align(left); it }
#show raw: set text(font:"CaskaydiaCove NFM", lang: "en", weight: "light", size: 9pt)
#show sym.emptyset : sym.diameter 

#let reft(reft) = box(width: 8pt, place(dx: -1pt, dy: -7pt, 
  box(radius: 100%, width: 9pt, height: 9pt, inset: 1pt, stroke: .5pt, // fill: black,
    align(center + horizon, text(font: "CaskaydiaCove NFM", size: 7pt, repr(reft)))
  )
))

#show raw: r => { show regex("t(\d)"): it => { reft(int(repr(it).slice(2, -1))) }; r }

// #let note(body) = block(inset: 1em, stroke: (thickness: .1pt, dash: "dashed"), [#body])
// #let note(body) = block(width: 100%, inset: 1em, stroke: (paint: silver, dash: "dashed"), body)
#let note(body) = box(width: 100%, inset: 1em, fill: luma(254), stroke: (paint: silver), body)
// #let note(body) = block(inset: 1em, fill: luma(254), stroke: (thickness: 1pt, paint: luma(245)), body)

#page(align(center + horizon)[
    #heading(outlined: false, numbering: none, text(size: 1.5em)[Wind Tunnel Simulation]) 
    #text(size: 1.3em)[Cicio Ionuț] \
    #text(size: 1em)[#link("https://github.com/CuriousCI/wind")[https://github.com/CuriousCI/wind]]
      
    #align(bottom, datetime.today().display("[day]/[month]/[year]"))
  ]
)

// #page(outline(indent: auto))

#set page(numbering: "1")

= Wind

#quote(attribution: [`handout.pdf`, page 1 Introduction], block: true)[
The software simulates, in a simplified way, how the air pressure spreads in a Wind Tunnel, with *fixed obstacles* and *flying particles*. [...] All particles have a mass and a resistance to the flow air, partially blocking and deviating it.
]

The sequential code has two main parallelization targets: the *update of the air flow*, and the *interactions with the partilces*; the latter is a big obstacle to parallelization due to a few important reasons:

- the ```c move_particle()``` function is very expensive and requires a copy of the matrix with the air flow
- accesses to memory depend on the positions of the particles 
- air flow positions with particles don't need to be always updated 
- *some* the particles move, which has both pros and cons: 
  - pre-calculating the optimal way to access the memory is virtually impossible, but flying particles move close to eachother, increasing the cache hit percentage as the simulation goes 

// - TODO: Intel Vtune, performance with particles, and performance without particles
// TODO: cost of moving particles
  // TODO: parallel sorting algorithm with qsort
  // TODO: https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/bitonic_openmp.html

The next few pages are dedicated to analyzing the tricks used to optimize memory usage. These optimizations not only lead to a better scale-up realtive to the sequential code, but also better efficiency when increasing the number of processes / threads.

// === Abstraction the enemy of performance
// === Clean code: horrible performance
// == How particles affect the performance <particles-performance>
== The effects of particles on performance <particles-performance>

From this point on fixed obstacles will be referred to as *fixed particles*, and flying partilces as *moving particles* to be consistent with the code.

=== Take what you need, leave what you don't

In the sequential code, a particle (either moving or fixed) is described by a ```c struct Particle``` s.t.

#figure(caption: `wind.c`)[
```c
typedef struct {
    unsigned char extra;
    int pos_row, pos_col; t1
    int mass; t2
    int resistance; t3
    int speed_row, speed_col; t4
    int old_flow; t5
} particle_t; // Particle was renamed to particle_t for consistency
```
]

This way of storing particles is very costly in terms of memory usage: not all particles need all the attributes, and not all attributes are needed everywhere. In fact:
- the only attributes fixed particles need are ```c pos_row,pos_col``` #reft(1) and ```c resistance``` #reft(3), and these never change
- moving particles need the position #reft(1), the speed #reft(4) and the mass #reft(2) only when moving, the rest of the time, the position #reft(1) is enough
- no particle actually needs the ```c extra``` field and the ```c old_flow``` #reft(5)

To solve this problem the idea is to partially use the _"*struct of arrays* instead of *array of structs*"_ strategy:

#figure(caption: `util.h`)[
```c
typedef struct { int row, col; } vec2_t; t1
typedef struct { vec2_t pos, speed; } particle_m_t; t2
```
]

The ```c particle_m_t struct``` #reft(2) contains the attributes needed by moving particles, and is way smaller than ```c particle_t```. // TODO mass in particle_m_t

#figure(caption: `wind_omp_2.c`)[
```c
particle_m_t *particles_m = ...; t3
vec2_t *particles_pos = ...; t4
int *particles_res = ..., 
    *particles_back = ...,
    *particles_m_mass = ...;
```
]

This way the attributes are separated in different arrays. It's interesting to note that the positions of moving particles are stored twice: 
- in ```c particles_m``` #reft(3)
- in ```c particles_pos``` #reft(4)
That's because the position in ```c particles_m``` is multiplied by a certain ```c PRECISION```, and the actual position inside the matrix is the one in ```c particles_pos```, having two advantages:
- the actual position inside the matrix can be pre-calculated for the fixed particles
- the data that needs to be transferred for moving particles is just a ```c vec2_t```, half the size of a ```c particle_m_t``` and 28% of a ```c particle_t``` ($"64B" / "232B"$).

=== Bringing order in a world of chaos <order-chaos>

The particles are generated randomly across the matrix, this means that the probabilty of cache misses increases drastically if the matrix is bigger. This problem is easy enough to handle: the particles need to be sorted by position (first row then column).

// #image("particles-sort.svg") 
// TODO: maybe describe; `particle_cmp` too
#figure(caption: [`qsort` is a standard C library function works in $O(n log(n))$])[
```c
qsort(particles, num_particles_f, ..., particle_cmp); t1
```
```c
qsort(
    particles + num_particles_f, 
    num_particles_m, ..., 
    particle_cmp
); t2
```
]

The array with the particles is divided in two parts: on the left the fixed particles #reft(1), and on the right the moving particles #reft(2). To improve the performance the best bet is to sort each part individually, first of all due to implementation reasons: 
- for MPI it's easier to use ```c MPI_Gatherv``` to gather the results of ```c move_particle()``` 
- for OpenMP: the workload of ```c move_particle()``` is more evenly distributed and cache-friendly, and it makes possibile to parallelize parts which otherwise would have been sequential 
// - for CUDA the accesses to memory are all adjacent when using ```c move_particles_kernel()```

Secondly, due to the movement patterns of the particles, separating the types of particles greatly improves the cache hit percentage. The moving particles tend to move towards the bottom of the matrix and stay there (see @particles-movement-pattern, the moving particles are marked by `[ ]`). After some iterations, all the accesses to the matrix when working with moving particles are close to eachother, without having to sort again.

#figure(caption: [particles movement pattern from top to bottom])[
#columns()[
#align(center)[

#raw(
"Iteration: 1\n" +
"+------------------------+\n" +
"| 9  9  9  *  *  *  *  * |\n" +
"| 6  7  7  8  8  8  8  8 |\n" +
"|[ ][ ][ ]   [ ]   [ ]   |\n" +
"|   [ ][ ]   [ ][ ]      |\n" +
"|                        |\n" +
"|                        |\n" +
"|                        |\n" +
"|                        |\n" +
"|                        |\n" +
"|                        |\n" +
"+------------------------+\n"
)

#colbreak()

#raw(
"Iteration: 73\n" +
"+------------------------+\n" +
"| *  *  *  *  *  *  *  * |\n" +
"| 9  *  *  *  *  *  *  * |\n" +
"| 8  8  9  9  9  9  9  9 |\n" +
"| 8  8  8  9  9  9  9  9 |\n" +
"| 8  8  8  8  9  9  9  9 |\n" +
"| 8  8  8  8  8  9  9  9 |\n" +
"| 7  8  8  8  8  9  8  8 |\n" +
"| *  9  6  9  4  9  8  8 |\n" +
"|[2] 7 [4] 6 [4] *  7  8 |\n" +
"|[1] 5  5  5 [4][+][4] 7 |\n" +
"+------------------------+\n"
)

]
]
] <particles-movement-pattern>

It's useful to sort only once at the beginnning, there is no benefit in doint it again later:
- the fixed particles don't move, so there's no reason to sort them again
- the moving particles move closer to eachother until they overlap at the bottom, so there's no benefit in sorting them again

 _(I tried different strategies for sorting multiple times during the execution, but there weren't any speed-up improvements. I don't belive a distributed sorting algorithm would benefit either)_

#pagebreak()

To stress the importance of sorting, this is the output of ```bash perf``` on OpenMP in different sorting conditions.

```bash
perf -d wind_omp 500 500 1000 0.5 0 500 1 499 1.0 1 499 1.0 1902 9019 2901
```

```c
// No sorting 
L1-dcache-load-misses       21.88% of all L1-dcache accesses
// Sorting only moving particles
L1-dcache-load-misses       13.86% of all L1-dcache accesses
// Sorting all particles
L1-dcache-load-misses       2.85%  of all L1-dcache accesses
```

// TODO: === Improve ```c memcpy()``` efficiency by doing less of it <motion>

#pagebreak()

== OpenMP

=== Implementation 

Many of the for-loops that work on particles are separated. 
#figure(caption: `wind_omp.c`)[
```c
if (num_particles_m > 0)
    #pragma omp for nowait
    for (int p = 0; p < num_particles_m; p++) 
        update_flow(particles_pos[num_particles_f + p]);

if (num_particles_f > 0)
    #pragma omp for
    for (int p = 0; p < num_particles_f; p++) 
        update_flow(particles_pos[p]);
```
] <separate-particles-for>

#grid(columns: (auto, auto), gutter: 1em,
figure()[
  #raw(
"+---------------------------+\n" +
"|                           |\n" +
"|         [ ]   [F][M]      |\n" +
"|      [ ][ ][ ]   [ ]   [ ]|\n" +
"|         [ ][ ][ ][ ]      |\n" +
"|      [ ][ ]               |\n" +
"|                     [ ]   |\n" +
"|[ ]                        |\n" +
"|         [ ][ ][ ]         |\n" +
"|   [ ]   [ ]      [ ][ ]   |\n" +
"|         [ ][ ][ ]         |\n" +
"|   [ ]   [ ]      [ ][ ]   |\n" +
"|[ ][ ]                  [ ]|\n" +
"+---------------------------+\n" 
)],
[ #v(5pt) This works very well because fixed and moving particles are sorted separately. If the loops were merged, it could happen that two different threads would be assigned particles close to eachother (e.g. fixed particle `[F]` to thread 1 and moving particle `[M]` to thread 3 in @separate-particles-for), leading to *false sharing* when updating the flow. ] // Separating the loops increases the efficiency too. ]
)

There are two parts of the code that must be sequential. The first one is the inlet flow update (@inlet-flow-update) because it uses ```c erand48()``` #reft(1) which is ```bash MT-Unsafe```.

// ```bash
// ┌───────────────────────────────────────┬───────────────┬──────────────┐
// │Interface                              │ Attribute     │ Value        │
// ├───────────────────────────────────────┼───────────────┼──────────────┤
// │drand48 (), erand48 (), lrand48 (),    │ Thread safety │   MT-Unsafe  │
// │nrand48 (), mrand48 (), jrand48 (),    │               │ race:drand48 │
// │srand48 (), seed48 (), lcong48 ()      │               │              │
// └───────────────────────────────────────┴───────────────┴──────────────┘
// ```

#figure(caption: [Inlet flow update])[
```c
for (j = inlet_pos; j < inlet_pos + inlet_size; j++) { 
    double noise = 0.5 - erand48(random_seq); t1
    accessMat(flow, 0, j) = ... (pressure_level + noise);
}
```
] <inlet-flow-update>

The other part is when the resistance of the particles changes the flow.

```c
#pragma omp single
for (int p = 0; p < num_particles_m; p++) {
    int row = ..., col = ..., back = ...;
    accessMat(flow, row, col) -= particles_m_back[p]; t1
    accessMat(flow, row - 1, col) -= particles_m_back[p]; t1
    accessMat(flow, row - 1, col - 1) -= particles_m_back[p]; t1
    accessMat(flow, row - 1, col + 1) -= particles_m_back[p]; t1
}
```

This part is problematic because multiple particles can overlap, so changes to the same position #reft(1) by different particles must be atomic. For this block the compiler uses *vectorized instructions*, which are really fast. So fast, in fact, that non of the methods I've tried can beat vectorized instructions: 
- ```c #pragma omp atomic``` not only is slower because make the instruction atomic, it also provents the compiler from using vectorized instructions in this context
- I tried using ```c omp_lock_t``` in two different ways: 
  - the first attempt was to create a matrix of locks, to lock each required cell individually, but the overhead to use locks was too big
  - the second attempt was to lock entire rows (to reduce the overhead for the locks by working with bigger sections), but this also proved to be inefficient
- ```c #pragma omp reduction()``` didn't work either, because reducing big arrays in OpenMP can break *very easily* the stack limit; even when setting the stack limit to unlimited the performance wasn't good 
- I tried doing something similar to a reduction manually by using ```c #pragma omp threadprivate``` and allocating each thread's section on the heap, but this also required some kind of sequential code at some point

There's nothing that can be done for *moving particles*, and it's not worth to try to parallelize this part, as most of the execution time is spent elsewhere (see @intel-vtune-move-particle).

#figure(caption: [Intel Vtune analysis])[
  #image("./intel-vtune-cpu-time.png", width: 80%)
]  <intel-vtune-move-particle>

Even so, this part can be parallelized _very efficiently_ for *fixed particles*.

```c
#pragma omp parallel
{
  int thread_num = omp_get_thread_num();
  for (int p = f_displs[thread_num]; t1 
      p < f_displs[thread_num] t1 + f_counts[thread_num]; t2 
      p++) {
      int row = ..., col = ..., back = ...;
      accessMat(flow, row, col) -= back;
      accessMat(flow, row - 1, col) += (back / 2);
      accessMat(flow, row - 1, col - 1) += (back / 4);
      accessMat(flow, row - 1, col + 1) += (back / 4);
}
```

The idea is to distribute all the fixed particles among the threads by pre-calculating the displacements #reft(1) and the counts #reft(2) in such a way that all particles that have the same position are assigned to the same thread, so no race conditions can happen. The fixed particles are already sorted, so it's enough to distribute the particles evenly among the threads, and check the particles on the borders.




// The hardest part to parallelize was 
// ```c
// int num_threads;
// #pragma omp parallel
// {
//     #pragma omp single
//     num_threads = omp_get_num_threads();
// }
//
// int *particles_f_counts = calloc(num_threads, sizeof(int)),
//     *particles_f_displs = calloc(num_threads, sizeof(int));
//
// ```

=== Speed-up & efficiency
#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => {
    if y == 0 {
      (bottom: .5pt + black, top: .5pt + black)
    }
    if y == 1 or y == 8 { 
      (bottom: .5pt + black)
    }

    (x: .5pt + black)
  },
)

// tunnel_size 10000 iter 
#figure(caption: [Run-times without any particles with 10000 iterations])[
  #table(
    columns: (auto,auto,auto,auto,auto,auto),
    align: center,
    table.cell(rowspan: 2)[], table.cell(colspan: 5)[Order of Tunnel],
    [64],[128],[256],[512],[1024],
    // gutter: 3pt,
    // table.cell(colspan: 6)[],
    [seq],[0.0354],[0.1238],[0.4659],[1.7831],[10.2625],
    // [cuda],[0.1466],[0.1486],[0.1536],[0.1714],[0.2631],
    [omp_1],[0.0242],[0.0808],[0.2976],[1.1357],[4.3416],
    [omp_2],[0.026],[0.064],[0.1765],[0.6305],[2.2364],
    [omp_4],[0.0389],[0.0604],[0.1398],[0.3748],[1.2126],
    [omp_8],[0.0635],[0.0784],[0.1253],[0.3288],[0.7496],
    [omp_16],[0.1459],[0.1516],[0.1805],[0.2808],[0.5982],
    [omp_32],[0.3169],[0.3057],[0.3279],[0.4117],[0.5811],
  )
]


== CUDA

=== Implementation 

```c
__global__ void move_particles_kernel()
__global__ void update_particles_flow_kernel()
__global__ void update_back_kernel()
__global__ void propagate_waves_kernel()
```

// __global__ void propagate_waves_kernel()
```c
__shared__ int temp[32 + 2];

int w = wave_front + blockIdx.y * 8, col = ...;
if (/* not valid ... */) return;

int s_idx = threadIdx.x + 1;

temp[s_idx] = accessMat(d_flow, wave - 1, col);
if (col > 0) 
    temp[s_idx - 1] = accessMat(d_flow, wave - 1, col - 1);
if (col < columns - 1) 
    temp[s_idx + 1] = accessMat(d_flow, wave - 1, col + 1);

__syncthreads();
```


=== Speed-up & efficiency

== MPI

== MPI + OpenMP

= Fails

== Pthread

As the problem looked very complex at the beginnning and really hard to parallelize, I originally started working with ```c pthread```. While trying different strategies with ```c pthread``` I found out ways to simplify the code, at the point I was able to actually use OpenMP to make significant improvements with way less effort.

== OpenMP

== MPI + CUDA

I've tried the MPI version + CUDA, but the amount of memory needed to be transferred made everything way slower (3x)

// TODO:
// - [X] data vs task parallelism
// - [X] MPI_init
// - [X] MPI_finalize
// - [X] mpirun -n ...
// - [X] MPI_Comm_rank
// - [X] MPI_Comm_size
// - [ ] MPI_Send
//   - [ ] non overtaking (for the same process)
// - [ ] MPI_Recv
// - [ ] MPI tags
// - [ ] Custom MPI communicators?
// - [ ] MPI_ANY_SOURCE
// - [ ] MPI_ANY_TAG
// - [ ] MPI_Status
// - [ ] MPI_Get_cout
// - [ ] multiple mpi hosts
// - [ ] Non blocking
// - [ ] MPI_Wait
// - [ ] MPI_Test
// - [ ] MPI_Reduce
// - [X] MPI_BCast
// - [ ] MPI_Allreduce
// - [ ] https://engineering.fb.com/2021/07/15/open-source/fsdp
// - [ ] NCCL (Nvidia, hard to try, I have only 1 gpu)
// - [ ] MPI_Scatter
// - [X] MPI_Gather
// - [ ] MPI_Barrier
// - [ ] MPI_Allgather
// - [ ] MPI_Reducescatter
// - [ ] MPI_Alltoall
// - [ ] MPI_Wtime (not my problem)
// - [ ] distribution of timings
// - [ ] PMC7 - slide 15
// - [ ] comm_sz vs order of matrix (table)
//   - [ ] $T_"serial" (n)$
//   - [ ] $T_"parallel" (n, p)$
//   - [ ] Speedup $S(n, p) = (T_"serial" (n) ) / (T_"parallel" (n, p))$
//   - [ ] Ideally $S(n, p) = p$ (linear speedup)
//   - [ ] speedup better when increasing problem size (PMC7 - slide 23)
//   - [ ] Scalability $S(n, p) = (T_"parallel" (n, 1)) / (T_"parallel" (n, p))$
//   - [ ] Efficiency $E(n, p) = S(n, p) / p = T_"serial"(n) / (p dot.c T_"parallel" (n, p))$
//   - [ ] Ideally $E(n, p) = 1$ (worst with smaller examples)
//   - [ ] strong vs weak scaling
//     - [ ] strong: fixed size, increase the number of processes
//     - [ ] weak: increase both processes and size at same scale
//   - [ ] Amdahl's law
//     - [ ] $T_"parallel" (p) = (1 - alpha) T_"serial" + alpha T_"serial" / p$
//     - [ ] $0 <= alpha <= 1$ is the fraction that can be parallelized
//     - [ ] Speedup Amdahl $S(p) = T_"serial" / ((1 - alpha) T_"serial" + alpha (T_"serial" / p))$
//     - [ ] $lim_(p -> inf) S(p) = 1 / (1 - alpha)$
//     - [ ] PDF7 - slide 37
//   - [ ] Gustafson's law
//     - [ ] $S(n, p) = (1 - alpha) + alpha p$
// - [ ] MPI_Sendrecv
// - [ ] In place
// - [ ] MPI_Type
// - [ ] MPI_Type_free
//
// Pthread
// - [ ] pthread_create
// - [ ] pthread_join
// - [ ] pthraad_self
// - [ ] pthread_equal
// - [ ] thread_handles (name for array)
// - [ ] -lpthread
// - [ ] pthread_mutex_t
// - [ ] pthread_mutex_init
// - [ ] pthread_mutex_destroy
// - [ ] pthread_mutex_lock
// - [ ] pthread_mutex_unlock
// - [ ] pthread_mutex_trylock
// - [ ] problems
//   - [ ] starvation
//   - [ ] deadlocks
//   - [ ] produce consumer
// - [ ] sem_init
// - [ ] sem_destroy
// - [ ] sem_post
// - [ ] sem_wait
// - [ ] lscup PMC 10 - slide 9
// - [ ] pthread_barrier (to sync)
// - [ ] pthread_cond_t
// - [ ] pthread_cond_init
// - [ ] pthread_cond_wait
// - [ ] pthread_cond_broadcast PMC 10 - slide 58
// - [ ] pthread_cond_destroy
// - [ ] TODO: attributes?
// - [ ] phtread_rwlock_init
// - [ ] pthread_rwlock_destroy
// - [ ] pthread_rwlock_rdlock
// - [ ] pthread_rwlock_wrlock
// - [ ] pthread_rwlock_unlock
//
// OpenMP
// - [ ] ```c #pragma omp parallel```
// - [X] export OMP_NUM_THREADS=4
// - [ ] export OMP_SCHEDULE="dynamic,25"
// - [X] omp_get_num_threads()
// - [X] omp_get_thread_num()
// - [X] ```c #pragma omp parallel num_threads(thread_count)```
// - [ ] number of threads isn't guaranteed
// - [ ] implicit barrier after block is completed
// - [x] names
//   - [x] team of threads (that execute a block in parallel)
//   - [x] master: original thread of execution
//   - [x] parent: thread that started a team of threads
//   - [x] child: each thread started by a parent
// - [ ] ```c #pragma omp end parallel```
// - [ ] ```c #ifndef _OPENMP #else #endif```
// - [ ] ```c #pragma omp critical``` on critical section
//   - [ ] it can have a name ```cpp critical(name)``` : different names can be executed at the same time
// - [x] ```c #pragma omp atomic``` only for 1 instruction, iff parallelizable
// - [ ] scope: set of threads that can access the variable
//   - [ ] shared: accessed by all the threads in the team (default for otuside variables)
//   - [ ] private: accessed by a single thread (default for variables in scope)
// - [x] reduction operator: binary operator
// - [x] reduction: computation that repeatedly applies the same reduction operator to a sequence of operands in order to get a single result
// - [ ] vectorized instructions
// - [x] ```c #pragma omp parallel for reduction(<operator>: <variable list>)```
// - [ ] ```c default(none)``` for scope
// - [ ] ```c shared()```
// - [ ] ```c private(x)``` x is not initialized
// - [ ] ```c firstprivate(x)``` same as private, but value is initialized to outside value
// - [ ] ```c lastprivate(x)``` same as private, but the thread with the *last iteration* sets it value outside
// - [ ] ```c threadprivate(x)``` thread specific persistent storage for global data (the variable must be global or static)
// - [ ] ```c copyin``` used with threadprivate to initialize threadprivate copies from the master thread's variables
// - [ ] ```c single copyprivate(private_var)```
// - [ ] ```c #pragma omp parallel for```
// - [ ] nested:
//   - [ ] invert loops... (not a bad idea)
//   - [ ] collapse in one loop
//   - [ ] ```c #pragma omp parallel for collapse(2)```
//   - [ ] nested parallelism is disabled by default
// - [ ] data dependencies PMC 14 - slide 29:
//   - [ ] loop-carried dependence
//   - [ ] flow dependence (RAW)
//   - [ ] anti-flow dependence (WAR)
//   - [ ] output dependence (WAW)
//   - [ ] input dependence (RAR)
// - [ ] flow dependence removal (TODO)
//   1. [ ] reduction/induction variable fix
//   2. [ ] loop skewing
//   3. [ ] partial parallelization
//   4. [ ] refactoring
//   5. [ ] fissioning  
//   6. [ ] algorithm change
// - [ ] scheduling
//   - [ ] default (0: first 10, 1: second 10 etc...)
//   - [ ] cyclic (0: 1, 11, 21, etc..., 1: 2, 12, 22 etc...)
//     - [ ] ```c schedule(static, 1)```, assigned before the loop is executed
//     - [ ] dynamic or guided (iterations assigned while the loop is executing)
//     - [ ] auto (compiler / runtime system decides)
//     - [ ] runtime (runtime system decides)
// - [x] omp_lock_t write_lock;
// - [x] omp_init_lock
// - [x] omp_set_lock
// - [x] omp_unset_lock
// - [x] omp_destroy_lock
// - [ ] master/single directives:
//   - [ ] only 1 thread executes it
//   - [x] with "single" a barrier is put at the end of the block
// - [ ] ```c #pragma omp barrier```
// - [ ] ```c #pragma omp parallel sections```
//   - [ ] ```c #pragma omp section```
// - [ ] ```c #pragma omp for ordered```
//   - [ ] ```c #pragma omp ordered```
// - [ ] PMC15 - slide 54 MPI + omp/pthread
//
// CUDA
// - [ ] GPU and CPU memories are disjoint
// - [ ] not the same floating point representation
// - [ ] typical program
//   - [ ] allocate GPU memory
//   - [ ] transfer from host to GPU
//   - [ ] run CUDA kernel
//   - [ ] copy results from GPU to host
// - [ ] 6D structure
//   - [ ] each thread has a (x, y, z) in a block
//   - [ ] each block has a (x, y, z) in a grid
//   - [ ] PMC16 - slide 26 (different capability, different sizes possible)
//   - [ ] TODO: get capability of my CPU (6.1)
// - [ ] kernel (function)
//   - [ ] dim3 block(3, 2)
//     - [ ] max 1024 threads per block
//   - [ ] dim3 grid(4, 3, 2)
//   - [ ] always void
//   - [ ] types:
//     - [ ] `__global__`: called from both device and host; from CC 3.5 can be executed on host too
//     - [ ] `__device__`: runs on the GPU and can be called only by a kernel
//     - [ ] `__host__`: used in combination with `__device__` to generate 2 codes
//   - [ ] `threadIdx`: position of thread within block
//   - [ ] `blockIdx`: position of thread's block within grid
//   - [ ] `int myId = stuff`
// - [ ] threads are executed in *warps* (size 32)
//   - [ ] threads in a *block* are split in *warps*
//   - [ ] multiple warps can be executed together, but at different execution paths
//   - [ ] warp divergence
//
// - [ ] CUDA device (8 blocks per SM, 1024 threads per SM, 512 threads per block)
//   - [ ] $8 times 8$ blocks
//     - [ ] 64 threads per block
//     - [ ] 1024 / 64 blocks needed (which is 16, and not all are used)
//   - [ ] $16 times 16$
//     - [ ] 256 threads per block
//     - [ ] 1024 / 256 = 4 which is a good amount of blocks: 4
//   - [ ] $32 times 32$
//     - [ ] 1024 threads per block
//     - [ ] more than the 512 allowed
//
// Dev 0
//   warpSize 32
//   maxGridSize 2147483647
//   maxBlocksPerMultiProcessor 32
//   maxThreadsPerBlock 1024
//   maxThreadsDim 1024
//   maxThreadsPerMultiProcessor 2048
//   sharedMemPerBlock 49152
//   regsPerMultiprocessor 65536
//   regsPerBlock 65536
//
// - [ ] cudaMalloc()
// - [ ] cudaFree()
// - [ ] cudaMemcpy()
//   - [ ] cudaMemcpyHostToHost
//   - [ ] cudaMemcpyHostToDevice
//   - [ ] cudaMemcpyDeviceToHost
//   - [ ] cudaMemcpyDeviceToDevice
//   - [ ] cudaMemcpyDefault (when Unified Virtual Address space capable)
//   - [ ] PMC 16 - slide 64
// - [ ] Memory Types
//   - [ ] registers: hold local variables
//     - [ ] store local variables
//     - [ ] split among resident threads
//     - [ ] maxNumRegisters determined by CC // TODO: determin maxNumRegisters
//     - [ ] if number is exceeded, variables are stored in global
//     - [ ] decided by the compiler
//     - [ ] nvcc --Xptxas
//     - [ ] *occupancy* = $"resident_warps" / "maximum_warps"$
//       - [ ] close to 1 is desirable (to hide latencies)
//       - [ ] analyzed via profiler
//   - [ ] shared memory: fast on-chip, used to hold frequently used data (used to exchange data between cores of the same SM)
//     - [ ] used like L1 "user-managed" cache
//     - [ ] `__shared__` specifier used to indicate data that must go on shared memory
//     - [ ] void `__syncthreads()` used to avoid WAR, RAW, WAW hazards
//   - [ ] L1/L2 cache: transparent to the programmer
//   - [ ] other
//     - [ ] global memory: off-chip, high capacity, relatively slow (only part accessible by the host)
//     - [ ] texture and surface memory: permits fast implementation of filtering/interpolation operator
//     - [ ] constant memory: cached, non modifiable, broadcast constants to all threads in a wrap
//   - [ ] variable declaration:
//     - [ ] automatic variables (no arrays): *registers* (scope thread) - lifetime kernel
//     - [ ] automatic arrays: *global memory* (scope thread) - lifetime kernel
//     - [ ] `__device__` `__shared__` int sharedVar; *shared memory* (scope block) - lifetime kernel
//     - [ ] `__device__` int globalVar; *global memory* (scope grid) - lifetime application
//     - [ ] `__device__` `__constant__` int constVar; *constant memory* (scope grid) - lifetime application
//   - [ ] Roofline model
//     - [ ] https://docs.nersc.gov/tools/performance/roofline/
//     - [ ] https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf
//     - [ ] $beta$ = 0.5GB/s
//     - [ ] PMC20
//     - [ ] PMC18
//   - [ ] matrix multiplication
//   - [ ] atomicAdd
//   - [ ] MPI+GPU
//     - [ ] PMC20j
//
//
// - [ ] cudaDeviceSynchronize(); // barrier
// - [ ] nvcc --arch=sm_20 (cc 2.0) // cuda capability
//
//
// Other
// - message matching
//   - recv type = send type
//   - recv buff size > send buff size
// - P2P
//   - buffered
//   - synchronous
//   - ready
//
// #page(bibliography("bibl.bib"))
//
// // rows=250 
// // columns=250 
// // max_iter=2000 
// // var_threshold=0.5
// // inlet_pos=0 
// // inlet_size=$rows
// // particles_f_band_pos=1 
// // particles_f_band_size=$((rows-1)) 
// // particles_f_density=0.5
// // particles_m_band_pos=1 
// // particles_m_band_size=$((rows-1)) 
// // particles_m_density=0.5
// // short_rnd1=$((RANDOM % 10000 + 1)) short_rnd2=$((RANDOM % 10000 + 1)) short_rnd3=$((RANDOM % 10000 + 1))
//
// // = Introduction
// //
// // You are provided with a sequential code that simulates in a simplified way, how the air pressure spreads in a Wind Tunnel, with fixed obstacles and flying particles. 
// //
// // - the simulation represents a zenithal view of the tunnel, by a 2-dimensional cut of the tunnel along the main air-flow axis
// // - the space is modelled as a set of discrete and evenly spaced points or positions 
// // - a *bidimensional array* is used to represent the value of *air pressure at each point* in a given *time step*
// // - another *unidimensional array* stores the information of a collection of *flying particles* (they can be pushed and moved by the air flow), or *fixed structures* that cannot move
// // - all particles have a *mass* and a *resistance* to the flow of air, partially _blocking_ and _deviating_ it
// //
// //
// // / Fan inlet:
// // - there is an air inlet at the start of the tunnel with a big fan
// // - the values of air pressure introduced by the fan are represented in the row 0 of the array
// // - after a given number of simulation steps the fan *turning-phase* is slightly changed recomputing the values of air pressure introduced, also adding some random noise 
// // - the fan phase is cyclically repeated
// // - every time step, the values of air pressure in the array are updated with the old values in upper rows, propagating the air flow to the lower rows
// // - on each iteration only some rows are updated
// // - they represent wave-fronts, separated by a given number of rows.
// //
// // / Fixed and flying particles:
// // - the information of a particle includes:
// //   - position coordinates
// //   - a two dimensional velocity vector
// //   - mass
// //   - and air flow resistance
// // - all of them are numbers with decimals represented in fixed position 
// // - the particles let some of the air flow to advance depending on their resistance 
// // - the air flow that is blocked by the particle is deviated creating more pressure in the front of the particle
// // - the flying particles are also pushed by the air when the pressure in their front is enough to move their mass
// // - the updates of deviated pressure and flying particles movements are done after a given number of time steps (the same as the distance between propagation wave fronts)
// //
// // / End of simulation:
// // - the simulation ends:
// //   - after a fixed number of iterations (input argument) 
// //   - or if the maximum variation of air pressure in any cell of the array is less than a given threshold
// // - at the end, the program outputs a results line with:
// //   - the number of iterations done 
// //   - the value of the maximum variability of an update in the last iteration
// //   - and the air pressure value of several chosen array positions in three rows: 
// //     - near the inlet 
// //     - in the middle of the tunnel
// //     - in the last row of the tunnel; 
// //     - or in the last row updated by the first wave front in case that the simulation stops before the air flow arrives at the last row
// //
// //
// // Symbols:
// //   - `[ ]` One or more particles/obstacles in the position
// //   - `.` Air pressu in the range $[0, 0.5)$
// //   - `n` A number $n$ represents a pressure between $[n, n + 1)$
// //   - `*` Air pressure equal or greater than 10
// //
// // #align(center, {
// //   set text(font: "CaskaydiaCove NF")
// //   image("wind.svg")
// // })
// //
// // = Scalabilità
// //
// // - rispetto alla dimensione della griglia
// // - rispetto alla dimensione dell'inlet
// // - rispetto al numero di particelle e al tipo
// //
// // Il tutto testato con diversi numeri casuali (usando numeri primi)
// //
// // = Correttezza
// // - valgrind e roba cicli CPU etc...
// // - perf stat
// // - gprof
// //
// // = Architetture
// //
// // - statistiche varie architetture e roba simile 
// //   - getconf LEVEL3_CACHE_ASSOC
// //   - lscpu| grep -E '^Thread|^Core|^Socket|^CPU\('
// //
// // = Alcune note 
// //
// // #note[
// // / Task parallelism:
// // - Partition various tasks among the cores
// // - The “temporal parallelism” we have seen in circuits is a specific type of task parallelism
// //
// // / Data parallelism:
// // - Partition the data used in solving the problem among the cores
// // - Each core carries out similar operations on it’s part of the data
// // - Similar to “spatial parallelism” in circuits
// // ]
// //
// //
// // Which one can i use here? Maybe both?
// // Tasks:
// // - update the inlet
// // - move the particles  
// // - update the flow
// // - propagate
// //
// // To propagate I need all the above... I can propagate multiple times for one of the above (a.k.a 8 times)
// //
// // If I want to parallelize the data, I have to work on distributing the data of the matrices
// // The problem with distributing the data is that some data depends on other data... maybe? Not as much as the tasks.
// //
// // #note[
// // In practice, cores need to coordinate, for different reasons:
// // / Communication: e.g., one core sends its partial sum to another core
// // / Load balancing: share the work evenly so that one is not heavily loaded (what if some file to compress is much bigger than the others?)
// // - If not, p-1 one cores could wait for the slowest (wasted resources & power!)
// // / Synchronization: each core works at its own pace, make sure some core does not get too far ahead
// // - E.g., one core fills the list of files to compress. If the other ones start too early they might miss some files
// // ]
// //
// // Load balancing is secondary here... can be done at the start probably. Communication and Synchronization are the important ones, but how?
// //
// // #note[
// // / Foster’s methodology:
// //
// // / Partitioning: divide the computation to be performed and the data operated on by the computation into small tasks. The focus here should be on identifying tasks that can be executed in parallel.
// //
// // / Communication: determine what communication needs to be carried out among the tasks identified in the previous step
// //
// // / Agglomeration or aggregation: combine tasks and communications identified in the first step into larger tasks. For example, if task A must be executed before task B can be executed, it may make sense to aggregate them into a single composite task. 
// //
// // / Mapping: assign the composite tasks identified in the previous step to processes/threads. This should be done so that communication is minimized, and each process/thread gets roughly the same amount of work.
// // ]
// // PMC7
// // PMC8 datatypes
//

// - performance difference between rows and columns, and why
// - performance when adding moving particles
// - performance when adding fixed particles
// - performance inlet

// #show figure: set block(breakable: true)
// #show heading: set block(above: 1.4em, below: 1em, sticky: false)
// #show sym.emptyset : sym.diameter 
// #show raw.where(block: true): block.with(inset: 1em, stroke: (y: (thickness: .1pt, dash: "dashed")))

// #show outline.entry.where(level: 1): it => {
//   show repeat : none
//   v(1.1em, weak: true)
//   text(size: 1em, strong(it))
// }

// #show raw: r => {
//   show regex("t(\d)"): it => {
//     box(baseline: .2em, circle(stroke: .5pt, inset: .1em, align(center + horizon, 
//           text(size: .8em, strong(repr(it).slice(2, -1)))
//     )))
//   }
//
//   r
// }
//
// #let reft(reft) = box(width: 9pt, height: 9pt, clip: true, radius: 100%, stroke: .5pt, baseline: 1pt,
//   align(center + horizon,
//     text(font: "CaskaydiaCove NF", size: 6pt, strong(str(reft)))
//   )
// )

