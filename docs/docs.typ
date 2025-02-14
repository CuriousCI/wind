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
    #text(size: 1em)[#link("https://github.com/CuriousCI/wind")[https://github.com/CuriousCI/wind]] \ \
    #box(width: 75%)[
      #text(size: 1em)[*Note:* to compile in DEBUG mode, the ```c inline``` keyword must be removed from the functions. The report is lengthy because the project is very big and it took a lot of effort, tries and errors.]
    ]
      
    #align(bottom, datetime.today().display("[day]/[month]/[year]"))
  ]
)

// #page(outline(indent: auto))

#set page(numbering: "1")

= Wind

#quote(attribution: [`handout.pdf`, page 1 Introduction], block: true)[
The software simulates, in a simplified way, how the air pressure spreads in a Wind Tunnel, with *fixed obstacles* and *flying particles*. [...] All particles have a mass and a resistance to the *flow air*, partially *blocking* and *deviating* it.
]
// #grid(columns: (auto, auto), gutter: 1em,
// text(font: "CaskaydiaCove NF",
// "  +------------------------+\n" +
// "  | 8  8  9  *  *  *  *  * |\n" +
// text(fill: green, "->| 6  7  7 [ ] 8 [ ] 8 [ ]|\n") +
// text(fill: blue,  "  |                  [ ][ ]|\n") +
// "  |               [ ][ ]   |\n" +
// "  |         [ ]         [ ]|\n" +
// "  |         [ ]      [ ][ ]|\n" +
// "  |         [ ]   [ ]      |\n" +
// "  |[ ]   [ ]      [ ]   [ ]|\n" +
// "  |[ ]                  [ ]|\n" +
// text(fill: green, "->|   [ ]      [ ][ ]      |\n") +
// text(fill: blue, "  |         [ ][ ]   [ ][ ]|\n") +
// "  |[ ]         [ ]         |\n" +
// "  |[ ]   [ ]   [ ][ ]      |\n" +
// "  |      [ ]               |\n" +
// "  |   [ ][ ][ ]            |\n" +
// "  |               [ ]      |\n" +
// "  |      [ ]               |\n" +
// text(fill: green, "->|[ ][ ]      [ ]   [ ]   |\n") +
// text(fill: blue, "  |   [ ]         [ ]      |\n") +
// "  +------------------------+\n" 
// ), 
// )


The sequential code has two main parallelization targets: *flow updates* and *interactions with partilces*; the latter is a big obstacle to parallelization for a number of reasons:

- *accesses to memory* depend on the positions of the particles 
- the ```c move_particle()``` function is very expensive and requires a *copy of the air flow*
- air flow positions with particles don't need to be always updated 
- *some particles move*, can end up on top of eachother and in different sections of the tunnel, making it very hard to distribute the workload and avoid *race conditions*

  // - pre-calculating the optimal way to access the memory is virtually impossible, but flying particles move close to eachother, increasing the cache hit percentage as the simulation goes 

  // TODO: parallel sorting algorithm with qsort
  // TODO: https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/bitonic_openmp.html

The next few pages are dedicated to analyzing the techniques used in all the implementations. These optimizations not only lead to a better scale-up realtive to the sequential code, but also better efficiency when increasing the number of processes / threads.

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
} particle_t; // Particle -> particle_t to fit my code style 
```
]

This way of storing particles is very costly in terms of memory usage: not all particles need all the attributes, and not all attributes are needed everywhere. In fact:
- the only attributes fixed particles need are the position #reft(1) and the ```c resistance``` #reft(3), and these never change
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
vec2_t *particles_pos = ...;
vec2_t *particles_m_pos = particles_pos + num_particles_f; t4
int *particles_res = ...; 
int *particles_back = ...;
int *particles_m_mass = ...;
```
]

This way the attributes are separated in different arrays. It's interesting to note that the positions of moving particles are stored twice: 
- in ```c particles_m``` #reft(3)
- in ```c particles_m_pos``` #reft(4)
That's because the position in ```c particles_m``` is multiplied by a certain ```c PRECISION```, and the actual position inside the matrix is the one in ```c particles_pos```, having two advantages:
- the actual position inside the matrix can be pre-calculated for the fixed particles
- the data that needs to be transferred for moving particles is just a ```c vec2_t```, half the size of a ```c particle_m_t``` and 28% of a ```c particle_t``` ($"64B" / "232B"$)
Using multiple arrays benefits each implementation in different ways.

=== Bringing order in a world of chaos <order-chaos>

The particles are *generated randomly across the matrix*, this means that the probabilty of *cache misses* increases drastically. This problem is easy enough to handle: the particles need to be sorted by position.

// #image("particles-sort.svg") 
// TODO: maybe describe; `particle_cmp` too
#figure(caption: [`qsort` is a standard C library function works in $O(n log(n))$])[
```c
qsort(particles, num_particles_f, ..., particle_cmp); t1
qsort(
    particles + num_particles_f, 
    num_particles_m, ..., 
    particle_cmp
); t2
```
]

The array with the particles is divided in two parts: on the left the fixed particles #reft(1), and on the right the moving particles #reft(2). To improve the performance the best bet is to sort each part individually, because:
- each type of particle is treated differently, it's more efficient to have moving particles close to each other when processing them
- if the sorting was across all the particles, moving particles would quickly mess up the order 

//, first of all due to implementation reasons: 
// - for MPI it's easier to use ```c MPI_Gatherv``` to gather the results of ```c move_particle()``` 
// - for OpenMP: the workload of ```c move_particle()``` is more evenly distributed and cache-friendly, and it makes possibile to parallelize parts which otherwise would have been sequential 
// - for CUDA the accesses to memory are all adjacent when using ```c move_particles_kernel()```

Due to the movement patterns of moving particles, separating the types of particles greatly improves the cache hit percentage. The moving particles tend to move towards the bottom of the matrix and stay there (see @particles-movement-pattern, the moving particles are marked by #text(fill: blue, `[ ]`)). After some iterations, all the accesses to the matrix when working with moving particles are close to eachother.

#figure(caption: [particles movement pattern from top to bottom])[
#columns()[
#align(center)[

#text(font: "CaskaydiaCove NF", size: 9pt, 
"Iteration: 1\n" +
"+------------------------+\n" +
"| 9  9  9  *  *  *  *  * |\n" +
"| 6  7  7  8  8  8  8  8 |\n" +
"|" + text(fill: blue, "[ ][ ][ ]   [ ]   [ ]   ") + "|\n" +
"|" + text(fill: blue, "   [ ][ ]   [ ][ ]      ") + "|\n" +
"|                        |\n" +
"|                        |\n" +
"|                        |\n" +
"|                        |\n" +
"+------------------------+\n"
)

#colbreak()

#text(font: "CaskaydiaCove NF", size: 9pt, 
"Iteration: 73\n" +
"+------------------------+\n" +
"| *  *  *  *  *  *  *  * |\n" +
"| 9  *  *  *  *  *  *  * |\n" +
"| 8  8  9  9  9  9  9  9 |\n" +
"| 8  8  8  9  9  9  9  9 |\n" +
"| 8  8  8  8  9  9  9  9 |\n" +
"| *  9  6  9  4  9  8  8 |\n" +
"|" + text(fill: blue, "[2] 7 [4] 6 [4] *  7  8 ") + "|\n" +
"|" + text(fill: blue, "[1] 5  5  5 [4][+][4] 7 ") + "|\n" +
"+------------------------+\n"
)

]
]
] <particles-movement-pattern>

It's useful to sort only once at the beginning, there is no benefit in doint it again later:
- fixed particles don't move, so there's no reason to sort them again
- moving particles don't need to be sorted due to the movement patterns in @particles-movement-pattern _(I tried different strategies for sorting multiple times during the execution, but there weren't any speed-up improvements)_

To stress the importance of sorting, this is the output of ```bash perf stat``` on OpenMP in different sorting conditions.

```bash
perf stat -d ./wind_omp 500 500 1000 0.5 0 500 1 499 1.0 1 499 1.0 1902 9019 2901
```

```c
// No sorting 
L1-dcache-load-misses:u     27.29% of all L1-dcache accesses
// Sorting only fixed particles
L1-dcache-load-misses:u     13.19% of all L1-dcache accesses
// Sorting only moving particles
L1-dcache-load-misses:u     17.01% of all L1-dcache accesses
// Sorting all types of particles
L1-dcache-load-misses:u      3.48% of all L1-dcache accesses
```

// TODO: === Improve ```c memcpy()``` efficiency by doing less of it <motion>

=== Computational cost <computational-cost>

Adding particles to the simulation is very expensive: not only the ```c move_particle()``` function takes a lot of time (4.111s, @intel-vtune-seq-particles) and it's very hard to optimize, but the workload of the ```c main()``` function increases from #sym.approx 60ms to 1.360s due to all the processing that must be done on the particles (updating the flow on the positions marked by particles, calculating the pressure of each particle and changing the flow around each particle by adding the pressure of the particle).

The graphs below aren't representative of the most optimal solution (particles aren't sorted, particles' attributes aren't split into multiple arrays etc...), but it's a good enough representation of what happens.  

#figure(caption: [Intel Vtune, "Microarchitecture Exploration" analysis of `wind_seq` with a tunnel full of particles on input\ `512 512 1000 0.5 0 512 1 511 1.0 1 511 1.0 4556 7307 1911` ])[
#image("./intel-vtune-seq-particles.png")
] <intel-vtune-seq-particles>


#figure(caption: [Intel Vtune, "Microarchitecture Exploration" analysis of `wind_seq` without any particles on input \ `512 512 10000 0.5 0 512 1 511 0.0 1 511 0.0 4556 7307 1911` ])[
#image("./intel-vtune-seq-no-particles.png")
] <intel-vtune-seq-no-particles>

The same measurements were made with ```c valgrind --tool=callgrind``` and the results are similar _(for callgrind I compiled with ```bash -ggdb3``` and without ```bash -O3``` to have a more detailed breakdown of each line's cost)_.

For this reason measurements are presented for both a situation without particles and a situtation with a lot of particles (for some implementations the efficiency and scale-up are way better when there aren't any particles).



#pagebreak()

== OpenMP

=== Wave propagation implementation 

```c
int last_wave = iter + 1 < rows ? iter + 1 : rows;
#pragma omp parallel for reduction(max : max_var) 
for (int wave = wave_front; wave < last_wave; wave += STEPS) {
    for (int col = 0; col < columns; col++) {
        /* do stuff... */
    }
} /* End propagation */
```

I want to discuss this section briefly because it has a nested loop. I tried everything: collapsing the loop, using a different types of scheduling (static with different sizes, dynamic with different sizes, guided), nothing improves the efficiency of the basic pragma... this is because *waves* are distant (each wave is 8 rows apart), so collapsing the loops can cause inefficient memory accesses (it's better for a thread to handle the whole row, instead of assigning different parts of it to different threads). This solution is bad when the input has very few rows and a lot of columns.

=== Particles interactions implementation

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

This part is problematic because multiple particles can overlap, so changes to the same position #reft(1) by different particles must be atomic. The compiler uses *vectorized instructions*, which are really fast. So fast, in fact, that none of the other methods I tried is more efficient: 
- ```c #pragma omp atomic``` is slower because make the instruction atomic and provents the compiler from optimizing
- I tried ```c omp_lock_t``` in two different ways: 
  - by creating a matrix of locks, to lock each required cell individually, but the overhead to use locks was too big
  - the second idea was to lock entire rows (to reduce the overhead), but this also proved to be inefficient
- ```c #pragma omp reduction()``` didn't work either, because reducing big arrays in OpenMP can break *very easily* the *stack limit*
- I tried doing something similar to a reduction manually by using ```c #pragma omp threadprivate``` and allocating each thread's section on the heap, but this also proved inefficient 

There's nothing that can be done for *moving particles*.

// and it's not worth to try to parallelize this part, as most of the execution time is spent elsewhere (see @intel-vtune-move-particle).

#figure(caption: [Intel Vtune, "Microarchitecture Exploration analysis" on `wind_omp` with a tunnel full of particles /* on input \ `512 512 1000 0.5 0 512 1 511 1.0 1 511 1.0 4556 7307 1911`*/])[
  #image("./intel-vtune-cpu-time.png")
]  <intel-vtune-move-particle>

Nonetheless, this part can be parallelized _very efficiently without race conditions_ for *fixed particles* (see the code in `wind_omp.c`, lines `656-720` and `857-888`). The idea is to distribute the fixed particles among the threads in such a way that all particles that have the same position are assigned to the same thread. Then particles on the border between two threads must be removed, and handled later.

=== Notes on sorting

Sorting the particles at the beginning is quite expensive, so I decided to parallelize this part too (even if it's done only once).

#figure(caption: [`wind_omp.c / wind_mpi_omp.c / wind_omp_cuda.cu` sorting])[
```c
#pragma omp parallel
{
    #pragma omp single
    {
        if (num_particles_f > 0) 
            #pragma omp task
            qsort(particles, num_particles_f, ...);

        if (num_particles_m > 0) 
            #pragma omp task
            qsort(particles_moving, num_particles_m, ...);
    }
}
```
]

The simplest solution I found was to create 2 tasks, so fixed and moving particles can be sorted at the same time... this solution is quick, but it doesn't scale with more than 2 threads. A better solution, enough time provided, would be to implement a a merge sort, or use other solutions like #underline(link("https://ieeexplore.ieee.org/document/5372752")[this one]) that uses the standard ```c qsort```, or a #underline(link("https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/bitonic_openmp.html")[bitonic sort]).


#[
#set page(margin: (x: 0.8in, y: 1in))

#grid(columns: (auto, auto), align: center, gutter: 1em,
  [#figure(caption: [$x$ axis tunnel size, no particles])[#image("./plot/tunnel_size_4_seq_omp_1_omp_2.png")]],
  [#figure(caption: [$x$ axis particles band size])[#image("./plot/full_system_4_seq_omp_1_omp_2.png")]],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*speed-up*],
      [], table.cell(colspan: 4)[*tunnel size*],
      [program],[128],[256],[512],[1024],
      [omp_1],[2.16],[2.06],[2.0],[3.06],
      [omp_2],[2.13],[2.68],[3.57],[5.39],
      [omp_4],[1.96],[3.06],[4.56],[8.29],
      [omp_8],[1.56],[3.17],[6.11],[11.82],
      [omp_16],[0.82],[2.41],[5.74],[15.72],
      [omp_32],[0.37],[1.18],[4.13],[15.03],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*speed-up*],
      [], table.cell(colspan: 4)[*particles band size*],
      [program], [64], [128],[256],[512],
      [omp_1],[1.35],[1.33],[1.44],[1.43],
      [omp_2],[1.95],[2.11],[2.24],[2.44],
      [omp_4],[2.53],[3.09],[3.65],[4.03],
      [omp_8],[3.44],[3.77],[5.35],[6.05],
      [omp_16],[4.23],[5.27],[7.12],[9.02],
      [omp_32],[2.08],[5.7],[4.31],[4.66],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*efficiency*],
      [], table.cell(colspan: 4)[*tunnel size*],
      [program],[128],[256],[512],[1024],
      [omp_1],[2.16],[2.06],[2.0],[3.06],
      [omp_2],[1.06],[1.34],[1.79],[2.69],
      [omp_4],[0.49],[0.76],[1.14],[2.07],
      [omp_8],[0.2],[0.4],[0.76],[1.48],
      [omp_16],[0.05],[0.15],[0.36],[0.98],
      [omp_32],[0.01],[0.04],[0.13],[0.47],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*efficiency*],
      [], table.cell(colspan: 4)[*particles band size*],
      [program],[64],[128],[256],[512],
      [omp_1],[1.35],[1.33],[1.44],[1.43],
      [omp_2],[0.98],[1.06],[1.12],[1.22],
      [omp_4],[0.63],[0.77],[0.91],[1.01],
      [omp_8],[0.43],[0.47],[0.67],[0.76],
      [omp_16],[0.26],[0.33],[0.44],[0.56],
      [omp_32],[0.07],[0.18],[0.13],[0.15],
    )
  ]
)

\

#block(inset: (x: 1.75in - 0.8in))[
  This data looks strange, because the efficiency in some cases (e.g. `omp_1`, `omp_2`) is greater than 1. This happens on other implementations too, because the sequential code does a lot of useless/inefficient work:
  - it always copies the flow matrix
  - it always resets the particle_locations
  - it always recalculates the position inside the matrix of *all* particles  
  - it doesn't sort the particles
  - it doesn't take advantage of how fixed and moving particles are separated
]
]

#pagebreak()

#[

#set page(margin: 1in)

#block(inset: (x: 1.75in - 1in, top: 1.75in - 1in))[
== CUDA

=== Implementation 

The implementation requires 4 kernels, 3 of which behave like the respective sequential code. 

```c
__global__ void move_particles_kernel()
__global__ void update_particles_flow_kernel()
__global__ void update_back_kernel()
__global__ void propagate_waves_kernel()
```

The interesting one is ```c propagate_waves_kernel()```, which can be optimized by bringing the row above the section of the current wave assigned to the block into *shared memory* (including the halo region).

```c
__shared__ int temp[WAVE_BLOCK_SIZE + 2];
int s_idx = threadIdx.x + 1;
temp[s_idx] = accessMat(d_flow, wave - 1, col);
if (col > 0) 
    temp[s_idx - 1] = accessMat(d_flow, wave - 1, col - 1);
if (col < columns - 1) 
    temp[s_idx + 1] = accessMat(d_flow, wave - 1, col + 1);
__syncthreads();
```

The CUDA version is very fast because all copies from host to device are made at the beginning, only 1 variable needs to be moved from the device to the host each iteration and a single row must be copied from host to device every ```c STEPS``` iterations.

=== Speed-up
]

#grid(columns: (auto, auto), gutter: 1em, align: center,
  [#figure(caption: [$x$ axis tunnel size, no particles])[#image("./plot/tunnel_size_4_seq_cuda.png")]],
  [#figure(caption: [$x$ axis particles band size])[#image("./plot/full_system_4_seq_cuda.png")]],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*speed-up*],
      table.cell(rowspan: 2)[], table.cell(colspan: 4)[*tunnel size*],
      [128],[256],[512],[1024],
      [cuda],[1.02],[3.58],[9.38],[44.75],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*speed-up*],
      table.cell(rowspan: 2)[], table.cell(colspan: 4)[*particles band size*],
      [64],[128],[256],[512],
      [cuda],[43.37],[70.32],[117.37],[167.83],
    )
  ]
)
]

#pagebreak()

== MPI

#grid(columns: (auto, auto), gutter: 1.2em,
[
  The idea around the MPI implementation is to assign each process some consecutive *sections* of the tunnel (a section is a slice of ```c STEPS``` rows, like the one in red/purle), because each iteration only one row in each section is changed. At the end of each ```c STEPS``` iterations, each processe sends to its successor the last row it modified (in blue/purple), so the successor can calculate the first row by using the last row of the predecessor.

  This works very well when there are no particles in the input. When there are particles, every ```c STEPS``` iterations an ```c MPI_Allgatherv()``` is called on the sections of the matrix, so each process can move the *moving partilces* assigned to it. After moving all the particles, process 0 gathers with ```c MPI_Gatherv``` the new positions of the particles (by using a *custom datatype* ```c MPI_VEC2_T```), and does all the processing on the particles (updates the flow around the particles etc...). Here the separation of the position comes handy, because it's faster to transfer less data. 
],
text(font:"CaskaydiaCove NFM", lang: "en", weight: "light", size: 9pt,
"  +------------------------+\n" +
"  | 8  8  9  *  *  *  *  * |\n" +
text(fill: red, "->| 6  7  7 [ ] 8 [ ] 8 [ ]|\n") +
text(fill: red, "  |                  [ ][ ]|\n") +
text(fill: red, "  |               [ ][ ]   |\n") +
text(fill: red, "  |         [ ]         [ ]|\n") +
text(fill: red, "  |         [ ]      [ ][ ]|\n") +
text(fill: red, "  |         [ ]   [ ]      |\n") +
text(fill: red, "  |[ ]   [ ]      [ ]   [ ]|\n") +
text(fill: purple, "  |[-]------------------[-]|\n") +
"->|   [ ]      [ ][ ]      |\n" +
"  |         [ ][ ]   [ ][ ]|\n" +
"  |[ ]         [ ]         |\n" +
"  |[ ]   [ ]   [ ][ ]      |\n" +
"  |      [ ]               |\n" +
"  |   [ ][ ][ ]            |\n" +
"  |               [ ]      |\n" +
text(fill: blue, "  |------[-]---------------|\n") +
"->|[ ][ ]      [ ]   [ ]   |\n" +
"  |   [ ]         [ ]      |\n" +
"  |      [ ]      [ ]      |\n" +
"  |[ ]   [ ]            [ ]|\n" +
"  |   [ ]            [ ][ ]|\n" +
"  |[ ]   [ ]         [ ]   |\n" +
"  |            [ ]   [ ]   |\n" +
text(fill: blue, "  |[-][-][-]---------------|\n") +
"->|[ ]               [ ]   |\n" +
"  |[ ]               [ ][ ]|\n" +
"  |[ ]         [ ][ ]      |\n" +
"  |   [ ]      [ ][ ][ ]   |\n" +
"  |      [ ]   [ ][ ]   [ ]|\n" +
"  |      [ ][ ]         [ ]|\n" +
"  |[ ]         [ ]      [ ]|\n" +
"  |[ ][ ][ ]               |\n" +
"  +------------------------+\n" 
),
)

The problem with this implementation is that process 0 does all the post-processing on the particles, but, as discussed in the previous sections, it would be really hard to distribute the work among the processors due to the amount of checks on inconsistencies. It would also require two more ```c MPI_Allgatherv()``` on the flow to keep the data consistent (and process 0 still would have to process the flow around moving particles by itself).

Another important note: I didn't use ```c MPI_Sendrecv()``` to exchange the rows on the border of sections because the first process doesn't receive any row, and the last process doesn't send any row.

#[
#set page(margin: (x: 0.8in, y: 1in))

#grid(columns: (auto, auto), align: center, gutter: 1em,
  [#figure(caption: [$x$ axis tunnel size, no particles])[#image("./plot/tunnel_size_4_seq_mpi_1_mpi_2.png")]],
  [#figure(caption: [$x$ axis particles band size])[#image("./plot/full_system_4_seq_mpi_1_mpi_2.png")]],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*speed-up*],
      [], table.cell(colspan: 4)[*tunnel size*],
      [program],[128],[256],[512],[1024],
      [mpi_1],[_1.88_],[2.07],[2.37],[3.42],
      [mpi_2],[3.42],[_3.63_],[3.9],[5.86],
      [mpi_4],[2.73],[4.53],[_6.29_],[9.52],
      [mpi_8],[2.04],[4.65],[7.95],[_13.19_],
      [mpi_16],[1.72],[4.81],[9.34],[17.33],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*speed-up*],
      [], table.cell(colspan: 4)[*particles band size*],
      [program],[64],[128],[256],[512],
      [mpi_1],[1.33],[1.36],[1.42],[1.4],
      [mpi_2],[1.98],[2.08],[2.26],[2.28],
      [mpi_4],[2.31],[2.72],[3.0],[3.13],
      [mpi_8],[2.19],[2.76],[3.4],[3.53],
      [mpi_16],[1.94],[2.61],[3.29],[3.78],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*efficiency*],
      [], table.cell(colspan: 4)[*tunnel size*],
      [program],[128],[256],[512],[1024],
      [mpi_1],[1.88],[2.07],[2.37],[3.42],
      [mpi_2],[1.71],[1.82],[1.95],[2.93],
      [mpi_4],[0.68],[1.13],[1.57],[2.38],
      [mpi_8],[0.25],[0.58],[0.99],[1.65],
      [mpi_16],[0.11],[0.3],[0.58],[1.08],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*efficiency*],
      [], table.cell(colspan: 4)[*particles band size*],
      [program],[64],[128],[256],[512],
      [mpi_1],[1.33],[1.36],[1.42],[1.4],
      [mpi_2],[0.99],[1.04],[1.13],[1.14],
      [mpi_4],[0.58],[0.68],[0.75],[0.78],
      [mpi_8],[0.27],[0.34],[0.42],[0.44],
      [mpi_16],[0.12],[0.16],[0.21],[0.24],
    )
  ]
)
\

#block(inset: (x: 1.75in - 0.8in))[
  The same problem with the efficiency that OpenMP had arises here too, for the same reasons. It's interesting to note how, when particles are added, the implementation is way less scalable, because process 0 does a lot of the work alone and it can't be distributed very easily. When there aren't any particles the implementation is *very weakly scalable* and *less strongly scalable*. When there are particles, it scales very badly.
]
]


#[
#set page(margin: (x: 0.8in, y: 1in))

#block(inset: (x: 1.75in - 0.8in, top: 1.75in - 0.8in))[
== MPI + OpenMP

The MPI+OpenMP version doesn't add very much: it has the same division in sections like the MPI version, and moving particles are distributed equally among processes. OpenMP parallelizes the movement of the particles in each process, and parallelizes the interactions with the particles for process 0.
]

#grid(columns: (auto, auto), align: center, gutter: 1em,
  // [#figure(caption: [$x$ axis tunnel size, no particles])[#image("./plot/tunnel_size_4_seq_mpi_omp_1_1_mpi_omp_2_1.png")]],
  // [#figure(caption: [$x$ axis particles band size])[#image("./plot/full_system_4_seq_mpi_omp_1_1_mpi_omp_2_1.png")]],
  [#figure(caption: [$x$ axis tunnel size, no particles])[#image("./plot/tunnel_size_4_seq_mpi_omp_1_1_mpi_omp_4_1.png")]],
  [#figure(caption: [$x$ axis particles band size])[#image("./plot/full_system_4_seq_mpi_omp_1_1_mpi_omp_4_1.png")]],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*speed-up*],
      [], table.cell(colspan: 4)[*tunnel size*],
      [program],[128],[256],[512],[1024],
      // [mpi_omp_2_1],[2.1],[3.64],[3.66],[5.6],
      [mpi_omp_2_2],[1.74],[3.74],[3.91],[8.28],
      [mpi_omp_2_4],[1.54],[3.65],[5.84],[11.46],
      [mpi_omp_2_8],[1.17],[2.82],[8.05],[15.71],
      // [mpi_omp_4_1],[1.71],[3.24],[4.55],[7.92],
      [mpi_omp_4_2],[1.54],[3.8],[6.35],[11.09],
      [mpi_omp_4_4],[1.16],[3.12],[7.02],[15.13],
      [mpi_omp_4_8],[0.72],[1.71],[6.39],[18.76],
    )
  ], 
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*speed-up*],
      [], table.cell(colspan: 4)[*particles band size*],
      [program],[64],[128],[256],[512],
      // [mpi_omp_2_1],[1.93],[2.06],[2.32],[2.46],
      [mpi_omp_2_2],[2.4],[2.99],[3.21],[3.55],
      [mpi_omp_2_4],[2.55],[3.47],[4.7],[4.65],
      [mpi_omp_2_8],[3.27],[3.67],[5.18],[5.88],
      // [mpi_omp_4_1],[2.17],[2.82],[3.2],[3.29],
      [mpi_omp_4_2],[2.21],[3.13],[3.73],[4.85],
      [mpi_omp_4_4],[2.41],[3.34],[4.62],[5.56],
      [mpi_omp_4_8],[1.49],[2.05],[3.07],[4.29],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*efficiency*],
      [], table.cell(colspan: 4)[*tunnel size*],
      [program],[128],[256],[512],[1024],
      // [mpi_omp_2_1],[1.05],[1.82],[1.83],[2.8],
      [mpi_omp_2_2],[0.43],[0.93],[0.98],[2.07],
      [mpi_omp_2_4],[0.19],[0.46],[0.73],[1.43],
      [mpi_omp_2_8],[0.07],[0.18],[0.5],[0.98],
      // [mpi_omp_4_1],[0.43],[0.81],[1.14],[1.98],
      [mpi_omp_4_2],[0.19],[0.48],[0.79],[1.39],
      [mpi_omp_4_4],[0.07],[0.19],[0.44],[0.95],
      [mpi_omp_4_8],[0.02],[0.05],[0.2],[0.59],
    )
  ],
  [
    #table(stroke: 0.1pt, columns: (auto, auto, auto, auto, auto),
      table.cell(colspan: 5)[*efficiency*],
      [], table.cell(colspan: 4)[*particles band size*],
      [program],[64],[128],[256],[512],
      // [mpi_omp_2_1],[0.96],[1.03],[1.16],[1.23],
      [mpi_omp_2_2],[0.6],[0.75],[0.8],[0.89],
      [mpi_omp_2_4],[0.32],[0.43],[0.59],[0.58],
      [mpi_omp_2_8],[0.2],[0.23],[0.32],[0.37],
      // [mpi_omp_4_1],[0.54],[0.7],[0.8],[0.82],
      [mpi_omp_4_2],[0.28],[0.39],[0.47],[0.61],
      [mpi_omp_4_4],[0.15],[0.21],[0.29],[0.35],
      [mpi_omp_4_8],[0.05],[0.06],[0.1],[0.13],
    )
  ]
)

]


== OpenMP + CUDA 

The only difference between the the OpenMP + CUDA implementation and the simple CUDA implementation is the parallelization of the particles' pre-processing before copying the data from the host to device.

#figure(caption: [CUDA vs OpenMP+CUDA, $x$ axis size of particles band])[
  #image("./plot/full_system_4_cuda_omp_cuda_1_omp_cuda_2.png", width: 84%)
] 

It doesn't scale much over 2 threads because the sorting isn't scalable, but I wrote it to to squeeze a bit more out of the CUDA implementation.

#align(center)[
  #table(stroke: 0.1pt, columns: (auto, auto, auto, auto,auto),
    table.cell(colspan: 5)[*speed-up*],
    table.cell(rowspan: 2, align: center + horizon)[], table.cell(colspan: 4)[*particles band size*],
    [64],[128],[256],[512],
    [omp_cuda_1],[43.24],[69.5],[117.21],[166.81],
    [omp_cuda_2],[43.36],[72.87],[126.1],[188.1],
    [omp_cuda_4],[43.54],[71.71],[124.34],[190.76],
  )
]

== Other

In the `arch` folder other _"failed"_ implementations / attempts can be found. I tried implementing a MPI + CUDA code too, but it proved to be very inefficient, due to the amount of data needed to be transferred to the GPU to do the calculations. 
