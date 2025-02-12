/*
 * Simplified simulation of air flow in a wind tunnel
 *
 * MPI version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2020/2021
 *
 * v1.4
 *
 * (c) 2021 Arturo Gonzalez Escribano
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International
 * License. https://creativecommons.org/licenses/by-sa/4.0/
 */
#include "mpi_util.h"
#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Headers for the MPI assignment versions */
#include <mpi.h>
#include <stddef.h>

#define PRECISION 10000
#define STEPS 8

/*
 * Student: Comment these macro definitions lines to eliminate modules
 *	Module2: Activate effects of particles in the air pressure
 *	Module3: Activate moving particles
 */
#define MODULE2
#define MODULE3

#define WAVE_BLOCK_SIZE 512
#define CUDA_ERROR_CHECK(value)                                                                \
    {                                                                                          \
        cudaError_t _m_cudaStat = value;                                                       \
        if (_m_cudaStat != cudaSuccess) {                                                      \
            fprintf(                                                                           \
                stderr,                                                                        \
                "Error %s at line %d in file %s\n",                                            \
                cudaGetErrorString(_m_cudaStat),                                               \
                __LINE__,                                                                      \
                __FILE__                                                                       \
            );                                                                                 \
            exit(1);                                                                           \
        }                                                                                      \
    }

/* Structure to represent a solid particle in the tunnel surface */
typedef struct {
    unsigned char extra;      // Extra field for student's usage
    int pos_row, pos_col;     // Position in the grid
    int mass;                 // Particle mass
    int resistance;           // Resistance to air flow
    int speed_row, speed_col; // Movement direction and speed
    int old_flow;             // To annotate the flow before applying effects
} particle_t;

/*
 * Function: Particle compare
 * 	This function compares two particles by position
 */
static int particle_cmp(const void *p1, const void *p2) {
    particle_t *p_1 = (particle_t *)p1, *p_2 = (particle_t *)p2;

    int pos_1_row = p_1->pos_row / PRECISION;
    int pos_1_col = p_1->pos_col / PRECISION;
    int pos_2_row = p_2->pos_row / PRECISION;
    int pos_2_col = p_2->pos_col / PRECISION;

    if (pos_1_row < pos_2_row) {
        return -1;
    }

    if (pos_1_row > pos_2_row) {
        return 1;
    }

    if (pos_1_col < pos_2_col) {
        return -1;
    }

    if (pos_1_col > pos_2_col) {
        return 1;
    }

    return 0;
}

/*
 * Function to get wall time
 */
double cp_Wtime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be changed and/or optimized by the students
 *
 */
#define accessMat(arr, exp1, exp2) arr[(int)(exp1) * columns + (int)(exp2)]

/*
 * Function: Update flow in a matrix position
 * 	This function can be changed and/or optimized by the students
 */
__global__ void update_particles_flow_kernel(
    int *d_flow,
    int *d_flow_copy,
    int *d_particle_locations,
    vec2_t *d_particles_pos,
    int *d_particles_back,
    int *d_particles_res,
    int columns,
    int num_particles
) {
    int particle = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle >= num_particles) {
        return;
    }

    int row = d_particles_pos[particle].row;
    int col = d_particles_pos[particle].col;

    int temp_flow = accessMat(d_flow_copy, row, col) + accessMat(d_flow_copy, row - 1, col) * 2;

    if (col > 0) {
        temp_flow += accessMat(d_flow_copy, row - 1, col - 1);
    }

    if (col < columns - 1) {
        temp_flow += accessMat(d_flow_copy, row - 1, col + 1);
    }

    int div = 4;
    if (col > 0 && col < columns - 1) {
        div = 5;
    }

    temp_flow /= div;

    accessMat(d_flow, row, col) = temp_flow;
    d_particles_back[particle] = (int)((long)temp_flow * d_particles_res[particle] / PRECISION)
                                 / accessMat(d_particle_locations, row, col);
}

/*
 * Function: Update back flow in a matrix position
 * 	This function applies the pressure given by particles resistance
 */
__global__ void update_back_flow_kernel(
    int *d_flow,
    vec2_t *d_particles_pos,
    int *d_particles_back,
    int *d_particles_res,
    int num_particles,
    int columns
) {
    int particle = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle >= num_particles) {
        return;
    }

    int row = d_particles_pos[particle].row;
    int col = d_particles_pos[particle].col;
    int back = d_particles_back[particle];

    atomicSub(&accessMat(d_flow, row, col), back);
    int temp_flow = back / 2;

    if (col > 0) {
        atomicAdd(&accessMat(d_flow, row - 1, col - 1), back / 4);
    } else {
        temp_flow += back / 4;
    }

    if (col < columns - 1) {
        atomicAdd(&accessMat(d_flow, row - 1, col + 1), back / 4);
    } else {
        temp_flow += back / 4;
    }

    atomicAdd(&accessMat(d_flow, row - 1, col), temp_flow);
}

/*
 * Function: Propagate waves
 * 	This function propagates the flow waves at each iteration
 */
__global__ void propagate_waves_kernel(
    int iter,
    int wave_front,
    int last_wave,
    int *d_flow,
    int *d_particle_locations,
    int *d_max_var,
    int columns
) {
    __shared__ int temp[WAVE_BLOCK_SIZE + 2];

    int wave = wave_front + blockIdx.y * 8;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= columns || wave >= last_wave) {
        return;
    }

    if (accessMat(d_particle_locations, wave, col) != 0) {
        return;
    }

    int shared_id = threadIdx.x + 1;

    temp[shared_id] = accessMat(d_flow, wave - 1, col);

    if (col > 0) {
        temp[shared_id - 1] = accessMat(d_flow, wave - 1, col - 1);
    }

    if (col < columns - 1) {
        temp[shared_id + 1] = accessMat(d_flow, wave - 1, col + 1);
    }

    __syncthreads();

    int temp_flow = accessMat(d_flow, wave, col) + temp[shared_id] * 2;

    if (col > 0) {
        temp_flow += temp[shared_id - 1];
    }

    if (col < columns - 1) {
        temp_flow += temp[shared_id + 1];
    }

    int div = 4;
    if (col > 0 && col < columns - 1) {
        div = 5;
    }

    temp_flow /= div;

    int old = atomicExch(&accessMat(d_flow, wave, col), temp_flow);
    int var = abs(old - temp_flow);
    atomicMax(d_max_var, var);
}

/*
 * Function: Move particle
 * 	This function can be changed and/or optimized by the students
 */
__global__ void move_particles_kernel(
    int *d_flow,
    int *d_particle_locations,
    particle_m_t *d_particles_m,
    vec2_t *d_particles_m_pos,
    int *d_particles_m_mass,
    int num_particles_m,
    int border_rows,
    int border_columns,
    int columns
) {
    int particle = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle >= num_particles_m) {
        return;
    }

    int row = d_particles_m_pos[particle].row;
    int col = d_particles_m_pos[particle].col;

    /* Subtract old position */
    atomicSub(&accessMat(d_particle_locations, row, col), 1);

    /* Compute movement for each step */
    for (int step = 0; step < STEPS; step++) {
        /* Highly simplified phisical model */
        int pressure = accessMat(d_flow, row - 1, col);
        int left = 0;
        int right = 0;

        if (col > 0) {
            left = pressure - accessMat(d_flow, row - 1, col - 1);
        }

        if (col < columns - 1) {
            right = pressure - accessMat(d_flow, row - 1, col + 1);
        }

        int flow_row = (int)((float)pressure / d_particles_m_mass[particle] * PRECISION);
        int flow_col = (int)((float)(right - left) / d_particles_m_mass[particle] * PRECISION);

        /* Speed change */
        d_particles_m[particle].speed.row = (d_particles_m[particle].speed.row + flow_row) / 2;
        d_particles_m[particle].speed.col = (d_particles_m[particle].speed.col + flow_col) / 2;

        /* Movement */
        d_particles_m[particle].pos.row += d_particles_m[particle].speed.row / STEPS / 2;
        d_particles_m[particle].pos.col += d_particles_m[particle].speed.col / STEPS / 2;

        /* Control limits */

        if (d_particles_m[particle].pos.row >= border_rows) {
            d_particles_m[particle].pos.row = border_rows - 1;
        }

        if (d_particles_m[particle].pos.col < 0) {
            d_particles_m[particle].pos.col = 0;
        }

        if (d_particles_m[particle].pos.col >= border_columns) {
            d_particles_m[particle].pos.col = border_columns - 1;
        }

        row = (d_particles_m[particle].pos.row / PRECISION);
        col = (d_particles_m[particle].pos.col / PRECISION);
    }

    d_particles_m_pos[particle].row = row;
    d_particles_m_pos[particle].col = col;

    /* Annotate new position */
    atomicAdd(&accessMat(d_particle_locations, row, col), 1);
}

/*
 * Function: Update flow in a matrix position
 * 	This function can be changed and/or optimized by the students
 */
void update_flow(int *flow, int *flow_copy, int row, int col, int columns) {
    int temp_flow = accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2;

    if (col > 0) {
        temp_flow += accessMat(flow_copy, row - 1, col - 1);
    }

    if (col < columns - 1) {
        temp_flow += accessMat(flow_copy, row - 1, col + 1);
    }

    if (col > 0 && col < columns - 1) {
        temp_flow /= 5;
    } else {
        temp_flow /= 4;
    }

    accessMat(flow, row, col) = temp_flow;
}

/*
 * Function: Move particle
 * 	This function can be changed and/or optimized by the students
 */
void move_particle(
    int *flow,
    particle_m_t *particles_m,
    int particle,
    int border_rows,
    int border_columns,
    int columns,
    vec2_t *particles_m_pos,
    const int *particles_m_mass
) {
    int row = particles_m_pos[particle].row;
    int col = particles_m_pos[particle].col;

    /* Compute movement for each step */
    for (int step = 0; step < STEPS; step++) {
        /* Highly simplified phisical model */
        int pressure = accessMat(flow, row - 1, col);
        int left = 0;
        int right = 0;

        if (col > 0) {
            left = pressure - accessMat(flow, row - 1, col - 1);
        }

        if (col < columns - 1) {
            right = pressure - accessMat(flow, row - 1, col + 1);
        }

        int flow_row = (int)((float)pressure / particles_m_mass[particle] * PRECISION);
        int flow_col = (int)((float)(right - left) / particles_m_mass[particle] * PRECISION);

        /* Speed change */
        particles_m[particle].speed.row = (particles_m[particle].speed.row + flow_row) / 2;
        particles_m[particle].speed.col = (particles_m[particle].speed.col + flow_col) / 2;

        /* Movement */
        particles_m[particle].pos.row += particles_m[particle].speed.row / STEPS / 2;
        particles_m[particle].pos.col += particles_m[particle].speed.col / STEPS / 2;

        /* Control limits */

        if (particles_m[particle].pos.row >= border_rows) {
            particles_m[particle].pos.row = border_rows - 1;
        }

        if (particles_m[particle].pos.col < 0) {
            particles_m[particle].pos.col = 0;
        }

        if (particles_m[particle].pos.col >= border_columns) {
            particles_m[particle].pos.col = border_columns - 1;
        }

        row = (particles_m[particle].pos.row / PRECISION);
        col = (particles_m[particle].pos.col / PRECISION);
    }

    /* Update position realtive to matrix */
    particles_m_pos[particle].row = row;
    particles_m_pos[particle].col = col;
}

#ifdef DEBUG
/*
 * Function: Print the current state of the simulation
 */
void print_status(
    int iteration,
    int rows,
    int columns,
    int *flow,
    int num_particles,
    int *particle_locations,
    int max_var
) {
    /*
     * You don't need to optimize this function, it is only for pretty
     * printing and debugging purposes.
     * It is not compiled in the production versions of the program.
     * Thus, it is never used when measuring times in the leaderboard
     */
    int i, j;
    printf("Iteration: %d, max_var: %f\n", iteration, (float)max_var / PRECISION);

    printf("  +");
    for (j = 0; j < columns; j++) {
        printf("---");
    }
    printf("+\n");
    for (i = 0; i < rows; i++) {
        if (i % STEPS == iteration % STEPS) {
            printf("->|");
        } else {
            printf("  |");
        }

        for (j = 0; j < columns; j++) {
            char symbol;
            if (accessMat(flow, i, j) >= 10 * PRECISION) {
                symbol = '*';
            } else if (accessMat(flow, i, j) >= 1 * PRECISION) {
                symbol = '0' + accessMat(flow, i, j) / PRECISION;
            } else if (accessMat(flow, i, j) >= 0.5 * PRECISION) {
                symbol = '+';
            } else if (accessMat(flow, i, j) > 0) {
                symbol = '.';
            } else {
                symbol = ' ';
            }

            if (accessMat(particle_locations, i, j) > 0) {
                printf("[%c]", symbol);
            } else {
                printf(" %c ", symbol);
            }
        }
        printf("|\n");
    }
    printf("  +");
    for (j = 0; j < columns; j++) {
        printf("---");
    }
    printf("+\n\n");
}
#endif

/*
 * Function: Print usage line in stderr
 */
void show_usage(char *program_name) {
    fprintf(stderr, "Usage: %s ", program_name);
    fprintf(
        stderr,
        "<rows> <columns> <maxIter> <threshold> <inlet_pos> <inlet_size> <fixed_particles_pos> "
        "<fixed_particles_size> <fixed_particles_density> <moving_particles_pos> "
        "<moving_particles_size> <moving_particles_density> <short_rnd1> <short_rnd2> "
        "<short_rnd3> [ <fixed_row> <fixed_col> <fixed_resistance> ... ]\n"
    );
    fprintf(stderr, "\n");
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i, j;

    // Simulation data
    int max_iter;      // Maximum number of simulation steps
    int var_threshold; // Threshold of variability to continue the simulation
    int rows, columns; // Cultivation area sizes

    int *flow = NULL;               // Wind tunnel air-flow
    int *flow_copy = NULL;          // Wind tunnel air-flow (ancillary copy)
    int *particle_locations = NULL; // To quickly locate places with particles

    int inlet_pos;             // First position of the inlet
    int inlet_size;            // Inlet size
    int particles_f_band_pos;  // First position of the band where fixed particles start
    int particles_f_band_size; // Size of the band where fixed particles start
    int particles_m_band_pos;  // First position of the band where moving particles start
    int particles_m_band_size; // Size of the band where moving particles start
    float particles_f_density; // Density of starting fixed particles
    float particles_m_density; // Density of starting moving particles

    unsigned short random_seq[3]; // Status of the random sequence

    int num_particles;     // Number of particles
    particle_t *particles; // List to store cells information

    /* 0. Initialize MPI */
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* 1. Read simulation arguments */
    /* 1.1. Check minimum number of arguments */
    if (argc < 16) {
        fprintf(
            stderr,
            "-- Error: Not enough arguments when reading configuration from the command "
            "line\n\n"
        );
        show_usage(argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* 1.2. Read simulation area sizes, maximum number of iterations and threshold */
    rows = atoi(argv[1]);
    columns = atoi(argv[2]);
    max_iter = atoi(argv[3]);
    var_threshold = (int)(atof(argv[4]) * PRECISION);

    /* 1.3. Read inlet data and band of moving particles data */
    inlet_pos = atoi(argv[5]);
    inlet_size = atoi(argv[6]);
    particles_f_band_pos = atoi(argv[7]);
    particles_f_band_size = atoi(argv[8]);
    particles_f_density = atof(argv[9]);
    particles_m_band_pos = atoi(argv[10]);
    particles_m_band_size = atoi(argv[11]);
    particles_m_density = atof(argv[12]);

    /* 1.4. Read random sequences initializer */
    for (i = 0; i < 3; i++) {
        random_seq[i] = (unsigned short)atoi(argv[13 + i]);
    }

    /* 1.5. Allocate particles */
    num_particles = 0;
    // Check correct number of parameters for fixed particles
    if (argc > 16) {
        if ((argc - 16) % 3 != 0) {
            fprintf(stderr, "-- Error in number of fixed position particles\n\n");
            show_usage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        // Get number of fixed particles
        num_particles = (argc - 16) / 3;
    }
    // Add number of fixed and moving particles in the bands
    int num_particles_f_band = (int)(particles_f_band_size * columns * particles_f_density);
    int num_particles_m_band = (int)(particles_m_band_size * columns * particles_m_density);
    num_particles += num_particles_f_band;
    num_particles += num_particles_m_band;

    // Allocate space for particles
    if (num_particles > 0) {
        particles = (particle_t *)malloc(num_particles * sizeof(particle_t));
        if (particles == NULL) {
            fprintf(
                stderr, "-- Error allocating particles structure for size: %d\n", num_particles
            );
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    } else {
        particles = NULL;
    }

    /* 1.6.1. Read fixed particles */
    int particle = 0;
    if (argc > 16) {
        int fixed_particles = (argc - 16) / 3;
        for (particle = 0; particle < fixed_particles; particle++) {
            particles[particle].pos_row = atoi(argv[16 + particle * 3]) * PRECISION;
            particles[particle].pos_col = atoi(argv[17 + particle * 3]) * PRECISION;
            particles[particle].mass = 0;
            particles[particle].resistance = (int)(atof(argv[18 + particle * 3]) * PRECISION);
            particles[particle].speed_row = 0;
            particles[particle].speed_col = 0;
        }
    }
    /* 1.6.2. Generate fixed particles in the band */
    for (; particle < num_particles - num_particles_m_band; particle++) {
        particles[particle].pos_row
            = (int)(PRECISION
                    * (particles_f_band_pos + particles_f_band_size * erand48(random_seq)));
        particles[particle].pos_col = (int)(PRECISION * columns * erand48(random_seq));
        particles[particle].mass = 0;
        particles[particle].resistance = (int)(PRECISION * erand48(random_seq));
        particles[particle].speed_row = 0;
        particles[particle].speed_col = 0;
    }

    /* 1.7. Generate moving particles in the band */
    for (; particle < num_particles; particle++) {
        particles[particle].pos_row
            = (int)(PRECISION
                    * (particles_m_band_pos + particles_m_band_size * erand48(random_seq)));
        particles[particle].pos_col = (int)(PRECISION * columns * erand48(random_seq));
        particles[particle].mass = (int)(PRECISION * (1 + 5 * erand48(random_seq)));
        particles[particle].resistance = (int)(PRECISION * erand48(random_seq));
        particles[particle].speed_row = 0;
        particles[particle].speed_col = 0;
    }

#ifdef DEBUG
    // 1.8. Print arguments
    if (rank == 0) {
        printf(
            "Arguments, Rows: %d, Columns: %d, max_iter: %d, threshold: %f\n",
            rows,
            columns,
            max_iter,
            (float)var_threshold / PRECISION
        );
        printf(
            "Arguments, Inlet: %d, %d  Band of fixed particles: %d, %d, %f  Band of moving "
            "particles: %d, %d, %f\n",
            inlet_pos,
            inlet_size,
            particles_f_band_pos,
            particles_f_band_size,
            particles_f_density,
            particles_m_band_pos,
            particles_m_band_size,
            particles_m_density
        );
        printf(
            "Arguments, Init Random Sequence: %hu,%hu,%hu\n",
            random_seq[0],
            random_seq[1],
            random_seq[2]
        );
        printf("Particles: %d\n", num_particles);
        for (int particle = 0; particle < num_particles; particle++) {
            printf(
                "Particle[%d] = { %d, %d, %d, %d, %d, %d }\n",
                particle,
                particles[particle].pos_row,
                particles[particle].pos_col,
                particles[particle].mass,
                particles[particle].resistance,
                particles[particle].speed_row,
                particles[particle].speed_col
            );
        }
        printf("\n");
        fflush(stdout);
    }
#endif // DEBUG

    /* 2. Start global timer */
    MPI_Barrier(MPI_COMM_WORLD);
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     *
     */

    const int border_rows = PRECISION * rows;
    const int border_columns = PRECISION * columns;
    const int num_particles_f = num_particles - num_particles_m_band;
    const int num_particles_m = num_particles_m_band;

    particle_t *particles_moving = particles + num_particles_f;

    qsort(particles, num_particles_f, sizeof(particle_t), particle_cmp);
    qsort(particles_moving, num_particles_m, sizeof(particle_t), particle_cmp);

    // Declare variables used in later output
    int max_var = INT_MAX;
    int iter = 0;
    int resultsA[6];
    int resultsB[6];
    int resultsC[6];

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    /* 3. Initialization */
    flow = (int *)calloc((size_t)rows * (size_t)columns, sizeof(int));
    flow_copy = (int *)calloc((size_t)rows * (size_t)columns, sizeof(int));
    particle_locations = (int *)calloc((size_t)rows * (size_t)columns, sizeof(int));

    if (flow == NULL || flow_copy == NULL || particle_locations == NULL) {
        fprintf(
            stderr, "-- Error allocating culture structures for size: %d x %d \n", rows, columns
        );
        exit(EXIT_FAILURE);
    }

    MPI_Datatype MPI_VEC2_T;
    MPI_Vec2_t(&MPI_VEC2_T);

    int *particles_m_counts = (int *)calloc(comm_size, sizeof(int));
    int *particles_m_displs = (int *)calloc(comm_size, sizeof(int));
    // int *particles_m_finals = (int *)malloc(comm_size * sizeof(int));

    /* Distribute moving particles among processes */
    distribute(num_particles_m, comm_size, particles_m_counts, particles_m_displs);
    // for (int rank = 0; rank < comm_size; rank++) {
    //     particles_m_finals[rank] = particles_m_displs[rank] + particles_m_counts[rank];
    // }

    int *sect_counts = (int *)calloc(comm_size, sizeof(int));
    int *sect_displs = (int *)calloc(comm_size, sizeof(int));

    /* Divide the matrix into blocks, each with STEPS rows and 'columns' columns
     * Distribute the blocks among the processes, ignoring the row with the inlet. */
    distribute(
        (rows - 1) / STEPS + (((rows - 1) % STEPS) != 0 ? 1 : 0),
        comm_size,
        sect_counts,
        sect_displs
    );

    int *flow_counts = (int *)calloc(comm_size, sizeof(int)),
        *flow_displs = (int *)calloc(comm_size, sizeof(int));

    {
        int units = columns;
        for (int rank = 0; rank < comm_size; rank++) {
            flow_displs[rank] = sect_displs[rank] * STEPS * columns;
            int count = sect_counts[rank] * STEPS * columns;

            if (count == 0) {
                flow_counts[rank] = 0;
            } else if (units + count < rows * columns) {
                flow_counts[rank] = count;
            } else {
                flow_counts[rank] = (rows * columns) - units;
            }

            units += flow_counts[rank];
        }
    }

    // int *flow_ranks = malloc(rows * columns * sizeof(int));
    // for (int rank = 0; rank < comm_size; rank++) {
    //     for (int unit = flow_displs[rank]; unit < flow_displs[rank] + flow_counts[rank];
    //          unit++) {
    //         flow_ranks[unit] = rank;
    //     }
    // }

    // if (rank == 0) {
    //     fflush(stderr);
    //     fprintf(
    //         stderr,
    //         "rows: %d - divided: %d + %d\n",
    //         rows,
    //         (rows - 1) / STEPS,
    //         (rows - 1) % STEPS
    //     );
    //     for (int rank = 0; rank < comm_size; rank++) {
    //         fprintf(
    //             stderr,
    //             "rank: %d - displ: %d - count: %d\n",
    //             rank,
    //             sect_displs[rank],
    //             sect_counts[rank]
    //         );
    //         fflush(stderr);
    //     }
    //
    //     for (int rank = 0; rank < comm_size; rank++) {
    //         fprintf(
    //             stderr,
    //             "rank: %d - displ: %d - count: %d\n",
    //             rank,
    //             flow_displs[rank],
    //             flow_counts[rank]
    //         );
    //         fflush(stderr);
    //     }
    // }

    // MPI_STATUS_IGNORE

    // if (rank == 0) {
    //     for (int rank = 0; rank < comm_size; rank++) {
    //         fprintf(
    //             stderr,
    //             "rank: %d - displ: %d - count: %d\n",
    //             rank,
    //             blocks_displs[rank],
    //             blocks_counts[rank]
    //         );
    //     }
    //     for (int rank = 0; rank < comm_size; rank++) {
    //         fprintf(
    //             stderr,
    //             "rank: %d - displ: %d - count: %d\n",
    //             rank,
    //             flow_displs[rank],
    //             flow_counts[rank]
    //         );
    //     }
    // }

    vec2_t *particles_pos = (vec2_t *)malloc(num_particles * sizeof(vec2_t));
    int *particles_res = (int *)malloc(num_particles * sizeof(int)),
        *particles_back = (int *)malloc(num_particles * sizeof(int));

    particle_m_t *particles_m = (particle_m_t *)malloc(num_particles_m * sizeof(particle_m_t));
    int *particles_m_mass = (int *)malloc(num_particles_m * sizeof(float));
    vec2_t *particles_m_pos = particles_pos + num_particles_f;
    int *particles_m_res = particles_res + num_particles_f,
        *particles_m_back = particles_back + num_particles_f;

    for (int particle = 0; particle < num_particles_m; particle++) {
        particles_m[particle] = (particle_m_t){
            .pos = {
                .row = particles_moving[particle].pos_row,
                .col = particles_moving[particle].pos_col,
            },
            .speed = {
                .row = particles_moving[particle].speed_row,
                .col = particles_moving[particle].speed_col,
            },
        };

        particles_m_mass[particle] = particles_moving[particle].mass;
    }

    for (int particle = 0; particle < num_particles; particle++) {
        particles_pos[particle].row = particles[particle].pos_row / PRECISION;
        particles_pos[particle].col = particles[particle].pos_col / PRECISION;
        particles_res[particle] = particles[particle].resistance;
    }

    if (rank == 0) {
        for (int particle = 0; particle < num_particles; particle++) {
            accessMat(
                particle_locations, particles_pos[particle].row, particles_pos[particle].col
            )++;
        }
    }

    int *d_flow;
    int *d_flow_copy;
    int *d_particle_locations;

    particle_m_t *d_particles_m;
    vec2_t *d_particles_pos, *d_particles_m_pos;
    int *d_particles_res;
    int *d_particles_m_mass;
    int *d_particles_back;

    CUDA_ERROR_CHECK(cudaMalloc(&d_flow, rows * columns * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_flow_copy, rows * columns * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_particle_locations, rows * columns * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMemset(d_flow, 0, rows * columns * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMemset(d_flow_copy, 0, rows * columns * sizeof(int)));

    CUDA_ERROR_CHECK(cudaMalloc(&d_particles_m, num_particles_m * sizeof(particle_m_t)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_particles_pos, num_particles * sizeof(vec2_t)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_particles_res, num_particles * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_particles_m_mass, num_particles_m * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_particles_back, num_particles * sizeof(int)));

    d_particles_m_pos = d_particles_pos + num_particles_f;

    CUDA_ERROR_CHECK(cudaMemcpy(
        d_particle_locations,
        particle_locations,
        rows * columns * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    CUDA_ERROR_CHECK(cudaMemcpy(
        d_particles_m,
        particles_m,
        num_particles_m * sizeof(particle_m_t),
        cudaMemcpyHostToDevice
    ));

    CUDA_ERROR_CHECK(cudaMemcpy(
        d_particles_pos, particles_pos, num_particles * sizeof(vec2_t), cudaMemcpyHostToDevice
    ));

    CUDA_ERROR_CHECK(cudaMemcpy(
        d_particles_m_mass,
        particles_m_mass,
        num_particles_m * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    CUDA_ERROR_CHECK(cudaMemcpy(
        d_particles_res, particles_res, num_particles * sizeof(int), cudaMemcpyHostToDevice
    ));

    dim3 waves_block_dim(WAVE_BLOCK_SIZE, 1);
    dim3 waves_gridx_dim((columns / WAVE_BLOCK_SIZE) + 1, (rows / 8) + 1);

    int *d_max_var;
    CUDA_ERROR_CHECK(cudaMalloc(&d_max_var, sizeof(int)));

    /* 4. Simulation */
    for (iter = 1; iter <= max_iter && max_var > var_threshold; iter++) {

        if (iter % STEPS == 1) {

            if (rank == 0) {
                /* 4.1. Change inlet values each STEP iterations */
                for (j = inlet_pos; j < inlet_pos + inlet_size; j++) {
                    /* 4.1.1. Change the fans phase */
                    double phase = iter / STEPS * (M_PI / 4); // NOLINT
                    double phase_step = M_PI / 2 / inlet_size;
                    double pressure_level = 9 + 2 * sin(phase + (j - inlet_pos) * phase_step);

                    /* 4.1.2. Add some random noise */
                    double noise = 0.5 - erand48(random_seq);

                    /* 4.1.3. Store level in the first row of the ancillary structure */
                    accessMat(flow, 0, j) = (int)(PRECISION * (pressure_level + noise));
                }
            }

            if (num_particles_m > 0) {
                MPI_Bcast(flow, columns, MPI_INT, 0, MPI_COMM_WORLD);
                // CUDA_ERROR_CHECK(
                //     cudaMemcpy(d_flow, flow, columns * sizeof(int), cudaMemcpyHostToDevice)
                // );
            }

#ifdef MODULE2
#ifdef MODULE3

            /* 4.2. Particles movement each STEPS iterations */

            if (rank == 0) {
                if (num_particles_m > 0) {
                    /* Subtract old positions */
                    for (int particle = 0; particle < num_particles_m; particle++) {
                        accessMat(
                            particle_locations,
                            particles_m_pos[particle].row,
                            particles_m_pos[particle].col
                        )--;
                    }
                }
            }

            if (num_particles_m > 0) {
                MPI_Allgatherv(
                    rank == 0 ? MPI_IN_PLACE : flow + columns + flow_displs[rank],
                    flow_counts[rank],
                    MPI_INT,
                    flow + columns,
                    flow_counts,
                    flow_displs,
                    MPI_INT,
                    MPI_COMM_WORLD
                );
            } else if (num_particles_f > 0) {
                MPI_Gatherv(
                    rank == 0 ? MPI_IN_PLACE : flow + columns + flow_displs[rank],
                    flow_counts[rank],
                    MPI_INT,
                    flow + columns,
                    flow_counts,
                    flow_displs,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD
                );
            }

            if (num_particles_m > 0) {
                for (int particle = particles_m_displs[rank];
                     particle < particles_m_displs[rank] + particles_m_counts[rank];
                     particle++) {
                    move_particle(
                        flow,
                        particles_m,
                        particle,
                        border_rows,
                        border_columns,
                        columns,
                        particles_m_pos,
                        particles_m_mass
                    );
                }
            }

            if (num_particles_m > 0) {
                MPI_Gatherv(
                    rank == 0 ? MPI_IN_PLACE : particles_m_pos + particles_m_displs[rank],
                    particles_m_counts[rank],
                    MPI_VEC2_T,
                    particles_m_pos,
                    particles_m_counts,
                    particles_m_displs,
                    MPI_VEC2_T,
                    0,
                    MPI_COMM_WORLD
                );
            }

            if (rank == 0) {
                if (num_particles_m > 0) {
                    for (int particle = 0; particle < num_particles_m; particle++) {
                        accessMat(
                            particle_locations,
                            particles_m_pos[particle].row,
                            particles_m_pos[particle].col
                        )++;
                    }
                }
            }

            if (num_particles_m > 0) {
                MPI_Scatterv(
                    particle_locations + columns,
                    flow_counts,
                    flow_displs,
                    MPI_INT,
                    particle_locations + columns + flow_displs[rank],
                    flow_counts[rank],
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD
                );
            }
#endif // MODULE3

            if (num_particles > 0) {
                if (rank == 0) {
                    cudaMemcpy(
                        d_flow, flow, rows * columns * sizeof(int), cudaMemcpyHostToDevice
                    );

                    cudaMemcpy(
                        d_particle_locations,
                        particle_locations,
                        rows * columns * sizeof(int),
                        cudaMemcpyHostToDevice
                    );

                    cudaMemcpy(
                        d_particles_pos,
                        particles_pos,
                        num_particles * sizeof(vec2_t),
                        cudaMemcpyHostToDevice
                    );

                    update_particles_flow_kernel<<<(num_particles / 64) + 1, 64>>>(
                        d_flow,
                        d_flow_copy,
                        d_particle_locations,
                        d_particles_pos,
                        d_particles_back,
                        d_particles_res,
                        columns,
                        num_particles
                    );

                    update_back_flow_kernel<<<(num_particles / 64) + 1, 64>>>(
                        d_flow,
                        d_particles_pos,
                        d_particles_back,
                        d_particles_res,
                        num_particles,
                        columns
                    );

                    cudaMemcpy(flow, d_flow, rows * columns, cudaMemcpyDeviceToHost);

                    // for (int particle = 0; particle < num_particles; particle++) {
                    //     int row = particles_pos[particle].row;
                    //     int col = particles_pos[particle].col;
                    //
                    //     update_flow(flow, flow_copy, row, col, columns);
                    //
                    //     particles_back[particle] = (int)((long)accessMat(flow, row, col)
                    //                                      * particles_res[particle] /
                    //                                      PRECISION)
                    //                                / accessMat(particle_locations, row, col);
                    // }
                    //
                    // for (int particle = 0; particle < num_particles; particle++) {
                    //     int row = particles_pos[particle].row;
                    //     int col = particles_pos[particle].col;
                    //
                    //     int back = particles_back[particle];
                    //
                    //     accessMat(flow, row, col) -= back;
                    //     accessMat(flow, row - 1, col) += back / 2;
                    //
                    //     if (col > 0) {
                    //         accessMat(flow, row - 1, col - 1) += back / 4;
                    //     } else {
                    //         accessMat(flow, row - 1, col) += back / 4;
                    //     }
                    //
                    //     if (col < columns - 1) {
                    //         accessMat(flow, row - 1, col + 1) += back / 4;
                    //     } else {
                    //         accessMat(flow, row - 1, col) += back / 4;
                    //     }
                    // }
                }

                MPI_Scatterv(
                    flow + columns,
                    flow_counts,
                    flow_displs,
                    MPI_INT,
                    flow + columns + flow_displs[rank],
                    flow_counts[rank],
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD
                );

                if (rank < comm_size - 1 && flow_counts[rank + 1] > 0) {
                    MPI_Send(
                        flow + flow_displs[rank] + flow_counts[rank],
                        columns,
                        MPI_INT,
                        rank + 1,
                        0,
                        MPI_COMM_WORLD
                    );
                }

                if (rank > 0 && flow_counts[rank] > 0) {
                    MPI_Recv(
                        flow + flow_displs[rank - 1] + flow_counts[rank - 1],
                        columns,
                        MPI_INT,
                        rank - 1,
                        0,
                        MPI_COMM_WORLD,
                        NULL
                    );
                }
            }

            max_var = 0;
#endif // MODULE2
        }

        int wave_front = iter % STEPS;
        if (wave_front == 0) {
            wave_front = STEPS;
            if (num_particles > 0) {
                MPI_Gatherv(
                    flow + columns + flow_displs[rank],
                    flow_counts[rank],
                    MPI_INT,
                    flow_copy + columns,
                    flow_counts,
                    flow_displs,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD
                );

                if (rank == 0) {
                    memcpy(flow_copy, flow, columns * sizeof(int));
                    cudaMemcpy(
                        d_flow_copy,
                        flow_copy,
                        rows * columns * sizeof(int),
                        cudaMemcpyHostToDevice
                    );
                }
            }
        }

        int last_wave = iter + 1 < rows ? iter + 1 : rows;

        for (int sect = sect_displs[rank]; sect < sect_displs[rank] + sect_counts[rank];
             sect++) {
            int wave = sect * STEPS + wave_front;
            if (wave < last_wave) {
                for (int col = 0; col < columns; col++) {
                    if (num_particles == 0 || accessMat(particle_locations, wave, col) == 0) {
                        int prev = accessMat(flow, wave, col);
                        update_flow(flow, flow, wave, col, columns);
                        int var = abs(prev - accessMat(flow, wave, col));
                        if (var > max_var) {
                            max_var = var;
                        }
                    }
                }
            }
        }

        if (wave_front == STEPS) {
            if (rank < comm_size - 1 && flow_counts[rank + 1] > 0) {
                MPI_Send(
                    flow + flow_displs[rank] + flow_counts[rank],
                    columns,
                    MPI_INT,
                    rank + 1,
                    0,
                    MPI_COMM_WORLD
                );
            }
            if (rank > 0 && flow_counts[rank] > 0) {
                MPI_Recv(
                    flow + flow_displs[rank - 1] + flow_counts[rank - 1],
                    columns,
                    MPI_INT,
                    rank - 1,
                    0,
                    MPI_COMM_WORLD,
                    NULL
                );
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &max_var, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

#ifdef DEBUG
        MPI_Gatherv(
            flow + columns + flow_displs[rank],
            flow_counts[rank],
            MPI_INT,
            flow + columns,
            flow_counts,
            flow_displs,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        // 4.7. DEBUG: Print the current state of the simulation at the end of each iteration
        if (rank == 0) {
            fflush(stdout);
            print_status(iter, rows, columns, flow, num_particles, particle_locations, max_var);
            fflush(stdout);
        }
#endif

    } // End iterations

    MPI_Gatherv(
        flow + columns + flow_displs[rank],
        flow_counts[rank],
        MPI_INT,
        flow + columns,
        flow_counts,
        flow_displs,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // MPI: Fill result arrays used for later output
    if (rank == 0) {
        int ind;
        for (ind = 0; ind < 6; ind++) {
            resultsA[ind] = accessMat(flow, STEPS - 1, ind * columns / 6);
        }

        int res_row = (iter - 1 < rows - 1) ? iter - 1 : rows - 1;
        for (ind = 0; ind < 6; ind++) {
            resultsB[ind] = accessMat(flow, res_row / 2, ind * columns / 6);
        }

        for (ind = 0; ind < 6; ind++) {
            resultsC[ind] = accessMat(flow, res_row, ind * columns / 6);
        }
    }

    /*
     *
     * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
     *
     */

    /* 5. Stop global timer */
    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    ttotal = cp_Wtime() - ttotal;

    /* 6. Output for leaderboard */
    if (rank == 0) {
        printf("\n");
        /* 6.1. Total computation time */
        printf("Time: %lf\n", ttotal);

        /* 6.2. Results: Statistics */
        printf("Result: %d, %d", iter - 1, max_var);
        int i;
        for (i = 0; i < 6; i++) {
            printf(", %d", resultsA[i]);
        }
        for (i = 0; i < 6; i++) {
            printf(", %d", resultsB[i]);
        }
        for (i = 0; i < 6; i++) {
            printf(", %d", resultsC[i]);
        }
        printf("\n");
    }

    /* 7. Free resources */
    free(flow);
    free(flow_copy);
    free(particle_locations);
    free(particles);

    /* 8. End */
    MPI_Finalize();
    return 0;
}

// int update_flow(
//     int *flow,
//     int *flow_copy,
//     int *particle_locations,
//     int row,
//     int col,
//     int columns,
//     int skip_particles
// ) {
//     // Skip update in particle positions
//     if (skip_particles && accessMat(particle_locations, row, col) != 0)
//         return 0;
//
//     // Update if border left
//     if (col == 0) {
//         accessMat(flow, row, col)
//             = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
//                + accessMat(flow_copy, row - 1, col + 1))
//               / 4;
//     }
//     // Update if border right
//     if (col == columns - 1) {
//         accessMat(flow, row, col)
//             = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
//                + accessMat(flow_copy, row - 1, col - 1))
//               / 4;
//     }
//     // Update in central part
//     if (col > 0 && col < columns - 1) {
//         accessMat(flow, row, col)
//             = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
//                + accessMat(flow_copy, row - 1, col - 1)
//                + accessMat(flow_copy, row - 1, col + 1))
//               / 5;
//     }
//
//     // Return flow variation at this position
//     return abs(accessMat(flow_copy, row, col) - accessMat(flow, row, col));
// }
// MPI_Barrier(MPI_COMM_WORLD);

// void move_particle(int *flow, particle_t *particles, int particle, int rows, int columns) {
//     // Compute movement for each step
//     int step;
//     for (step = 0; step < STEPS; step++) {
//         // Highly simplified phisical model
//         int row = particles[particle].pos_row / PRECISION;
//         int col = particles[particle].pos_col / PRECISION;
//         int pressure = accessMat(flow, row - 1, col);
//         int left, right;
//         if (col == 0)
//             left = 0;
//         else
//             left = pressure - accessMat(flow, row - 1, col - 1);
//         if (col == columns - 1)
//             right = 0;
//         else
//             right = pressure - accessMat(flow, row - 1, col + 1);
//
//         int flow_row = (int)((float)pressure / particles[particle].mass * PRECISION);
//         int flow_col = (int)((float)(right - left) / particles[particle].mass * PRECISION);
//
//         // Speed change
//         particles[particle].speed_row = (particles[particle].speed_row + flow_row) / 2;
//         particles[particle].speed_col = (particles[particle].speed_col + flow_col) / 2;
//
//         // Movement
//         particles[particle].pos_row
//             = particles[particle].pos_row + particles[particle].speed_row / STEPS / 2;
//         particles[particle].pos_col
//             = particles[particle].pos_col + particles[particle].speed_col / STEPS / 2;
//
//         // Control limits
//         if (particles[particle].pos_row >= PRECISION * rows)
//             particles[particle].pos_row = PRECISION * rows - 1;
//         if (particles[particle].pos_col < 0)
//             particles[particle].pos_col = 0;
//         if (particles[particle].pos_col >= PRECISION * columns)
//             particles[particle].pos_col = PRECISION * columns - 1;
//     }
// }

// if (rank == 0)
//     if (num_particles_f > 0) {
//         for (int particle = 0; particle < num_particles_f; particle++) {
//             int row = particles_pos[particle].row;
//             int col = particles_pos[particle].col;
//
//             accessMat(flow_copy, row, col) = accessMat(flow, row, col);
//             accessMat(flow_copy, row - 1, col) = accessMat(flow, row - 1, col);
//             if (col > 0)
//                 accessMat(flow_copy, row - 1, col - 1)
//                     = accessMat(flow, row - 1, col - 1);
//             if (col < columns - 1)
//                 accessMat(flow_copy, row - 1, col + 1)
//                     = accessMat(flow, row - 1, col + 1);
//         }
//     }
//
// if (rank == 0)
//     if (num_particles_m > 0) {
//         int min_row = INT_MAX;
//
//         for (int particle = 0; particle < num_particles_m; particle++)
//             if (particles_m_pos[particle].row < min_row)
//                 min_row = particles_m_pos[particle].row;
//
//         if (min_row > 1)
//             min_row--;
//         memcpy(
//             flow_copy + min_row * columns,
//             flow + min_row * columns,
//             (rows - min_row) * columns * sizeof(int)
//         );
//     }

// if (rank == 0)
//     for (int wave = wave_front; wave < last_wave; wave += STEPS) {
//         for (int col = 0; col < columns; col++)
//             if (accessMat(particle_locations, wave, col) == 0) {
//                 int prev = accessMat(flow, wave, col);
//                 update_flow(flow, flow, wave, col, columns);
//                 int var = abs(prev - accessMat(flow, wave, col));
//                 if (var > max_var)
//                     max_var = var;
//             }
//     }

// if (rank == 0)
//     if (num_particles_m > 0)
//         for (int particle = 0; particle < num_particles_m; particle++)
//             move_particle(
//                 flow,
//                 particles_m,
//                 particle,
//                 precision_rows,
//                 precision_columns,
//                 columns,
//                 particles_m_pos,
//                 particles_m_mass
//             );
// MPI_Bcast(flow, rows * columns, MPI_INT, 0, MPI_COMM_WORLD);

// MPI_Scatterv(
//     flow,
//     flow_counts,
//     flow_displs,
//     MPI_INT,
//     flow + flow_displs[rank],
//     flow_counts[rank],
//     MPI_INT,
//     0,
//     MPI_COMM_WORLD
// );

// int *particles_counts = (int *)malloc(comm_size * sizeof(int));
// int *particles_displs = (int *)malloc(comm_size * sizeof(int));
// distribute(num_particles, comm_size, particles_counts, particles_displs);

// if (col == 0) { // Update if border left
//     accessMat(flow, row, col)
//         = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
//            + accessMat(flow_copy, row - 1, col + 1))
//           / 4;
// } else if (col == columns - 1) { // Update if border right
//     accessMat(flow, row, col)
//         = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
//            + accessMat(flow_copy, row - 1, col - 1))
//           / 4;
// } else { // Update in central part
//     accessMat(flow, row, col)
//         = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
//            + accessMat(flow_copy, row - 1, col - 1)
//            + accessMat(flow_copy, row - 1, col + 1))
//           / 5;
// }

// if (rank < comm_size - 1 && blocks_counts[rank + 1] > 0) {
// if (rank > 0 && rank < comm_size - 1) {
//     int rank_last_wave
//         = (blocks_displs[rank] + blocks_counts[rank] - 1) * STEPS + wave_front;
//     int prev_rank_last_wave
//         = (blocks_displs[rank - 1] + blocks_counts[rank - 1] - 1) * STEPS
//           + wave_front;
//
//     MPI_Sendrecv(
//         flow + rank_last_wave * columns,
//         columns,
//         MPI_INT,
//         rank + 1,
//         0,
//         flow + rank_last_wave * columns,
//         columns,
//         MPI_INT,
//         rank - 1,
//         0,
//         MPI_COMM_WORLD,
//         NULL
//     );
// }

// MPI_COMM_WORLD,

// MPI_Isend(
//     flow + rank_last_wave * columns,
//     columns,
//     MPI_INT,
//     rank + 1,
//     0,
//     MPI_COMM_WORLD,
//     &request
// );
// int prev_rank_last_wave
//     = (blocks_displs[rank - 1] + blocks_counts[rank - 1] - 1) * STEPS
//       + wave_front;

// if (rank > 0 && blocks_counts[rank] > 0) {

// if (num_particles > 0) {
// MPI_Scatterv(
//     flow,
//     flow_counts,
//     flow_displs,
//     MPI_INT,
//     flow + flow_displs[rank],
//     flow_counts[rank],
//     MPI_INT,
//     0,
//     MPI_COMM_WORLD
// );
// MPI_Bcast(flow, rows * columns, MPI_INT, 0, MPI_COMM_WORLD);
// if (rank < comm_size - 1 && sect_counts[rank + 1] > 0) {
//     // int rank_last_wave = (sect_displs[rank] + sect_counts[rank]) *
//     STEPS;
//     MPI_Isend(
//         flow + flow_displs[rank] + flow_counts[rank] - columns,
//         // flow + rank_last_wave * columns,
//         columns,
//         MPI_INT,
//         rank + 1,
//         0,
//         MPI_COMM_WORLD,
//         &send_row_request
//     );
// }
//
// if (rank > 0 && sect_counts[rank] > 0) {
//     // int prev_rank_last_wave
//     //     = (sect_displs[rank - 1] + sect_counts[rank - 1]) * STEPS;
//
//     MPI_Irecv(
//         flow + flow_displs[rank - 1] + flow_counts[rank - 1] - columns,
//         // flow + prev_rank_last_wave * columns,
//         columns,
//         MPI_INT,
//         rank - 1,
//         0,
//         MPI_COMM_WORLD,
//         &recv_row_request
//     );
// }
// }
// if (particle_locations_request != NULL) {
//     MPI_Wait(&particle_locations_request, NULL);
//     particle_locations_request = NULL;
// }
