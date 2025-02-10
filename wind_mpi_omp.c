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

/* Structure to represent a solid particle in the tunnel surface */
typedef struct {
    unsigned char extra;      // Extra field for student's usage
    int pos_row, pos_col;     // Position in the grid
    int mass;                 // Particle mass
    int resistance;           // Resistance to air flow
    int speed_row, speed_col; // Movement direction and speed
    int old_flow;             // To annotate the flow before applying effects
} particle_t;

static int particle_cmp(const void *p1, const void *p2) {
    particle_t *p_1 = (particle_t *)p1, *p_2 = (particle_t *)p2;

    if (p_1->pos_row < p_2->pos_row)
        return -1;
    if (p_1->pos_row > p_2->pos_row)
        return 1;
    if (p_1->pos_col < p_2->pos_col)
        return -1;
    if (p_1->pos_col > p_2->pos_col)
        return 1;

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
void update_flow(int *flow, int *flow_copy, int row, int col, int columns) {
    if (col == 0) { // Update if border left
        accessMat(flow, row, col)
            = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
               + accessMat(flow_copy, row - 1, col + 1))
              / 4;
    } else if (col == columns - 1) { // Update if border right
        accessMat(flow, row, col)
            = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
               + accessMat(flow_copy, row - 1, col - 1))
              / 4;
    } else { // Update in central part
        accessMat(flow, row, col)
            = (accessMat(flow_copy, row, col) + accessMat(flow_copy, row - 1, col) * 2
               + accessMat(flow_copy, row - 1, col - 1)
               + accessMat(flow_copy, row - 1, col + 1))
              / 5;
    }
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

/*
 * Function: Move particle
 * 	This function can be changed and/or optimized by the students
 */
void move_particle(
    int *flow,
    particle_m_t *particles_m,
    int particle,
    int precision_rows,
    int precision_columns,
    int columns,
    vec2_t *particles_m_pos,
    const int *particles_m_mass
) {
    // Compute movement for each step
    int row = particles_m_pos[particle].row;
    int col = particles_m_pos[particle].col;
    int x = col;
    int y = row;

    for (int step = 0; step < STEPS; step++) {
        // Highly simplified phisical model
        int pressure = accessMat(flow, row - 1, col);
        int left = 0, right = 0;
        int calc = 0;
        if (col > 0)
            left = pressure - accessMat(flow, row - 1, col - 1);
        if (col < columns - 1)
            right = pressure - accessMat(flow, row - 1, col + 1);

        int flow_row = (int)((float)pressure / particles_m_mass[particle] * PRECISION);
        int flow_col = (int)((float)(right - left) / particles_m_mass[particle] * PRECISION);

        // Speed change
        particles_m[particle].speed.row = (particles_m[particle].speed.row + flow_row) / 2;
        particles_m[particle].speed.col = (particles_m[particle].speed.col + flow_col) / 2;

        // Movement
        particles_m[particle].pos.row += particles_m[particle].speed.row / STEPS / 2;
        particles_m[particle].pos.col += particles_m[particle].speed.col / STEPS / 2;

        // Control limits
        if (particles_m[particle].pos.row >= precision_rows)
            particles_m[particle].pos.row = precision_rows - 1;
        if (particles_m[particle].pos.col < 0)
            particles_m[particle].pos.col = 0;
        if (particles_m[particle].pos.col >= precision_columns)
            particles_m[particle].pos.col = precision_columns - 1;

        row = (particles_m[particle].pos.row / PRECISION);
        col = (particles_m[particle].pos.col / PRECISION);
    }

    particles_m_pos[particle].row = row;
    particles_m_pos[particle].col = col;
}
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
    for (j = 0; j < columns; j++)
        printf("---");
    printf("+\n");
    for (i = 0; i < rows; i++) {
        if (i % STEPS == iteration % STEPS)
            printf("->|");
        else
            printf("  |");

        for (j = 0; j < columns; j++) {
            char symbol;
            if (accessMat(flow, i, j) >= 10 * PRECISION)
                symbol = '*';
            else if (accessMat(flow, i, j) >= 1 * PRECISION)
                symbol = '0' + accessMat(flow, i, j) / PRECISION;
            else if (accessMat(flow, i, j) >= 0.5 * PRECISION)
                symbol = '+';
            else if (accessMat(flow, i, j) > 0)
                symbol = '.';
            else
                symbol = ' ';

            if (accessMat(particle_locations, i, j) > 0)
                printf("[%c]", symbol);
            else
                printf(" %c ", symbol);
        }
        printf("|\n");
    }
    printf("  +");
    for (j = 0; j < columns; j++)
        printf("---");
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
    } else
        particles = NULL;

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
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     *
     */

    const int precision_rows = PRECISION * rows,
              precision_columns = PRECISION * columns,
              num_particles_f = num_particles - num_particles_m_band,
              num_particles_m = num_particles_m_band;

    particle_t *particles_moving = particles + num_particles_f;

    // if (rank == 0) {
    qsort(particles, num_particles_f, sizeof(particle_t), particle_cmp);
    qsort(particles_moving, num_particles_m, sizeof(particle_t), particle_cmp);
    // }

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

    MPI_Datatype MPI_VEC2_T;
    MPI_Vec2_t(&MPI_VEC2_T);

    int *particles_counts = malloc(comm_size * sizeof(int)),
        *particles_displs = malloc(comm_size * sizeof(int)),
        *particles_m_counts = malloc(comm_size * sizeof(int)),
        *particles_m_displs = malloc(comm_size * sizeof(int)),
        *particles_m_finals = malloc(comm_size * sizeof(int));

    distribute(num_particles, comm_size, particles_counts, particles_displs);
    distribute(num_particles_m, comm_size, particles_m_counts, particles_m_displs);

    for (int rank = 0; rank < comm_size; rank++)
        particles_m_finals[rank] = particles_m_displs[rank] + particles_m_counts[rank];

    vec2_t *particles_pos = malloc(num_particles * sizeof(vec2_t));
    int *particles_res = malloc(num_particles * sizeof(int)),
        *particles_back = malloc(num_particles * sizeof(int));

    particle_m_t *particles_m = malloc(num_particles_m * sizeof(particle_m_t));
    int *particles_m_mass = malloc(num_particles_m * sizeof(float));
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

    for (int particle = 0; particle < num_particles; particle++)
        accessMat(
            particle_locations, particles_pos[particle].row, particles_pos[particle].col
        )++;

    /* 4. Simulation */
    for (iter = 1; iter <= max_iter && max_var > var_threshold; iter++) {
        if (iter % STEPS == 1) {
            if (rank == 0)
                for (j = inlet_pos; j < inlet_pos + inlet_size; j++) {
                    double phase = iter / STEPS * (M_PI / 4);
                    double phase_step = M_PI / 2 / inlet_size;
                    double pressure_level = 9 + 2 * sin(phase + (j - inlet_pos) * phase_step);
                    double noise = 0.5 - erand48(random_seq);
                    accessMat(flow, 0, j) = (int)(PRECISION * (pressure_level + noise));
                }
            // End inlet update

#ifdef MODULE2
#ifdef MODULE3
            // 4.2. Particles movement each STEPS iterations
            // Clean particle positions

            if (rank == 0)
                if (num_particles_m > 0)
#pragma omp parallel for
                    for (int particle = 0; particle < num_particles_m; particle++)
#pragma omp atomic
                        accessMat(
                            particle_locations,
                            particles_m_pos[particle].row,
                            particles_m_pos[particle].col
                        )--;

            // TODO: copy just the necessary space, precalc minrow etc... with a reduction maybe?
            MPI_Bcast(flow, rows * columns, MPI_INT, 0, MPI_COMM_WORLD);

            if (num_particles_m > 0)
#pragma omp parallel for
                for (int particle = particles_m_displs[rank]; particle < particles_m_finals[rank]; particle++)
                    move_particle(
                        flow,
                        particles_m,
                        particle,
                        precision_rows,
                        precision_columns,
                        columns,
                        particles_m_pos,
                        particles_m_mass
                    );

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

            if (rank == 0)
                if (num_particles_m > 0)
#pragma omp parallel for
                    for (int particle = 0; particle < num_particles_m; particle++)
#pragma omp atomic
                        accessMat(
                            particle_locations,
                            particles_m_pos[particle].row,
                            particles_m_pos[particle].col
                        )++;
#endif // MODULE3

            if (rank == 0)
                if (num_particles > 0)
#pragma omp parallel for
                    for (int particle = 0; particle < num_particles; particle++)
                        update_flow(
                            flow,
                            flow_copy,
                            particles_pos[particle].row,
                            particles_pos[particle].col,
                            columns
                        );

            if (rank == 0)
                if (num_particles > 0)
#pragma omp parallel for
                    for (int particle = 0; particle < num_particles; particle++) {
                        int row = particles_pos[particle].row;
                        int col = particles_pos[particle].col;
                        particles_back[particle] = (int)((long)accessMat(flow, row, col)
                                                         * particles_res[particle] / PRECISION)
                                                   / accessMat(particle_locations, row, col);
                    }

            if (rank == 0)
                if (num_particles > 0)
                    for (int particle = 0; particle < num_particles; particle++) {
                        int row = particles_pos[particle].row;
                        int col = particles_pos[particle].col;
                        int back = particles_back[particle];

                        accessMat(flow, row, col) -= back;
                        accessMat(flow, row - 1, col) += back / 2;

                        if (col > 0)
                            accessMat(flow, row - 1, col - 1) += back / 4;
                        else
                            accessMat(flow, row - 1, col) += back / 4;

                        if (col < columns - 1)
                            accessMat(flow, row - 1, col + 1) += back / 4;
                        else
                            accessMat(flow, row - 1, col) += back / 4;
                    }

            max_var = 0;
#endif // MODULE2
        }

        int wave_front = iter % STEPS;
        if (wave_front == 0) {
            wave_front = STEPS;

            // if (rank == 0)
            //     if (num_particles > 0)
            //         memcpy(flow_copy, flow, rows * columns * sizeof(int));

            if (rank == 0)
                if (num_particles_f > 0) {
#pragma omp parallel for
                    for (int particle = 0; particle < num_particles_f; particle++) {
                        int row = particles_pos[particle].row;
                        int col = particles_pos[particle].col;

                        accessMat(flow_copy, row, col) = accessMat(flow, row, col);
                        accessMat(flow_copy, row - 1, col) = accessMat(flow, row - 1, col);
                        if (col > 0)
                            accessMat(flow_copy, row - 1, col - 1)
                                = accessMat(flow, row - 1, col - 1);
                        if (col < columns - 1)
                            accessMat(flow_copy, row - 1, col + 1)
                                = accessMat(flow, row - 1, col + 1);
                    }
                }

            if (rank == 0)
                if (num_particles_m > 0) {
                    int min_row = INT_MAX;
                    int max_row = 0;

#pragma omp parallel for reduction(min : min_row) reduction(max : max_row)
                    for (int particle = 0; particle < num_particles_m; particle++) {
                        if (particles_m_pos[particle].row < min_row)
                            min_row = particles_m_pos[particle].row;
                        if (particles_m_pos[particle].row > max_row)
                            max_row = particles_m_pos[particle].row;
                    }

                    if (min_row > 1)
                        min_row--;
                    max_row += (rows - max_row < 3 ? rows - max_row : 3);
                    memcpy(
                        flow_copy + min_row * columns,
                        flow + min_row * columns,
                        (max_row - min_row) * columns * sizeof(int)
                    );
                }
        }

        int last_wave = iter + 1 < rows ? iter + 1 : rows;
        if (rank == 0)
#pragma omp parallel for reduction(max : max_var)
            for (int wave = wave_front; wave < last_wave; wave += STEPS) {
                for (int col = 0; col < columns; col++)
                    if (accessMat(particle_locations, wave, col) == 0) {
                        int prev = accessMat(flow, wave, col);
                        update_flow(flow, flow, wave, col, columns);
                        int var = abs(prev - accessMat(flow, wave, col));
                        if (var > max_var)
                            max_var = var;
                    }
            }

        // MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&max_var, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG
        // 4.7. DEBUG: Print the current state of the simulation at the end of each iteration
        if (rank == 0) {
            fflush(stdout);
            print_status(iter, rows, columns, flow, num_particles, particle_locations, max_var);
            fflush(stdout);
        }
#endif

    } // End iterations

    // MPI_Bcast(flow, rows * columns, MPI_INT, 0, MPI_COMM_WORLD);

    // MPI: Fill result arrays used for later output
    int ind;
    for (ind = 0; ind < 6; ind++)
        resultsA[ind] = accessMat(flow, STEPS - 1, ind * columns / 6);

    int res_row = (iter - 1 < rows - 1) ? iter - 1 : rows - 1;
    for (ind = 0; ind < 6; ind++)
        resultsB[ind] = accessMat(flow, res_row / 2, ind * columns / 6);

    for (ind = 0; ind < 6; ind++)
        resultsC[ind] = accessMat(flow, res_row, ind * columns / 6);

    // MPI Version: Eliminate this conditional-end to start doing the work in parallel
    /*}*/

    /*
     *
     * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
     *
     */

    /* 5. Stop global timer */
    MPI_Barrier(MPI_COMM_WORLD);
    ttotal = cp_Wtime() - ttotal;

    /* 6. Output for leaderboard */
    if (rank == 0) {
        printf("\n");
        /* 6.1. Total computation time */
        printf("Time: %lf\n", ttotal);

        /* 6.2. Results: Statistics */
        printf("Result: %d, %d", iter - 1, max_var);
        int i;
        for (i = 0; i < 6; i++)
            printf(", %d", resultsA[i]);
        for (i = 0; i < 6; i++)
            printf(", %d", resultsB[i]);
        for (i = 0; i < 6; i++)
            printf(", %d", resultsC[i]);
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
