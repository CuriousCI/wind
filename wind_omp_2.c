/*
 * Simplified simulation of air flow in a wind tunnel
 *
 * OpemMP version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2020/2021
 *
 * v1.4
 *
 * (c) 2021 Arturo Gonzalez Escribano
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0
 * International License. https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "util.h"

/* Headers for the OpenMP assignment versions */
#include <omp.h>

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
 * Macro function to simplify accessing with two coordinates to a flattened
 * array This macro-function can be changed and/or optimized by the students
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
        "<rows> <columns> <maxIter> <threshold> <inlet_pos> "
        "<inlet_size> <fixed_particles_pos> <fixed_particles_size> "
        "<fixed_particles_density> <moving_particles_pos> "
        "<moving_particles_size> <moving_particles_density> "
        "<short_rnd1> <short_rnd2> <short_rnd3> [ <fixed_row> "
        "<fixed_col> <fixed_resistance> ... ]\n"
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

    int *flow;               // Wind tunnel air-flow
    int *flow_copy;          // Wind tunnel air-flow (ancillary copy)
    int *particle_locations; // To quickly locate places with particles

    int inlet_pos;             // First position of the inlet
    int inlet_size;            // Inlet size
    int particles_f_band_pos;  // First position of the band where fixed
                               // particles start
    int particles_f_band_size; // Size of the band where fixed particles start
    int particles_m_band_pos;  // First position of the band where moving
                               // particles start
    int particles_m_band_size; // Size of the band where moving particles start
    float particles_f_density; // Density of starting fixed particles
    float particles_m_density; // Density of starting moving particles

    unsigned short random_seq[3]; // Status of the random sequence

    int num_particles;     // Number of particles
    particle_t *particles; // List to store cells information

    /* 1. Read simulation arguments */
    /* 1.1. Check minimum number of arguments */
    if (argc < 16) {
        fprintf(
            stderr,
            "-- Error: Not enough arguments when reading "
            "configuration from the command line\n\n"
        );
        show_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    /* 1.2. Read simulation area sizes, maximum number of iterations and
     * threshold */
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
            exit(EXIT_FAILURE);
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
            exit(EXIT_FAILURE);
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
    printf(
        "Arguments, Rows: %d, Columns: %d, max_iter: %d, threshold: %f\n",
        rows,
        columns,
        max_iter,
        (float)var_threshold / PRECISION
    );
    printf(
        "Arguments, Inlet: %d, %d  Band of fixed particles: %d, %d, %f  "
        "Band of moving particles: %d, %d, %f\n",
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
#endif // DEBUG

    /* 2. Start global timer */
    double ttotal = cp_Wtime();

    /*
     *
     * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
     *
     */

    const int precision_rows = PRECISION * rows, precision_columns = PRECISION * columns,
              num_particles_f = num_particles - num_particles_m_band,
              num_particles_m = num_particles_m_band;

    particle_t *moving_particles = particles + num_particles_f;

    /* 3. Initialization */
    flow = calloc(rows * columns, sizeof(int));
    flow_copy = calloc(rows * columns, sizeof(int));
    particle_locations = calloc(rows * columns, sizeof(int));

    qsort(particles, num_particles_f, sizeof(particle_t), particle_cmp);
    qsort(moving_particles, num_particles_m, sizeof(particle_t), particle_cmp);

    particle_m_t *particles_m = malloc(num_particles_m_band * sizeof(particle_m_t));

    vec2_t *particles_pos = malloc(num_particles * sizeof(vec2_t));
    vec2_t *particles_m_pos = particles_pos + num_particles_f;

    int *particles_back = malloc(num_particles * sizeof(int)),
        *particles_m_back = particles_back + num_particles_f;

    int *particles_res = malloc(num_particles * sizeof(int));
    int *particles_m_mass_part = malloc(num_particles_m * sizeof(float));

#pragma omp parallel for
    for (int particle = 0; particle < num_particles_m; particle++) {
        particles_m[particle] = (particle_m_t){
            .pos = {
                .row = moving_particles[particle].pos_row,
                .col = moving_particles[particle].pos_col,
            },
            .speed = {
                .row = moving_particles[particle].speed_row,
                .col = moving_particles[particle].speed_col,
            },
        };

        particles_m_mass_part[particle] = moving_particles[particle].mass;
    }

    const int *particles_m_mass = particles_m_mass_part;
    const int *particles_m_res = particles_res + num_particles_f;

#pragma omp parallel for
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
    int max_var = INT_MAX, iter;
    for (iter = 1; iter <= max_iter && max_var > var_threshold; iter++) {
        // 4.1. Change inlet values each STEP iterations
        if (iter % STEPS == 1) {
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

#pragma omp parallel
            {
#pragma omp for
                for (int particle = 0; particle < num_particles_m; particle++) {
#pragma omp atomic
                    accessMat(
                        particle_locations,
                        particles_m_pos[particle].row,
                        particles_m_pos[particle].col
                    )--;

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

#pragma omp atomic
                    accessMat(
                        particle_locations,
                        particles_m_pos[particle].row,
                        particles_m_pos[particle].col
                    )++;
                }

#endif // MODULE3

#pragma omp for
                for (int particle = 0; particle < num_particles_f; particle++)
                    update_flow(
                        flow,
                        flow_copy,
                        particles_pos[particle].row,
                        particles_pos[particle].col,
                        columns
                    );

#pragma omp for
                for (int particle = 0; particle < num_particles_m; particle++)
                    update_flow(
                        flow,
                        flow_copy,
                        particles_m_pos[particle].row,
                        particles_m_pos[particle].col,
                        columns
                    );

#pragma omp for
                for (int particle = 0; particle < num_particles_f; particle++) {
                    int row = particles_pos[particle].row;
                    int col = particles_pos[particle].col;
                    particles_back[particle] = (int)((long)accessMat(flow, row, col)
                                                     * particles_res[particle] / PRECISION)
                                               / accessMat(particle_locations, row, col);
                }

#pragma omp for nowait
                for (int particle = 0; particle < num_particles_m; particle++) {
                    int row = particles_m_pos[particle].row;
                    int col = particles_m_pos[particle].col;
                    particles_m_back[particle] = (int)((long)accessMat(flow, row, col)
                                                       * particles_m_res[particle] / PRECISION)
                                                 / accessMat(particle_locations, row, col);
                }

#pragma omp for
                for (int particle = 0; particle < num_particles; particle++) {
                    int row = particles_pos[particle].row;
                    int col = particles_pos[particle].col;
                    int back = particles_back[particle];

                    accessMat(flow, row, col) -= back;
                    accessMat(flow, row - 1, col) += back / 2;

                    if (col > 0) {
                        accessMat(flow, row - 1, col - 1) += back / 4;
                    } else {
                        accessMat(flow, row - 1, col) += back / 4;
                    }

                    if (col < columns - 1) {
                        accessMat(flow, row - 1, col + 1) += back / 4;
                    } else {
                        accessMat(flow, row - 1, col) += back / 4;
                    }
                }

                // #pragma omp for
                //                 for (int particle = 0; particle < num_particles_m;
                //                 particle++) {
                //                     int row = particles_m_pos[particle].row;
                //                     int col = particles_m_pos[particle].col;
                //                     int back = particles_m_back[particle];
                //
                //                     accessMat(flow, row, col) -= back;
                //                     accessMat(flow, row - 1, col) += back / 2;
                //
                //                     if (col > 0) {
                //                         accessMat(flow, row - 1, col - 1) += back / 4;
                //                     } else {
                //                         accessMat(flow, row - 1, col) += back / 4;
                //                     }
                //
                //                     if (col < columns - 1) {
                //                         accessMat(flow, row - 1, col + 1) += back / 4;
                //                     } else {
                //                         accessMat(flow, row - 1, col) += back / 4;
                //                     }
                //                 }

#pragma omp for
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

#pragma omp for nowait
                for (int particle = 0; particle < num_particles_m; particle++) {
                    int row = particles_m_pos[particle].row;
                    int col = particles_m_pos[particle].col;

                    accessMat(flow_copy, row, col) = accessMat(flow, row, col);
                    accessMat(flow_copy, row - 1, col) = accessMat(flow, row - 1, col);
                    if (col > 0)
                        accessMat(flow_copy, row - 1, col - 1)
                            = accessMat(flow, row - 1, col - 1);
                    if (col < columns - 1)
                        accessMat(flow_copy, row - 1, col + 1)
                            = accessMat(flow, row - 1, col + 1);
                }

                int last_wave = iter < rows ? iter : rows;
#pragma omp for
                for (int wave = 0; wave < last_wave; wave += STEPS)
                    memcpy(
                        flow_copy + wave * columns, flow + wave * columns, columns * sizeof(int)
                    );
            }

            max_var = 0;
#endif // MODULE2
        }

        int wave_front = iter % STEPS;
        if (wave_front == 0)
            wave_front = STEPS;
        int last_wave = iter + 1 < rows ? iter + 1 : rows;

        // default(none) firstprivate(           \ flow, flow_copy, wave_front,
        // last_wave, particle_locations, columns    \)

        // if (wave_front != STEPS) {
// #pragma omp parallel for reduction(max : max_var)
//     for (int wave = wave_front; wave < last_wave; wave += STEPS) {
//         for (int col = 0; col < columns; col++)
//             if (accessMat(particle_locations, wave, col) == 0) {
//                 int prev = accessMat(flow_copy, wave, col);
//                 update_flow(flow, flow_copy, wave, col, columns);
//
//                 int var = abs(prev - accessMat(flow, wave, col));
//                 if (var > max_var)
//                     max_var = var;
//             }
//     }
// } else {
#pragma omp parallel for reduction(max : max_var)
        for (int wave = wave_front; wave < last_wave; wave += STEPS) {
            for (int col = 0; col < columns; col++)
                if (accessMat(particle_locations, wave, col) == 0) {
                    update_flow(flow, flow_copy, wave, col, columns);

                    int var = abs(accessMat(flow_copy, wave, col) - accessMat(flow, wave, col));
                    if (var > max_var)
                        max_var = var;

                    if (wave_front != STEPS)
                        accessMat(flow_copy, wave, col) = accessMat(flow, wave, col);
                }
        }
        // }

        // if (wave_front != STEPS)
        //     memcpy(
        //         flow_copy + wave * columns, flow + wave * columns, columns * sizeof(int)
        //     );

#ifdef DEBUG
        // 4.7. DEBUG: Print the current state of the simulation at
        // the end of each iteration
        print_status(iter, rows, columns, flow, num_particles, particle_locations, max_var);
#endif
    } // End iterations
    /*
     *
     * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
     *
     */

    /* 5. Stop global timer */
    ttotal = cp_Wtime() - ttotal;

    /* 6. Output for leaderboard */
    printf("\n");
    /* 6.1. Total computation time */
    printf("Time: %lf\n", ttotal);

    /* 6.2. Results: Statistics */
    printf("Result: %d, %d", iter - 1, max_var);
    int res_row = (iter - 1 < rows - 1) ? iter - 1 : rows - 1;
    int ind;
    for (ind = 0; ind < 6; ind++)
        printf(", %d", accessMat(flow, STEPS - 1, ind * columns / 6));
    for (ind = 0; ind < 6; ind++)
        printf(", %d", accessMat(flow, res_row / 2, ind * columns / 6));
    for (ind = 0; ind < 6; ind++)
        printf(", %d", accessMat(flow, res_row, ind * columns / 6));
    printf("\n");

    /* 7. Free resources */
    free(flow);
    free(flow_copy);
    free(particle_locations);
    free(particles);

    /* 8. End */
    return 0;
}

/*for (int wave_front = 1; wave_front <= STEPS && iter <= max_iter &&
 * max_var > var_threshold; wave_front++, iter++) {*/
/*    if (wave_front == 1)*/
/*        max_var = 0;*/
/**/
/*    // 4.4. Copy data in the ancillary structure*/
/*    memcpy(flow_copy, flow, (iter < rows ? iter : rows) * columns *
 * sizeof(int));*/
/**/
/*    // 4.5.2. Execute propagation on the wave fronts*/
/*    int stop = iter + 1 <= rows ? iter + 1 : rows;*/
/**/
/*    #pragma omp parallel for num_threads(6)*/
/*    for (int col = 0; col < columns; col++) {*/
/*        for (int wave = wave_front; wave < stop; wave += STEPS) {*/
/*            int var = update_flow(flow, flow_copy, particle_locations,
 * wave, col, columns, 1);*/
/*            if (var > max_var)*/
/*                max_var = var;*/
/*        }*/
/*    }*/
/*}*/

/*omp_lock_t *matrix_locks = malloc(rows * columns * sizeof(omp_lock_t));*/
/*for (size_t i = 0; i < rows; i++)*/
/*    for (size_t j = 0; j < columns; j++)*/
/*        omp_init_lock(&accessMat(matrix_locks, i, j));*/

/*#pragma omp parallel for schedule(static, 1)*/
/*omp_set_lock(&accessMat(matrix_locks, row, col));*/
/*omp_set_lock(&accessMat(matrix_locks, row - 1, col));*/
/*omp_unset_lock(&accessMat(matrix_locks, row, col));*/
/*omp_unset_lock(&accessMat(matrix_locks, row - 1, col));*/

/*for (size_t i = 0; i < rows; i++)*/
/*    for (size_t j = 0; j < columns; j++)*/
/*        omp_destroy_lock(&accessMat(matrix_locks, i, j));*/
/*for (int particle = 0; particle < num_particles_f; particle++) {*/
/*    particles_pos[particle].row = particles[particle].pos_row /
 * PRECISION;*/
/*    particles_pos[particle].col = particles[particle].pos_col /
 * PRECISION;*/
/*    accessMat(fixed_particle_locations, particles_pos[particle].row,
 * particles_pos[particle].col)++;*/
/*}*/
/*accessMat(fixed_particle_locations, particles_pos[particle].row,
 * particles_pos[particle].col)++;*/

/*particles_pos[particle].col;*/
/* TODO: basically, calculate displacements based on the number of threads,
 * and parallelize that */
/*memcpy(particle_locations, fixed_particle_locations, rows * columns *
 * sizeof(int));*/
/*for (int particle = num_particles_f; particle < num_particles;
 * particle++)*/
/*    accessMat(particle_locations, particles_pos[particle].row,
 * particles_pos[particle].col)++;*/
/*int *fixed_particle_locations = calloc(rows * columns, sizeof(int));*/
/*for (size_t i = 0; i < rows; i++)*/
/*    for (size_t j = 0; j < columns; j++)*/
/*        omp_destroy_lock(&accessMat(matrix_locks, i, j));*/
/* TODO: OMP, get_num_threads, get_thread_id, do particle iff in range */
/* TODO: build dynamic matrix*/
/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles; particle++)*/
/*backs[particle] = (int)((long)particles[particle].old_flow *
 * particles[particle].resistance / PRECISION) /
 * accessMat(particle_locations, particles_pos[particle].row,
 * particles_pos[particle].col);*/
/*#pragma omp parallel for num_threads(pool_size) schedule(static, 1)*/
/*            for (int rank = 0; rank < pool_size; rank++) {*/
/*                memcpy(*/
/*                    flow_copy + rows_displs[rank] * columns,*/
/*                    flow + rows_displs[rank] * columns,*/
/*                    rows_counts[rank] * columns * sizeof(int));*/
/*            }*/
/*if (min_particle_row > 0)*/
/*    memcpy(flow_copy, flow, columns * sizeof(int));*/
/*memcpy(flow_copy + min_particle_row * columns, flow, particle_row_dist *
 * columns * sizeof(int));*/
/*int *rows_displs = malloc(pool_size * sizeof(int)),*/
/*    *rows_counts = malloc(pool_size * sizeof(int));*/
/*scatter_all(rows, pool_size, rows_displs, rows_counts);*/
/*sect_t rows_sect = sector(rows, );*/

/*int update_flow(int *flow, int *flow_copy, int *particle_locations, int
 * row, int col, int columns, int skip_particles) {*/
/*    // Skip update in particle positions*/
/*    if (skip_particles && accessMat(particle_locations, row, col) != 0)*/
/*        return 0;*/
/**/
/*    if (col == 0) { // Update if border left*/
/*        accessMat(flow, row, col) =*/
/*            (accessMat(flow_copy, row, col) +*/
/*             accessMat(flow_copy, row - 1, col) * 2 +*/
/*             accessMat(flow_copy, row - 1, col + 1)) /*/
/*            4;*/
/*    } else if (col == columns - 1) { // Update if border right*/
/*        accessMat(flow, row, col) =*/
/*            (accessMat(flow_copy, row, col) +*/
/*             accessMat(flow_copy, row - 1, col) * 2 +*/
/*             accessMat(flow_copy, row - 1, col - 1)) /*/
/*            4;*/
/*    } else { // Update in central part*/
/*        accessMat(flow, row, col) =*/
/*            (accessMat(flow_copy, row, col) +*/
/*             accessMat(flow_copy, row - 1, col) * 2 +*/
/*             accessMat(flow_copy, row - 1, col - 1) +*/
/*             accessMat(flow_copy, row - 1, col + 1)) /*/
/*            5;*/
/*    }*/
/**/
/*    // Return flow variation at this position*/
/*    return abs(accessMat(flow_copy, row, col) - accessMat(flow, row,
 * col));*/
/*}*/
/*int pool_size = omp_get_num_threads();*/
/*particles_m[particle].base.old_flow = accessMat(flow, row, col);*/
/*backs[f_particle] = (int)((long)particles_m[particle].base.old_flow *
 * particles_m[particle].base.resistance / PRECISION) /
 * accessMat(particle_locations, particles_pos[f_particle].row,
 * particles_pos[f_particle].col);*/

/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles; particle++)
 * {*/
/*                int row = particles_mat_pos[particle].row;*/
/*                int col = particles_mat_pos[particle].col;*/
/**/
/*                // update_flow(flow, flow_copy, particle_locations, row,
 * col, columns, 0);*/
/*                update_flow(flow, flow_copy, row, col, columns);*/
/*                particles[particle].old_flow = accessMat(flow, row,
 * col);*/
/*                particles_back[particle] =
 * (int)((long)particles[particle].old_flow * particles[particle].resistance
 * / PRECISION) / accessMat(particle_locations,
 * particles_mat_pos[particle].row, particles_mat_pos[particle].col);*/
/*            }*/

/*void move_particle(int *flow, particle_t *particles, int particle, int
 * rows, int columns, vec2_t *particles_pos) {*/
/*    // Compute movement for each step*/
/*    for (int step = 0; step < STEPS; step++) {*/
/*        int row = particles_pos[particle].row;*/
/*        int col = particles_pos[particle].col;*/
/*        // Highly simplified phisical model*/
/*        // TODO: loro due*/
/*        // int row = particles[particle].pos_row / PRECISION;*/
/*        // int col = particles[particle].pos_col / PRECISION;*/
/**/
/*        int pressure = accessMat(flow, row - 1, col);*/
/*        int left, right;*/
/*        if (col == 0)*/
/*            left = 0;*/
/*        else*/
/*            left = pressure - accessMat(flow, row - 1, col - 1);*/
/*        if (col == columns - 1)*/
/*            right = 0;*/
/*        else*/
/*            right = pressure - accessMat(flow, row - 1, col + 1);*/
/**/
/*        //TODO: loro due */
/*        int flow_row = (int)((float)pressure / particles[particle].mass *
 * PRECISION);*/
/*        int flow_col = (int)((float)(right - left) /
 * particles[particle].mass
 * * PRECISION);*/
/**/
/*        // Speed change*/
/*        particles[particle].speed_row = (particles[particle].speed_row +
 * flow_row) / 2;*/
/*        particles[particle].speed_col = (particles[particle].speed_col +
 * flow_col) / 2;*/
/**/
/*        // Movement*/
/*        particles[particle].pos_row += particles[particle].speed_row /
 * STEPS / 2;*/
/*        particles[particle].pos_col += particles[particle].speed_col /
 * STEPS / 2;*/
/**/
/*        // Control limits*/
/*        // TODO: don't do stuff if already on border*/
/*        if (particles[particle].pos_row >= PRECISION * rows)*/
/*            particles[particle].pos_row = PRECISION * rows - 1;*/
/*        if (particles[particle].pos_col < 0)*/
/*            particles[particle].pos_col = 0;*/
/*        if (particles[particle].pos_col >= PRECISION * columns)*/
/*            particles[particle].pos_col = PRECISION * columns - 1;*/
/**/
/*        particles_pos[particle].row = (row = particles[particle].pos_row /
 * PRECISION);*/
/*        particles_pos[particle].col = (col = particles[particle].pos_col /
 * PRECISION);*/
/*    }*/
/*}*/
/*int var = update_flow(flow, flow_copy, particle_locations, wave, col,
 * columns, 1);*/

/*#pragma omp parallel for schedule(static, 1)*/
/*            for (int particle = 0; particle < num_particles_m;
 * particle++)*/
/*#pragma omp atomic*/
/*                accessMat(particle_locations,
 * particles_m_pos[particle].row, particles_m_pos[particle].col)--;*/
/*#pragma omp parallel for schedule(static, 1)*/
/*            for (int particle = 0; particle < num_particles_m;
 * particle++)*/
/*#pragma omp atomic*/
/*                accessMat(particle_locations,
 * particles_m_pos[particle].row, particles_m_pos[particle].col)++;*/

/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles_f;
 * particle++)*/
/*                particles_old_flow[particle] = accessMat(flow,
 * particles_pos[particle].row, particles_pos[particle].col);*/
/**/
/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles_m;
 * particle++)*/
/*                particles_old_flow[particle] = accessMat(flow,
 * particles_m_pos[particle].row, particles_m_pos[particle].col);*/
/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles_f;
 * particle++)*/
/*                particles_back[particle] =
 * (int)((long)particles_old_flow[particle] * particles_res[particle] /
 * PRECISION) / accessMat(particle_locations, particles_pos[particle].row,
 * particles_pos[particle].col);*/
/**/
/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles_m;
 * particle++)*/
/*                particles_back[particle] =
 * (int)((long)particles_old_flow[particle] * particles_res[particle] /
 * PRECISION) / accessMat(particle_locations, particles_m_pos[particle].row,
 * particles_m_pos[particle].col);*/

/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles_f; particle++)
 * {*/
/*                int row = particles_pos[particle].row;*/
/*                int col = particles_pos[particle].col;*/
/**/
/*                update_flow(flow, flow_copy, row, col, columns);*/
/*                particles_old_flow[particle] = accessMat(flow, row,
 * col);*/
/*                particles_back[particle] =
 * (int)((long)particles_old_flow[particle] * particles_res[particle] /
 * PRECISION) / accessMat(particle_locations, row, col);*/
/*            }*/
/**/
/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles_m; particle++)
 * {*/
/*                int row = particles_m_pos[particle].row;*/
/*                int col = particles_m_pos[particle].col;*/
/**/
/*                update_flow(flow, flow_copy, row, col, columns);*/
/*                particles_m_old_flow[particle] = accessMat(flow, row,
 * col);*/
/*                particles_m_back[particle] =
 * (int)((long)particles_m_old_flow[particle] * particles_m_res[particle] /
 * PRECISION) / accessMat(particle_locations, row, col);*/
/*            }*/

/* PAZZO */
/*                if (col > 0) {*/
/*#pragma omp atomic*/
/*                    accessMat(flow, row - 1, col - 1) += back / 4;*/
/*                } else {*/
/*#pragma omp atomic*/
/*                    accessMat(flow, row - 1, col) += back / 4;*/
/*                }*/
/**/
/*                if (col < columns - 1) {*/
/*#pragma omp atomic*/
/*                    accessMat(flow, row - 1, col + 1) += back / 4;*/
/*                } else {*/
/*#pragma omp atomic*/
/*                    accessMat(flow, row - 1, col) += back / 4;*/
/*                }*/
/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles; particle++)
 * {*/
/*                int row = particles_pos[particle].row;*/
/*                int col = particles_pos[particle].col;*/
/**/
/*                update_flow(flow, flow_copy, row, col, columns);*/
/*                particles_old_flow[particle] = accessMat(flow, row,
 * col);*/
/*                particles_back[particle] =
 * (int)((long)particles_old_flow[particle] * particles_res[particle] /
 * PRECISION) / accessMat(particle_locations, particles_pos[particle].row,
 * particles_pos[particle].col);*/
/*            }*/

// 4.3. Effects due to particles each STEPS iterations
/**/
/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles; particle++)
 * {*/
/*                particles_old_flow[particle] = accessMat(flow,
 * particles_pos[particle].row, particles_pos[particle].col);*/
/*                particles_back[particle] =
 * (int)((long)particles_old_flow[particle] * particles_res[particle] /
 * PRECISION) / accessMat(particle_locations, particles_pos[particle].row,
 * particles_pos[particle].col);*/
/*            }*/

/*#pragma omp parallel for*/
/*            for (int particle = 0; particle < num_particles; particle++)*/
/*particles_back[particle] = (int)((long)particles_old_flow[particle] *
 * particles_res[particle] / PRECISION) / accessMat(particle_locations,
 * particles_pos[particle].row, particles_pos[particle].col);*/

// Return flow variation at this position
/*return abs(accessMat(flow_copy, row, col) - accessMat(flow, row, col));*/
/*return 0;*/

/*else {*/
/*int wave_front = (iter - 1) % STEPS;*/
/*if (wave_front == 0)*/
/*    wave_front = STEPS;*/

/*#pragma omp parallel for*/
/*            for (int wave = wave_front; wave < (iter < rows ? iter :
 * rows); wave += STEPS)*/
/*                memcpy(flow_copy + wave * columns, flow + wave * columns,
 * columns * sizeof(int));*/
/*}*/
/*memcpy(flow_copy + wave * columns, flow + wave * columns, columns);*/
/*if (iter % STEPS != 0)*/
/*    for (int col = 0; col < columns; col++)*/
/*        accessMat(flow_copy, wave, col) = accessMat(flow, wave, col);*/
/*memcpy(flow_copy, flow, rows * columns * sizeof(int));*/
/*else {*/
/*    int wave_front = (iter - 1) % STEPS;*/
/*    if (wave_front == 0)*/
/*        wave_front = STEPS;*/
/*    int last_wave = iter < rows ? iter : rows;*/
/*    for (int wave = 0; wave < last_wave; wave += STEPS)*/
/*        memcpy(*/
/*            flow_copy + wave * columns,*/
/*            flow + wave * columns,*/
/*            columns * sizeof(int)*/
/*        );*/
/*}*/

/*for (j = inlet_pos; j < inlet_pos + inlet_size; j++) {*/
/*    accessMat(flow_copy, 0, j) = accessMat(flow, 0, j);*/
/*}*/
/*for (int particle = 0; particle < num_particles; particle++) {*/
/*    int row = particles_pos[particle].row;*/
/*    int col = particles_pos[particle].col;*/
/*    accessMat(flow_copy, row, col) = accessMat(flow, row, col);*/
/*}*/

// 4.4. Copy data in the ancillary structure
/*if (iter % STEPS == 1)*/
/*    memcpy(flow_copy, flow, rows * columns * sizeof(int));*/
/*memcpy(*/
/*    flow_copy,*/
/*    flow,*/
/*    (iter + 1 < rows ? iter + 1 : rows) * columns * sizeof(int)*/
/*);*/
/*int num_particles_f_dist = 0;*/
/*vec2_t *particles_f_pos_dist*/
/*    = malloc(num_particles_f_dist * sizeof(vec2_t));*/
/*#pragma omp parallel for*/
/*            for (int particle = 0; particle <
 * num_particles_f_dist; particle++)*/
/*                update_flow(*/
/*                    flow,*/
/*                    flow_copy,*/
/*                    particles_f_pos_dist[particle].row,*/
/*                    particles_f_pos_dist[particle].col,*/
/*                    columns*/
/*                );*/

/*for (int particle = 0; particle < num_particles_f; particle++) {*/
/*    int row = particles[particle].pos_row / PRECISION;*/
/*    int col = particles[particle].pos_col / PRECISION;*/
/**/
/*    particles_pos[particle].row = row;*/
/*    particles_pos[particle].col = col;*/
/*    particles_res[particle] = particles[particle].resistance;*/
/**/
/*    if (num_particles_f_dist == 0) {*/
/*        num_particles_f_dist++;*/
/*        particles_f_pos_dist = realloc(*/
/*            particles_f_pos_dist, num_particles_f_dist * sizeof(vec2_t)*/
/*        );*/
/**/
/*        particles_f_pos_dist[num_particles_f_dist - 1]*/
/*            = (vec2_t){.row = row, .col = col};*/
/*    } else if (particles_f_pos_dist[num_particles_f_dist - 1].row != row*/
/*               || particles_f_pos_dist[num_particles_f_dist - 1].col*/
/*                      != col) {*/
/*        num_particles_f_dist++;*/
/*        particles_f_pos_dist = realloc(*/
/*            particles_f_pos_dist, num_particles_f_dist * sizeof(vec2_t)*/
/*        );*/
/**/
/*        particles_f_pos_dist[num_particles_f_dist - 1]*/
/*            = (vec2_t){.row = row, .col = col};*/
/*    }*/
/*}*/

/**particles_m_old_flow = particles_old_flow + num_particles_f;*/
/*int *particles_m_res = particles_res + num_particles_f;*/
/*TODO: loro due */
/*int row = particles[particle].pos.row / PRECISION;*/
/*int col = particles[particle].pos.col / PRECISION;*/
/*if (col == 0)*/
/*    left = 0;*/
/*else*/
/*if (col == columns - 1)*/
/*    right = 0;*/
/*else*/

/*#pragma omp parallel for default(none) \*/
/*    shared(particle_locations, \*/
/*               particles_m, \*/
/*               particles_m_pos, \*/
/*               particles_m_mass, \*/
/*               flow, \*/
/*               precision_rows, \*/
/*               precision_columns, \*/
/*               columns, \*/
/*               num_particles_m)*/
/*for (int sect = 0; sect < num_threads; sect++) {*/
/*    for (int particle = particles_f_sect[sect].displ;*/
/*         particle < particles_f_sect[sect].displ*/
/*                        + particles_f_sect[sect].count;*/
/*particle++) {*/

/*sect_t *particles_f_sect = malloc(num_threads * sizeof(sect_t));*/
/*sect_t *particles_m_sect = malloc(num_threads * sizeof(sect_t));*/

/*scatter_all(num_particles_f, num_threads, particles_f_sect);*/
/*for (int sect = 0; sect < num_threads - 1; sect++) {*/
/*    int border*/
/*        = particles_f_sect[sect].displ + particles_f_sect[sect].count;*/
/*    while (particles_pos[border].row == particles_pos[border + 1].row*/
/*           && particles_pos[border].col == particles_pos[border + 1].col)
 * {*/
/*        particles_f_sect[sect].count++;*/
/*        particles_f_sect[sect + 1].displ++;*/
/*        particles_f_sect[sect + 1].count--;*/
/*        int border*/
/*            = particles_f_sect[sect].displ +
 * particles_f_sect[sect].count;*/
/*    }*/
/**/
/*    printf(*/
/*        "%d - %d, %d\n",*/
/*        sect,*/
/*        particles_f_sect[sect].displ,*/
/*        particles_f_sect[sect].count*/
/*    );*/
/*}*/
/*particles_old_flow, \*/
/*particles_m_old_flow, \*/
/*particles_old_flow[particle] = accessMat(flow, row,
 * col);*/
/*particles_m_old_flow[particle] = accessMat(flow, row,
 * col);*/
/*int *particles_old_flow = malloc(num_particles * sizeof(int));*/
/*int *particles_m_old_flow = particles_old_flow + num_particles_f;*/

// #pragma omp for
//                 for (particle = 0; particle <
//                 num_particles_m; particle++) {
//                     int row = particles_m_pos[particle].row;
//                     int col = particles_m_pos[particle].col;
//                     int back = particles_m_back[particle];
//                     int pos;
//
//                     pos = row * columns + col;
// #pragma omp atomic
//                     flow[pos] -= back;
//
//                     pos = (row - 1) * columns + col;
// #pragma omp atomic
//                     flow[pos] += back / 2;
//
//                     if (col > 0) {
//                         pos = (row - 1) * columns + (col
//                         - 1);
// #pragma omp atomic
//                         flow[pos] += back / 4;
//                     } else {
//                         pos = (row - 1) * columns + col;
// #pragma omp atomic
//                         flow[pos] += back / 4;
//                     }
//
//                     if (col < columns - 1) {
//                         pos = (row - 1) * columns + (col
//                         + 1);
// #pragma omp atomic
//                         flow[pos] += back / 4;
//                     } else {
//                         pos = (row - 1) * columns + col;
// #pragma omp atomic
//                         flow[pos] += back / 4;
//                     }

// #pragma omp atomic
//                     accessMat(flow, row, col) -= back;
// #pragma omp atomic
//                     accessMat(flow, row - 1, col) += back /
//                     2; if (col > 0)
// #pragma omp atomic
//                         accessMat(flow, row - 1, col - 1) +=
//                         back / 4;
//                     else
// #pragma omp atomic
//                         accessMat(flow, row - 1, col) += back
//                         / 4;
//                     if (col < columns - 1)
// #pragma omp atomic
//                         accessMat(flow, row - 1, col + 1) +=
//                         back / 4;
//                     else
// #pragma omp atomic
//                         accessMat(flow, row - 1, col) += back
//                         / 4;
//                 }

//                     int pos;
//
//                     pos = row * columns + col;
// #pragma omp atomic
//                     flow[pos] -= back;
//
//                     pos = (row - 1) * columns + col;
// #pragma omp atomic
//                     flow[pos] += back / 2;
//
//                     if (col > 0) {
//                         pos = (row - 1) * columns + (col
//                         - 1);
// #pragma omp atomic
//                         flow[pos] += back / 4;
//                     } else {
//                         pos = (row - 1) * columns + col;
// #pragma omp atomic
//                         flow[pos] += back / 4;
//                     }
//
//                     if (col < columns - 1) {
//                         pos = (row - 1) * columns + (col
//                         + 1);
// #pragma omp atomic
//                         flow[pos] += back / 4;
//                     } else {
//                         pos = (row - 1) * columns + col;
// #pragma omp atomic
//                         flow[pos] += back / 4;
//                     }
//          default(none) firstprivate(                               \
//         particle_locations, \
//             flow, \
//             flow_copy, \
//             rows, \
//             columns, \
//             precision_rows, \
//             precision_columns, \
//             iter, \
//             particle, \
//             particles_pos, \
//             particles_res, \
//             particles_back, \
//             particles_m, \
//             particles_m_pos, \
//             particles_m_res, \
//             particles_m_mass, \
//             particles_m_back, \
//             num_particles, \
//             num_particles_f, \
//             num_particles_m, \
// )

// int *t_flow;
// #pragma omp threadprivate(t_flow)
// #pragma omp parallel
//     {
//         t_flow = malloc(rows * columns * sizeof(int));
//     }
// memset(t_flow, 0, rows * columns * sizeof(int));

// int row_col = row * columns + col;
// int row_1_col = (row - 1) * columns + col;
// int row_1_col_m_1 = (row - 1) * columns + col - 1;
// int row_1_col_p_1 = (row - 1) * columns + col + 1;
// int back_2 = back / 2;
// int back_4 = back / 4;
// int row_1_col_back = back_2;
//
// omp_set_lock(&flow_locks[row_col]);
// flow[row_col] -= back;
// omp_unset_lock(&flow_locks[row_col]);
//
// if (col > 0) {
//     omp_set_lock(&flow_locks[row_1_col_m_1]);
//     flow[row_1_col_m_1] += back_4;
//     omp_unset_lock(&flow_locks[row_1_col_m_1]);
// } else {
//     row_1_col_back += back_4;
// }
//
// if (col < columns - 1) {
//     omp_set_lock(&flow_locks[row_1_col_p_1]);
//     flow[row_1_col_m_1] += back_4;
//     omp_unset_lock(&flow_locks[row_1_col_p_1]);
// } else {
//     row_1_col_back += back_4;
// }
//
// omp_set_lock(&flow_locks[row_1_col]);
// flow[row_1_col] += row_1_col_back;
// omp_unset_lock(&flow_locks[row_1_col]);
// omp_lock_t *flow_locks = malloc(rows * columns * sizeof(omp_lock_t));
// for (int i = 0; i < rows * columns; i++)
//     omp_init_lock(flow_locks + i);

// #pragma omp for reduction(+ : flow[ : rows * columns])
//                 for (particle = 0; particle <
//                 num_particles_m; particle++) {
//                     int row = particles_m_pos[particle].row;
//                     int col = particles_m_pos[particle].col;
//                     int back = particles_m_back[particle];
//
//                     // #pragma omp atomic
//                     accessMat(flow, row, col) -= back;
//                     // #pragma omp atomic
//                     accessMat(flow, row - 1, col) += back /
//                     2; if (col > 0)
//                         // #pragma omp atomic
//                         accessMat(flow, row - 1, col - 1) +=
//                         back / 4;
//                     else
//                         // #pragma omp atomic
//                         accessMat(flow, row - 1, col) += back
//                         / 4;
//                     if (col < columns - 1)
//                         // #pragma omp atomic
//                         accessMat(flow, row - 1, col + 1) +=
//                         back / 4;
//                     else
//                         // #pragma omp atomic
//                         accessMat(flow, row - 1, col) += back
//                         / 4;
//                 }

// #pragma omp for
//                 for (particle = 0; particle <
//                 num_particles_m; particle++) {
//                     int row = particles_m_pos[particle].row;
//                     int col = particles_m_pos[particle].col;
//                     int back = particles_m_back[particle];
//
//                     accessMat(flow, row, col) -= back;
//                     accessMat(flow, row - 1, col) += back /
//                     2; if (col > 0)
//                         accessMat(flow, row - 1, col - 1) +=
//                         back / 4;
//                     else
//                         accessMat(flow, row - 1, col) += back
//                         / 4;
//                     if (col < columns - 1)
//                         accessMat(flow, row - 1, col + 1) +=
//                         back / 4;
//                     else
//                         accessMat(flow, row - 1, col) += back
//                         / 4;
//                 }

// int num_flow_free = 0;
// vec2_t *flow_free = malloc(0);
//
// for (int row = 0; row < rows; row++)
//     for (int col = 0; col < columns; col++) {
//
//     }

// int thread_num = omp_get_thread_num();
//
// for (int particle = 0; particle < num_particles_f; particle++) {
//     int row = particles_pos[particle].row;
//     int col = particles_pos[particle].col;
//     int back = particles_back[particle];
//
//     if (accessMat(thread_assign, row, col) == thread_num)
//         accessMat(flow, row, col) -= back;
//
//     if (accessMat(thread_assign, row - 1, col) == thread_num)
//         accessMat(flow, row - 1, col) += back / 2;
//
//     if (col > 0) {
//         if (accessMat(thread_assign, row - 1, col - 1) == thread_num)
//             accessMat(flow, row - 1, col - 1) += back / 4;
//     } else {
//         if (accessMat(thread_assign, row - 1, col) == thread_num)
//             accessMat(flow, row - 1, col) += back / 4;
//     }
//
//     if (col < columns - 1) {
//         if (accessMat(thread_assign, row - 1, col + 1) == thread_num)
//             accessMat(flow, row - 1, col + 1) += back / 4;
//     } else {
//         if (accessMat(thread_assign, row - 1, col) == thread_num)
//             accessMat(flow, row - 1, col) += back / 4;
//     }
// }
//
// for (int particle = 0; particle < num_particles_m; particle++) {
//     int row = particles_m_pos[particle].row;
//     int col = particles_m_pos[particle].col;
//     int back = particles_m_back[particle];
//
//     if (accessMat(thread_assign, row, col) == thread_num)
//         accessMat(flow, row, col) -= back;
//
//     if (accessMat(thread_assign, row - 1, col) == thread_num)
//         accessMat(flow, row - 1, col) += back / 2;
//
//     if (col > 0) {
//         if (accessMat(thread_assign, row - 1, col - 1) == thread_num)
//             accessMat(flow, row - 1, col - 1) += back / 4;
//     } else {
//         if (accessMat(thread_assign, row - 1, col) == thread_num)
//             accessMat(flow, row - 1, col) += back / 4;
//     }
//
//     if (col < columns - 1) {
//         if (accessMat(thread_assign, row - 1, col + 1) == thread_num)
//             accessMat(flow, row - 1, col + 1) += back / 4;
//     } else {
//         if (accessMat(thread_assign, row - 1, col) == thread_num)
//             accessMat(flow, row - 1, col) += back / 4;
//     }
// }

// for (particle = 0; particle < num_particles_m; particle++) {
//     int row = particles_m_pos[particle].row;
//     int col = particles_m_pos[particle].col;
//     int back = particles_m_back[particle];
//
//     if (accessMat(thread_assign, row, col) == thread_num)
//         accessMat(flow, row, col) -= back;
//
//     if (accessMat(thread_assign, row - 1, col) == thread_num)
//         accessMat(flow, row - 1, col) += back / 2;
//
//     if (col > 0) {
//         if (accessMat(thread_assign, row - 1, col - 1) == thread_num)
//             accessMat(flow, row - 1, col - 1) += back / 4;
//     } else {
//         if (accessMat(thread_assign, row - 1, col) == thread_num)
//             accessMat(flow, row - 1, col) += back / 4;
//     }
//
//     if (col < columns - 1) {
//         if (accessMat(thread_assign, row - 1, col + 1) == thread_num)
//             accessMat(flow, row - 1, col + 1) += back / 4;
//     } else {
//         if (accessMat(thread_assign, row - 1, col) == thread_num)
//             accessMat(flow, row - 1, col) += back / 4;
//     }
//
//     // if (accessMat(thread_assign, row, col) == thread_num) {
//     //     int back = particles_m_back[particle];
//     //     accessMat(flow, row, col) -= back;
//     //     accessMat(flow, row - 1, col) += back / 2;
//     //     if (col > 0)
//     //         accessMat(flow, row - 1, col - 1) += back / 4;
//     //     else
//     //         accessMat(flow, row - 1, col) += back / 4;
//     //     if (col < columns - 1)
//     //         accessMat(flow, row - 1, col + 1) += back / 4;
//     //     else
//     //         accessMat(flow, row - 1, col) += back / 4;
//     // }
// }

//     int num_threads;
// #pragma omp parallel default(none) shared(num_threads)
//     {
// #pragma omp master
//         {
//             num_threads = omp_get_num_threads();
//         }
//     }
//
//     int *thread_assign = calloc(rows * columns, sizeof(int));
//     {
//         int disp = 0, size = (rows * columns) / num_threads,
//             rest = (rows * columns) % num_threads;
//
//         for (int thread_num = 0; thread_num < num_threads; thread_num++) {
//             int count = size + (thread_num < rest);
//             for (int i = disp; i < disp + count; i++)
//                 thread_assign[i] = thread_num;
//             disp += count;
//         }
//     }
