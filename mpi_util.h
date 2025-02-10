#ifndef MPI_UTIL_H_
#define MPI_UTIL_H_

#include "util.h"
#include <mpi.h>

void MPI_Vec2_t(MPI_Datatype *MPI_Vec2_t) { // NOLINT
    MPI_Aint addr[2],
        vec2_t_displacements[2] = {0};
    int vec2_t_block_lenghts[] = {1, 1};
    MPI_Datatype vec2_t_types[] = {MPI_INT, MPI_INT};
    vec2_t v = {};

    MPI_Get_address(&v.row, addr);
    MPI_Get_address(&v.col, addr + 1);
    vec2_t_displacements[1] = addr[1] - addr[0];

    MPI_Type_create_struct(
        2,
        vec2_t_block_lenghts,
        vec2_t_displacements,
        vec2_t_types,
        MPI_Vec2_t
    );
    MPI_Type_commit(MPI_Vec2_t);
}

#endif

// MPI_Aint addr[8], particle_t_displs[8] = {0};
// int particle_t_block_lenghts[] = {1, 1, 1, 1, 1, 1, 1, 1};
// MPI_Datatype particle_t_types[]
//     = {MPI_UNSIGNED_CHAR, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
//     MPI_INT, MPI_INT
//     };
// particle_t t = {0};
// MPI_Get_address(&t.extra, addr);
// MPI_Get_address(&t.pos_row, addr + 1);
// MPI_Get_address(&t.pos_col, addr + 2);
// MPI_Get_address(&t.mass, addr + 3);
// MPI_Get_address(&t.resistance, addr + 4);
// MPI_Get_address(&t.speed_row, addr + 5);
// MPI_Get_address(&t.speed_col, addr + 6);
// MPI_Get_address(&t.old_flow, addr + 7);
// for (int i = 1; i < 8; i++)
//     particles_displs[i] = addr[i] - addr[0];
// MPI_Type_create_struct(
//     8, particle_t_block_lenghts, particle_t_displs, particle_t_types,
//     &MPI_PARTICLE
// );
// MPI_Type_commit(&MPI_PARTICLE);

// {
//     MPI_Aint addr[8], particle_t_displs[8] = {0};
//     int particle_t_block_lenghts[] = {1, 1, 1, 1, 1, 1, 1, 1};
//     MPI_Datatype particle_t_types[]
//         = {MPI_UNSIGNED_CHAR, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT
//         };
//     particle_t t = {0};
//
//     MPI_Get_address(&t.extra, addr);
//     MPI_Get_address(&t.pos_row, addr + 1);
//     MPI_Get_address(&t.pos_col, addr + 2);
//     MPI_Get_address(&t.mass, addr + 3);
//     MPI_Get_address(&t.resistance, addr + 4);
//     MPI_Get_address(&t.speed_row, addr + 5);
//     MPI_Get_address(&t.speed_col, addr + 6);
//     MPI_Get_address(&t.old_flow, addr + 7);
//
//     for (int i = 1; i < 8; i++)
//         particles_displs[i] = addr[i] - addr[0];
//
//     MPI_Type_create_struct(
//         8, particle_t_block_lenghts, particle_t_displs, particle_t_types, &MPI_PARTICLE
//     );
//     MPI_Type_commit(&MPI_PARTICLE);
// }
