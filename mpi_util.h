#ifndef MPI_UTIL_H_
#define MPI_UTIL_H_

#include "util.h"
#include <mpi.h>

/* Creates a MPI_Datatype for VEC2_T at the specified address */
void MPI_Vec2_t(MPI_Datatype *MPI_Vec2_t) { // NOLINT
    MPI_Aint addr[2], vec2_t_displacements[2] = {0};
    int vec2_t_block_lenghts[] = {1, 1};
    MPI_Datatype vec2_t_types[] = {MPI_INT, MPI_INT};
    vec2_t v = {};

    MPI_Get_address(&v.row, addr);
    MPI_Get_address(&v.col, addr + 1);
    vec2_t_displacements[1] = addr[1] - addr[0];

    MPI_Type_create_struct(
        2, vec2_t_block_lenghts, vec2_t_displacements, vec2_t_types, MPI_Vec2_t
    );

    MPI_Type_commit(MPI_Vec2_t);
}

#endif
