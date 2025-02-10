#ifndef UTIL_H_
#define UTIL_H_

typedef struct {
    int row, col;
} vec2_t;

typedef struct {
    vec2_t pos, speed;
} particle_m_t;

typedef struct {
    int displ, count;
} sect_t;

void distribute(int items_count, int comm_size, int *counts, int *displs) { // NOLINT
    int displ = 0,
        items_per_rank = items_count / comm_size,
        remaining_items = items_count % comm_size;

    for (int rank = 0; rank < comm_size; rank++) {
        displs[rank] = displ;
        counts[rank] = items_per_rank + (rank < remaining_items ? 1 : 0);
        displ += counts[rank];
    }
}

#endif

// int particle_m_cmp(const void *p1, const void *p2) {
//     particle_m_t *p_1 = (particle_m_t *)p1, *p_2 = (particle_m_t *)p2;
//
//     if (p_1->pos.row < p_2->pos.row)
//         return -1;
//     if (p_1->pos.row > p_2->pos.row)
//         return 1;
//     if (p_1->pos.col < p_2->pos.col)
//         return -1;
//     if (p_1->pos.col > p_2->pos.col)
//         return 1;
//
//     return 0;
// }

/*int mass;*/
/*typedef struct {*/
/*    vec2_t pos;*/
/*} base_particle_t;*/

/*int resistance,*/
/*    old_flow;*/

/*base_particle_t base;*/

/*int base_particle_cmp(const void *p1, const void *p2) {*/
/*    base_particle_t *p_1 = (base_particle_t *)p1,*/
/*                    *p_2 = (base_particle_t *)p2;*/
/**/
/*    if (p_1->pos.row < p_2->pos.row)*/
/*        return -1;*/
/*    if (p_1->pos.row > p_2->pos.row)*/
/*        return 1;*/
/*    if (p_1->pos.col < p_2->pos.col)*/
/*        return -1;*/
/*    if (p_1->pos.col > p_2->pos.col)*/
/*        return 1;*/
/**/
/*    return 0;*/
/*}*/

/*int moving_particle_cmp(const void *p1, const void *p2) {*/
/*    particle_m_t *p_1 = (particle_m_t *)p1,*/
/*                 *p_2 = (particle_m_t *)p2;*/
/**/
/*    if (p_1->pos.row < p_2->pos.row)*/
/*        return -1;*/
/*    if (p_1->pos.row > p_2->pos.row)*/
/*        return 1;*/
/*    if (p_1->pos.col < p_2->pos.col)*/
/*        return -1;*/
/*    if (p_1->pos.col > p_2->pos.col)*/
/*        return 1;*/
/**/
/*    return 0;*/
/*}*/
/*typedef struct {*/
/*    int displ, count;*/
/*} sect_t;*/

// sects[id] = (sect_t){.displ = offset, .count = rank_vals + (id < left_vals)};
// offset += sects[id].count;

// sect_t sector(int vals, int rank, int size) { // NOLINT
//     int offset = 0, rank_items = vals / size, left_items = vals % size, displ = 0, count = 0;
//
//     for (int thread = 0; thread <= rank; thread++) {
//         displ = offset;
//         count = rank_items + (thread < left_items);
//         offset += count;
//     }
//
//     return (sect_t){displ, count};
// }
