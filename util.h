#ifndef UTIL_H_
#define UTIL_H_

/* 2D Vector */
typedef struct {
    int row, col;
} vec2_t;

/* Used to locate and move a MOVING PARTICLE */
typedef struct {
    vec2_t pos, speed;
} particle_m_t;

/* Calculate displs and counts for each process/thread given a number of items and the size of
 * the communicator/team */
void distribute(int items_count, int pool_size, int *counts, int *displs) { // NOLINT
    int displ = 0, items_per_entity = items_count / pool_size,
        rest_items = items_count % pool_size;

    for (int entity = 0; entity < pool_size; entity++) {
        displs[entity] = displ;
        counts[entity] = items_per_entity + (entity < rest_items ? 1 : 0);
        displ += counts[entity];
    }
}

#endif
