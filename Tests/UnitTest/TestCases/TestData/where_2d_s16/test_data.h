// Hand-crafted test data for arm_where_s16
#pragma once
#include <stdint.h>

// Input: shape [2, 3], condition values (non-zero = true)
// Condition: [[0, 100, 0], [200, 0, 300]]
// Non-zero at: [0,1], [1,0], [1,2]
// Output coords: [[0,1], [1,0], [1,2]]

#define WHERE_2D_S16_RANK 2
#define WHERE_2D_S16_SIZE 6
#define WHERE_2D_S16_NUM_TRUE 3

static const int32_t where_2d_s16_shape[2] = {2, 3};
static const int16_t where_2d_s16_condition[6] = {0, 100, 0, 200, 0, 300};
static const int32_t where_2d_s16_output[6] = {0, 1, 1, 0, 1, 2};
