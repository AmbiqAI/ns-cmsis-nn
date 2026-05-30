// Hand-crafted test data for arm_where_s8
#pragma once
#include <stdint.h>

// Condition: shape [2, 3] = [[1, 0, 1], [0, 1, 0]]
// Non-zero positions: (0,0), (0,2), (1,1)
// Output coords (flattened): [0,0, 0,2, 1,1]

#define WHERE_2D_S8_RANK 2
#define WHERE_2D_S8_COND_SIZE 6
#define WHERE_2D_S8_NUM_TRUE 3

static const int32_t where_2d_s8_shape[2] = {2, 3};

static const int8_t where_2d_s8_condition[6] = {1, 0, 1, 0, 1, 0};
static const int32_t where_2d_s8_output[6] = {0, 0, 0, 2, 1, 1};
