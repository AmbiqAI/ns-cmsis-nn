// Hand-crafted test data for arm_select_v2_s8 (with broadcast)
#pragma once
#include <stdint.h>

// condition: shape [3] = [1, 0, 1]  (broadcast over dim 0)
// x: shape [2, 3] = [[10, 20, 30], [40, 50, 60]]
// y: shape [2, 3] = [[-1, -2, -3], [-4, -5, -6]]
// output: shape [2, 3]
// cond broadcasts: cond_strides=[0, 1], x_strides=[3, 1], y_strides=[3, 1]
// output[i][j] = cond[j] ? x[i][j] : y[i][j]
// = [[10, -2, 30], [40, -5, 60]]

#define SELECT_V2_BROADCAST_S8_RANK 2
#define SELECT_V2_BROADCAST_S8_OUTPUT_SIZE 6

static const int32_t select_v2_broadcast_s8_output_shape[2] = {2, 3};
static const int32_t select_v2_broadcast_s8_cond_strides[2] = {0, 1};
static const int32_t select_v2_broadcast_s8_x_strides[2] = {3, 1};
static const int32_t select_v2_broadcast_s8_y_strides[2] = {3, 1};

static const int8_t select_v2_broadcast_s8_condition[3] = {1, 0, 1};
static const int8_t select_v2_broadcast_s8_x[6] = {10, 20, 30, 40, 50, 60};
static const int8_t select_v2_broadcast_s8_y[6] = {-1, -2, -3, -4, -5, -6};
static const int8_t select_v2_broadcast_s8_output[6] = {10, -2, 30, 40, -5, 60};
