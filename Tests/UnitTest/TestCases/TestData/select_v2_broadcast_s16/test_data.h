// Hand-crafted test data for arm_select_v2_s16 (with broadcast)
#pragma once
#include <stdint.h>

// condition: shape [3] = [1, 0, 1]  (broadcast over dim 0)
// x: shape [2, 3] = [[100, 200, 300], [400, 500, 600]]
// y: shape [2, 3] = [[-10, -20, -30], [-40, -50, -60]]
// output: shape [2, 3]
// output[i][j] = cond[j] ? x[i][j] : y[i][j]
// = [[100, -20, 300], [400, -50, 600]]

#define SELECT_V2_BROADCAST_S16_RANK 2
#define SELECT_V2_BROADCAST_S16_OUTPUT_SIZE 6

static const int32_t select_v2_broadcast_s16_output_shape[2] = {2, 3};
static const int32_t select_v2_broadcast_s16_cond_strides[2] = {0, 1};
static const int32_t select_v2_broadcast_s16_x_strides[2] = {3, 1};
static const int32_t select_v2_broadcast_s16_y_strides[2] = {3, 1};

static const int8_t select_v2_broadcast_s16_condition[3] = {1, 0, 1};
static const int16_t select_v2_broadcast_s16_x[6] = {100, 200, 300, 400, 500, 600};
static const int16_t select_v2_broadcast_s16_y[6] = {-10, -20, -30, -40, -50, -60};
static const int16_t select_v2_broadcast_s16_output[6] = {100, -20, 300, 400, -50, 600};
