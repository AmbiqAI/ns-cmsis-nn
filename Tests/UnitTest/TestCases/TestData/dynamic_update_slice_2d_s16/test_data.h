// Hand-crafted test data for arm_dynamic_update_slice_s16
#pragma once
#include <stdint.h>

// Operand: shape [3, 4], Update: shape [2, 2], start_indices: [1, 1]
// Operand: [[100, 200, 300, 400], [500, 600, 700, 800], [900, 1000, 1100, 1200]]
// Update:  [[2000, 2100], [2200, 2300]]
// After slice update at [1,1]:
// Output:  [[100, 200, 300, 400], [500, 2000, 2100, 800], [900, 2200, 2300, 1200]]

#define DYNAMIC_UPDATE_SLICE_2D_S16_RANK 2
#define DYNAMIC_UPDATE_SLICE_2D_S16_OPERAND_SIZE 12
#define DYNAMIC_UPDATE_SLICE_2D_S16_UPDATE_SIZE 4

static const int32_t dynamic_update_slice_2d_s16_operand_shape[2] = {3, 4};
static const int32_t dynamic_update_slice_2d_s16_update_shape[2] = {2, 2};
static const int32_t dynamic_update_slice_2d_s16_operand_strides[2] = {4, 1};
static const int32_t dynamic_update_slice_2d_s16_start_indices[2] = {1, 1};

static const int16_t dynamic_update_slice_2d_s16_operand[12] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200};
static const int16_t dynamic_update_slice_2d_s16_update[4] = {2000, 2100, 2200, 2300};
static const int16_t dynamic_update_slice_2d_s16_output[12] = {100, 200, 300, 400, 500, 2000, 2100, 800, 900, 2200, 2300, 1200};
