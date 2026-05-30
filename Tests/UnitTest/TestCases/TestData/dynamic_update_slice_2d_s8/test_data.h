// Hand-crafted test data for arm_dynamic_update_slice_s8
#pragma once
#include <stdint.h>

// Operand: shape [3, 4], Update: shape [2, 2], start_indices: [1, 1]
// Operand: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
// Update:  [[20, 21], [22, 23]]
// After slice update at [1,1]:
// Output:  [[1, 2, 3, 4], [5, 20, 21, 8], [9, 22, 23, 12]]

#define DYNAMIC_UPDATE_SLICE_2D_S8_RANK 2
#define DYNAMIC_UPDATE_SLICE_2D_S8_OPERAND_SIZE 12
#define DYNAMIC_UPDATE_SLICE_2D_S8_UPDATE_SIZE 4

static const int32_t dynamic_update_slice_2d_s8_operand_shape[2] = {3, 4};
static const int32_t dynamic_update_slice_2d_s8_update_shape[2] = {2, 2};
static const int32_t dynamic_update_slice_2d_s8_operand_strides[2] = {4, 1};
static const int32_t dynamic_update_slice_2d_s8_start_indices[2] = {1, 1};

static const int8_t dynamic_update_slice_2d_s8_operand[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const int8_t dynamic_update_slice_2d_s8_update[4] = {20, 21, 22, 23};
static const int8_t dynamic_update_slice_2d_s8_output[12] = {1, 2, 3, 4, 5, 20, 21, 8, 9, 22, 23, 12};
