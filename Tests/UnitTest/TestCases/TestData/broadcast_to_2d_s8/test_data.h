// Hand-crafted test data for arm_broadcast_to_s8
#pragma once
#include <stdint.h>

// Input: shape [1, 3], Output: shape [2, 3]
// Input:  [[10, 20, 30]]
// Output: [[10, 20, 30], [10, 20, 30]]

#define BROADCAST_TO_2D_S8_RANK 2
#define BROADCAST_TO_2D_S8_INPUT_SIZE 3
#define BROADCAST_TO_2D_S8_OUTPUT_SIZE 6

static const int32_t broadcast_to_2d_s8_input_shape[2] = {1, 3};
static const int32_t broadcast_to_2d_s8_output_shape[2] = {2, 3};

static const int8_t broadcast_to_2d_s8_input[3] = {10, 20, 30};
static const int8_t broadcast_to_2d_s8_output[6] = {10, 20, 30, 10, 20, 30};
