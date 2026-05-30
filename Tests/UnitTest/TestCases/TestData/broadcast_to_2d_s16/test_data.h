// Hand-crafted test data for arm_broadcast_to_s16
#pragma once
#include <stdint.h>

// Input: shape [1, 3], Output: shape [2, 3]
// Input:  [[100, 200, 300]]
// Output: [[100, 200, 300], [100, 200, 300]]

#define BROADCAST_TO_2D_S16_RANK 2
#define BROADCAST_TO_2D_S16_INPUT_SIZE 3
#define BROADCAST_TO_2D_S16_OUTPUT_SIZE 6

static const int32_t broadcast_to_2d_s16_input_shape[2] = {1, 3};
static const int32_t broadcast_to_2d_s16_output_shape[2] = {2, 3};

static const int16_t broadcast_to_2d_s16_input[3] = {100, 200, 300};
static const int16_t broadcast_to_2d_s16_output[6] = {100, 200, 300, 100, 200, 300};
