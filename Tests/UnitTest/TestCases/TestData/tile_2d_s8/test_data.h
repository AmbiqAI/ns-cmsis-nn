// Hand-crafted test data for arm_tile_s8
#pragma once
#include <stdint.h>

// Input: shape [2, 3], multiples [1, 2]
// Output: shape [2, 6]
// Input:  [[1, 2, 3], [4, 5, 6]]
// Output: [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]]

#define TILE_2D_S8_RANK 2
#define TILE_2D_S8_INPUT_SIZE 6
#define TILE_2D_S8_OUTPUT_SIZE 12

static const int32_t tile_2d_s8_input_shape[2] = {2, 3};
static const int32_t tile_2d_s8_multiples[2] = {1, 2};

static const int8_t tile_2d_s8_input[6] = {1, 2, 3, 4, 5, 6};
static const int8_t tile_2d_s8_output[12] = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
