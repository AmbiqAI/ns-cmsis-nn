// Hand-crafted test data for arm_tile_s16
#pragma once
#include <stdint.h>

// Input: shape [2, 3], multiples [1, 2]
// Output: shape [2, 6]
// Input:  [[100, 200, 300], [400, 500, 600]]
// Output: [[100, 200, 300, 100, 200, 300], [400, 500, 600, 400, 500, 600]]

#define TILE_2D_S16_RANK 2
#define TILE_2D_S16_INPUT_SIZE 6
#define TILE_2D_S16_OUTPUT_SIZE 12

static const int32_t tile_2d_s16_input_shape[2] = {2, 3};
static const int32_t tile_2d_s16_multiples[2] = {1, 2};

static const int16_t tile_2d_s16_input[6] = {100, 200, 300, 400, 500, 600};
static const int16_t tile_2d_s16_output[12] = {100, 200, 300, 100, 200, 300, 400, 500, 600, 400, 500, 600};
