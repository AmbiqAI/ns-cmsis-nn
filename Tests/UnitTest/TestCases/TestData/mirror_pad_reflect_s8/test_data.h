// Hand-crafted test data for arm_mirror_pad_s8 (REFLECT mode)
#pragma once
#include <stdint.h>

// Input: shape [3], pad_before [2], mode=REFLECT
// Output: shape [7] (pad_before=2, pad_after=2)
// Input:  [1, 2, 3]
// REFLECT pad: [3, 2, 1, 2, 3, 2, 1]

#define MIRROR_PAD_REFLECT_S8_RANK 1
#define MIRROR_PAD_REFLECT_S8_MODE 0
#define MIRROR_PAD_REFLECT_S8_INPUT_SIZE 3
#define MIRROR_PAD_REFLECT_S8_OUTPUT_SIZE 7

static const int32_t mirror_pad_reflect_s8_input_shape[1] = {3};
static const int32_t mirror_pad_reflect_s8_output_shape[1] = {7};
static const int32_t mirror_pad_reflect_s8_pad_before[1] = {2};

static const int8_t mirror_pad_reflect_s8_input[3] = {1, 2, 3};
static const int8_t mirror_pad_reflect_s8_output[7] = {3, 2, 1, 2, 3, 2, 1};
