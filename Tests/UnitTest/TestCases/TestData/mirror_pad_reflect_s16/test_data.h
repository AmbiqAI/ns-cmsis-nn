// Hand-crafted test data for arm_mirror_pad_s16 (REFLECT mode)
#pragma once
#include <stdint.h>

// Input: shape [3], pad_before [2], mode=REFLECT
// Output: shape [7] (pad_before=2, pad_after=2)
// Input:  [100, 200, 300]
// REFLECT pad: [300, 200, 100, 200, 300, 200, 100]

#define MIRROR_PAD_REFLECT_S16_RANK 1
#define MIRROR_PAD_REFLECT_S16_MODE 0
#define MIRROR_PAD_REFLECT_S16_INPUT_SIZE 3
#define MIRROR_PAD_REFLECT_S16_OUTPUT_SIZE 7

static const int32_t mirror_pad_reflect_s16_input_shape[1] = {3};
static const int32_t mirror_pad_reflect_s16_output_shape[1] = {7};
static const int32_t mirror_pad_reflect_s16_pad_before[1] = {2};

static const int16_t mirror_pad_reflect_s16_input[3] = {100, 200, 300};
static const int16_t mirror_pad_reflect_s16_output[7] = {300, 200, 100, 200, 300, 200, 100};
