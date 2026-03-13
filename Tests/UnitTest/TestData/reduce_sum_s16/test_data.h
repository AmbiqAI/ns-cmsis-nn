#pragma once

#include <stdint.h>

const cmsis_nn_dims reduce_sum_s16_in_dim = {4, 2, 3, 2};
const cmsis_nn_dims reduce_sum_s16_axis_dim = {1, 0, 0, 0};
const cmsis_nn_dims reduce_sum_s16_out_dim = {4, 2, 3, 1};


const int16_t reduce_sum_s16_input_tensor[] = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

const int16_t reduce_sum_s16_output[] = {
    3,  7,  11, 15, 19, 23,
    27, 31, 35, 39, 43, 47};

const int32_t reduce_sum_s16_input_offset = 0;
const int32_t reduce_sum_s16_output_offset = 0;
const int32_t reduce_sum_s16_output_multiplier = 1;
const int32_t reduce_sum_s16_output_shift = 0;

#define REDUCE_SUM_S16_OUTPUT_SIZE (sizeof(reduce_sum_s16_output) / sizeof(reduce_sum_s16_output[0]))
