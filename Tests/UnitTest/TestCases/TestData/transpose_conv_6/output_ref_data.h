// Hand-authored (not TensorFlow-generated -- no tensorflow available in
// this environment, unlike transpose_conv_1..4). Golden output computed by
// an independent pure-Python int32-accumulate + per-channel-requantize
// reference implementing the documented arm_transpose_conv_s8 contract,
// NOT derived from arm_transpose_conv_s8.c's row-flush scheduling under
// test. Regression pin for ns-cmsis-nn issue #261:
// filter_h (1) < stride_h (2) under TFLite VALID padding, where the main-loop
// flush of MAX(0, input_h * stride_h - pad_h) rows runs past output_h and the
// leftover loop (filter_h - stride_h < 0 iterations) cannot compensate.
#pragma once
#include <stdint.h>

const int8_t transpose_conv_6_output_ref[60] = {
    0, 52, 45, 28, -22, 29, 3, 24, 3, 24, 3, 24, 105, 72, -128, 9, 19, 22, 3, 24,
    3, 24, 3, 24, 48, 29, -89, -2, 47, 26, -88, 17, -14, 43, -57, 15, 3, 24, 3, 24,
    3, 24, -60, -3, 127, 81, -118, 17, 3, 24, 3, 24, 3, 24, 86, 37, -125, -7, 73, 24};
