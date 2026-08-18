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

const int8_t transpose_conv_6_input[36] = {
    -36, 96, -58, -80, -101, 30, -63, -42, -113, 6, 60, 61, -125, 62, 114, 77, 44, 34, 100, 88,
    -36, 28, 75, -92, 91, 23, 53, -112, -34, -87, -53, -58, 18, 88, 91, 81};
