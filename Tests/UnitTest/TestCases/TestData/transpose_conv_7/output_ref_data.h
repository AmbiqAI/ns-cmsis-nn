// Hand-authored (not TensorFlow-generated -- no tensorflow available in
// this environment, unlike transpose_conv_1..4). Golden output computed by
// an independent pure-Python int32-accumulate + per-channel-requantize
// reference implementing the documented arm_transpose_conv_s8 contract,
// NOT derived from arm_transpose_conv_s8.c's row-flush scheduling under
// test. Regression pin for ns-cmsis-nn issue #261:
// pad_h (2) > input_h * stride_h (1) -- a tall kernel on a 1-row input, reachable
// from plain TFLite SAME padding -- where the leftover-row loop emitted rows from
// the wrong rolling-buffer slot.
#pragma once
#include <stdint.h>

const int8_t transpose_conv_7_output_ref[36] = {
    -43, -16, 14, -36, -27, 127, -49, -31, 30, -1, -19, 68, -74, -19, -41, 59, -7, -11, -94, -21,
    -30, -22, -9, 11, -38, -9, -75, 127, -1, -44, 3, -11, -93, 46, -7, -17};
