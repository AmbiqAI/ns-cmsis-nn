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

const int8_t transpose_conv_7_weights[60] = {
    -111, 127, -113, 27, 14, 118, -3, 72, -28, 35, -27, -49, 113, 28, -81, -17, 121, -45, -82, -125,
    -39, -91, -92, 4, 45, -2, -122, 124, -77, 115, -85, 107, 81, 96, -70, -53, 10, -64, -111, -85,
    -14, 64, -32, -81, -16, 117, 31, 64, 70, -65, -97, -114, 21, 67, 6, -20, 43, 124, -4, 24};
