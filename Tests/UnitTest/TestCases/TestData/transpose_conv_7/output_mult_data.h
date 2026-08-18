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

const int32_t transpose_conv_7_output_mult[3] = {
    1560497773, 1307215495, 1544936597};
