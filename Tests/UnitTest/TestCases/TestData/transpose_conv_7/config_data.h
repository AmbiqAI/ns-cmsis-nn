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
#define TRANSPOSE_CONV_7_OUT_CH 3
#define TRANSPOSE_CONV_7_IN_CH 2
#define TRANSPOSE_CONV_7_INPUT_W 3
#define TRANSPOSE_CONV_7_INPUT_H 1
#define TRANSPOSE_CONV_7_DST_SIZE 36
#define TRANSPOSE_CONV_7_INPUT_SIZE 6
#define TRANSPOSE_CONV_7_OUT_ACTIVATION_MIN -128
#define TRANSPOSE_CONV_7_OUT_ACTIVATION_MAX 127
#define TRANSPOSE_CONV_7_INPUT_BATCHES 1
#define TRANSPOSE_CONV_7_FILTER_X 2
#define TRANSPOSE_CONV_7_FILTER_Y 5
#define TRANSPOSE_CONV_7_STRIDE_X 1
#define TRANSPOSE_CONV_7_STRIDE_Y 1
#define TRANSPOSE_CONV_7_PAD_X 0
#define TRANSPOSE_CONV_7_PAD_Y 2
#define TRANSPOSE_CONV_7_OUTPUT_W 4
#define TRANSPOSE_CONV_7_OUTPUT_H 3
#define TRANSPOSE_CONV_7_INPUT_OFFSET 7
#define TRANSPOSE_CONV_7_OUTPUT_OFFSET -11
#define TRANSPOSE_CONV_7_DILATION_X 1
#define TRANSPOSE_CONV_7_DILATION_Y 1
#define TRANSPOSE_CONV_7_PAD_X_WITH_OFFSET 0
#define TRANSPOSE_CONV_7_PAD_Y_WITH_OFFSET 0
