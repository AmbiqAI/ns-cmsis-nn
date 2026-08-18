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
#define TRANSPOSE_CONV_6_OUT_CH 2
#define TRANSPOSE_CONV_6_IN_CH 3
#define TRANSPOSE_CONV_6_INPUT_W 2
#define TRANSPOSE_CONV_6_INPUT_H 3
#define TRANSPOSE_CONV_6_DST_SIZE 60
#define TRANSPOSE_CONV_6_INPUT_SIZE 36
#define TRANSPOSE_CONV_6_OUT_ACTIVATION_MIN -128
#define TRANSPOSE_CONV_6_OUT_ACTIVATION_MAX 127
#define TRANSPOSE_CONV_6_INPUT_BATCHES 2
#define TRANSPOSE_CONV_6_FILTER_X 2
#define TRANSPOSE_CONV_6_FILTER_Y 1
#define TRANSPOSE_CONV_6_STRIDE_X 1
#define TRANSPOSE_CONV_6_STRIDE_Y 2
#define TRANSPOSE_CONV_6_PAD_X 0
#define TRANSPOSE_CONV_6_PAD_Y 0
#define TRANSPOSE_CONV_6_OUTPUT_W 3
#define TRANSPOSE_CONV_6_OUTPUT_H 5
#define TRANSPOSE_CONV_6_INPUT_OFFSET -13
#define TRANSPOSE_CONV_6_OUTPUT_OFFSET 9
#define TRANSPOSE_CONV_6_DILATION_X 1
#define TRANSPOSE_CONV_6_DILATION_Y 1
#define TRANSPOSE_CONV_6_PAD_X_WITH_OFFSET 0
#define TRANSPOSE_CONV_6_PAD_Y_WITH_OFFSET 0
