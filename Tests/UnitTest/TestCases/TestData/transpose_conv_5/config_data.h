// Hand-authored (not TensorFlow-generated -- no tensorflow available in
// this environment, unlike transpose_conv_1..4). Golden output computed by
// an independent pure-Python int32-accumulate + per-channel-requantize
// reference implementing the documented arm_transpose_conv_s8 contract,
// NOT derived from arm_transpose_conv_s8.c's row-flush scheduling under
// test. Mirrors upstream CMSIS-NN issue #230's own repro shape (input
// 4x4x8, filter OHWI 4x4x4x8, stride 2x2, pad 1x1) and deliberately uses
// pad_h % stride_h != 0 (ns-cmsis-nn issue #258): the one axis/remainder
// combination no existing transpose_conv_N / reverse_transpose_conv_N
// case exercises.
#pragma once
#define TRANSPOSE_CONV_5_OUT_CH 4
#define TRANSPOSE_CONV_5_IN_CH 8
#define TRANSPOSE_CONV_5_INPUT_W 4
#define TRANSPOSE_CONV_5_INPUT_H 4
#define TRANSPOSE_CONV_5_DST_SIZE 256
#define TRANSPOSE_CONV_5_INPUT_SIZE 128
#define TRANSPOSE_CONV_5_OUT_ACTIVATION_MIN -128
#define TRANSPOSE_CONV_5_OUT_ACTIVATION_MAX 127
#define TRANSPOSE_CONV_5_INPUT_BATCHES 1
#define TRANSPOSE_CONV_5_FILTER_X 4
#define TRANSPOSE_CONV_5_FILTER_Y 4
#define TRANSPOSE_CONV_5_STRIDE_X 2
#define TRANSPOSE_CONV_5_STRIDE_Y 2
#define TRANSPOSE_CONV_5_PAD_X 1
#define TRANSPOSE_CONV_5_PAD_Y 1
#define TRANSPOSE_CONV_5_OUTPUT_W 8
#define TRANSPOSE_CONV_5_OUTPUT_H 8
#define TRANSPOSE_CONV_5_INPUT_OFFSET -18
#define TRANSPOSE_CONV_5_OUTPUT_OFFSET 17
#define TRANSPOSE_CONV_5_DILATION_X 1
#define TRANSPOSE_CONV_5_DILATION_Y 1
#define TRANSPOSE_CONV_5_PAD_X_WITH_OFFSET 1
#define TRANSPOSE_CONV_5_PAD_Y_WITH_OFFSET 1
