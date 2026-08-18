// Hand-authored (not TensorFlow-generated -- no tensorflow available in
// this environment, unlike transpose_conv_1..4). Golden output computed by
// an independent pure-Python int32-accumulate + per-channel-requantize
// reference implementing the documented arm_transpose_conv_s8 contract,
// NOT derived from arm_transpose_conv_s8.c's row-flush scheduling under
// test. Regression pin for ns-cmsis-nn issue #261:
// pad_h (2) > input_h * stride_h (1) -- a tall kernel on a 1-row input -- where the
// leftover-row loop emitted rows from the wrong rolling-buffer slot.
//
// Provenance: the defect CLASS is TFLite-reachable. TFLite SAME padding with
// input_h = 1, stride_h = 1, filter_h = 5 gives output_h = 1 and
// pad_h = ((input_h - 1) * stride_h + filter_h - output_h) / 2 = 2, i.e. pad_h = 2 > 1.
// This case is a hand-widened variant of that shape: output_h is opened up from 1 to 3
// (and in_ch/out_ch/input_w from 1) so the one-row slip shows across several rows and
// channels instead of a single value. The widened (pad_h = 2, output_h = 3) pair is NOT
// itself something a TFLite converter emits -- ComputePadding for output_h = 3 yields
// pad_h = 1 -- so treat these exact numbers as a hand-authored probe of the class, not
// as a captured converter output.
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
