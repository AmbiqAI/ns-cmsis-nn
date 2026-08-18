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
#include <stdint.h>

const int8_t transpose_conv_7_output_ref[36] = {
    -43, -16, 14, -36, -27, 127, -49, -31, 30, -1, -19, 68, -74, -19, -41, 59, -7, -11, -94, -21,
    -30, -22, -9, 11, -38, -9, -75, 127, -1, -44, 3, -11, -93, 46, -7, -17};
