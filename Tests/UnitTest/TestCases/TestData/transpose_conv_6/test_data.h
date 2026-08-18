// Hand-authored (not TensorFlow-generated -- no tensorflow available in
// this environment, unlike transpose_conv_1..4). Golden output computed by
// an independent pure-Python int32-accumulate + per-channel-requantize
// reference implementing the documented arm_transpose_conv_s8 contract,
// NOT derived from arm_transpose_conv_s8.c's row-flush scheduling under
// test. Regression pin for ns-cmsis-nn issue #261:
// filter_h (1) < stride_h (2) under TFLite VALID padding, where the main-loop
// flush of MAX(0, input_h * stride_h - pad_h) rows runs past output_h and the
// leftover loop (filter_h - stride_h < 0 iterations) cannot compensate.
#include "biases_data.h"
#include "config_data.h"
#include "input_data.h"
#include "output_mult_data.h"
#include "output_ref_data.h"
#include "output_shift_data.h"
#include "weights_data.h"
