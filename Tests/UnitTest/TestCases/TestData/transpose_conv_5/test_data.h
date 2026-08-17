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
#include "biases_data.h"
#include "config_data.h"
#include "input_data.h"
#include "output_mult_data.h"
#include "output_ref_data.h"
#include "output_shift_data.h"
#include "weights_data.h"
