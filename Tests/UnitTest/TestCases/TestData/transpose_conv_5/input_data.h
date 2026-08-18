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
#include <stdint.h>

const int8_t transpose_conv_5_input[128] = {
    -15, 31,  32,   -101, -109, -8,  -95,  110, 72,   -122, -128, -99, -51,  34,   123,  -70, 50,  -91, 6,
    -26, -84, -79,  -2,   -30,  97,  -75,  -39, 48,   -52,  -97,  -47, -123, -38,  -21,  -90, -3,  0,   17,
    73,  -10, 111,  53,   -94,  9,   -80,  104, 34,   -104, 68,   11,  88,   -110, 29,   -6,  94,  99,  -43,
    -12, 24,  62,   45,   127,  -12, -114, 17,  -88,  4,    96,   38,  -47,  40,   86,   81,  -7,  71,  75,
    111, 32,  -119, 30,   -67,  39,  62,   -55, -111, -117, -65,  -7,  -62,  88,   -100, 117, -77, 110, 86,
    107, 41,  -29,  93,   -27,  71,  -23,  69,  122,  80,   103,  10,  -72,  20,   12,   -24, -89, 31,  -49,
    114, 7,   -18,  -88,  -20,  18,  22,   71,  20,   58,   -64,  -55, 94,   -65};
