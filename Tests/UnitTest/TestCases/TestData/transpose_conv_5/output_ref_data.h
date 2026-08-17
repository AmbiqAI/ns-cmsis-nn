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

const int8_t transpose_conv_5_output_ref[256] = {
    -46,  20,  13, -3,   69,   -59, 12, 127,  14,   -123, 17, -107, 127,  -71,  11, -115, -128, -62, 9,  -128,
    9,    -27, 15, 127,  -118, -76, 12, 127,  43,   -38,  15, -69,  -23,  -3,   19, -70,  -128, 22,  5,  41,
    -94,  76,  31, 81,   -36,  -35, 19, -37,  -128, 19,   21, 127,  -15,  -52,  11, 127,  -128, 65,  21, 5,
    127,  35,  20, -128, 80,   -42, 15, 127,  -128, -112, 8,  127,  127,  -111, 19, -128, 127,  -76, 17, -128,
    -128, 23,  8,  -128, -128, 8,   13, 109,  39,   -110, 14, 38,   127,  -72,  19, 127,  -99,  9,   15, -128,
    124,  13,  12, 127,  -93,  69,  21, -41,  127,  77,   30, 127,  127,  -14,  23, 127,  24,   11,  13, 108,
    15,   -1,  8,  -128, 85,   35,  13, 127,  22,   19,   13, 91,   -31,  -49,  11, 48,   17,   -9,  16, 127,
    -128, 27,  22, 127,  -51,  64,  13, 127,  -30,  58,   12, -34,  96,   -105, 14, 23,   127,  58,  19, -55,
    71,   8,   17, 109,  -25,  -5,  19, 66,   127,  78,   20, 127,  62,   -70,  20, -27,  104,  54,  2,  38,
    65,   -26, 19, -95,  -128, 57,  17, 127,  -83,  13,   11, 127,  -67,  53,   18, -23,  -128, 19,  21, 22,
    49,   -23, 30, 78,   127,  1,   18, 77,   34,   42,   7,  38,   -100, -16,  14, 127,  45,   -66, 14, 127,
    111,  70,  10, 127,  122,  -28, 21, 127,  75,   -42,  15, 43,   -70,  62,   16, -71,  -29,  13,  16, 68,
    127,  -29, 21, 102,  35,   -53, 20, -128, -128, 64,   18, 23,   -8,   32,   14, 5};
