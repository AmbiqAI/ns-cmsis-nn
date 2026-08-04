/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated golden data (numpy reference).
 */
#pragma once

#define SLICE_F16_INPUT_N 1
#define SLICE_F16_INPUT_H 4
#define SLICE_F16_INPUT_W 4
#define SLICE_F16_INPUT_C 2
#define SLICE_F16_BEGIN_N 0
#define SLICE_F16_BEGIN_H 0
#define SLICE_F16_BEGIN_W 0
#define SLICE_F16_BEGIN_C 0
#define SLICE_F16_STRIDES_N 1
#define SLICE_F16_STRIDES_H 2
#define SLICE_F16_STRIDES_W 2
#define SLICE_F16_STRIDES_C 1
#define SLICE_F16_OUTPUT_N 1
#define SLICE_F16_OUTPUT_H 2
#define SLICE_F16_OUTPUT_W 2
#define SLICE_F16_OUTPUT_C 2
#define SLICE_F16_OUTPUT_SIZE 8

static const float16_t slice_f16_input[] = {
    -0.48657227f, 1.6347656f,  -1.5888672f, 0.26855469f,  -0.22155762f, 1.0869141f, 0.44921875f,  5.1015625f,
    2.9980469f,   2.9941406f,  -4.078125f,  -0.68066406f, -1.2167969f,  1.0654297f, -4.5585938f,  2.3496094f,
    2.1347656f,   -2.6035156f, -1.9570312f, -1.6025391f,  0.086608887f, 1.2822266f, 4.0976562f,   -0.39477539f,
    1.5351562f,   0.31079102f, 3.5195312f,  1.484375f,    2.7363281f,   -2.15625f,  -0.38452148f, -1.6279297f};
static const float16_t slice_f16_output_ref[] =
    {-0.48657227f, 1.6347656f, -0.22155762f, 1.0869141f, 2.1347656f, -2.6035156f, 0.086608887f, 1.2822266f};
