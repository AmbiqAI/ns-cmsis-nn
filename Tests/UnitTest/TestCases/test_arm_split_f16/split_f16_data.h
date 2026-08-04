/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated golden data (numpy reference).
 */
#pragma once

#define SPLIT_F16_INPUT_DIMS 3
#define SPLIT_F16_AXIS 1
#define SPLIT_F16_NUM_SPLITS 3
#define SPLIT_F16_SPLIT_SIZE 12

static const int32_t split_f16_input_shape[] = {2, 6, 3};
static const int32_t split_f16_split_dims[] = {2, 2, 2};
static const float16_t split_f16_input[] = {
    1.4189453f,  2.3125f,      -4.3164062f,  -0.99609375f,  0.65625f,     -1.21875f,   3.1816406f,     -2.3828125f,
    0.70898438f, -2.0976562f,  2.8125f,      -0.043304443f, -0.74462891f, -3.4355469f, 3.3632812f,     1.5058594f,
    1.5068359f,  2.2753906f,   0.69824219f,  -1.2783203f,   -1.6005859f,  -1.6005859f, 2.7402344f,     -2.9199219f,
    -1.1923828f, -0.64257812f, 0.44921875f,  1.1503906f,    -2.4980469f,  -3.4609375f, -0.0088272095f, 2.4277344f,
    1.5136719f,  0.43139648f,  -0.63427734f, 0.58642578f};
static const float16_t split_f16_output_ref_0[] = {1.4189453f,
                                                   2.3125f,
                                                   -4.3164062f,
                                                   -0.99609375f,
                                                   0.65625f,
                                                   -1.21875f,
                                                   0.69824219f,
                                                   -1.2783203f,
                                                   -1.6005859f,
                                                   -1.6005859f,
                                                   2.7402344f,
                                                   -2.9199219f};
static const float16_t split_f16_output_ref_1[] = {3.1816406f,
                                                   -2.3828125f,
                                                   0.70898438f,
                                                   -2.0976562f,
                                                   2.8125f,
                                                   -0.043304443f,
                                                   -1.1923828f,
                                                   -0.64257812f,
                                                   0.44921875f,
                                                   1.1503906f,
                                                   -2.4980469f,
                                                   -3.4609375f};
static const float16_t split_f16_output_ref_2[] = {-0.74462891f,
                                                   -3.4355469f,
                                                   3.3632812f,
                                                   1.5058594f,
                                                   1.5068359f,
                                                   2.2753906f,
                                                   -0.0088272095f,
                                                   2.4277344f,
                                                   1.5136719f,
                                                   0.43139648f,
                                                   -0.63427734f,
                                                   0.58642578f};
