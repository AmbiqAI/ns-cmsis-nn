/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated golden data (numpy reference, float32 accumulation).
 */
#pragma once

#define MEAN_F16_INPUT_SIZE 30

static const float16_t mean_f16_input[MEAN_F16_INPUT_SIZE] = {
    -3.5f, 1.25f, 2.0f,  4.5f,  -0.75f, 6.0f,  -2.5f, 0.5f,  3.25f, 1.75f,
    -1.0f, 5.5f,  -4.0f, 2.25f, 7.0f,   0.25f, -6.5f, 3.5f,  1.0f,  -2.0f,
    8.0f,  0.75f, -1.5f, 4.0f,  2.5f,   -5.0f, 2.0f,  6.5f,  -3.0f, 1.5f};

static const float16_t mean_f16_ref_c[] = {0.7001953125f, 1.7998046875f, 1.9501953125f,
                                           -0.75f,       2.75f,         0.39990234375f};
static const float16_t mean_f16_ref_h[] = {0.5f,
                                           1.4169921875f,
                                           -0.5f,
                                           3.333984375f,
                                           2.666015625f,
                                           1.0830078125f,
                                           -1.25f,
                                           2.833984375f,
                                           0.66650390625f,
                                           0.66650390625f};
static const float16_t mean_f16_ref_hc[] = {1.4833984375f, 0.7998046875f};
static const float16_t mean_f16_ref_all[] = {1.1416015625f};
