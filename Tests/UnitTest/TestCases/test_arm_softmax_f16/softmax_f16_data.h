/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated golden data, from Tests/UnitTest/softmax_settings_flt.py
 * --dataset softmax_f16: PyTorch softmax computed in float32 and rounded once
 * to float16.
 */
#pragma once

#define SOFTMAX_F16_NUM_ROWS 2
#define SOFTMAX_F16_ROW_SIZE 5
#define SOFTMAX_F16_DST_SIZE 10

static const float16_t softmax_f16_input[] = {
    1.0f, 0.25f, -0.5f, 2.0f, -1.25f, -2.0f, -0.125f, 0.75f, 3.0f, 1.5f};

static const float16_t softmax_f16_output_ref[] = {0.221313477f,
                                                   0.104553223f,
                                                   0.0493774414f,
                                                   0.6015625f,
                                                   0.0233154297f,
                                                   0.0048866272f,
                                                   0.0318603516f,
                                                   0.0764160156f,
                                                   0.725097656f,
                                                   0.161743164f};
