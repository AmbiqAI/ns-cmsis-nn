/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated golden data (numpy reference, float32 evaluation).
 */
#pragma once

#define RSQRT_F16_DST_SIZE 18

static const float16_t rsqrt_f16_input[RSQRT_F16_DST_SIZE] = {
    0x1.1p-20f, 0.0625f, 0.125f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f,
    3.0f,       4.0f,    5.0f,   8.0f,  9.0f, 16.0f, 25.0f, 64.0f, 100.0f};

static const float16_t rsqrt_f16_output_ref[RSQRT_F16_DST_SIZE] = {
    993.5f,        4.0f,          2.828125f,     2.0f,          1.4140625f,   1.154296875f,
    1.0f,          0.81640625f,   0.70703125f,   0.5771484375f, 0.5f,         0.447265625f,
    0.353515625f,  0.333251953125f, 0.25f,       0.199951171875f, 0.125f,     0.0999755859375f};
