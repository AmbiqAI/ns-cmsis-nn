/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 *
 * Generated golden data (numpy reference, float32 accumulation).
 */
#pragma once

#define MEAN_F32_INPUT_SIZE 30

static const float32_t mean_f32_input[MEAN_F32_INPUT_SIZE] = {
    -3.5f, 1.25f, 2.0f,  4.5f,  -0.75f, 6.0f,  -2.5f, 0.5f,  3.25f, 1.75f,
    -1.0f, 5.5f,  -4.0f, 2.25f, 7.0f,   0.25f, -6.5f, 3.5f,  1.0f,  -2.0f,
    8.0f,  0.75f, -1.5f, 4.0f,  2.5f,   -5.0f, 2.0f,  6.5f,  -3.0f, 1.5f};

static const float32_t mean_f32_ref_c[] = {0.699999988079071f, 1.7999999523162842f, 1.9500000476837158f,
                                           -0.75f,             2.75f,               0.4000000059604645f};
static const float32_t mean_f32_ref_h[] = {0.5f,
                                           1.4166666269302368f,
                                           -0.5f,
                                           3.3333332538604736f,
                                           2.6666667461395264f,
                                           1.0833333730697632f,
                                           -1.25f,
                                           2.8333332538604736f,
                                           0.6666666865348816f,
                                           0.6666666865348816f};
static const float32_t mean_f32_ref_hc[] = {1.4833333492279053f, 0.800000011920929f};
static const float32_t mean_f32_ref_all[] = {1.1416666507720947f};
