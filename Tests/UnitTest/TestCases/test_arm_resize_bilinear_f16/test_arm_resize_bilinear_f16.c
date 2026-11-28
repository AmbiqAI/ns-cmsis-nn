/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "unity.h"
static const uint16_t resize_special_bits[] = {0x7e53, 0x8000, 1};
#define RESIZE_MAX 65504.0f
#define RESIZE_TYPE float16_t
#define RESIZE_ENTRY arm_resize_bilinear_f16
// Allow half-weight and accumulation rounding relative to the largest contributing input.
#define RESIZE_TOLERANCE(scale) (0.0025f * (scale))
#include "../Utils/resize_bilinear_flt_tests.h"

static void resize_bilinear_finite_extrema(void)
{
    const cmsis_nn_dims in = {1, 2, 2, 9}, out = {1, 7, 7, 9};
    const cmsis_nn_resize_params params = {false, false};
    float16_t input[2 * 2 * 9], output[7 * 7 * 9];
    const float extrema[] = {-65504.0f, 65504.0f};
    for (int i = 0; i < 2 * 2 * 9; ++i)
        input[i] = (float16_t)extrema[(i % 9) % 2];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RESIZE_ENTRY(&params, &in, input, &out, output));
    for (int i = 0; i < 7 * 7 * 9; ++i)
    {
        uint16_t bits;
        memcpy(&bits, &output[i], sizeof(bits));
        TEST_ASSERT_TRUE((bits & 0x7fffu) <= 0x7bffu);
        TEST_ASSERT_FLOAT_WITHIN(96.0f, extrema[(i % 9) % 2], (float)output[i]);
    }

    // The center combines four equally weighted, alternating-sign maxima.
    const cmsis_nn_dims centered = {1, 3, 3, 9};
    const cmsis_nn_resize_params aligned = {true, false};
    for (int i = 0; i < 2 * 2 * 9; ++i)
        input[i] = (float16_t)extrema[((i / 9) + (i % 9)) % 2];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RESIZE_ENTRY(&aligned, &in, input, &centered, output));
    for (int c = 0; c < 9; ++c)
        TEST_ASSERT_EQUAL_FLOAT(0.0f, (float)output[(1 * 3 + 1) * 9 + c]);
}
