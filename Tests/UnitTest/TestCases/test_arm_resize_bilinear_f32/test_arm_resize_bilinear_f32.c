/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "unity.h"
static const uint32_t resize_special_bits[] = {0x7fc12345, 0x80000000, 1};
#define RESIZE_TYPE float32_t
#define RESIZE_ENTRY arm_resize_bilinear_f32
#define RESIZE_TOLERANCE(scale) 1e-05f
#include "../Utils/resize_bilinear_flt_tests.h"

static void resize_bilinear_fractional_coordinate(void)
{
    // LiteRT BUILTIN_REF evaluates the fractional coordinate in FP32 at x = 195.
    const cmsis_nn_dims in = {1, 1, 192, 1}, out = {1, 1, 257, 1};
    const cmsis_nn_resize_params params = {false, false};
    float input[192], output[257];
    for (int x = 0; x < in.w; ++x)
    {
        input[x] = (x & 1) ? 1.0f : -1.0f;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_resize_bilinear_f32(&params, &in, input, &out, output));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -0.36184692f, output[195]);
}
