/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#include "sub_f32_data.h"

/*
 * Bit-pattern NaN test. isnan() is not usable here: -ffinite-math-only, which the
 * -Ofast this suite is built with by default implies, licenses the compiler to fold
 * isnan() to a constant false, so it cannot observe a NaN result even on toolchains
 * where the kernel returns one.
 */
static bool sub_f32_bits_are_nan(float32_t x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
}

void sub_f32_arm_elementwise_sub_f32(void)
{
    float32_t output[SUB_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_elementwise_sub_f32(sub_f32_input1,
                                              sub_f32_input2,
                                              output,
                                              SUB_F32_OUT_ACTIVATION_MIN,
                                              SUB_F32_OUT_ACTIVATION_MAX,
                                              SUB_F32_DST_SIZE));

    for (int i = 0; i < SUB_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, sub_f32_output_ref[i], output[i]);
    }
}

void sub_f32_nan_inf_arm_elementwise_sub_f32(void)
{
    // Non-finite inputs are not supported (#333), so this case is a regression pin, not a contract. What
    // comes back for the two NaN-producing lanes is toolchain dependent: some toolchains fold the clamp's
    // NaN check away and return a clamp bound, others keep it and return a real NaN. Those lanes are
    // therefore required only to be a NaN or exactly one of the two bounds - anything else, such as a stray
    // 0.0f or an unclamped Inf, is still a failure. The Inf-overflow lanes are pinned to the bounds.
    const float32_t inf = (float32_t)INFINITY;
    const float32_t in1[4] = {inf, (float32_t)NAN, inf, -inf};
    const float32_t in2[4] = {inf, 0.0f, 1.0f, 1.0f};
    float32_t output[4] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_sub_f32(in1, in2, output, -6.0f, 6.0f, 4));

    for (int32_t i = 0; i < 2; i++)
    {
        const float32_t y = output[i];
        TEST_ASSERT_TRUE(sub_f32_bits_are_nan(y) || y == -6.0f || y == 6.0f);
    }
    TEST_ASSERT_EQUAL_FLOAT(6.0f, output[2]);
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, output[3]);
}

void sub_f32_arg_error_arm_elementwise_sub_f32(void)
{
    float32_t output[SUB_F32_DST_SIZE] = {0};
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_elementwise_sub_f32(
            NULL, sub_f32_input2, output, SUB_F32_OUT_ACTIVATION_MIN, SUB_F32_OUT_ACTIVATION_MAX, SUB_F32_DST_SIZE));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_elementwise_sub_f32(
            sub_f32_input1, sub_f32_input2, output, SUB_F32_OUT_ACTIVATION_MIN, SUB_F32_OUT_ACTIVATION_MAX, 0));
}
