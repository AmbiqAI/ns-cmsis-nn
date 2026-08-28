/*
 * SPDX-FileCopyrightText: 2026 Ambiq
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

#include "sub_f16_data.h"

/*
 * Bit-pattern NaN test. isnan() is not usable here: -ffinite-math-only, which the
 * -Ofast this suite is built with by default implies, licenses the compiler to fold
 * isnan() to a constant false, so it cannot observe a NaN result even on toolchains
 * where the kernel returns one.
 */
static bool sub_f16_bits_are_nan(float16_t x)
{
    uint16_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return ((bits & 0x7C00u) == 0x7C00u) && ((bits & 0x03FFu) != 0u);
}

void sub_f16_arm_elementwise_sub_f16(void)
{
    float16_t output[SUB_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_elementwise_sub_f16(sub_f16_input1,
                                              sub_f16_input2,
                                              output,
                                              SUB_F16_OUT_ACTIVATION_MIN,
                                              SUB_F16_OUT_ACTIVATION_MAX,
                                              SUB_F16_DST_SIZE));

    for (int i = 0; i < SUB_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(6.0e-3f, (float)sub_f16_output_ref[i], (float)output[i]);
    }
}

void sub_f16_nan_inf_arm_elementwise_sub_f16(void)
{
    // Non-finite inputs are not supported (#333), so this case is a regression pin, not a contract. What
    // comes back for the two NaN-producing lanes is toolchain dependent: some toolchains fold the clamp's
    // NaN check away and return a clamp bound, others keep it and return a real NaN. Those lanes are
    // therefore required only to be a NaN or exactly one of the two bounds - anything else, such as a stray
    // 0.0f or an unclamped Inf, is still a failure. The Inf-overflow lanes are pinned to the bounds.
    const float16_t inf = (float16_t)(_Float16)INFINITY;
    const float16_t in1[4] = {inf, (float16_t)(_Float16)NAN, inf, (float16_t)(_Float16)(-(_Float16)INFINITY)};
    const float16_t in2[4] = {inf, (float16_t)(_Float16)0.0f, (float16_t)(_Float16)1.0f, (float16_t)(_Float16)1.0f};
    float16_t output[4] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_elementwise_sub_f16(in1,
                                              in2,
                                              output,
                                              (float16_t)(_Float16)-6.0f,
                                              (float16_t)(_Float16)6.0f,
                                              4));

    for (int32_t i = 0; i < 2; i++)
    {
        const float32_t y = (float32_t)output[i];
        TEST_ASSERT_TRUE(sub_f16_bits_are_nan(output[i]) || y == -6.0f || y == 6.0f);
    }
    TEST_ASSERT_EQUAL_FLOAT(6.0f, (float)output[2]);
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, (float)output[3]);
}

void sub_f16_arg_error_arm_elementwise_sub_f16(void)
{
    float16_t output[SUB_F16_DST_SIZE] = {0};
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_elementwise_sub_f16(
            NULL, sub_f16_input2, output, SUB_F16_OUT_ACTIVATION_MIN, SUB_F16_OUT_ACTIVATION_MAX, SUB_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_elementwise_sub_f16(
            sub_f16_input1, sub_f16_input2, output, SUB_F16_OUT_ACTIVATION_MIN, SUB_F16_OUT_ACTIVATION_MAX, 0));
}
