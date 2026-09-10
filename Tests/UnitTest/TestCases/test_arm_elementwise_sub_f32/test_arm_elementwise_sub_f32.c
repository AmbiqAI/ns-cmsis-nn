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
 * Bit-pattern NaN test. Harness test TUs are built with -fno-finite-math-only
 * (Tests/UnitTest/CMakeLists.txt), so isnan() would also work here; the bit-pattern
 * check is defense-in-depth for a standalone -Ofast build of this TU, where the
 * implied -ffinite-math-only licenses folding isnan() to a constant false.
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

/*
 * Rebuild a float from bits held in a volatile so the non-finite inputs are created
 * at run time. Harness test TUs are built with -fno-finite-math-only
 * (Tests/UnitTest/CMakeLists.txt), so the staging is not required there; it is
 * defense-in-depth for a standalone -Ofast build of this TU, where the implied
 * -ffinite-math-only licenses constant-folding arithmetic on the compile-time
 * Inf/NaN so the kernel is never handed a non-finite value at all.
 */
static float32_t sub_f32_from_bits(volatile const uint32_t *bits)
{
    const uint32_t b = *bits;
    float32_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

void sub_f32_nan_inf_arm_elementwise_sub_f32(void)
{
    // NaN propagates through the clamp (TFLite semantics) at every optimization level, including the
    // shipped -Ofast: the clamp classifies NaN on the integer bit pattern, which -ffinite-math-only has
    // no license to fold (#333, #334). Inf - Inf and NaN - 0 must therefore come back as NaN, and the
    // Inf-overflow lanes must clamp to the activation bounds.
    volatile uint32_t inf_bits = 0x7F800000u;
    volatile uint32_t nan_bits = 0x7FC00000u;
    const float32_t inf = sub_f32_from_bits(&inf_bits);
    const float32_t nan = sub_f32_from_bits(&nan_bits);
    const float32_t in1[4] = {inf, nan, inf, -inf};
    const float32_t in2[4] = {inf, 0.0f, 1.0f, 1.0f};
    float32_t output[4] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_sub_f32(in1, in2, output, -6.0f, 6.0f, 4));

    TEST_ASSERT_TRUE_MESSAGE(sub_f32_bits_are_nan(output[0]), "Expected NaN from Inf - Inf");
    TEST_ASSERT_TRUE_MESSAGE(sub_f32_bits_are_nan(output[1]), "Expected NaN from NaN - 0");
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
