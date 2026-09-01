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

/*
 * Rebuild a float16 from bits held in a volatile so the non-finite inputs are created
 * at run time. Without the volatile staging, -Ofast constant-folds arithmetic on the
 * compile-time Inf/NaN under -ffinite-math-only and the kernel is never handed a
 * non-finite value at all.
 */
static float16_t sub_f16_from_bits(volatile const uint16_t *bits)
{
    const uint16_t b = *bits;
    float16_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

void sub_f16_nan_inf_arm_elementwise_sub_f16(void)
{
    // NaN propagates through the clamp (TFLite semantics) at every optimization level, including the
    // shipped -Ofast: the clamp classifies NaN on the integer bit pattern, which -ffinite-math-only has
    // no license to fold (#333, #334). Inf - Inf and NaN - 0 must therefore come back as NaN, and the
    // Inf-overflow lanes must clamp to the activation bounds.
    volatile uint16_t inf_bits = 0x7C00u;
    volatile uint16_t nan_bits = 0x7E00u;
    volatile uint16_t ninf_bits = 0xFC00u;
    const float16_t inf = sub_f16_from_bits(&inf_bits);
    const float16_t nan = sub_f16_from_bits(&nan_bits);
    const float16_t ninf = sub_f16_from_bits(&ninf_bits);
    const float16_t in1[4] = {inf, nan, inf, ninf};
    const float16_t in2[4] = {inf, (float16_t)(_Float16)0.0f, (float16_t)(_Float16)1.0f, (float16_t)(_Float16)1.0f};
    float16_t output[4] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_elementwise_sub_f16(in1, in2, output, (float16_t)(_Float16)-6.0f, (float16_t)(_Float16)6.0f, 4));

    TEST_ASSERT_TRUE_MESSAGE(sub_f16_bits_are_nan(output[0]), "Expected NaN from Inf - Inf");
    TEST_ASSERT_TRUE_MESSAGE(sub_f16_bits_are_nan(output[1]), "Expected NaN from NaN - 0");
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
