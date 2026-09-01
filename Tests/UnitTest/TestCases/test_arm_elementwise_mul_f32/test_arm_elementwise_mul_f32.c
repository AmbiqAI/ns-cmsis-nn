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

/*
 * Bit-pattern NaN test. isnan() is not usable here: -ffinite-math-only, which the
 * -Ofast this suite is built with by default implies, licenses the compiler to fold
 * isnan() to a constant false, so it cannot observe a NaN result even on toolchains
 * where the kernel returns one.
 */
static bool mul_f32_bits_are_nan(float32_t x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
}

/*
 * Rebuild a float from bits held in a volatile so the non-finite inputs are created at
 * run time. Without the volatile staging, -Ofast constant-folds arithmetic on the
 * compile-time Inf/NaN under -ffinite-math-only and the kernel is never handed a
 * non-finite value at all.
 */
static float32_t mul_f32_from_bits(volatile const uint32_t *bits)
{
    const uint32_t b = *bits;
    float32_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

void mul_f32_arm_elementwise_mul_f32(void)
{
    // Exactly representable values, so the reference is bit-exact; the last lanes
    // exercise both clamp bounds.
    const float32_t in1[9] = {0.5f, -2.0f, 3.25f, 1.75f, -0.125f, 8.0f, -8.0f, 2.5f, 1.0f};
    const float32_t in2[9] = {0.25f, 1.0f, -1.5f, 2.0f, 4.0f, 1.0f, 1.0f, -2.0f, 0.0f};
    const float32_t ref[9] = {0.125f, -2.0f, -4.875f, 3.5f, -0.5f, 6.0f, -6.0f, -5.0f, 0.0f};
    float32_t output[9] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_mul_f32(in1, in2, output, -6.0f, 6.0f, 9));

    for (int32_t i = 0; i < 9; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT(ref[i], output[i]);
    }
}

void mul_f32_nan_inf_arm_elementwise_mul_f32(void)
{
    // NaN propagates through the clamp (TFLite semantics) at every optimization level, including the
    // shipped -Ofast: the clamp classifies NaN on the integer bit pattern, which -ffinite-math-only has
    // no license to fold (#333, #334). Inf * 0 and a NaN operand must therefore come back as NaN,
    // and the Inf-overflow lanes must clamp to the activation bounds.
    volatile uint32_t inf_bits = 0x7F800000u;
    volatile uint32_t nan_bits = 0x7FC00000u;
    const float32_t inf = mul_f32_from_bits(&inf_bits);
    const float32_t nan = mul_f32_from_bits(&nan_bits);
    const float32_t in1[4] = {inf, nan, inf, -inf};
    const float32_t in2[4] = {0.0f, 1.0f, 1.0f, 1.0f};
    float32_t output[4] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_mul_f32(in1, in2, output, -6.0f, 6.0f, 4));

    TEST_ASSERT_TRUE_MESSAGE(mul_f32_bits_are_nan(output[0]), "Expected NaN from Inf * 0");
    TEST_ASSERT_TRUE_MESSAGE(mul_f32_bits_are_nan(output[1]), "Expected NaN from NaN operand");
    TEST_ASSERT_EQUAL_FLOAT(6.0f, output[2]);
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, output[3]);
}

void mul_f32_arg_error_arm_elementwise_mul_f32(void)
{
    float32_t in[2] = {0.0f, 0.0f};
    float32_t output[2] = {0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_mul_f32(NULL, in, output, -6.0f, 6.0f, 2));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_mul_f32(in, in, output, -6.0f, 6.0f, 0));
}
