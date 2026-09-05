/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Host -Ofast NaN/Inf probe for the f16 reduction kernels (#429).
 *
 * The Unity harness builds every test TU with -fno-finite-math-only
 * (Tests/UnitTest/CMakeLists.txt), so its suites cannot observe what the shipped
 * -Ofast does to the reductions' non-finite contract: arm_reduce_sum_f16 states
 * that NaN and Inf propagate, and the library is the one target the harness
 * leaves on finite-math assumptions. This probe compiles this TU AND its own
 * copies of arm_nn_mean_f16/arm_reduce_sum_f16 at -Ofast -ffinite-math-only, the
 * flag pair under which a floating-point non-finite test is a codegen property
 * rather than a contract, and checks the results as bit patterns.
 *
 * Both dispatch paths are covered: a reduced contiguous suffix takes the flatten
 * path, and a reduced interior axis takes the generic path.
 *
 * Scope: on a host neither ARM_MATH_MVEF nor ARM_MATH_MVE_FLOAT16 is defined, so
 * only the SCALAR legs run here. The MVE legs are covered on target under QEMU
 * mps3-an547 by the Unity suites.
 *
 * Inputs are staged through volatile bit patterns so the compiler cannot
 * constant-fold the non-finite arithmetic away, and the classification is a
 * bit-pattern test for the same reason isnan()/isinf() are not used: under
 * -ffinite-math-only both may fold to a constant false.
 *
 * Each case also asserts the finite lanes exactly, so a change that propagated
 * NaN by breaking the ordinary values would still fail. Exit status is the
 * number of failed checks.
 */

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define F16_NAN 0x7E00u
#define F16_POS_INF 0x7C00u
#define F16_NEG_INF 0xFC00u
#define F16_ZERO 0x0000u
#define F16_ONE 0x3C00u
#define F16_TWO 0x4000u
#define F16_THREE 0x4200u
#define F16_FOUR 0x4400u
#define F16_FIVE 0x4500u
#define F16_SIX 0x4600u
#define F16_EIGHT 0x4800u

static int failures;

static uint16_t f16_bits(float16_t x)
{
    uint16_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

// volatile so the staged non-finite input survives to the call site.
static void f16_stage(float16_t *dst, const uint16_t *pattern, int32_t count)
{
    for (int32_t i = 0; i < count; ++i)
    {
        volatile uint16_t bits = pattern[i];
        const uint16_t b = bits;
        memcpy(&dst[i], &b, sizeof(dst[i]));
    }
}

static void expect_nan(const char *what, int32_t lane, float16_t value)
{
    const uint16_t bits = f16_bits(value);
    if ((uint16_t)(bits & 0x7FFFu) <= F16_POS_INF)
    {
        ++failures;
        printf("FAIL %s[%d]: expected NaN, got 0x%04x\n", what, (int)lane, bits);
    }
}

static void expect_bits(const char *what, int32_t lane, float16_t value, uint16_t expected)
{
    const uint16_t bits = f16_bits(value);
    if (bits != expected)
    {
        ++failures;
        printf("FAIL %s[%d]: expected 0x%04x, got 0x%04x\n", what, (int)lane, expected, bits);
    }
}

// Reduced axis is the trailing one, so this takes the flatten path.
static void mean_f16_flatten_path(void)
{
    const uint16_t pattern[6] = {F16_ONE, F16_POS_INF, F16_TWO, F16_NAN, F16_ONE, F16_TWO};
    float16_t input[6];
    float16_t output[2] = {0};
    const cmsis_nn_dims input_dims = {1, 1, 2, 3};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 2, 1};

    f16_stage(input, pattern, 6);
    if (arm_nn_mean_f16(input, &input_dims, &axis_dims, output, &output_dims) != ARM_CMSIS_NN_SUCCESS)
    {
        ++failures;
        printf("FAIL mean_f16_flatten: status\n");
        return;
    }

    expect_bits("mean_f16_flatten", 0, output[0], F16_POS_INF);
    expect_nan("mean_f16_flatten", 1, output[1]);
}

// Reducing an interior axis leaves no contiguous reduced suffix, so this takes
// the generic path.
static void mean_f16_generic_path(void)
{
    const uint16_t pattern[8] =
        {F16_NAN, F16_ONE, F16_TWO, F16_THREE, F16_FOUR, F16_FIVE, F16_SIX, F16_POS_INF};
    float16_t input[8];
    float16_t output[4] = {0};
    const cmsis_nn_dims input_dims = {1, 2, 2, 2};
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {1, 1, 2, 2};

    f16_stage(input, pattern, 8);
    if (arm_nn_mean_f16(input, &input_dims, &axis_dims, output, &output_dims) != ARM_CMSIS_NN_SUCCESS)
    {
        ++failures;
        printf("FAIL mean_f16_generic: status\n");
        return;
    }

    expect_nan("mean_f16_generic", 0, output[0]);
    expect_bits("mean_f16_generic", 1, output[1], F16_THREE);
    expect_bits("mean_f16_generic", 2, output[2], F16_FOUR);
    expect_bits("mean_f16_generic", 3, output[3], F16_POS_INF);
}

static void reduce_sum_f16_flatten_path(void)
{
    const uint16_t pattern[9] = {F16_ONE,
                                 F16_POS_INF,
                                 F16_TWO,
                                 F16_POS_INF,
                                 F16_NEG_INF,
                                 F16_ZERO,
                                 F16_NAN,
                                 F16_ONE,
                                 F16_TWO};
    float16_t input[9];
    float16_t output[3] = {0};
    const cmsis_nn_dims input_dims = {1, 1, 3, 3};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 3, 1};

    f16_stage(input, pattern, 9);
    if (arm_reduce_sum_f16(input, &input_dims, &axis_dims, output, &output_dims) != ARM_CMSIS_NN_SUCCESS)
    {
        ++failures;
        printf("FAIL rsum_f16_flatten: status\n");
        return;
    }

    expect_bits("rsum_f16_flatten", 0, output[0], F16_POS_INF);
    expect_nan("rsum_f16_flatten", 1, output[1]);
    expect_nan("rsum_f16_flatten", 2, output[2]);
}

static void reduce_sum_f16_generic_path(void)
{
    const uint16_t pattern[8] =
        {F16_NAN, F16_ONE, F16_TWO, F16_THREE, F16_FOUR, F16_FIVE, F16_SIX, F16_POS_INF};
    float16_t input[8];
    float16_t output[4] = {0};
    const cmsis_nn_dims input_dims = {1, 2, 2, 2};
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {1, 1, 2, 2};

    f16_stage(input, pattern, 8);
    if (arm_reduce_sum_f16(input, &input_dims, &axis_dims, output, &output_dims) != ARM_CMSIS_NN_SUCCESS)
    {
        ++failures;
        printf("FAIL rsum_f16_generic: status\n");
        return;
    }

    expect_nan("rsum_f16_generic", 0, output[0]);
    expect_bits("rsum_f16_generic", 1, output[1], F16_SIX);
    expect_bits("rsum_f16_generic", 2, output[2], F16_EIGHT);
    expect_bits("rsum_f16_generic", 3, output[3], F16_POS_INF);
}

int main(void)
{
    mean_f16_flatten_path();
    mean_f16_generic_path();
    reduce_sum_f16_flatten_path();
    reduce_sum_f16_generic_path();

    if (failures == 0)
    {
        printf("reduction_nan_ofast_probe: PASS\n");
    }

    return failures;
}
