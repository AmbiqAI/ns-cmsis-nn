/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

/* After arm_nnfunctions.h: the guard is enabled from its target macros. */
#include "../Utils/mpu_guard.h"

/* Values are compared and classified on their bits; no conditional select touches float16 (GCC PR 118460). */
static uint16_t gelu_bits(float16_t x)
{
    uint16_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static float16_t gelu_half(uint16_t u)
{
    float16_t x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static bool gelu_is_nan(uint16_t u) { return (u & 0x7fffU) > 0x7c00U; }

/* C adaptation of GeluTransform in TensorFlow v2.16.1 reference/gelu.h, evaluated in float32 on the widened input
 * and rounded to float16 once. Keep the decimal producer constant independent of the kernel's encoding.
 */
static uint16_t gelu_producer(uint16_t u)
{
    /* Force the linked libm call, not compiler-folded transcendental results. */
    float (*volatile runtime_erfc)(float) = erfcf;
    const float x = (float)gelu_half(u);
    return gelu_bits((float16_t)(0.5f * x * runtime_erfc(x * (float)-0.70710678118654752440)));
}

static bool gelu_equal(uint16_t expected, uint16_t actual)
{
    if (gelu_is_nan(expected))
    {
        return gelu_is_nan(actual);
    }
    return expected == actual;
}

/* x * Phi(x) in double, independent of the float32 erfcf the kernel and producer share. */
static double gelu_reference(uint16_t u)
{
    const double x = (double)(float)gelu_half(u);
    return 0.5 * x * erfc(-x / sqrt(2.0));
}

/* Every float16 bit pattern, one scalar call each: an array call may take a vectorized libm. */
void gelu_f16_exhaustive(void)
{
    uint32_t mismatches = 0, first = 0;
    for (uint32_t u = 0; u <= 0xffffU; ++u)
    {
        const float16_t input = gelu_half((uint16_t)u);
        float16_t output = gelu_half(0x5a5aU);
        if (arm_nn_gelu_f16(&input, &output, 1) != ARM_CMSIS_NN_SUCCESS ||
            !gelu_equal(gelu_producer((uint16_t)u), gelu_bits(output)))
        {
            first = mismatches++ ? first : u;
        }
    }
    char message[48];
    snprintf(message, sizeof(message), "first mismatch at input 0x%04lx", (unsigned long)first);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0, mismatches, message);
}

/* The documented bound, |out - ref| <= 2^-10 * |ref| + 2^-24 with FPSCR.FZ16 clear, for every finite input. */
void gelu_f16_accuracy(void)
{
    uint32_t violations = 0, first = 0;
    for (uint32_t u = 0; u <= 0xffffU; ++u)
    {
        if ((u & 0x7c00U) == 0x7c00U)
        {
            continue;
        }
        const float16_t input = gelu_half((uint16_t)u);
        float16_t output;
        const double ref = gelu_reference((uint16_t)u);
        const bool ok = arm_nn_gelu_f16(&input, &output, 1) == ARM_CMSIS_NN_SUCCESS &&
            fabs((double)(float)output - ref) <= ldexp(fabs(ref), -10) + ldexp(1.0, -24);
        if (!ok)
        {
            first = violations++ ? first : u;
        }
    }
    char message[48];
    snprintf(message, sizeof(message), "first violation at input 0x%04lx", (unsigned long)first);
    TEST_ASSERT_EQUAL_UINT32_MESSAGE(0, violations, message);
}

/* Fixed points of the contract that do not depend on the producer. */
void gelu_f16_special(void)
{
    static const uint16_t cases[][2] = {
        {0x0000, 0x0000}, /* +0 */
        {0x8000, 0x8000}, /* -0 keeps its sign */
        {0x7c00, 0x7c00}, /* +Inf */
        {0x7bff, 0x7bff}, /* 65504: Phi rounds to 1 */
        {0x4800, 0x4800}, /* 8 */
        {0xfbff, 0x8000}, /* -65504 underflows to -0 */
        {0x3c00, 0x3abb}, /* 1 -> 0.8413 */
        {0xbc00, 0xb114}, /* -1 -> -0.1587 */
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        const float16_t input = gelu_half(cases[i][0]);
        float16_t output;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f16(&input, &output, 1));
        TEST_ASSERT_EQUAL_HEX16(cases[i][1], gelu_bits(output));
    }

    /* -Inf * Phi(-Inf) is -Inf * 0, NaN as in TFLite; NaN inputs stay NaN. */
    static const uint16_t to_nan[] = {0xfc00, 0x7e00, 0xfe00, 0x7c01, 0x7fff};
    for (unsigned i = 0; i < sizeof(to_nan) / sizeof(to_nan[0]); ++i)
    {
        const float16_t input = gelu_half(to_nan[i]);
        float16_t output;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f16(&input, &output, 1));
        TEST_ASSERT_TRUE(gelu_is_nan(gelu_bits(output)));
    }

    /* Subnormal inputs widen exactly, so a tiny x maps to about x / 2, still subnormal. */
    const float16_t tiny = gelu_half(0x0002);
    float16_t half_tiny;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f16(&tiny, &half_tiny, 1));
    TEST_ASSERT_EQUAL_HEX16(0x0001, gelu_bits(half_tiny));
}

void gelu_f16_counts(void)
{
    const int counts[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33};
    float16_t input[36], output[36];
    for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c)
    {
        for (int i = 0; i < 36; ++i)
        {
            /* Here erfc rounds to 2, so GELU is exactly the input on any libm path. */
            input[i] = (float16_t)(32.0f + (float)i);
            output[i] = (float16_t)99.0f;
        }
        const int n = counts[c];
        /* Two-byte offset: no vector alignment may be required. */
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f16(input + 1, output + 1, n));
        TEST_ASSERT_EQUAL_HEX16(gelu_bits((float16_t)99.0f), gelu_bits(output[0]));
        TEST_ASSERT_EQUAL_HEX16(gelu_bits((float16_t)99.0f), gelu_bits(output[n + 1]));
        for (int i = 1; i <= n; ++i)
        {
            TEST_ASSERT_EQUAL_HEX16(gelu_bits(input[i]), gelu_bits(output[i]));
        }
    }
}

void gelu_f16_in_place(void)
{
    float16_t values[17], reference[17];
    for (int i = 0; i < 17; ++i)
    {
        values[i] = reference[i] = (float16_t)(32.0f + (float)i);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f16(values, values, 17));
    for (int i = 0; i < 17; ++i)
    {
        TEST_ASSERT_EQUAL_HEX16(gelu_bits(reference[i]), gelu_bits(values[i]));
    }
    /* Nontrivial in-place arithmetic, one element at a time against the producer. */
    for (int i = 0; i < 17; ++i)
    {
        float16_t value = (float16_t)((float)(i - 12) / 4.0f);
        const uint16_t expected = gelu_producer(gelu_bits(value));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f16(&value, &value, 1));
        TEST_ASSERT_TRUE(gelu_equal(expected, gelu_bits(value)));
    }
}

void gelu_f16_arguments(void)
{
    float16_t x = (float16_t)1.0f, y = (float16_t)99.0f;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f16(NULL, NULL, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_gelu_f16(&x, &y, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_gelu_f16(&x, &y, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_gelu_f16(NULL, &y, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_gelu_f16(&x, NULL, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_gelu_f16(NULL, NULL, INT32_MIN));
    TEST_ASSERT_EQUAL_HEX16(0x5630, gelu_bits(y));
}

/*
 * With FPSCR.FZ16 set, a result whose reference is below 2^-14 in magnitude may come back as a zero of the same
 * sign; every other result must be unchanged. Expectations are taken with FZ16 clear. Armv8.1-M conversions between
 * half and single precision do not apply FZ16, so on those targets this checks that nothing changes.
 */
void gelu_f16_flush_to_zero(void)
{
#if defined(USING_FVP_CORSTONE_300) && defined(__FPU_PRESENT) && (__FPU_PRESENT == 1U)
    const uint32_t fpscr = __get_FPSCR();
    const uint32_t fz16 = 1UL << 19;
    uint32_t violations = 0;
    for (uint32_t u = 0; u <= 0xffffU; ++u)
    {
        __set_FPSCR(fpscr & ~fz16);
        const uint16_t expected = gelu_producer((uint16_t)u);
        const float16_t input = gelu_half((uint16_t)u);
        float16_t output = gelu_half(0x5a5aU);
        __set_FPSCR(fpscr | fz16);
        const arm_cmsis_nn_status status = arm_nn_gelu_f16(&input, &output, 1);
        __set_FPSCR(fpscr);

        const uint16_t actual = gelu_bits(output);
        const bool flushed = fabs(gelu_reference((uint16_t)u)) < ldexp(1.0, -14) && actual == (expected & 0x8000U);
        if (status != ARM_CMSIS_NN_SUCCESS || !(gelu_equal(expected, actual) || flushed))
        {
            ++violations;
        }
    }
    TEST_ASSERT_EQUAL_UINT32(0, violations);
#else
    TEST_IGNORE_MESSAGE("FPSCR.FZ16 is set only on the Corstone-300 FVP");
#endif
}

void gelu_f16_bounds(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    float16_t input[17], output[17], expected[17];
    const int counts[] = {1, 3, 4, 5, 17};
    for (int i = 0; i < 17; ++i)
    {
        input[i] = expected[i] = (float16_t)(32.0f + (float)i);
    }
    for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c)
    {
        const int n = counts[c];
        const float16_t *guarded_input = guard_place(input, n * sizeof(float16_t));
        guard_gap_enable();
        arm_cmsis_nn_status status = arm_nn_gelu_f16(guarded_input, output, n);
        guard_gap_disable();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
        for (int i = 0; i < n; ++i)
        {
            TEST_ASSERT_EQUAL_HEX16(gelu_bits(expected[i]), gelu_bits(output[i]));
        }
        float16_t *guarded_output = guard_end(n * sizeof(float16_t));
        guard_gap_enable();
        status = arm_nn_gelu_f16(input, guarded_output, n);
        guard_gap_disable();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
        for (int i = 0; i < n; ++i)
        {
            TEST_ASSERT_EQUAL_HEX16(gelu_bits(expected[i]), gelu_bits(guarded_output[i]));
        }
    }
#else
    TEST_IGNORE_MESSAGE("MPU guard requires Corstone-300/MVE");
#endif
}
