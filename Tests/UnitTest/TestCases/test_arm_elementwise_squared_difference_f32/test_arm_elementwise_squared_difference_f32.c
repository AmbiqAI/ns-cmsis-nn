/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */
#include "arm_nnfunctions.h"
#include "squared_difference_f32_data.h"
#include "unity.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static uint32_t bits_of(float32_t x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}
static float32_t from_bits(uint32_t u)
{
    float32_t x;
    memcpy(&x, &u, sizeof(x));
    return x;
}
static void expect_bits(uint32_t expected, float32_t actual)
{
    if ((expected & 0x7fffffffu) > 0x7f800000u)
        TEST_ASSERT_TRUE((bits_of(actual) & 0x7fffffffu) > 0x7f800000u);
    else
        TEST_ASSERT_EQUAL_HEX32(expected, bits_of(actual));
}
/* Column 2 is exact, column 3 flushes operands and results. The MVE path redoes flushed squares on the scalar FPU,
 * so it gives column 2 at FZ=0 like the scalar path (#877). Autovectorized MVE FP32 arithmetic uses
 * StandardFPSCRValue (FZ=1), so those builds set scalar FZ as well and mixed scalar/vector tails have one oracle. */
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2) && defined(ARM_MATH_AUTOVECTORIZE)
#define EXPECTED_COLUMN 3
#else
#define EXPECTED_COLUMN 2
#endif

enum { N = sizeof(squared_difference_cases) / sizeof(squared_difference_cases[0]) };

static void squared_difference_f32_check_column(int column)
{
    const int lengths[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, N};
    float32_t output[N + 2];
    for (unsigned k = 0; k < sizeof(lengths) / sizeof(lengths[0]); ++k)
    {
        const int n = lengths[k];
        /* Exact allocation also makes an unpredicated tail load visible to ASan. */
        float32_t *a = malloc(n * sizeof(*a));
        float32_t *b = malloc(n * sizeof(*b));
        TEST_ASSERT_NOT_NULL(a);
        TEST_ASSERT_NOT_NULL(b);
        for (int i = 0; i < n; ++i)
        {
            a[i] = from_bits(squared_difference_cases[i][0]);
            b[i] = from_bits(squared_difference_cases[i][1]);
        }
        for (int i = 0; i < N + 2; ++i)
            output[i] = 123.0f;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_squared_difference_f32(a, b, output + 1, n));
        for (int i = 0; i < n; ++i)
            expect_bits(squared_difference_cases[i][column], output[i + 1]);
        expect_bits(bits_of(123.0f), output[0]);
        for (int i = n + 1; i < N + 2; ++i)
            expect_bits(bits_of(123.0f), output[i]);
        free(a);
        free(b);
    }
}

static void squared_difference_f32_check_alias(int column)
{
    float32_t a[N], b[N];
    for (int which = 0; which < 2; ++which)
    {
        for (int i = 0; i < N; ++i)
        {
            a[i] = from_bits(squared_difference_cases[i][0]);
            b[i] = from_bits(squared_difference_cases[i][1]);
        }
        float32_t *out = which ? a : b;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_squared_difference_f32(a, b, out, N));
        for (int i = 0; i < N; ++i)
            expect_bits(squared_difference_cases[i][column], out[i]);
    }
}

void squared_difference_f32_values(void) { squared_difference_f32_check_column(EXPECTED_COLUMN); }

/* With FPSCR.FZ set the scalar path flushes operands and results as MVE does, so the MVE path skips its redo, in
 * place too (#877) */
void squared_difference_f32_flush_to_zero(void)
{
#if defined(ARM_NN_FP_CONTROL_AVAILABLE)
    const arm_nn_fp_control_t saved = arm_nn_get_fp_control();
    arm_nn_set_fp_control(saved | (1u << 24));
    squared_difference_f32_check_column(3);
    squared_difference_f32_check_alias(3);
    arm_nn_set_fp_control(saved);
#else
    TEST_IGNORE_MESSAGE("FP controls require Arm floating-point hardware");
#endif
}

void squared_difference_f32_alias(void)
{
    float32_t a[N];
    squared_difference_f32_check_alias(EXPECTED_COLUMN);
    for (int i = 0; i < N; ++i)
        a[i] = (float32_t)(i - 250);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_squared_difference_f32(a, a, a, N));
    for (int i = 0; i < N; ++i)
        expect_bits(0, a[i]);
}

/* One risky lane among zero differences, so it alone decides whether the MVE path's check reruns: 0x1FFFFFFF is the
 * largest difference whose square MVE flushes and 2^-63 the smallest with a normal square (#877) */
void squared_difference_f32_threshold(void)
{
    const uint32_t cases[][3] = {
        /* difference, exact square, flushed square */
        {0x1fffffffu, 0x007fffffu, 0x00000000u},
        {0x9fffffffu, 0x007fffffu, 0x00000000u},
        {0x20000000u, 0x00800000u, 0x00800000u},
        {0x00800000u, 0x00000000u, 0x00000000u},
    };
    const int lengths[] = {1, 6, 9};
    float32_t a[9], b[9], out[9];
    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c)
    {
        for (unsigned k = 0; k < sizeof(lengths) / sizeof(lengths[0]); ++k)
        {
            const int n = lengths[k];
            for (int lane = 0; lane < n; ++lane)
            {
                for (int i = 0; i < n; ++i)
                {
                    a[i] = (float32_t)(i + 1);
                    b[i] = a[i];
                }
                a[lane] = from_bits(cases[c][0]);
                b[lane] = 0.0f;
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_squared_difference_f32(a, b, out, n));
                for (int i = 0; i < n; ++i)
                    expect_bits(i == lane ? cases[c][EXPECTED_COLUMN - 1] : 0u, out[i]);
            }
        }
    }
}

void squared_difference_f32_arg_error(void)
{
    const float32_t a = 1.0f;
    float32_t out = 123.0f;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f32(NULL, &a, &out, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f32(&a, NULL, &out, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f32(&a, &a, NULL, 1));
    const int invalid[] = {0, -1, INT32_MIN, INT32_MAX / 4 + 1, INT32_MAX};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f32(&a, &a, &out, invalid[i]));
    expect_bits(bits_of(123.0f), out);
}
