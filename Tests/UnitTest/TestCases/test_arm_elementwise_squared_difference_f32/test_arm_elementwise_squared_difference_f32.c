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
/* MVE FP32 arithmetic uses StandardFPSCRValue (FZ=1). Autovectorized
 * builds set scalar FZ as well, so mixed scalar/vector tails have one oracle. */
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
#define EXPECTED_COLUMN 3
#else
#define EXPECTED_COLUMN 2
#endif

enum { N = sizeof(squared_difference_cases) / sizeof(squared_difference_cases[0]) };

void squared_difference_f32_values(void)
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
            expect_bits(squared_difference_cases[i][EXPECTED_COLUMN], output[i + 1]);
        expect_bits(bits_of(123.0f), output[0]);
        for (int i = n + 1; i < N + 2; ++i)
            expect_bits(bits_of(123.0f), output[i]);
        free(a);
        free(b);
    }
}

void squared_difference_f32_alias(void)
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
            expect_bits(squared_difference_cases[i][EXPECTED_COLUMN], out[i]);
    }
    for (int i = 0; i < N; ++i)
        a[i] = (float32_t)(i - 250);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_squared_difference_f32(a, a, a, N));
    for (int i = 0; i < N; ++i)
        expect_bits(0, a[i]);
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
