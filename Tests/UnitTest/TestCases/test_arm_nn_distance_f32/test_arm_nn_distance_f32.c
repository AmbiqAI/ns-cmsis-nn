/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "arm_nnfunctions.h"
#include "unity.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define MAX_F 65
#define MAX_N 1000
/* Static arrays keep the embedded test's stack bounded. */
static float input[MAX_F], refs[MAX_F * MAX_N], actual[MAX_N];
typedef arm_cmsis_nn_status (*distance_fn)(const float *, const float *, int32_t, int32_t, float *);
static distance_fn kernels[] = {arm_nn_l1_distance_f32, arm_nn_sq_l2_distance_f32};

static void compare_chain(int32_t n, int32_t f, int square)
{
    const cmsis_nn_dims row_dims = {1, 1, 1, f};
    float deltas[MAX_F], terms[MAX_F];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, kernels[square](input, refs, n, f, actual));
    /* Each reference row has an independent reduction. Reuse the oracle's
     * scratch for each row while exercising the full batch in the fused call. */
    for (int i = 0; i < n; ++i)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_elementwise_sub_broadcast_f32(
                              input, &row_dims, refs + i * f, &row_dims, deltas, &row_dims, -FLT_MAX, FLT_MAX));
        if (square)
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                              arm_elementwise_mul_f32(deltas, deltas, terms, -FLT_MAX, FLT_MAX, f));
        }
        else
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_abs_f32(deltas, terms, f));
        }
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        /* The fused entry accumulates in scalar order on MVE. */
        volatile float sum = 0.0f;
        for (int col = 0; col < f; ++col)
        {
            sum += terms[col];
        }
        const float wanted = sum;
#else
        float wanted;
        const cmsis_nn_dims axis = {0, 0, 0, 1}, out_dims = {1, 1, 1, 1};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(terms, &row_dims, &axis, &wanted, &out_dims));
#endif
        if (isnan(wanted))
        {
            TEST_ASSERT_TRUE(isnan(actual[i]));
        }
        else
        {
            uint32_t a, b;
            memcpy(&a, actual + i, sizeof(a));
            memcpy(&b, &wanted, sizeof(b));
            TEST_ASSERT_EQUAL_HEX32(b, a);
        }
    }
}
static void distances_random(void)
{
    const int ns[] = {1, 3, 4, 5, 1000}, fs[] = {1, 2, 64, 65};
    uint32_t seed = 0x17397531u;
    for (int i = 0; i < MAX_F; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        input[i] = (float)(seed % 1009) / 32.0f - 15.0f;
    }
    for (int i = 0; i < MAX_F * MAX_N; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        refs[i] = (float)(seed % 1237) / 64.0f - 10.0f;
    }
    for (int n = 0; n < 5; ++n)
        for (int f = 0; f < 4; ++f)
            for (int s = 0; s < 2; ++s)
                compare_chain(ns[n], fs[f], s);
}
static void distances_special(void)
{
    const float values[] = {NAN, INFINITY, -INFINITY, -0.0f, 0.0f, FLT_MAX, -FLT_MAX, 1.0f, -1.0f};
    for (int x = 0; x < 9; ++x)
        for (int r = 0; r < 9; ++r)
        {
            for (int f = 0; f < 5; ++f)
            {
                input[f] = values[x];
                refs[f] = values[r];
            }
            for (int s = 0; s < 2; ++s)
                compare_chain(1, 5, s);
        }
}
static void distances_invalid(void)
{
    for (int s = 0; s < 2; ++s)
    {
        actual[0] = 123.0f;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](NULL, refs, 1, 1, actual));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](input, NULL, 1, 1, actual));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](input, refs, 1, 1, NULL));
        const int bad[] = {0, -1, INT32_MIN, INT32_MAX};
        for (int i = 0; i < 4; ++i)
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](input, refs, bad[i], 1, actual));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](input, refs, 1, bad[i], actual));
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](input, refs, 65536, 65536, actual));
        TEST_ASSERT_EQUAL_FLOAT(123.0f, actual[0]);
    }
}
static void distances_overlap(void)
{
    for (int s = 0; s < 2; ++s)
    {
        input[0] = 123.0f;
        refs[0] = 456.0f;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](input, refs, 2, 3, input));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](input, refs, 2, 3, input + 1));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, kernels[s](input, refs, 2, 3, refs + 1));
        TEST_ASSERT_EQUAL_FLOAT(123.0f, input[0]);
        TEST_ASSERT_EQUAL_FLOAT(456.0f, refs[0]);
    }
}

/* With -Ofast, inlined floating comparisons against FLT_MAX were removed,
 * even though finite inputs can overflow when squared. */
static void distances_finite_overflow(void)
{
    input[0] = -2.1541898e21f;
    refs[0] = -1493.83362f;
    compare_chain(1, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(FLT_MAX, actual[0]);
}
