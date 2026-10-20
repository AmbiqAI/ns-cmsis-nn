/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnfunctions.h"
#include "div_data.h"
#include "unity.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static uint16_t bits_of(float16_t x)
{
    uint16_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}
static float16_t from_bits(uint16_t u)
{
    float16_t x;
    memcpy(&x, &u, sizeof(x));
    return x;
}
static void expect_bits(uint16_t expected, float16_t actual)
{
    const uint16_t magnitude = expected & 0x7fffu;
    if (magnitude > 0x7c00u)
        TEST_ASSERT_TRUE((bits_of(actual) & 0x7fffu) > 0x7c00u);
    else
        TEST_ASSERT_EQUAL_HEX16(expected, bits_of(actual));
}

void div_values(void)
{
    enum
    {
        N = sizeof(div_cases) / sizeof(div_cases[0])
    };
    float16_t a[N], b[N], out[N + 2];
    for (int i = 0; i < N; ++i)
    {
        a[i] = from_bits(div_cases[i][0]);
        b[i] = from_bits(div_cases[i][1]);
    }
    const int lengths[] = {1, 3, 4, 7, 8, 9, 17, 31, 33, N};
    for (unsigned k = 0; k < sizeof(lengths) / sizeof(lengths[0]); ++k)
    {
        for (int i = 0; i < N + 2; ++i)
            out[i] = (float16_t)123;
        const int n = lengths[k];
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_div_f16(a, b, out + 1, -INFINITY, INFINITY, n));
        for (int i = 0; i < n; ++i)
            expect_bits(div_cases[i][2], out[i + 1]);
        TEST_ASSERT_EQUAL_HEX16(bits_of((float16_t)123), bits_of(out[0]));
        TEST_ASSERT_EQUAL_HEX16(bits_of((float16_t)123), bits_of(out[n + 1]));
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_div_f16(a, b, a, -INFINITY, INFINITY, N));
    for (int i = 0; i < N; ++i)
        expect_bits(div_cases[i][2], a[i]);
    for (int i = 0; i < N; ++i)
        a[i] = from_bits(div_cases[i][0]);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_div_f16(a, b, b, -INFINITY, INFINITY, N));
    for (int i = 0; i < N; ++i)
        expect_bits(div_cases[i][2], b[i]);
    for (int i = 0; i < N; ++i)
        b[i] = from_bits(div_cases[i][1]);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_div_f16(a, b, out, -3, 3, N));
    for (int i = 0; i < N; ++i)
    {
        float x = (float)from_bits(div_cases[i][2]);
        if (x < -3)
            x = -3;
        if (x > 3)
            x = 3;
        expect_bits(bits_of((float16_t)x), out[i]);
    }
}

static int count(cmsis_nn_dims s) { return s.n * s.h * s.w * s.c; }
static int offset(cmsis_nn_dims s, int n, int h, int w, int ch)
{
    return (((n % s.n) * s.h + h % s.h) * s.w + w % s.w) * s.c + ch % s.c;
}
static void check_shape(cmsis_nn_dims a_shape, cmsis_nn_dims b_shape, cmsis_nn_dims out_shape)
{
    static float16_t a[3232], b[3232], out[3234];
    for (int i = 0; i < count(a_shape); ++i)
        a[i] = (float16_t)((i % 37) - 19);
    for (int i = 0; i < count(b_shape); ++i)
        b[i] = (float16_t)((i % 13) + 1);
    out[0] = out[count(out_shape) + 1] = (float16_t)123;
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_elementwise_div_broadcast_f16(a, &a_shape, b, &b_shape, out + 1, &out_shape, -INFINITY, INFINITY));
    int i = 1;
    for (int n = 0; n < out_shape.n; ++n)
        for (int h = 0; h < out_shape.h; ++h)
            for (int w = 0; w < out_shape.w; ++w)
                for (int ch = 0; ch < out_shape.c; ++ch)
                {
                    const double av = (double)a[offset(a_shape, n, h, w, ch)];
                    const double bv = (double)b[offset(b_shape, n, h, w, ch)];
                    expect_bits(bits_of((float16_t)(av / bv)), out[i++]);
                }
    TEST_ASSERT_EQUAL_HEX16(bits_of((float16_t)123), bits_of(out[0]));
    TEST_ASSERT_EQUAL_HEX16(bits_of((float16_t)123), bits_of(out[i]));
}

void div_broadcast(void)
{
    const int extents[4] = {2, 3, 5, 7};
    for (int mask = 0; mask < 81; ++mask)
    {
        int a[4], b[4], code = mask;
        for (int j = 0; j < 4; ++j)
        {
            a[j] = extents[j];
            b[j] = extents[j];
            if (code % 3 == 1)
                a[j] = 1;
            if (code % 3 == 2)
                b[j] = 1;
            code /= 3;
        }
        check_shape((cmsis_nn_dims){a[0], a[1], a[2], a[3]},
                    (cmsis_nn_dims){b[0], b[1], b[2], b[3]},
                    (cmsis_nn_dims){2, 3, 5, 7});
    }
    check_shape((cmsis_nn_dims){1, 1, 1, 1}, (cmsis_nn_dims){1, 1, 1, 1}, (cmsis_nn_dims){1, 1, 1, 1});
    check_shape((cmsis_nn_dims){1, 1, 33, 16}, (cmsis_nn_dims){1, 1, 1, 1}, (cmsis_nn_dims){1, 1, 33, 16});
    check_shape((cmsis_nn_dims){1, 1, 101, 32}, (cmsis_nn_dims){1, 1, 101, 1}, (cmsis_nn_dims){1, 1, 101, 32});
    /* Each arithmetic fixture also exercises scalar numerator and denominator broadcasts. */
    cmsis_nn_dims scalar = {1, 1, 1, 1}, shape = {1, 1, 1, 9};
    for (unsigned k = 0; k < sizeof(div_cases) / sizeof(div_cases[0]); ++k)
    {
        float16_t a[9], b[9], out[9];
        for (int i = 0; i < 9; ++i)
        {
            a[i] = from_bits(div_cases[k][0]);
            b[i] = from_bits(div_cases[k][1]);
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_elementwise_div_broadcast_f16(a, &scalar, b, &shape, out, &shape, -INFINITY, INFINITY));
        for (int i = 0; i < 9; ++i)
            expect_bits(div_cases[k][2], out[i]);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_elementwise_div_broadcast_f16(a, &shape, b, &scalar, out, &shape, -INFINITY, INFINITY));
        for (int i = 0; i < 9; ++i)
            expect_bits(div_cases[k][2], out[i]);
    }
}

void div_invalid(void)
{
    float16_t a = 1, b = 2, out = 123;
    cmsis_nn_dims one = {1, 1, 1, 1};
    const int lengths[] = {0, -1, INT32_MIN, INT32_MAX};
    for (unsigned k = 0; k < sizeof(lengths) / sizeof(lengths[0]); ++k)
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_elementwise_div_f16(&a, &b, &out, -INFINITY, INFINITY, lengths[k]));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_div_f16(NULL, &b, &out, -INFINITY, INFINITY, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_div_f16(&a, NULL, &out, -INFINITY, INFINITY, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_div_f16(&a, &b, NULL, -INFINITY, INFINITY, 1));
    for (int k = 0; k < 6; ++k)
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_elementwise_div_broadcast_f16(k == 0 ? NULL : &a,
                                                            k == 1 ? NULL : &one,
                                                            k == 2 ? NULL : &b,
                                                            k == 3 ? NULL : &one,
                                                            k == 4 ? NULL : &out,
                                                            k == 5 ? NULL : &one,
                                                            -INFINITY,
                                                            INFINITY));
    const int invalid[][4] = {{0, 1, 1, 1},
                              {1, -1, 1, 1},
                              {1, 1, INT32_MIN, 1},
                              {1, 1, 1, 0},
                              {1, 1, 1, INT32_MAX},
                              {65536, 65536, 1, 1},
                              {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX}};
    for (unsigned k = 0; k < sizeof(invalid) / sizeof(invalid[0]); ++k)
    {
        cmsis_nn_dims s = {invalid[k][0], invalid[k][1], invalid[k][2], invalid[k][3]};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_elementwise_div_broadcast_f16(&a, &s, &b, &one, &out, &s, -INFINITY, INFINITY));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_elementwise_div_broadcast_f16(&a, &one, &b, &s, &out, &s, -INFINITY, INFINITY));
    }
    for (int axis = 0; axis < 4; ++axis)
    {
        int d[4] = {1, 1, 1, 1};
        d[axis] = 2;
        cmsis_nn_dims a2 = {d[0], d[1], d[2], d[3]};
        d[axis] = 3;
        cmsis_nn_dims b3 = {d[0], d[1], d[2], d[3]};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_elementwise_div_broadcast_f16(&a, &a2, &b, &b3, &out, &b3, -INFINITY, INFINITY));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_elementwise_div_broadcast_f16(&a, &a2, &b, &one, &out, &one, -INFINITY, INFINITY));
    }
    TEST_ASSERT_EQUAL_HEX16(bits_of((float16_t)123), bits_of(out));
}
