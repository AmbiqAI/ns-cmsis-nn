/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "unity.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef arm_cmsis_nn_status (*binary_fn)(const bool *,
                                         const cmsis_nn_dims *,
                                         const bool *,
                                         const cmsis_nn_dims *,
                                         bool *,
                                         const cmsis_nn_dims *);

static int count(const cmsis_nn_dims d) { return d.n * d.h * d.w * d.c; }

static int oracle_index(int i, const cmsis_nn_dims out, const cmsis_nn_dims in)
{
    const int c = i % out.c;
    i /= out.c;
    const int w = i % out.w;
    i /= out.w;
    const int h = i % out.h;
    const int n = i / out.h;
    return (((n % in.n) * in.h + h % in.h) * in.w + w % in.w) * in.c + c % in.c;
}

static void check_binary(cmsis_nn_dims a_dims, cmsis_nn_dims b_dims, cmsis_nn_dims out_dims)
{
    const int size = count(out_dims);
    /* Offset by one to exercise unaligned packed and vector accesses. */
    bool *a_storage = malloc((size_t)count(a_dims) + 1);
    bool *b_storage = malloc((size_t)count(b_dims) + 1);
    unsigned char *storage = malloc((size_t)size + 2);
    TEST_ASSERT_NOT_NULL(a_storage);
    TEST_ASSERT_NOT_NULL(b_storage);
    TEST_ASSERT_NOT_NULL(storage);
    bool *a = a_storage + 1;
    bool *b = b_storage + 1;
    bool *out = (bool *)(storage + 1);
    for (int i = 0; i < count(a_dims); ++i)
    {
        a[i] = (i % 3) == 0;
    }
    for (int i = 0; i < count(b_dims); ++i)
    {
        b[i] = (i % 5) < 2;
    }
    const binary_fn functions[2] = {arm_logical_and_bool, arm_logical_or_bool};
    for (int op = 0; op < 2; ++op)
    {
        memset(storage, 0xa5, (size_t)size + 2);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, functions[op](a, &a_dims, b, &b_dims, out, &out_dims));
        for (int i = 0; i < size; ++i)
        {
            const bool x = a[oracle_index(i, out_dims, a_dims)];
            const bool y = b[oracle_index(i, out_dims, b_dims)];
            const unsigned char expected = op ? (x || y) : (x && y);
            TEST_ASSERT_EQUAL_UINT8(expected, storage[i + 1]);
        }
        TEST_ASSERT_EQUAL_HEX8(0xa5, storage[0]);
        TEST_ASSERT_EQUAL_HEX8(0xa5, storage[size + 1]);
    }
    free(a_storage);
    free(b_storage);
    free(storage);
}

void logical_truth_and_tails(void)
{
    bool a[35], b[35];
    unsigned char guarded[37];
    for (int n = 1; n <= 35; ++n)
    {
        const cmsis_nn_dims dims = {1, 1, 1, n};
        for (int i = 0; i < n; ++i)
        {
            a[i] = (i & 1) != 0;
            b[i] = (i & 2) != 0;
        }
        for (int op = 0; op < 3; ++op)
        {
            memset(guarded, 0xa5, sizeof(guarded));
            bool *out = (bool *)(guarded + 1);
            arm_cmsis_nn_status status = op == 0 ? arm_logical_and_bool(a, &dims, b, &dims, out, &dims)
                : op == 1                        ? arm_logical_or_bool(a, &dims, b, &dims, out, &dims)
                                                 : arm_logical_not_bool(a, out, n);
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
            for (int i = 0; i < n; ++i)
            {
                TEST_ASSERT_EQUAL_UINT8(op == 0 ? a[i] && b[i] : op == 1 ? a[i] || b[i] : !a[i], guarded[i + 1]);
            }
            TEST_ASSERT_EQUAL_HEX8(0xa5, guarded[0]);
            for (int i = n + 1; i < 37; ++i)
            {
                TEST_ASSERT_EQUAL_HEX8(0xa5, guarded[i]);
            }
        }
        check_binary(dims, dims, dims);
    }
}

void logical_broadcast(void)
{
    const cmsis_nn_dims pairs[][2] = {{{1, 1, 1, 1}, {2, 3, 4, 5}},
                                      {{2, 1, 1, 1}, {1, 3, 4, 5}},
                                      {{2, 3, 1, 5}, {1, 1, 4, 1}},
                                      {{1, 3, 4, 1}, {2, 1, 1, 5}},
                                      {{2, 1, 4, 5}, {1, 3, 4, 5}},
                                      {{2, 3, 1, 1}, {1, 1, 4, 5}},
                                      {{1, 4, 256, 256}, {1, 1, 256, 256}}};
    for (unsigned i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i)
    {
        const cmsis_nn_dims a = pairs[i][0], b = pairs[i][1];
        const cmsis_nn_dims out = {
            a.n > b.n ? a.n : b.n, a.h > b.h ? a.h : b.h, a.w > b.w ? a.w : b.w, a.c > b.c ? a.c : b.c};
        check_binary(a, b, out);
        check_binary(b, a, out);
    }
    /* Both values of a broadcast scalar, in either operand position. */
    const cmsis_nn_dims scalar = {1, 1, 1, 1}, dims = {1, 1, 1, 17};
    bool vec[17], out[17];
    for (int i = 0; i < 17; ++i)
    {
        vec[i] = i % 2 != 0;
    }
    for (int value = 0; value < 2; ++value)
    {
        const bool s = value != 0;
        const binary_fn functions[2] = {arm_logical_and_bool, arm_logical_or_bool};
        for (int op = 0; op < 2; ++op)
        {
            for (int swap = 0; swap < 2; ++swap)
            {
                TEST_ASSERT_EQUAL(
                    ARM_CMSIS_NN_SUCCESS,
                    functions[op](
                        swap ? vec : &s, swap ? &dims : &scalar, swap ? &s : vec, swap ? &scalar : &dims, out, &dims));
                for (int i = 0; i < 17; ++i)
                {
                    TEST_ASSERT_EQUAL(op ? (s || vec[i]) : (s && vec[i]), out[i]);
                }
            }
        }
    }
}

void logical_rejects_without_writes(void)
{
    bool input = true;
    unsigned char output = 0xa5;
    const cmsis_nn_dims one = {1, 1, 1, 1};
    const cmsis_nn_dims invalid[] = {
        {0, 1, 1, 1}, {1, -1, 1, 1}, {INT32_MAX, 2, 1, 1}, {1, 46341, 46341, 1}, {1, 1, 2, INT32_MAX}};
    const binary_fn functions[2] = {arm_logical_and_bool, arm_logical_or_bool};
    for (int op = 0; op < 2; ++op)
    {
        for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                              functions[op](&input, &invalid[i], &input, &one, (bool *)&output, &invalid[i]));
            TEST_ASSERT_EQUAL_HEX8(0xa5, output);
        }
        const cmsis_nn_dims two = {1, 1, 1, 2}, three = {1, 1, 1, 3};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &two, &input, &three, (bool *)&output, &three));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &one, &input, &one, (bool *)&output, &two));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](NULL, &one, &input, &one, (bool *)&output, &one));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, NULL, &input, &one, (bool *)&output, &one));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &one, &input, &one, NULL, &one));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &one, &input, &one, (bool *)&output, NULL));
        TEST_ASSERT_EQUAL_HEX8(0xa5, output);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_logical_not_bool(&input, (bool *)&output, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_logical_not_bool(&input, (bool *)&output, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_logical_not_bool(NULL, (bool *)&output, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_logical_not_bool(&input, NULL, 1));
    TEST_ASSERT_EQUAL_HEX8(0xa5, output);
}
