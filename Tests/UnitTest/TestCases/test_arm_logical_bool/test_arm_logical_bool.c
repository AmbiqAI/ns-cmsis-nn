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

static bool binary_expected(bool a, bool b, int op)
{
    switch (op)
    {
    case 0: return a && b;
    case 1: return a || b;
    case 2: return a == b;
    default: return a != b;
    }
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
    const binary_fn functions[4] = {arm_logical_and_bool, arm_logical_or_bool, arm_equal_bool, arm_not_equal_bool};
    for (int op = 0; op < 4; ++op)
    {
        memset(storage, 0xa5, (size_t)size + 2);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, functions[op](a, &a_dims, b, &b_dims, out, &out_dims));
        for (int i = 0; i < size; ++i)
        {
            const bool x = a[oracle_index(i, out_dims, a_dims)];
            const bool y = b[oracle_index(i, out_dims, b_dims)];
            const unsigned char expected = binary_expected(x, y, op);
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
                                      {{1, 4, 17, 19}, {1, 1, 17, 19}}};
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
        const binary_fn functions[4] = {arm_logical_and_bool, arm_logical_or_bool, arm_equal_bool, arm_not_equal_bool};
        for (int op = 0; op < 4; ++op)
        {
            for (int swap = 0; swap < 2; ++swap)
            {
                TEST_ASSERT_EQUAL(
                    ARM_CMSIS_NN_SUCCESS,
                    functions[op](
                        swap ? vec : &s, swap ? &dims : &scalar, swap ? &s : vec, swap ? &scalar : &dims, out, &dims));
                for (int i = 0; i < 17; ++i)
                {
                    TEST_ASSERT_EQUAL(binary_expected(s, vec[i], op), out[i]);
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
    const binary_fn functions[4] = {arm_logical_and_bool, arm_logical_or_bool, arm_equal_bool, arm_not_equal_bool};
    for (int op = 0; op < 4; ++op)
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

typedef arm_cmsis_nn_status (*reduce_fn)(const bool *, const cmsis_nn_dims *, const cmsis_nn_dims *, bool *, const cmsis_nn_dims *);

void logical_reductions(void)
{
    const cmsis_nn_dims shapes[] = {{2,3,5,17}, {2,1,3,33}, {1,1,1,1}, {2,0,3,17},
                                    {0,3,5,1}, {2,3,0,17}, {2,3,5,0}};
    bool input_store[512], expected[512];
    unsigned char output_store[514];
    bool *input = input_store + 1;
    const reduce_fn functions[2] = {arm_reduce_any_bool, arm_reduce_all_bool};
    for (unsigned shape = 0; shape < sizeof(shapes)/sizeof(shapes[0]); ++shape)
    {
        const cmsis_nn_dims d = shapes[shape];
        const int length = count(d);
        for (int mask = 0; mask < 16; ++mask)
        {
            const cmsis_nn_dims axis = {(mask >> 3)&1, (mask >> 2)&1, (mask >> 1)&1, mask&1};
            const cmsis_nn_dims out = {axis.n ? 1 : d.n, axis.h ? 1 : d.h, axis.w ? 1 : d.w, axis.c ? 1 : d.c};
            const int size = count(out);
            for (int op = 0; op < 2; ++op)
            {
                for (int pattern = 0; pattern < 4; ++pattern)
                {
                    for (int i = 0; i < length; ++i)
                    {
                        input[i] = pattern == 0 ? false : pattern == 1 ? true :
                            pattern == 2 ? (i % 7 == 0) : ((i != length - 1) == (op != 0));
                    }
                    for (int i = 0; i < size; ++i) expected[i] = op != 0;
                    /* Independent flat-index oracle; no block collapsing or shared traversal. */
                    for (int i = 0; i < length; ++i)
                    {
                        int q = i;
                        const int cc = q % d.c; q /= d.c;
                        const int ww = q % d.w; q /= d.w;
                        const int hh = q % d.h; q /= d.h;
                        const int oi = (((axis.n ? 0 : q)*out.h + (axis.h ? 0 : hh))*out.w +
                                        (axis.w ? 0 : ww))*out.c + (axis.c ? 0 : cc);
                        expected[oi] = op ? (expected[oi] && input[i]) : (expected[oi] || input[i]);
                    }
                    memset(output_store, 0xa5, sizeof(output_store));
                    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, functions[op](input, &d, &axis, (bool *)(output_store+1), &out));
                    for (int i = 0; i < size; ++i) TEST_ASSERT_EQUAL_UINT8(expected[i], output_store[i+1]);
                    TEST_ASSERT_EQUAL_HEX8(0xa5, output_store[0]);
                    for (unsigned i = (unsigned)size+1; i < sizeof(output_store); ++i) TEST_ASSERT_EQUAL_HEX8(0xa5, output_store[i]);
                }
            }
        }
    }
    /* Every SIMD tail with the only decisive element last, including ALL's inactive lanes. */
    for (int length = 1; length <= 35; ++length)
    {
        const cmsis_nn_dims d = {1,1,1,length}, axis = {0,0,0,1}, out = {1,1,1,1};
        for (int op = 0; op < 2; ++op)
        {
            memset(input, op, (size_t)length);
            bool result = !op;
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, functions[op](input, &d, &axis, &result, &out));
            TEST_ASSERT_EQUAL(op, result);
            input[length-1] = !op;
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, functions[op](input, &d, &axis, &result, &out));
            TEST_ASSERT_EQUAL(!op, result);
        }
    }
}

void logical_reduction_rejects(void)
{
    const reduce_fn functions[2] = {arm_reduce_any_bool, arm_reduce_all_bool};
    const cmsis_nn_dims one = {1,1,1,1}, all = {1,1,1,1}, none = {0,0,0,0};
    const cmsis_nn_dims bad_input[] = {{-1,1,1,1}, {1,2,INT32_MAX,1}, {1,46341,46341,1}};
    const cmsis_nn_dims bad_axis[] = {{2,0,0,0}, {0,-1,0,0}, {0,0,2,0}, {0,0,0,2}};
    bool input = true;
    unsigned char output = 0xa5;
    for (int op = 0; op < 2; ++op)
    {
        for (unsigned i = 0; i < sizeof(bad_input)/sizeof(bad_input[0]); ++i)
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &bad_input[i], &all, (bool *)&output, &one));
        for (unsigned i = 0; i < sizeof(bad_axis)/sizeof(bad_axis[0]); ++i)
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &one, &bad_axis[i], (bool *)&output, &one));
        const cmsis_nn_dims two = {1,1,1,2}, empty = {0,INT32_MAX,2,1}, empty_axis = {1,0,0,0}, huge = {1,INT32_MAX,2,1};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &one, &none, (bool *)&output, &two));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &empty, &empty_axis, (bool *)&output, &huge));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](NULL, &one, &all, (bool *)&output, &one));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, NULL, &all, (bool *)&output, &one));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &one, NULL, (bool *)&output, &one));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &one, &all, NULL, &one));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, functions[op](&input, &one, &all, (bool *)&output, NULL));
        TEST_ASSERT_EQUAL_HEX8(0xa5, output);
        /* A zero extent avoids spurious intermediate product overflow. */
        const cmsis_nn_dims late_zero = {INT32_MAX,INT32_MAX,0,1};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, functions[op](&input, &late_zero, &all, (bool *)&output, &one));
        TEST_ASSERT_EQUAL_UINT8(op, output);
        output = 0xa5;
    }
}
