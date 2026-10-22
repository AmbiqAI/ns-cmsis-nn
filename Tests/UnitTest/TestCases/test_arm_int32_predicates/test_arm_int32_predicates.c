/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */
#include "arm_nnfunctions.h"
#include "unity.h"
#include <limits.h>
#include <string.h>

typedef arm_cmsis_nn_status (*compare_fn)(const int32_t *,
                                          const cmsis_nn_dims *,
                                          const int32_t *,
                                          const cmsis_nn_dims *,
                                          bool *,
                                          const cmsis_nn_dims *);
static const compare_fn entries[] =
    {arm_equal_s32, arm_not_equal_s32, arm_greater_s32, arm_greater_equal_s32, arm_less_s32, arm_less_equal_s32};
static const int32_t values[] = {INT32_MIN, INT32_MAX, 0, -1, 1, INT32_MIN + 1, INT32_MAX - 1, -12345, 67890};
static bool relation(int op, int32_t a, int32_t b)
{
    switch (op)
    {
    case 0:
        return a == b;
    case 1:
        return a != b;
    case 2:
        return a > b;
    case 3:
        return a >= b;
    case 4:
        return a < b;
    default:
        return a <= b;
    }
}
static int32_t index4(int32_t flat, const int32_t *shape, const int32_t *out)
{
    int32_t index = 0, stride = 1;
    for (int d = 3; d >= 0; --d)
    {
        index += (shape[d] == 1 ? 0 : flat % out[d]) * stride;
        stride *= shape[d];
        flat /= out[d];
    }
    return index;
}
static void comparisons(void)
{
    int32_t a[60], b[60];
    bool guarded[62];
    for (int i = 0; i < 60; ++i)
    {
        a[i] = values[i % 9];
        b[i] = values[(i * 7 + 1) % 9];
    }
    for (int ma = 0; ma < 16; ++ma)
        for (int mb = 0; mb < 16; ++mb)
        {
            const int32_t full[4] = {2, 2, 3, 5};
            int32_t da[4], db[4], dout[4], count = 1;
            for (int d = 0; d < 4; ++d)
            {
                da[d] = (ma & (1 << d)) ? full[d] : 1;
                db[d] = (mb & (1 << d)) ? full[d] : 1;
                dout[d] = da[d] > db[d] ? da[d] : db[d];
                count *= dout[d];
            }
            cmsis_nn_dims ad = {da[0], da[1], da[2], da[3]}, bd = {db[0], db[1], db[2], db[3]};
            cmsis_nn_dims od = {dout[0], dout[1], dout[2], dout[3]};
            for (int op = 0; op < 6; ++op)
            {
                memset(guarded, 0xa5, sizeof(guarded));
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, entries[op](a, &ad, b, &bd, guarded + 1, &od));
                TEST_ASSERT_EQUAL_HEX8(0xa5, ((uint8_t *)guarded)[0]);
                TEST_ASSERT_EQUAL_HEX8(0xa5, ((uint8_t *)guarded)[count + 1]);
                for (int i = 0; i < count; ++i)
                    TEST_ASSERT_EQUAL(relation(op, a[index4(i, da, dout)], b[index4(i, db, dout)]), guarded[i + 1]);
            }
        }
}
static void selections(void)
{
    bool cond[1024];
    int32_t x[1024], y[1024], output[258];
    for (int i = 0; i < 1024; ++i)
    {
        cond[i] = i % 3 == 1;
        x[i] = values[i % 9];
        y[i] = values[(i * 5 + 2) % 9];
    }
    for (int rank = 1; rank <= 8; ++rank)
        for (int mask = 0; mask < 64; ++mask)
        {
            int32_t shape[8], cs[8], xs[8], ys[8], count = 1;
            int32_t *strides[3] = {cs, xs, ys};
            for (int d = 0; d < rank; ++d)
            {
                shape[d] = rank <= 3 ? (d == rank - 1 ? 5 : 2) : 2;
                count *= shape[d];
            }
            for (int j = 0; j < 3; ++j)
            {
                int32_t stride = 1;
                for (int d = rank - 1; d >= 0; --d)
                {
                    strides[j][d] = (mask & (1 << (j * 2 + d % 2))) ? stride : 0;
                    if (strides[j][d])
                        stride *= shape[d];
                }
            }
            cmsis_nn_select_v2_params params = {rank, shape, cs, xs, ys};
            memset(output, 0xa5, sizeof(output));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_select_v2_s32(cond, x, y, &params, output + 1));
            TEST_ASSERT_EQUAL_HEX32(0xa5a5a5a5u, output[0]);
            TEST_ASSERT_EQUAL_HEX32(0xa5a5a5a5u, output[count + 1]);
            for (int i = 0; i < count; ++i)
            {
                int32_t offsets[3] = {0}, flat = i;
                for (int d = rank - 1; d >= 0; --d)
                {
                    int32_t coord = flat % shape[d];
                    flat /= shape[d];
                    for (int j = 0; j < 3; ++j)
                        offsets[j] += coord * strides[j][d];
                }
                TEST_ASSERT_EQUAL_INT32(cond[offsets[0]] ? x[offsets[1]] : y[offsets[2]], output[i + 1]);
            }
        }
    int32_t shape[] = {2, 5}, cs[] = {15, 2}, xs[] = {17, 3}, ys[] = {20, 4};
    cmsis_nn_select_v2_params params = {2, shape, cs, xs, ys};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_select_v2_s32(cond, x, y, &params, output));
    for (int i = 0; i < 10; ++i)
        TEST_ASSERT_EQUAL_INT32(cond[i / 5 * 15 + i % 5 * 2] ? x[i / 5 * 17 + i % 5 * 3] : y[i / 5 * 20 + i % 5 * 4],
                                output[i]);
}
static void rejected_arguments(void)
{
    int32_t x = INT32_MIN, y = INT32_MAX, out = 123;
    bool condition = true, result = true;
    cmsis_nn_dims one = {1, 1, 1, 1}, bad = {1, 1, 1, 2}, huge = {1, 1, 1, INT32_MAX}, zero = {0, 1, 1, 1};
    for (int op = 0; op < 6; ++op)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, entries[op](&x, &one, &y, &one, &result, &bad));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, entries[op](&x, &huge, &y, &huge, &result, &huge));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, entries[op](&x, &zero, &y, &one, &result, &one));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, entries[op](&x, NULL, &y, &one, &result, &one));
        TEST_ASSERT_TRUE(result);
    }
    int32_t shape[2] = {2, 3}, strides[2] = {3, 1};
    cmsis_nn_select_v2_params p = {2, shape, strides, strides, strides};
    for (int which = 0; which < 5; ++which)
    {
        shape[0] = 2;
        strides[0] = 3;
        p.rank = 2;
        if (which == 0)
            shape[0] = 0;
        if (which == 1)
            shape[0] = INT32_MAX;
        if (which == 2)
            strides[0] = -1;
        if (which == 3)
            strides[0] = INT32_MAX;
        if (which == 4)
            p.rank = 9;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_select_v2_s32(&condition, &x, &y, &p, &out));
        TEST_ASSERT_EQUAL_INT32(123, out);
    }
    p.rank = 2;
    p.output_shape = NULL;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_select_v2_s32(&condition, &x, &y, &p, &out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_select_v2_s32(&condition, &x, &y, NULL, &out));
    TEST_ASSERT_EQUAL_INT32(123, out);
}
