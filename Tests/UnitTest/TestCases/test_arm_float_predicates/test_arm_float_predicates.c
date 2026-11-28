/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */
#include <Internal/arm_nn_fp_control.h>
#include "arm_nnfunctions.h"
#include "float_predicates_data.h"
#include "unity.h"
#include <limits.h>
#include <string.h>

static void predicates_reference(void)
{
    const cmsis_nn_dims ad = {2, 1, 1, 9}, bd = {1, 3, 1, 1}, md = {2, 3, 1, 9};
    const int32_t shape[] = {2, 3, 2, 9}, cs[] = {27, 9, 0, 1}, xs[] = {0, 2, 1, 0}, ys[] = {0, 0, 0, 1};
    const cmsis_nn_select_v2_params p = {4, shape, cs, xs, ys};
    float32_t a[18], b[3], x[6], y[9], result[110];
    bool mask[56] = {true};
    mask[55] = true;
    memset(result, 0x55, sizeof(result));
    memcpy(a, ref_a, sizeof(a));
    memcpy(b, ref_b, sizeof(b));
    memcpy(x, ref_x, sizeof(x));
    memcpy(y, ref_y, sizeof(y));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_greater_f32(a, &ad, b, &bd, mask + 1, &md));
    TEST_ASSERT_EQUAL_UINT8_ARRAY(ref_mask, mask + 1, 54);
    TEST_ASSERT_TRUE(mask[0] && mask[55]);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_select_v2_f32(mask + 1, x, y, &p, result + 1));
    TEST_ASSERT_EQUAL_MEMORY(ref_selected, result + 1, sizeof(ref_selected));
    const uint32_t guard = 0x55555555;
    TEST_ASSERT_EQUAL_MEMORY(&guard, result, 4);
    TEST_ASSERT_EQUAL_MEMORY(&guard, result + 109, 4);
}

/* Restore the caller's control word before Unity can return on a failure. */
static arm_cmsis_nn_status compare_with_flush(const float32_t *a, const cmsis_nn_dims *ad,
                                             const float32_t *b, const cmsis_nn_dims *bd,
                                             bool *out, const cmsis_nn_dims *od, bool flush)
{
#if defined(ARM_NN_FP_CONTROL_AVAILABLE)
    const arm_nn_fp_control_t saved = arm_nn_get_fp_control();
    const arm_nn_fp_control_t fz = 1u << 24;
    arm_nn_set_fp_control((saved & ~fz) | (flush ? fz : 0));
#else
    (void)flush;
#endif
    const arm_cmsis_nn_status status = arm_greater_f32(a, ad, b, bd, out, od);
#if defined(ARM_NN_FP_CONTROL_AVAILABLE)
    arm_nn_set_fp_control(saved);
#endif
    return status;
}

static void predicates_order(void)
{
    /* Ordered values, then signaling and quiet NaNs of either sign. */
    const uint32_t bits[] = {0xff800000,
                             0xbf800000,
                             0x80000001,
                             0x80000000,
                             0,
                             1,
                             0x3f800000,
                             0x7f800000,
                             0x7f800001,
                             0x7fc12345,
                             0xff800001,
                             0xffc12345};
    const int rank[] = {0, 1, 2, 3, 3, 4, 5, 6};
    const cmsis_nn_dims many = {1, 1, 1, 12}, one = {1, 1, 1, 1};
    float32_t data[12], repeated[12];
    bool out[12];
    memcpy(data, bits, sizeof(bits));
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 12; j++)
            memcpy(repeated + j, data + i, 4);
        for (int mode = 0; mode < 3; mode++)
        {
            const bool left = mode == 1;
            const float32_t *a = left ? data + i : data;
            const float32_t *b = left ? data : (mode == 0 ? data + i : repeated);
            for (int flush = 0; flush <= 1; ++flush)
            {
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                                  compare_with_flush(a, left ? &one : &many, b, mode == 0 ? &one : &many, out, &many, flush));
                for (int j = 0; j < 12; j++)
                    TEST_ASSERT_EQUAL(i < 8 && j < 8 && (left ? rank[i] > rank[j] : rank[j] > rank[i]), out[j]);
            }
        }
    }
}

/* Keep the only exceptional comparison in the last active lane of a partial vector. */
static void predicates_tail(void)
{
    const cmsis_nn_dims five = {1, 1, 1, 5};
    const uint32_t cases[][3] = {
        {0x00000001, 0, 1},
        {0, 0x80000001, 1},
        {0x007fffff, 1, 1},
        {0x00800000, 0x007fffff, 1},
        {0, 0x80000000, 0},
        {0x7f800001, 0xff800000, 0},
        {0x7f800000, 0xff800001, 0},
        {0x7f800000, 0x7f7fffff, 1},
    };
    float32_t a[5] = {1, 2, 3, 4, 0}, b[5] = {0, 1, 2, 3, 0};
    bool mask[5];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        memcpy(a + 4, cases[i], 4);
        memcpy(b + 4, cases[i] + 1, 4);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_greater_f32(a, &five, b, &five, mask, &five));
        for (int j = 0; j < 4; j++)
            TEST_ASSERT_TRUE(mask[j]);
        TEST_ASSERT_EQUAL(cases[i][2], mask[4]);
    }
}

/* Consecutive groups and a scalar tail, with a byte-offset output mask. */
static void predicates_packed_mask(void)
{
    const cmsis_nn_dims dims = {1, 1, 1, 9};
    const int32_t length = 9, stride = 1;
    const cmsis_nn_select_v2_params params = {1, &length, &stride, &stride, &stride};
    const uint32_t left[] = {1, 0x80000001, 0, 0x80000000, 0x7f800000, 0x7f800001, 0xff800000, 0x807fffff, 1};
    const uint32_t right[] = {0, 0x80000000, 0x80000000, 0, 0x7f800000, 0xffc12345, 0xff800000, 0x80800000, 0};
    const bool expected[] = {true, false, false, false, false, false, false, true, true};
    float32_t a[9], b[9], selected[9];
    memcpy(a, left, sizeof(a));
    memcpy(b, right, sizeof(b));

    for (int flush = 0; flush <= 1; ++flush)
    {
        bool mask[11] = {true};
        mask[10] = true;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, compare_with_flush(a, &dims, b, &dims, mask + 1, &dims, flush));
        TEST_ASSERT_TRUE(mask[0] && mask[10]);
        TEST_ASSERT_EQUAL_MEMORY(expected, mask + 1, sizeof(expected));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_select_v2_f32(mask + 1, a, b, &params, selected));
        for (int i = 0; i < length; ++i)
        {
            TEST_ASSERT_EQUAL_MEMORY((expected[i] ? left : right) + i, selected + i, sizeof(uint32_t));
        }
    }

    /* Two equal trailing groups (including a NaN) follow a true first lane. */
    memcpy(b + 1, a + 1, 8 * sizeof(*a));
    bool mask[11];
    for (int i = 0; i < 11; ++i) mask[i] = true;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, compare_with_flush(a, &dims, b, &dims, mask + 1, &dims, true));
    TEST_ASSERT_TRUE(mask[0] && mask[1] && mask[10]);
    for (int i = 2; i < 10; ++i) TEST_ASSERT_FALSE(mask[i]);
}

static void predicates_strides(void)
{
    const int32_t shape[8] = {1, 1, 1, 1, 1, 1, 2, 3}, cs[8] = {0}, xs[8] = {0, 0, 0, 0, 0, 0, 1, 2}, ys[8] = {0};
    const cmsis_nn_select_v2_params p = {8, shape, cs, xs, ys};
    const uint32_t words[] = {0x7f800001, 0x80000000, 1, 0xffc12345, 0xff800000, 0x80000001};
    const uint32_t expected[] = {words[0], words[2], words[4], words[1], words[3], words[5]};
    float32_t x[6], out[6];
    memcpy(x, words, sizeof(x));
    bool choose = true;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_select_v2_f32(&choose, x, x + 1, &p, out));
    TEST_ASSERT_EQUAL_MEMORY(expected, out, sizeof(expected));
    choose = false;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_select_v2_f32(&choose, x, x + 1, &p, out));
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_EQUAL_MEMORY(words + 1, out + i, 4);
}

static void predicates_reject(void)
{
    const cmsis_nn_dims one = {1, 1, 1, 1}, two = {1, 1, 1, 2}, huge = {1, 1, 1, INT32_MAX};
    const int32_t good = 1, bad = -1, overflow = INT32_MAX;
    float32_t x = 0, out = 1;
    bool mask = false;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_greater_f32(&x, &one, &x, &one, &mask, &two));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_greater_f32(&x, &huge, &x, &one, &mask, &huge));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_greater_f32(NULL, &one, &x, &one, &mask, &one));
    cmsis_nn_select_v2_params p = {9, &good, &good, &good, &good};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_select_v2_f32(&mask, &x, &x, &p, &out));
    p.rank = 1;
    p.x_strides = &bad;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_select_v2_f32(&mask, &x, &x, &p, &out));
    p.x_strides = &good;
    p.output_shape = &overflow;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_select_v2_f32(&mask, &x, &x, &p, &out));
    TEST_ASSERT_FALSE(mask);
    TEST_ASSERT_TRUE(out == 1);
}
