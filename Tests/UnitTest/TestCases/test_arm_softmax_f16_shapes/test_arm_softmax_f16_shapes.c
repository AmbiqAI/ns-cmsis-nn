/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <string.h>
#include <unity.h>

#include "softmax_f16_shapes_data.h"

// Row widths that land on either side of the MVE leg's 8-lane boundary.
// test_arm_softmax_f16 carries the upstream dataset and the accuracy claim;
// this suite is about the loop: which arms run for a given trip count, and
// whether the tail picks up where the full passes left off.

// Reference from softmax_settings_flt.py: PyTorch in float32, rounded once to
// float16.
void softmax_f16_shapes_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_softmax_f16(softmax_f16_input, SOFTMAX_F16_NUM_ROWS, SOFTMAX_F16_ROW_SIZE, output));

    for (int i = 0; i < SOFTMAX_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)softmax_f16_output_ref[i], (float)output[i]);
    }
}

// The normalization itself, independent of the reference: a row whose lanes do
// not sum to one is a widen/narrow or reduction fault, not an accuracy one.
void softmax_f16_shapes_rows_sum_to_one_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_softmax_f16(softmax_f16_input, SOFTMAX_F16_NUM_ROWS, SOFTMAX_F16_ROW_SIZE, output));

    for (int r = 0; r < SOFTMAX_F16_NUM_ROWS; ++r)
    {
        float sum = 0.0f;
        for (int c = 0; c < SOFTMAX_F16_ROW_SIZE; ++c)
        {
            sum += (float)output[r * SOFTMAX_F16_ROW_SIZE + c];
        }
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 1.0f, sum);
    }
}

// cols == 16: two full 8-lane passes and no tail, so the unpredicated arm runs
// on its own. The 5-column case above only ever runs the predicated one.
void softmax_f16_shapes_full_vector_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_VEC_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_softmax_f16(softmax_f16_vec_input, SOFTMAX_F16_VEC_NUM_ROWS, SOFTMAX_F16_VEC_ROW_SIZE, output));

    for (int i = 0; i < SOFTMAX_F16_VEC_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)softmax_f16_vec_output_ref[i], (float)output[i]);
    }
}

// cols == 20: two full passes then a 4-lane tail, so both arms run in the
// same call and the tail has to pick up where the full passes left off.
void softmax_f16_shapes_wide_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_WIDE_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_softmax_f16(softmax_f16_wide_input, SOFTMAX_F16_WIDE_NUM_ROWS, SOFTMAX_F16_WIDE_ROW_SIZE, output));

    for (int i = 0; i < SOFTMAX_F16_WIDE_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)softmax_f16_wide_output_ref[i], (float)output[i]);
    }
}

void softmax_f16_shapes_full_vector_rows_sum_to_one_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_WIDE_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_softmax_f16(softmax_f16_wide_input, SOFTMAX_F16_WIDE_NUM_ROWS, SOFTMAX_F16_WIDE_ROW_SIZE, output));

    for (int r = 0; r < SOFTMAX_F16_WIDE_NUM_ROWS; ++r)
    {
        float sum = 0.0f;
        for (int c = 0; c < SOFTMAX_F16_WIDE_ROW_SIZE; ++c)
        {
            sum += (float)output[r * SOFTMAX_F16_WIDE_ROW_SIZE + c];
        }
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 1.0f, sum);
    }
}

void softmax_f16_shapes_arg_error_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_softmax_f16(NULL, SOFTMAX_F16_NUM_ROWS, SOFTMAX_F16_ROW_SIZE, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_softmax_f16(softmax_f16_input, SOFTMAX_F16_NUM_ROWS, SOFTMAX_F16_ROW_SIZE, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_softmax_f16(softmax_f16_input, 0, SOFTMAX_F16_ROW_SIZE, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_softmax_f16(softmax_f16_input, SOFTMAX_F16_NUM_ROWS, 0, output));
}

// Input values never change the status, and a row with no finite maximum (NaN,
// +Inf, only -Inf) leaves the next row exactly as a call on that row alone
// computes it. Bits are set through memcpy so no float16 select is generated.
// Refs #854.
void softmax_f16_shapes_nonfinite_rows_arm_softmax_f16(void)
{
    enum
    {
        cols = 5
    };
    const uint16_t nan = 0x7e00u;
    const uint16_t pos_inf = 0x7c00u;
    const uint16_t neg_inf = 0xfc00u;
    const uint16_t token_rows[3][cols] = {{0x3c00u, nan, 0x4000u, 0x4200u, 0x3800u},
                                          {0x3c00u, pos_inf, 0x4000u, 0x4200u, 0x3800u},
                                          {neg_inf, neg_inf, neg_inf, neg_inf, neg_inf}};
    const uint16_t finite_bits[cols] = {0x3400u, 0xbc00u, 0x4000u, 0x3800u, 0x3c00u};
    float16_t finite[cols];
    float16_t expected[cols];
    memcpy(finite, finite_bits, sizeof(finite));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_softmax_f16(finite, 1, cols, expected));

    for (int t = 0; t < 3; ++t)
    {
        float16_t input[2 * cols];
        float16_t output[2 * cols];
        memcpy(input, token_rows[t], sizeof(token_rows[t]));
        memcpy(input + cols, finite_bits, sizeof(finite_bits));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_softmax_f16(input, 2, cols, output));
        TEST_ASSERT_EQUAL_MEMORY(expected, output + cols, sizeof(expected));
    }

    // The 1x2 specialization takes its own path.
    const uint16_t pair_bits[2] = {nan, 0x3c00u};
    float16_t pair[2];
    float16_t pair_output[2];
    memcpy(pair, pair_bits, sizeof(pair));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_softmax_f16(pair, 1, 2, pair_output));
}
