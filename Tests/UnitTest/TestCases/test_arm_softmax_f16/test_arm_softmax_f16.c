/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "softmax_f16_data.h"

// Reference from softmax_settings_flt.py: PyTorch in float32, rounded once to
// float16. The MVE leg reaches the half<->single conversions through
// arm_nn_vexpq_poly_mve_f16, which is why this suite runs on the oldest gated
// GCC as well (AmbiqAI/ns-cmsis-nn#427).
void softmax_f16_arm_softmax_f16(void)
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
void softmax_f16_rows_sum_to_one_arm_softmax_f16(void)
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

// cols == 16: two full 8-lane passes and no tail, so every conversion runs
// with all lanes live. The 5-column case above only ever runs the predicated
// arm, where a wrong destination register can land in a lane nothing reads.
void softmax_f16_full_vector_arm_softmax_f16(void)
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
void softmax_f16_wide_arm_softmax_f16(void)
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

void softmax_f16_full_vector_rows_sum_to_one_arm_softmax_f16(void)
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

void softmax_f16_arg_error_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_softmax_f16(NULL, SOFTMAX_F16_NUM_ROWS, SOFTMAX_F16_ROW_SIZE, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_softmax_f16(softmax_f16_input, SOFTMAX_F16_NUM_ROWS, SOFTMAX_F16_ROW_SIZE, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_softmax_f16(softmax_f16_input, 0, SOFTMAX_F16_ROW_SIZE, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_softmax_f16(softmax_f16_input, SOFTMAX_F16_NUM_ROWS, 0, output));
}
