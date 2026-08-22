/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <math.h>
#include <unity.h>

#include "mean_f16_data.h"

static const cmsis_nn_dims mean_f16_input_dims = {2, 3, 1, 5};

static void mean_f16_check(const cmsis_nn_dims *axis_dims, const cmsis_nn_dims *output_dims, const float16_t *expected)
{
    float16_t output[MEAN_F16_INPUT_SIZE] = {0};
    const int32_t output_size = output_dims->n * output_dims->h * output_dims->w * output_dims->c;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_mean_f16(mean_f16_input, &mean_f16_input_dims, axis_dims, output, output_dims));
    for (int32_t i = 0; i < output_size; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-3f, (float)expected[i], (float)output[i]);
    }
}

void mean_f16_axis_c_arm_mean_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 1, 1};
    mean_f16_check(&axis_dims, &output_dims, mean_f16_ref_c);
}

void mean_f16_axis_h_arm_mean_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 5};
    mean_f16_check(&axis_dims, &output_dims, mean_f16_ref_h);
}

void mean_f16_axis_hc_arm_mean_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 1};
    const cmsis_nn_dims output_dims = {2, 1, 1, 1};
    mean_f16_check(&axis_dims, &output_dims, mean_f16_ref_hc);
}

void mean_f16_axis_all_arm_mean_f16(void)
{
    const cmsis_nn_dims axis_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    mean_f16_check(&axis_dims, &output_dims, mean_f16_ref_all);
}

void mean_f16_identity_arm_mean_f16(void)
{
    float16_t output[MEAN_F16_INPUT_SIZE] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_mean_f16(mean_f16_input, &mean_f16_input_dims, &axis_dims, output, &mean_f16_input_dims));
    for (int32_t i = 0; i < MEAN_F16_INPUT_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT((float)mean_f16_input[i], (float)output[i]);
    }
}

void mean_f16_large_sum_arm_mean_f16(void)
{
    float16_t input[17];
    float16_t output = 0.0f;
    const cmsis_nn_dims input_dims = {1, 1, 1, 17};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    for (int32_t i = 0; i < 17; ++i)
    {
        input[i] = (float16_t)65504.0f;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_mean_f16(input, &input_dims, &axis_dims, &output, &output_dims));
    TEST_ASSERT_EQUAL_FLOAT(65504.0f, (float)output);
}

void mean_f16_nan_inf_arm_mean_f16(void)
{
    const float16_t input[6] = {1.0f, (float16_t)INFINITY, 2.0f, (float16_t)NAN, 1.0f, 2.0f};
    float16_t output[2] = {0};
    const cmsis_nn_dims input_dims = {1, 1, 2, 3};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 2, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_mean_f16(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_FLOAT_IS_INF((float)output[0]);
    TEST_ASSERT_FLOAT_IS_NAN((float)output[1]);
}

void mean_f16_arg_error_arm_mean_f16(void)
{
    float16_t output[6] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 1, 1};
    const cmsis_nn_dims invalid_input_dims = {2, 0, 1, 5};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_mean_f16(NULL, &mean_f16_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_mean_f16(mean_f16_input, NULL, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_mean_f16(mean_f16_input, &mean_f16_input_dims, NULL, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_mean_f16(mean_f16_input, &mean_f16_input_dims, &axis_dims, NULL, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_mean_f16(mean_f16_input, &mean_f16_input_dims, &axis_dims, output, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_mean_f16(mean_f16_input, &invalid_input_dims, &axis_dims, output, &output_dims));
}
