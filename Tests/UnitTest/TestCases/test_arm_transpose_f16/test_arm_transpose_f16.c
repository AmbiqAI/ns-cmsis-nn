/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

// An output_dims that is not input_dims permuted by perm must be rejected before any write.
// see AmbiqAI/ns-cmsis-nn#443
void transpose_dims_mismatch_arm_transpose_f16(void)
{
    const int32_t buffer_size = 1 * 8 * 10 * 12;
    const float16_t poison = (float16_t)-7.5f;
    static float16_t input_data[1 * 8 * 10 * 12];
    static float16_t output_data[1 * 8 * 10 * 12];

    const cmsis_nn_dims input_dims = {1, 8, 10, 12};
    const cmsis_nn_dims output_dims = {12, 8, 10, 1};
    const cmsis_nn_transpose_params_f16 transpose_params = {4, {3, 2, 1, 0}, ARM_NN_LAYOUT_NHWC};
    const cmsis_nn_context ctx = {NULL, 0};

    for (int32_t i = 0; i < buffer_size; i++)
    {
        input_data[i] = (float16_t)i;
        output_data[i] = poison;
    }

    arm_cmsis_nn_status result =
        arm_transpose_f16(&ctx, &transpose_params, &input_dims, input_data, &output_dims, output_data);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    bool output_untouched = true;
    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}

// A perm with a repeated axis is not a permutation; the dims agree, so only the repeat can reject it.
// see AmbiqAI/ns-cmsis-nn#443
void transpose_perm_duplicate_axis_arm_transpose_f16(void)
{
    const int32_t input_size = 2 * 2 * 3 * 4;
    const int32_t output_size = 2 * 2 * 2 * 3;
    const float16_t poison = (float16_t)-7.5f;
    static float16_t input_data[2 * 2 * 3 * 4];
    static float16_t output_data[2 * 2 * 2 * 3];

    const cmsis_nn_dims input_dims = {2, 2, 3, 4};
    const cmsis_nn_dims output_dims = {2, 2, 2, 3};
    const cmsis_nn_transpose_params_f16 transpose_params = {4, {0, 0, 1, 2}, ARM_NN_LAYOUT_NHWC};
    const cmsis_nn_context ctx = {NULL, 0};

    for (int32_t i = 0; i < input_size; i++)
    {
        input_data[i] = (float16_t)i;
    }
    for (int32_t i = 0; i < output_size; i++)
    {
        output_data[i] = poison;
    }

    arm_cmsis_nn_status result =
        arm_transpose_f16(&ctx, &transpose_params, &input_dims, input_data, &output_dims, output_data);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    bool output_untouched = true;
    for (int32_t i = 0; i < output_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}
