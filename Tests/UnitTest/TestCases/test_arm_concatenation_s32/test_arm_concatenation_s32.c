/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "../TestData/concat_axis_c_ten_inputs_s32/test_data.h"
#include "../TestData/concat_axis_c_two_inputs_s32/test_data.h"
#include "../TestData/concat_axis_w_two_inputs_s32/test_data.h"
#include "../Utils/validate.h"
#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

void concat_axis_c_two_inputs_arm_concatenation_s32(void)
{
    const int32_t *input_ptrs[CONCAT_AXIS_C_TWO_INPUTS_S32_INPUTS_COUNT] = {
        concat_axis_c_two_inputs_s32_input_tensor_1, concat_axis_c_two_inputs_s32_input_tensor_2};
    const int32_t input_concat_dims[CONCAT_AXIS_C_TWO_INPUTS_S32_INPUTS_COUNT] =
        CONCAT_AXIS_C_TWO_INPUTS_S32_INPUT_CONCAT_DIMS;
    const int32_t output_shape[CONCAT_AXIS_C_TWO_INPUTS_S32_OUTPUT_DIMS] = CONCAT_AXIS_C_TWO_INPUTS_S32_OUTPUT_SHAPE;
    int32_t output_ptr[CONCAT_AXIS_C_TWO_INPUTS_S32_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_concatenation_s32(input_ptrs,
                                                             CONCAT_AXIS_C_TWO_INPUTS_S32_INPUTS_COUNT,
                                                             input_concat_dims,
                                                             CONCAT_AXIS_C_TWO_INPUTS_S32_AXIS,
                                                             output_ptr,
                                                             CONCAT_AXIS_C_TWO_INPUTS_S32_OUTPUT_DIMS,
                                                             output_shape);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    validate_s32(output_ptr, concat_axis_c_two_inputs_s32_output, CONCAT_AXIS_C_TWO_INPUTS_S32_OUTPUT_SIZE);
}

void concat_axis_w_two_inputs_arm_concatenation_s32(void)
{
    const int32_t *input_ptrs[CONCAT_AXIS_W_TWO_INPUTS_S32_INPUTS_COUNT] = {
        concat_axis_w_two_inputs_s32_input_tensor_1, concat_axis_w_two_inputs_s32_input_tensor_2};
    const int32_t input_concat_dims[CONCAT_AXIS_W_TWO_INPUTS_S32_INPUTS_COUNT] =
        CONCAT_AXIS_W_TWO_INPUTS_S32_INPUT_CONCAT_DIMS;
    const int32_t output_shape[CONCAT_AXIS_W_TWO_INPUTS_S32_OUTPUT_DIMS] = CONCAT_AXIS_W_TWO_INPUTS_S32_OUTPUT_SHAPE;
    int32_t output_ptr[CONCAT_AXIS_W_TWO_INPUTS_S32_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_concatenation_s32(input_ptrs,
                                                             CONCAT_AXIS_W_TWO_INPUTS_S32_INPUTS_COUNT,
                                                             input_concat_dims,
                                                             CONCAT_AXIS_W_TWO_INPUTS_S32_AXIS,
                                                             output_ptr,
                                                             CONCAT_AXIS_W_TWO_INPUTS_S32_OUTPUT_DIMS,
                                                             output_shape);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    validate_s32(output_ptr, concat_axis_w_two_inputs_s32_output, CONCAT_AXIS_W_TWO_INPUTS_S32_OUTPUT_SIZE);
}

void concat_axis_c_ten_inputs_arm_concatenation_s32(void)
{
    const int32_t *input_ptrs[CONCAT_AXIS_C_TEN_INPUTS_S32_INPUTS_COUNT] = {
        concat_axis_c_ten_inputs_s32_input_tensor_1, concat_axis_c_ten_inputs_s32_input_tensor_2};
    const int32_t input_concat_dims[CONCAT_AXIS_C_TEN_INPUTS_S32_INPUTS_COUNT] =
        CONCAT_AXIS_C_TEN_INPUTS_S32_INPUT_CONCAT_DIMS;
    const int32_t output_shape[CONCAT_AXIS_C_TEN_INPUTS_S32_OUTPUT_DIMS] = CONCAT_AXIS_C_TEN_INPUTS_S32_OUTPUT_SHAPE;
    int32_t output_ptr[CONCAT_AXIS_C_TEN_INPUTS_S32_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_concatenation_s32(input_ptrs,
                                                             CONCAT_AXIS_C_TEN_INPUTS_S32_INPUTS_COUNT,
                                                             input_concat_dims,
                                                             CONCAT_AXIS_C_TEN_INPUTS_S32_AXIS,
                                                             output_ptr,
                                                             CONCAT_AXIS_C_TEN_INPUTS_S32_OUTPUT_DIMS,
                                                             output_shape);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    validate_s32(output_ptr, concat_axis_c_ten_inputs_s32_output, CONCAT_AXIS_C_TEN_INPUTS_S32_OUTPUT_SIZE);
}

/* More than ten inputs (11, 12 and 16, each 1 to 3 wide on the concatenation axis), joined on the channel axis and on
 * the height axis, so that both a single outer step and several outer steps run. The expected output is indexed per
 * output element, and two guard words after the output must stay untouched. */
void concat_many_inputs_arm_concatenation_s32(void)
{
    enum
    {
        max_inputs = 16,
        max_input_size = 24,
        max_output_size = 248,
        guard = 0x5A5A5A5A
    };
    static int32_t inputs[max_inputs][max_input_size];
    static int32_t output[max_output_size + 2];
    const int32_t input_counts[] = {11, 12, 16};
    const int32_t axes[] = {3, 1};

    for (size_t a = 0; a < sizeof(axes) / sizeof(axes[0]); a++)
    {
        for (size_t n = 0; n < sizeof(input_counts) / sizeof(input_counts[0]); n++)
        {
            const int32_t axis = axes[a];
            const int32_t inputs_count = input_counts[n];
            const int32_t *input_ptrs[max_inputs];
            int32_t input_concat_dims[max_inputs];
            int32_t concat_dim_sum = 0;
            for (int32_t i = 0; i < inputs_count; i++)
            {
                input_concat_dims[i] = 1 + i % 3;
                concat_dim_sum += input_concat_dims[i];
            }
            int32_t output_shape[4] = {2, 2, 2, 2};
            output_shape[axis] = concat_dim_sum;
            int32_t outer_size = 1;
            for (int32_t d = 0; d < axis; d++)
            {
                outer_size *= output_shape[d];
            }
            int32_t inner_size = 1;
            for (int32_t d = axis + 1; d < 4; d++)
            {
                inner_size *= output_shape[d];
            }
            const int32_t output_size = outer_size * concat_dim_sum * inner_size;
            TEST_ASSERT_TRUE(output_size <= max_output_size);

            for (int32_t i = 0; i < inputs_count; i++)
            {
                const int32_t input_size = outer_size * input_concat_dims[i] * inner_size;
                TEST_ASSERT_TRUE(input_size <= max_input_size);
                for (int32_t j = 0; j < input_size; j++)
                {
                    inputs[i][j] = (i + 1) * 1000 + j;
                }
                input_ptrs[i] = inputs[i];
            }
            for (int32_t j = 0; j < max_output_size + 2; j++)
            {
                output[j] = guard;
            }

            const arm_cmsis_nn_status result =
                arm_concatenation_s32(input_ptrs, inputs_count, input_concat_dims, axis, output, 4, output_shape);
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);

            for (int32_t o = 0; o < outer_size; o++)
            {
                int32_t start = 0;
                for (int32_t i = 0; i < inputs_count; i++)
                {
                    for (int32_t c = 0; c < input_concat_dims[i]; c++)
                    {
                        for (int32_t r = 0; r < inner_size; r++)
                        {
                            const int32_t out_index = (o * concat_dim_sum + start + c) * inner_size + r;
                            const int32_t in_index = (o * input_concat_dims[i] + c) * inner_size + r;
                            TEST_ASSERT_EQUAL_INT32(inputs[i][in_index], output[out_index]);
                        }
                    }
                    start += input_concat_dims[i];
                }
            }
            TEST_ASSERT_EQUAL_INT32(guard, output[output_size]);
            TEST_ASSERT_EQUAL_INT32(guard, output[output_size + 1]);
        }
    }
}

/* Twelve inputs whose last input is invalid: a zero concatenation dimension, a dimension sum that does not match the
 * output, and a copy size above UINT32_MAX (65536 * 65535 words per step, doubled for the last input). Each call must
 * return ARM_CMSIS_NN_ARG_ERROR before writing any output. */
void concat_many_inputs_rejected_arm_concatenation_s32(void)
{
    enum
    {
        inputs_count = 12,
        guard = 0x5A5A5A5A
    };
    static const int32_t input[4] = {1, 2, 3, 4};
    int32_t output[4] = {guard, guard, guard, guard};
    const int32_t *input_ptrs[inputs_count];
    int32_t input_concat_dims[inputs_count];
    for (int32_t i = 0; i < inputs_count; i++)
    {
        input_ptrs[i] = input;
        input_concat_dims[i] = 1;
    }

    input_concat_dims[inputs_count - 1] = 0;
    const int32_t zero_dim_shape[4] = {1, 1, 1, inputs_count - 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_concatenation_s32(input_ptrs, inputs_count, input_concat_dims, 3, output, 4, zero_dim_shape));

    input_concat_dims[inputs_count - 1] = 1;
    const int32_t wrong_sum_shape[4] = {1, 1, 1, inputs_count + 1};
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_concatenation_s32(input_ptrs, inputs_count, input_concat_dims, 3, output, 4, wrong_sum_shape));

    input_concat_dims[inputs_count - 1] = 2;
    const int32_t oversized_shape[4] = {1, inputs_count + 1, 65536, 65535};
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_concatenation_s32(input_ptrs, inputs_count, input_concat_dims, 1, output, 4, oversized_shape));

    for (int32_t i = 0; i < 4; i++)
    {
        TEST_ASSERT_EQUAL_INT32(guard, output[i]);
    }
}
