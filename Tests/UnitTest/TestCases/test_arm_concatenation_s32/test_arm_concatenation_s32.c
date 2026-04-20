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
