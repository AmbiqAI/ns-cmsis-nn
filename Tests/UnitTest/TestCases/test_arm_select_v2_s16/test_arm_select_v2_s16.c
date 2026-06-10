/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/select_v2_broadcast_s16/test_data.h"
#include "../TestData/select_v2_row_mask_s16/test_data.h"
#include "../TestData/select_v2_scalar_condition_true_s16/test_data.h"
#include "../Utils/validate.h"

void select_v2_broadcast_s16_arm_select_v2_s16(void)
{
    const int32_t output_shape[] = SELECT_V2_BROADCAST_S16_OUTPUT_SHAPE;
    const int32_t cond_strides[] = SELECT_V2_BROADCAST_S16_COND_STRIDES;
    const int32_t x_strides[] = SELECT_V2_BROADCAST_S16_X_STRIDES;
    const int32_t y_strides[] = SELECT_V2_BROADCAST_S16_Y_STRIDES;
    const cmsis_nn_select_v2_params params = {
        .rank = SELECT_V2_BROADCAST_S16_RANK,
        .output_shape = output_shape,
        .cond_strides = cond_strides,
        .x_strides = x_strides,
        .y_strides = y_strides,
    };
    int16_t output[SELECT_V2_BROADCAST_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_select_v2_s16(select_v2_broadcast_s16_condition_input_tensor,
                                                         select_v2_broadcast_s16_x_input_tensor,
                                                         select_v2_broadcast_s16_y_input_tensor,
                                                         &params,
                                                         output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, select_v2_broadcast_s16_output, SELECT_V2_BROADCAST_S16_OUTPUT_SIZE));
}

void select_v2_scalar_condition_true_s16_arm_select_v2_s16(void)
{
    const int32_t output_shape[] = SELECT_V2_SCALAR_CONDITION_TRUE_S16_OUTPUT_SHAPE;
    const int32_t cond_strides[] = SELECT_V2_SCALAR_CONDITION_TRUE_S16_COND_STRIDES;
    const int32_t x_strides[] = SELECT_V2_SCALAR_CONDITION_TRUE_S16_X_STRIDES;
    const int32_t y_strides[] = SELECT_V2_SCALAR_CONDITION_TRUE_S16_Y_STRIDES;
    const cmsis_nn_select_v2_params params = {
        .rank = SELECT_V2_SCALAR_CONDITION_TRUE_S16_RANK,
        .output_shape = output_shape,
        .cond_strides = cond_strides,
        .x_strides = x_strides,
        .y_strides = y_strides,
    };
    int16_t output[SELECT_V2_SCALAR_CONDITION_TRUE_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_select_v2_s16(select_v2_scalar_condition_true_s16_condition_input_tensor,
                                                         select_v2_scalar_condition_true_s16_x_input_tensor,
                                                         select_v2_scalar_condition_true_s16_y_input_tensor,
                                                         &params,
                                                         output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                                  select_v2_scalar_condition_true_s16_output,
                                  SELECT_V2_SCALAR_CONDITION_TRUE_S16_OUTPUT_SIZE));
}

void select_v2_row_mask_s16_arm_select_v2_s16(void)
{
    const int32_t output_shape[] = SELECT_V2_ROW_MASK_S16_OUTPUT_SHAPE;
    const int32_t cond_strides[] = SELECT_V2_ROW_MASK_S16_COND_STRIDES;
    const int32_t x_strides[] = SELECT_V2_ROW_MASK_S16_X_STRIDES;
    const int32_t y_strides[] = SELECT_V2_ROW_MASK_S16_Y_STRIDES;
    const cmsis_nn_select_v2_params params = {
        .rank = SELECT_V2_ROW_MASK_S16_RANK,
        .output_shape = output_shape,
        .cond_strides = cond_strides,
        .x_strides = x_strides,
        .y_strides = y_strides,
    };
    int16_t output[SELECT_V2_ROW_MASK_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_select_v2_s16(select_v2_row_mask_s16_condition_input_tensor,
                                                         select_v2_row_mask_s16_x_input_tensor,
                                                         select_v2_row_mask_s16_y_input_tensor,
                                                         &params,
                                                         output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, select_v2_row_mask_s16_output, SELECT_V2_ROW_MASK_S16_OUTPUT_SIZE));
}