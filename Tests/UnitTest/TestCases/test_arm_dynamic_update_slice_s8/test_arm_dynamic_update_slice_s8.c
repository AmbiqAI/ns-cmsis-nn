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

#include "../TestData/dynamic_update_slice_3d_edge_s8/test_data.h"
#include "../TestData/dynamic_update_slice_2d_s8/test_data.h"
#include "../TestData/dynamic_update_slice_start_zero_s8/test_data.h"
#include "../Utils/validate.h"

void dynamic_update_slice_2d_s8_arm_dynamic_update_slice_s8(void)
{
    const int32_t operand_shape[] = DYNAMIC_UPDATE_SLICE_2D_S8_OPERAND_SHAPE;
    const int32_t update_shape[] = DYNAMIC_UPDATE_SLICE_2D_S8_UPDATE_SHAPE;
    const int32_t operand_strides[] = DYNAMIC_UPDATE_SLICE_2D_S8_OPERAND_STRIDES;
    const cmsis_nn_dynamic_update_slice_params params = {
        .rank = DYNAMIC_UPDATE_SLICE_2D_S8_RANK,
        .operand_shape = operand_shape,
        .update_shape = update_shape,
        .operand_size = DYNAMIC_UPDATE_SLICE_2D_S8_OPERAND_SIZE,
        .update_size = DYNAMIC_UPDATE_SLICE_2D_S8_UPDATE_SIZE,
        .operand_strides = operand_strides,
    };
    int8_t output[DYNAMIC_UPDATE_SLICE_2D_S8_OPERAND_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_dynamic_update_slice_s8(dynamic_update_slice_2d_s8_operand_input_tensor,
                                                                   dynamic_update_slice_2d_s8_update_input_tensor,
                                                                   dynamic_update_slice_2d_s8_start_indices,
                                                                   &params,
                                                                   output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, dynamic_update_slice_2d_s8_output, DYNAMIC_UPDATE_SLICE_2D_S8_OPERAND_SIZE));
}

void dynamic_update_slice_start_zero_s8_arm_dynamic_update_slice_s8(void)
{
    const int32_t operand_shape[] = DYNAMIC_UPDATE_SLICE_START_ZERO_S8_OPERAND_SHAPE;
    const int32_t update_shape[] = DYNAMIC_UPDATE_SLICE_START_ZERO_S8_UPDATE_SHAPE;
    const int32_t operand_strides[] = DYNAMIC_UPDATE_SLICE_START_ZERO_S8_OPERAND_STRIDES;
    const cmsis_nn_dynamic_update_slice_params params = {
        .rank = DYNAMIC_UPDATE_SLICE_START_ZERO_S8_RANK,
        .operand_shape = operand_shape,
        .update_shape = update_shape,
        .operand_size = DYNAMIC_UPDATE_SLICE_START_ZERO_S8_OPERAND_SIZE,
        .update_size = DYNAMIC_UPDATE_SLICE_START_ZERO_S8_UPDATE_SIZE,
        .operand_strides = operand_strides,
    };
    int8_t output[DYNAMIC_UPDATE_SLICE_START_ZERO_S8_OPERAND_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_dynamic_update_slice_s8(dynamic_update_slice_start_zero_s8_operand_input_tensor,
                                                                   dynamic_update_slice_start_zero_s8_update_input_tensor,
                                                                   dynamic_update_slice_start_zero_s8_start_indices,
                                                                   &params,
                                                                   output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output,
                              dynamic_update_slice_start_zero_s8_output,
                              DYNAMIC_UPDATE_SLICE_START_ZERO_S8_OPERAND_SIZE));
}

void dynamic_update_slice_3d_edge_s8_arm_dynamic_update_slice_s8(void)
{
    const int32_t operand_shape[] = DYNAMIC_UPDATE_SLICE_3D_EDGE_S8_OPERAND_SHAPE;
    const int32_t update_shape[] = DYNAMIC_UPDATE_SLICE_3D_EDGE_S8_UPDATE_SHAPE;
    const int32_t operand_strides[] = DYNAMIC_UPDATE_SLICE_3D_EDGE_S8_OPERAND_STRIDES;
    const cmsis_nn_dynamic_update_slice_params params = {
        .rank = DYNAMIC_UPDATE_SLICE_3D_EDGE_S8_RANK,
        .operand_shape = operand_shape,
        .update_shape = update_shape,
        .operand_size = DYNAMIC_UPDATE_SLICE_3D_EDGE_S8_OPERAND_SIZE,
        .update_size = DYNAMIC_UPDATE_SLICE_3D_EDGE_S8_UPDATE_SIZE,
        .operand_strides = operand_strides,
    };
    int8_t output[DYNAMIC_UPDATE_SLICE_3D_EDGE_S8_OPERAND_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_dynamic_update_slice_s8(dynamic_update_slice_3d_edge_s8_operand_input_tensor,
                                                                   dynamic_update_slice_3d_edge_s8_update_input_tensor,
                                                                   dynamic_update_slice_3d_edge_s8_start_indices,
                                                                   &params,
                                                                   output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, dynamic_update_slice_3d_edge_s8_output, DYNAMIC_UPDATE_SLICE_3D_EDGE_S8_OPERAND_SIZE));
}