/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "arm_nn_types.h"
#include "unity.h"

#include "../TestData/dynamic_update_slice_2d_s16/test_data.h"

void dynamic_update_slice_2d_arm_dynamic_update_slice_s16(void)
{
    int16_t output[DYNAMIC_UPDATE_SLICE_2D_S16_OPERAND_SIZE] = {0};

    const cmsis_nn_dynamic_update_slice_params params = {
        .rank = DYNAMIC_UPDATE_SLICE_2D_S16_RANK,
        .operand_shape = dynamic_update_slice_2d_s16_operand_shape,
        .update_shape = dynamic_update_slice_2d_s16_update_shape,
        .operand_size = DYNAMIC_UPDATE_SLICE_2D_S16_OPERAND_SIZE,
        .update_size = DYNAMIC_UPDATE_SLICE_2D_S16_UPDATE_SIZE,
        .operand_strides = dynamic_update_slice_2d_s16_operand_strides,
    };

    const arm_cmsis_nn_status result = arm_dynamic_update_slice_s16(
        dynamic_update_slice_2d_s16_operand,
        dynamic_update_slice_2d_s16_update,
        dynamic_update_slice_2d_s16_start_indices,
        &params,
        output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(dynamic_update_slice_2d_s16_output, output, DYNAMIC_UPDATE_SLICE_2D_S16_OPERAND_SIZE);
}
