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

#include "../TestData/select_v2_broadcast_s8/test_data.h"

void select_v2_broadcast_arm_select_v2_s8(void)
{
    int8_t output[SELECT_V2_BROADCAST_S8_OUTPUT_SIZE] = {0};

    const cmsis_nn_select_v2_params params = {
        .rank = SELECT_V2_BROADCAST_S8_RANK,
        .output_shape = select_v2_broadcast_s8_output_shape,
        .cond_strides = select_v2_broadcast_s8_cond_strides,
        .x_strides = select_v2_broadcast_s8_x_strides,
        .y_strides = select_v2_broadcast_s8_y_strides,
    };

    const arm_cmsis_nn_status result = arm_select_v2_s8(
        select_v2_broadcast_s8_condition,
        select_v2_broadcast_s8_x,
        select_v2_broadcast_s8_y,
        &params,
        output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT8_ARRAY(select_v2_broadcast_s8_output, output, SELECT_V2_BROADCAST_S8_OUTPUT_SIZE);
}
