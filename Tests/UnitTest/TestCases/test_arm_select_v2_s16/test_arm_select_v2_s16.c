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

#include "../TestData/select_v2_broadcast_s16/test_data.h"

void select_v2_broadcast_arm_select_v2_s16(void)
{
    int16_t output[SELECT_V2_BROADCAST_S16_OUTPUT_SIZE] = {0};

    const cmsis_nn_select_v2_params params = {
        .rank = SELECT_V2_BROADCAST_S16_RANK,
        .output_shape = select_v2_broadcast_s16_output_shape,
        .cond_strides = select_v2_broadcast_s16_cond_strides,
        .x_strides = select_v2_broadcast_s16_x_strides,
        .y_strides = select_v2_broadcast_s16_y_strides,
    };

    const arm_cmsis_nn_status result = arm_select_v2_s16(
        select_v2_broadcast_s16_condition, select_v2_broadcast_s16_x,
        select_v2_broadcast_s16_y, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(select_v2_broadcast_s16_output, output, SELECT_V2_BROADCAST_S16_OUTPUT_SIZE);
}
