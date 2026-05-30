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

#include "../TestData/broadcast_to_2d_s16/test_data.h"

void broadcast_to_2d_arm_broadcast_to_s16(void)
{
    int16_t output[BROADCAST_TO_2D_S16_OUTPUT_SIZE] = {0};

    const cmsis_nn_broadcast_to_params params = {
        .rank = BROADCAST_TO_2D_S16_RANK,
        .input_shape = broadcast_to_2d_s16_input_shape,
        .output_shape = broadcast_to_2d_s16_output_shape,
    };

    const arm_cmsis_nn_status result = arm_broadcast_to_s16(broadcast_to_2d_s16_input, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(broadcast_to_2d_s16_output, output, BROADCAST_TO_2D_S16_OUTPUT_SIZE);
}
