/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "unity.h"
#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>

#include "../Utils/validate.h"

#include "../TestData/prelu_per_channel_s8/test_data.h"


void prelu_per_channel_s8_arm_prelu_s8(void)
{
    const cmsis_nn_dims input_dims = {PRELU_PER_CHANNEL_S8_INPUT_N,
                                      PRELU_PER_CHANNEL_S8_INPUT_H,
                                      PRELU_PER_CHANNEL_S8_INPUT_W,
                                      PRELU_PER_CHANNEL_S8_INPUT_C};
    const cmsis_nn_dims alpha_dims = {PRELU_PER_CHANNEL_S8_ALPHA_N,
                                      PRELU_PER_CHANNEL_S8_ALPHA_H,
                                      PRELU_PER_CHANNEL_S8_ALPHA_W,
                                      PRELU_PER_CHANNEL_S8_ALPHA_C};
    const cmsis_nn_dims output_dims = {PRELU_PER_CHANNEL_S8_OUTPUT_N,
                                       PRELU_PER_CHANNEL_S8_OUTPUT_H,
                                       PRELU_PER_CHANNEL_S8_OUTPUT_W,
                                       PRELU_PER_CHANNEL_S8_OUTPUT_C};

    int8_t output[PRELU_PER_CHANNEL_S8_OUTPUT_LEN] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s8(&input_dims,
                     prelu_per_channel_s8_input_tensor,
                     &alpha_dims,
                     prelu_per_channel_s8_alpha_input_tensor,
                     PRELU_PER_CHANNEL_S8_INPUT_OFFSET,
                     PRELU_PER_CHANNEL_S8_ALPHA_OFFSET,
                     PRELU_PER_CHANNEL_S8_OUTPUT_OFFSET,
                     PRELU_PER_CHANNEL_S8_OUTPUT_MULTIPLIER_1,
                     PRELU_PER_CHANNEL_S8_OUTPUT_SHIFT_1,
                     PRELU_PER_CHANNEL_S8_OUTPUT_MULTIPLIER_2,
                     PRELU_PER_CHANNEL_S8_OUTPUT_SHIFT_2,
                     &output_dims,
                     output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, prelu_per_channel_s8_output, PRELU_PER_CHANNEL_S8_OUTPUT_LEN));
}
