/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "../TestData/pad_int16_1/test_data.h"
#include "../TestData/pad_int16_2/test_data.h"
#include "../Utils/validate.h"
#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

void pad_int16_1_arm_pad_s16(void)
{
    const int16_t *input_ptr = pad_int16_1_input_tensor;
    int16_t output_ptr[PAD_INT16_1_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_size = {PAD_INT16_1_INPUT_N, PAD_INT16_1_INPUT_H, PAD_INT16_1_INPUT_W, PAD_INT16_1_INPUT_C};
    const cmsis_nn_dims pre_pad = {
        PAD_INT16_1_PRE_PAD_N, PAD_INT16_1_PRE_PAD_H, PAD_INT16_1_PRE_PAD_W, PAD_INT16_1_PRE_PAD_C};
    const cmsis_nn_dims post_pad = {
        PAD_INT16_1_POST_PAD_N, PAD_INT16_1_POST_PAD_H, PAD_INT16_1_POST_PAD_W, PAD_INT16_1_POST_PAD_C};

    const arm_cmsis_nn_status result =
        arm_pad_s16(input_ptr, output_ptr, PAD_INT16_1_PAD_VALUE, &input_size, &pre_pad, &post_pad);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output_ptr, pad_int16_1_output, PAD_INT16_1_OUTPUT_SIZE));
}

void pad_int16_2_arm_pad_s16(void)
{
    const int16_t *input_ptr = pad_int16_2_input_tensor;
    int16_t output_ptr[PAD_INT16_2_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_size = {PAD_INT16_2_INPUT_N, PAD_INT16_2_INPUT_H, PAD_INT16_2_INPUT_W, PAD_INT16_2_INPUT_C};
    const cmsis_nn_dims pre_pad = {
        PAD_INT16_2_PRE_PAD_N, PAD_INT16_2_PRE_PAD_H, PAD_INT16_2_PRE_PAD_W, PAD_INT16_2_PRE_PAD_C};
    const cmsis_nn_dims post_pad = {
        PAD_INT16_2_POST_PAD_N, PAD_INT16_2_POST_PAD_H, PAD_INT16_2_POST_PAD_W, PAD_INT16_2_POST_PAD_C};

    const arm_cmsis_nn_status result =
        arm_pad_s16(input_ptr, output_ptr, PAD_INT16_2_PAD_VALUE, &input_size, &pre_pad, &post_pad);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output_ptr, pad_int16_2_output, PAD_INT16_2_OUTPUT_SIZE));
}
