/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
