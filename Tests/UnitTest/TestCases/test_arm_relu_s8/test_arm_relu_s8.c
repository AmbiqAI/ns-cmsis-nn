/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "unity.h"
#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>

#include "../TestData/relu_basic_s8/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (2)

void relu_basic_arm_relu_s8(void)
{

    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *input_data = relu_basic_s8_input_tensor;
    int8_t output[RELU_BASIC_S8_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_relu_s8(
        input_data,
        RELU_BASIC_S8_INPUT_OFFSET,
        RELU_BASIC_S8_OUTPUT_OFFSET,
        RELU_BASIC_S8_OUTPUT_MULTIPLIER,
        RELU_BASIC_S8_OUTPUT_SHIFT,
        output,
        RELU_BASIC_S8_OUTPUT_LEN
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, relu_basic_s8_output, RELU_BASIC_S8_OUTPUT_LEN));
}
