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
#include <arm_nnactivations.h>

#include "../TestData/leaky_relu/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (2)

void leaky_relu_arm_leaky_relu_s8(void)
{

    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *input_data = leaky_relu_input;
    int8_t output[LEAKY_RELU_DST_SIZE];

    arm_cmsis_nn_status result = arm_leaky_relu_s8(
        input_data,
        LEAKY_RELU_OFFSET,
        LEAKY_RELU_OFFSET,
        LEAKY_RELU_OUTPUT_MULT_ALPHA,
        LEAKY_RELU_OUTPUT_SHIFT_ALPHA,
        LEAKY_RELU_OUTPUT_MULT_IDENTITY,
        LEAKY_RELU_OUTPUT_SHIFT_IDENTITY,
        output,
        LEAKY_RELU_DST_SIZE
    );

    TEST_ASSERT_EQUAL(expected, result);
    // TEST_ASSERT_TRUE(validate(output, leaky_relu_output_ref, LEAKY_RELU_DST_SIZE));
}
