/*
 * Copyright (C) 2022 Arm Limited or its affiliates.
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

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/abs_long_row_s8/test_data.h"
#include "../TestData/abs_multi_batch_s8/test_data.h"
#include "../TestData/abs_small_tensor_s8/test_data.h"

#include "../Utils/validate.h"

void abs_small_tensor_s8_arm_abs_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *input_data = abs_small_tensor_s8_input_tensor;
    int8_t output[ABS_SMALL_TENSOR_S8_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_abs_s8(input_data,
                                            ABS_SMALL_TENSOR_S8_INPUT_OFFSET,
                                            output,
                                            ABS_SMALL_TENSOR_S8_OUTPUT_OFFSET,
                                            ABS_SMALL_TENSOR_S8_OUTPUT_MULTIPLIER,
                                            ABS_SMALL_TENSOR_S8_OUTPUT_SHIFT,
                                            ABS_SMALL_TENSOR_S8_NEEDS_RESCALE,
                                            ABS_SMALL_TENSOR_S8_OUT_ACTIVATION_MIN,
                                            ABS_SMALL_TENSOR_S8_OUT_ACTIVATION_MAX,
                                            ABS_SMALL_TENSOR_S8_BLOCK_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, abs_small_tensor_s8_output, ABS_SMALL_TENSOR_S8_OUTPUT_LEN));
}

void abs_long_row_s8_arm_abs_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *input_data = abs_long_row_s8_input_tensor;
    int8_t output[ABS_LONG_ROW_S8_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_abs_s8(input_data,
                                            ABS_LONG_ROW_S8_INPUT_OFFSET,
                                            output,
                                            ABS_LONG_ROW_S8_OUTPUT_OFFSET,
                                            ABS_LONG_ROW_S8_OUTPUT_MULTIPLIER,
                                            ABS_LONG_ROW_S8_OUTPUT_SHIFT,
                                            ABS_LONG_ROW_S8_NEEDS_RESCALE,
                                            ABS_LONG_ROW_S8_OUT_ACTIVATION_MIN,
                                            ABS_LONG_ROW_S8_OUT_ACTIVATION_MAX,
                                            ABS_LONG_ROW_S8_BLOCK_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, abs_long_row_s8_output, ABS_LONG_ROW_S8_OUTPUT_LEN));
}

void abs_multi_batch_s8_arm_abs_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *input_data = abs_multi_batch_s8_input_tensor;
    int8_t output[ABS_MULTI_BATCH_S8_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_abs_s8(input_data,
                                            ABS_MULTI_BATCH_S8_INPUT_OFFSET,
                                            output,
                                            ABS_MULTI_BATCH_S8_OUTPUT_OFFSET,
                                            ABS_MULTI_BATCH_S8_OUTPUT_MULTIPLIER,
                                            ABS_MULTI_BATCH_S8_OUTPUT_SHIFT,
                                            ABS_MULTI_BATCH_S8_NEEDS_RESCALE,
                                            ABS_MULTI_BATCH_S8_OUT_ACTIVATION_MIN,
                                            ABS_MULTI_BATCH_S8_OUT_ACTIVATION_MAX,
                                            ABS_MULTI_BATCH_S8_BLOCK_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, abs_multi_batch_s8_output, ABS_MULTI_BATCH_S8_OUTPUT_LEN));
}
