/*
 * Copyright (C) 2026 Arm Limited or its affiliates.
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
#include <math.h>

#include "../TestData/sqrt_small_tensor_s16/test_data.h"
#include "../TestData/sqrt_long_row_s16/test_data.h"
#include "../TestData/sqrt_multi_batch_s16/test_data.h"
#include "../TestData/sqrt_float_tensor_s16/test_data.h"
#include "../Utils/validate.h"
#include <stdio.h>
void sqrt_small_tensor_s16_arm_sqrt_s16(void)
{
    // Small 2x3x4 tensor: mixed negatives/positives to verify clamp-to-zero and basic sqrt scaling.
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *input_data = sqrt_small_tensor_s16_input_tensor;
    int16_t output[SQRT_SMALL_TENSOR_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_sqrt_s16(input_data,
                                              SQRT_SMALL_TENSOR_S16_INPUT_OFFSET,
                                              output,
                                              SQRT_SMALL_TENSOR_S16_OUTPUT_OFFSET,
                                              SQRT_SMALL_TENSOR_S16_OUTPUT_MULTIPLIER,
                                              SQRT_SMALL_TENSOR_S16_OUTPUT_SHIFT,
                                              SQRT_SMALL_TENSOR_S16_NEEDS_RESCALE,
                                              SQRT_SMALL_TENSOR_S16_OUT_ACTIVATION_MIN,
                                              SQRT_SMALL_TENSOR_S16_OUT_ACTIVATION_MAX,
                                              SQRT_SMALL_TENSOR_S16_BLOCK_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sqrt_small_tensor_s16_output, SQRT_SMALL_TENSOR_S16_OUTPUT_LEN));
}

void sqrt_long_row_s16_arm_sqrt_s16(void)
{
    // Long 1x32 row with input offset applied: exercises offset handling and a wider value spread.
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *input_data = sqrt_long_row_s16_input_tensor;
    int16_t output[SQRT_LONG_ROW_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_sqrt_s16(input_data,
                                              SQRT_LONG_ROW_S16_INPUT_OFFSET,
                                              output,
                                              SQRT_LONG_ROW_S16_OUTPUT_OFFSET,
                                              SQRT_LONG_ROW_S16_OUTPUT_MULTIPLIER,
                                              SQRT_LONG_ROW_S16_OUTPUT_SHIFT,
                                              SQRT_LONG_ROW_S16_NEEDS_RESCALE,
                                              SQRT_LONG_ROW_S16_OUT_ACTIVATION_MIN,
                                              SQRT_LONG_ROW_S16_OUT_ACTIVATION_MAX,
                                              SQRT_LONG_ROW_S16_BLOCK_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sqrt_long_row_s16_output, SQRT_LONG_ROW_S16_OUTPUT_LEN));
}

void sqrt_multi_batch_s16_arm_sqrt_s16(void)
{
    // Multi-batch 2x2x3x4 tensor: validates batch handling and varied magnitudes.
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *input_data = sqrt_multi_batch_s16_input_tensor;
    int16_t output[SQRT_MULTI_BATCH_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_sqrt_s16(input_data,
                                              SQRT_MULTI_BATCH_S16_INPUT_OFFSET,
                                              output,
                                              SQRT_MULTI_BATCH_S16_OUTPUT_OFFSET,
                                              SQRT_MULTI_BATCH_S16_OUTPUT_MULTIPLIER,
                                              SQRT_MULTI_BATCH_S16_OUTPUT_SHIFT,
                                              SQRT_MULTI_BATCH_S16_NEEDS_RESCALE,
                                              SQRT_MULTI_BATCH_S16_OUT_ACTIVATION_MIN,
                                              SQRT_MULTI_BATCH_S16_OUT_ACTIVATION_MAX,
                                              SQRT_MULTI_BATCH_S16_BLOCK_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sqrt_multi_batch_s16_output, SQRT_MULTI_BATCH_S16_OUTPUT_LEN));
}

void sqrt_float_tensor_s16_arm_sqrt_s16(void)
{
    // Float-derived inputs: quantize abs(N(0,1))*10 to s16, then validate scale-based sqrt.
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *input_data = sqrt_float_tensor_s16_input_tensor;
    int16_t output[SQRT_FLOAT_TENSOR_S16_OUTPUT_LEN];
    // const float si = SQRT_FLOAT_TENSOR_S16_INPUT_SCALE;
    // const float so = SQRT_FLOAT_TENSOR_S16_OUTPUT_SCALE;

    arm_cmsis_nn_status result = arm_sqrt_s16(input_data,
                                              SQRT_FLOAT_TENSOR_S16_INPUT_OFFSET,
                                              output,
                                              SQRT_FLOAT_TENSOR_S16_OUTPUT_OFFSET,
                                              SQRT_FLOAT_TENSOR_S16_OUTPUT_MULTIPLIER,
                                              SQRT_FLOAT_TENSOR_S16_OUTPUT_SHIFT,
                                              SQRT_FLOAT_TENSOR_S16_NEEDS_RESCALE,
                                              SQRT_FLOAT_TENSOR_S16_OUT_ACTIVATION_MIN,
                                              SQRT_FLOAT_TENSOR_S16_OUT_ACTIVATION_MAX,
                                              SQRT_FLOAT_TENSOR_S16_BLOCK_SIZE);
    // for (int i = 0 ; i < SQRT_FLOAT_TENSOR_S16_BLOCK_SIZE; i++)
    // {
    //     double input_float = (double)(input_data[i] - SQRT_FLOAT_TENSOR_S16_INPUT_OFFSET) * (double)si;
    //     double output_float = (double)(output[i] - SQRT_FLOAT_TENSOR_S16_OUTPUT_OFFSET) * (double)so;
    //     double expected_float = sqrt(input_float);
    //     printf("Input[%d] as float: %lf, Output[%d] as float: (%lf, %lf) (est, expected)\r\n",
    //            i,
    //            input_float,
    //            i,
    //            output_float,
    //            expected_float);
    // }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sqrt_float_tensor_s16_output, SQRT_FLOAT_TENSOR_S16_OUTPUT_LEN));
}
