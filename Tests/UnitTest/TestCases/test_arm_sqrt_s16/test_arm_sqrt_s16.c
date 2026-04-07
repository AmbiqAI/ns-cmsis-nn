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

#include <stdint.h>

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/sqrt_small_tensor_s16/test_data.h"
#include "../TestData/sqrt_long_row_s16/test_data.h"
#include "../TestData/sqrt_multi_batch_s16/test_data.h"

#include "../Utils/validate.h"

void sqrt_small_tensor_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_SMALL_TENSOR_S16_IN_DIM;
    const int16_t *input_data = sqrt_small_tensor_s16_input_tensor;
    int16_t output_data[SQRT_SMALL_TENSOR_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result =
        arm_sqrt_s16(input_data, &input_dims, output_data, (int16_t *)sqrt_small_tensor_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_small_tensor_s16_output, SQRT_SMALL_TENSOR_S16_OUTPUT_LEN));
}

void sqrt_long_row_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_LONG_ROW_S16_IN_DIM;
    const int16_t *input_data = sqrt_long_row_s16_input_tensor;
    int16_t output_data[SQRT_LONG_ROW_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result =
        arm_sqrt_s16(input_data, &input_dims, output_data, (int16_t *)sqrt_long_row_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_long_row_s16_output, SQRT_LONG_ROW_S16_OUTPUT_LEN));
}

void sqrt_multi_batch_s16_arm_sqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_MULTI_BATCH_S16_IN_DIM;
    const int16_t *input_data = sqrt_multi_batch_s16_input_tensor;
    int16_t output_data[SQRT_MULTI_BATCH_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result =
        arm_sqrt_s16(input_data, &input_dims, output_data, (int16_t *)sqrt_multi_batch_s16_sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, sqrt_multi_batch_s16_output, SQRT_MULTI_BATCH_S16_OUTPUT_LEN));
}
