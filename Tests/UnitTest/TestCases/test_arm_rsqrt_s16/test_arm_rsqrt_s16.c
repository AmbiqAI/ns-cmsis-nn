/*
 * Copyright (C) 2026 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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

#include "../TestData/rsqrt_float_tensor_s16/test_data.h"
#include "../Utils/validate.h"

void rsqrt_float_tensor_s16_arm_rsqrt_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *input_data = rsqrt_float_tensor_s16_input_tensor;
    int16_t output[RSQRT_FLOAT_TENSOR_S16_OUTPUT_LEN];
    // const float si = RSQRT_FLOAT_TENSOR_S16_INPUT_SCALE;
    // const float so = RSQRT_FLOAT_TENSOR_S16_OUTPUT_SCALE;

    arm_cmsis_nn_status result = arm_rsqrt_s16(input_data,
                                               RSQRT_FLOAT_TENSOR_S16_INPUT_OFFSET,
                                               output,
                                               RSQRT_FLOAT_TENSOR_S16_OUTPUT_OFFSET,
                                               RSQRT_FLOAT_TENSOR_S16_OUTPUT_MULTIPLIER,
                                               RSQRT_FLOAT_TENSOR_S16_OUTPUT_SHIFT,
                                               RSQRT_FLOAT_TENSOR_S16_NEEDS_RESCALE,
                                               RSQRT_FLOAT_TENSOR_S16_OUT_ACTIVATION_MIN,
                                               RSQRT_FLOAT_TENSOR_S16_OUT_ACTIVATION_MAX,
                                               RSQRT_FLOAT_TENSOR_S16_BLOCK_SIZE);

    // for (int i = 0; i < RSQRT_FLOAT_TENSOR_S16_BLOCK_SIZE; i++)
    // {
    //     const double input_float = (double)(input_data[i] - RSQRT_FLOAT_TENSOR_S16_INPUT_OFFSET) * (double)si;
    //     const double output_float = (double)(output[i] - RSQRT_FLOAT_TENSOR_S16_OUTPUT_OFFSET) * (double)so;
    //     const double expected_float = 1.0 / sqrt(input_float);
    //     printf("Input[%d] as float: %lf, Output[%d] as float: %lf, Rsqrt(input): %lf\r\n",
    //            i,
    //            input_float,
    //            i,
    //            output_float,
    //            expected_float);
    // }

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, rsqrt_float_tensor_s16_output, RSQRT_FLOAT_TENSOR_S16_OUTPUT_LEN));
}
