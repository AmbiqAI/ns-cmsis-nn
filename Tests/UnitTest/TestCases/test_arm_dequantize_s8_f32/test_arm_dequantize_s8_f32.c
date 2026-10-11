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

 #include "../TestData/dequantize_s8_f32/test_data.h"
 #include "../Utils/validate.h"

void test_arm_dequantize_s8_f32(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    float   scale      = DEQUANTIZE_S8_F32_QUANT_INPUT_SCALE_FLOAT32_T;      // e.g. 2.725261e-05
    int32_t zero_point = DEQUANTIZE_S8_F32_QUANT_INPUT_ZERO_POINT_FLOAT32_T; // e.g. 0

    const int size = DEQUANTIZE_S8_F32_OUTPUT_LEN;

    // Prepare a buffer for the function output
    float output_f32[DEQUANTIZE_S8_F32_OUTPUT_LEN];

    // Call your custom function
    arm_cmsis_nn_status result = arm_dequantize_s8_f32(
        dequantize_s8_f32_input_tensor_1,
        output_f32,
        size,
        zero_point,
        scale
    );

    TEST_ASSERT_EQUAL(expected, result);

    const float tolerance = 1e-5f;
    for(int i = 0; i < size; i++)
    {
        TEST_ASSERT_FLOAT_WITHIN(
            tolerance,
            dequantize_s8_f32_output[i],
            output_f32[i]
        );
    }
}

#define DEQUANTIZE_INT8_GUARD (-1234.5f)

// A zero point outside the input type is rejected before any access (issue #784).
void dequantize_s8_f32_rejects_out_of_range_zero_point(void)
{
    const int8_t input[4] = {1, 2, 3, 4};
    float output[4];
    const int32_t zero_points[] = {INT8_MIN - 1, INT8_MAX + 1, INT32_MIN, INT32_MAX};

    for (size_t z = 0; z < sizeof(zero_points) / sizeof(zero_points[0]); z++)
    {
        for (int i = 0; i < 4; i++)
        {
            output[i] = DEQUANTIZE_INT8_GUARD;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_dequantize_s8_f32(input, output, 4, zero_points[z], 0.5f));
        for (int i = 0; i < 4; i++)
        {
            TEST_ASSERT_EQUAL_FLOAT(DEQUANTIZE_INT8_GUARD, output[i]);
        }
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_s8_f32(input, output, 4, INT8_MIN, 0.5f));
    for (int i = 0; i < 4; i++)
    {
        TEST_ASSERT_EQUAL_FLOAT(((float)input[i] - (float)INT8_MIN) * 0.5f, output[i]);
    }
}

// A negative size is rejected and nothing is written; the Helium loop would read it as a huge count (issue #784).
void dequantize_s8_f32_rejects_negative_size(void)
{
    const int8_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float output[8];
    const int32_t sizes[] = {-1, -5, INT32_MIN};

    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++)
    {
        for (int i = 0; i < 8; i++)
        {
            output[i] = DEQUANTIZE_INT8_GUARD;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_dequantize_s8_f32(input, output, sizes[s], 0, 0.5f));
        for (int i = 0; i < 8; i++)
        {
            TEST_ASSERT_EQUAL_FLOAT(DEQUANTIZE_INT8_GUARD, output[i]);
        }
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_s8_f32(input, output, 0, 0, 0.5f));
    for (int i = 0; i < 8; i++)
    {
        TEST_ASSERT_EQUAL_FLOAT(DEQUANTIZE_INT8_GUARD, output[i]);
    }
}
