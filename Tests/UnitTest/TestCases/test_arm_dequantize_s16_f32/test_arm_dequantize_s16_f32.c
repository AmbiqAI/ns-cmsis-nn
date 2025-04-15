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
 
 #include "../TestData/dequantize_s16_f32/test_data.h"
 #include "../Utils/validate.h"
 
void test_arm_dequantize_s16_f32(void)
{
    // from config_data.h
    float   scale      = DEQUANTIZE_S16_F32_QUANT_INPUT_SCALE_FLOAT32_T;      // e.g. 2.725261e-05
    int32_t zero_point = DEQUANTIZE_S16_F32_QUANT_INPUT_ZERO_POINT_FLOAT32_T; // e.g. 0

    // We'll dequantize 8 elements
    const int size = DEQUANTIZE_S16_F32_OUTPUT_LEN;

    // Prepare a buffer for the function output
    float output_f32[DEQUANTIZE_S16_F32_OUTPUT_LEN];

    // Call your custom function
    arm_dequantize_s16_f32(
        dequantize_s16_f32_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    );

    // Compare with TFLite reference
    // We'll allow some small tolerance for floating comparisons:
    const float tolerance = 1e-5f;

    for(int i = 0; i < size; i++)
    {
        TEST_ASSERT_FLOAT_WITHIN(tolerance,
                                 dequantize_s16_f32_output[i],
                                 output_f32[i]);
    }
}