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
 
 #include "../TestData/quantize_f32_s8/test_data.h"
 #include "../Utils/validate.h"
 
void test_arm_quantize_f32_s8(void)
{
    const float   scale      = QUANTIZE_F32_S8_QUANT_OUTPUT_SCALE_INT8_T;
    const int32_t zero_point = QUANTIZE_F32_S8_QUANT_OUTPUT_ZERO_POINT_INT8_T;

    int8_t output[8];

    arm_quantize_f32_s8(
        quantize_f32_s8_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        8,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    );

    TEST_ASSERT_EQUAL_INT8_ARRAY(quantize_f32_s8_output, output, 8);
}