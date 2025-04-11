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
 
 #include "../TestData/quantize_f32_s16/test_data.h"
 #include "../Utils/validate.h"
 
void test_arm_quantize_f32_s16(void)
{
    const float   scale      = QUANTIZE_F32_S16_QUANT_OUTPUT_SCALE_INT16_T;
    const int32_t zero_point = QUANTIZE_F32_S16_QUANT_OUTPUT_ZERO_POINT_INT16_T;

    int16_t output[8];

    arm_quantize_f32_s16(
        quantize_f32_s16_input_tensor_1,
        output,
        8,
        zero_point,
        scale
    );

    TEST_ASSERT_EQUAL_INT16_ARRAY(quantize_f32_s16_output, output, 8);
}