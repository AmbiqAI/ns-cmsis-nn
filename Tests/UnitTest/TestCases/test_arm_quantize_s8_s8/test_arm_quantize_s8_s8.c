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

 #include "../TestData/quantize_s8_s8/test_data.h"
 #include "../Utils/validate.h"


void test_arm_quantize_s8_s8(void)
{
    // Prepare test data
    const int8_t input[] = {-128, -64, 0, 63, 64, 127};
    enum { INPUT_LEN = (int)(sizeof(input) / sizeof(input[0])) };

    // Parameters for identity
    const int32_t effective_scale_multiplier = 0x40000000; // Q1.31 format close to 1.0
    const int32_t effective_scale_shift = 0;
    const int32_t input_zeropoint = 0;
    const int32_t output_zeropoint = 0;

    int8_t output[INPUT_LEN];
    int8_t expected[INPUT_LEN];

    // Compute the "reference" expected results in float
    for (size_t i = 0; i < INPUT_LEN; i++)
    {
        float ref_val = 0.5f * input[i]; // scale by 0.5
        // Round to nearest integer, then clamp to [-128, 127]
        int32_t rounded = (int32_t)((ref_val >= 0) ? (ref_val + 0.5f) : (ref_val - 0.5f));
        if (rounded > 127)
        {
            rounded = 127;
        }
        else if (rounded < -128)
        {
            rounded = -128;
        }
        expected[i] = (int8_t)rounded;
    }
    
    // Call the function under test
    arm_requantize_s8_s8(input,
                         output,
                         INPUT_LEN,
                         effective_scale_multiplier,
                         effective_scale_shift,
                         input_zeropoint,
                         output_zeropoint);

    // Verify the results
    for (size_t i = 0; i < INPUT_LEN; i++)
    {
        // For identity transform, output[i] should match input[i]
        TEST_ASSERT_EQUAL_INT8_MESSAGE(expected[i], output[i], "Identity requantization failed");
    }
}