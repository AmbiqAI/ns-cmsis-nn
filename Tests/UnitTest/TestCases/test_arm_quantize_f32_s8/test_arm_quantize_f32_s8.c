/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

 #include "arm_nnfunctions.h"
 #include "unity.h"

 #include "../TestData/quantize_f32_s8/test_data.h"
 #include "../Utils/validate.h"

void test_arm_quantize_f32_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const float   scale      = QUANTIZE_F32_S8_QUANT_OUTPUT_SCALE_INT8_T;
    const int32_t zero_point = QUANTIZE_F32_S8_QUANT_OUTPUT_ZERO_POINT_INT8_T;

    int8_t output[QUANTIZE_F32_S8_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_quantize_f32_s8(
        quantize_f32_s8_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        QUANTIZE_F32_S8_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_EQUAL_INT8_ARRAY(quantize_f32_s8_output, output, QUANTIZE_F32_S8_OUTPUT_LEN);
}
