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

 #include "../TestData/quantize_f32_s16/test_data.h"
 #include "../Utils/validate.h"

void test_arm_quantize_f32_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const float   scale      = QUANTIZE_F32_S16_QUANT_OUTPUT_SCALE_INT16_T;
    const int32_t zero_point = QUANTIZE_F32_S16_QUANT_OUTPUT_ZERO_POINT_INT16_T;

    int16_t output[QUANTIZE_F32_S16_OUTPUT_LEN];

    arm_cmsis_nn_status result = arm_quantize_f32_s16(
        quantize_f32_s16_input_tensor_1,
        output,
        QUANTIZE_F32_S16_OUTPUT_LEN,
        zero_point,
        scale
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(quantize_f32_s16_output, output, QUANTIZE_F32_S16_OUTPUT_LEN);
}
