/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <math.h>

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

// Ties must round half away from zero on every leg (roundf and TensorFlow Lite semantics); the MVE leg
// used to round ties to even through VRNDN before VCVTA. With scale 0.5 the scaled values are exactly
// 2.5, -2.5, 3.5, -3.5, 0.5, -0.5, 4 and -1.
void test_arm_quantize_f32_s8_ties_away_from_zero(void)
{
    const float input[8] = {1.25f, -1.25f, 1.75f, -1.75f, 0.25f, -0.25f, 2.0f, -0.5f};
    const int8_t expected[8] = {3, -3, 4, -4, 1, -1, 4, -1};
    int8_t output[8] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, 8, 0, 0.5f));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, 8);
}

// Values beyond the int32 range, including infinities, must clamp to the int8_t bounds after the zero
// point is applied; adding the zero point to a saturated int32 conversion used to wrap to the wrong end.
void test_arm_quantize_f32_s8_saturates_with_zero_point(void)
{
    const float input[6] = {1.0e12f, -1.0e12f, (float)INFINITY, -(float)INFINITY, 3.0e9f, -3.0e9f};
    const int8_t expected[6] = {127, -128, 127, -128, 127, -128};
    int8_t output[6] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, 6, 100, 1.0f));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, 6);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, 6, -100, 1.0f));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, 6);
}

// A zero point outside the output type is rejected rather than folded into the clamp bounds.
void test_arm_quantize_f32_s8_rejects_out_of_range_zero_point(void)
{
    const float input[1] = {0.0f};
    int8_t output[1] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s8(input, output, 1, INT8_MAX + 1, 1.0f));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s8(input, output, 1, INT8_MIN - 1, 1.0f));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s8(input, output, 1, INT8_MAX, 1.0f));
    TEST_ASSERT_EQUAL(INT8_MAX, output[0]);
}
