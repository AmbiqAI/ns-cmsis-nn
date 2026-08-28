/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <math.h>
#include <string.h>

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

// Ties must round half away from zero on every leg (roundf and TensorFlow Lite semantics); the MVE leg
// used to round ties to even through VRNDN before VCVTA. With scale 0.5 the scaled values are exactly
// 2.5, -2.5, 3.5, -3.5, 0.5, -0.5, 4 and -1.
void test_arm_quantize_f32_s16_ties_away_from_zero(void)
{
    const float input[8] = {1.25f, -1.25f, 1.75f, -1.75f, 0.25f, -0.25f, 2.0f, -0.5f};
    const int16_t expected[8] = {3, -3, 4, -4, 1, -1, 4, -1};
    int16_t output[8] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s16(input, output, 8, 0, 0.5f));
    TEST_ASSERT_EQUAL_INT16_ARRAY(expected, output, 8);
}

// Values beyond the int32 range, including infinities, must clamp to the int16_t bounds after the zero
// point is applied; adding the zero point to a saturated int32 conversion used to wrap to the wrong end.
// NaN maps to the zero point on both legs. It is built from its bit pattern so that a -ffinite-math-only
// build of this test cannot fold it, and only the integer outputs are compared.
void test_arm_quantize_f32_s16_saturates_with_zero_point(void)
{
    const uint32_t nan_bits = 0x7fc00000U;
    float nan_value;
    memcpy(&nan_value, &nan_bits, sizeof(nan_value));

    const float input[7] = {1.0e12f, -1.0e12f, (float)INFINITY, -(float)INFINITY, 3.0e9f, -3.0e9f, nan_value};
    const int16_t expected_positive[7] = {32767, -32768, 32767, -32768, 32767, -32768, 30000};
    const int16_t expected_negative[7] = {32767, -32768, 32767, -32768, 32767, -32768, -30000};
    int16_t output[7] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s16(input, output, 7, 30000, 1.0f));
    TEST_ASSERT_EQUAL_INT16_ARRAY(expected_positive, output, 7);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s16(input, output, 7, -30000, 1.0f));
    TEST_ASSERT_EQUAL_INT16_ARRAY(expected_negative, output, 7);
}

// A zero point outside the output type is rejected rather than folded into the clamp bounds.
void test_arm_quantize_f32_s16_rejects_out_of_range_zero_point(void)
{
    const float input[1] = {0.0f};
    int16_t output[1] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s16(input, output, 1, INT16_MAX + 1, 1.0f));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_quantize_f32_s16(input, output, 1, INT16_MIN - 1, 1.0f));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_quantize_f32_s16(input, output, 1, INT16_MAX, 1.0f));
    TEST_ASSERT_EQUAL(INT16_MAX, output[0]);
}
