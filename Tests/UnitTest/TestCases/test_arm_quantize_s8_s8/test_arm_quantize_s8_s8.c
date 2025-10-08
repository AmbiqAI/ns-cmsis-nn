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

#include "../Utils/validate.h"


void test_arm_quantize_s8_s8(void)
{
    const arm_cmsis_nn_status expected_status = ARM_CMSIS_NN_SUCCESS;
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
    arm_cmsis_nn_status result = arm_requantize_s8_s8(
        input,
        output,
        INPUT_LEN,
        effective_scale_multiplier,
        effective_scale_shift,
        input_zeropoint,
        output_zeropoint
    );

    // Verify the results
    TEST_ASSERT_EQUAL(expected_status, result);
    for (size_t i = 0; i < INPUT_LEN; i++)
    {
        // For identity transform, output[i] should match input[i]
        TEST_ASSERT_EQUAL_INT8_MESSAGE(expected[i], output[i], "Identity requantization failed");
    }
}
