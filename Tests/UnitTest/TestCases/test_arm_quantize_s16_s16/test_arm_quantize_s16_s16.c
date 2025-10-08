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


static int32_t reference_single_rounding(int32_t val,
                                         int32_t multiplier,
                                         int32_t shift)
{
    // This matches arm_nn_requantize() for single-rounding
    int64_t total_shift = 31 - shift;
    int64_t new_val = (int64_t)val * (int64_t)multiplier;
    // Shift right by (total_shift - 1)
    int32_t result = (int32_t)(new_val >> (total_shift - 1));
    // Then shift by 1 again with a +1 for rounding
    result = (result + 1) >> 1;
    return result;
}

/**
 * @brief Test for int16 -> int16 requantization with scale = 0.5
 */
static void test_arm_quantize_s16_s16(void)
{
    const arm_cmsis_nn_status expected_status = ARM_CMSIS_NN_SUCCESS;
    // Prepare representative test data
    // Cover negative extremes, positive extremes, and some mid-range values
    const int16_t input[] = {
        INT16_MIN,      // -32768
        -20000,
        -12345,
        -1,
        0,
        1,
        12345,
        20000,
        INT16_MAX       // 32767
    };
    enum { INPUT_LEN = (int)(sizeof(input) / sizeof(input[0])) };

    // We want a scale ~0.5 in Q1.31 format:
    // 0x40000000 = 2^30, representing 0.5 in 32-bit Q1.31
    const int32_t effective_scale_multiplier = 0x40000000;
    const int32_t effective_scale_shift = 0;
    const int32_t input_zeropoint = 0;
    const int32_t output_zeropoint = 0;

    int16_t output[INPUT_LEN];
    int16_t expected[INPUT_LEN];

    // Compute expected using the same integer math
    for (size_t i = 0; i < INPUT_LEN; i++)
    {
        // 1) Subtract input zero point
        int32_t val = (int32_t)input[i] - input_zeropoint;

        // 2) Single-rounding “requantize”
        val = reference_single_rounding(val, effective_scale_multiplier, effective_scale_shift);

        // 3) Add output zero point
        val += output_zeropoint;

        // 4) Clamp to int16 range
        if (val > INT16_MAX)
        {
            val = INT16_MAX;
        }
        else if (val < INT16_MIN)
        {
            val = INT16_MIN;
        }
        expected[i] = (int16_t)val;
    }

    // Call the function under test
    arm_cmsis_nn_status result = arm_requantize_s16_s16(
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
        TEST_ASSERT_EQUAL_INT16_MESSAGE(expected[i], output[i],
                                        "Mismatch in single-rounding approach");
    }
}
