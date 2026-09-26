/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
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

/*
 * The MVE path runs whole four-element blocks unpredicated and peels the remainder,
 * so every tail residue, the zero-block case and the store bounds need covering.
 */
static void requantize_ref_s16(const int16_t *input,
                               int16_t *output,
                               const int32_t size,
                               const int32_t multiplier,
                               const int32_t shift,
                               const int32_t input_zeropoint,
                               const int32_t output_zeropoint)
{
    for (int32_t i = 0; i < size; i++)
    {
        int32_t val = input[i] - input_zeropoint;
        val = arm_nn_requantize(val, multiplier, shift);
        val += output_zeropoint;
        output[i] = (int16_t)ARM_NN_CLAMP(val, INT16_MAX, INT16_MIN);
    }
}

#define REQ_MAX_SIZE 40
#define REQ_GUARD ((int16_t)0x5A5A)

/* Returns how many outputs landed on an INT16 bound, so a caller can assert the clamp ran. */
static int requantize_case_s16(uint32_t seed,
                               int32_t size,
                               int32_t multiplier,
                               int32_t shift,
                               int32_t izp,
                               int32_t ozp)
{
    int16_t input[REQ_MAX_SIZE];
    int16_t expected[REQ_MAX_SIZE];
    int16_t output[REQ_MAX_SIZE];

    uint32_t rng = seed;
    for (int32_t i = 0; i < REQ_MAX_SIZE; i++)
    {
        rng = rng * 1103515245u + 12345u;
        input[i] = (int16_t)((rng >> 8) % 65536u);
        expected[i] = REQ_GUARD;
        output[i] = REQ_GUARD;
    }

    requantize_ref_s16(input, expected, size, multiplier, shift, izp, ozp);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_requantize_s16_s16(input, output, size, multiplier, shift, izp, ozp));

    int saturated = 0;
    for (int32_t i = 0; i < size; i++)
    {
        TEST_ASSERT_EQUAL_INT16(expected[i], output[i]);
        if (expected[i] == INT16_MAX || expected[i] == INT16_MIN)
        {
            saturated++;
        }
    }
    for (int32_t i = size; i < REQ_MAX_SIZE; i++)
    {
        TEST_ASSERT_EQUAL_INT16_MESSAGE(REQ_GUARD, output[i], "wrote past size");
    }
    return saturated;
}

/* size 0..33 covers blocks==0, all four tail residues and several whole blocks. */
static void test_arm_requantize_s16_s16_sizes(void)
{
    for (int32_t size = 0; size <= 33; size++)
    {
        requantize_case_s16(1u + (uint32_t)size, size, 0x4F1A2B3C, -6, 0, 0);
        requantize_case_s16(91u + (uint32_t)size, size, 0x7FFFFFFF, -3, 17, -23);
    }
}

/* Positive shifts scale the result out of int16 range, so the clamp has to engage. */
static void test_arm_requantize_s16_s16_saturation(void)
{
    int saturated = 0;
    for (int32_t size = 1; size <= 17; size++)
    {
        saturated += requantize_case_s16(7u + (uint32_t)size, size, 0x7FFFFFFF, 6, 0, 0);
        saturated += requantize_case_s16(51u + (uint32_t)size, size, 0x7FFFFFFF, 4, -1000, 1000);
    }
    TEST_ASSERT_GREATER_THAN_INT_MESSAGE(0, saturated, "clamp was never exercised");
}

/* Zero points at their extremes push the pre- and post-offsets in opposite directions. */
static void test_arm_requantize_s16_s16_zero_points(void)
{
    for (int32_t size = 1; size <= 9; size++)
    {
        requantize_case_s16(13u + (uint32_t)size, size, 0x40000000, -8, INT16_MIN, INT16_MAX);
        requantize_case_s16(29u + (uint32_t)size, size, 0x40000000, -8, INT16_MAX, INT16_MIN);
    }
}
