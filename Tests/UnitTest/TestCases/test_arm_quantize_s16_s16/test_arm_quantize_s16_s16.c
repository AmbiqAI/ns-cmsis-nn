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


/* Independent integer oracle: floor division with explicit half-up/away rounding, no kernel helper calls. */
static int64_t s16_floor_div(const int64_t value, const int32_t shift)
{
    const int64_t divisor = (int64_t)1 << shift;
    return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
}

static int64_t s16_reference(const int32_t value, const int32_t multiplier, const int32_t shift)
{
#if defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    const int32_t right = 31 - shift;
    return s16_floor_div((int64_t)value * multiplier + ((int64_t)1 << (right - 1)), right);
#else
    if (shift > 0)
    {
        return s16_floor_div((int64_t)value * multiplier + ((int64_t)1 << (30 - shift)), 31 - shift);
    }
    const int64_t high = s16_floor_div((int64_t)value * multiplier + ((int64_t)1 << 30), 31);
    if (shift == 0)
    {
        return high;
    }
    const int64_t half = (int64_t)1 << (-shift - 1);
    return high >= 0 ? s16_floor_div(high + half, -shift) : -s16_floor_div(-high + half, -shift);
#endif
}

static void test_arm_quantize_s16_s16(void)
{
    const int16_t inputs[] = {INT16_MIN, -20000, -1000, -1, 0, 1, 1000, 20000, INT16_MAX};
    const int32_t zero_points[] = {INT16_MIN, -5, 0, INT16_MAX};
    const int32_t scales[][2] = {
        {1 << 30, 0}, {1 << 30, 1}, {1771674010, 2}, {1690522173, -6},
        {INT32_MAX, 14}, {1 << 30, 15}, {1 << 30, 16}, {1 << 30, 22},
        {INT32_MAX, 30}, {1073741823, 30}, {1, 15}, {1, 30}, {0, 30}, {INT32_MAX, -31}};
    enum { LENGTH = sizeof(inputs) / sizeof(inputs[0]) };
    int16_t output[LENGTH + 2];
    for (size_t scale = 0; scale < sizeof(scales) / sizeof(scales[0]); scale++)
    {
        for (size_t zi = 0; zi < sizeof(zero_points) / sizeof(zero_points[0]); zi++)
        {
            for (size_t zo = 0; zo < sizeof(zero_points) / sizeof(zero_points[0]); zo++)
            {
                for (int32_t length = 1; length <= LENGTH; length++)
                {
                    for (size_t i = 0; i < LENGTH + 2; i++) output[i] = 0x5151;
                    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                        arm_requantize_s16_s16(inputs, output, length, scales[scale][0], scales[scale][1],
                                              zero_points[zi], zero_points[zo]));
                    for (int32_t i = 0; i < length; i++)
                    {
                        int64_t expected = s16_reference((int32_t)inputs[i] - zero_points[zi],
                                                         scales[scale][0], scales[scale][1]) + zero_points[zo];
                        expected = expected < INT16_MIN ? INT16_MIN : (expected > INT16_MAX ? INT16_MAX : expected);
                        TEST_ASSERT_EQUAL_INT16((int16_t)expected, output[i]);
                    }
                    for (int32_t i = length; i < LENGTH + 2; i++) TEST_ASSERT_EQUAL_HEX16(0x5151, output[i]);
                }
            }
        }
    }

    /* Exact-sized tails and the original saturation reproducer. */
    const int16_t one_input = 1000;
    int16_t one_output = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
        arm_requantize_s16_s16(&one_input, &one_output, 1, 1 << 30, 22, 0, 0));
    TEST_ASSERT_EQUAL_INT16(INT16_MAX, one_output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
        arm_requantize_s16_s16(&one_input, &one_output, 1, 1 << 30, 1, 0, 0));
    TEST_ASSERT_EQUAL_INT16(one_input, one_output);
    const int16_t three_input[3] = {INT16_MIN, 0, INT16_MAX};
    int16_t three_output[3] = {0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
        arm_requantize_s16_s16(three_input, three_output, 3, 1 << 30, 1, 0, 0));
    TEST_ASSERT_EQUAL_INT16_ARRAY(three_input, three_output, 3);
    const int32_t empty_sizes[] = {0, -1, INT32_MIN};
    for (size_t i = 0; i < sizeof(empty_sizes) / sizeof(empty_sizes[0]); i++)
    {
        one_output = 0x5151;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
            arm_requantize_s16_s16(NULL, &one_output, empty_sizes[i], 1 << 30, 1, 0, 0));
        TEST_ASSERT_EQUAL_HEX16(0x5151, one_output);
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
    for (int32_t i = size > 0 ? size : 0; i < REQ_MAX_SIZE; i++)
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

/* A negative length must write nothing. Note size & 3 is 3 for size == -1, so a tail
   derived from the mask alone would store past the start of the output buffer. */
static void test_arm_requantize_s16_s16_negative_size(void)
{
    for (int32_t size = -8; size < 0; size++)
    {
        requantize_case_s16(41u + (uint32_t)(-size), size, 0x7FFFFFFF, -3, 0, 0);
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
