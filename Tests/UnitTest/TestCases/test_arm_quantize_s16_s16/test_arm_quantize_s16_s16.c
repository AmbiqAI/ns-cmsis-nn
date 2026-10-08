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
