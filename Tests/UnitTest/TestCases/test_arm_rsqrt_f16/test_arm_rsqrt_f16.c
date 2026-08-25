/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#include "rsqrt_f16_data.h"

static uint16_t f16_bits(float16_t value)
{
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void assert_f16_within_one_ulp(float16_t expected, float16_t actual)
{
    TEST_ASSERT_INT_WITHIN(1, f16_bits(expected), f16_bits(actual));
}

void rsqrt_f16_arm_rsqrt_f16(void)
{
    float16_t output[RSQRT_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(rsqrt_f16_input, output, RSQRT_F16_DST_SIZE));
    for (int32_t i = 0; i < RSQRT_F16_DST_SIZE; ++i)
    {
        assert_f16_within_one_ulp(rsqrt_f16_output_ref[i], output[i]);
    }
}

void rsqrt_f16_in_place_arm_rsqrt_f16(void)
{
    float16_t values[RSQRT_F16_DST_SIZE];
    memcpy(values, rsqrt_f16_input, sizeof(values));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(values, values, RSQRT_F16_DST_SIZE));
    for (int32_t i = 0; i < RSQRT_F16_DST_SIZE; ++i)
    {
        assert_f16_within_one_ulp(rsqrt_f16_output_ref[i], values[i]);
    }
}

void rsqrt_f16_special_values_arm_rsqrt_f16(void)
{
    const uint16_t input_bits[] = {0x0000, 0x8000, 0xBC00, 0x7C00, 0xFC00, 0x7D55};
    const uint16_t expected_bits[] = {0x7C00, 0xFC00, 0x7E00, 0x0000, 0x7E00, 0x7F55};
    float16_t input[6];
    float16_t output[6] = {0};
    memcpy(input, input_bits, sizeof(input));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(input, output, 6));
    for (int32_t i = 0; i < 6; ++i)
    {
        TEST_ASSERT_EQUAL_HEX16(expected_bits[i], f16_bits(output[i]));
    }
}

void rsqrt_f16_arg_error_arm_rsqrt_f16(void)
{
    float16_t output[RSQRT_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(NULL, output, RSQRT_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(rsqrt_f16_input, NULL, RSQRT_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(rsqrt_f16_input, output, 0));
}
