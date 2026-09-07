/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "rsqrt_f16_data.h"

#define RSQRT_F16_EXHAUSTIVE_BLOCK 1000

static uint16_t rsqrt_f16_bits(float16_t value)
{
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

#if !defined(__ARM_FP16_FORMAT_ALTERNATIVE)
// Contract reference (Include/arm_nnfunctions_flt.h, arm_rsqrt_f16): specials by bit
// pattern; positive finite inputs are the float64 result rounded once to
// float16, independent of the kernel's float32 evaluation.
static uint16_t rsqrt_f16_reference_bits(uint16_t in_bits)
{
    const uint16_t magnitude = in_bits & UINT16_C(0x7FFF);
    float16_t x;

    if (magnitude == 0)
    {
        return (uint16_t)(in_bits | UINT16_C(0x7C00));
    }
    if (magnitude > UINT16_C(0x7C00))
    {
        return (uint16_t)(in_bits | UINT16_C(0x0200));
    }
    if (in_bits == UINT16_C(0x7C00))
    {
        return UINT16_C(0);
    }
    if (in_bits & UINT16_C(0x8000))
    {
        return UINT16_C(0x7E00);
    }
    memcpy(&x, &in_bits, sizeof(x));
    return rsqrt_f16_bits((float16_t)(1.0 / sqrt((double)x)));
}
#endif

void rsqrt_f16_arm_rsqrt_f16(void)
{
    float16_t output[RSQRT_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(rsqrt_f16_input, output, RSQRT_F16_DST_SIZE));
    for (int32_t i = 0; i < RSQRT_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_HEX16(rsqrt_f16_bits(rsqrt_f16_output_ref[i]), rsqrt_f16_bits(output[i]));
    }
}

void rsqrt_f16_in_place_arm_rsqrt_f16(void)
{
    float16_t values[RSQRT_F16_DST_SIZE];
    memcpy(values, rsqrt_f16_input, sizeof(values));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(values, values, RSQRT_F16_DST_SIZE));
    for (int32_t i = 0; i < RSQRT_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_HEX16(rsqrt_f16_bits(rsqrt_f16_output_ref[i]), rsqrt_f16_bits(values[i]));
    }
}

void rsqrt_f16_special_values_arm_rsqrt_f16(void)
{
#if defined(__ARM_FP16_FORMAT_ALTERNATIVE)
    TEST_IGNORE_MESSAGE("Arm alternative half precision has no infinity or NaN encodings");
#else
    // +0, -0, -1, +Inf, -Inf, sNaN payload, qNaN, -qNaN, -min subnormal, -max finite
    const uint16_t input_bits[] = {0x0000, 0x8000, 0xBC00, 0x7C00, 0xFC00, 0x7D55, 0x7E01, 0xFE01, 0x8001, 0xFBFF};
    const uint16_t expected_bits[] = {0x7C00, 0xFC00, 0x7E00, 0x0000, 0x7E00, 0x7F55, 0x7E01, 0xFE01, 0x7E00, 0x7E00};
    const int32_t count = (int32_t)(sizeof(input_bits) / sizeof(input_bits[0]));
    float16_t input[10];
    float16_t output[10] = {0};
    memcpy(input, input_bits, sizeof(input));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(input, output, count));
    for (int32_t i = 0; i < count; ++i)
    {
        TEST_ASSERT_EQUAL_HEX16(expected_bits[i], rsqrt_f16_bits(output[i]));
    }
#endif
}

// Every float16 bit pattern, in blocks that are not a lane multiple, against
// the float64 reference: proves the documented 0-ulp contract (#295).
void rsqrt_f16_exhaustive_arm_rsqrt_f16(void)
{
#if defined(__ARM_FP16_FORMAT_ALTERNATIVE)
    TEST_IGNORE_MESSAGE("Arm alternative half precision has no infinity or NaN encodings");
#else
    static float16_t input[RSQRT_F16_EXHAUSTIVE_BLOCK];
    static float16_t output[RSQRT_F16_EXHAUSTIVE_BLOCK];
    char msg[48];

    for (uint32_t base = 0; base < 0x10000u; base += RSQRT_F16_EXHAUSTIVE_BLOCK)
    {
        for (int32_t i = 0; i < RSQRT_F16_EXHAUSTIVE_BLOCK; ++i)
        {
            const uint16_t bits = (uint16_t)(base + (uint32_t)i);
            memcpy(&input[i], &bits, sizeof(bits));
            output[i] = (float16_t)0.0f;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(input, output, RSQRT_F16_EXHAUSTIVE_BLOCK));
        for (int32_t i = 0; i < RSQRT_F16_EXHAUSTIVE_BLOCK; ++i)
        {
            const uint16_t bits = (uint16_t)(base + (uint32_t)i);
            const uint16_t expected = rsqrt_f16_reference_bits(bits);
            const uint16_t actual = rsqrt_f16_bits(output[i]);
            if (expected != actual)
            {
                snprintf(msg, sizeof(msg), "input 0x%04x", (unsigned)bits);
                TEST_ASSERT_EQUAL_HEX16_MESSAGE(expected, actual, msg);
            }
        }
    }
#endif
}

void rsqrt_f16_arg_error_arm_rsqrt_f16(void)
{
    float16_t output[RSQRT_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(NULL, output, RSQRT_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(rsqrt_f16_input, NULL, RSQRT_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(rsqrt_f16_input, output, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(rsqrt_f16_input, output, -1));
}
