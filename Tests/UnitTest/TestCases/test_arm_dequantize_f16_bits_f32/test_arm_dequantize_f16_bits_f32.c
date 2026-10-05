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
#include <string.h>

#include "unity.h"

/* Reference widening, written independently of the kernel: the value from its definition, a NaN from its bits */
static uint32_t dq_reference(const uint32_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    const int32_t exp = (int32_t)((h >> 10) & 0x1Fu);
    const uint32_t mant = h & 0x3FFu;
    if (exp == 0x1F)
    {
        return sign | 0x7F800000u | (mant << 13);
    }
    const float magnitude = exp == 0 ? ldexpf((float)mant, -24) : ldexpf((float)(1024u + mant), exp - 25);
    uint32_t bits;
    memcpy(&bits, &magnitude, sizeof(bits));
    return sign | bits;
}

/* Every half bit pattern, in blocks of varying size so that vector tails of every length run */
void all_halves_arm_dequantize_f16_bits_f32(void)
{
    static uint16_t in[1024];
    static float out[1024 + 1];
    const int32_t blocks[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 1024};
    uint32_t h = 0;
    size_t b = 0;
    while (h < 0x10000u)
    {
        int32_t n = blocks[b++ % (sizeof(blocks) / sizeof(blocks[0]))];
        if (h + (uint32_t)n > 0x10000u)
        {
            n = (int32_t)(0x10000u - h);
        }
        for (int32_t i = 0; i < n; i++)
        {
            in[i] = (uint16_t)(h + (uint32_t)i);
        }
        const uint32_t guard = 0xDEADBEEFu;
        memcpy(&out[n], &guard, sizeof(guard));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_f16_bits_f32(in, out, n));
        for (int32_t i = 0; i < n; i++)
        {
            uint32_t got;
            memcpy(&got, &out[i], sizeof(got));
            TEST_ASSERT_EQUAL_HEX32(dq_reference(in[i]), got);
        }
        uint32_t after;
        memcpy(&after, &out[n], sizeof(after));
        TEST_ASSERT_EQUAL_HEX32(guard, after);
        h += (uint32_t)n;
    }
}

void arguments_arm_dequantize_f16_bits_f32(void)
{
    uint16_t in[1] = {0x3C00u};
    float out[1] = {0.0f};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_f16_bits_f32(in, out, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_f16_bits_f32(NULL, NULL, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_dequantize_f16_bits_f32(in, out, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_dequantize_f16_bits_f32(NULL, out, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_dequantize_f16_bits_f32(in, NULL, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_f16_bits_f32(in, out, 1));
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out[0]);
}
