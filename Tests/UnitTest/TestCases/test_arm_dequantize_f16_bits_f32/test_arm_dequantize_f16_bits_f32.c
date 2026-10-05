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

/* Every half bit pattern, in blocks of varying size so that vector tails of every length run. The second pass
   starts one element into the buffer, so each half is also converted at an odd index, next to other partners in
   the two-halves-per-word path. */
static void dq_sweep(const int32_t offset)
{
    static uint16_t buffer[1024 + 1];
    static float out[1024 + 1];
    uint16_t *in = buffer + offset;
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

void all_halves_arm_dequantize_f16_bits_f32(void)
{
    dq_sweep(0);
    dq_sweep(1);
}

/* Inf and NaN next to finite halves in both positions of a word, including both signaling NaNs that border Inf */
void special_pairs_arm_dequantize_f16_bits_f32(void)
{
    static const uint16_t specials[] = {0x7C00u, 0xFC00u, 0x7C01u, 0xFC01u, 0x7E00u, 0x7DFFu, 0xFFFFu};
    static const uint16_t finites[] = {0x0000u, 0x8000u, 0x0001u, 0x3C00u, 0x7BFFu, 0xFBFFu};
    uint16_t in[2];
    float out[2];
    for (size_t a = 0; a < sizeof(specials) / sizeof(specials[0]); a++)
    {
        for (size_t b = 0; b < sizeof(finites) / sizeof(finites[0]); b++)
        {
            for (int order = 0; order < 2; order++)
            {
                in[order] = specials[a];
                in[1 - order] = finites[b];
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_dequantize_f16_bits_f32(in, out, 2));
                for (int k = 0; k < 2; k++)
                {
                    uint32_t got;
                    memcpy(&got, &out[k], sizeof(got));
                    TEST_ASSERT_EQUAL_HEX32(dq_reference(in[k]), got);
                }
            }
        }
    }
}

/* With FPSCR.AHP set the FPU reads exponent 31 as a number; the result must not change */
void alternative_half_arm_dequantize_f16_bits_f32(void)
{
#if defined(__ARM_FP) && defined(__ARM_ARCH_PROFILE) && (__ARM_ARCH_PROFILE == 'M')
    static const uint16_t in[] = {0x7C00u, 0xFC00u, 0x7C01u, 0x7E00u, 0x3C00u, 0x0001u, 0x7BFFu, 0xFFFFu, 0x7C00u};
    const int32_t n = (int32_t)(sizeof(in) / sizeof(in[0]));
    float out[sizeof(in) / sizeof(in[0])];
    uint32_t fpscr;
    __asm volatile("vmrs %0, fpscr" : "=r"(fpscr));
    __asm volatile("vmsr fpscr, %0" : : "r"(fpscr | (1u << 26)) : "memory");
    const arm_cmsis_nn_status status = arm_dequantize_f16_bits_f32(in, out, n);
    __asm volatile("vmsr fpscr, %0" : : "r"(fpscr) : "memory");
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    for (int32_t k = 0; k < n; k++)
    {
        uint32_t got;
        memcpy(&got, &out[k], sizeof(got));
        TEST_ASSERT_EQUAL_HEX32(dq_reference(in[k]), got);
    }
#else
    TEST_IGNORE_MESSAGE("no M-profile FPU");
#endif
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
