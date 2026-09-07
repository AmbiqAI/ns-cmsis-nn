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

#include "rsqrt_f32_data.h"

#define RSQRT_F32_SWEEP_BLOCK 512
#define RSQRT_F32_SWEEP_RANDOM 8192
#define RSQRT_F32_ULP_TOLERANCE 1

static uint32_t rsqrt_f32_bits(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Contract reference (Include/arm_nnfunctions_flt.h, arm_rsqrt_f32): specials by bit
// pattern; positive finite inputs are the float64 result rounded once to
// float32, independent of the kernel's float32 evaluation.
static uint32_t rsqrt_f32_reference_bits(uint32_t in_bits)
{
    const uint32_t magnitude = in_bits & UINT32_C(0x7FFFFFFF);
    float32_t x;

    if (magnitude == 0)
    {
        return in_bits | UINT32_C(0x7F800000);
    }
    if (magnitude > UINT32_C(0x7F800000))
    {
        return in_bits | UINT32_C(0x00400000);
    }
    if (in_bits == UINT32_C(0x7F800000))
    {
        return UINT32_C(0);
    }
    if (in_bits & UINT32_C(0x80000000))
    {
        return UINT32_C(0x7FC00000);
    }
    memcpy(&x, &in_bits, sizeof(x));
    return rsqrt_f32_bits((float32_t)(1.0 / sqrt((double)x)));
}

// Positive finite outputs share a sign and order monotonically with their bit
// pattern, so the bit distance is the ulp distance.
static void rsqrt_f32_assert_ulp(uint32_t expected, uint32_t actual, uint32_t in_bits)
{
    const int64_t distance = (int64_t)expected - (int64_t)actual;
    char msg[48];

    if (distance > RSQRT_F32_ULP_TOLERANCE || distance < -RSQRT_F32_ULP_TOLERANCE)
    {
        snprintf(msg, sizeof(msg), "input 0x%08lx", (unsigned long)in_bits);
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(expected, actual, msg);
    }
}

static void rsqrt_f32_check_block(const uint32_t *in_bits, int32_t count)
{
    float32_t input[RSQRT_F32_SWEEP_BLOCK];
    float32_t output[RSQRT_F32_SWEEP_BLOCK] = {0};

    memcpy(input, in_bits, (size_t)count * sizeof(input[0]));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f32(input, output, count));
    for (int32_t i = 0; i < count; ++i)
    {
        rsqrt_f32_assert_ulp(rsqrt_f32_reference_bits(in_bits[i]), rsqrt_f32_bits(output[i]), in_bits[i]);
    }
}

void rsqrt_f32_arm_rsqrt_f32(void)
{
    float32_t output[RSQRT_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f32(rsqrt_f32_input, output, RSQRT_F32_DST_SIZE));
    for (int32_t i = 0; i < RSQRT_F32_DST_SIZE; ++i)
    {
        rsqrt_f32_assert_ulp(rsqrt_f32_bits(rsqrt_f32_output_ref[i]), rsqrt_f32_bits(output[i]), rsqrt_f32_bits(rsqrt_f32_input[i]));
    }
}

void rsqrt_f32_in_place_arm_rsqrt_f32(void)
{
    float32_t values[RSQRT_F32_DST_SIZE];
    memcpy(values, rsqrt_f32_input, sizeof(values));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f32(values, values, RSQRT_F32_DST_SIZE));
    for (int32_t i = 0; i < RSQRT_F32_DST_SIZE; ++i)
    {
        rsqrt_f32_assert_ulp(rsqrt_f32_bits(rsqrt_f32_output_ref[i]), rsqrt_f32_bits(values[i]), rsqrt_f32_bits(rsqrt_f32_input[i]));
    }
}

void rsqrt_f32_special_values_arm_rsqrt_f32(void)
{
    // +0, -0, -1, +Inf, -Inf, sNaN payload, qNaN, -qNaN payload, -min subnormal, -max finite
    const uint32_t input_bits[] = {0x00000000, 0x80000000, 0xBF800000, 0x7F800000, 0xFF800000,
                                   0x7FA00001, 0x7FC00000, 0xFFC12345, 0x80000001, 0xFF7FFFFF};
    const uint32_t expected_bits[] = {0x7F800000, 0xFF800000, 0x7FC00000, 0x00000000, 0x7FC00000, 0x7FE00001, 0x7FC00000, 0xFFC12345, 0x7FC00000, 0x7FC00000};
    const int32_t count = (int32_t)(sizeof(input_bits) / sizeof(input_bits[0]));
    float32_t input[10];
    float32_t output[10] = {0};
    memcpy(input, input_bits, sizeof(input));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f32(input, output, count));
    for (int32_t i = 0; i < count; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(expected_bits[i], rsqrt_f32_bits(output[i]));
    }
}

// Every exponent boundary (2^e and its two neighbours, e = -149..127), every
// subnormal power of two, the extremes, and a log-uniform random sample,
// against the float64 reference at the documented tolerance (#295).
void rsqrt_f32_sweep_arm_rsqrt_f32(void)
{
    static uint32_t bits[RSQRT_F32_SWEEP_BLOCK];
    int32_t n = 0;
    uint32_t state = 0x2950001u;

    for (uint32_t e = 1; e <= 254; ++e)
    {
        const uint32_t pow2 = e << 23;
        bits[n++] = pow2 - 1;
        bits[n++] = pow2;
        bits[n++] = pow2 + 1;
        if (n + 3 > RSQRT_F32_SWEEP_BLOCK)
        {
            rsqrt_f32_check_block(bits, n);
            n = 0;
        }
    }
    for (uint32_t s = 0; s < 23; ++s)
    {
        bits[n++] = UINT32_C(1) << s;
    }
    bits[n++] = UINT32_C(0x00000003);
    bits[n++] = UINT32_C(0x000FFFFF);
    bits[n++] = UINT32_C(0x007FFFFF);
    bits[n++] = UINT32_C(0x7F7FFFFF);
    bits[n++] = UINT32_C(0x7F7FFFFE);
    rsqrt_f32_check_block(bits, n);
    n = 0;

    for (int32_t i = 0; i < RSQRT_F32_SWEEP_RANDOM; ++i)
    {
        // xorshift32; a uniform bit pattern over the positive normals is a
        // log-uniform value over every binade.
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        bits[n++] = UINT32_C(0x00800000) + (state % (UINT32_C(0x7F7FFFFF) - UINT32_C(0x00800000) + 1));
        if (n == RSQRT_F32_SWEEP_BLOCK)
        {
            rsqrt_f32_check_block(bits, n);
            n = 0;
        }
    }
    if (n > 0)
    {
        rsqrt_f32_check_block(bits, n);
    }

    // 1 / sqrt(4^k) = 2^-k is representable, so the contract is exact there
    // (every normal 4^k: exponent field 127 + 2k in 1..253).
    for (int32_t k = -63; k <= 63; ++k)
    {
        const uint32_t in_bits = (uint32_t)(127 + 2 * k) << 23;
        const uint32_t expected = (uint32_t)(127 - k) << 23;
        float32_t x, y;
        memcpy(&x, &in_bits, sizeof(x));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f32(&x, &y, 1));
        TEST_ASSERT_EQUAL_HEX32(expected, rsqrt_f32_bits(y));
    }
}

void rsqrt_f32_arg_error_arm_rsqrt_f32(void)
{
    float32_t output[RSQRT_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f32(NULL, output, RSQRT_F32_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f32(rsqrt_f32_input, NULL, RSQRT_F32_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f32(rsqrt_f32_input, output, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f32(rsqrt_f32_input, output, -1));
}
