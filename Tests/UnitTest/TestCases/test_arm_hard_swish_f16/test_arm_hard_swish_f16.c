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
#include <unity.h>

#include "hard_swish_f16_data.h"

static uint16_t hard_swish_f16_bits(float16_t x)
{
    uint16_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static float16_t hard_swish_f16_from_bits(uint16_t bits)
{
    float16_t x;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

// The golden data models the single-rounded scalar leg; the MVE leg rounds the
// gate and the product separately in float16 and sits up to 2 ulp away from it
// in the curved region. Both legs are held to the combined absolute-plus-
// relative reading the kernel documents, not to an absolute bound alone.
#define HARD_SWISH_F16_TOL_ABS 1.0e-3f
#define HARD_SWISH_F16_TOL_REL 1.0e-3f
#define HARD_SWISH_F16_ASSERT_CLOSE(ref, out)                                                                          \
    TEST_ASSERT_FLOAT_WITHIN(HARD_SWISH_F16_TOL_ABS + HARD_SWISH_F16_TOL_REL * fabsf(ref), (ref), (out))

// Curved region against the reference x * relu6(x + 3) / 6 (rounded through the
// generator's compute-in-f32, round-once model); size 31 exercises the MVE
// tail-predicated final iteration. The last four inputs are the exhaustive
// sweep's worst case for the MVE leg, which is outside an absolute-only 1e-3
// and inside the combined reading asserted here.
void hard_swish_f16_curved_arm_hard_swish_f16(void)
{
    float16_t output[HARD_SWISH_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f16(hard_swish_f16_input, output, HARD_SWISH_F16_DST_SIZE));

    for (int i = 0; i < HARD_SWISH_F16_DST_SIZE; ++i)
    {
        HARD_SWISH_F16_ASSERT_CLOSE((float)hard_swish_f16_output_ref[i], (float)output[i]);
    }
}

// Identity region: x >= 3 returns x bit-exactly (gate 1, widen-narrow round
// trip is exact), up to and including the float16 maximum 65504.
void hard_swish_f16_identity_region_arm_hard_swish_f16(void)
{
    float16_t output[HARD_SWISH_F16_IDENT_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f16(hard_swish_f16_ident, output, HARD_SWISH_F16_IDENT_SIZE));

    for (int i = 0; i < HARD_SWISH_F16_IDENT_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_HEX16(hard_swish_f16_bits(hard_swish_f16_ident[i]), hard_swish_f16_bits(output[i]));
    }
}

// Zero region: x <= -3 returns exactly zero (a negative zero: negative * +0.0).
void hard_swish_f16_zero_region_arm_hard_swish_f16(void)
{
    float16_t output[HARD_SWISH_F16_IDENT_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_hard_swish_f16(hard_swish_f16_zero_region, output, HARD_SWISH_F16_IDENT_SIZE));

    for (int i = 0; i < HARD_SWISH_F16_IDENT_SIZE; ++i)
    {
        // Exactly the documented negative zero (negative * +0.0), not just
        // any zero: pin the sign bit.
        TEST_ASSERT_EQUAL_HEX16(0x8000u, hard_swish_f16_bits(output[i]));
    }
}

// Knot points: hard_swish(-3) = -0, hard_swish(0) = 0, hard_swish(3) = 3,
// hard_swish(6) = 6, all exact.
void hard_swish_f16_knots_arm_hard_swish_f16(void)
{
    const float16_t knots[4] = {(float16_t)-3.0f, (float16_t)0.0f, (float16_t)3.0f, (float16_t)6.0f};
    float16_t output[4] = {(float16_t)77.0f, (float16_t)77.0f, (float16_t)77.0f, (float16_t)77.0f};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f16(knots, output, 4));

    TEST_ASSERT_EQUAL_HEX16(0x8000u, hard_swish_f16_bits(output[0]));
    TEST_ASSERT_EQUAL_HEX16(hard_swish_f16_bits((float16_t)0.0f), hard_swish_f16_bits(output[1]));
    TEST_ASSERT_EQUAL_HEX16(hard_swish_f16_bits((float16_t)3.0f), hard_swish_f16_bits(output[2]));
    TEST_ASSERT_EQUAL_HEX16(hard_swish_f16_bits((float16_t)6.0f), hard_swish_f16_bits(output[3]));
}

// NaN propagates; +Inf returns +Inf; -Inf returns NaN ((-Inf) * 0 at the
// saturated gate, the documented IEEE consequence matching TFLite's float
// hard-swish reference).
void hard_swish_f16_nan_inf_arm_hard_swish_f16(void)
{
    const uint16_t f16_qnan = 0x7e00;
    const uint16_t f16_pos_inf = 0x7c00;
    const uint16_t f16_neg_inf = 0xfc00;
    const float16_t input[5] = {hard_swish_f16_from_bits(f16_qnan),
                                hard_swish_f16_from_bits(f16_pos_inf),
                                hard_swish_f16_from_bits(f16_neg_inf),
                                (float16_t)1.0f,
                                (float16_t)-1.0f};
    float16_t output[5] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f16(input, output, 5));

    // NaN: all-ones exponent, non-zero mantissa. Classified on the bit
    // pattern so the check cannot be folded by finite-math flags.
    TEST_ASSERT_TRUE((hard_swish_f16_bits(output[0]) & 0x7fff) > 0x7c00);
    TEST_ASSERT_EQUAL_HEX16(f16_pos_inf, hard_swish_f16_bits(output[1]));
    TEST_ASSERT_TRUE((hard_swish_f16_bits(output[2]) & 0x7fff) > 0x7c00);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, 0.6666667f, (float)output[3]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, -0.3333333f, (float)output[4]);
}

// Float16 subnormal inputs: the gate is exactly 0.5 on both legs (float16
// (x + 3) is 3 for any subnormal x, and 3 * float16(1/6) rounds to 0.5), the
// halving is exact, and the rounding to nearest-even halves the smallest
// subnormal to a signed zero on the tie.
void hard_swish_f16_denormal_arm_hard_swish_f16(void)
{
    float16_t input[HARD_SWISH_F16_DENORMAL_SIZE];
    float16_t output[HARD_SWISH_F16_DENORMAL_SIZE] = {0};

    for (int i = 0; i < HARD_SWISH_F16_DENORMAL_SIZE; ++i)
    {
        input[i] = hard_swish_f16_from_bits(hard_swish_f16_denormal_in_bits[i]);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f16(input, output, HARD_SWISH_F16_DENORMAL_SIZE));

    for (int i = 0; i < HARD_SWISH_F16_DENORMAL_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_HEX16(hard_swish_f16_denormal_ref_bits[i], hard_swish_f16_bits(output[i]));
    }
}

// Rounding shape, per leg. The witness input is one where the two legs land
// one ulp apart, so it pins each leg to its own value rather than letting a
// drifting leg hide behind a tolerance: the scalar leg widens to float32 and
// rounds once, the MVE leg rounds the gate and the product separately in
// float16 (see AmbiqAI/ns-cmsis-nn#427).
void hard_swish_f16_round_once_arm_hard_swish_f16(void)
{
    const float16_t input[1] = {hard_swish_f16_from_bits(HARD_SWISH_F16_WITNESS_IN_BITS)};
    float16_t output[1] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f16(input, output, 1));

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    TEST_ASSERT_EQUAL_HEX16(HARD_SWISH_F16_WITNESS_TWICE_BITS, hard_swish_f16_bits(output[0]));
#else
    TEST_ASSERT_EQUAL_HEX16(HARD_SWISH_F16_WITNESS_ONCE_BITS, hard_swish_f16_bits(output[0]));
#endif
    // Belt and braces: the two values really are different, so the assertion
    // above cannot pass on the wrong leg.
    TEST_ASSERT_TRUE(HARD_SWISH_F16_WITNESS_ONCE_BITS != HARD_SWISH_F16_WITNESS_TWICE_BITS);
}

// Every size 1..17: the MVE leg's tail predication must neither drop elements
// nor write past size (sentinels stay untouched).
void hard_swish_f16_tail_sizes_arm_hard_swish_f16(void)
{
    for (int32_t size = 1; size <= 17; ++size)
    {
        float16_t output[24];
        for (int i = 0; i < 24; ++i)
        {
            output[i] = (float16_t)77.0f;
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f16(hard_swish_f16_input, output, size));

        for (int32_t i = 0; i < size; ++i)
        {
            HARD_SWISH_F16_ASSERT_CLOSE((float)hard_swish_f16_output_ref[i], (float)output[i]);
        }
        for (int32_t i = size; i < 24; ++i)
        {
            TEST_ASSERT_EQUAL_FLOAT(77.0f, (float)output[i]);
        }
    }
}

// The scalar leg's float32 gate must be a correctly rounded fused multiply-add:
// input 0x3bff is the single float16 value (exhaustive sweep) where a
// separately rounded x * (1/6f) + 0.5f gate lands the narrowed product one ulp
// away. The MVE leg reaches the same bits by a different route, so this runs
// unconditionally.
void hard_swish_f16_fma_witness_arm_hard_swish_f16(void)
{
    const float16_t input[1] = {hard_swish_f16_from_bits(HARD_SWISH_F16_FMA_WITNESS_IN_BITS)};
    float16_t output[1] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f16(input, output, 1));

    TEST_ASSERT_EQUAL_HEX16(HARD_SWISH_F16_FMA_WITNESS_FUSED_BITS, hard_swish_f16_bits(output[0]));
    // Belt and braces: the unfused value really is different.
    TEST_ASSERT_TRUE(HARD_SWISH_F16_FMA_WITNESS_FUSED_BITS != HARD_SWISH_F16_FMA_WITNESS_UNFUSED_BITS);
}

void hard_swish_f16_arg_error_arm_hard_swish_f16(void)
{
    float16_t output[HARD_SWISH_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_hard_swish_f16(NULL, output, HARD_SWISH_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_hard_swish_f16(hard_swish_f16_input, NULL, HARD_SWISH_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_hard_swish_f16(hard_swish_f16_input, output, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_hard_swish_f16(hard_swish_f16_input, output, -1));
}
