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

#include "hard_swish_f32_data.h"

static uint32_t hard_swish_f32_bits(float32_t x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

// Curved region against the float64 reference x * relu6(x + 3) / 6 (rounded
// through the generator's once-rounded model); size 35 exercises the MVE
// tail-predicated final iteration.
void hard_swish_f32_curved_arm_hard_swish_f32(void)
{
    float32_t output[HARD_SWISH_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f32(hard_swish_f32_input, output, HARD_SWISH_F32_DST_SIZE));

    for (int i = 0; i < HARD_SWISH_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, hard_swish_f32_output_ref[i], output[i]);
    }
}

// Identity region: x >= 3 returns x bit-exactly (the gate saturates to 1).
void hard_swish_f32_identity_region_arm_hard_swish_f32(void)
{
    float32_t output[HARD_SWISH_F32_IDENT_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f32(hard_swish_f32_ident, output, HARD_SWISH_F32_IDENT_SIZE));

    for (int i = 0; i < HARD_SWISH_F32_IDENT_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(hard_swish_f32_bits(hard_swish_f32_ident[i]), hard_swish_f32_bits(output[i]));
    }
}

// Zero region: x <= -3 returns exactly zero (a negative zero: negative * +0.0).
void hard_swish_f32_zero_region_arm_hard_swish_f32(void)
{
    float32_t output[HARD_SWISH_F32_IDENT_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_hard_swish_f32(hard_swish_f32_zero_region, output, HARD_SWISH_F32_IDENT_SIZE));

    for (int i = 0; i < HARD_SWISH_F32_IDENT_SIZE; ++i)
    {
        // Exactly the documented negative zero (negative * +0.0), not just
        // any zero: pin the sign bit.
        TEST_ASSERT_EQUAL_HEX32(0x80000000u, hard_swish_f32_bits(output[i]));
    }
}

// Knot points: hard_swish(-3) = -0, hard_swish(0) = 0, hard_swish(3) = 3,
// hard_swish(6) = 6, all exact.
void hard_swish_f32_knots_arm_hard_swish_f32(void)
{
    const float32_t knots[4] = {-3.0f, 0.0f, 3.0f, 6.0f};
    float32_t output[4] = {77.0f, 77.0f, 77.0f, 77.0f};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f32(knots, output, 4));

    TEST_ASSERT_EQUAL_HEX32(0x80000000u, hard_swish_f32_bits(output[0]));
    TEST_ASSERT_EQUAL_HEX32(hard_swish_f32_bits(0.0f), hard_swish_f32_bits(output[1]));
    TEST_ASSERT_EQUAL_HEX32(hard_swish_f32_bits(3.0f), hard_swish_f32_bits(output[2]));
    TEST_ASSERT_EQUAL_HEX32(hard_swish_f32_bits(6.0f), hard_swish_f32_bits(output[3]));
}

// NaN propagates; +Inf returns +Inf; -Inf returns NaN ((-Inf) * 0 at the
// saturated gate, the documented IEEE consequence matching TFLite's float
// hard-swish reference).
void hard_swish_f32_nan_inf_arm_hard_swish_f32(void)
{
    const float32_t input[5] = {NAN, INFINITY, -INFINITY, 1.0f, -1.0f};
    float32_t output[5] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f32(input, output, 5));

    TEST_ASSERT_FLOAT_IS_NAN(output[0]);
    TEST_ASSERT_FLOAT_IS_INF(output[1]);
    TEST_ASSERT_FLOAT_IS_NAN(output[2]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, 0.6666667f, output[3]);
    TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, -0.3333333f, output[4]);
}

// Denormal inputs: the gate is 0.5 there, and the halved result narrows with
// round-to-nearest-even (the smallest subnormal rounds to a signed zero).
void hard_swish_f32_denormal_arm_hard_swish_f32(void)
{
    float32_t output[HARD_SWISH_F32_DENORMAL_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_hard_swish_f32(hard_swish_f32_denormal_in, output, HARD_SWISH_F32_DENORMAL_SIZE));

    for (int i = 0; i < HARD_SWISH_F32_DENORMAL_SIZE; ++i)
    {
        const uint32_t got = hard_swish_f32_bits(output[i]);
#if defined(USING_FVP_CORSTONE_300)
        // The FVP Corstone-300 model's MVE pipeline flushes f32 subnormal
        // operands and results to a signed zero regardless of FPSCR.FZ
        // (FPSCR reads FZ=0 there while scalar VFP arithmetic on the same
        // run keeps subnormals; real silicon with FZ clear is expected to
        // keep them, but is unverified). Arm GCC's -Ofast additionally
        // links crtfastmath.o, which really does set FZ at startup. So on
        // the FVP accept the flushed signed zero from either cause; exact
        // once-rounded bits are required everywhere else.
        const uint32_t flushed = hard_swish_f32_bits(hard_swish_f32_denormal_in[i]) & 0x80000000u;
        if (got == flushed)
        {
            continue;
        }
#endif
        TEST_ASSERT_EQUAL_HEX32(hard_swish_f32_denormal_ref_bits[i], got);
    }
}

// Every size 1..9: the MVE leg's tail predication must neither drop elements
// nor write past size (sentinels stay untouched).
void hard_swish_f32_tail_sizes_arm_hard_swish_f32(void)
{
    for (int32_t size = 1; size <= 9; ++size)
    {
        float32_t output[16];
        for (int i = 0; i < 16; ++i)
        {
            output[i] = 77.0f;
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f32(hard_swish_f32_input, output, size));

        for (int32_t i = 0; i < size; ++i)
        {
            TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, hard_swish_f32_output_ref[i], output[i]);
        }
        for (int32_t i = size; i < 16; ++i)
        {
            TEST_ASSERT_EQUAL_FLOAT(77.0f, output[i]);
        }
    }
}

// The gate must be a correctly rounded fused multiply-add: at this input a
// separately rounded x * (1/6f) + 0.5f gate yields a product one ulp away.
// Bit-exact so an unfused-gate regression fails here.
void hard_swish_f32_fma_witness_arm_hard_swish_f32(void)
{
    float32_t input[1];
    float32_t output[1] = {0};
    const uint32_t in_bits = HARD_SWISH_F32_FMA_WITNESS_IN_BITS;
    memcpy(&input[0], &in_bits, sizeof(input[0]));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_hard_swish_f32(input, output, 1));

    TEST_ASSERT_EQUAL_HEX32(HARD_SWISH_F32_FMA_WITNESS_FUSED_BITS, hard_swish_f32_bits(output[0]));
    // Belt and braces: the unfused value really is different.
    TEST_ASSERT_TRUE(HARD_SWISH_F32_FMA_WITNESS_FUSED_BITS != HARD_SWISH_F32_FMA_WITNESS_UNFUSED_BITS);
}

void hard_swish_f32_arg_error_arm_hard_swish_f32(void)
{
    float32_t output[HARD_SWISH_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_hard_swish_f32(NULL, output, HARD_SWISH_F32_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_hard_swish_f32(hard_swish_f32_input, NULL, HARD_SWISH_F32_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_hard_swish_f32(hard_swish_f32_input, output, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_hard_swish_f32(hard_swish_f32_input, output, -1));
}
