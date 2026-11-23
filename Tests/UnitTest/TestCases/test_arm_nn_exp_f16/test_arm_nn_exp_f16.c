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
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "../../../../Include/Internal/arm_nn_exp_flt.h"
#include "exp_f16_data.h"

#define EXP_F16_SENTINEL UINT16_C(0xA5A5)
#define EXP_F16_GUARD 16 /* eight 32-bit words */
#define EXP_F16_MAX_TAIL 17
#define EXP_F16_LONG 4096
#define EXP_F16_BLOCK 4096

// The Helium float16 body is compiled where arm_nn_exp_f16.c selects it.
#if defined(ARM_NN_EXP_MVE_F32) && defined(ARM_MATH_MVE_FLOAT16)
    #define EXP_F16_HELIUM 1
#else
    #define EXP_F16_HELIUM 0
#endif

/*
 * Inputs whose float16 result on the Helium leg differs from the scalar leg, out of all
 * 65 536, as the header states (Include/arm_nnfunctions_flt.h, arm_nn_exp_f16).
 */
#define EXP_F16_HELIUM_DIFFERENCES 0

static uint16_t exp_f16_bits(float16_t value)
{
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float16_t exp_f16_from_bits(uint16_t bits)
{
    float16_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/*
 * The scalar leg's contract: a NaN is the input with its quiet bit set; every other
 * input is the float32 scalar body at the widened input, rounded once to float16. The
 * volatile read keeps a vectorizing compiler from folding the reference into lanes.
 */
static uint16_t exp_f16_scalar_reference(uint16_t in_bits)
{
    if ((in_bits & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
    {
        return (uint16_t)(in_bits | UINT16_C(0x0200));
    }
    volatile float16_t x = exp_f16_from_bits(in_bits);
    const float32_t y = arm_nn_exp_scalar_f32((float32_t)x);
    return exp_f16_bits((float16_t)y);
}

// A pseudo-random binary16 pattern, every value class included.
static uint16_t exp_f16_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return (uint16_t)(*state >> 16);
}

// All 65 536 inputs against the scalar leg's contract; the Helium leg may differ only on
// the inputs the header names, and those are counted exactly.
void exp_f16_exhaustive_arm_nn_exp_f16(void)
{
    static uint16_t input[EXP_F16_BLOCK];
    static uint16_t output[EXP_F16_BLOCK];
    uint32_t differences = 0;
    char msg[64];

    for (uint32_t base = 0; base < UINT32_C(0x10000); base += EXP_F16_BLOCK)
    {
        for (uint32_t i = 0; i < EXP_F16_BLOCK; ++i)
        {
            input[i] = (uint16_t)(base + i);
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_exp_f16((const float16_t *)input, (float16_t *)output, EXP_F16_BLOCK));
        for (uint32_t i = 0; i < EXP_F16_BLOCK; ++i)
        {
            const uint16_t expected = exp_f16_scalar_reference(input[i]);
            if (expected == output[i])
            {
                continue;
            }
#if EXP_F16_HELIUM
            ++differences;
            continue;
#endif
            snprintf(msg, sizeof(msg), "input 0x%04x", (unsigned)input[i]);
            TEST_ASSERT_EQUAL_HEX16_MESSAGE(expected, output[i], msg);
        }
    }
#if EXP_F16_HELIUM
    TEST_ASSERT_EQUAL_UINT32(EXP_F16_HELIUM_DIFFERENCES, differences);
#else
    TEST_ASSERT_EQUAL_UINT32(0, differences);
#endif
}

// The generated inputs against the binary16 nearest their exponential: a NaN is the
// input quieted, and every other result is that nearest binary16 or one of its two
// neighbours, which one rounding of a float32 result within EXP_F32 allows.
void exp_f16_golden_arm_nn_exp_f16(void)
{
    static uint16_t output[EXP_F16_CASES];
    char msg[64];

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_exp_f16((const float16_t *)exp_f16_input_bits, (float16_t *)output, EXP_F16_CASES));
    for (int32_t i = 0; i < EXP_F16_CASES; ++i)
    {
        const uint16_t in_bits = exp_f16_input_bits[i];
        const uint16_t reference = exp_f16_reference_bits[i];
        snprintf(msg, sizeof(msg), "input 0x%04x output 0x%04x", (unsigned)in_bits, (unsigned)output[i]);
        if ((in_bits & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
        {
            TEST_ASSERT_EQUAL_HEX16_MESSAGE(in_bits | UINT16_C(0x0200), output[i], msg);
            continue;
        }
        TEST_ASSERT_TRUE_MESSAGE(output[i] <= UINT16_C(0x7C00), msg);
        const int32_t distance = (int32_t)output[i] - (int32_t)reference;
        TEST_ASSERT_TRUE_MESSAGE(distance >= -1 && distance <= 1, msg);
    }
}

void exp_f16_special_values_arm_nn_exp_f16(void)
{
    // +0, -0, +inf, -inf, qNaN, -qNaN payload, sNaN payload, -sNaN, +-min normal,
    // smallest and largest subnormals, +-max, 11.0859375 (largest finite result), 11.09375
    const uint16_t input_bits[] = {0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0xFE5A, 0x7C01, 0xFD00,
                                   0x0400, 0x8400, 0x0001, 0x83FF, 0x7BFF, 0xFBFF, 0x498B, 0x498C};
    const uint16_t expected_bits[] = {0x3C00, 0x3C00, 0x7C00, 0x0000, 0x7E00, 0xFE5A, 0x7E01, 0xFF00,
                                      0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x7C00, 0x0000, 0x7BF7, 0x7C00};
    const int32_t count = (int32_t)(sizeof(input_bits) / sizeof(input_bits[0]));
    uint16_t output[sizeof(input_bits) / sizeof(input_bits[0])];

    memset(output, 0, sizeof(output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_exp_f16((const float16_t *)input_bits, (float16_t *)output, count));
    TEST_ASSERT_EQUAL_HEX16_ARRAY(expected_bits, output, count);
}

void exp_f16_arg_error_arm_nn_exp_f16(void)
{
    const int32_t sizes[] = {0, -1, INT32_MIN};
    const uint16_t input[4] = {0x0000, 0x3C00, 0xBC00, 0x4000};
    uint16_t output[4];

    for (int32_t i = 0; i < 4; ++i)
    {
        output[i] = EXP_F16_SENTINEL;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_exp_f16(NULL, (float16_t *)output, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_exp_f16((const float16_t *)input, NULL, 4));
    for (uint32_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_nn_exp_f16((const float16_t *)input, (float16_t *)output, sizes[s]));
    }
    for (int32_t i = 0; i < 4; ++i)
    {
        TEST_ASSERT_EQUAL_HEX16(EXP_F16_SENTINEL, output[i]);
    }
}

// output == input gives the bits separate buffers give, at every tail length and a long block.
void exp_f16_in_place_arm_nn_exp_f16(void)
{
    static uint16_t input[EXP_F16_LONG];
    static uint16_t separate[EXP_F16_LONG];
    static uint16_t values[EXP_F16_LONG];
    uint32_t state = 0x1E5u;

    for (int32_t i = 0; i < EXP_F16_LONG; ++i)
    {
        input[i] = exp_f16_next(&state);
    }
    for (int32_t length = 1; length <= EXP_F16_MAX_TAIL + 1; ++length)
    {
        const int32_t n = length <= EXP_F16_MAX_TAIL ? length : EXP_F16_LONG;
        memcpy(values, input, (size_t)n * sizeof(values[0]));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_exp_f16((const float16_t *)input, (float16_t *)separate, n));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_exp_f16((const float16_t *)values, (float16_t *)values, n));
        TEST_ASSERT_EQUAL_HEX16_ARRAY(separate, values, n);
    }
}

// Sixteen sentinel halves (eight words) on both sides of the input and the output are
// unchanged, and so is the input, for every length 1 to 17: each tail of the four-lane
// and the eight-lane predicates.
void exp_f16_guards_arm_nn_exp_f16(void)
{
    uint16_t input[EXP_F16_GUARD + EXP_F16_MAX_TAIL + EXP_F16_GUARD];
    uint16_t output[EXP_F16_GUARD + EXP_F16_MAX_TAIL + EXP_F16_GUARD];
    uint16_t values[EXP_F16_MAX_TAIL];
    uint32_t state = 0xC0FFEEu;

    for (int32_t length = 1; length <= EXP_F16_MAX_TAIL; ++length)
    {
        for (int32_t i = 0; i < EXP_F16_GUARD + EXP_F16_MAX_TAIL + EXP_F16_GUARD; ++i)
        {
            input[i] = EXP_F16_SENTINEL;
            output[i] = EXP_F16_SENTINEL;
        }
        for (int32_t i = 0; i < length; ++i)
        {
            values[i] = exp_f16_next(&state);
            input[EXP_F16_GUARD + i] = values[i];
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_exp_f16((const float16_t *)&input[EXP_F16_GUARD],
                                         (float16_t *)&output[EXP_F16_GUARD],
                                         length));
        for (int32_t i = 0; i < EXP_F16_GUARD; ++i)
        {
            TEST_ASSERT_EQUAL_HEX16(EXP_F16_SENTINEL, input[i]);
            TEST_ASSERT_EQUAL_HEX16(EXP_F16_SENTINEL, output[i]);
            TEST_ASSERT_EQUAL_HEX16(EXP_F16_SENTINEL, input[EXP_F16_GUARD + length + i]);
            TEST_ASSERT_EQUAL_HEX16(EXP_F16_SENTINEL, output[EXP_F16_GUARD + length + i]);
        }
        TEST_ASSERT_EQUAL_HEX16_ARRAY(values, &input[EXP_F16_GUARD], length);
    }
}

/*
 * The FPSCR contract of the header, on Arm targets: with FZ, FZ16 and DN set every one
 * of the 65 536 results keeps the bits it has with them clear, because a NaN is decided
 * on its bits, a float32 result that FZ flushes is below every binary16, and a conversion
 * to binary16 does not flush. The control bits are restored and checked unchanged by the
 * call; the cumulative exception flags are not asserted. Compiles away on the host.
 */
void exp_f16_fp_controls_arm_nn_exp_f16(void)
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
    static uint16_t input[EXP_F16_BLOCK];
    static uint16_t plain[EXP_F16_BLOCK];
    static uint16_t flushed[EXP_F16_BLOCK];
    /* FPSCR: FZ16[19], FZ[24], DN[25]. */
    const uint32_t controls = (UINT32_C(1) << 19) | (UINT32_C(1) << 24) | (UINT32_C(1) << 25);
    const uint32_t rounding = UINT32_C(3) << 22;
    uint32_t original;
    uint32_t after;

    __asm volatile("vmrs %0, fpscr" : "=r"(original));
    const uint32_t base = original & ~(controls | rounding);
    for (uint32_t start = 0; start < UINT32_C(0x10000); start += EXP_F16_BLOCK)
    {
        for (uint32_t i = 0; i < EXP_F16_BLOCK; ++i)
        {
            input[i] = (uint16_t)(start + i);
        }
        __asm volatile("vmsr fpscr, %0" ::"r"(base) : "memory");
        const arm_cmsis_nn_status plain_status =
            arm_nn_exp_f16((const float16_t *)input, (float16_t *)plain, EXP_F16_BLOCK);
        __asm volatile("vmsr fpscr, %0" ::"r"(base | controls) : "memory");
        const arm_cmsis_nn_status flushed_status =
            arm_nn_exp_f16((const float16_t *)input, (float16_t *)flushed, EXP_F16_BLOCK);
        __asm volatile("vmrs %0, fpscr" : "=r"(after));
        __asm volatile("vmsr fpscr, %0" ::"r"(original) : "memory");

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, plain_status);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, flushed_status);
        TEST_ASSERT_EQUAL_HEX32(controls, after & (controls | rounding));
        TEST_ASSERT_EQUAL_HEX16_ARRAY(plain, flushed, EXP_F16_BLOCK);
    }
#endif
}

// Every generated input against the exact bits arm_nn_exp_f16 returns, which the generator
// computes from a port of the scalar body in exact arithmetic, rounded once to binary16.
void exp_f16_exact_bits_arm_nn_exp_f16(void)
{
    static uint16_t output[EXP_F16_CASES];

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_exp_f16((const float16_t *)exp_f16_input_bits, (float16_t *)output, EXP_F16_CASES));
    TEST_ASSERT_EQUAL_HEX16_ARRAY(exp_f16_exact_bits, output, EXP_F16_CASES);
}
