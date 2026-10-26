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

#include "../../../../Include/Internal/arm_nn_exp_flt.h"
#include "exp_f32_data.h"

/*
 * The bound of the contract (Include/arm_nnfunctions_flt.h, arm_nn_exp_f32):
 * |y - e| <= EXP_F32_ATOL + EXP_F32_RTOL * e against the float64 exponential e.
 * The values are helia-ml's EXP_F32 (python/helia_ml/tolerance.py), measured there
 * over every float32 argument of the body this kernel ports bit for bit.
 */
#define EXP_F32_ATOL 0x1.0p-149
#define EXP_F32_RTOL 2.5e-7

#define EXP_F32_SENTINEL UINT32_C(0xA5A5A5A5)
#define EXP_F32_GUARD 8
#define EXP_F32_MAX_TAIL 17
#define EXP_F32_LONG 4096

// The tier comparison: every 4 099th bit pattern, and every input of two dense ranges.
#define EXP_F32_TIER_STRIDE UINT32_C(4099)
#define EXP_F32_TIER_BLOCK 1024
#define EXP_F32_DENSE_LOW_FIRST UINT32_C(0xC2AE0000)  /* -87.0 */
#define EXP_F32_DENSE_LOW_LAST UINT32_C(0xC2D00000)   /* -104.0 */
#define EXP_F32_DENSE_HIGH_FIRST UINT32_C(0x42B00000) /* 88.0 */
#define EXP_F32_DENSE_HIGH_LAST UINT32_C(0x42B18000)  /* 88.75 */
/*
 * Inputs whose scalar result is subnormal: every bit pattern from 0xC2AEAC50
 * (-87.3365479f) to 0xC2CFF0A4 (-103.97f), all inside the dense range [-104, -87].
 * The Helium body returns +0 for each of them and equals the scalar body elsewhere.
 */
#define EXP_F32_SUBNORMAL_RESULTS 2180181

static uint32_t exp_f32_bits(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float32_t exp_f32_from_bits(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static double exp_f32_from_bits64(uint64_t bits)
{
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// True where a result below 2^-126 may read +0: the Helium body flushes it whatever
// FPSCR.FZ holds, and the scalar body follows FZ (or a host's FTZ), probed at run time.
static int exp_f32_subnormal_may_flush(void)
{
#if defined(ARM_NN_EXP_MVE_F32)
    return 1;
#else
    volatile float32_t smallest_normal = 0x1.0p-126f;
    volatile float32_t half = 0.5f;
    return exp_f32_bits(smallest_normal * half) == 0;
#endif
}

// A pseudo-random float32 in [-110, 95], with a NaN or an infinity now and then.
static uint32_t exp_f32_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    switch (*state % 16)
    {
    case 0:
        return UINT32_C(0x7FC00000) | (*state >> 12);
    case 1:
        return (*state & UINT32_C(0x80000000)) | UINT32_C(0x7F800000);
    default:
        return exp_f32_bits(-110.0f + 205.0f * (float32_t)(*state >> 8) * 0x1.0p-24f);
    }
}

// The contract of Include/arm_nnfunctions_flt.h for one input: specials exact, every
// other result within the bound of the float64 exponential.
static void exp_f32_check(uint32_t in_bits, double reference, uint32_t out_bits)
{
    const uint32_t magnitude = in_bits & UINT32_C(0x7FFFFFFF);
    char msg[64];

    snprintf(msg, sizeof(msg), "input 0x%08lx output 0x%08lx", (unsigned long)in_bits, (unsigned long)out_bits);
    if (magnitude > UINT32_C(0x7F800000))
    {
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(in_bits | UINT32_C(0x00400000), out_bits, msg);
        return;
    }
    if ((in_bits & UINT32_C(0x80000000)) && magnitude > UINT32_C(0x42CFF0A4))
    {
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(0, out_bits, msg);
        return;
    }
    if (!(in_bits & UINT32_C(0x80000000)) && in_bits > UINT32_C(0x42B17217))
    {
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(UINT32_C(0x7F800000), out_bits, msg);
        return;
    }
    if (magnitude == 0)
    {
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(UINT32_C(0x3F800000), out_bits, msg);
        return;
    }
    TEST_ASSERT_TRUE_MESSAGE(out_bits < UINT32_C(0x7F800000), msg);
    if (out_bits == 0 && reference < 0x1.0p-126 && exp_f32_subnormal_may_flush())
    {
        return;
    }
    const double y = (double)exp_f32_from_bits(out_bits);
    TEST_ASSERT_TRUE_MESSAGE(fabs(y - reference) <= EXP_F32_ATOL + EXP_F32_RTOL * reference, msg);
}

// Every generated input: the specials, both thresholds and their neighbours, the
// subnormal boundary, each reduction boundary (k + 0.5) ln 2 and a stride of [-104, 89].
void exp_f32_golden_arm_nn_exp_f32(void)
{
    static float32_t input[EXP_F32_CASES];
    static float32_t output[EXP_F32_CASES];

    memcpy(input, exp_f32_input_bits, sizeof(input));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_exp_f32(input, output, EXP_F32_CASES));
    for (int32_t i = 0; i < EXP_F32_CASES; ++i)
    {
        exp_f32_check(exp_f32_input_bits[i], exp_f32_from_bits64(exp_f32_reference_bits[i]), exp_f32_bits(output[i]));
    }
}

void exp_f32_special_values_arm_nn_exp_f32(void)
{
    // +0, -0, +inf, -inf, qNaN, -qNaN payload, sNaN payload, -sNaN, +-FLT_MIN, smallest and
    // largest subnormals, 0x42B17217 (largest finite result), 0x42B17218, -103.97f's
    // successor, -FLT_MAX, +FLT_MAX
    const uint32_t input_bits[] = {0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC12345,
                                   0x7F800001, 0xFFA00000, 0x00800000, 0x80800000, 0x00000001, 0x807FFFFF,
                                   0x42B17217, 0x42B17218, 0xC2CFF0A5, 0xFF7FFFFF, 0x7F7FFFFF};
    const uint32_t expected_bits[] = {0x3F800000, 0x3F800000, 0x7F800000, 0x00000000, 0x7FC00000, 0xFFC12345,
                                      0x7FC00001, 0xFFE00000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000,
                                      0x7F7FFF84, 0x7F800000, 0x00000000, 0x00000000, 0x7F800000};
    const int32_t count = (int32_t)(sizeof(input_bits) / sizeof(input_bits[0]));
    float32_t input[sizeof(input_bits) / sizeof(input_bits[0])];
    float32_t output[sizeof(input_bits) / sizeof(input_bits[0])];

    memcpy(input, input_bits, sizeof(input));
    memset(output, 0, sizeof(output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_exp_f32(input, output, count));
    for (int32_t i = 0; i < count; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(expected_bits[i], exp_f32_bits(output[i]));
    }
}

void exp_f32_arg_error_arm_nn_exp_f32(void)
{
    const int32_t sizes[] = {0, -1, INT32_MIN};
    float32_t input[4] = {0.0f, 1.0f, -1.0f, 2.0f};
    uint32_t output[4];

    for (int32_t i = 0; i < 4; ++i)
    {
        output[i] = EXP_F32_SENTINEL;
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_exp_f32(NULL, (float32_t *)output, 4));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_exp_f32(input, NULL, 4));
    for (uint32_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_exp_f32(input, (float32_t *)output, sizes[s]));
    }
    for (int32_t i = 0; i < 4; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(EXP_F32_SENTINEL, output[i]);
    }
}

// output == input gives the bits separate buffers give, at every tail length and a long block.
void exp_f32_in_place_arm_nn_exp_f32(void)
{
    static float32_t input[EXP_F32_LONG];
    static float32_t separate[EXP_F32_LONG];
    static float32_t values[EXP_F32_LONG];
    uint32_t state = 0x1E5u;

    for (int32_t i = 0; i < EXP_F32_LONG; ++i)
    {
        input[i] = exp_f32_from_bits(exp_f32_next(&state));
    }
    for (int32_t length = 1; length <= EXP_F32_MAX_TAIL + 1; ++length)
    {
        const int32_t n = length <= EXP_F32_MAX_TAIL ? length : EXP_F32_LONG;
        memcpy(values, input, (size_t)n * sizeof(values[0]));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_exp_f32(input, separate, n));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_exp_f32(values, values, n));
        TEST_ASSERT_EQUAL_HEX32_ARRAY((const uint32_t *)separate, (const uint32_t *)values, n);
    }
}

// Eight sentinel words on both sides of the input and the output are unchanged, and so
// is the input, for every length 1 to 17: each tail of the four-lane predicate.
void exp_f32_guards_arm_nn_exp_f32(void)
{
    uint32_t input[EXP_F32_GUARD + EXP_F32_MAX_TAIL + EXP_F32_GUARD];
    uint32_t output[EXP_F32_GUARD + EXP_F32_MAX_TAIL + EXP_F32_GUARD];
    uint32_t values[EXP_F32_MAX_TAIL];
    uint32_t state = 0xC0FFEEu;

    for (int32_t length = 1; length <= EXP_F32_MAX_TAIL; ++length)
    {
        for (int32_t i = 0; i < EXP_F32_GUARD + EXP_F32_MAX_TAIL + EXP_F32_GUARD; ++i)
        {
            input[i] = EXP_F32_SENTINEL;
            output[i] = EXP_F32_SENTINEL;
        }
        for (int32_t i = 0; i < length; ++i)
        {
            values[i] = exp_f32_next(&state);
            input[EXP_F32_GUARD + i] = values[i];
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_exp_f32((const float32_t *)&input[EXP_F32_GUARD],
                                         (float32_t *)&output[EXP_F32_GUARD],
                                         length));
        for (int32_t i = 0; i < EXP_F32_GUARD; ++i)
        {
            TEST_ASSERT_EQUAL_HEX32(EXP_F32_SENTINEL, input[i]);
            TEST_ASSERT_EQUAL_HEX32(EXP_F32_SENTINEL, output[i]);
            TEST_ASSERT_EQUAL_HEX32(EXP_F32_SENTINEL, input[EXP_F32_GUARD + length + i]);
            TEST_ASSERT_EQUAL_HEX32(EXP_F32_SENTINEL, output[EXP_F32_GUARD + length + i]);
        }
        TEST_ASSERT_EQUAL_HEX32_ARRAY(values, &input[EXP_F32_GUARD], length);
    }
}

/*
 * The FPSCR contract of the header, on Arm targets: with FZ and DN set, as the
 * Apollo510 Lite and Apollo330M Plus images start, a result below 2^-126 is +0 on
 * every leg, a NaN keeps its payload, a subnormal input gives 1.0f and a normal result
 * keeps its bits. The control bits are restored and checked unchanged by the call; the
 * cumulative exception flags are not asserted (an exponential raises inexact and
 * underflow by nature). Compiles away on the host.
 */
void exp_f32_fp_controls_arm_nn_exp_f32(void)
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
    // Subnormal results: -103.97f, -100, -95, -90, -88 and 0xC2AEAC50, the input nearest
    // zero whose result is subnormal; then 0xC2AEAC4F, the first normal result, kept.
    // Then NaNs, subnormal inputs and normal results.
    const uint32_t input_bits[] = {0xC2CFF0A4, 0xC2C80000, 0xC2BE0000, 0xC2B40000, 0xC2B00000, 0xC2AEAC50,
                                   0xC2AEAC4F, 0x7FA00001, 0xFFC12345, 0x7F800001, 0x00000001, 0x807FFFFF,
                                   0x3F800000, 0xC2A00000, 0x42B17217, 0xBF800000};
    const int32_t count = (int32_t)(sizeof(input_bits) / sizeof(input_bits[0]));
    /* FPSCR: FZ[24], DN[25]. */
    const uint32_t controls = (UINT32_C(1) << 24) | (UINT32_C(1) << 25);
    const uint32_t rounding = UINT32_C(3) << 22;
    float32_t input[sizeof(input_bits) / sizeof(input_bits[0])];
    float32_t plain[sizeof(input_bits) / sizeof(input_bits[0])];
    float32_t flushed[sizeof(input_bits) / sizeof(input_bits[0])];
    uint32_t original;
    uint32_t after;

    memcpy(input, input_bits, sizeof(input));
    __asm volatile("vmrs %0, fpscr" : "=r"(original));
    const uint32_t base = original & ~(controls | rounding);
    __asm volatile("vmsr fpscr, %0" ::"r"(base) : "memory");
    const arm_cmsis_nn_status plain_status = arm_nn_exp_f32(input, plain, count);
    __asm volatile("vmsr fpscr, %0" ::"r"(base | controls) : "memory");
    const arm_cmsis_nn_status flushed_status = arm_nn_exp_f32(input, flushed, count);
    __asm volatile("vmrs %0, fpscr" : "=r"(after));
    __asm volatile("vmsr fpscr, %0" ::"r"(original) : "memory");

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, plain_status);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, flushed_status);
    TEST_ASSERT_EQUAL_HEX32(controls, after & (controls | rounding));
    for (int32_t i = 0; i < 6; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(0, exp_f32_bits(flushed[i]));
    }
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x00800026), exp_f32_bits(flushed[6]));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x7FE00001), exp_f32_bits(flushed[7]));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0xFFC12345), exp_f32_bits(flushed[8]));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x7FC00001), exp_f32_bits(flushed[9]));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x3F800000), exp_f32_bits(flushed[10]));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x3F800000), exp_f32_bits(flushed[11]));
    for (int32_t i = 6; i < count; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(exp_f32_bits(plain[i]), exp_f32_bits(flushed[i]));
    }
#endif
}

// The scalar body through a volatile read, so a vectorizing compiler cannot fold the
// reference into Helium lanes that flush.
static uint32_t exp_f32_scalar_reference(uint32_t in_bits)
{
    volatile float32_t x = exp_f32_from_bits(in_bits);
    return exp_f32_bits(arm_nn_exp_scalar_f32(x));
}

static uint32_t exp_f32_tier_block(const uint32_t *in_bits, int32_t count)
{
    static float32_t input[EXP_F32_TIER_BLOCK];
    static float32_t output[EXP_F32_TIER_BLOCK];
    uint32_t flushed = 0;
    char msg[64];

    memcpy(input, in_bits, (size_t)count * sizeof(input[0]));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_exp_f32(input, output, count));
    for (int32_t i = 0; i < count; ++i)
    {
        const uint32_t expected = exp_f32_scalar_reference(in_bits[i]);
        const uint32_t actual = exp_f32_bits(output[i]);
        if (expected == actual)
        {
            continue;
        }
    #if defined(ARM_NN_EXP_MVE_F32)
        if (expected != 0 && expected < UINT32_C(0x00800000) && actual == 0)
        {
            ++flushed;
            continue;
        }
    #endif
        snprintf(msg, sizeof(msg), "input 0x%08lx", (unsigned long)in_bits[i]);
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(expected, actual, msg);
    }
    return flushed;
}

static uint32_t exp_f32_tier_range(uint32_t first, uint32_t last)
{
    static uint32_t bits[EXP_F32_TIER_BLOCK];
    uint32_t flushed = 0;
    int32_t n = 0;

    for (uint32_t b = first;; ++b)
    {
        bits[n++] = b;
        if (n == EXP_F32_TIER_BLOCK || b == last)
        {
            flushed += exp_f32_tier_block(bits, n);
            n = 0;
        }
        if (b == last)
        {
            return flushed;
        }
    }
}

/*
 * arm_nn_exp_f32 against the internal scalar body, arm_nn_exp_scalar_f32: equal bits on
 * every input, except that the Helium body returns +0 where the scalar result is
 * subnormal. The exception is counted and must be exactly the subnormal-result inputs
 * on a Helium build and nothing elsewhere.
 */
void exp_f32_tiers_arm_nn_exp_f32(void)
{
    static uint32_t bits[EXP_F32_TIER_BLOCK];
    uint32_t flushed = 0;
    int32_t n = 0;

    for (uint64_t u = 0; u <= UINT64_C(0xFFFFFFFF); u += EXP_F32_TIER_STRIDE)
    {
        const uint32_t b = (uint32_t)u;
        if ((b >= EXP_F32_DENSE_LOW_FIRST && b <= EXP_F32_DENSE_LOW_LAST) ||
            (b >= EXP_F32_DENSE_HIGH_FIRST && b <= EXP_F32_DENSE_HIGH_LAST))
        {
            continue;
        }
        bits[n++] = b;
        if (n == EXP_F32_TIER_BLOCK)
        {
            flushed += exp_f32_tier_block(bits, n);
            n = 0;
        }
    }
    if (n > 0)
    {
        flushed += exp_f32_tier_block(bits, n);
    }
    flushed += exp_f32_tier_range(EXP_F32_DENSE_LOW_FIRST, EXP_F32_DENSE_LOW_LAST);
    flushed += exp_f32_tier_range(EXP_F32_DENSE_HIGH_FIRST, EXP_F32_DENSE_HIGH_LAST);
    #if defined(ARM_NN_EXP_MVE_F32)
    TEST_ASSERT_EQUAL_UINT32(EXP_F32_SUBNORMAL_RESULTS, flushed);
    #else
    TEST_ASSERT_EQUAL_UINT32(0, flushed);
    #endif
}
