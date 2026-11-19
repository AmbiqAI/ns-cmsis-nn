/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 * SPDX-License-Identifier: Apache-2.0
 */
/* Each executable selects its mode independently of the library configuration. */
#if defined(TEST_REQUANTIZE_SINGLE_ROUNDING)
#ifndef CMSIS_NN_USE_SINGLE_ROUNDING
#define CMSIS_NN_USE_SINGLE_ROUNDING
#endif
#else
#undef CMSIS_NN_USE_SINGLE_ROUNDING
#endif
#include "arm_nnsupportfunctions.h"
#include "unity.h"

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

void setUp(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
}
void tearDown(void) {}

static void shifted_requantize_matches_unfused(void)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const int32_t values[][4] = {
        {0, 1, -1, 3}, {INT16_MIN, INT16_MAX, -3, -5}, {INT32_MIN, INT32_MAX, -1073741824, 1073741823}};
    const int32_t multipliers[] = {0, 1, 1073741824, INT32_MAX};
    const int32_t shifts[] = {-31, -15, -1, 0, 1, 7, 15};
    const int32_t pre_shifts[] = {0, 1, 4, 12, 20};
    for (unsigned v = 0; v < sizeof(values) / sizeof(values[0]); ++v)
    {
        for (unsigned m = 0; m < sizeof(multipliers) / sizeof(multipliers[0]); ++m)
        {
            for (unsigned s = 0; s < sizeof(shifts) / sizeof(shifts[0]); ++s)
            {
                for (unsigned p = 0; p < sizeof(pre_shifts) / sizeof(pre_shifts[0]); ++p)
                {
                    const int shift = shifts[s];
    #ifdef CMSIS_NN_USE_SINGLE_ROUNDING
                    const int right = ARM_NN_MIN(-1, shift);
                    const int left = shift - right;
    #else
                    const int left = LEFT_SHIFT(shift);
    #endif
                    if (pre_shifts[p] + left > 31)
                    {
                        continue;
                    }
                    const int32x4_t x = vld1q_s32(values[v]);
                    const int32x4_t shifted = vshlq_s32(x, vdupq_n_s32(pre_shifts[p]));
                    int32x4_t expected;
    #ifdef CMSIS_NN_USE_SINGLE_ROUNDING
                    expected = vqdmulhq_n_s32(vshlq_s32(shifted, vdupq_n_s32(left)), multipliers[m]);
                    expected = vrshlq_s32(expected, vdupq_n_s32(right));
    #else
                    expected = vqrdmulhq_n_s32(vshlq_s32(shifted, vdupq_n_s32(left)), multipliers[m]);
                    const int32x4_t exponent = vdupq_n_s32(-RIGHT_SHIFT(shift));
                    const int32x4_t fixup = vshrq_n_s32(vandq_s32(expected, exponent), 31);
                    expected = vrshlq_s32(vqaddq_s32(expected, fixup), exponent);
    #endif
                    int32_t reference[4], actual[4], uniform[4], per_lane[4];
                    vst1q_s32(reference, expected);
                    vst1q_s32(actual, arm_requantize_shifted_mve(x, pre_shifts[p], multipliers[m], shift));
                    TEST_ASSERT_EQUAL_INT32_ARRAY(reference, actual, 4);
                    vst1q_s32(uniform, arm_requantize_mve(shifted, multipliers[m], shift));
                    TEST_ASSERT_EQUAL_INT32_ARRAY(reference, uniform, 4);
                    vst1q_s32(per_lane,
                              arm_requantize_mve_32x4(shifted, vdupq_n_s32(multipliers[m]), vdupq_n_s32(shift)));
                    TEST_ASSERT_EQUAL_INT32_ARRAY(per_lane, uniform, 4);
                }
            }
        }
    }
#else
    TEST_IGNORE_MESSAGE("Requires explicit MVE integer intrinsics");
#endif
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(shifted_requantize_matches_unfused);
    return UNITY_END();
}
