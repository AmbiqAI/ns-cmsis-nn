/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>
#include <math.h>

#include "../../../../Include/Internal/arm_nn_activation_flt.h"

#include "../TestData/softmax_f16/test_data.h"

void softmax_f16_arm_softmax_f16(void)
{
    float16_t output[SOFTMAX_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_softmax_f16(softmax_f16_input, SOFTMAX_F16_NUM_ROWS, SOFTMAX_F16_ROW_SIZE, output));

    for (int i = 0; i < SOFTMAX_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-3f, (float)softmax_f16_output_ref[i], (float)output[i]);
    }
}

/* Exercise the normal/subnormal transition and final rounding to zero without a
 * tolerance large enough to conceal the former smallest-normal exponent floor. */
void softmax_f16_scalar_exp_lower_range(void)
{
    const float32_t values[] = {-9.5f, -9.703125f, -10.0f, -14.0f, -16.0f, -16.75f,
                               -17.0f, -17.328125f, -17.34375f, -20.0f, -80.0f};
    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i)
    {
        const float16_t x = (float16_t)values[i];
        const double expected = exp((double)x);
        const float32_t tolerance = (float32_t)(expected * 0.001 + 0x1p-25);
        TEST_ASSERT_FLOAT_WITHIN(tolerance, (float32_t)expected, (float32_t)arm_nn_softmax_exp_lut_f16(x));
        TEST_ASSERT_FLOAT_WITHIN(tolerance, (float32_t)expected, (float32_t)arm_nn_softmax_exp_taylor_f16(x));
        TEST_ASSERT_FLOAT_WITHIN(tolerance,
                                 (float32_t)(expected / (1.0 + expected)),
                                 (float32_t)arm_nn_sigmoid_scalar_f16(x));
    }
}

void softmax_f16_large_gap(void)
{
    const float32_t gaps[] = {10.0f, 16.75f, 20.0f};
    for (unsigned i = 0; i < sizeof(gaps) / sizeof(gaps[0]); ++i)
    {
        for (int peak = 0; peak < 2; ++peak)
        {
            float16_t input[2] = {(float16_t)-gaps[i], (float16_t)-gaps[i]};
            float16_t output[2];
            input[peak] = (float16_t)0.0f;
            const double e = exp((double)input[1 - peak]);
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_softmax_f16(input, 1, 2, output));
            TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, (float32_t)(1.0 / (1.0 + e)), (float32_t)output[peak]);
            TEST_ASSERT_FLOAT_WITHIN((float32_t)(e * 0.001 + 0x1p-25),
                                     (float32_t)(e / (1.0 + e)),
                                     (float32_t)output[1 - peak]);
        }
    }
}

void softmax_f16_wide_large_gap(void)
{
    /* This exercises the generic row path even when 1x2 specialization is enabled. */
    static float16_t input[1000];
    static float16_t output[1000];
    for (int i = 0; i < 1000; ++i)
    {
        input[i] = (float16_t)-20.0f;
    }
    input[0] = (float16_t)0.0f;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_softmax_f16(input, 1, 1000, output));
    TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, 1.0f, (float32_t)output[0]);
    for (int i = 1; i < 1000; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, (float32_t)output[i]);
    }
}

void softmax_f16_lut_lane_boundaries(void)
{
    const int32_t widths[] = {3, 4, 5, 7, 8, 9, 33};
    float16_t input[66];
    float16_t output[68];
    for (unsigned w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w)
    {
        const int32_t cols = widths[w];
        /* Different LUT fractions across vector groups and scalar tails. The
         * second row also checks the small nonzero probabilities at exp(-10). */
        for (int32_t c = 0; c < cols; ++c)
        {
            input[c] = (float16_t)(-0.125f * (float32_t)((c * 7) % 23));
            input[cols + c] = (float16_t)-10.0f;
        }
        input[cols] = (float16_t)0.0f;
        output[0] = (float16_t)123.0f;
        output[2 * cols + 1] = (float16_t)123.0f;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_softmax_f16(input, 2, cols, output + 1));
        TEST_ASSERT_EQUAL_FLOAT(123.0f, (float32_t)output[0]);
        TEST_ASSERT_EQUAL_FLOAT(123.0f, (float32_t)output[2 * cols + 1]);
        for (int32_t r = 0; r < 2; ++r)
        {
            double sum = 0.0;
            for (int32_t c = 0; c < cols; ++c)
            {
                sum += exp((double)input[r * cols + c]);
            }
            for (int32_t c = 0; c < cols; ++c)
            {
                const double expected = exp((double)input[r * cols + c]) / sum;
                /* Allow half-precision exponential and normalization rounding. */
                TEST_ASSERT_FLOAT_WITHIN((float32_t)(expected * 0.002 + 0x1p-24),
                                         (float32_t)expected,
                                         (float32_t)output[1 + r * cols + c]);
            }
        }
    }
}
