/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "unity.h"
#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>

#include "../TestData/softmax/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (2)

void softmax_arm_softmax_s8(void)
{
    const int32_t num_rows = SOFTMAX_NUM_ROWS;
    const int32_t row_size = SOFTMAX_ROW_SIZE;
    const int32_t mult = SOFTMAX_INPUT_MULT;
    const int32_t shift = SOFTMAX_INPUT_LEFT_SHIFT;
    const int32_t diff_min = SOFTMAX_DIFF_MIN;
    const int8_t *input_data = softmax_input;
    int8_t output[SOFTMAX_DST_SIZE];

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);
        TEST_ASSERT_TRUE(validate(output, softmax_output_ref, SOFTMAX_DST_SIZE));
    }
}

void softmax_invalid_diff_min_arm_softmax_s8(void)
{
    const int32_t num_rows = SOFTMAX_NUM_ROWS;
    const int32_t row_size = SOFTMAX_ROW_SIZE;
    const int32_t mult = SOFTMAX_INPUT_MULT;
    const int32_t shift = SOFTMAX_INPUT_LEFT_SHIFT;
    const int32_t diff_min = 0x7FFFFFFF;
    const int8_t *input_data = softmax_input;
    int8_t output[SOFTMAX_DST_SIZE];

    int8_t *softmax_expect_invalid_output = malloc(SOFTMAX_DST_SIZE);
    for (int i = 0; i < SOFTMAX_DST_SIZE; i++)
    {
        softmax_expect_invalid_output[i] = -128;
    }

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);
        TEST_ASSERT_TRUE(validate(output, softmax_expect_invalid_output, SOFTMAX_DST_SIZE));
    }
    free(softmax_expect_invalid_output);
}

/* Rows of n equal values: every output is 1/n, rounded (the sizes avoid the exact tie at 512). From 256 elements
   the normalisation reaches a divide by 2^31, and from 512 a quotient that rounds to 0 (#710). From 4096 the row
   sum passes int32_t (#705); 8193 is a size where a wrapped int32_t sum stays positive and gives a wrong output. */
void softmax_long_equal_rows_arm_softmax_s8(void)
{
    static int8_t input[8193];
    static int8_t output[8193];
    const int32_t sizes[] = {256, 601, 8193};
    const int8_t expected[] = {-127, -128, -128};

    memset(input, 5, sizeof(input));
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++)
    {
        arm_softmax_s8(input, 1, sizes[k], SOFTMAX_INPUT_MULT, SOFTMAX_INPUT_LEFT_SHIFT, SOFTMAX_DIFF_MIN, output);
        for (int32_t i = 0; i < sizes[k]; i++)
        {
            TEST_ASSERT_EQUAL_INT8(expected[k], output[i]);
        }
    }
}

/* arm_nn_divide_by_power_of_two(), which the softmax kernels round with, for exponents 0 to 31 on edge and random
   dividends against a 64-bit reference that rounds midpoints away from zero. Builds with
   CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY take its assembly branch. */
void divide_by_power_of_two_arm_softmax_s8(void)
{
    const int32_t edges[] = {
        INT32_MIN, INT32_MIN + 1, -1073741825, -65537, -3, -2, -1, 0, 1, 2, 3, 65535, 1073741824, INT32_MAX - 1, INT32_MAX};
    const int32_t num_edges = (int32_t)(sizeof(edges) / sizeof(edges[0]));
    uint32_t seed = 713u;
    for (int32_t exponent = 0; exponent <= 31; exponent++)
    {
        for (int32_t i = 0; i < num_edges + 64; i++)
        {
            int32_t dividend;
            if (i < num_edges)
            {
                dividend = edges[i];
            }
            else
            {
                seed = seed * 1664525u + 1013904223u;
                dividend = (int32_t)seed;
            }
            const int64_t divisor = (int64_t)1 << exponent;
            const int64_t magnitude = ((dividend < 0 ? -(int64_t)dividend : dividend) + divisor / 2) / divisor;
            const int32_t expected = (int32_t)(dividend < 0 ? -magnitude : magnitude);
            TEST_ASSERT_EQUAL_INT32(expected, arm_nn_divide_by_power_of_two(dividend, exponent));
        }
    }
}

void softmax_shift31_arm_softmax_s8(void)
{
    int8_t out[9];
    int16_t out16[9];
    uint8_t out_u8[9];
    /* Doubling differences at shift 30 is equivalent to shift 31. Nine lanes
     * cover both the MVE vector body and its scalar tail, including a -1 input. */
    const int8_t edge[9] = {0, -1, 0, -1, 0, -1, 0, -1, -1};
    const int8_t doubled[9] = {0, -2, 0, -2, 0, -2, 0, -2, -2};
    const uint8_t edge_u8[9] = {2, 1, 2, 1, 2, 1, 2, 1, 1};
    const uint8_t doubled_u8[9] = {2, 0, 2, 0, 2, 0, 2, 0, 0};
    int8_t ref[9];
    int16_t ref16[9];
    uint8_t ref_u8[9];
    arm_softmax_s8(doubled, 1, 9, 1077952576, 30, -2, ref);
    arm_softmax_s8_s16(doubled, 1, 9, 1077952576, 30, -2, ref16);
    arm_softmax_u8(doubled_u8, 1, 9, 1077952576, 30, -2, ref_u8);
    arm_softmax_s8(edge, 1, 9, 1077952576, 31, -1, out);
    arm_softmax_s8_s16(edge, 1, 9, 1077952576, 31, -1, out16);
    arm_softmax_u8(edge_u8, 1, 9, 1077952576, 31, -1, out_u8);
    TEST_ASSERT_EQUAL_INT8_ARRAY(ref, out, 9);
    TEST_ASSERT_EQUAL_INT16_ARRAY(ref16, out16, 9);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(ref_u8, out_u8, 9);
}
