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
