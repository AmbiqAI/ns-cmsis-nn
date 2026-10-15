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

#include "../Common/Softmax/exp_lut_data.h"
#include "../Common/Softmax/one_by_one_lut_data.h"
#include "../TestData/softmax_s16/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (2)

void softmax_s16_arm_softmax_s16(void)
{
    const int32_t num_rows = SOFTMAX_S16_NUM_ROWS;
    const int32_t row_size = SOFTMAX_S16_ROW_SIZE;
    const int32_t mult = SOFTMAX_S16_INPUT_MULT;
    const int32_t shift = SOFTMAX_S16_INPUT_LEFT_SHIFT;
    const int16_t *input_data = softmax_s16_input;
    const cmsis_nn_softmax_lut_s16 softmax_params = {.exp_lut = softmax_s16_exp_lut,
                                                     .one_by_one_lut = softmax_s16_one_by_one_lut};
    int16_t output[SOFTMAX_S16_DST_SIZE];

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_softmax_s16(input_data, num_rows, row_size, mult, shift, &softmax_params, output);
        TEST_ASSERT_TRUE(validate_s16(output, softmax_s16_output_ref, SOFTMAX_S16_DST_SIZE));
    }
}

/* Exact integer/LUT interpolation reference: at shift 31 the Q31 scale is the multiplier itself. */
void softmax_s16_shift31_arm_softmax_s16(void)
{
    const cmsis_nn_softmax_lut_s16 params = {softmax_s16_exp_lut, softmax_s16_one_by_one_lut};
    const struct
    {
        int16_t input[3];
        int32_t multiplier;
        int16_t expected[3];
    } cases[] = {
        {{0, -1000, -2000}, 1, {12625, 10839, 9305}},
        {{0, 0, -2}, 1077952576, {16384, 16384, 1}},
        {{INT16_MAX, INT16_MAX, INT16_MIN}, INT32_MAX, {16384, 16384, 1}},
        {{0, -1000, -2000}, 0, {10922, 10922, 10922}},
        {{INT16_MAX, INT16_MAX, INT16_MIN}, INT32_MIN, {10922, 10922, 10922}},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        int16_t output[3] = {0};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_softmax_s16(cases[i].input, 1, 3, cases[i].multiplier, 31, &params, output));
        TEST_ASSERT_EQUAL_INT16_ARRAY(cases[i].expected, output, 3);
    }
}

void softmax_s16_row_domains_arm_softmax_s16(void)
{
    const cmsis_nn_softmax_lut_s16 params = {softmax_s16_exp_lut, softmax_s16_one_by_one_lut};
    const int32_t rejected[][2] = {{-1, 3}, {1, -1}, {2, -1}, {0, -1}, {-1, 0}, {-1, -1},
                                   {INT32_MIN, INT32_MAX}, {INT32_MAX, INT32_MIN}};
    for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++)
    {
        int16_t output[3] = {0x5555, 0x5555, 0x5555};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_softmax_s16(NULL, rejected[i][0], rejected[i][1], 1, 0, &params, output));
        for (size_t j = 0; j < sizeof(output) / sizeof(output[0]); j++)
        {
            TEST_ASSERT_EQUAL_INT16(0x5555, output[j]);
        }
    }

    const int32_t empty[][2] = {{0, 3}, {3, 0}, {0, 0}, {INT32_MAX, 0}, {0, 65535}};
    for (size_t i = 0; i < sizeof(empty) / sizeof(empty[0]); i++)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_softmax_s16(NULL, empty[i][0], empty[i][1], 1, 0, &params, NULL));
    }

    const cmsis_nn_softmax_lut_s16 missing_exp = {NULL, softmax_s16_one_by_one_lut};
    const cmsis_nn_softmax_lut_s16 missing_reciprocal = {softmax_s16_exp_lut, NULL};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_softmax_s16(NULL, 0, 0, 1, 0, &missing_exp, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_softmax_s16(NULL, -1, 3, 1, 0, &missing_reciprocal, NULL));
}

void softmax_s16_saturation_arm_softmax_s16(void)
{
    const cmsis_nn_softmax_lut_s16 params = {softmax_s16_exp_lut, softmax_s16_one_by_one_lut};
    const int16_t input[] = {INT16_MIN, -12345, -1, 0, 1, 777, INT16_MAX};
    int16_t output[7];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_softmax_s16(input, 7, 1, SOFTMAX_S16_INPUT_MULT,
                                      SOFTMAX_S16_INPUT_LEFT_SHIFT, &params, output));
    for (int i = 0; i < 7; i++)
    {
        TEST_ASSERT_EQUAL_INT16(INT16_MAX, output[i]);
    }
}

void softmax_s16_normalization_boundary_arm_softmax_s16(void)
{
    const cmsis_nn_softmax_lut_s16 params = {softmax_s16_exp_lut, softmax_s16_one_by_one_lut};
    /* Standard LUT interpolation gives eight exponents of 32762 and one of 47.
     * Their sum, 262143, normalizes and rounds to 131072, beyond the LUT domain.
     * Rounded 64-bit probabilities are 4095 and 6, respectively. */
    const int16_t input[9] = {32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767, -10176};
    int16_t output[9];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_softmax_s16(input, 1, 9, 1, 31, &params, output));
    for (int i = 0; i < 9; i++)
    {
        const int64_t exponent = i == 8 ? 47 : 32762;
        const int16_t expected = (int16_t)((exponent * 32768 + 262143 / 2) / 262143);
        TEST_ASSERT_EQUAL_INT16(expected, output[i]);
    }
}

void softmax_s16_output_tails_arm_softmax_s16(void)
{
    const cmsis_nn_softmax_lut_s16 params = {softmax_s16_exp_lut, softmax_s16_one_by_one_lut};
    const int32_t lengths[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17};
    /* Frozen scalar results for the same standard LUTs and quantization. */
    const int16_t expected[] = {
        32767, 16134, 16633, 10592, 10919, 11257, 7822, 8063, 8312, 8569, 6161, 6352,
        6547, 6749, 6958, 4265, 4397, 4532, 4673, 4817, 4965, 5119, 3673, 3787,
        3904, 4024, 4149, 4277, 4409, 4545, 3214, 3313, 3415, 3521, 3630, 3742,
        3857, 3976, 4099, 1750, 1804, 1860, 1917, 1977, 2038, 2101, 2166, 2233,
        2302, 2373, 2446, 2522, 2599, 2680, 1614, 1664, 1716, 1768, 1823, 1879,
        1937, 1998, 2059, 2123, 2188, 2256, 2326, 2397, 2471, 2548, 1494, 1541,
        1588, 1637, 1688, 1740, 1794, 1849, 1906, 1965, 2026, 2089, 2153, 2220,
        2288, 2359, 2432,
    };
    int16_t input[17];
    for (int i = 0; i < 17; i++)
    {
        input[i] = (int16_t)((i * 997) % 20001 - 10000);
    }
    int offset = 0;
    for (size_t k = 0; k < sizeof(lengths) / sizeof(lengths[0]); k++)
    {
        int16_t output[19];
        for (int i = 0; i < 19; i++)
        {
            output[i] = 0x5555;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_softmax_s16(input, 1, lengths[k], 1718013132, -2, &params, output + 1));
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected + offset, output + 1, lengths[k]);
        TEST_ASSERT_EQUAL_INT16(0x5555, output[0]);
        for (int i = lengths[k] + 1; i < 19; i++)
        {
            TEST_ASSERT_EQUAL_INT16(0x5555, output[i]);
        }
        offset += lengths[k];
    }
}
