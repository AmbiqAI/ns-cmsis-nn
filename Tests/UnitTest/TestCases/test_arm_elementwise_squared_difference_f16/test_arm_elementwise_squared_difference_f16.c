/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#include "squared_difference_f16_data.h"

static bool squared_difference_f16_bits_are_nan(float16_t value)
{
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return ((bits & 0x7C00u) == 0x7C00u) && ((bits & 0x03FFu) != 0u);
}

static bool squared_difference_f16_bits_are_inf(float16_t value)
{
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7FFFu) == 0x7C00u;
}

static float16_t squared_difference_f16_from_bits(volatile const uint16_t *bits)
{
    const uint16_t value_bits = *bits;
    float16_t value;
    memcpy(&value, &value_bits, sizeof(value));
    return value;
}

void squared_difference_f16(void)
{
    const float16_t input_1[5] = {(float16_t)3.0f, (float16_t)-2.0f, (float16_t)1.0f, (float16_t)0.0f, (float16_t)8.0f};
    const float16_t input_2[5] = {(float16_t)1.0f, (float16_t)4.0f, (float16_t)-2.0f, (float16_t)0.0f, (float16_t)5.0f};
    const float16_t expected[5] = {
        (float16_t)4.0f, (float16_t)36.0f, (float16_t)9.0f, (float16_t)0.0f, (float16_t)9.0f};
    float16_t output[5] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_squared_difference_f16(input_1, input_2, output, 5));

    for (int i = 0; i < 5; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(6.0e-3f, (float)expected[i], (float)output[i]);
    }
}

void squared_difference_f16_complex(void)
{
    float16_t output[SQUARED_DIFFERENCE_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_elementwise_squared_difference_f16(
            squared_difference_f16_input1, squared_difference_f16_input2, output, SQUARED_DIFFERENCE_F16_DST_SIZE));

    for (int i = 0; i < SQUARED_DIFFERENCE_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(6.0e-3f, (float)squared_difference_f16_output_ref[i], (float)output[i]);
    }
}

void squared_difference_f16_nan_inf(void)
{
    volatile uint16_t inf_bits = 0x7C00u;
    volatile uint16_t nan_bits = 0x7E00u;
    volatile uint16_t ninf_bits = 0xFC00u;
    const float16_t inf = squared_difference_f16_from_bits(&inf_bits);
    const float16_t nan = squared_difference_f16_from_bits(&nan_bits);
    const float16_t ninf = squared_difference_f16_from_bits(&ninf_bits);
    const float16_t input_1[5] = {inf, nan, inf, ninf, (float16_t)1.0f};
    const float16_t input_2[5] = {inf, (float16_t)0.0f, (float16_t)1.0f, (float16_t)1.0f, nan};
    float16_t output[5] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_squared_difference_f16(input_1, input_2, output, 5));

    TEST_ASSERT_TRUE_MESSAGE(squared_difference_f16_bits_are_nan(output[0]), "Expected NaN from Inf - Inf");
    TEST_ASSERT_TRUE_MESSAGE(squared_difference_f16_bits_are_nan(output[1]), "Expected NaN from NaN - 0");
    TEST_ASSERT_TRUE_MESSAGE(squared_difference_f16_bits_are_inf(output[2]), "Expected Inf from Inf - 1");
    TEST_ASSERT_TRUE_MESSAGE(squared_difference_f16_bits_are_inf(output[3]), "Expected Inf from -Inf - 1");
    TEST_ASSERT_TRUE_MESSAGE(squared_difference_f16_bits_are_nan(output[4]), "Expected NaN from 1 - NaN");
}

void squared_difference_f16_arg_error(void)
{
    const float16_t input[1] = {(float16_t)1.0f};
    float16_t output[1] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f16(NULL, input, output, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f16(input, NULL, output, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f16(input, input, NULL, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f16(input, input, output, 0));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_elementwise_squared_difference_f16(input, input, output, -1));
}

typedef float16_t broadcast_element;
#define broadcast_kernel arm_elementwise_squared_difference_broadcast_f16
#include "../Common/squared_difference_broadcast_tests.h"

static void broadcast_f16_compare_flat(void)
{
    const cmsis_nn_dims rows = {1, 1, 2, 1}, full = {1, 1, 2, 9};
    const uint16_t inf = 0x7c00, nan = 0x7e01, tiny = 0x1000;
    const float16_t scalars[] = {0, squared_difference_f16_from_bits(&inf)};
    float16_t vec[18] = {0}, expanded[18], expected[18], out[18];
    vec[8] = squared_difference_f16_from_bits(&tiny); /* (2^-11)^2 is subnormal. */
    vec[9] = scalars[1];
    vec[10] = squared_difference_f16_from_bits(&nan);
    for (int i = 0; i < 18; ++i)
        expanded[i] = scalars[i / 9];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_elementwise_squared_difference_f16(vec, expanded, expected, 18));
    for (int swap = 0; swap <= 1; ++swap)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, broadcast_kernel(swap ? scalars : vec, swap ? &rows : &full,
            swap ? vec : scalars, swap ? &full : &rows, out, &full));
        for (int i = 0; i < 18; ++i)
        {
            if (squared_difference_f16_bits_are_nan(expected[i]))
                TEST_ASSERT_TRUE(squared_difference_f16_bits_are_nan(out[i]));
            else
                TEST_ASSERT_EQUAL_MEMORY(expected + i, out + i, sizeof(*out));
        }
    }
}
