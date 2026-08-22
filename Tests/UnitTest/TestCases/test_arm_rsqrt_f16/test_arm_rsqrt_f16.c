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
#include <string.h>
#include <unity.h>

#include "rsqrt_f16_data.h"

void rsqrt_f16_arm_rsqrt_f16(void)
{
    float16_t output[RSQRT_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(rsqrt_f16_input, output, RSQRT_F16_DST_SIZE));
    for (int32_t i = 0; i < RSQRT_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)rsqrt_f16_output_ref[i], (float)output[i]);
    }
}

void rsqrt_f16_in_place_arm_rsqrt_f16(void)
{
    float16_t values[RSQRT_F16_DST_SIZE];
    memcpy(values, rsqrt_f16_input, sizeof(values));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(values, values, RSQRT_F16_DST_SIZE));
    for (int32_t i = 0; i < RSQRT_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)rsqrt_f16_output_ref[i], (float)values[i]);
    }
}

void rsqrt_f16_special_values_arm_rsqrt_f16(void)
{
    const float16_t input[] = {
        (float16_t)0.0f, (float16_t)-0.0f, (float16_t)-1.0f, (float16_t)INFINITY, (float16_t)NAN};
    float16_t output[5] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_rsqrt_f16(input, output, 5));
    TEST_ASSERT_FLOAT_IS_INF((float)output[0]);
    TEST_ASSERT_FALSE(signbit((float)output[0]));
    TEST_ASSERT_FLOAT_IS_NEG_INF((float)output[1]);
    TEST_ASSERT_FLOAT_IS_NAN((float)output[2]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, (float)output[3]);
    TEST_ASSERT_FLOAT_IS_NAN((float)output[4]);
}

void rsqrt_f16_arg_error_arm_rsqrt_f16(void)
{
    float16_t output[RSQRT_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(NULL, output, RSQRT_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(rsqrt_f16_input, NULL, RSQRT_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_rsqrt_f16(rsqrt_f16_input, output, 0));
}
