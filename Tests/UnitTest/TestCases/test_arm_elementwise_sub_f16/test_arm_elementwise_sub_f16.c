/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "sub_f16_data.h"

void sub_f16_arm_elementwise_sub_f16(void)
{
    float16_t output[SUB_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_elementwise_sub_f16(sub_f16_input1,
                                              sub_f16_input2,
                                              output,
                                              SUB_F16_OUT_ACTIVATION_MIN,
                                              SUB_F16_OUT_ACTIVATION_MAX,
                                              SUB_F16_DST_SIZE));

    for (int i = 0; i < SUB_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(6.0e-3f, (float)sub_f16_output_ref[i], (float)output[i]);
    }
}

void sub_f16_arg_error_arm_elementwise_sub_f16(void)
{
    float16_t output[SUB_F16_DST_SIZE] = {0};
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_elementwise_sub_f16(
            NULL, sub_f16_input2, output, SUB_F16_OUT_ACTIVATION_MIN, SUB_F16_OUT_ACTIVATION_MAX, SUB_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_elementwise_sub_f16(
            sub_f16_input1, sub_f16_input2, output, SUB_F16_OUT_ACTIVATION_MIN, SUB_F16_OUT_ACTIVATION_MAX, 0));
}
