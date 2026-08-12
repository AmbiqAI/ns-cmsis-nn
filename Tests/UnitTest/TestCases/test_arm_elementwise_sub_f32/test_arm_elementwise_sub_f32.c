/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "sub_f32_data.h"

void sub_f32_arm_elementwise_sub_f32(void)
{
    float32_t output[SUB_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_elementwise_sub_f32(sub_f32_input1,
                                              sub_f32_input2,
                                              output,
                                              SUB_F32_OUT_ACTIVATION_MIN,
                                              SUB_F32_OUT_ACTIVATION_MAX,
                                              SUB_F32_DST_SIZE));

    for (int i = 0; i < SUB_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, sub_f32_output_ref[i], output[i]);
    }
}

void sub_f32_arg_error_arm_elementwise_sub_f32(void)
{
    float32_t output[SUB_F32_DST_SIZE] = {0};
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_elementwise_sub_f32(
            NULL, sub_f32_input2, output, SUB_F32_OUT_ACTIVATION_MIN, SUB_F32_OUT_ACTIVATION_MAX, SUB_F32_DST_SIZE));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_elementwise_sub_f32(
            sub_f32_input1, sub_f32_input2, output, SUB_F32_OUT_ACTIVATION_MIN, SUB_F32_OUT_ACTIVATION_MAX, 0));
}
