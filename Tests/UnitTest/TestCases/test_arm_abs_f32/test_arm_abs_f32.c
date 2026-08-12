/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "abs_f32_data.h"

void abs_f32_arm_abs_f32(void)
{
    float32_t output[ABS_F32_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_abs_f32(abs_f32_input, output, ABS_F32_DST_SIZE));

    for (int i = 0; i < ABS_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, abs_f32_output_ref[i], output[i]);
    }
}

void abs_f32_arg_error_arm_abs_f32(void)
{
    float32_t output[ABS_F32_DST_SIZE] = {0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_abs_f32(NULL, output, ABS_F32_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_abs_f32(abs_f32_input, output, 0));
}
