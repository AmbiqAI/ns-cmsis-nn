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

#include "abs_f16_data.h"

void abs_f16_arm_nn_abs_f16(void)
{
    float16_t output[ABS_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_abs_f16(abs_f16_input, output, ABS_F16_DST_SIZE));

    for (int i = 0; i < ABS_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)abs_f16_output_ref[i], (float)output[i]);
    }
}

void abs_f16_arg_error_arm_nn_abs_f16(void)
{
    float16_t output[ABS_F16_DST_SIZE] = {0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_abs_f16(NULL, output, ABS_F16_DST_SIZE));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_abs_f16(abs_f16_input, output, 0));
}
