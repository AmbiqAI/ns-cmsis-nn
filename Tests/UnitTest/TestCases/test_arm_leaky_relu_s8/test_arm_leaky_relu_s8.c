/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "unity.h"
#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>

#include "../TestData/leaky_relu/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (2)

void leaky_relu_arm_leaky_relu_s8(void)
{

    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *input_data = leaky_relu_input;
    int8_t output[LEAKY_RELU_DST_SIZE];

    arm_cmsis_nn_status result = arm_leaky_relu_s8(
        input_data,
        LEAKY_RELU_INPUT_OFFSET,
        LEAKY_RELU_OUTPUT_OFFSET,
        LEAKY_RELU_OUTPUT_MULT_ALPHA,
        LEAKY_RELU_OUTPUT_SHIFT_ALPHA,
        LEAKY_RELU_OUTPUT_MULT_IDENTITY,
        LEAKY_RELU_OUTPUT_SHIFT_IDENTITY,
        output,
        LEAKY_RELU_DST_SIZE
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, leaky_relu_output_ref, LEAKY_RELU_DST_SIZE));
}
