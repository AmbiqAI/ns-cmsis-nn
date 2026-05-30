/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "arm_nn_types.h"
#include "unity.h"

#include "../TestData/mirror_pad_reflect_s8/test_data.h"

void mirror_pad_reflect_arm_mirror_pad_s8(void)
{
    int8_t output[MIRROR_PAD_REFLECT_S8_OUTPUT_SIZE] = {0};

    const cmsis_nn_mirror_pad_params params = {
        .rank = MIRROR_PAD_REFLECT_S8_RANK,
        .input_shape = mirror_pad_reflect_s8_input_shape,
        .output_shape = mirror_pad_reflect_s8_output_shape,
        .pad_before = mirror_pad_reflect_s8_pad_before,
        .mode = MIRROR_PAD_REFLECT_S8_MODE,
    };

    const arm_cmsis_nn_status result = arm_mirror_pad_s8(mirror_pad_reflect_s8_input, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT8_ARRAY(mirror_pad_reflect_s8_output, output, MIRROR_PAD_REFLECT_S8_OUTPUT_SIZE);
}
