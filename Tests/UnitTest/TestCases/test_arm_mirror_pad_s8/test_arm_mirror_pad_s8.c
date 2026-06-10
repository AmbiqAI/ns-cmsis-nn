/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/mirror_pad_reflect_2d_s8/test_data.h"
#include "../TestData/mirror_pad_reflect_s8/test_data.h"
#include "../TestData/mirror_pad_reflect_wide_s8/test_data.h"
#include "../Utils/validate.h"

void mirror_pad_reflect_s8_arm_mirror_pad_s8(void)
{
    const int32_t input_shape[] = MIRROR_PAD_REFLECT_S8_INPUT_SHAPE;
    const int32_t output_shape[] = MIRROR_PAD_REFLECT_S8_OUTPUT_SHAPE;
    const int32_t pad_before[] = MIRROR_PAD_REFLECT_S8_PAD_BEFORE;
    const cmsis_nn_mirror_pad_params params = {
        .rank = MIRROR_PAD_REFLECT_S8_RANK,
        .input_shape = input_shape,
        .output_shape = output_shape,
        .pad_before = pad_before,
        .mode = MIRROR_PAD_REFLECT_S8_MODE,
    };
    int8_t output[MIRROR_PAD_REFLECT_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_mirror_pad_s8(mirror_pad_reflect_s8_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, mirror_pad_reflect_s8_output, MIRROR_PAD_REFLECT_S8_OUTPUT_SIZE));
}

void mirror_pad_reflect_2d_s8_arm_mirror_pad_s8(void)
{
    const int32_t input_shape[] = MIRROR_PAD_REFLECT_2D_S8_INPUT_SHAPE;
    const int32_t output_shape[] = MIRROR_PAD_REFLECT_2D_S8_OUTPUT_SHAPE;
    const int32_t pad_before[] = MIRROR_PAD_REFLECT_2D_S8_PAD_BEFORE;
    const cmsis_nn_mirror_pad_params params = {
        .rank = MIRROR_PAD_REFLECT_2D_S8_RANK,
        .input_shape = input_shape,
        .output_shape = output_shape,
        .pad_before = pad_before,
        .mode = MIRROR_PAD_REFLECT_2D_S8_MODE,
    };
    int8_t output[MIRROR_PAD_REFLECT_2D_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_mirror_pad_s8(mirror_pad_reflect_2d_s8_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, mirror_pad_reflect_2d_s8_output, MIRROR_PAD_REFLECT_2D_S8_OUTPUT_SIZE));
}

void mirror_pad_reflect_wide_s8_arm_mirror_pad_s8(void)
{
    const int32_t input_shape[] = MIRROR_PAD_REFLECT_WIDE_S8_INPUT_SHAPE;
    const int32_t output_shape[] = MIRROR_PAD_REFLECT_WIDE_S8_OUTPUT_SHAPE;
    const int32_t pad_before[] = MIRROR_PAD_REFLECT_WIDE_S8_PAD_BEFORE;
    const cmsis_nn_mirror_pad_params params = {
        .rank = MIRROR_PAD_REFLECT_WIDE_S8_RANK,
        .input_shape = input_shape,
        .output_shape = output_shape,
        .pad_before = pad_before,
        .mode = MIRROR_PAD_REFLECT_WIDE_S8_MODE,
    };
    int8_t output[MIRROR_PAD_REFLECT_WIDE_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_mirror_pad_s8(mirror_pad_reflect_wide_s8_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, mirror_pad_reflect_wide_s8_output, MIRROR_PAD_REFLECT_WIDE_S8_OUTPUT_SIZE));
}