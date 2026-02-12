/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "../TestData/slice_center_3x3_s32/test_data.h"
#include "../TestData/slice_strided_2x3_s32/test_data.h"
#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

static void validate_s32(const int32_t *actual, const int32_t *expected, const int32_t length)
{
    for (int32_t i = 0; i < length; ++i)
    {
        TEST_ASSERT_EQUAL_INT32(expected[i], actual[i]);
    }
}

void slice_center_3x3_arm_strided_slice_s32(void)
{
    const int32_t *input_ptr = slice_center_3x3_s32_input_tensor;
    int32_t output_ptr[SLICE_CENTER_3X3_S32_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {
        SLICE_CENTER_3X3_S32_INPUT_N, SLICE_CENTER_3X3_S32_INPUT_H, SLICE_CENTER_3X3_S32_INPUT_W, SLICE_CENTER_3X3_S32_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        SLICE_CENTER_3X3_S32_BEGIN_N, SLICE_CENTER_3X3_S32_BEGIN_H, SLICE_CENTER_3X3_S32_BEGIN_W, SLICE_CENTER_3X3_S32_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_CENTER_3X3_S32_STRIDES_N, SLICE_CENTER_3X3_S32_STRIDES_H, SLICE_CENTER_3X3_S32_STRIDES_W, SLICE_CENTER_3X3_S32_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        SLICE_CENTER_3X3_S32_OUTPUT_N, SLICE_CENTER_3X3_S32_OUTPUT_H, SLICE_CENTER_3X3_S32_OUTPUT_W, SLICE_CENTER_3X3_S32_OUTPUT_C};

    const arm_cmsis_nn_status result =
        arm_strided_slice_s32(input_ptr, output_ptr, &input_dims, &begin_dims, &stride_dims, &output_dims);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    validate_s32(output_ptr, slice_center_3x3_s32_output, SLICE_CENTER_3X3_S32_OUTPUT_SIZE);
}

void slice_strided_2x3_arm_strided_slice_s32(void)
{
    const int32_t *input_ptr = slice_strided_2x3_s32_input_tensor;
    int32_t output_ptr[SLICE_STRIDED_2X3_S32_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_STRIDED_2X3_S32_INPUT_N,
                                      SLICE_STRIDED_2X3_S32_INPUT_H,
                                      SLICE_STRIDED_2X3_S32_INPUT_W,
                                      SLICE_STRIDED_2X3_S32_INPUT_C};
    const cmsis_nn_dims begin_dims = {SLICE_STRIDED_2X3_S32_BEGIN_N,
                                      SLICE_STRIDED_2X3_S32_BEGIN_H,
                                      SLICE_STRIDED_2X3_S32_BEGIN_W,
                                      SLICE_STRIDED_2X3_S32_BEGIN_C};
    const cmsis_nn_dims stride_dims = {SLICE_STRIDED_2X3_S32_STRIDES_N,
                                       SLICE_STRIDED_2X3_S32_STRIDES_H,
                                       SLICE_STRIDED_2X3_S32_STRIDES_W,
                                       SLICE_STRIDED_2X3_S32_STRIDES_C};
    const cmsis_nn_dims output_dims = {SLICE_STRIDED_2X3_S32_OUTPUT_N,
                                       SLICE_STRIDED_2X3_S32_OUTPUT_H,
                                       SLICE_STRIDED_2X3_S32_OUTPUT_W,
                                       SLICE_STRIDED_2X3_S32_OUTPUT_C};

    const arm_cmsis_nn_status result =
        arm_strided_slice_s32(input_ptr, output_ptr, &input_dims, &begin_dims, &stride_dims, &output_dims);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    validate_s32(output_ptr, slice_strided_2x3_s32_output, SLICE_STRIDED_2X3_S32_OUTPUT_SIZE);
}
