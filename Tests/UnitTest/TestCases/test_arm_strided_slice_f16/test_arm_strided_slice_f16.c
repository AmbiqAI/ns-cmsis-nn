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

#include "slice_f16_data.h"

void slice_f16_arm_strided_slice_f16(void)
{
    float16_t output[SLICE_F16_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_F16_INPUT_N, SLICE_F16_INPUT_H, SLICE_F16_INPUT_W, SLICE_F16_INPUT_C};
    const cmsis_nn_dims begin_dims = {SLICE_F16_BEGIN_N, SLICE_F16_BEGIN_H, SLICE_F16_BEGIN_W, SLICE_F16_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_F16_STRIDES_N, SLICE_F16_STRIDES_H, SLICE_F16_STRIDES_W, SLICE_F16_STRIDES_C};
    const cmsis_nn_dims output_dims = {SLICE_F16_OUTPUT_N, SLICE_F16_OUTPUT_H, SLICE_F16_OUTPUT_W, SLICE_F16_OUTPUT_C};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_strided_slice_f16(slice_f16_input, output, &input_dims, &begin_dims, &stride_dims, &output_dims));

    for (int i = 0; i < SLICE_F16_OUTPUT_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)slice_f16_output_ref[i], (float)output[i]);
    }
}
