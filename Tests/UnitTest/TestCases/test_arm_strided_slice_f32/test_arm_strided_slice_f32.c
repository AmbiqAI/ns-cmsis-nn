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

#include "slice_f32_data.h"

void slice_f32_arm_strided_slice_f32(void)
{
    float32_t output[SLICE_F32_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_F32_INPUT_N, SLICE_F32_INPUT_H, SLICE_F32_INPUT_W, SLICE_F32_INPUT_C};
    const cmsis_nn_dims begin_dims = {SLICE_F32_BEGIN_N, SLICE_F32_BEGIN_H, SLICE_F32_BEGIN_W, SLICE_F32_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_F32_STRIDES_N, SLICE_F32_STRIDES_H, SLICE_F32_STRIDES_W, SLICE_F32_STRIDES_C};
    const cmsis_nn_dims output_dims = {SLICE_F32_OUTPUT_N, SLICE_F32_OUTPUT_H, SLICE_F32_OUTPUT_W, SLICE_F32_OUTPUT_C};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_strided_slice_f32(slice_f32_input, output, &input_dims, &begin_dims, &stride_dims, &output_dims));

    for (int i = 0; i < SLICE_F32_OUTPUT_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, slice_f32_output_ref[i], output[i]);
    }
}

void slice_f32_general_arm_strided_slice_f32(void)
{
    float32_t output[SLICE_F32_GEN_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_F32_INPUT_N, SLICE_F32_INPUT_H, SLICE_F32_INPUT_W, SLICE_F32_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        SLICE_F32_GEN_BEGIN_N, SLICE_F32_GEN_BEGIN_H, SLICE_F32_GEN_BEGIN_W, SLICE_F32_GEN_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_F32_GEN_STRIDES_N, SLICE_F32_GEN_STRIDES_H, SLICE_F32_GEN_STRIDES_W, SLICE_F32_GEN_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        SLICE_F32_GEN_OUTPUT_N, SLICE_F32_GEN_OUTPUT_H, SLICE_F32_GEN_OUTPUT_W, SLICE_F32_GEN_OUTPUT_C};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_strided_slice_f32(slice_f32_input, output, &input_dims, &begin_dims, &stride_dims, &output_dims));

    for (int i = 0; i < SLICE_F32_GEN_OUTPUT_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-6f, slice_f32_gen_output_ref[i], output[i]);
    }
}
