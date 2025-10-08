/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "../TestData/slice_center_3x3/test_data.h"
#include "../TestData/slice_strided_2x3/test_data.h"
#include "../TestData/slice_whole_slab/test_data.h"
#include "../TestData/slice_case2/test_data.h"
#include "../TestData/slice_case3/test_data.h"
#include "../TestData/slice_case4/test_data.h"
#include "../Utils/validate.h"
#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

void slice_center_3x3_arm_strided_slice_s8(void)
{
    const int8_t *input_ptr = slice_center_3x3_input_tensor;
    int8_t output_ptr[SLICE_CENTER_3X3_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_CENTER_3X3_INPUT_N, SLICE_CENTER_3X3_INPUT_H, SLICE_CENTER_3X3_INPUT_W, SLICE_CENTER_3X3_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        SLICE_CENTER_3X3_BEGIN_N, SLICE_CENTER_3X3_BEGIN_H, SLICE_CENTER_3X3_BEGIN_W, SLICE_CENTER_3X3_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_CENTER_3X3_STRIDES_N, SLICE_CENTER_3X3_STRIDES_H, SLICE_CENTER_3X3_STRIDES_W, SLICE_CENTER_3X3_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        SLICE_CENTER_3X3_OUTPUT_N, SLICE_CENTER_3X3_OUTPUT_H, SLICE_CENTER_3X3_OUTPUT_W, SLICE_CENTER_3X3_OUTPUT_C};


    const arm_cmsis_nn_status result = arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output_ptr, slice_center_3x3_output_tensor, SLICE_CENTER_3X3_OUTPUT_SIZE));
}

void slice_strided_2x3_arm_strided_slice_s8(void)
{

    const int8_t *input_ptr = slice_strided_2x3_input_tensor;
    int8_t output_ptr[SLICE_STRIDED_2X3_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_STRIDED_2X3_INPUT_N, SLICE_STRIDED_2X3_INPUT_H, SLICE_STRIDED_2X3_INPUT_W, SLICE_STRIDED_2X3_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        SLICE_STRIDED_2X3_BEGIN_N, SLICE_STRIDED_2X3_BEGIN_H, SLICE_STRIDED_2X3_BEGIN_W, SLICE_STRIDED_2X3_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_STRIDED_2X3_STRIDES_N, SLICE_STRIDED_2X3_STRIDES_H, SLICE_STRIDED_2X3_STRIDES_W, SLICE_STRIDED_2X3_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        SLICE_STRIDED_2X3_OUTPUT_N, SLICE_STRIDED_2X3_OUTPUT_H, SLICE_STRIDED_2X3_OUTPUT_W, SLICE_STRIDED_2X3_OUTPUT_C};

    const arm_cmsis_nn_status result = arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output_ptr, slice_strided_2x3_output_tensor, SLICE_STRIDED_2X3_OUTPUT_SIZE));
}

void slice_whole_slab_arm_strided_slice_s8(void)
{
    const int8_t *input_ptr = slice_whole_slab_input_tensor;
    int8_t output_ptr[SLICE_WHOLE_SLAB_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_WHOLE_SLAB_INPUT_N, SLICE_WHOLE_SLAB_INPUT_H, SLICE_WHOLE_SLAB_INPUT_W, SLICE_WHOLE_SLAB_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        SLICE_WHOLE_SLAB_BEGIN_N, SLICE_WHOLE_SLAB_BEGIN_H, SLICE_WHOLE_SLAB_BEGIN_W, SLICE_WHOLE_SLAB_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_WHOLE_SLAB_STRIDES_N, SLICE_WHOLE_SLAB_STRIDES_H, SLICE_WHOLE_SLAB_STRIDES_W, SLICE_WHOLE_SLAB_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        SLICE_WHOLE_SLAB_OUTPUT_N, SLICE_WHOLE_SLAB_OUTPUT_H, SLICE_WHOLE_SLAB_OUTPUT_W, SLICE_WHOLE_SLAB_OUTPUT_C};

    const arm_cmsis_nn_status result = arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output_ptr, slice_whole_slab_output_tensor, SLICE_WHOLE_SLAB_OUTPUT_SIZE));
}

void slice_case2_arm_strided_slice_s8(void)
{
    const int8_t *input_ptr = slice_case2_input_tensor;
    int8_t output_ptr[SLICE_CASE2_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_CASE2_INPUT_N, SLICE_CASE2_INPUT_H, SLICE_CASE2_INPUT_W, SLICE_CASE2_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        SLICE_CASE2_BEGIN_N, SLICE_CASE2_BEGIN_H, SLICE_CASE2_BEGIN_W, SLICE_CASE2_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_CASE2_STRIDES_N, SLICE_CASE2_STRIDES_H, SLICE_CASE2_STRIDES_W, SLICE_CASE2_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        SLICE_CASE2_OUTPUT_N, SLICE_CASE2_OUTPUT_H, SLICE_CASE2_OUTPUT_W, SLICE_CASE2_OUTPUT_C};

    const arm_cmsis_nn_status result = arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output_ptr, slice_case2_output_tensor, SLICE_CASE2_OUTPUT_SIZE));
}

void slice_case3_arm_strided_slice_s8(void)
{
    const int8_t *input_ptr = slice_case3_input_tensor;
    int8_t output_ptr[SLICE_CASE3_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_CASE3_INPUT_N, SLICE_CASE3_INPUT_H, SLICE_CASE3_INPUT_W, SLICE_CASE3_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        SLICE_CASE3_BEGIN_N, SLICE_CASE3_BEGIN_H, SLICE_CASE3_BEGIN_W, SLICE_CASE3_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_CASE3_STRIDES_N, SLICE_CASE3_STRIDES_H, SLICE_CASE3_STRIDES_W, SLICE_CASE3_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        SLICE_CASE3_OUTPUT_N, SLICE_CASE3_OUTPUT_H, SLICE_CASE3_OUTPUT_W, SLICE_CASE3_OUTPUT_C};

    const arm_cmsis_nn_status result = arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output_ptr, slice_case3_output_tensor, SLICE_CASE3_OUTPUT_SIZE));
}

void slice_case4_arm_strided_slice_s8(void)
{
    const int8_t *input_ptr = slice_case4_input_tensor;
    int8_t output_ptr[SLICE_CASE4_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {SLICE_CASE4_INPUT_N, SLICE_CASE4_INPUT_H, SLICE_CASE4_INPUT_W, SLICE_CASE4_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        SLICE_CASE4_BEGIN_N, SLICE_CASE4_BEGIN_H, SLICE_CASE4_BEGIN_W, SLICE_CASE4_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        SLICE_CASE4_STRIDES_N, SLICE_CASE4_STRIDES_H, SLICE_CASE4_STRIDES_W, SLICE_CASE4_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        SLICE_CASE4_OUTPUT_N, SLICE_CASE4_OUTPUT_H, SLICE_CASE4_OUTPUT_W, SLICE_CASE4_OUTPUT_C};

    const arm_cmsis_nn_status result = arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output_ptr, slice_case4_output_tensor, SLICE_CASE4_OUTPUT_SIZE));
}
