/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/reduce_min_axis_n_s16/test_data.h"
#include "../TestData/reduce_min_axis_h_s16/test_data.h"
#include "../TestData/reduce_min_axis_w_s16/test_data.h"
#include "../TestData/reduce_min_axis_c_s16/test_data.h"
#include "../TestData/reduce_min_axis_hw_s16/test_data.h"
#include "../TestData/reduce_min_axis_nhwc_s16/test_data.h"
#include "../TestData/reduce_min_axis_hwc_s16/test_data.h"
#include "../TestData/reduce_min_axis_wc_s16/test_data.h"


#include "../Utils/validate.h"


void reduce_min_axis_n_arm_reduce_min_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_MIN_AXIS_N_S16_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_MIN_AXIS_N_S16_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_MIN_AXIS_N_S16_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_MIN_AXIS_N_S16_OUT_DIM;

    const int16_t *input_data = reduce_min_axis_n_s16_input_tensor;
    const int16_t *const output_ref = reduce_min_axis_n_s16_output;
    const int32_t output_ref_size = REDUCE_MIN_AXIS_N_S16_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_min_s16(
        input_data,
        &input_dims,
        &axis_dims,
        output_ptr,
        &output_dims
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void reduce_min_axis_h_arm_reduce_min_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_MIN_AXIS_H_S16_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_MIN_AXIS_H_S16_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_MIN_AXIS_H_S16_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_MIN_AXIS_H_S16_OUT_DIM;

    const int16_t *input_data = reduce_min_axis_h_s16_input_tensor;
    const int16_t *const output_ref = reduce_min_axis_h_s16_output;
    const int32_t output_ref_size = REDUCE_MIN_AXIS_H_S16_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_min_s16(
        input_data,
        &input_dims,
        &axis_dims,
        output_ptr,
        &output_dims
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void reduce_min_axis_w_arm_reduce_min_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_MIN_AXIS_W_S16_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_MIN_AXIS_W_S16_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_MIN_AXIS_W_S16_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_MIN_AXIS_W_S16_OUT_DIM;

    const int16_t *input_data = reduce_min_axis_w_s16_input_tensor;
    const int16_t *const output_ref = reduce_min_axis_w_s16_output;
    const int32_t output_ref_size = REDUCE_MIN_AXIS_W_S16_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_min_s16(
        input_data,
        &input_dims,
        &axis_dims,
        output_ptr,
        &output_dims
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void reduce_min_axis_c_arm_reduce_min_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_MIN_AXIS_C_S16_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_MIN_AXIS_C_S16_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_MIN_AXIS_C_S16_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_MIN_AXIS_C_S16_OUT_DIM;

    const int16_t *input_data = reduce_min_axis_c_s16_input_tensor;
    const int16_t *const output_ref = reduce_min_axis_c_s16_output;
    const int32_t output_ref_size = REDUCE_MIN_AXIS_C_S16_OUTPUT_SIZE;


    arm_cmsis_nn_status result = arm_reduce_min_s16(
        input_data,
        &input_dims,
        &axis_dims,
        output_ptr,
        &output_dims
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void reduce_min_axis_hw_arm_reduce_min_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_MIN_AXIS_HW_S16_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_MIN_AXIS_HW_S16_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_MIN_AXIS_HW_S16_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_MIN_AXIS_HW_S16_OUT_DIM;

    const int16_t *input_data = reduce_min_axis_hw_s16_input_tensor;
    const int16_t *const output_ref = reduce_min_axis_hw_s16_output;
    const int32_t output_ref_size = REDUCE_MIN_AXIS_HW_S16_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_min_s16(
        input_data,
        &input_dims,
        &axis_dims,
        output_ptr,
        &output_dims
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void reduce_min_axis_nhwc_arm_reduce_min_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_MIN_AXIS_NHWC_S16_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_MIN_AXIS_NHWC_S16_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_MIN_AXIS_NHWC_S16_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_MIN_AXIS_NHWC_S16_OUT_DIM;

    const int16_t *input_data = reduce_min_axis_nhwc_s16_input_tensor;
    const int16_t *const output_ref = reduce_min_axis_nhwc_s16_output;
    const int32_t output_ref_size = REDUCE_MIN_AXIS_NHWC_S16_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_min_s16(
        input_data,
        &input_dims,
        &axis_dims,
        output_ptr,
        &output_dims
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}
void reduce_min_axis_hwc_arm_reduce_min_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_MIN_AXIS_HWC_S16_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_MIN_AXIS_HWC_S16_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_MIN_AXIS_HWC_S16_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_MIN_AXIS_HWC_S16_OUT_DIM;

    const int16_t *input_data = reduce_min_axis_hwc_s16_input_tensor;
    const int16_t *const output_ref = reduce_min_axis_hwc_s16_output;
    const int32_t output_ref_size = REDUCE_MIN_AXIS_HWC_S16_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_min_s16(
        input_data,
        &input_dims,
        &axis_dims,
        output_ptr,
        &output_dims
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}
void reduce_min_axis_wc_arm_reduce_min_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[REDUCE_MIN_AXIS_WC_S16_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = REDUCE_MIN_AXIS_WC_S16_IN_DIM;
    const cmsis_nn_dims axis_dims = REDUCE_MIN_AXIS_WC_S16_AXIS_DIM;
    const cmsis_nn_dims output_dims = REDUCE_MIN_AXIS_WC_S16_OUT_DIM;

    const int16_t *input_data = reduce_min_axis_wc_s16_input_tensor;
    const int16_t *const output_ref = reduce_min_axis_wc_s16_output;
    const int32_t output_ref_size = REDUCE_MIN_AXIS_WC_S16_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_reduce_min_s16(
        input_data,
        &input_dims,
        &axis_dims,
        output_ptr,
        &output_dims
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}
