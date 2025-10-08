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

#include "../TestData/mean_axis_n_s8/test_data.h"
#include "../TestData/mean_axis_h_s8/test_data.h"
#include "../TestData/mean_axis_w_s8/test_data.h"
#include "../TestData/mean_axis_c_s8/test_data.h"
#include "../TestData/mean_axis_hw_s8/test_data.h"
#include "../TestData/mean_axis_nhwc_s8/test_data.h"
#include "../TestData/mean_axis_hwc_s8/test_data.h"
#include "../TestData/mean_axis_wc_s8/test_data.h"


#include "../Utils/validate.h"


void mean_axis_n_arm_mean_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[MEAN_AXIS_N_S8_OUTPUT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = MEAN_AXIS_N_S8_IN_DIM;
    const cmsis_nn_dims axis_dims = MEAN_AXIS_N_S8_AXIS_DIM;
    const cmsis_nn_dims output_dims = MEAN_AXIS_N_S8_OUT_DIM;

    const int8_t *input_data = mean_axis_n_s8_input_tensor;
    const int8_t *const output_ref = mean_axis_n_s8_output;
    const int32_t output_ref_size = MEAN_AXIS_N_S8_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_mean_s8(
        input_data,
        &input_dims,
        MEAN_AXIS_N_S8_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        MEAN_AXIS_N_S8_OUTPUT_OFFSET,
        MEAN_AXIS_N_S8_OUTPUT_MULTIPLIER,
        MEAN_AXIS_N_S8_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void mean_axis_h_arm_mean_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[MEAN_AXIS_H_S8_OUTPUT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = MEAN_AXIS_H_S8_IN_DIM;
    const cmsis_nn_dims axis_dims = MEAN_AXIS_H_S8_AXIS_DIM;
    const cmsis_nn_dims output_dims = MEAN_AXIS_H_S8_OUT_DIM;

    const int8_t *input_data = mean_axis_h_s8_input_tensor;
    const int8_t *const output_ref = mean_axis_h_s8_output;
    const int32_t output_ref_size = MEAN_AXIS_H_S8_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_mean_s8(
        input_data,
        &input_dims,
        MEAN_AXIS_H_S8_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        MEAN_AXIS_H_S8_OUTPUT_OFFSET,
        MEAN_AXIS_H_S8_OUTPUT_MULTIPLIER,
        MEAN_AXIS_H_S8_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void mean_axis_w_arm_mean_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[MEAN_AXIS_W_S8_OUTPUT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = MEAN_AXIS_W_S8_IN_DIM;
    const cmsis_nn_dims axis_dims = MEAN_AXIS_W_S8_AXIS_DIM;
    const cmsis_nn_dims output_dims = MEAN_AXIS_W_S8_OUT_DIM;

    const int8_t *input_data = mean_axis_w_s8_input_tensor;
    const int8_t *const output_ref = mean_axis_w_s8_output;
    const int32_t output_ref_size = MEAN_AXIS_W_S8_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_mean_s8(
        input_data,
        &input_dims,
        MEAN_AXIS_W_S8_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        MEAN_AXIS_W_S8_OUTPUT_OFFSET,
        MEAN_AXIS_W_S8_OUTPUT_MULTIPLIER,
        MEAN_AXIS_W_S8_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void mean_axis_c_arm_mean_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[MEAN_AXIS_C_S8_OUTPUT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = MEAN_AXIS_C_S8_IN_DIM;
    const cmsis_nn_dims axis_dims = MEAN_AXIS_C_S8_AXIS_DIM;
    const cmsis_nn_dims output_dims = MEAN_AXIS_C_S8_OUT_DIM;

    const int8_t *input_data = mean_axis_c_s8_input_tensor;
    const int8_t *const output_ref = mean_axis_c_s8_output;
    const int32_t output_ref_size = MEAN_AXIS_C_S8_OUTPUT_SIZE;


    arm_cmsis_nn_status result = arm_mean_s8(
        input_data,
        &input_dims,
        MEAN_AXIS_C_S8_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        MEAN_AXIS_C_S8_OUTPUT_OFFSET,
        MEAN_AXIS_C_S8_OUTPUT_MULTIPLIER,
        MEAN_AXIS_C_S8_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void mean_axis_hw_arm_mean_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[MEAN_AXIS_HW_S8_OUTPUT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = MEAN_AXIS_HW_S8_IN_DIM;
    const cmsis_nn_dims axis_dims = MEAN_AXIS_HW_S8_AXIS_DIM;
    const cmsis_nn_dims output_dims = MEAN_AXIS_HW_S8_OUT_DIM;

    const int8_t *input_data = mean_axis_hw_s8_input_tensor;
    const int8_t *const output_ref = mean_axis_hw_s8_output;
    const int32_t output_ref_size = MEAN_AXIS_HW_S8_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_mean_s8(
        input_data,
        &input_dims,
        MEAN_AXIS_HW_S8_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        MEAN_AXIS_HW_S8_OUTPUT_OFFSET,
        MEAN_AXIS_HW_S8_OUTPUT_MULTIPLIER,
        MEAN_AXIS_HW_S8_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void mean_axis_nhwc_arm_mean_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[MEAN_AXIS_NHWC_S8_OUTPUT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = MEAN_AXIS_NHWC_S8_IN_DIM;
    const cmsis_nn_dims axis_dims = MEAN_AXIS_NHWC_S8_AXIS_DIM;
    const cmsis_nn_dims output_dims = MEAN_AXIS_NHWC_S8_OUT_DIM;

    const int8_t *input_data = mean_axis_nhwc_s8_input_tensor;
    const int8_t *const output_ref = mean_axis_nhwc_s8_output;
    const int32_t output_ref_size = MEAN_AXIS_NHWC_S8_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_mean_s8(
        input_data,
        &input_dims,
        MEAN_AXIS_NHWC_S8_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        MEAN_AXIS_NHWC_S8_OUTPUT_OFFSET,
        MEAN_AXIS_NHWC_S8_OUTPUT_MULTIPLIER,
        MEAN_AXIS_NHWC_S8_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}
void mean_axis_hwc_arm_mean_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[MEAN_AXIS_HWC_S8_OUTPUT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = MEAN_AXIS_HWC_S8_IN_DIM;
    const cmsis_nn_dims axis_dims = MEAN_AXIS_HWC_S8_AXIS_DIM;
    const cmsis_nn_dims output_dims = MEAN_AXIS_HWC_S8_OUT_DIM;

    const int8_t *input_data = mean_axis_hwc_s8_input_tensor;
    const int8_t *const output_ref = mean_axis_hwc_s8_output;
    const int32_t output_ref_size = MEAN_AXIS_HWC_S8_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_mean_s8(
        input_data,
        &input_dims,
        MEAN_AXIS_HWC_S8_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        MEAN_AXIS_HWC_S8_OUTPUT_OFFSET,
        MEAN_AXIS_HWC_S8_OUTPUT_MULTIPLIER,
        MEAN_AXIS_HWC_S8_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}
void mean_axis_wc_arm_mean_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[MEAN_AXIS_WC_S8_OUTPUT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = MEAN_AXIS_WC_S8_IN_DIM;
    const cmsis_nn_dims axis_dims = MEAN_AXIS_WC_S8_AXIS_DIM;
    const cmsis_nn_dims output_dims = MEAN_AXIS_WC_S8_OUT_DIM;

    const int8_t *input_data = mean_axis_wc_s8_input_tensor;
    const int8_t *const output_ref = mean_axis_wc_s8_output;
    const int32_t output_ref_size = MEAN_AXIS_WC_S8_OUTPUT_SIZE;

    arm_cmsis_nn_status result = arm_mean_s8(
        input_data,
        &input_dims,
        MEAN_AXIS_WC_S8_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        MEAN_AXIS_WC_S8_OUTPUT_OFFSET,
        MEAN_AXIS_WC_S8_OUTPUT_MULTIPLIER,
        MEAN_AXIS_WC_S8_OUTPUT_SHIFT
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}
