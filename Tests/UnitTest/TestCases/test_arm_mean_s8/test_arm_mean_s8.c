/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <stdlib.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
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

/* axis = [H, W] against a plain reference over H and W 1..9, C 1..37 (both sides of the four-channel groups) and
 * batches 1..2, with output offsets and shifts that reach both clamp ends (#678). */
void mean_axis_hw_sweep_arm_mean_s8(void)
{
    static int8_t in[2 * 9 * 9 * 37];
    static int8_t out[2 * 37 + 4];
    for (int32_t i = 0; i < (int32_t)sizeof(in); i++)
    {
        in[i] = (int8_t)(i * 97 + 13);
    }
    const cmsis_nn_dims axis = {0, 1, 1, 0};
    int32_t cases = 0;
    for (int32_t n = 1; n <= 2; n++)
    {
        for (int32_t h = 1; h <= 9; h += 2)
        {
            for (int32_t w = 1; w <= 9; w++)
            {
                for (int32_t c = 1; c <= 37; c += (c < 9 ? 1 : 7))
                {
                    const cmsis_nn_dims in_dims = {n, h, w, c};
                    const cmsis_nn_dims out_dims = {n, 1, 1, c};
                    const int32_t in_off = (cases % 3) - 1 + (cases % 5) * 20;
                    const int32_t out_off = (cases % 7) * 30 - 90;
                    const int32_t out_mult = 1073741824 + (cases % 11) * 97000000;
                    const int32_t out_shift = -((cases % 9) + 1);
                    memset(out, 0x5A, sizeof(out));
                    TEST_ASSERT_EQUAL(
                        ARM_CMSIS_NN_SUCCESS,
                        arm_mean_s8(in, &in_dims, in_off, &axis, out, &out_dims, out_off, out_mult, out_shift));
                    for (int32_t b = 0; b < n; b++)
                    {
                        for (int32_t ch = 0; ch < c; ch++)
                        {
                            int32_t acc = in_off * h * w;
                            for (int32_t i = 0; i < h * w; i++)
                            {
                                acc += in[(b * h * w + i) * c + ch];
                            }
                            acc = arm_nn_requantize(acc, out_mult, out_shift) + out_off;
                            acc = acc < -128 ? -128 : (acc > 127 ? 127 : acc);
                            TEST_ASSERT_EQUAL_INT8((int8_t)acc, out[b * c + ch]);
                        }
                    }
                    for (int32_t i = n * c; i < (int32_t)sizeof(out); i++)
                    {
                        TEST_ASSERT_EQUAL_INT8(0x5A, out[i]);
                    }
                    cases++;
                }
            }
        }
    }
}
