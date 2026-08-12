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

#include "rsum_f32_data.h"

static const cmsis_nn_dims rsum_f32_input_dims = {2, 3, 4, 5};

static void rsum_f32_check(const cmsis_nn_dims *axis_dims, const cmsis_nn_dims *output_dims, const float32_t *expected)
{
    float32_t output[120] = {0};
    const int32_t out_size = output_dims->n * output_dims->h * output_dims->w * output_dims->c;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_reduce_sum_f32(rsum_f32_input, &rsum_f32_input_dims, axis_dims, output, output_dims));

    for (int i = 0; i < out_size; ++i)
    {
        // Accumulation-order tolerance: scalar vs MVE partial sums
        TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, expected[i], output[i]);
    }
}

void rsum_f32_axis_c_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 4, 1};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_c);
}

void rsum_f32_axis_hwc_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 1, 1};
    const cmsis_nn_dims output_dims = {2, 1, 1, 1};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_hwc);
}

void rsum_f32_axis_all_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_all);
}

void rsum_f32_axis_hw_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 1, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 5};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_hw);
}

void rsum_f32_axis_h_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {2, 1, 4, 5};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_h);
}

void rsum_f32_arg_error_arm_reduce_sum_f32(void)
{
    float32_t output[8] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 4, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f32(NULL, &rsum_f32_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f32(rsum_f32_input, &rsum_f32_input_dims, NULL, output, &output_dims));
}
