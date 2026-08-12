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

#include "rsum_f16_data.h"

static const cmsis_nn_dims rsum_f16_input_dims = {2, 3, 4, 5};

static void rsum_f16_check(const cmsis_nn_dims *axis_dims, const cmsis_nn_dims *output_dims, const float16_t *expected)
{
    float16_t output[120] = {0};
    const int32_t out_size = output_dims->n * output_dims->h * output_dims->w * output_dims->c;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_reduce_sum_f16(rsum_f16_input, &rsum_f16_input_dims, axis_dims, output, output_dims));

    for (int i = 0; i < out_size; ++i)
    {
        // Kernel and golden both accumulate in float32; tolerance covers
        // accumulation order plus the final float16 rounding.
        TEST_ASSERT_FLOAT_WITHIN(3.0e-2f, (float)expected[i], (float)output[i]);
    }
}

void rsum_f16_axis_c_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 4, 1};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_c);
}

void rsum_f16_axis_hwc_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 1, 1};
    const cmsis_nn_dims output_dims = {2, 1, 1, 1};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_hwc);
}

void rsum_f16_axis_all_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_all);
}

void rsum_f16_axis_hw_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 1, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 5};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_hw);
}

void rsum_f16_axis_h_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {2, 1, 4, 5};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_h);
}

void rsum_f16_arg_error_arm_reduce_sum_f16(void)
{
    float16_t output[8] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 4, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f16(NULL, &rsum_f16_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f16(rsum_f16_input, &rsum_f16_input_dims, NULL, output, &output_dims));
}
