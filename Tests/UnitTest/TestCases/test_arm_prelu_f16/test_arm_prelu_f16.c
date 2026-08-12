/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "prelu_f16_data.h"

static const cmsis_nn_dims prelu_f16_input_dims = {1, 3, 4, 5};

static void prelu_f16_check(const cmsis_nn_dims *alpha_dims, const float16_t *alpha, const float16_t *expected)
{
    float16_t output[PRELU_F16_DST_SIZE] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_prelu_f16(&prelu_f16_input_dims,
                                    prelu_f16_input,
                                    alpha_dims,
                                    alpha,
                                    &prelu_f16_input_dims,
                                    output));

    for (int i = 0; i < PRELU_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(6.0e-3f, (float)expected[i], (float)output[i]);
    }
}

void prelu_f16_alpha_per_channel_arm_prelu_f16(void)
{
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 5};
    prelu_f16_check(&alpha_dims, prelu_f16_alpha_chan, prelu_f16_ref_chan);
}

void prelu_f16_alpha_full_arm_prelu_f16(void)
{
    const cmsis_nn_dims alpha_dims = {1, 3, 4, 5};
    prelu_f16_check(&alpha_dims, prelu_f16_alpha_full, prelu_f16_ref_full);
}

void prelu_f16_alpha_scalar_arm_prelu_f16(void)
{
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 1};
    prelu_f16_check(&alpha_dims, prelu_f16_alpha_scal, prelu_f16_ref_scal);
}

void prelu_f16_alpha_broadcast_w_arm_prelu_f16(void)
{
    const cmsis_nn_dims alpha_dims = {1, 3, 1, 5};
    prelu_f16_check(&alpha_dims, prelu_f16_alpha_gen, prelu_f16_ref_gen);
}

void prelu_f16_arg_error_arm_prelu_f16(void)
{
    float16_t output[PRELU_F16_DST_SIZE] = {0};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 5};
    // Alpha dim that is neither 1 nor the input dim
    const cmsis_nn_dims bad_alpha_dims = {1, 1, 1, 3};
    // Output dims that differ from input dims
    const cmsis_nn_dims bad_output_dims = {1, 3, 4, 4};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_prelu_f16(
            &prelu_f16_input_dims, NULL, &alpha_dims, prelu_f16_alpha_chan, &prelu_f16_input_dims, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_f16(&prelu_f16_input_dims,
                                    prelu_f16_input,
                                    &bad_alpha_dims,
                                    prelu_f16_alpha_chan,
                                    &prelu_f16_input_dims,
                                    output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_f16(&prelu_f16_input_dims,
                                    prelu_f16_input,
                                    &alpha_dims,
                                    prelu_f16_alpha_chan,
                                    &bad_output_dims,
                                    output));
}
