/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "unity.h"
#include <arm_nnfunctions.h>

#include "../Utils/validate.h"

static const int32_t prelu_identity_multiplier = 1073741824;
static const int32_t prelu_identity_shift = 1;

void prelu_scalar_alpha_s16_arm_prelu_s16(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 1, 4};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 4};
    const int16_t input_data[] = {-4, -1, 2, 5};
    const int16_t alpha_data[] = {3};
    const int16_t expected[] = {-12, -3, 2, 5};
    int16_t output[4] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s16(&input_dims,
                      input_data,
                      &alpha_dims,
                      alpha_data,
                      0,
                      0,
                      0,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      &output_dims,
                      output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, expected, 4));
}

void prelu_per_channel_s16_arm_prelu_s16(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 1, 4};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 4};
    const cmsis_nn_dims output_dims = {1, 1, 1, 4};
    const int16_t input_data[] = {-4, -1, 2, 5};
    const int16_t alpha_data[] = {3, 2, 1, 4};
    const int16_t expected[] = {-12, -2, 2, 5};
    int16_t output[4] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s16(&input_dims,
                      input_data,
                      &alpha_dims,
                      alpha_data,
                      0,
                      0,
                      0,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      &output_dims,
                      output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, expected, 4));
}

void prelu_alpha_width_broadcast_s16_arm_prelu_s16(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 2, 2};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 2};
    const cmsis_nn_dims output_dims = {1, 1, 2, 2};
    const int16_t input_data[] = {-4, 6, -2, 3};
    const int16_t alpha_data[] = {3, 5};
    const int16_t expected[] = {-12, 6, -6, 3};
    int16_t output[4] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s16(&input_dims,
                      input_data,
                      &alpha_dims,
                      alpha_data,
                      0,
                      0,
                      0,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      &output_dims,
                      output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, expected, 4));
}

void prelu_output_shape_mismatch_s16_arm_prelu_s16(void)
{
    /* PReLU is elementwise in its input, so the output shape must be the input shape. That is a
     * stricter rule than the shared broadcast check: an alpha wider than the input broadcasts to a
     * legal shape, and only this guard rejects it. */
    const cmsis_nn_dims input_dims_1c = {1, 1, 1, 1};
    const cmsis_nn_dims alpha_dims_3c = {1, 1, 1, 3};
    const cmsis_nn_dims output_dims_3c = {1, 1, 1, 3};
    const int16_t wide_alpha[3] = {2, 2, 2};
    int16_t wide_output[3] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_s16(&input_dims_1c,
                                    wide_alpha,
                                    &alpha_dims_3c,
                                    wide_alpha,
                                    0,
                                    0,
                                    0,
                                    prelu_identity_multiplier,
                                    prelu_identity_shift,
                                    prelu_identity_multiplier,
                                    prelu_identity_shift,
                                    &output_dims_3c,
                                    wide_output));

    /* an output shape that is the broadcast shape of neither operand */
    const cmsis_nn_dims input_dims = {1, 1, 1, 2};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 2};
    const cmsis_nn_dims output_dims = {1, 1, 1, 3};
    const int16_t input_data[] = {-2, 4};
    const int16_t alpha_data[] = {2, 2};
    int16_t output[3] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s16(&input_dims,
                      input_data,
                      &alpha_dims,
                      alpha_data,
                      0,
                      0,
                      0,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      &output_dims,
                      output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
}
/* Regression for the NHWC broadcast walk (issue #336): alpha is a per-batch scalar (2,1,1,1)
 * against an input whose rows are a single pixel with more than one channel (2,2,1,2). Each row
 * of alpha is a single element; the previous walk failed to advance its row pointer and then
 * rewound it off the front of the buffer, reading out of bounds and returning SUCCESS with wrong
 * values. PReLU is elementwise in its input, so only this operand order is representable.
 * The quantization is the identity, so a negative input maps to alpha * input. */
void prelu_broadcast_batch_scalar_alpha_s16_arm_prelu_s16(void)
{
    const int16_t input[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    const int16_t alpha[2] = {2, 3};
    const int16_t expected[8] = {-2, 2, -6, 4, -15, 6, -21, 8};
    int16_t output[8] = {0};
    const cmsis_nn_dims input_dims = {2, 2, 1, 2};
    const cmsis_nn_dims alpha_dims = {2, 1, 1, 1};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_prelu_s16(
            &input_dims, input, &alpha_dims, alpha, 0, 0, 0, 1073741824, 1, 1073741824, 1, &output_dims, output));
    TEST_ASSERT_TRUE(validate_s16(output, expected, 8));
}

/* The walk indexes alpha by its own dims, so an alpha that does not broadcast into the input is
 * now rejected instead of silently producing a partial result. */
void prelu_dims_arg_error_s16_arm_prelu_s16(void)
{
    const int16_t input[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    const int16_t alpha[4] = {1, 2, 3, 4};
    int16_t output[8] = {0};
    const cmsis_nn_dims input_dims = {2, 2, 1, 2};
    const cmsis_nn_dims input_dims_1c = {2, 2, 1, 1};
    const cmsis_nn_dims alpha_dims_3h = {1, 3, 1, 1};
    const cmsis_nn_dims alpha_dims_0h = {2, 0, 1, 1};
    const cmsis_nn_dims alpha_dims_ok = {2, 1, 1, 1};

    /* alpha h = 3 does not broadcast into input h = 2 */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_prelu_s16(
            &input_dims, input, &alpha_dims_3h, alpha, 0, 0, 0, 1073741824, 1, 1073741824, 1, &input_dims, output));
    /* an empty alpha dimension is rejected rather than treated as a no-op */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_s16(&input_dims_1c,
                                    input,
                                    &alpha_dims_0h,
                                    alpha,
                                    0,
                                    0,
                                    0,
                                    1073741824,
                                    1,
                                    1073741824,
                                    1,
                                    &input_dims_1c,
                                    output));
    /* a null operand is rejected */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_prelu_s16(
            &input_dims, NULL, &alpha_dims_ok, alpha, 0, 0, 0, 1073741824, 1, 1073741824, 1, &input_dims, output));
}
