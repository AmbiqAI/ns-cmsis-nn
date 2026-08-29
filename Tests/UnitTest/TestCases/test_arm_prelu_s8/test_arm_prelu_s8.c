/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "unity.h"
#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>

#include "../Utils/validate.h"

#include "../TestData/prelu_alpha_height_broadcast_s8/test_data.h"
#include "../TestData/prelu_alpha_hw_broadcast_s8/test_data.h"
#include "../TestData/prelu_alpha_width_broadcast_s8/test_data.h"
#include "../TestData/prelu_per_channel_s8/test_data.h"
#include "../TestData/prelu_scalar_alpha_s8/test_data.h"
#include "../TestData/prelu_scalar_input_s8/test_data.h"


void prelu_per_channel_s8_arm_prelu_s8(void)
{
    const cmsis_nn_dims input_dims = {PRELU_PER_CHANNEL_S8_INPUT_N,
                                      PRELU_PER_CHANNEL_S8_INPUT_H,
                                      PRELU_PER_CHANNEL_S8_INPUT_W,
                                      PRELU_PER_CHANNEL_S8_INPUT_C};
    const cmsis_nn_dims alpha_dims = {PRELU_PER_CHANNEL_S8_ALPHA_N,
                                      PRELU_PER_CHANNEL_S8_ALPHA_H,
                                      PRELU_PER_CHANNEL_S8_ALPHA_W,
                                      PRELU_PER_CHANNEL_S8_ALPHA_C};
    const cmsis_nn_dims output_dims = {PRELU_PER_CHANNEL_S8_OUTPUT_N,
                                       PRELU_PER_CHANNEL_S8_OUTPUT_H,
                                       PRELU_PER_CHANNEL_S8_OUTPUT_W,
                                       PRELU_PER_CHANNEL_S8_OUTPUT_C};

    int8_t output[PRELU_PER_CHANNEL_S8_OUTPUT_LEN] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s8(&input_dims,
                     prelu_per_channel_s8_input_tensor,
                     &alpha_dims,
                     prelu_per_channel_s8_alpha_input_tensor,
                     PRELU_PER_CHANNEL_S8_INPUT_OFFSET,
                     PRELU_PER_CHANNEL_S8_ALPHA_OFFSET,
                     PRELU_PER_CHANNEL_S8_OUTPUT_OFFSET,
                     PRELU_PER_CHANNEL_S8_OUTPUT_MULTIPLIER_1,
                     PRELU_PER_CHANNEL_S8_OUTPUT_SHIFT_1,
                     PRELU_PER_CHANNEL_S8_OUTPUT_MULTIPLIER_2,
                     PRELU_PER_CHANNEL_S8_OUTPUT_SHIFT_2,
                     &output_dims,
                     output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, prelu_per_channel_s8_output, PRELU_PER_CHANNEL_S8_OUTPUT_LEN));
}

void prelu_scalar_input_s8_arm_prelu_s8(void)
{
    const cmsis_nn_dims input_dims = {PRELU_SCALAR_INPUT_S8_INPUT_N,
                                      PRELU_SCALAR_INPUT_S8_INPUT_H,
                                      PRELU_SCALAR_INPUT_S8_INPUT_W,
                                      PRELU_SCALAR_INPUT_S8_INPUT_C};
    const cmsis_nn_dims alpha_dims = {PRELU_SCALAR_INPUT_S8_ALPHA_N,
                                      PRELU_SCALAR_INPUT_S8_ALPHA_H,
                                      PRELU_SCALAR_INPUT_S8_ALPHA_W,
                                      PRELU_SCALAR_INPUT_S8_ALPHA_C};
    const cmsis_nn_dims output_dims = {PRELU_SCALAR_INPUT_S8_OUTPUT_N,
                                       PRELU_SCALAR_INPUT_S8_OUTPUT_H,
                                       PRELU_SCALAR_INPUT_S8_OUTPUT_W,
                                       PRELU_SCALAR_INPUT_S8_OUTPUT_C};

    int8_t output[PRELU_SCALAR_INPUT_S8_OUTPUT_LEN] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s8(&input_dims,
                     prelu_scalar_input_s8_input_tensor,
                     &alpha_dims,
                     prelu_scalar_input_s8_alpha_input_tensor,
                     PRELU_SCALAR_INPUT_S8_INPUT_OFFSET,
                     PRELU_SCALAR_INPUT_S8_ALPHA_OFFSET,
                     PRELU_SCALAR_INPUT_S8_OUTPUT_OFFSET,
                     PRELU_SCALAR_INPUT_S8_OUTPUT_MULTIPLIER_1,
                     PRELU_SCALAR_INPUT_S8_OUTPUT_SHIFT_1,
                     PRELU_SCALAR_INPUT_S8_OUTPUT_MULTIPLIER_2,
                     PRELU_SCALAR_INPUT_S8_OUTPUT_SHIFT_2,
                     &output_dims,
                     output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, prelu_scalar_input_s8_output, PRELU_SCALAR_INPUT_S8_OUTPUT_LEN));
}

void prelu_scalar_alpha_s8_arm_prelu_s8(void)
{
    const cmsis_nn_dims input_dims = {PRELU_SCALAR_ALPHA_S8_INPUT_N,
                                      PRELU_SCALAR_ALPHA_S8_INPUT_H,
                                      PRELU_SCALAR_ALPHA_S8_INPUT_W,
                                      PRELU_SCALAR_ALPHA_S8_INPUT_C};
    const cmsis_nn_dims alpha_dims = {PRELU_SCALAR_ALPHA_S8_ALPHA_N,
                                      PRELU_SCALAR_ALPHA_S8_ALPHA_H,
                                      PRELU_SCALAR_ALPHA_S8_ALPHA_W,
                                      PRELU_SCALAR_ALPHA_S8_ALPHA_C};
    const cmsis_nn_dims output_dims = {PRELU_SCALAR_ALPHA_S8_OUTPUT_N,
                                       PRELU_SCALAR_ALPHA_S8_OUTPUT_H,
                                       PRELU_SCALAR_ALPHA_S8_OUTPUT_W,
                                       PRELU_SCALAR_ALPHA_S8_OUTPUT_C};

    int8_t output[PRELU_SCALAR_ALPHA_S8_OUTPUT_LEN] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s8(&input_dims,
                     prelu_scalar_alpha_s8_input_tensor,
                     &alpha_dims,
                     prelu_scalar_alpha_s8_alpha_input_tensor,
                     PRELU_SCALAR_ALPHA_S8_INPUT_OFFSET,
                     PRELU_SCALAR_ALPHA_S8_ALPHA_OFFSET,
                     PRELU_SCALAR_ALPHA_S8_OUTPUT_OFFSET,
                     PRELU_SCALAR_ALPHA_S8_OUTPUT_MULTIPLIER_1,
                     PRELU_SCALAR_ALPHA_S8_OUTPUT_SHIFT_1,
                     PRELU_SCALAR_ALPHA_S8_OUTPUT_MULTIPLIER_2,
                     PRELU_SCALAR_ALPHA_S8_OUTPUT_SHIFT_2,
                     &output_dims,
                     output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, prelu_scalar_alpha_s8_output, PRELU_SCALAR_ALPHA_S8_OUTPUT_LEN));
}

void prelu_alpha_width_broadcast_s8_arm_prelu_s8(void)
{
    const cmsis_nn_dims input_dims = {PRELU_ALPHA_WIDTH_BROADCAST_S8_INPUT_N,
                                      PRELU_ALPHA_WIDTH_BROADCAST_S8_INPUT_H,
                                      PRELU_ALPHA_WIDTH_BROADCAST_S8_INPUT_W,
                                      PRELU_ALPHA_WIDTH_BROADCAST_S8_INPUT_C};
    const cmsis_nn_dims alpha_dims = {PRELU_ALPHA_WIDTH_BROADCAST_S8_ALPHA_N,
                                      PRELU_ALPHA_WIDTH_BROADCAST_S8_ALPHA_H,
                                      PRELU_ALPHA_WIDTH_BROADCAST_S8_ALPHA_W,
                                      PRELU_ALPHA_WIDTH_BROADCAST_S8_ALPHA_C};
    const cmsis_nn_dims output_dims = {PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_N,
                                       PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_H,
                                       PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_W,
                                       PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_C};

    int8_t output[PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_LEN] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s8(&input_dims,
                     prelu_alpha_width_broadcast_s8_input_tensor,
                     &alpha_dims,
                     prelu_alpha_width_broadcast_s8_alpha_input_tensor,
                     PRELU_ALPHA_WIDTH_BROADCAST_S8_INPUT_OFFSET,
                     PRELU_ALPHA_WIDTH_BROADCAST_S8_ALPHA_OFFSET,
                     PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_OFFSET,
                     PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_MULTIPLIER_1,
                     PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_SHIFT_1,
                     PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_MULTIPLIER_2,
                     PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_SHIFT_2,
                     &output_dims,
                     output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(
        validate(output, prelu_alpha_width_broadcast_s8_output, PRELU_ALPHA_WIDTH_BROADCAST_S8_OUTPUT_LEN));
}

void prelu_alpha_height_broadcast_s8_arm_prelu_s8(void)
{
    const cmsis_nn_dims input_dims = {PRELU_ALPHA_HEIGHT_BROADCAST_S8_INPUT_N,
                                      PRELU_ALPHA_HEIGHT_BROADCAST_S8_INPUT_H,
                                      PRELU_ALPHA_HEIGHT_BROADCAST_S8_INPUT_W,
                                      PRELU_ALPHA_HEIGHT_BROADCAST_S8_INPUT_C};
    const cmsis_nn_dims alpha_dims = {PRELU_ALPHA_HEIGHT_BROADCAST_S8_ALPHA_N,
                                      PRELU_ALPHA_HEIGHT_BROADCAST_S8_ALPHA_H,
                                      PRELU_ALPHA_HEIGHT_BROADCAST_S8_ALPHA_W,
                                      PRELU_ALPHA_HEIGHT_BROADCAST_S8_ALPHA_C};
    const cmsis_nn_dims output_dims = {PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_N,
                                       PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_H,
                                       PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_W,
                                       PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_C};

    int8_t output[PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_LEN] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s8(&input_dims,
                     prelu_alpha_height_broadcast_s8_input_tensor,
                     &alpha_dims,
                     prelu_alpha_height_broadcast_s8_alpha_input_tensor,
                     PRELU_ALPHA_HEIGHT_BROADCAST_S8_INPUT_OFFSET,
                     PRELU_ALPHA_HEIGHT_BROADCAST_S8_ALPHA_OFFSET,
                     PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_OFFSET,
                     PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_MULTIPLIER_1,
                     PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_SHIFT_1,
                     PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_MULTIPLIER_2,
                     PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_SHIFT_2,
                     &output_dims,
                     output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(
        validate(output, prelu_alpha_height_broadcast_s8_output, PRELU_ALPHA_HEIGHT_BROADCAST_S8_OUTPUT_LEN));
}

void prelu_alpha_hw_broadcast_s8_arm_prelu_s8(void)
{
    const cmsis_nn_dims input_dims = {PRELU_ALPHA_HW_BROADCAST_S8_INPUT_N,
                                      PRELU_ALPHA_HW_BROADCAST_S8_INPUT_H,
                                      PRELU_ALPHA_HW_BROADCAST_S8_INPUT_W,
                                      PRELU_ALPHA_HW_BROADCAST_S8_INPUT_C};
    const cmsis_nn_dims alpha_dims = {PRELU_ALPHA_HW_BROADCAST_S8_ALPHA_N,
                                      PRELU_ALPHA_HW_BROADCAST_S8_ALPHA_H,
                                      PRELU_ALPHA_HW_BROADCAST_S8_ALPHA_W,
                                      PRELU_ALPHA_HW_BROADCAST_S8_ALPHA_C};
    const cmsis_nn_dims output_dims = {PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_N,
                                       PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_H,
                                       PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_W,
                                       PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_C};

    int8_t output[PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_LEN] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s8(&input_dims,
                     prelu_alpha_hw_broadcast_s8_input_tensor,
                     &alpha_dims,
                     prelu_alpha_hw_broadcast_s8_alpha_input_tensor,
                     PRELU_ALPHA_HW_BROADCAST_S8_INPUT_OFFSET,
                     PRELU_ALPHA_HW_BROADCAST_S8_ALPHA_OFFSET,
                     PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_OFFSET,
                     PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_MULTIPLIER_1,
                     PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_SHIFT_1,
                     PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_MULTIPLIER_2,
                     PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_SHIFT_2,
                     &output_dims,
                     output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, prelu_alpha_hw_broadcast_s8_output, PRELU_ALPHA_HW_BROADCAST_S8_OUTPUT_LEN));
}

/* Regression for the NHWC broadcast walk (issue #336): alpha is a per-batch scalar (2,1,1,1)
 * against an input whose rows are a single pixel with more than one channel (2,2,1,2). Each row
 * of alpha is a single element; the previous walk failed to advance its row pointer and then
 * rewound it off the front of the buffer, reading out of bounds and returning SUCCESS with wrong
 * values. PReLU is elementwise in its input, so only this operand order is representable.
 * The quantization is the identity, so a negative input maps to alpha * input. */
void prelu_broadcast_batch_scalar_alpha_s8_arm_prelu_s8(void)
{
    const int8_t input[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    const int8_t alpha[2] = {2, 3};
    const int8_t expected[8] = {-2, 2, -6, 4, -15, 6, -21, 8};
    int8_t output[8] = {0};
    const cmsis_nn_dims input_dims = {2, 2, 1, 2};
    const cmsis_nn_dims alpha_dims = {2, 1, 1, 1};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_prelu_s8(
            &input_dims, input, &alpha_dims, alpha, 0, 0, 0, 1073741824, 1, 1073741824, 1, &output_dims, output));
    TEST_ASSERT_TRUE(validate(output, expected, 8));
}

/* The walk indexes alpha by its own dims, so an alpha that does not broadcast into the input is
 * now rejected instead of silently producing a partial result. */
void prelu_dims_arg_error_s8_arm_prelu_s8(void)
{
    const int8_t input[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    const int8_t alpha[4] = {1, 2, 3, 4};
    int8_t output[8] = {0};
    const cmsis_nn_dims input_dims = {2, 2, 1, 2};
    const cmsis_nn_dims input_dims_1c = {2, 2, 1, 1};
    const cmsis_nn_dims alpha_dims_3h = {1, 3, 1, 1};
    const cmsis_nn_dims alpha_dims_0h = {2, 0, 1, 1};
    const cmsis_nn_dims alpha_dims_ok = {2, 1, 1, 1};

    /* alpha h = 3 does not broadcast into input h = 2 */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_prelu_s8(
            &input_dims, input, &alpha_dims_3h, alpha, 0, 0, 0, 1073741824, 1, 1073741824, 1, &input_dims, output));
    /* an empty alpha dimension is rejected rather than treated as a no-op */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_s8(&input_dims_1c,
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
        arm_prelu_s8(
            &input_dims, NULL, &alpha_dims_ok, alpha, 0, 0, 0, 1073741824, 1, 1073741824, 1, &input_dims, output));
}

/* PReLU is elementwise in its input, so the output shape must be the input shape. That is a
 * stricter rule than the shared broadcast check: an alpha wider than the input broadcasts to a
 * legal shape, and only this guard rejects it. The s16 suite has the same case. */
void prelu_output_shape_mismatch_s8_arm_prelu_s8(void)
{
    const int8_t input_data[2] = {-2, 4};
    const int8_t alpha_data[3] = {2, 2, 2};
    int8_t output[3] = {0};
    const cmsis_nn_dims input_dims_1c = {1, 1, 1, 1};
    const cmsis_nn_dims input_dims_2c = {1, 1, 1, 2};
    const cmsis_nn_dims alpha_dims_2c = {1, 1, 1, 2};
    const cmsis_nn_dims alpha_dims_3c = {1, 1, 1, 3};
    const cmsis_nn_dims output_dims_3c = {1, 1, 1, 3};

    /* alpha is wider than the input: (1,1,1,1) and (1,1,1,3) broadcast to (1,1,1,3), so the shared
     * check passes, but the output shape is not the input shape */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_s8(&input_dims_1c,
                                   input_data,
                                   &alpha_dims_3c,
                                   alpha_data,
                                   0,
                                   0,
                                   0,
                                   1073741824,
                                   1,
                                   1073741824,
                                   1,
                                   &output_dims_3c,
                                   output));
    /* an output shape that is the broadcast shape of neither operand */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_prelu_s8(&input_dims_2c,
                                   input_data,
                                   &alpha_dims_2c,
                                   alpha_data,
                                   0,
                                   0,
                                   0,
                                   1073741824,
                                   1,
                                   1073741824,
                                   1,
                                   &output_dims_3c,
                                   output));
}
