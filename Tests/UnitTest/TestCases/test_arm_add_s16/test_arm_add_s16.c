/*
 * Copyright (C) 2022 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/add_scalar_s16/test_data.h"
#include "../TestData/add_ident_s16/test_data.h"
#include "../TestData/add_broadcast_h_s16/test_data.h"
#include "../TestData/add_broadcast_w_s16/test_data.h"
#include "../TestData/add_broadcast_c_s16/test_data.h"
#include "../TestData/add_broadcast_hc_s16/test_data.h"

#include "../Utils/validate.h"

void add_scalar_s16_arm_add_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_SCALAR_S16_LHS_N;
    lhs_dims.h = ADD_SCALAR_S16_LHS_H;
    lhs_dims.w = ADD_SCALAR_S16_LHS_W;
    lhs_dims.c = ADD_SCALAR_S16_LHS_C;

    rhs_dims.n = ADD_SCALAR_S16_RHS_N;
    rhs_dims.h = ADD_SCALAR_S16_RHS_H;
    rhs_dims.w = ADD_SCALAR_S16_RHS_W;
    rhs_dims.c = ADD_SCALAR_S16_RHS_C;

    out_dims.n = ADD_SCALAR_S16_OUTPUT_N;
    out_dims.h = ADD_SCALAR_S16_OUTPUT_H;
    out_dims.w = ADD_SCALAR_S16_OUTPUT_W;
    out_dims.c = ADD_SCALAR_S16_OUTPUT_C;

    const int16_t *lhs = add_scalar_s16_lhs_input_tensor;
    const int16_t *rhs = add_scalar_s16_rhs_input_tensor;
    int16_t output[ADD_SCALAR_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_SCALAR_S16_LHS_OFFSET,
        ADD_SCALAR_S16_LHS_MULT,
        ADD_SCALAR_S16_LHS_SHIFT,
        ADD_SCALAR_S16_RHS_OFFSET,
        ADD_SCALAR_S16_RHS_MULT,
        ADD_SCALAR_S16_RHS_SHIFT,
        ADD_SCALAR_S16_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_SCALAR_S16_OUTPUT_OFFSET,
        ADD_SCALAR_S16_OUTPUT_MULT,
        ADD_SCALAR_S16_OUTPUT_SHIFT,
        ADD_SCALAR_S16_ACTIVATION_MIN,
        ADD_SCALAR_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, add_scalar_s16_output_ref, ADD_SCALAR_S16_DST_SIZE));
}


void add_ident_s16_arm_add_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_IDENT_S16_LHS_N;
    lhs_dims.h = ADD_IDENT_S16_LHS_H;
    lhs_dims.w = ADD_IDENT_S16_LHS_W;
    lhs_dims.c = ADD_IDENT_S16_LHS_C;

    rhs_dims.n = ADD_IDENT_S16_RHS_N;
    rhs_dims.h = ADD_IDENT_S16_RHS_H;
    rhs_dims.w = ADD_IDENT_S16_RHS_W;
    rhs_dims.c = ADD_IDENT_S16_RHS_C;

    out_dims.n = ADD_IDENT_S16_OUTPUT_N;
    out_dims.h = ADD_IDENT_S16_OUTPUT_H;
    out_dims.w = ADD_IDENT_S16_OUTPUT_W;
    out_dims.c = ADD_IDENT_S16_OUTPUT_C;

    const int16_t *lhs = add_ident_s16_lhs_input_tensor;
    const int16_t *rhs = add_ident_s16_rhs_input_tensor;
    int16_t output[ADD_IDENT_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_IDENT_S16_LHS_OFFSET,
        ADD_IDENT_S16_LHS_MULT,
        ADD_IDENT_S16_LHS_SHIFT,
        ADD_IDENT_S16_RHS_OFFSET,
        ADD_IDENT_S16_RHS_MULT,
        ADD_IDENT_S16_RHS_SHIFT,
        ADD_IDENT_S16_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_IDENT_S16_OUTPUT_OFFSET,
        ADD_IDENT_S16_OUTPUT_MULT,
        ADD_IDENT_S16_OUTPUT_SHIFT,
        ADD_IDENT_S16_ACTIVATION_MIN,
        ADD_IDENT_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, add_ident_s16_output_ref, ADD_IDENT_S16_DST_SIZE));
}


void add_broadcast_h_s16_arm_add_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_BROADCAST_H_S16_LHS_N;
    lhs_dims.h = ADD_BROADCAST_H_S16_LHS_H;
    lhs_dims.w = ADD_BROADCAST_H_S16_LHS_W;
    lhs_dims.c = ADD_BROADCAST_H_S16_LHS_C;

    rhs_dims.n = ADD_BROADCAST_H_S16_RHS_N;
    rhs_dims.h = ADD_BROADCAST_H_S16_RHS_H;
    rhs_dims.w = ADD_BROADCAST_H_S16_RHS_W;
    rhs_dims.c = ADD_BROADCAST_H_S16_RHS_C;

    out_dims.n = ADD_BROADCAST_H_S16_OUTPUT_N;
    out_dims.h = ADD_BROADCAST_H_S16_OUTPUT_H;
    out_dims.w = ADD_BROADCAST_H_S16_OUTPUT_W;
    out_dims.c = ADD_BROADCAST_H_S16_OUTPUT_C;

    const int16_t *lhs = add_broadcast_h_s16_lhs_input_tensor;
    const int16_t *rhs = add_broadcast_h_s16_rhs_input_tensor;
    int16_t output[ADD_BROADCAST_H_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_BROADCAST_H_S16_LHS_OFFSET,
        ADD_BROADCAST_H_S16_LHS_MULT,
        ADD_BROADCAST_H_S16_LHS_SHIFT,
        ADD_BROADCAST_H_S16_RHS_OFFSET,
        ADD_BROADCAST_H_S16_RHS_MULT,
        ADD_BROADCAST_H_S16_RHS_SHIFT,
        ADD_BROADCAST_H_S16_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_BROADCAST_H_S16_OUTPUT_OFFSET,
        ADD_BROADCAST_H_S16_OUTPUT_MULT,
        ADD_BROADCAST_H_S16_OUTPUT_SHIFT,
        ADD_BROADCAST_H_S16_ACTIVATION_MIN,
        ADD_BROADCAST_H_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, add_broadcast_h_s16_output_ref, ADD_BROADCAST_H_S16_DST_SIZE));
}

void add_broadcast_w_s16_arm_add_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_BROADCAST_W_S16_LHS_N;
    lhs_dims.h = ADD_BROADCAST_W_S16_LHS_H;
    lhs_dims.w = ADD_BROADCAST_W_S16_LHS_W;
    lhs_dims.c = ADD_BROADCAST_W_S16_LHS_C;

    rhs_dims.n = ADD_BROADCAST_W_S16_RHS_N;
    rhs_dims.h = ADD_BROADCAST_W_S16_RHS_H;
    rhs_dims.w = ADD_BROADCAST_W_S16_RHS_W;
    rhs_dims.c = ADD_BROADCAST_W_S16_RHS_C;

    out_dims.n = ADD_BROADCAST_W_S16_OUTPUT_N;
    out_dims.h = ADD_BROADCAST_W_S16_OUTPUT_H;
    out_dims.w = ADD_BROADCAST_W_S16_OUTPUT_W;
    out_dims.c = ADD_BROADCAST_W_S16_OUTPUT_C;

    const int16_t *lhs = add_broadcast_w_s16_lhs_input_tensor;
    const int16_t *rhs = add_broadcast_w_s16_rhs_input_tensor;
    int16_t output[ADD_BROADCAST_W_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_BROADCAST_W_S16_LHS_OFFSET,
        ADD_BROADCAST_W_S16_LHS_MULT,
        ADD_BROADCAST_W_S16_LHS_SHIFT,
        ADD_BROADCAST_W_S16_RHS_OFFSET,
        ADD_BROADCAST_W_S16_RHS_MULT,
        ADD_BROADCAST_W_S16_RHS_SHIFT,
        ADD_BROADCAST_W_S16_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_BROADCAST_W_S16_OUTPUT_OFFSET,
        ADD_BROADCAST_W_S16_OUTPUT_MULT,
        ADD_BROADCAST_W_S16_OUTPUT_SHIFT,
        ADD_BROADCAST_W_S16_ACTIVATION_MIN,
        ADD_BROADCAST_W_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, add_broadcast_w_s16_output_ref, ADD_BROADCAST_W_S16_DST_SIZE));
}

void add_broadcast_c_s16_arm_add_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_BROADCAST_C_S16_LHS_N;
    lhs_dims.h = ADD_BROADCAST_C_S16_LHS_H;
    lhs_dims.w = ADD_BROADCAST_C_S16_LHS_W;
    lhs_dims.c = ADD_BROADCAST_C_S16_LHS_C;

    rhs_dims.n = ADD_BROADCAST_C_S16_RHS_N;
    rhs_dims.h = ADD_BROADCAST_C_S16_RHS_H;
    rhs_dims.w = ADD_BROADCAST_C_S16_RHS_W;
    rhs_dims.c = ADD_BROADCAST_C_S16_RHS_C;

    out_dims.n = ADD_BROADCAST_C_S16_OUTPUT_N;
    out_dims.h = ADD_BROADCAST_C_S16_OUTPUT_H;
    out_dims.w = ADD_BROADCAST_C_S16_OUTPUT_W;
    out_dims.c = ADD_BROADCAST_C_S16_OUTPUT_C;

    const int16_t *lhs = add_broadcast_c_s16_lhs_input_tensor;
    const int16_t *rhs = add_broadcast_c_s16_rhs_input_tensor;
    int16_t output[ADD_BROADCAST_C_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_BROADCAST_C_S16_LHS_OFFSET,
        ADD_BROADCAST_C_S16_LHS_MULT,
        ADD_BROADCAST_C_S16_LHS_SHIFT,
        ADD_BROADCAST_C_S16_RHS_OFFSET,
        ADD_BROADCAST_C_S16_RHS_MULT,
        ADD_BROADCAST_C_S16_RHS_SHIFT,
        ADD_BROADCAST_C_S16_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_BROADCAST_C_S16_OUTPUT_OFFSET,
        ADD_BROADCAST_C_S16_OUTPUT_MULT,
        ADD_BROADCAST_C_S16_OUTPUT_SHIFT,
        ADD_BROADCAST_C_S16_ACTIVATION_MIN,
        ADD_BROADCAST_C_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, add_broadcast_c_s16_output_ref, ADD_BROADCAST_C_S16_DST_SIZE));
}

void add_broadcast_hc_s16_arm_add_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_BROADCAST_HC_S16_LHS_N;
    lhs_dims.h = ADD_BROADCAST_HC_S16_LHS_H;
    lhs_dims.w = ADD_BROADCAST_HC_S16_LHS_W;
    lhs_dims.c = ADD_BROADCAST_HC_S16_LHS_C;

    rhs_dims.n = ADD_BROADCAST_HC_S16_RHS_N;
    rhs_dims.h = ADD_BROADCAST_HC_S16_RHS_H;
    rhs_dims.w = ADD_BROADCAST_HC_S16_RHS_W;
    rhs_dims.c = ADD_BROADCAST_HC_S16_RHS_C;

    out_dims.n = ADD_BROADCAST_HC_S16_OUTPUT_N;
    out_dims.h = ADD_BROADCAST_HC_S16_OUTPUT_H;
    out_dims.w = ADD_BROADCAST_HC_S16_OUTPUT_W;
    out_dims.c = ADD_BROADCAST_HC_S16_OUTPUT_C;

    const int16_t *lhs = add_broadcast_hc_s16_lhs_input_tensor;
    const int16_t *rhs = add_broadcast_hc_s16_rhs_input_tensor;
    int16_t output[ADD_BROADCAST_HC_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_BROADCAST_HC_S16_LHS_OFFSET,
        ADD_BROADCAST_HC_S16_LHS_MULT,
        ADD_BROADCAST_HC_S16_LHS_SHIFT,
        ADD_BROADCAST_HC_S16_RHS_OFFSET,
        ADD_BROADCAST_HC_S16_RHS_MULT,
        ADD_BROADCAST_HC_S16_RHS_SHIFT,
        ADD_BROADCAST_HC_S16_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_BROADCAST_HC_S16_OUTPUT_OFFSET,
        ADD_BROADCAST_HC_S16_OUTPUT_MULT,
        ADD_BROADCAST_HC_S16_OUTPUT_SHIFT,
        ADD_BROADCAST_HC_S16_ACTIVATION_MIN,
        ADD_BROADCAST_HC_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, add_broadcast_hc_s16_output_ref, ADD_BROADCAST_HC_S16_DST_SIZE));
}

/* Regression for the NHWC broadcast walk (issue #336): input 1 is a per-batch scalar (2,1,1,1)
 * and input 2 broadcasts along the batch with h > 1 (1,2,1,2). Each row of the per-batch scalar
 * is a single element; the previous walk failed to advance its row pointer and then rewound it
 * off the front of the buffer, reading out of bounds and returning SUCCESS with wrong values.
 * Checked in both operand orders. The quantization is the identity, so the expected output is
 * the plain elementwise result. */
void add_broadcast_batch_scalar_s16_arm_add_s16(void)
{
    const int16_t input_1[2] = {10, 20};
    const int16_t input_2[4] = {1, 2, 3, 4};
    const int16_t expected_1_2[8] = {11, 12, 13, 14, 21, 22, 23, 24};
    const int16_t expected_2_1[8] = {11, 12, 13, 14, 21, 22, 23, 24};
    int16_t output[8] = {0};
    const cmsis_nn_dims input_1_dims = {2, 1, 1, 1};
    const cmsis_nn_dims input_2_dims = {1, 2, 1, 2};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_add_s16(input_1,
                                  &input_1_dims,
                                  input_2,
                                  &input_2_dims,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  output,
                                  &output_dims,
                                  0,
                                  1073741824,
                                  1,
                                  -32768,
                                  32767));
    TEST_ASSERT_TRUE(validate_s16(output, expected_1_2, 8));

    int16_t output_reversed[8] = {0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_add_s16(input_2,
                                  &input_2_dims,
                                  input_1,
                                  &input_1_dims,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  output_reversed,
                                  &output_dims,
                                  0,
                                  1073741824,
                                  1,
                                  -32768,
                                  32767));
    TEST_ASSERT_TRUE(validate_s16(output_reversed, expected_2_1, 8));
}

/* The walk indexes each operand by its own dims, so shapes that do not broadcast to the output
 * shape are now rejected instead of silently producing a partial result. */
void add_dims_arg_error_s16_arm_add_s16(void)
{
    const int16_t input_a[4] = {1, 2, 3, 4};
    const int16_t input_b[4] = {5, 6, 7, 8};
    int16_t output[8] = {0};
    const cmsis_nn_dims dims_2n = {2, 1, 1, 1};
    const cmsis_nn_dims dims_3n = {3, 1, 1, 1};
    const cmsis_nn_dims dims_1h2c = {1, 2, 1, 2};
    const cmsis_nn_dims dims_out = {2, 2, 1, 2};
    const cmsis_nn_dims dims_0h = {2, 0, 1, 1};

    /* n = 2 against n = 3 does not broadcast */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_add_s16(input_a,
                                  &dims_2n,
                                  input_b,
                                  &dims_3n,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  output,
                                  &dims_3n,
                                  0,
                                  1073741824,
                                  1,
                                  -32768,
                                  32767));
    /* the output shape must be the broadcast shape of the two inputs */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_add_s16(input_a,
                                  &dims_2n,
                                  input_b,
                                  &dims_1h2c,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  output,
                                  &dims_2n,
                                  0,
                                  1073741824,
                                  1,
                                  -32768,
                                  32767));
    /* a null operand is rejected */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_add_s16(NULL,
                                  &dims_2n,
                                  input_b,
                                  &dims_1h2c,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  output,
                                  &dims_out,
                                  0,
                                  1073741824,
                                  1,
                                  -32768,
                                  32767));
    /* a non-positive dimension is rejected */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_add_s16(input_a,
                                  &dims_0h,
                                  input_b,
                                  &dims_0h,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  1073741824,
                                  1,
                                  0,
                                  output,
                                  &dims_0h,
                                  0,
                                  1073741824,
                                  1,
                                  -32768,
                                  32767));
}
