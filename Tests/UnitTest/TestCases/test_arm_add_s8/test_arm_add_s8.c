/*
 * Copyright (C) 2022 Arm Limited or its affiliates.
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
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
#include "arm_nnsupportfunctions.h"
#include "unity.h"
#include <string.h>

#include "../TestData/add_scalar_s8/test_data.h"
#include "../TestData/add_ident_s8/test_data.h"
#include "../TestData/add_scalar_neg_offset_s8/test_data.h"
#include "../TestData/add_ident_neg_offset_s8/test_data.h"
#include "../TestData/add_broadcast_h_s8/test_data.h"
#include "../TestData/add_broadcast_w_s8/test_data.h"
#include "../TestData/add_broadcast_c_s8/test_data.h"
#include "../TestData/add_broadcast_hc_s8/test_data.h"

#include "../Utils/validate.h"

void add_scalar_s8_arm_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_SCALAR_S8_LHS_N;
    lhs_dims.h = ADD_SCALAR_S8_LHS_H;
    lhs_dims.w = ADD_SCALAR_S8_LHS_W;
    lhs_dims.c = ADD_SCALAR_S8_LHS_C;

    rhs_dims.n = ADD_SCALAR_S8_RHS_N;
    rhs_dims.h = ADD_SCALAR_S8_RHS_H;
    rhs_dims.w = ADD_SCALAR_S8_RHS_W;
    rhs_dims.c = ADD_SCALAR_S8_RHS_C;

    out_dims.n = ADD_SCALAR_S8_OUTPUT_N;
    out_dims.h = ADD_SCALAR_S8_OUTPUT_H;
    out_dims.w = ADD_SCALAR_S8_OUTPUT_W;
    out_dims.c = ADD_SCALAR_S8_OUTPUT_C;

    const int8_t *lhs = add_scalar_s8_lhs_input_tensor;
    const int8_t *rhs = add_scalar_s8_rhs_input_tensor;
    int8_t output[ADD_SCALAR_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_SCALAR_S8_LHS_OFFSET,
        ADD_SCALAR_S8_LHS_MULT,
        ADD_SCALAR_S8_LHS_SHIFT,
        ADD_SCALAR_S8_RHS_OFFSET,
        ADD_SCALAR_S8_RHS_MULT,
        ADD_SCALAR_S8_RHS_SHIFT,
        ADD_SCALAR_S8_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_SCALAR_S8_OUTPUT_OFFSET,
        ADD_SCALAR_S8_OUTPUT_MULT,
        ADD_SCALAR_S8_OUTPUT_SHIFT,
        ADD_SCALAR_S8_ACTIVATION_MIN,
        ADD_SCALAR_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_scalar_s8_output_ref, ADD_SCALAR_S8_DST_SIZE));
}


void add_ident_s8_arm_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_IDENT_S8_LHS_N;
    lhs_dims.h = ADD_IDENT_S8_LHS_H;
    lhs_dims.w = ADD_IDENT_S8_LHS_W;
    lhs_dims.c = ADD_IDENT_S8_LHS_C;

    rhs_dims.n = ADD_IDENT_S8_RHS_N;
    rhs_dims.h = ADD_IDENT_S8_RHS_H;
    rhs_dims.w = ADD_IDENT_S8_RHS_W;
    rhs_dims.c = ADD_IDENT_S8_RHS_C;

    out_dims.n = ADD_IDENT_S8_OUTPUT_N;
    out_dims.h = ADD_IDENT_S8_OUTPUT_H;
    out_dims.w = ADD_IDENT_S8_OUTPUT_W;
    out_dims.c = ADD_IDENT_S8_OUTPUT_C;

    const int8_t *lhs = add_ident_s8_lhs_input_tensor;
    const int8_t *rhs = add_ident_s8_rhs_input_tensor;
    int8_t output[ADD_IDENT_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_IDENT_S8_LHS_OFFSET,
        ADD_IDENT_S8_LHS_MULT,
        ADD_IDENT_S8_LHS_SHIFT,
        ADD_IDENT_S8_RHS_OFFSET,
        ADD_IDENT_S8_RHS_MULT,
        ADD_IDENT_S8_RHS_SHIFT,
        ADD_IDENT_S8_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_IDENT_S8_OUTPUT_OFFSET,
        ADD_IDENT_S8_OUTPUT_MULT,
        ADD_IDENT_S8_OUTPUT_SHIFT,
        ADD_IDENT_S8_ACTIVATION_MIN,
        ADD_IDENT_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_ident_s8_output_ref, ADD_IDENT_S8_DST_SIZE));
}

void add_scalar_neg_offset_s8_arm_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_SCALAR_NEG_OFFSET_S8_LHS_N;
    lhs_dims.h = ADD_SCALAR_NEG_OFFSET_S8_LHS_H;
    lhs_dims.w = ADD_SCALAR_NEG_OFFSET_S8_LHS_W;
    lhs_dims.c = ADD_SCALAR_NEG_OFFSET_S8_LHS_C;

    rhs_dims.n = ADD_SCALAR_NEG_OFFSET_S8_RHS_N;
    rhs_dims.h = ADD_SCALAR_NEG_OFFSET_S8_RHS_H;
    rhs_dims.w = ADD_SCALAR_NEG_OFFSET_S8_RHS_W;
    rhs_dims.c = ADD_SCALAR_NEG_OFFSET_S8_RHS_C;

    out_dims.n = ADD_SCALAR_NEG_OFFSET_S8_OUTPUT_N;
    out_dims.h = ADD_SCALAR_NEG_OFFSET_S8_OUTPUT_H;
    out_dims.w = ADD_SCALAR_NEG_OFFSET_S8_OUTPUT_W;
    out_dims.c = ADD_SCALAR_NEG_OFFSET_S8_OUTPUT_C;

    const int8_t *lhs = add_scalar_neg_offset_s8_lhs_input_tensor;
    const int8_t *rhs = add_scalar_neg_offset_s8_rhs_input_tensor;
    int8_t output[ADD_SCALAR_NEG_OFFSET_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_SCALAR_NEG_OFFSET_S8_LHS_OFFSET,
        ADD_SCALAR_NEG_OFFSET_S8_LHS_MULT,
        ADD_SCALAR_NEG_OFFSET_S8_LHS_SHIFT,
        ADD_SCALAR_NEG_OFFSET_S8_RHS_OFFSET,
        ADD_SCALAR_NEG_OFFSET_S8_RHS_MULT,
        ADD_SCALAR_NEG_OFFSET_S8_RHS_SHIFT,
        ADD_SCALAR_NEG_OFFSET_S8_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_SCALAR_NEG_OFFSET_S8_OUTPUT_OFFSET,
        ADD_SCALAR_NEG_OFFSET_S8_OUTPUT_MULT,
        ADD_SCALAR_NEG_OFFSET_S8_OUTPUT_SHIFT,
        ADD_SCALAR_NEG_OFFSET_S8_ACTIVATION_MIN,
        ADD_SCALAR_NEG_OFFSET_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_scalar_neg_offset_s8_output_ref, ADD_SCALAR_NEG_OFFSET_S8_DST_SIZE));
}


void add_ident_neg_offset_s8_arm_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_IDENT_NEG_OFFSET_S8_LHS_N;
    lhs_dims.h = ADD_IDENT_NEG_OFFSET_S8_LHS_H;
    lhs_dims.w = ADD_IDENT_NEG_OFFSET_S8_LHS_W;
    lhs_dims.c = ADD_IDENT_NEG_OFFSET_S8_LHS_C;

    rhs_dims.n = ADD_IDENT_NEG_OFFSET_S8_RHS_N;
    rhs_dims.h = ADD_IDENT_NEG_OFFSET_S8_RHS_H;
    rhs_dims.w = ADD_IDENT_NEG_OFFSET_S8_RHS_W;
    rhs_dims.c = ADD_IDENT_NEG_OFFSET_S8_RHS_C;

    out_dims.n = ADD_IDENT_NEG_OFFSET_S8_OUTPUT_N;
    out_dims.h = ADD_IDENT_NEG_OFFSET_S8_OUTPUT_H;
    out_dims.w = ADD_IDENT_NEG_OFFSET_S8_OUTPUT_W;
    out_dims.c = ADD_IDENT_NEG_OFFSET_S8_OUTPUT_C;

    const int8_t *lhs = add_ident_neg_offset_s8_lhs_input_tensor;
    const int8_t *rhs = add_ident_neg_offset_s8_rhs_input_tensor;
    int8_t output[ADD_IDENT_NEG_OFFSET_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_IDENT_NEG_OFFSET_S8_LHS_OFFSET,
        ADD_IDENT_NEG_OFFSET_S8_LHS_MULT,
        ADD_IDENT_NEG_OFFSET_S8_LHS_SHIFT,
        ADD_IDENT_NEG_OFFSET_S8_RHS_OFFSET,
        ADD_IDENT_NEG_OFFSET_S8_RHS_MULT,
        ADD_IDENT_NEG_OFFSET_S8_RHS_SHIFT,
        ADD_IDENT_NEG_OFFSET_S8_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_IDENT_NEG_OFFSET_S8_OUTPUT_OFFSET,
        ADD_IDENT_NEG_OFFSET_S8_OUTPUT_MULT,
        ADD_IDENT_NEG_OFFSET_S8_OUTPUT_SHIFT,
        ADD_IDENT_NEG_OFFSET_S8_ACTIVATION_MIN,
        ADD_IDENT_NEG_OFFSET_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_ident_neg_offset_s8_output_ref, ADD_IDENT_NEG_OFFSET_S8_DST_SIZE));
}


void add_broadcast_h_s8_arm_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_BROADCAST_H_S8_LHS_N;
    lhs_dims.h = ADD_BROADCAST_H_S8_LHS_H;
    lhs_dims.w = ADD_BROADCAST_H_S8_LHS_W;
    lhs_dims.c = ADD_BROADCAST_H_S8_LHS_C;

    rhs_dims.n = ADD_BROADCAST_H_S8_RHS_N;
    rhs_dims.h = ADD_BROADCAST_H_S8_RHS_H;
    rhs_dims.w = ADD_BROADCAST_H_S8_RHS_W;
    rhs_dims.c = ADD_BROADCAST_H_S8_RHS_C;

    out_dims.n = ADD_BROADCAST_H_S8_OUTPUT_N;
    out_dims.h = ADD_BROADCAST_H_S8_OUTPUT_H;
    out_dims.w = ADD_BROADCAST_H_S8_OUTPUT_W;
    out_dims.c = ADD_BROADCAST_H_S8_OUTPUT_C;

    const int8_t *lhs = add_broadcast_h_s8_lhs_input_tensor;
    const int8_t *rhs = add_broadcast_h_s8_rhs_input_tensor;
    int8_t output[ADD_BROADCAST_H_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_BROADCAST_H_S8_LHS_OFFSET,
        ADD_BROADCAST_H_S8_LHS_MULT,
        ADD_BROADCAST_H_S8_LHS_SHIFT,
        ADD_BROADCAST_H_S8_RHS_OFFSET,
        ADD_BROADCAST_H_S8_RHS_MULT,
        ADD_BROADCAST_H_S8_RHS_SHIFT,
        ADD_BROADCAST_H_S8_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_BROADCAST_H_S8_OUTPUT_OFFSET,
        ADD_BROADCAST_H_S8_OUTPUT_MULT,
        ADD_BROADCAST_H_S8_OUTPUT_SHIFT,
        ADD_BROADCAST_H_S8_ACTIVATION_MIN,
        ADD_BROADCAST_H_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_broadcast_h_s8_output_ref, ADD_BROADCAST_H_S8_DST_SIZE));
}

void add_broadcast_w_s8_arm_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_BROADCAST_W_S8_LHS_N;
    lhs_dims.h = ADD_BROADCAST_W_S8_LHS_H;
    lhs_dims.w = ADD_BROADCAST_W_S8_LHS_W;
    lhs_dims.c = ADD_BROADCAST_W_S8_LHS_C;

    rhs_dims.n = ADD_BROADCAST_W_S8_RHS_N;
    rhs_dims.h = ADD_BROADCAST_W_S8_RHS_H;
    rhs_dims.w = ADD_BROADCAST_W_S8_RHS_W;
    rhs_dims.c = ADD_BROADCAST_W_S8_RHS_C;

    out_dims.n = ADD_BROADCAST_W_S8_OUTPUT_N;
    out_dims.h = ADD_BROADCAST_W_S8_OUTPUT_H;
    out_dims.w = ADD_BROADCAST_W_S8_OUTPUT_W;
    out_dims.c = ADD_BROADCAST_W_S8_OUTPUT_C;

    const int8_t *lhs = add_broadcast_w_s8_lhs_input_tensor;
    const int8_t *rhs = add_broadcast_w_s8_rhs_input_tensor;
    int8_t output[ADD_BROADCAST_W_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_BROADCAST_W_S8_LHS_OFFSET,
        ADD_BROADCAST_W_S8_LHS_MULT,
        ADD_BROADCAST_W_S8_LHS_SHIFT,
        ADD_BROADCAST_W_S8_RHS_OFFSET,
        ADD_BROADCAST_W_S8_RHS_MULT,
        ADD_BROADCAST_W_S8_RHS_SHIFT,
        ADD_BROADCAST_W_S8_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_BROADCAST_W_S8_OUTPUT_OFFSET,
        ADD_BROADCAST_W_S8_OUTPUT_MULT,
        ADD_BROADCAST_W_S8_OUTPUT_SHIFT,
        ADD_BROADCAST_W_S8_ACTIVATION_MIN,
        ADD_BROADCAST_W_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_broadcast_w_s8_output_ref, ADD_BROADCAST_W_S8_DST_SIZE));
}

void add_broadcast_c_s8_arm_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_BROADCAST_C_S8_LHS_N;
    lhs_dims.h = ADD_BROADCAST_C_S8_LHS_H;
    lhs_dims.w = ADD_BROADCAST_C_S8_LHS_W;
    lhs_dims.c = ADD_BROADCAST_C_S8_LHS_C;

    rhs_dims.n = ADD_BROADCAST_C_S8_RHS_N;
    rhs_dims.h = ADD_BROADCAST_C_S8_RHS_H;
    rhs_dims.w = ADD_BROADCAST_C_S8_RHS_W;
    rhs_dims.c = ADD_BROADCAST_C_S8_RHS_C;

    out_dims.n = ADD_BROADCAST_C_S8_OUTPUT_N;
    out_dims.h = ADD_BROADCAST_C_S8_OUTPUT_H;
    out_dims.w = ADD_BROADCAST_C_S8_OUTPUT_W;
    out_dims.c = ADD_BROADCAST_C_S8_OUTPUT_C;

    const int8_t *lhs = add_broadcast_c_s8_lhs_input_tensor;
    const int8_t *rhs = add_broadcast_c_s8_rhs_input_tensor;
    int8_t output[ADD_BROADCAST_C_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_BROADCAST_C_S8_LHS_OFFSET,
        ADD_BROADCAST_C_S8_LHS_MULT,
        ADD_BROADCAST_C_S8_LHS_SHIFT,
        ADD_BROADCAST_C_S8_RHS_OFFSET,
        ADD_BROADCAST_C_S8_RHS_MULT,
        ADD_BROADCAST_C_S8_RHS_SHIFT,
        ADD_BROADCAST_C_S8_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_BROADCAST_C_S8_OUTPUT_OFFSET,
        ADD_BROADCAST_C_S8_OUTPUT_MULT,
        ADD_BROADCAST_C_S8_OUTPUT_SHIFT,
        ADD_BROADCAST_C_S8_ACTIVATION_MIN,
        ADD_BROADCAST_C_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_broadcast_c_s8_output_ref, ADD_BROADCAST_C_S8_DST_SIZE));
}

void add_broadcast_hc_s8_arm_add_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = ADD_BROADCAST_HC_S8_LHS_N;
    lhs_dims.h = ADD_BROADCAST_HC_S8_LHS_H;
    lhs_dims.w = ADD_BROADCAST_HC_S8_LHS_W;
    lhs_dims.c = ADD_BROADCAST_HC_S8_LHS_C;

    rhs_dims.n = ADD_BROADCAST_HC_S8_RHS_N;
    rhs_dims.h = ADD_BROADCAST_HC_S8_RHS_H;
    rhs_dims.w = ADD_BROADCAST_HC_S8_RHS_W;
    rhs_dims.c = ADD_BROADCAST_HC_S8_RHS_C;

    out_dims.n = ADD_BROADCAST_HC_S8_OUTPUT_N;
    out_dims.h = ADD_BROADCAST_HC_S8_OUTPUT_H;
    out_dims.w = ADD_BROADCAST_HC_S8_OUTPUT_W;
    out_dims.c = ADD_BROADCAST_HC_S8_OUTPUT_C;

    const int8_t *lhs = add_broadcast_hc_s8_lhs_input_tensor;
    const int8_t *rhs = add_broadcast_hc_s8_rhs_input_tensor;
    int8_t output[ADD_BROADCAST_HC_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_add_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        ADD_BROADCAST_HC_S8_LHS_OFFSET,
        ADD_BROADCAST_HC_S8_LHS_MULT,
        ADD_BROADCAST_HC_S8_LHS_SHIFT,
        ADD_BROADCAST_HC_S8_RHS_OFFSET,
        ADD_BROADCAST_HC_S8_RHS_MULT,
        ADD_BROADCAST_HC_S8_RHS_SHIFT,
        ADD_BROADCAST_HC_S8_LEFT_SHIFT,
        output,
        &out_dims,
        ADD_BROADCAST_HC_S8_OUTPUT_OFFSET,
        ADD_BROADCAST_HC_S8_OUTPUT_MULT,
        ADD_BROADCAST_HC_S8_OUTPUT_SHIFT,
        ADD_BROADCAST_HC_S8_ACTIVATION_MIN,
        ADD_BROADCAST_HC_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, add_broadcast_hc_s8_output_ref, ADD_BROADCAST_HC_S8_DST_SIZE));
}

/* Regression for the NHWC broadcast walk (issue #336): input 1 is a per-batch scalar (2,1,1,1)
 * and input 2 broadcasts along the batch with h > 1 (1,2,1,2). Each row of the per-batch scalar
 * is a single element; the previous walk failed to advance its row pointer and then rewound it
 * off the front of the buffer, reading out of bounds and returning SUCCESS with wrong values.
 * Checked in both operand orders. The quantization is the identity, so the expected output is
 * the plain elementwise result. */
void add_broadcast_batch_scalar_s8_arm_add_s8(void)
{
    const int8_t input_1[2] = {10, 20};
    const int8_t input_2[4] = {1, 2, 3, 4};
    const int8_t expected_1_2[8] = {11, 12, 13, 14, 21, 22, 23, 24};
    const int8_t expected_2_1[8] = {11, 12, 13, 14, 21, 22, 23, 24};
    int8_t output[8] = {0};
    const cmsis_nn_dims input_1_dims = {2, 1, 1, 1};
    const cmsis_nn_dims input_2_dims = {1, 2, 1, 2};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_add_s8(input_1,
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
                                 -128,
                                 127));
    TEST_ASSERT_TRUE(validate(output, expected_1_2, 8));

    int8_t output_reversed[8] = {0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_add_s8(input_2,
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
                                 -128,
                                 127));
    TEST_ASSERT_TRUE(validate(output_reversed, expected_2_1, 8));
}

/* The walk indexes each operand by its own dims, so shapes that do not broadcast to the output
 * shape are now rejected instead of silently producing a partial result. */
void add_dims_arg_error_s8_arm_add_s8(void)
{
    const int8_t input_a[4] = {1, 2, 3, 4};
    const int8_t input_b[4] = {5, 6, 7, 8};
    int8_t output[8] = {0};
    const cmsis_nn_dims dims_2n = {2, 1, 1, 1};
    const cmsis_nn_dims dims_3n = {3, 1, 1, 1};
    const cmsis_nn_dims dims_1h2c = {1, 2, 1, 2};
    const cmsis_nn_dims dims_out = {2, 2, 1, 2};
    const cmsis_nn_dims dims_0h = {2, 0, 1, 1};

    /* n = 2 against n = 3 does not broadcast */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_add_s8(input_a,
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
                                 -128,
                                 127));
    /* the output shape must be the broadcast shape of the two inputs */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_add_s8(input_a,
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
                                 -128,
                                 127));
    /* a null operand is rejected */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_add_s8(NULL,
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
                                 -128,
                                 127));
    /* a non-positive dimension is rejected */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_add_s8(input_a,
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
                                 -128,
                                 127));
}

typedef struct
{
    int32_t off_1, mult_1, shift_1, off_2, mult_2, shift_2, left_shift, out_offset, out_mult, out_shift, act_min,
        act_max;
} add_q_params;

static int8_t add_reference(int32_t x1, int32_t x2, const add_q_params *q)
{
    const int32_t a = arm_nn_requantize((x1 + q->off_1) * (1 << q->left_shift), q->mult_1, q->shift_1);
    const int32_t b = arm_nn_requantize((x2 + q->off_2) * (1 << q->left_shift), q->mult_2, q->shift_2);
    int32_t r = arm_nn_requantize(a + b, q->out_mult, q->out_shift) + q->out_offset;
    r = r < q->act_min ? q->act_min : (r > q->act_max ? q->act_max : r);
    return (int8_t)r;
}

/* Parameter sets: both input shifts 0, each input shift non-zero, and an out_shift outside the specialized range. */
static const add_q_params add_q_sets[] = {
    {128, 1073741824, 0, 128, 1073741824, 0, 20, -128, 1073741824, -18, -128, 127},
    {17, 1073741824, 0, -5, 1395864371, -1, 20, 3, 1518500250, -19, -128, 127},
    {-9, 1395864371, -2, 30, 1073741824, 0, 20, -7, 1518500250, -19, -100, 110},
    {5, 1518500250, -1, -12, 1395864371, -3, 20, 11, 1073741824, -20, -128, 127},
    {1, 1073741824, 0, 1, 1073741824, 0, 20, 0, 1073741824, 0, -128, 127},
};

/* One operand broadcast along W with matching C, including swapped operands, N/H broadcast and C not a multiple of
   4, compared against a scalar reference, with guard bytes after the output. */
#define ADD_ROWB_MAX (2 * 3 * 240 * 8)
static int8_t add_rowb_full[ADD_ROWB_MAX], add_rowb_row[ADD_ROWB_MAX], add_rowb_out[ADD_ROWB_MAX + 16];

static int32_t add_rowb_index(const cmsis_nn_dims *d, int32_t n, int32_t h, int32_t w, int32_t c)
{
    return (((d->n == 1 ? 0 : n) * d->h + (d->h == 1 ? 0 : h)) * d->w + (d->w == 1 ? 0 : w)) * d->c + c;
}

static void add_row_broadcast_case(
    const cmsis_nn_dims full, const cmsis_nn_dims row, const cmsis_nn_dims out, int32_t swap, const add_q_params *q)
{
    const int32_t size = out.n * out.h * out.w * out.c;
    uint32_t seed = (uint32_t)(full.w * 131 + full.c * 7 + swap);
    for (int32_t i = 0; i < size; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        add_rowb_full[i] = (int8_t)(seed >> 24);
        add_rowb_row[i] = (int8_t)(seed >> 16);
    }
    memset(add_rowb_out, 0x5A, sizeof(add_rowb_out));
    const arm_cmsis_nn_status status = swap ? arm_add_s8(add_rowb_row,
                                                         &row,
                                                         add_rowb_full,
                                                         &full,
                                                         q->off_2,
                                                         q->mult_2,
                                                         q->shift_2,
                                                         q->off_1,
                                                         q->mult_1,
                                                         q->shift_1,
                                                         q->left_shift,
                                                         add_rowb_out,
                                                         &out,
                                                         q->out_offset,
                                                         q->out_mult,
                                                         q->out_shift,
                                                         q->act_min,
                                                         q->act_max)
                                            : arm_add_s8(add_rowb_full,
                                                         &full,
                                                         add_rowb_row,
                                                         &row,
                                                         q->off_1,
                                                         q->mult_1,
                                                         q->shift_1,
                                                         q->off_2,
                                                         q->mult_2,
                                                         q->shift_2,
                                                         q->left_shift,
                                                         add_rowb_out,
                                                         &out,
                                                         q->out_offset,
                                                         q->out_mult,
                                                         q->out_shift,
                                                         q->act_min,
                                                         q->act_max);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    for (int32_t n = 0; n < out.n; n++)
        for (int32_t h = 0; h < out.h; h++)
            for (int32_t w = 0; w < out.w; w++)
                for (int32_t c = 0; c < out.c; c++)
                {
                    TEST_ASSERT_EQUAL_INT8(add_reference(add_rowb_full[add_rowb_index(&full, n, h, w, c)],
                                                         add_rowb_row[add_rowb_index(&row, n, h, w, c)],
                                                         q),
                                           add_rowb_out[add_rowb_index(&out, n, h, w, c)]);
                }
    for (int32_t i = 0; i < 16; i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, add_rowb_out[size + i]);
    }
}

void add_row_broadcast_s8_arm_add_s8(void)
{
    /* {W operand, W == 1 operand, output} */
    const cmsis_nn_dims shapes[][3] = {
        {{1, 1, 240, 8}, {1, 1, 1, 8}, {1, 1, 240, 8}},
        {{1, 1, 17, 3}, {1, 1, 1, 3}, {1, 1, 17, 3}},
        {{2, 3, 5, 12}, {1, 1, 1, 12}, {2, 3, 5, 12}},
        {{2, 3, 5, 17}, {2, 3, 1, 17}, {2, 3, 5, 17}},
        {{1, 1, 5, 12}, {2, 3, 1, 12}, {2, 3, 5, 12}},
        {{1, 2, 9, 40}, {1, 2, 1, 40}, {1, 2, 9, 40}},
        {{1, 1, 2, 2}, {1, 1, 1, 2}, {1, 1, 2, 2}},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
    {
        for (int32_t swap = 0; swap < 2; swap++)
        {
            for (size_t k = 0; k < sizeof(add_q_sets) / sizeof(add_q_sets[0]); k++)
            {
                add_row_broadcast_case(shapes[i][0], shapes[i][1], shapes[i][2], swap, &add_q_sets[k]);
            }
        }
    }
}
