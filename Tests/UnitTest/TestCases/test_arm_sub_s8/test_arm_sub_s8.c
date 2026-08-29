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

#include "../TestData/sub_scalar_s8/test_data.h"
#include "../TestData/sub_ident_s8/test_data.h"
#include "../TestData/sub_broadcast_h_s8/test_data.h"
#include "../TestData/sub_broadcast_w_s8/test_data.h"
#include "../TestData/sub_broadcast_c_s8/test_data.h"
#include "../TestData/sub_broadcast_hc_s8/test_data.h"

#include "../Utils/validate.h"

void sub_scalar_s8_arm_sub_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_SCALAR_S8_LHS_N;
    lhs_dims.h = SUB_SCALAR_S8_LHS_H;
    lhs_dims.w = SUB_SCALAR_S8_LHS_W;
    lhs_dims.c = SUB_SCALAR_S8_LHS_C;

    rhs_dims.n = SUB_SCALAR_S8_RHS_N;
    rhs_dims.h = SUB_SCALAR_S8_RHS_H;
    rhs_dims.w = SUB_SCALAR_S8_RHS_W;
    rhs_dims.c = SUB_SCALAR_S8_RHS_C;

    out_dims.n = SUB_SCALAR_S8_OUTPUT_N;
    out_dims.h = SUB_SCALAR_S8_OUTPUT_H;
    out_dims.w = SUB_SCALAR_S8_OUTPUT_W;
    out_dims.c = SUB_SCALAR_S8_OUTPUT_C;

    const int8_t *lhs = sub_scalar_s8_lhs_input_tensor;
    const int8_t *rhs = sub_scalar_s8_rhs_input_tensor;
    int8_t output[SUB_SCALAR_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_SCALAR_S8_LHS_OFFSET,
        SUB_SCALAR_S8_LHS_MULT,
        SUB_SCALAR_S8_LHS_SHIFT,
        SUB_SCALAR_S8_RHS_OFFSET,
        SUB_SCALAR_S8_RHS_MULT,
        SUB_SCALAR_S8_RHS_SHIFT,
        SUB_SCALAR_S8_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_SCALAR_S8_OUTPUT_OFFSET,
        SUB_SCALAR_S8_OUTPUT_MULT,
        SUB_SCALAR_S8_OUTPUT_SHIFT,
        SUB_SCALAR_S8_ACTIVATION_MIN,
        SUB_SCALAR_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, sub_scalar_s8_output_ref, SUB_SCALAR_S8_DST_SIZE));
}


void sub_ident_s8_arm_sub_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_IDENT_S8_LHS_N;
    lhs_dims.h = SUB_IDENT_S8_LHS_H;
    lhs_dims.w = SUB_IDENT_S8_LHS_W;
    lhs_dims.c = SUB_IDENT_S8_LHS_C;

    rhs_dims.n = SUB_IDENT_S8_RHS_N;
    rhs_dims.h = SUB_IDENT_S8_RHS_H;
    rhs_dims.w = SUB_IDENT_S8_RHS_W;
    rhs_dims.c = SUB_IDENT_S8_RHS_C;

    out_dims.n = SUB_IDENT_S8_OUTPUT_N;
    out_dims.h = SUB_IDENT_S8_OUTPUT_H;
    out_dims.w = SUB_IDENT_S8_OUTPUT_W;
    out_dims.c = SUB_IDENT_S8_OUTPUT_C;

    const int8_t *lhs = sub_ident_s8_lhs_input_tensor;
    const int8_t *rhs = sub_ident_s8_rhs_input_tensor;
    int8_t output[SUB_IDENT_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_IDENT_S8_LHS_OFFSET,
        SUB_IDENT_S8_LHS_MULT,
        SUB_IDENT_S8_LHS_SHIFT,
        SUB_IDENT_S8_RHS_OFFSET,
        SUB_IDENT_S8_RHS_MULT,
        SUB_IDENT_S8_RHS_SHIFT,
        SUB_IDENT_S8_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_IDENT_S8_OUTPUT_OFFSET,
        SUB_IDENT_S8_OUTPUT_MULT,
        SUB_IDENT_S8_OUTPUT_SHIFT,
        SUB_IDENT_S8_ACTIVATION_MIN,
        SUB_IDENT_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, sub_ident_s8_output_ref, SUB_IDENT_S8_DST_SIZE));
}


void sub_broadcast_h_s8_arm_sub_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_BROADCAST_H_S8_LHS_N;
    lhs_dims.h = SUB_BROADCAST_H_S8_LHS_H;
    lhs_dims.w = SUB_BROADCAST_H_S8_LHS_W;
    lhs_dims.c = SUB_BROADCAST_H_S8_LHS_C;

    rhs_dims.n = SUB_BROADCAST_H_S8_RHS_N;
    rhs_dims.h = SUB_BROADCAST_H_S8_RHS_H;
    rhs_dims.w = SUB_BROADCAST_H_S8_RHS_W;
    rhs_dims.c = SUB_BROADCAST_H_S8_RHS_C;

    out_dims.n = SUB_BROADCAST_H_S8_OUTPUT_N;
    out_dims.h = SUB_BROADCAST_H_S8_OUTPUT_H;
    out_dims.w = SUB_BROADCAST_H_S8_OUTPUT_W;
    out_dims.c = SUB_BROADCAST_H_S8_OUTPUT_C;

    const int8_t *lhs = sub_broadcast_h_s8_lhs_input_tensor;
    const int8_t *rhs = sub_broadcast_h_s8_rhs_input_tensor;
    int8_t output[SUB_BROADCAST_H_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_BROADCAST_H_S8_LHS_OFFSET,
        SUB_BROADCAST_H_S8_LHS_MULT,
        SUB_BROADCAST_H_S8_LHS_SHIFT,
        SUB_BROADCAST_H_S8_RHS_OFFSET,
        SUB_BROADCAST_H_S8_RHS_MULT,
        SUB_BROADCAST_H_S8_RHS_SHIFT,
        SUB_BROADCAST_H_S8_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_BROADCAST_H_S8_OUTPUT_OFFSET,
        SUB_BROADCAST_H_S8_OUTPUT_MULT,
        SUB_BROADCAST_H_S8_OUTPUT_SHIFT,
        SUB_BROADCAST_H_S8_ACTIVATION_MIN,
        SUB_BROADCAST_H_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, sub_broadcast_h_s8_output_ref, SUB_BROADCAST_H_S8_DST_SIZE));
}

void sub_broadcast_w_s8_arm_sub_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_BROADCAST_W_S8_LHS_N;
    lhs_dims.h = SUB_BROADCAST_W_S8_LHS_H;
    lhs_dims.w = SUB_BROADCAST_W_S8_LHS_W;
    lhs_dims.c = SUB_BROADCAST_W_S8_LHS_C;

    rhs_dims.n = SUB_BROADCAST_W_S8_RHS_N;
    rhs_dims.h = SUB_BROADCAST_W_S8_RHS_H;
    rhs_dims.w = SUB_BROADCAST_W_S8_RHS_W;
    rhs_dims.c = SUB_BROADCAST_W_S8_RHS_C;

    out_dims.n = SUB_BROADCAST_W_S8_OUTPUT_N;
    out_dims.h = SUB_BROADCAST_W_S8_OUTPUT_H;
    out_dims.w = SUB_BROADCAST_W_S8_OUTPUT_W;
    out_dims.c = SUB_BROADCAST_W_S8_OUTPUT_C;

    const int8_t *lhs = sub_broadcast_w_s8_lhs_input_tensor;
    const int8_t *rhs = sub_broadcast_w_s8_rhs_input_tensor;
    int8_t output[SUB_BROADCAST_W_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_BROADCAST_W_S8_LHS_OFFSET,
        SUB_BROADCAST_W_S8_LHS_MULT,
        SUB_BROADCAST_W_S8_LHS_SHIFT,
        SUB_BROADCAST_W_S8_RHS_OFFSET,
        SUB_BROADCAST_W_S8_RHS_MULT,
        SUB_BROADCAST_W_S8_RHS_SHIFT,
        SUB_BROADCAST_W_S8_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_BROADCAST_W_S8_OUTPUT_OFFSET,
        SUB_BROADCAST_W_S8_OUTPUT_MULT,
        SUB_BROADCAST_W_S8_OUTPUT_SHIFT,
        SUB_BROADCAST_W_S8_ACTIVATION_MIN,
        SUB_BROADCAST_W_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, sub_broadcast_w_s8_output_ref, SUB_BROADCAST_W_S8_DST_SIZE));
}

void sub_broadcast_c_s8_arm_sub_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_BROADCAST_C_S8_LHS_N;
    lhs_dims.h = SUB_BROADCAST_C_S8_LHS_H;
    lhs_dims.w = SUB_BROADCAST_C_S8_LHS_W;
    lhs_dims.c = SUB_BROADCAST_C_S8_LHS_C;

    rhs_dims.n = SUB_BROADCAST_C_S8_RHS_N;
    rhs_dims.h = SUB_BROADCAST_C_S8_RHS_H;
    rhs_dims.w = SUB_BROADCAST_C_S8_RHS_W;
    rhs_dims.c = SUB_BROADCAST_C_S8_RHS_C;

    out_dims.n = SUB_BROADCAST_C_S8_OUTPUT_N;
    out_dims.h = SUB_BROADCAST_C_S8_OUTPUT_H;
    out_dims.w = SUB_BROADCAST_C_S8_OUTPUT_W;
    out_dims.c = SUB_BROADCAST_C_S8_OUTPUT_C;

    const int8_t *lhs = sub_broadcast_c_s8_lhs_input_tensor;
    const int8_t *rhs = sub_broadcast_c_s8_rhs_input_tensor;
    int8_t output[SUB_BROADCAST_C_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_BROADCAST_C_S8_LHS_OFFSET,
        SUB_BROADCAST_C_S8_LHS_MULT,
        SUB_BROADCAST_C_S8_LHS_SHIFT,
        SUB_BROADCAST_C_S8_RHS_OFFSET,
        SUB_BROADCAST_C_S8_RHS_MULT,
        SUB_BROADCAST_C_S8_RHS_SHIFT,
        SUB_BROADCAST_C_S8_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_BROADCAST_C_S8_OUTPUT_OFFSET,
        SUB_BROADCAST_C_S8_OUTPUT_MULT,
        SUB_BROADCAST_C_S8_OUTPUT_SHIFT,
        SUB_BROADCAST_C_S8_ACTIVATION_MIN,
        SUB_BROADCAST_C_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, sub_broadcast_c_s8_output_ref, SUB_BROADCAST_C_S8_DST_SIZE));
}

void sub_broadcast_hc_s8_arm_sub_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_BROADCAST_HC_S8_LHS_N;
    lhs_dims.h = SUB_BROADCAST_HC_S8_LHS_H;
    lhs_dims.w = SUB_BROADCAST_HC_S8_LHS_W;
    lhs_dims.c = SUB_BROADCAST_HC_S8_LHS_C;

    rhs_dims.n = SUB_BROADCAST_HC_S8_RHS_N;
    rhs_dims.h = SUB_BROADCAST_HC_S8_RHS_H;
    rhs_dims.w = SUB_BROADCAST_HC_S8_RHS_W;
    rhs_dims.c = SUB_BROADCAST_HC_S8_RHS_C;

    out_dims.n = SUB_BROADCAST_HC_S8_OUTPUT_N;
    out_dims.h = SUB_BROADCAST_HC_S8_OUTPUT_H;
    out_dims.w = SUB_BROADCAST_HC_S8_OUTPUT_W;
    out_dims.c = SUB_BROADCAST_HC_S8_OUTPUT_C;

    const int8_t *lhs = sub_broadcast_hc_s8_lhs_input_tensor;
    const int8_t *rhs = sub_broadcast_hc_s8_rhs_input_tensor;
    int8_t output[SUB_BROADCAST_HC_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_BROADCAST_HC_S8_LHS_OFFSET,
        SUB_BROADCAST_HC_S8_LHS_MULT,
        SUB_BROADCAST_HC_S8_LHS_SHIFT,
        SUB_BROADCAST_HC_S8_RHS_OFFSET,
        SUB_BROADCAST_HC_S8_RHS_MULT,
        SUB_BROADCAST_HC_S8_RHS_SHIFT,
        SUB_BROADCAST_HC_S8_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_BROADCAST_HC_S8_OUTPUT_OFFSET,
        SUB_BROADCAST_HC_S8_OUTPUT_MULT,
        SUB_BROADCAST_HC_S8_OUTPUT_SHIFT,
        SUB_BROADCAST_HC_S8_ACTIVATION_MIN,
        SUB_BROADCAST_HC_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, sub_broadcast_hc_s8_output_ref, SUB_BROADCAST_HC_S8_DST_SIZE));
}

/* Regression for the NHWC broadcast walk (issue #336): input 1 is a per-batch scalar (2,1,1,1)
 * and input 2 broadcasts along the batch with h > 1 (1,2,1,2). Each row of the per-batch scalar
 * is a single element; the previous walk failed to advance its row pointer and then rewound it
 * off the front of the buffer, reading out of bounds and returning SUCCESS with wrong values.
 * Checked in both operand orders. The quantization is the identity, so the expected output is
 * the plain elementwise result. */
void sub_broadcast_batch_scalar_s8_arm_sub_s8(void)
{
    const int8_t input_1[2] = {10, 20};
    const int8_t input_2[4] = {1, 2, 3, 4};
    const int8_t expected_1_2[8] = {9, 8, 7, 6, 19, 18, 17, 16};
    const int8_t expected_2_1[8] = {-9, -8, -7, -6, -19, -18, -17, -16};
    int8_t output[8] = {0};
    const cmsis_nn_dims input_1_dims = {2, 1, 1, 1};
    const cmsis_nn_dims input_2_dims = {1, 2, 1, 2};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_sub_s8(input_1,
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
                      arm_sub_s8(input_2,
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
void sub_dims_arg_error_s8_arm_sub_s8(void)
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
                      arm_sub_s8(input_a,
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
                      arm_sub_s8(input_a,
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
                      arm_sub_s8(NULL,
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
                      arm_sub_s8(input_a,
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
