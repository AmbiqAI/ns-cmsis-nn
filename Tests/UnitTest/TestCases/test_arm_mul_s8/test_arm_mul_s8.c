/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/mul_scalar_s8/test_data.h"
#include "../TestData/mul_ident_s8/test_data.h"
#include "../TestData/mul_broadcast_h_s8/test_data.h"
#include "../TestData/mul_broadcast_w_s8/test_data.h"
#include "../TestData/mul_broadcast_c_s8/test_data.h"
#include "../TestData/mul_broadcast_hc_s8/test_data.h"

#include "../Utils/validate.h"

void mul_scalar_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_SCALAR_S8_LHS_N;
    lhs_dims.h = MUL_SCALAR_S8_LHS_H;
    lhs_dims.w = MUL_SCALAR_S8_LHS_W;
    lhs_dims.c = MUL_SCALAR_S8_LHS_C;

    rhs_dims.n = MUL_SCALAR_S8_RHS_N;
    rhs_dims.h = MUL_SCALAR_S8_RHS_H;
    rhs_dims.w = MUL_SCALAR_S8_RHS_W;
    rhs_dims.c = MUL_SCALAR_S8_RHS_C;

    out_dims.n = MUL_SCALAR_S8_OUTPUT_N;
    out_dims.h = MUL_SCALAR_S8_OUTPUT_H;
    out_dims.w = MUL_SCALAR_S8_OUTPUT_W;
    out_dims.c = MUL_SCALAR_S8_OUTPUT_C;

    const int8_t *lhs = mul_scalar_s8_lhs_input_tensor;
    const int8_t *rhs = mul_scalar_s8_rhs_input_tensor;
    int8_t output[MUL_SCALAR_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_SCALAR_S8_LHS_OFFSET,
        MUL_SCALAR_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_SCALAR_S8_OUTPUT_OFFSET,
        MUL_SCALAR_S8_OUTPUT_MULT,
        MUL_SCALAR_S8_OUTPUT_SHIFT,
        MUL_SCALAR_S8_ACTIVATION_MIN,
        MUL_SCALAR_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_scalar_s8_output_ref, MUL_SCALAR_S8_DST_SIZE));
}


void mul_ident_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_IDENT_S8_LHS_N;
    lhs_dims.h = MUL_IDENT_S8_LHS_H;
    lhs_dims.w = MUL_IDENT_S8_LHS_W;
    lhs_dims.c = MUL_IDENT_S8_LHS_C;

    rhs_dims.n = MUL_IDENT_S8_RHS_N;
    rhs_dims.h = MUL_IDENT_S8_RHS_H;
    rhs_dims.w = MUL_IDENT_S8_RHS_W;
    rhs_dims.c = MUL_IDENT_S8_RHS_C;

    out_dims.n = MUL_IDENT_S8_OUTPUT_N;
    out_dims.h = MUL_IDENT_S8_OUTPUT_H;
    out_dims.w = MUL_IDENT_S8_OUTPUT_W;
    out_dims.c = MUL_IDENT_S8_OUTPUT_C;

    const int8_t *lhs = mul_ident_s8_lhs_input_tensor;
    const int8_t *rhs = mul_ident_s8_rhs_input_tensor;
    int8_t output[MUL_IDENT_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_IDENT_S8_LHS_OFFSET,
        MUL_IDENT_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_IDENT_S8_OUTPUT_OFFSET,
        MUL_IDENT_S8_OUTPUT_MULT,
        MUL_IDENT_S8_OUTPUT_SHIFT,
        MUL_IDENT_S8_ACTIVATION_MIN,
        MUL_IDENT_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_ident_s8_output_ref, MUL_IDENT_S8_DST_SIZE));
}


void mul_broadcast_h_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_H_S8_LHS_N;
    lhs_dims.h = MUL_BROADCAST_H_S8_LHS_H;
    lhs_dims.w = MUL_BROADCAST_H_S8_LHS_W;
    lhs_dims.c = MUL_BROADCAST_H_S8_LHS_C;

    rhs_dims.n = MUL_BROADCAST_H_S8_RHS_N;
    rhs_dims.h = MUL_BROADCAST_H_S8_RHS_H;
    rhs_dims.w = MUL_BROADCAST_H_S8_RHS_W;
    rhs_dims.c = MUL_BROADCAST_H_S8_RHS_C;

    out_dims.n = MUL_BROADCAST_H_S8_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_H_S8_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_H_S8_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_H_S8_OUTPUT_C;

    const int8_t *lhs = mul_broadcast_h_s8_lhs_input_tensor;
    const int8_t *rhs = mul_broadcast_h_s8_rhs_input_tensor;
    int8_t output[MUL_BROADCAST_H_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_H_S8_LHS_OFFSET,
        MUL_BROADCAST_H_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_H_S8_OUTPUT_OFFSET,
        MUL_BROADCAST_H_S8_OUTPUT_MULT,
        MUL_BROADCAST_H_S8_OUTPUT_SHIFT,
        MUL_BROADCAST_H_S8_ACTIVATION_MIN,
        MUL_BROADCAST_H_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_broadcast_h_s8_output_ref, MUL_BROADCAST_H_S8_DST_SIZE));
}

void mul_broadcast_w_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_W_S8_LHS_N;
    lhs_dims.h = MUL_BROADCAST_W_S8_LHS_H;
    lhs_dims.w = MUL_BROADCAST_W_S8_LHS_W;
    lhs_dims.c = MUL_BROADCAST_W_S8_LHS_C;

    rhs_dims.n = MUL_BROADCAST_W_S8_RHS_N;
    rhs_dims.h = MUL_BROADCAST_W_S8_RHS_H;
    rhs_dims.w = MUL_BROADCAST_W_S8_RHS_W;
    rhs_dims.c = MUL_BROADCAST_W_S8_RHS_C;

    out_dims.n = MUL_BROADCAST_W_S8_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_W_S8_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_W_S8_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_W_S8_OUTPUT_C;

    const int8_t *lhs = mul_broadcast_w_s8_lhs_input_tensor;
    const int8_t *rhs = mul_broadcast_w_s8_rhs_input_tensor;
    int8_t output[MUL_BROADCAST_W_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_W_S8_LHS_OFFSET,
        MUL_BROADCAST_W_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_W_S8_OUTPUT_OFFSET,
        MUL_BROADCAST_W_S8_OUTPUT_MULT,
        MUL_BROADCAST_W_S8_OUTPUT_SHIFT,
        MUL_BROADCAST_W_S8_ACTIVATION_MIN,
        MUL_BROADCAST_W_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_broadcast_w_s8_output_ref, MUL_BROADCAST_W_S8_DST_SIZE));
}

void mul_broadcast_c_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_C_S8_LHS_N;
    lhs_dims.h = MUL_BROADCAST_C_S8_LHS_H;
    lhs_dims.w = MUL_BROADCAST_C_S8_LHS_W;
    lhs_dims.c = MUL_BROADCAST_C_S8_LHS_C;

    rhs_dims.n = MUL_BROADCAST_C_S8_RHS_N;
    rhs_dims.h = MUL_BROADCAST_C_S8_RHS_H;
    rhs_dims.w = MUL_BROADCAST_C_S8_RHS_W;
    rhs_dims.c = MUL_BROADCAST_C_S8_RHS_C;

    out_dims.n = MUL_BROADCAST_C_S8_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_C_S8_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_C_S8_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_C_S8_OUTPUT_C;

    const int8_t *lhs = mul_broadcast_c_s8_lhs_input_tensor;
    const int8_t *rhs = mul_broadcast_c_s8_rhs_input_tensor;
    int8_t output[MUL_BROADCAST_C_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_C_S8_LHS_OFFSET,
        MUL_BROADCAST_C_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_C_S8_OUTPUT_OFFSET,
        MUL_BROADCAST_C_S8_OUTPUT_MULT,
        MUL_BROADCAST_C_S8_OUTPUT_SHIFT,
        MUL_BROADCAST_C_S8_ACTIVATION_MIN,
        MUL_BROADCAST_C_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_broadcast_c_s8_output_ref, MUL_BROADCAST_C_S8_DST_SIZE));
}

void mul_broadcast_hc_s8_arm_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_HC_S8_LHS_N;
    lhs_dims.h = MUL_BROADCAST_HC_S8_LHS_H;
    lhs_dims.w = MUL_BROADCAST_HC_S8_LHS_W;
    lhs_dims.c = MUL_BROADCAST_HC_S8_LHS_C;

    rhs_dims.n = MUL_BROADCAST_HC_S8_RHS_N;
    rhs_dims.h = MUL_BROADCAST_HC_S8_RHS_H;
    rhs_dims.w = MUL_BROADCAST_HC_S8_RHS_W;
    rhs_dims.c = MUL_BROADCAST_HC_S8_RHS_C;

    out_dims.n = MUL_BROADCAST_HC_S8_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_HC_S8_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_HC_S8_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_HC_S8_OUTPUT_C;

    const int8_t *lhs = mul_broadcast_hc_s8_lhs_input_tensor;
    const int8_t *rhs = mul_broadcast_hc_s8_rhs_input_tensor;
    int8_t output[MUL_BROADCAST_HC_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_HC_S8_LHS_OFFSET,
        MUL_BROADCAST_HC_S8_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_HC_S8_OUTPUT_OFFSET,
        MUL_BROADCAST_HC_S8_OUTPUT_MULT,
        MUL_BROADCAST_HC_S8_OUTPUT_SHIFT,
        MUL_BROADCAST_HC_S8_ACTIVATION_MIN,
        MUL_BROADCAST_HC_S8_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_broadcast_hc_s8_output_ref, MUL_BROADCAST_HC_S8_DST_SIZE));
}

/* Regression for the NHWC broadcast walk (issue #336): see the add suite for the mechanism.
 * Identity output requantization and zero input offsets, so the expected output is the plain
 * elementwise product. */
void mul_broadcast_batch_scalar_s8_arm_mul_s8(void)
{
    const int8_t input_1[2] = {10, 20};
    const int8_t input_2[4] = {1, 2, 3, 4};
    const int8_t expected_1_2[8] = {10, 20, 30, 40, 20, 40, 60, 80};
    const int8_t expected_2_1[8] = {10, 20, 30, 40, 20, 40, 60, 80};
    int8_t output[8] = {0};
    int8_t output_reversed[8] = {0};
    const cmsis_nn_dims input_1_dims = {2, 1, 1, 1};
    const cmsis_nn_dims input_2_dims = {1, 2, 1, 2};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_mul_s8(
            input_1, &input_1_dims, input_2, &input_2_dims, 0, 0, output, &output_dims, 0, 1073741824, 1, -128, 127));
    TEST_ASSERT_TRUE(validate(output, expected_1_2, 8));

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_mul_s8(input_2,
                                 &input_2_dims,
                                 input_1,
                                 &input_1_dims,
                                 0,
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
void mul_dims_arg_error_s8_arm_mul_s8(void)
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
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_mul_s8(input_a, &dims_2n, input_b, &dims_3n, 0, 0, output, &dims_3n, 0, 1073741824, 1, -128, 127));
    /* the output shape must be the broadcast shape of the two inputs */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_mul_s8(input_a, &dims_2n, input_b, &dims_1h2c, 0, 0, output, &dims_2n, 0, 1073741824, 1, -128, 127));
    /* a null operand is rejected */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_mul_s8(NULL, &dims_2n, input_b, &dims_1h2c, 0, 0, output, &dims_out, 0, 1073741824, 1, -128, 127));
    /* a non-positive dimension is rejected */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_mul_s8(input_a, &dims_0h, input_b, &dims_0h, 0, 0, output, &dims_0h, 0, 1073741824, 1, -128, 127));
}
