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

#include "../TestData/mul_scalar_s16/test_data.h"
#include "../TestData/mul_ident_s16/test_data.h"
#include "../TestData/mul_broadcast_h_s16/test_data.h"
#include "../TestData/mul_broadcast_w_s16/test_data.h"
#include "../TestData/mul_broadcast_c_s16/test_data.h"
#include "../TestData/mul_broadcast_hc_s16/test_data.h"

#include "../Utils/validate.h"

void mul_scalar_s16_arm_mul_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_SCALAR_S16_LHS_N;
    lhs_dims.h = MUL_SCALAR_S16_LHS_H;
    lhs_dims.w = MUL_SCALAR_S16_LHS_W;
    lhs_dims.c = MUL_SCALAR_S16_LHS_C;

    rhs_dims.n = MUL_SCALAR_S16_RHS_N;
    rhs_dims.h = MUL_SCALAR_S16_RHS_H;
    rhs_dims.w = MUL_SCALAR_S16_RHS_W;
    rhs_dims.c = MUL_SCALAR_S16_RHS_C;

    out_dims.n = MUL_SCALAR_S16_OUTPUT_N;
    out_dims.h = MUL_SCALAR_S16_OUTPUT_H;
    out_dims.w = MUL_SCALAR_S16_OUTPUT_W;
    out_dims.c = MUL_SCALAR_S16_OUTPUT_C;

    const int16_t *lhs = mul_scalar_s16_lhs_input_tensor;
    const int16_t *rhs = mul_scalar_s16_rhs_input_tensor;
    int16_t output[MUL_SCALAR_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_SCALAR_S16_LHS_OFFSET,
        MUL_SCALAR_S16_RHS_OFFSET,
        output,
        &out_dims,
        MUL_SCALAR_S16_OUTPUT_OFFSET,
        MUL_SCALAR_S16_OUTPUT_MULT,
        MUL_SCALAR_S16_OUTPUT_SHIFT,
        MUL_SCALAR_S16_ACTIVATION_MIN,
        MUL_SCALAR_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, mul_scalar_s16_output_ref, MUL_SCALAR_S16_DST_SIZE));
}


void mul_ident_s16_arm_mul_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_IDENT_S16_LHS_N;
    lhs_dims.h = MUL_IDENT_S16_LHS_H;
    lhs_dims.w = MUL_IDENT_S16_LHS_W;
    lhs_dims.c = MUL_IDENT_S16_LHS_C;

    rhs_dims.n = MUL_IDENT_S16_RHS_N;
    rhs_dims.h = MUL_IDENT_S16_RHS_H;
    rhs_dims.w = MUL_IDENT_S16_RHS_W;
    rhs_dims.c = MUL_IDENT_S16_RHS_C;

    out_dims.n = MUL_IDENT_S16_OUTPUT_N;
    out_dims.h = MUL_IDENT_S16_OUTPUT_H;
    out_dims.w = MUL_IDENT_S16_OUTPUT_W;
    out_dims.c = MUL_IDENT_S16_OUTPUT_C;

    const int16_t *lhs = mul_ident_s16_lhs_input_tensor;
    const int16_t *rhs = mul_ident_s16_rhs_input_tensor;
    int16_t output[MUL_IDENT_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_IDENT_S16_LHS_OFFSET,
        MUL_IDENT_S16_RHS_OFFSET,
        output,
        &out_dims,
        MUL_IDENT_S16_OUTPUT_OFFSET,
        MUL_IDENT_S16_OUTPUT_MULT,
        MUL_IDENT_S16_OUTPUT_SHIFT,
        MUL_IDENT_S16_ACTIVATION_MIN,
        MUL_IDENT_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, mul_ident_s16_output_ref, MUL_IDENT_S16_DST_SIZE));
}


void mul_broadcast_h_s16_arm_mul_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_H_S16_LHS_N;
    lhs_dims.h = MUL_BROADCAST_H_S16_LHS_H;
    lhs_dims.w = MUL_BROADCAST_H_S16_LHS_W;
    lhs_dims.c = MUL_BROADCAST_H_S16_LHS_C;

    rhs_dims.n = MUL_BROADCAST_H_S16_RHS_N;
    rhs_dims.h = MUL_BROADCAST_H_S16_RHS_H;
    rhs_dims.w = MUL_BROADCAST_H_S16_RHS_W;
    rhs_dims.c = MUL_BROADCAST_H_S16_RHS_C;

    out_dims.n = MUL_BROADCAST_H_S16_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_H_S16_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_H_S16_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_H_S16_OUTPUT_C;

    const int16_t *lhs = mul_broadcast_h_s16_lhs_input_tensor;
    const int16_t *rhs = mul_broadcast_h_s16_rhs_input_tensor;
    int16_t output[MUL_BROADCAST_H_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_H_S16_LHS_OFFSET,
        MUL_BROADCAST_H_S16_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_H_S16_OUTPUT_OFFSET,
        MUL_BROADCAST_H_S16_OUTPUT_MULT,
        MUL_BROADCAST_H_S16_OUTPUT_SHIFT,
        MUL_BROADCAST_H_S16_ACTIVATION_MIN,
        MUL_BROADCAST_H_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, mul_broadcast_h_s16_output_ref, MUL_BROADCAST_H_S16_DST_SIZE));
}

void mul_broadcast_w_s16_arm_mul_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_W_S16_LHS_N;
    lhs_dims.h = MUL_BROADCAST_W_S16_LHS_H;
    lhs_dims.w = MUL_BROADCAST_W_S16_LHS_W;
    lhs_dims.c = MUL_BROADCAST_W_S16_LHS_C;

    rhs_dims.n = MUL_BROADCAST_W_S16_RHS_N;
    rhs_dims.h = MUL_BROADCAST_W_S16_RHS_H;
    rhs_dims.w = MUL_BROADCAST_W_S16_RHS_W;
    rhs_dims.c = MUL_BROADCAST_W_S16_RHS_C;

    out_dims.n = MUL_BROADCAST_W_S16_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_W_S16_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_W_S16_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_W_S16_OUTPUT_C;

    const int16_t *lhs = mul_broadcast_w_s16_lhs_input_tensor;
    const int16_t *rhs = mul_broadcast_w_s16_rhs_input_tensor;
    int16_t output[MUL_BROADCAST_W_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_W_S16_LHS_OFFSET,
        MUL_BROADCAST_W_S16_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_W_S16_OUTPUT_OFFSET,
        MUL_BROADCAST_W_S16_OUTPUT_MULT,
        MUL_BROADCAST_W_S16_OUTPUT_SHIFT,
        MUL_BROADCAST_W_S16_ACTIVATION_MIN,
        MUL_BROADCAST_W_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, mul_broadcast_w_s16_output_ref, MUL_BROADCAST_W_S16_DST_SIZE));
}

void mul_broadcast_c_s16_arm_mul_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_C_S16_LHS_N;
    lhs_dims.h = MUL_BROADCAST_C_S16_LHS_H;
    lhs_dims.w = MUL_BROADCAST_C_S16_LHS_W;
    lhs_dims.c = MUL_BROADCAST_C_S16_LHS_C;

    rhs_dims.n = MUL_BROADCAST_C_S16_RHS_N;
    rhs_dims.h = MUL_BROADCAST_C_S16_RHS_H;
    rhs_dims.w = MUL_BROADCAST_C_S16_RHS_W;
    rhs_dims.c = MUL_BROADCAST_C_S16_RHS_C;

    out_dims.n = MUL_BROADCAST_C_S16_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_C_S16_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_C_S16_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_C_S16_OUTPUT_C;

    const int16_t *lhs = mul_broadcast_c_s16_lhs_input_tensor;
    const int16_t *rhs = mul_broadcast_c_s16_rhs_input_tensor;
    int16_t output[MUL_BROADCAST_C_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_C_S16_LHS_OFFSET,
        MUL_BROADCAST_C_S16_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_C_S16_OUTPUT_OFFSET,
        MUL_BROADCAST_C_S16_OUTPUT_MULT,
        MUL_BROADCAST_C_S16_OUTPUT_SHIFT,
        MUL_BROADCAST_C_S16_ACTIVATION_MIN,
        MUL_BROADCAST_C_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, mul_broadcast_c_s16_output_ref, MUL_BROADCAST_C_S16_DST_SIZE));
}

void mul_broadcast_hc_s16_arm_mul_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = MUL_BROADCAST_HC_S16_LHS_N;
    lhs_dims.h = MUL_BROADCAST_HC_S16_LHS_H;
    lhs_dims.w = MUL_BROADCAST_HC_S16_LHS_W;
    lhs_dims.c = MUL_BROADCAST_HC_S16_LHS_C;

    rhs_dims.n = MUL_BROADCAST_HC_S16_RHS_N;
    rhs_dims.h = MUL_BROADCAST_HC_S16_RHS_H;
    rhs_dims.w = MUL_BROADCAST_HC_S16_RHS_W;
    rhs_dims.c = MUL_BROADCAST_HC_S16_RHS_C;

    out_dims.n = MUL_BROADCAST_HC_S16_OUTPUT_N;
    out_dims.h = MUL_BROADCAST_HC_S16_OUTPUT_H;
    out_dims.w = MUL_BROADCAST_HC_S16_OUTPUT_W;
    out_dims.c = MUL_BROADCAST_HC_S16_OUTPUT_C;

    const int16_t *lhs = mul_broadcast_hc_s16_lhs_input_tensor;
    const int16_t *rhs = mul_broadcast_hc_s16_rhs_input_tensor;
    int16_t output[MUL_BROADCAST_HC_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_mul_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        MUL_BROADCAST_HC_S16_LHS_OFFSET,
        MUL_BROADCAST_HC_S16_RHS_OFFSET,
        output,
        &out_dims,
        MUL_BROADCAST_HC_S16_OUTPUT_OFFSET,
        MUL_BROADCAST_HC_S16_OUTPUT_MULT,
        MUL_BROADCAST_HC_S16_OUTPUT_SHIFT,
        MUL_BROADCAST_HC_S16_ACTIVATION_MIN,
        MUL_BROADCAST_HC_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, mul_broadcast_hc_s16_output_ref, MUL_BROADCAST_HC_S16_DST_SIZE));
}
