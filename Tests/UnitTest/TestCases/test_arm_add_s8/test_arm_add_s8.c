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

#include "../TestData/add_scalar_s8/test_data.h"
#include "../TestData/add_ident_s8/test_data.h"
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
