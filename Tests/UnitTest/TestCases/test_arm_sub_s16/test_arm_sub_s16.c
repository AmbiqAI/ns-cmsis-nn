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

#include "../TestData/sub_scalar_s16/test_data.h"
#include "../TestData/sub_ident_s16/test_data.h"
#include "../TestData/sub_broadcast_h_s16/test_data.h"
#include "../TestData/sub_broadcast_w_s16/test_data.h"
#include "../TestData/sub_broadcast_c_s16/test_data.h"
#include "../TestData/sub_broadcast_hc_s16/test_data.h"

#include "../Utils/validate.h"

void sub_scalar_s16_arm_sub_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_SCALAR_S16_LHS_N;
    lhs_dims.h = SUB_SCALAR_S16_LHS_H;
    lhs_dims.w = SUB_SCALAR_S16_LHS_W;
    lhs_dims.c = SUB_SCALAR_S16_LHS_C;

    rhs_dims.n = SUB_SCALAR_S16_RHS_N;
    rhs_dims.h = SUB_SCALAR_S16_RHS_H;
    rhs_dims.w = SUB_SCALAR_S16_RHS_W;
    rhs_dims.c = SUB_SCALAR_S16_RHS_C;

    out_dims.n = SUB_SCALAR_S16_OUTPUT_N;
    out_dims.h = SUB_SCALAR_S16_OUTPUT_H;
    out_dims.w = SUB_SCALAR_S16_OUTPUT_W;
    out_dims.c = SUB_SCALAR_S16_OUTPUT_C;

    const int16_t *lhs = sub_scalar_s16_lhs_input_tensor;
    const int16_t *rhs = sub_scalar_s16_rhs_input_tensor;
    int16_t output[SUB_SCALAR_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_SCALAR_S16_LHS_OFFSET,
        SUB_SCALAR_S16_LHS_MULT,
        SUB_SCALAR_S16_LHS_SHIFT,
        SUB_SCALAR_S16_RHS_OFFSET,
        SUB_SCALAR_S16_RHS_MULT,
        SUB_SCALAR_S16_RHS_SHIFT,
        SUB_SCALAR_S16_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_SCALAR_S16_OUTPUT_OFFSET,
        SUB_SCALAR_S16_OUTPUT_MULT,
        SUB_SCALAR_S16_OUTPUT_SHIFT,
        SUB_SCALAR_S16_ACTIVATION_MIN,
        SUB_SCALAR_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sub_scalar_s16_output_ref, SUB_SCALAR_S16_DST_SIZE));
}


void sub_ident_s16_arm_sub_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_IDENT_S16_LHS_N;
    lhs_dims.h = SUB_IDENT_S16_LHS_H;
    lhs_dims.w = SUB_IDENT_S16_LHS_W;
    lhs_dims.c = SUB_IDENT_S16_LHS_C;

    rhs_dims.n = SUB_IDENT_S16_RHS_N;
    rhs_dims.h = SUB_IDENT_S16_RHS_H;
    rhs_dims.w = SUB_IDENT_S16_RHS_W;
    rhs_dims.c = SUB_IDENT_S16_RHS_C;

    out_dims.n = SUB_IDENT_S16_OUTPUT_N;
    out_dims.h = SUB_IDENT_S16_OUTPUT_H;
    out_dims.w = SUB_IDENT_S16_OUTPUT_W;
    out_dims.c = SUB_IDENT_S16_OUTPUT_C;

    const int16_t *lhs = sub_ident_s16_lhs_input_tensor;
    const int16_t *rhs = sub_ident_s16_rhs_input_tensor;
    int16_t output[SUB_IDENT_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_IDENT_S16_LHS_OFFSET,
        SUB_IDENT_S16_LHS_MULT,
        SUB_IDENT_S16_LHS_SHIFT,
        SUB_IDENT_S16_RHS_OFFSET,
        SUB_IDENT_S16_RHS_MULT,
        SUB_IDENT_S16_RHS_SHIFT,
        SUB_IDENT_S16_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_IDENT_S16_OUTPUT_OFFSET,
        SUB_IDENT_S16_OUTPUT_MULT,
        SUB_IDENT_S16_OUTPUT_SHIFT,
        SUB_IDENT_S16_ACTIVATION_MIN,
        SUB_IDENT_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sub_ident_s16_output_ref, SUB_IDENT_S16_DST_SIZE));
}


void sub_broadcast_h_s16_arm_sub_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_BROADCAST_H_S16_LHS_N;
    lhs_dims.h = SUB_BROADCAST_H_S16_LHS_H;
    lhs_dims.w = SUB_BROADCAST_H_S16_LHS_W;
    lhs_dims.c = SUB_BROADCAST_H_S16_LHS_C;

    rhs_dims.n = SUB_BROADCAST_H_S16_RHS_N;
    rhs_dims.h = SUB_BROADCAST_H_S16_RHS_H;
    rhs_dims.w = SUB_BROADCAST_H_S16_RHS_W;
    rhs_dims.c = SUB_BROADCAST_H_S16_RHS_C;

    out_dims.n = SUB_BROADCAST_H_S16_OUTPUT_N;
    out_dims.h = SUB_BROADCAST_H_S16_OUTPUT_H;
    out_dims.w = SUB_BROADCAST_H_S16_OUTPUT_W;
    out_dims.c = SUB_BROADCAST_H_S16_OUTPUT_C;

    const int16_t *lhs = sub_broadcast_h_s16_lhs_input_tensor;
    const int16_t *rhs = sub_broadcast_h_s16_rhs_input_tensor;
    int16_t output[SUB_BROADCAST_H_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_BROADCAST_H_S16_LHS_OFFSET,
        SUB_BROADCAST_H_S16_LHS_MULT,
        SUB_BROADCAST_H_S16_LHS_SHIFT,
        SUB_BROADCAST_H_S16_RHS_OFFSET,
        SUB_BROADCAST_H_S16_RHS_MULT,
        SUB_BROADCAST_H_S16_RHS_SHIFT,
        SUB_BROADCAST_H_S16_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_BROADCAST_H_S16_OUTPUT_OFFSET,
        SUB_BROADCAST_H_S16_OUTPUT_MULT,
        SUB_BROADCAST_H_S16_OUTPUT_SHIFT,
        SUB_BROADCAST_H_S16_ACTIVATION_MIN,
        SUB_BROADCAST_H_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sub_broadcast_h_s16_output_ref, SUB_BROADCAST_H_S16_DST_SIZE));
}

void sub_broadcast_w_s16_arm_sub_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_BROADCAST_W_S16_LHS_N;
    lhs_dims.h = SUB_BROADCAST_W_S16_LHS_H;
    lhs_dims.w = SUB_BROADCAST_W_S16_LHS_W;
    lhs_dims.c = SUB_BROADCAST_W_S16_LHS_C;

    rhs_dims.n = SUB_BROADCAST_W_S16_RHS_N;
    rhs_dims.h = SUB_BROADCAST_W_S16_RHS_H;
    rhs_dims.w = SUB_BROADCAST_W_S16_RHS_W;
    rhs_dims.c = SUB_BROADCAST_W_S16_RHS_C;

    out_dims.n = SUB_BROADCAST_W_S16_OUTPUT_N;
    out_dims.h = SUB_BROADCAST_W_S16_OUTPUT_H;
    out_dims.w = SUB_BROADCAST_W_S16_OUTPUT_W;
    out_dims.c = SUB_BROADCAST_W_S16_OUTPUT_C;

    const int16_t *lhs = sub_broadcast_w_s16_lhs_input_tensor;
    const int16_t *rhs = sub_broadcast_w_s16_rhs_input_tensor;
    int16_t output[SUB_BROADCAST_W_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_BROADCAST_W_S16_LHS_OFFSET,
        SUB_BROADCAST_W_S16_LHS_MULT,
        SUB_BROADCAST_W_S16_LHS_SHIFT,
        SUB_BROADCAST_W_S16_RHS_OFFSET,
        SUB_BROADCAST_W_S16_RHS_MULT,
        SUB_BROADCAST_W_S16_RHS_SHIFT,
        SUB_BROADCAST_W_S16_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_BROADCAST_W_S16_OUTPUT_OFFSET,
        SUB_BROADCAST_W_S16_OUTPUT_MULT,
        SUB_BROADCAST_W_S16_OUTPUT_SHIFT,
        SUB_BROADCAST_W_S16_ACTIVATION_MIN,
        SUB_BROADCAST_W_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sub_broadcast_w_s16_output_ref, SUB_BROADCAST_W_S16_DST_SIZE));
}

void sub_broadcast_c_s16_arm_sub_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_BROADCAST_C_S16_LHS_N;
    lhs_dims.h = SUB_BROADCAST_C_S16_LHS_H;
    lhs_dims.w = SUB_BROADCAST_C_S16_LHS_W;
    lhs_dims.c = SUB_BROADCAST_C_S16_LHS_C;

    rhs_dims.n = SUB_BROADCAST_C_S16_RHS_N;
    rhs_dims.h = SUB_BROADCAST_C_S16_RHS_H;
    rhs_dims.w = SUB_BROADCAST_C_S16_RHS_W;
    rhs_dims.c = SUB_BROADCAST_C_S16_RHS_C;

    out_dims.n = SUB_BROADCAST_C_S16_OUTPUT_N;
    out_dims.h = SUB_BROADCAST_C_S16_OUTPUT_H;
    out_dims.w = SUB_BROADCAST_C_S16_OUTPUT_W;
    out_dims.c = SUB_BROADCAST_C_S16_OUTPUT_C;

    const int16_t *lhs = sub_broadcast_c_s16_lhs_input_tensor;
    const int16_t *rhs = sub_broadcast_c_s16_rhs_input_tensor;
    int16_t output[SUB_BROADCAST_C_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_BROADCAST_C_S16_LHS_OFFSET,
        SUB_BROADCAST_C_S16_LHS_MULT,
        SUB_BROADCAST_C_S16_LHS_SHIFT,
        SUB_BROADCAST_C_S16_RHS_OFFSET,
        SUB_BROADCAST_C_S16_RHS_MULT,
        SUB_BROADCAST_C_S16_RHS_SHIFT,
        SUB_BROADCAST_C_S16_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_BROADCAST_C_S16_OUTPUT_OFFSET,
        SUB_BROADCAST_C_S16_OUTPUT_MULT,
        SUB_BROADCAST_C_S16_OUTPUT_SHIFT,
        SUB_BROADCAST_C_S16_ACTIVATION_MIN,
        SUB_BROADCAST_C_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sub_broadcast_c_s16_output_ref, SUB_BROADCAST_C_S16_DST_SIZE));
}

void sub_broadcast_hc_s16_arm_sub_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SUB_BROADCAST_HC_S16_LHS_N;
    lhs_dims.h = SUB_BROADCAST_HC_S16_LHS_H;
    lhs_dims.w = SUB_BROADCAST_HC_S16_LHS_W;
    lhs_dims.c = SUB_BROADCAST_HC_S16_LHS_C;

    rhs_dims.n = SUB_BROADCAST_HC_S16_RHS_N;
    rhs_dims.h = SUB_BROADCAST_HC_S16_RHS_H;
    rhs_dims.w = SUB_BROADCAST_HC_S16_RHS_W;
    rhs_dims.c = SUB_BROADCAST_HC_S16_RHS_C;

    out_dims.n = SUB_BROADCAST_HC_S16_OUTPUT_N;
    out_dims.h = SUB_BROADCAST_HC_S16_OUTPUT_H;
    out_dims.w = SUB_BROADCAST_HC_S16_OUTPUT_W;
    out_dims.c = SUB_BROADCAST_HC_S16_OUTPUT_C;

    const int16_t *lhs = sub_broadcast_hc_s16_lhs_input_tensor;
    const int16_t *rhs = sub_broadcast_hc_s16_rhs_input_tensor;
    int16_t output[SUB_BROADCAST_HC_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_sub_s16(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        SUB_BROADCAST_HC_S16_LHS_OFFSET,
        SUB_BROADCAST_HC_S16_LHS_MULT,
        SUB_BROADCAST_HC_S16_LHS_SHIFT,
        SUB_BROADCAST_HC_S16_RHS_OFFSET,
        SUB_BROADCAST_HC_S16_RHS_MULT,
        SUB_BROADCAST_HC_S16_RHS_SHIFT,
        SUB_BROADCAST_HC_S16_LEFT_SHIFT,
        output,
        &out_dims,
        SUB_BROADCAST_HC_S16_OUTPUT_OFFSET,
        SUB_BROADCAST_HC_S16_OUTPUT_MULT,
        SUB_BROADCAST_HC_S16_OUTPUT_SHIFT,
        SUB_BROADCAST_HC_S16_ACTIVATION_MIN,
        SUB_BROADCAST_HC_S16_ACTIVATION_MAX
    );

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, sub_broadcast_hc_s16_output_ref, SUB_BROADCAST_HC_S16_DST_SIZE));
}
