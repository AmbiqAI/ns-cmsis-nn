/*
 * Copyright (C) 2026 Arm Limited or its affiliates.
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

#include "../TestData/squared_difference_scalar_s16/test_data.h"
#include "../TestData/squared_difference_ident_s16/test_data.h"
#include "../TestData/squared_difference_broadcast_n_s16/test_data.h"
#include "../TestData/squared_difference_broadcast_h_s16/test_data.h"
#include "../TestData/squared_difference_broadcast_w_s16/test_data.h"
#include "../TestData/squared_difference_broadcast_c_s16/test_data.h"
#include "../TestData/squared_difference_broadcast_hc_s16/test_data.h"

#include "../Utils/validate.h"

void squared_difference_scalar_s16_arm_squared_difference_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SQUARED_DIFFERENCE_SCALAR_S16_LHS_N;
    lhs_dims.h = SQUARED_DIFFERENCE_SCALAR_S16_LHS_H;
    lhs_dims.w = SQUARED_DIFFERENCE_SCALAR_S16_LHS_W;
    lhs_dims.c = SQUARED_DIFFERENCE_SCALAR_S16_LHS_C;

    rhs_dims.n = SQUARED_DIFFERENCE_SCALAR_S16_RHS_N;
    rhs_dims.h = SQUARED_DIFFERENCE_SCALAR_S16_RHS_H;
    rhs_dims.w = SQUARED_DIFFERENCE_SCALAR_S16_RHS_W;
    rhs_dims.c = SQUARED_DIFFERENCE_SCALAR_S16_RHS_C;

    out_dims.n = SQUARED_DIFFERENCE_SCALAR_S16_OUTPUT_N;
    out_dims.h = SQUARED_DIFFERENCE_SCALAR_S16_OUTPUT_H;
    out_dims.w = SQUARED_DIFFERENCE_SCALAR_S16_OUTPUT_W;
    out_dims.c = SQUARED_DIFFERENCE_SCALAR_S16_OUTPUT_C;

    const int16_t *lhs = squared_difference_scalar_s16_lhs_input_tensor;
    const int16_t *rhs = squared_difference_scalar_s16_rhs_input_tensor;
    int16_t output[SQUARED_DIFFERENCE_SCALAR_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_squared_difference_s16(lhs,
                                                           &lhs_dims,
                                                           rhs,
                                                           &rhs_dims,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_INPUT1_OFFSET,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_INPUT1_MULT,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_INPUT1_SHIFT,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_INPUT2_OFFSET,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_INPUT2_MULT,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_INPUT2_SHIFT,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_LEFT_SHIFT,
                                                           output,
                                                           &out_dims,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_OUTPUT_OFFSET,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_OUTPUT_MULT,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_OUTPUT_SHIFT,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_ACTIVATION_MIN,
                                                           SQUARED_DIFFERENCE_SCALAR_S16_ACTIVATION_MAX);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, squared_difference_scalar_s16_output_ref, SQUARED_DIFFERENCE_SCALAR_S16_DST_SIZE));
}

void squared_difference_ident_s16_arm_squared_difference_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SQUARED_DIFFERENCE_IDENT_S16_LHS_N;
    lhs_dims.h = SQUARED_DIFFERENCE_IDENT_S16_LHS_H;
    lhs_dims.w = SQUARED_DIFFERENCE_IDENT_S16_LHS_W;
    lhs_dims.c = SQUARED_DIFFERENCE_IDENT_S16_LHS_C;

    rhs_dims.n = SQUARED_DIFFERENCE_IDENT_S16_RHS_N;
    rhs_dims.h = SQUARED_DIFFERENCE_IDENT_S16_RHS_H;
    rhs_dims.w = SQUARED_DIFFERENCE_IDENT_S16_RHS_W;
    rhs_dims.c = SQUARED_DIFFERENCE_IDENT_S16_RHS_C;

    out_dims.n = SQUARED_DIFFERENCE_IDENT_S16_OUTPUT_N;
    out_dims.h = SQUARED_DIFFERENCE_IDENT_S16_OUTPUT_H;
    out_dims.w = SQUARED_DIFFERENCE_IDENT_S16_OUTPUT_W;
    out_dims.c = SQUARED_DIFFERENCE_IDENT_S16_OUTPUT_C;

    const int16_t *lhs = squared_difference_ident_s16_lhs_input_tensor;
    const int16_t *rhs = squared_difference_ident_s16_rhs_input_tensor;
    int16_t output[SQUARED_DIFFERENCE_IDENT_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_squared_difference_s16(lhs,
                                                           &lhs_dims,
                                                           rhs,
                                                           &rhs_dims,
                                                           SQUARED_DIFFERENCE_IDENT_S16_INPUT1_OFFSET,
                                                           SQUARED_DIFFERENCE_IDENT_S16_INPUT1_MULT,
                                                           SQUARED_DIFFERENCE_IDENT_S16_INPUT1_SHIFT,
                                                           SQUARED_DIFFERENCE_IDENT_S16_INPUT2_OFFSET,
                                                           SQUARED_DIFFERENCE_IDENT_S16_INPUT2_MULT,
                                                           SQUARED_DIFFERENCE_IDENT_S16_INPUT2_SHIFT,
                                                           SQUARED_DIFFERENCE_IDENT_S16_LEFT_SHIFT,
                                                           output,
                                                           &out_dims,
                                                           SQUARED_DIFFERENCE_IDENT_S16_OUTPUT_OFFSET,
                                                           SQUARED_DIFFERENCE_IDENT_S16_OUTPUT_MULT,
                                                           SQUARED_DIFFERENCE_IDENT_S16_OUTPUT_SHIFT,
                                                           SQUARED_DIFFERENCE_IDENT_S16_ACTIVATION_MIN,
                                                           SQUARED_DIFFERENCE_IDENT_S16_ACTIVATION_MAX);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, squared_difference_ident_s16_output_ref, SQUARED_DIFFERENCE_IDENT_S16_DST_SIZE));
}

void squared_difference_broadcast_n_s16_arm_squared_difference_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_N_S16_LHS_N;
    lhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_N_S16_LHS_H;
    lhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_N_S16_LHS_W;
    lhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_N_S16_LHS_C;

    rhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_N_S16_RHS_N;
    rhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_N_S16_RHS_H;
    rhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_N_S16_RHS_W;
    rhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_N_S16_RHS_C;

    out_dims.n = SQUARED_DIFFERENCE_BROADCAST_N_S16_OUTPUT_N;
    out_dims.h = SQUARED_DIFFERENCE_BROADCAST_N_S16_OUTPUT_H;
    out_dims.w = SQUARED_DIFFERENCE_BROADCAST_N_S16_OUTPUT_W;
    out_dims.c = SQUARED_DIFFERENCE_BROADCAST_N_S16_OUTPUT_C;

    const int16_t *lhs = squared_difference_broadcast_n_s16_lhs_input_tensor;
    const int16_t *rhs = squared_difference_broadcast_n_s16_rhs_input_tensor;
    int16_t output[SQUARED_DIFFERENCE_BROADCAST_N_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_squared_difference_s16(lhs,
                                                           &lhs_dims,
                                                           rhs,
                                                           &rhs_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_INPUT1_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_INPUT1_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_INPUT1_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_INPUT2_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_INPUT2_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_INPUT2_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_LEFT_SHIFT,
                                                           output,
                                                           &out_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_OUTPUT_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_OUTPUT_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_OUTPUT_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_ACTIVATION_MIN,
                                                           SQUARED_DIFFERENCE_BROADCAST_N_S16_ACTIVATION_MAX);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                              squared_difference_broadcast_n_s16_output_ref,
                              SQUARED_DIFFERENCE_BROADCAST_N_S16_DST_SIZE));
}

void squared_difference_broadcast_h_s16_arm_squared_difference_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_H_S16_LHS_N;
    lhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_H_S16_LHS_H;
    lhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_H_S16_LHS_W;
    lhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_H_S16_LHS_C;

    rhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_H_S16_RHS_N;
    rhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_H_S16_RHS_H;
    rhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_H_S16_RHS_W;
    rhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_H_S16_RHS_C;

    out_dims.n = SQUARED_DIFFERENCE_BROADCAST_H_S16_OUTPUT_N;
    out_dims.h = SQUARED_DIFFERENCE_BROADCAST_H_S16_OUTPUT_H;
    out_dims.w = SQUARED_DIFFERENCE_BROADCAST_H_S16_OUTPUT_W;
    out_dims.c = SQUARED_DIFFERENCE_BROADCAST_H_S16_OUTPUT_C;

    const int16_t *lhs = squared_difference_broadcast_h_s16_lhs_input_tensor;
    const int16_t *rhs = squared_difference_broadcast_h_s16_rhs_input_tensor;
    int16_t output[SQUARED_DIFFERENCE_BROADCAST_H_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_squared_difference_s16(lhs,
                                                           &lhs_dims,
                                                           rhs,
                                                           &rhs_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_INPUT1_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_INPUT1_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_INPUT1_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_INPUT2_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_INPUT2_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_INPUT2_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_LEFT_SHIFT,
                                                           output,
                                                           &out_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_OUTPUT_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_OUTPUT_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_OUTPUT_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_ACTIVATION_MIN,
                                                           SQUARED_DIFFERENCE_BROADCAST_H_S16_ACTIVATION_MAX);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                              squared_difference_broadcast_h_s16_output_ref,
                              SQUARED_DIFFERENCE_BROADCAST_H_S16_DST_SIZE));
}

void squared_difference_broadcast_w_s16_arm_squared_difference_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_W_S16_LHS_N;
    lhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_W_S16_LHS_H;
    lhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_W_S16_LHS_W;
    lhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_W_S16_LHS_C;

    rhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_W_S16_RHS_N;
    rhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_W_S16_RHS_H;
    rhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_W_S16_RHS_W;
    rhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_W_S16_RHS_C;

    out_dims.n = SQUARED_DIFFERENCE_BROADCAST_W_S16_OUTPUT_N;
    out_dims.h = SQUARED_DIFFERENCE_BROADCAST_W_S16_OUTPUT_H;
    out_dims.w = SQUARED_DIFFERENCE_BROADCAST_W_S16_OUTPUT_W;
    out_dims.c = SQUARED_DIFFERENCE_BROADCAST_W_S16_OUTPUT_C;

    const int16_t *lhs = squared_difference_broadcast_w_s16_lhs_input_tensor;
    const int16_t *rhs = squared_difference_broadcast_w_s16_rhs_input_tensor;
    int16_t output[SQUARED_DIFFERENCE_BROADCAST_W_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_squared_difference_s16(lhs,
                                                           &lhs_dims,
                                                           rhs,
                                                           &rhs_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_INPUT1_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_INPUT1_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_INPUT1_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_INPUT2_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_INPUT2_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_INPUT2_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_LEFT_SHIFT,
                                                           output,
                                                           &out_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_OUTPUT_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_OUTPUT_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_OUTPUT_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_ACTIVATION_MIN,
                                                           SQUARED_DIFFERENCE_BROADCAST_W_S16_ACTIVATION_MAX);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                              squared_difference_broadcast_w_s16_output_ref,
                              SQUARED_DIFFERENCE_BROADCAST_W_S16_DST_SIZE));
}

void squared_difference_broadcast_c_s16_arm_squared_difference_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_C_S16_LHS_N;
    lhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_C_S16_LHS_H;
    lhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_C_S16_LHS_W;
    lhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_C_S16_LHS_C;

    rhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_C_S16_RHS_N;
    rhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_C_S16_RHS_H;
    rhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_C_S16_RHS_W;
    rhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_C_S16_RHS_C;

    out_dims.n = SQUARED_DIFFERENCE_BROADCAST_C_S16_OUTPUT_N;
    out_dims.h = SQUARED_DIFFERENCE_BROADCAST_C_S16_OUTPUT_H;
    out_dims.w = SQUARED_DIFFERENCE_BROADCAST_C_S16_OUTPUT_W;
    out_dims.c = SQUARED_DIFFERENCE_BROADCAST_C_S16_OUTPUT_C;

    const int16_t *lhs = squared_difference_broadcast_c_s16_lhs_input_tensor;
    const int16_t *rhs = squared_difference_broadcast_c_s16_rhs_input_tensor;
    int16_t output[SQUARED_DIFFERENCE_BROADCAST_C_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_squared_difference_s16(lhs,
                                                           &lhs_dims,
                                                           rhs,
                                                           &rhs_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_INPUT1_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_INPUT1_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_INPUT1_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_INPUT2_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_INPUT2_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_INPUT2_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_LEFT_SHIFT,
                                                           output,
                                                           &out_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_OUTPUT_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_OUTPUT_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_OUTPUT_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_ACTIVATION_MIN,
                                                           SQUARED_DIFFERENCE_BROADCAST_C_S16_ACTIVATION_MAX);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                              squared_difference_broadcast_c_s16_output_ref,
                              SQUARED_DIFFERENCE_BROADCAST_C_S16_DST_SIZE));
}

void squared_difference_broadcast_hc_s16_arm_squared_difference_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_HC_S16_LHS_N;
    lhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_HC_S16_LHS_H;
    lhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_HC_S16_LHS_W;
    lhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_HC_S16_LHS_C;

    rhs_dims.n = SQUARED_DIFFERENCE_BROADCAST_HC_S16_RHS_N;
    rhs_dims.h = SQUARED_DIFFERENCE_BROADCAST_HC_S16_RHS_H;
    rhs_dims.w = SQUARED_DIFFERENCE_BROADCAST_HC_S16_RHS_W;
    rhs_dims.c = SQUARED_DIFFERENCE_BROADCAST_HC_S16_RHS_C;

    out_dims.n = SQUARED_DIFFERENCE_BROADCAST_HC_S16_OUTPUT_N;
    out_dims.h = SQUARED_DIFFERENCE_BROADCAST_HC_S16_OUTPUT_H;
    out_dims.w = SQUARED_DIFFERENCE_BROADCAST_HC_S16_OUTPUT_W;
    out_dims.c = SQUARED_DIFFERENCE_BROADCAST_HC_S16_OUTPUT_C;

    const int16_t *lhs = squared_difference_broadcast_hc_s16_lhs_input_tensor;
    const int16_t *rhs = squared_difference_broadcast_hc_s16_rhs_input_tensor;
    int16_t output[SQUARED_DIFFERENCE_BROADCAST_HC_S16_DST_SIZE] = {0};

    arm_cmsis_nn_status result = arm_squared_difference_s16(lhs,
                                                           &lhs_dims,
                                                           rhs,
                                                           &rhs_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_INPUT1_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_INPUT1_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_INPUT1_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_INPUT2_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_INPUT2_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_INPUT2_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_LEFT_SHIFT,
                                                           output,
                                                           &out_dims,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_OUTPUT_OFFSET,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_OUTPUT_MULT,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_OUTPUT_SHIFT,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_ACTIVATION_MIN,
                                                           SQUARED_DIFFERENCE_BROADCAST_HC_S16_ACTIVATION_MAX);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                              squared_difference_broadcast_hc_s16_output_ref,
                              SQUARED_DIFFERENCE_BROADCAST_HC_S16_DST_SIZE));
}
