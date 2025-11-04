/*
 * Copyright (C) 2025 Arm Limited or its affiliates.
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

#include "../TestData/argmax_c_s8/test_data.h"
#include "../TestData/argmin_c_s8/test_data.h"
#include "../TestData/argmax_h_s8/test_data.h"
#include "../TestData/argmin_h_s8/test_data.h"
#include "../TestData/argmax_n_s8/test_data.h"
#include "../TestData/argmin_n_s8/test_data.h"

void argmax_axis_c_s8(void)
{
    const cmsis_nn_dims dims = {
        ARGMAX_C_S8_LHS_N,
        ARGMAX_C_S8_LHS_H,
        ARGMAX_C_S8_LHS_W,
        ARGMAX_C_S8_LHS_C
    };

    int32_t output[ARGMAX_C_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status status = arm_argmax_s8(argmax_c_s8_lhs_input_tensor,
                                               &dims,
                                               ARGMAX_C_S8_AXIS,
                                               output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT32_ARRAY(argmax_c_s8_output, output, ARGMAX_C_S8_DST_SIZE);
}

void argmin_axis_c_s8(void)
{
    const cmsis_nn_dims dims = {
        ARGMIN_C_S8_LHS_N,
        ARGMIN_C_S8_LHS_H,
        ARGMIN_C_S8_LHS_W,
        ARGMIN_C_S8_LHS_C
    };

    int32_t output[ARGMIN_C_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status status = arm_argmin_s8(argmin_c_s8_lhs_input_tensor,
                                               &dims,
                                               ARGMIN_C_S8_AXIS,
                                               output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT32_ARRAY(argmin_c_s8_output, output, ARGMIN_C_S8_DST_SIZE);
}

void argmax_axis_h_s8(void)
{
    const cmsis_nn_dims dims = {
        ARGMAX_H_S8_LHS_N,
        ARGMAX_H_S8_LHS_H,
        ARGMAX_H_S8_LHS_W,
        ARGMAX_H_S8_LHS_C
    };

    int32_t output[ARGMAX_H_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status status = arm_argmax_s8(argmax_h_s8_lhs_input_tensor,
                                               &dims,
                                               ARGMAX_H_S8_AXIS,
                                               output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT32_ARRAY(argmax_h_s8_output, output, ARGMAX_H_S8_DST_SIZE);
}

void argmin_axis_h_s8(void)
{
    const cmsis_nn_dims dims = {
        ARGMIN_H_S8_LHS_N,
        ARGMIN_H_S8_LHS_H,
        ARGMIN_H_S8_LHS_W,
        ARGMIN_H_S8_LHS_C
    };

    int32_t output[ARGMIN_H_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status status = arm_argmin_s8(argmin_h_s8_lhs_input_tensor,
                                               &dims,
                                               ARGMIN_H_S8_AXIS,
                                               output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT32_ARRAY(argmin_h_s8_output, output, ARGMIN_H_S8_DST_SIZE);
}

void argmax_axis_n_s8(void)
{
    const cmsis_nn_dims dims = {
        ARGMAX_N_S8_LHS_N,
        ARGMAX_N_S8_LHS_H,
        ARGMAX_N_S8_LHS_W,
        ARGMAX_N_S8_LHS_C
    };

    int32_t output[ARGMAX_N_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status status = arm_argmax_s8(argmax_n_s8_lhs_input_tensor,
                                               &dims,
                                               ARGMAX_N_S8_AXIS,
                                               output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT32_ARRAY(argmax_n_s8_output, output, ARGMAX_N_S8_DST_SIZE);
}

void argmin_axis_n_s8(void)
{
    const cmsis_nn_dims dims = {
        ARGMIN_N_S8_LHS_N,
        ARGMIN_N_S8_LHS_H,
        ARGMIN_N_S8_LHS_W,
        ARGMIN_N_S8_LHS_C
    };

    int32_t output[ARGMIN_N_S8_DST_SIZE] = {0};

    arm_cmsis_nn_status status = arm_argmin_s8(argmin_n_s8_lhs_input_tensor,
                                               &dims,
                                               ARGMIN_N_S8_AXIS,
                                               output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT32_ARRAY(argmin_n_s8_output, output, ARGMIN_N_S8_DST_SIZE);
}

void arg_axis_invalid_s8(void)
{
    const int8_t input[] = {1, 2, 3, 4};
    const cmsis_nn_dims dims = {
        .n = 1,
        .h = 1,
        .w = 1,
        .c = 4
    };

    int32_t output[4] = {0};

    arm_cmsis_nn_status status_max = arm_argmax_s8(input, &dims, 4, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, status_max);

    arm_cmsis_nn_status status_min = arm_argmin_s8(input, &dims, -1, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, status_min);
}
