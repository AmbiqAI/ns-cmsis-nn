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
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/comparison_equal_ident_s16/test_data.h"
#include "../TestData/comparison_greater_broadcast_h_s16/test_data.h"
#include "../TestData/comparison_greater_equal_broadcast_w_s16/test_data.h"
#include "../TestData/comparison_less_broadcast_c_s16/test_data.h"
#include "../TestData/comparison_less_equal_broadcast_hw_s16/test_data.h"
#include "../TestData/comparison_not_equal_scalar_s16/test_data.h"

#include "../Utils/validate.h"

static void fill_dims(cmsis_nn_dims *dims, int32_t n, int32_t h, int32_t w, int32_t c)
{
    dims->n = n;
    dims->h = h;
    dims->w = w;
    dims->c = c;
}

void comparison_equal_ident_s16_arm_equal_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    fill_dims(&input_1_dims,
              COMPARISON_EQUAL_IDENT_S16_INPUT_1_N,
              COMPARISON_EQUAL_IDENT_S16_INPUT_1_H,
              COMPARISON_EQUAL_IDENT_S16_INPUT_1_W,
              COMPARISON_EQUAL_IDENT_S16_INPUT_1_C);
    fill_dims(&input_2_dims,
              COMPARISON_EQUAL_IDENT_S16_INPUT_2_N,
              COMPARISON_EQUAL_IDENT_S16_INPUT_2_H,
              COMPARISON_EQUAL_IDENT_S16_INPUT_2_W,
              COMPARISON_EQUAL_IDENT_S16_INPUT_2_C);
    fill_dims(&output_dims,
              COMPARISON_EQUAL_IDENT_S16_OUTPUT_N,
              COMPARISON_EQUAL_IDENT_S16_OUTPUT_H,
              COMPARISON_EQUAL_IDENT_S16_OUTPUT_W,
              COMPARISON_EQUAL_IDENT_S16_OUTPUT_C);

    bool output[COMPARISON_EQUAL_IDENT_S16_DST_SIZE] = {false};

    arm_cmsis_nn_status result = arm_equal_s16(NULL,
                                               comparison_equal_ident_s16_input_1_input_tensor,
                                               &input_1_dims,
                                               comparison_equal_ident_s16_input_2_input_tensor,
                                               &input_2_dims,
                                               output,
                                               &output_dims,
                                               COMPARISON_EQUAL_IDENT_S16_INPUT_1_OFFSET,
                                               COMPARISON_EQUAL_IDENT_S16_INPUT_1_MULT,
                                               COMPARISON_EQUAL_IDENT_S16_INPUT_1_SHIFT,
                                               COMPARISON_EQUAL_IDENT_S16_INPUT_2_OFFSET,
                                               COMPARISON_EQUAL_IDENT_S16_INPUT_2_MULT,
                                               COMPARISON_EQUAL_IDENT_S16_INPUT_2_SHIFT,
                                               COMPARISON_EQUAL_IDENT_S16_LEFT_SHIFT);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(
        validate_bool(output, comparison_equal_ident_s16_output_ref, COMPARISON_EQUAL_IDENT_S16_DST_SIZE));
}

void comparison_not_equal_scalar_s16_arm_not_equal_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    fill_dims(&input_1_dims,
              COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_1_N,
              COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_1_H,
              COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_1_W,
              COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_1_C);
    fill_dims(&input_2_dims,
              COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_2_N,
              COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_2_H,
              COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_2_W,
              COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_2_C);
    fill_dims(&output_dims,
              COMPARISON_NOT_EQUAL_SCALAR_S16_OUTPUT_N,
              COMPARISON_NOT_EQUAL_SCALAR_S16_OUTPUT_H,
              COMPARISON_NOT_EQUAL_SCALAR_S16_OUTPUT_W,
              COMPARISON_NOT_EQUAL_SCALAR_S16_OUTPUT_C);

    bool output[COMPARISON_NOT_EQUAL_SCALAR_S16_DST_SIZE] = {false};

    arm_cmsis_nn_status result = arm_not_equal_s16(NULL,
                                                   comparison_not_equal_scalar_s16_input_1_input_tensor,
                                                   &input_1_dims,
                                                   comparison_not_equal_scalar_s16_input_2_input_tensor,
                                                   &input_2_dims,
                                                   output,
                                                   &output_dims,
                                                   COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_1_OFFSET,
                                                   COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_1_MULT,
                                                   COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_1_SHIFT,
                                                   COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_2_OFFSET,
                                                   COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_2_MULT,
                                                   COMPARISON_NOT_EQUAL_SCALAR_S16_INPUT_2_SHIFT,
                                                   COMPARISON_NOT_EQUAL_SCALAR_S16_LEFT_SHIFT);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_bool(
        output, comparison_not_equal_scalar_s16_output_ref, COMPARISON_NOT_EQUAL_SCALAR_S16_DST_SIZE));
}

void comparison_greater_broadcast_h_s16_arm_greater_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    fill_dims(&input_1_dims,
              COMPARISON_GREATER_BROADCAST_H_S16_INPUT_1_N,
              COMPARISON_GREATER_BROADCAST_H_S16_INPUT_1_H,
              COMPARISON_GREATER_BROADCAST_H_S16_INPUT_1_W,
              COMPARISON_GREATER_BROADCAST_H_S16_INPUT_1_C);
    fill_dims(&input_2_dims,
              COMPARISON_GREATER_BROADCAST_H_S16_INPUT_2_N,
              COMPARISON_GREATER_BROADCAST_H_S16_INPUT_2_H,
              COMPARISON_GREATER_BROADCAST_H_S16_INPUT_2_W,
              COMPARISON_GREATER_BROADCAST_H_S16_INPUT_2_C);
    fill_dims(&output_dims,
              COMPARISON_GREATER_BROADCAST_H_S16_OUTPUT_N,
              COMPARISON_GREATER_BROADCAST_H_S16_OUTPUT_H,
              COMPARISON_GREATER_BROADCAST_H_S16_OUTPUT_W,
              COMPARISON_GREATER_BROADCAST_H_S16_OUTPUT_C);

    bool output[COMPARISON_GREATER_BROADCAST_H_S16_DST_SIZE] = {false};

    arm_cmsis_nn_status result = arm_greater_s16(NULL,
                                                 comparison_greater_broadcast_h_s16_input_1_input_tensor,
                                                 &input_1_dims,
                                                 comparison_greater_broadcast_h_s16_input_2_input_tensor,
                                                 &input_2_dims,
                                                 output,
                                                 &output_dims,
                                                 COMPARISON_GREATER_BROADCAST_H_S16_INPUT_1_OFFSET,
                                                 COMPARISON_GREATER_BROADCAST_H_S16_INPUT_1_MULT,
                                                 COMPARISON_GREATER_BROADCAST_H_S16_INPUT_1_SHIFT,
                                                 COMPARISON_GREATER_BROADCAST_H_S16_INPUT_2_OFFSET,
                                                 COMPARISON_GREATER_BROADCAST_H_S16_INPUT_2_MULT,
                                                 COMPARISON_GREATER_BROADCAST_H_S16_INPUT_2_SHIFT,
                                                 COMPARISON_GREATER_BROADCAST_H_S16_LEFT_SHIFT);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_bool(
        output, comparison_greater_broadcast_h_s16_output_ref, COMPARISON_GREATER_BROADCAST_H_S16_DST_SIZE));
}

void comparison_greater_equal_broadcast_w_s16_arm_comparison_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    fill_dims(&input_1_dims,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_1_N,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_1_H,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_1_W,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_1_C);
    fill_dims(&input_2_dims,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_2_N,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_2_H,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_2_W,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_2_C);
    fill_dims(&output_dims,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_OUTPUT_N,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_OUTPUT_H,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_OUTPUT_W,
              COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_OUTPUT_C);

    bool output[COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_DST_SIZE] = {false};

    arm_cmsis_nn_status result = arm_comparison_s16(NULL,
                                                    comparison_greater_equal_broadcast_w_s16_input_1_input_tensor,
                                                    &input_1_dims,
                                                    comparison_greater_equal_broadcast_w_s16_input_2_input_tensor,
                                                    &input_2_dims,
                                                    output,
                                                    &output_dims,
                                                    COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_1_OFFSET,
                                                    COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_1_MULT,
                                                    COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_1_SHIFT,
                                                    COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_2_OFFSET,
                                                    COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_2_MULT,
                                                    COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_INPUT_2_SHIFT,
                                                    COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_LEFT_SHIFT,
                                                    COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_OPERATION);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_bool(output,
                                       comparison_greater_equal_broadcast_w_s16_output_ref,
                                       COMPARISON_GREATER_EQUAL_BROADCAST_W_S16_DST_SIZE));
}

void comparison_less_broadcast_c_s16_arm_less_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    fill_dims(&input_1_dims,
              COMPARISON_LESS_BROADCAST_C_S16_INPUT_1_N,
              COMPARISON_LESS_BROADCAST_C_S16_INPUT_1_H,
              COMPARISON_LESS_BROADCAST_C_S16_INPUT_1_W,
              COMPARISON_LESS_BROADCAST_C_S16_INPUT_1_C);
    fill_dims(&input_2_dims,
              COMPARISON_LESS_BROADCAST_C_S16_INPUT_2_N,
              COMPARISON_LESS_BROADCAST_C_S16_INPUT_2_H,
              COMPARISON_LESS_BROADCAST_C_S16_INPUT_2_W,
              COMPARISON_LESS_BROADCAST_C_S16_INPUT_2_C);
    fill_dims(&output_dims,
              COMPARISON_LESS_BROADCAST_C_S16_OUTPUT_N,
              COMPARISON_LESS_BROADCAST_C_S16_OUTPUT_H,
              COMPARISON_LESS_BROADCAST_C_S16_OUTPUT_W,
              COMPARISON_LESS_BROADCAST_C_S16_OUTPUT_C);

    bool output[COMPARISON_LESS_BROADCAST_C_S16_DST_SIZE] = {false};

    arm_cmsis_nn_status result = arm_less_s16(NULL,
                                              comparison_less_broadcast_c_s16_input_1_input_tensor,
                                              &input_1_dims,
                                              comparison_less_broadcast_c_s16_input_2_input_tensor,
                                              &input_2_dims,
                                              output,
                                              &output_dims,
                                              COMPARISON_LESS_BROADCAST_C_S16_INPUT_1_OFFSET,
                                              COMPARISON_LESS_BROADCAST_C_S16_INPUT_1_MULT,
                                              COMPARISON_LESS_BROADCAST_C_S16_INPUT_1_SHIFT,
                                              COMPARISON_LESS_BROADCAST_C_S16_INPUT_2_OFFSET,
                                              COMPARISON_LESS_BROADCAST_C_S16_INPUT_2_MULT,
                                              COMPARISON_LESS_BROADCAST_C_S16_INPUT_2_SHIFT,
                                              COMPARISON_LESS_BROADCAST_C_S16_LEFT_SHIFT);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_bool(
        output, comparison_less_broadcast_c_s16_output_ref, COMPARISON_LESS_BROADCAST_C_S16_DST_SIZE));
}

void comparison_less_equal_broadcast_hw_s16_arm_less_equal_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    fill_dims(&input_1_dims,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_1_N,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_1_H,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_1_W,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_1_C);
    fill_dims(&input_2_dims,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_2_N,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_2_H,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_2_W,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_2_C);
    fill_dims(&output_dims,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_OUTPUT_N,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_OUTPUT_H,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_OUTPUT_W,
              COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_OUTPUT_C);

    bool output[COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_DST_SIZE] = {false};

    arm_cmsis_nn_status result = arm_less_equal_s16(NULL,
                                                    comparison_less_equal_broadcast_hw_s16_input_1_input_tensor,
                                                    &input_1_dims,
                                                    comparison_less_equal_broadcast_hw_s16_input_2_input_tensor,
                                                    &input_2_dims,
                                                    output,
                                                    &output_dims,
                                                    COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_1_OFFSET,
                                                    COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_1_MULT,
                                                    COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_1_SHIFT,
                                                    COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_2_OFFSET,
                                                    COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_2_MULT,
                                                    COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_INPUT_2_SHIFT,
                                                    COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_LEFT_SHIFT);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_bool(output,
                                       comparison_less_equal_broadcast_hw_s16_output_ref,
                                       COMPARISON_LESS_EQUAL_BROADCAST_HW_S16_DST_SIZE));
}

// Regression for the NHWC broadcast walk: both operands have width 1, so each row of input 1 is a
// single element that must advance with the output row. The previous walk left input 1 on row 0
// and returned SUCCESS with rows 1..h-1 wrong. Identity requantization: offset 0, unit multiplier.
void comparison_broadcast_row_scalar_s16_arm_comparison_s16(void)
{
    const int16_t input_1[3] = {10, 20, 30};
    const int16_t input_2[12] = {1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24};
    const bool expected_greater[12] = {true, true, true, true, true, true, true, true, true, true, true, true};
    const bool expected_swapped[12] = {
        false, false, false, false, false, false, false, false, false, false, false, false};
    bool output[12] = {0};
    cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_1_dims = {1, 3, 1, 1};
    const cmsis_nn_dims input_2_dims = {1, 3, 1, 4};
    const cmsis_nn_dims output_dims = {1, 3, 1, 4};
    const int32_t unit_mult = 0x7fffffff;

    // input_1 > input_2 for every element: 10 > 1..4, 20 > 11..14, 30 > 21..24
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_comparison_s16(&ctx,
                                         input_1,
                                         &input_1_dims,
                                         input_2,
                                         &input_2_dims,
                                         output,
                                         &output_dims,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         ARM_COMPARE_GREATER));
    TEST_ASSERT_TRUE(validate_bool(output, expected_greater, 12));

    // swapped operands: input_2 > input_1 is false everywhere; the broken walk reported row 1 as true
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_comparison_s16(&ctx,
                                         input_2,
                                         &input_2_dims,
                                         input_1,
                                         &input_1_dims,
                                         output,
                                         &output_dims,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         ARM_COMPARE_GREATER));
    TEST_ASSERT_TRUE(validate_bool(output, expected_swapped, 12));
}

void comparison_arg_error_s16_arm_comparison_s16(void)
{
    int16_t data[12] = {0};
    bool output[12] = {0};
    cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims dims_3x1 = {1, 3, 1, 1};
    const cmsis_nn_dims dims_2x4 = {1, 2, 1, 4};
    const cmsis_nn_dims dims_3x4 = {1, 3, 1, 4};
    const int32_t unit_mult = 0x7fffffff;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_comparison_s16(&ctx,
                                         data,
                                         &dims_3x1,
                                         data,
                                         &dims_2x4,
                                         output,
                                         &dims_3x4,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         ARM_COMPARE_EQUAL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_comparison_s16(&ctx,
                                         data,
                                         &dims_3x1,
                                         data,
                                         &dims_3x4,
                                         output,
                                         &dims_3x1,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         ARM_COMPARE_EQUAL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_comparison_s16(&ctx,
                                         NULL,
                                         &dims_3x1,
                                         data,
                                         &dims_3x4,
                                         output,
                                         &dims_3x4,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         unit_mult,
                                         0,
                                         0,
                                         ARM_COMPARE_EQUAL));
}
