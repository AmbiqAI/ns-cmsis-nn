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

#include "../TestData/mul/test_data.h"
#include "../TestData/elementwise_mul_1_s8/test_data.h"
#include "../TestData/elementwise_mul_2_s8/test_data.h"
#include "../Utils/validate.h"

void mul_arm_elementwise_mul_2_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[ELEMENTWISE_MUL_2_S8_DST_SIZE] = {0};

    const int8_t *input_data1 = elementwise_mul_2_s8_lhs_input_tensor;
    const int8_t *input_data2 = elementwise_mul_2_s8_rhs_input_tensor;
    
    const int32_t input_1_offset = ELEMENTWISE_MUL_2_S8_LHS_OFFSET;
    const int32_t input_2_offset = ELEMENTWISE_MUL_2_S8_RHS_OFFSET;

    const int32_t out_offset = ELEMENTWISE_MUL_2_S8_OUTPUT_OFFSET;
    const int32_t out_mult = ELEMENTWISE_MUL_2_S8_OUTPUT_MULTIPLIER;
    const int32_t out_shift = ELEMENTWISE_MUL_2_S8_OUTPUT_SHIFT;

    const int32_t out_activation_min = ELEMENTWISE_MUL_2_S8_ACTIVATION_MIN;
    const int32_t out_activation_max = ELEMENTWISE_MUL_2_S8_ACTIVATION_MAX;

    arm_cmsis_nn_status result = arm_elementwise_mul_s8(input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        ELEMENTWISE_MUL_2_S8_DST_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, elementwise_mul_2_s8_output, ELEMENTWISE_MUL_2_S8_DST_SIZE));
}

void mul_arm_elementwise_mul_1_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[ELEMENTWISE_MUL_1_S8_DST_SIZE] = {0};

    const int8_t *input_data1 = elementwise_mul_1_s8_lhs_input_tensor;
    const int8_t *input_data2 = elementwise_mul_1_s8_rhs_input_tensor;
    
    const int32_t input_1_offset = ELEMENTWISE_MUL_1_S8_LHS_OFFSET;
    const int32_t input_2_offset = ELEMENTWISE_MUL_1_S8_RHS_OFFSET;

    const int32_t out_offset = ELEMENTWISE_MUL_1_S8_OUTPUT_OFFSET;
    const int32_t out_mult = ELEMENTWISE_MUL_1_S8_OUTPUT_MULTIPLIER;
    const int32_t out_shift = ELEMENTWISE_MUL_1_S8_OUTPUT_SHIFT;

    const int32_t out_activation_min = ELEMENTWISE_MUL_1_S8_ACTIVATION_MIN;
    const int32_t out_activation_max = ELEMENTWISE_MUL_1_S8_ACTIVATION_MAX;

    arm_cmsis_nn_status result = arm_elementwise_mul_s8(input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        ELEMENTWISE_MUL_1_S8_DST_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, elementwise_mul_1_s8_output, ELEMENTWISE_MUL_1_S8_DST_SIZE));
}

void mul_arm_elementwise_mul_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[MUL_DST_SIZE] = {0};

    const int8_t *input_data1 = mul_input1;
    const int8_t *input_data2 = mul_input2;

    const int32_t input_1_offset = MUL_INPUT1_OFFSET;
    const int32_t input_2_offset = MUL_INPUT2_OFFSET;

    const int32_t out_offset = MUL_OUTPUT_OFFSET;
    const int32_t out_mult = MUL_OUTPUT_MULT;
    const int32_t out_shift = MUL_OUTPUT_SHIFT;

    const int32_t out_activation_min = MUL_OUT_ACTIVATION_MIN;
    const int32_t out_activation_max = MUL_OUT_ACTIVATION_MAX;

    arm_cmsis_nn_status result = arm_elementwise_mul_s8(input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        MUL_DST_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_output_ref, MUL_DST_SIZE));
}

/* A large effective multiplier drives products past the int16 range before the output offset is added. The result
   must saturate to the activation clamp in every lane, whether the element falls in an 8-element block or the tail. */
#define MUL_SAT_LEN (8 * 4 + 3)
static void mul_saturation_expected(const int8_t *in1,
                                    const int8_t *in2,
                                    int32_t in1_offset,
                                    int32_t in2_offset,
                                    int32_t out_offset,
                                    int32_t mult,
                                    int32_t shift,
                                    int32_t scalar_in1,
                                    int8_t *expected)
{
    for (int32_t i = 0; i < MUL_SAT_LEN; i++)
    {
        const int32_t a = (scalar_in1 ? in1[0] : in1[i]) + in1_offset;
        int32_t r = arm_nn_requantize(a * (in2[i] + in2_offset), mult, shift) + out_offset;
        r = r < -128 ? -128 : (r > 127 ? 127 : r);
        expected[i] = (int8_t)r;
    }
}

void mul_saturation_arm_elementwise_mul_s8(void)
{
    int8_t in1[MUL_SAT_LEN], in2[MUL_SAT_LEN], out[MUL_SAT_LEN], expected[MUL_SAT_LEN];
    for (int32_t i = 0; i < MUL_SAT_LEN; i++)
    {
        in1[i] = (int8_t)(i % 2 ? 127 - i : -128 + i);
        in2[i] = (int8_t)(i % 3 ? -128 + 2 * i : 127 - i);
    }
    const int32_t mult = 1 << 30, shift = 8, in1_offset = 1, in2_offset = -2;
    const int32_t out_offsets[] = {100, -100, 127, -128};
    for (size_t o = 0; o < sizeof(out_offsets) / sizeof(out_offsets[0]); o++)
    {
        for (int32_t scalar_in1 = 0; scalar_in1 < 2; scalar_in1++)
        {
            mul_saturation_expected(
                in1, in2, in1_offset, in2_offset, out_offsets[o], mult, shift, scalar_in1, expected);
            const arm_cmsis_nn_status status = scalar_in1
                ? arm_mul_scalar_s8(
                      in1, in2, in1_offset, in2_offset, out, out_offsets[o], mult, shift, -128, 127, MUL_SAT_LEN)
                : arm_elementwise_mul_s8(
                      in1, in2, in1_offset, in2_offset, out, out_offsets[o], mult, shift, -128, 127, MUL_SAT_LEN);
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
            TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, MUL_SAT_LEN);
        }
    }
}
