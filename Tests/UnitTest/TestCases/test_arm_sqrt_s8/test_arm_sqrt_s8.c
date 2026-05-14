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

#include <math.h>
#include <stdint.h>

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/sqrt_long_row_s8/test_data.h"
#include "../TestData/sqrt_multi_batch_s8/test_data.h"
#include "../TestData/sqrt_small_tensor_s8/test_data.h"

#include "../Utils/validate.h"

static inline float clamp_f32(float v, float lo, float hi) { return (v < lo) ? lo : ((v > hi) ? hi : v); }

static void
make_sqrt_lut(float input_scale, int32_t input_zero_point, float output_scale, int32_t output_zero_point, int8_t *lut)
{
    for (int32_t i = -128; i < 128; ++i)
    {
        const float x = (i - input_zero_point) * input_scale;
        int32_t final_val = output_zero_point;

        if (x > 0.0f)
        {
            const float res = sqrtf(x);
            const float q = (res / output_scale) + output_zero_point;
            final_val = clamp_f32(q, -128, 127);
        }

        lut[(uint8_t)i] = (int8_t)final_val;
    }
}

void sqrt_small_tensor_s8_arm_sqrt_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_SMALL_TENSOR_S8_IN_DIM;
    const int8_t *input_data = sqrt_small_tensor_s8_input_tensor;
    int8_t output_data[SQRT_SMALL_TENSOR_S8_OUTPUT_LEN];
    int8_t sqrt_lut[256];

    make_sqrt_lut(SQRT_SMALL_TENSOR_S8_INPUT_SCALE,
                  SQRT_SMALL_TENSOR_S8_INPUT_ZERO_POINT,
                  SQRT_SMALL_TENSOR_S8_OUTPUT_SCALE,
                  SQRT_SMALL_TENSOR_S8_OUTPUT_ZERO_POINT,
                  sqrt_lut);

    arm_cmsis_nn_status result = arm_sqrt_s8(input_data, &input_dims, output_data, sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, sqrt_small_tensor_s8_output, SQRT_SMALL_TENSOR_S8_OUTPUT_LEN));
}

void sqrt_long_row_s8_arm_sqrt_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_LONG_ROW_S8_IN_DIM;
    const int8_t *input_data = sqrt_long_row_s8_input_tensor;
    int8_t output_data[SQRT_LONG_ROW_S8_OUTPUT_LEN];
    int8_t sqrt_lut[256];

    make_sqrt_lut(SQRT_LONG_ROW_S8_INPUT_SCALE,
                  SQRT_LONG_ROW_S8_INPUT_ZERO_POINT,
                  SQRT_LONG_ROW_S8_OUTPUT_SCALE,
                  SQRT_LONG_ROW_S8_OUTPUT_ZERO_POINT,
                  sqrt_lut);

    arm_cmsis_nn_status result = arm_sqrt_s8(input_data, &input_dims, output_data, sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, sqrt_long_row_s8_output, SQRT_LONG_ROW_S8_OUTPUT_LEN));
}

void sqrt_multi_batch_s8_arm_sqrt_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const cmsis_nn_dims input_dims = SQRT_MULTI_BATCH_S8_IN_DIM;
    const int8_t *input_data = sqrt_multi_batch_s8_input_tensor;
    int8_t output_data[SQRT_MULTI_BATCH_S8_OUTPUT_LEN];
    int8_t sqrt_lut[256];

    make_sqrt_lut(SQRT_MULTI_BATCH_S8_INPUT_SCALE,
                  SQRT_MULTI_BATCH_S8_INPUT_ZERO_POINT,
                  SQRT_MULTI_BATCH_S8_OUTPUT_SCALE,
                  SQRT_MULTI_BATCH_S8_OUTPUT_ZERO_POINT,
                  sqrt_lut);

    arm_cmsis_nn_status result = arm_sqrt_s8(input_data, &input_dims, output_data, sqrt_lut);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, sqrt_multi_batch_s8_output, SQRT_MULTI_BATCH_S8_OUTPUT_LEN));
}
