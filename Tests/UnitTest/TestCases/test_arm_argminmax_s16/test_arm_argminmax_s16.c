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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "arm_nnfunctions.h"
#include "unity.h"

static void assert_equal_int32(const int32_t *expected, const int32_t *actual, int32_t length)
{
    for (int32_t i = 0; i < length; ++i)
    {
        TEST_ASSERT_EQUAL_INT32_MESSAGE(expected[i], actual[i], "Mismatch in arg (min/max) output");
    }
}

void argmax_axis_c_s16(void)
{
    const int16_t input[] = {
        100,   500,  -200,
        700,   300,   400,
        -100, -500,  -300,
        200,   200,   100
    };

    const cmsis_nn_dims dims = {
        .n = 1,
        .h = 2,
        .w = 2,
        .c = 3
    };

    int32_t output[4] = {0};
    const int32_t expected[4] = {1, 0, 0, 0};

    arm_cmsis_nn_status status = arm_argmax_s16(input, &dims, 3, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    assert_equal_int32(expected, output, 4);
}

void argmin_axis_c_s16(void)
{
    const int16_t input[] = {
        100,   500,  -200,
        700,   300,   400,
        -100, -500,  -300,
        200,   200,   100
    };

    const cmsis_nn_dims dims = {
        .n = 1,
        .h = 2,
        .w = 2,
        .c = 3
    };

    int32_t output[4] = {0};
    const int32_t expected[4] = {2, 1, 1, 2};

    arm_cmsis_nn_status status = arm_argmin_s16(input, &dims, 3, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    assert_equal_int32(expected, output, 4);
}

void argmax_axis_h_s16(void)
{
    const int16_t input[] = {
        300,
        -10,
        700,
        400,
        500,
        -10
    };

    const cmsis_nn_dims dims = {
        .n = 1,
        .h = 3,
        .w = 2,
        .c = 1
    };

    int32_t output[2] = {0};
    const int32_t expected[2] = {1, 1};

    arm_cmsis_nn_status status = arm_argmax_s16(input, &dims, 1, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    assert_equal_int32(expected, output, 2);
}

void argmin_axis_h_s16(void)
{
    const int16_t input[] = {
        300,
        -10,
        700,
        400,
        500,
        -10
    };

    const cmsis_nn_dims dims = {
        .n = 1,
        .h = 3,
        .w = 2,
        .c = 1
    };

    int32_t output[2] = {0};
    const int32_t expected[2] = {0, 0};

    arm_cmsis_nn_status status = arm_argmin_s16(input, &dims, 1, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    assert_equal_int32(expected, output, 2);
}

void argmax_axis_n_s16(void)
{
    const int16_t input[] = {
        100, 3000,
        500,   0
    };

    const cmsis_nn_dims dims = {
        .n = 2,
        .h = 1,
        .w = 1,
        .c = 2
    };

    int32_t output[2] = {0};
    const int32_t expected[2] = {1, 0};

    arm_cmsis_nn_status status = arm_argmax_s16(input, &dims, 0, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    assert_equal_int32(expected, output, 2);
}

void argmin_axis_n_s16(void)
{
    const int16_t input[] = {
        100, 3000,
        500,   0
    };

    const cmsis_nn_dims dims = {
        .n = 2,
        .h = 1,
        .w = 1,
        .c = 2
    };

    int32_t output[2] = {0};
    const int32_t expected[2] = {0, 1};

    arm_cmsis_nn_status status = arm_argmin_s16(input, &dims, 0, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    assert_equal_int32(expected, output, 2);
}

void arg_axis_invalid_s16(void)
{
    const int16_t input[] = {100, 200, 300, 400};
    const cmsis_nn_dims dims = {
        .n = 1,
        .h = 1,
        .w = 1,
        .c = 4
    };

    int32_t output[4] = {0};

    arm_cmsis_nn_status status_max = arm_argmax_s16(input, &dims, 5, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, status_max);

    arm_cmsis_nn_status status_min = arm_argmin_s16(input, &dims, -2, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, status_min);
}
