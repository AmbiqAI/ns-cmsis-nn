/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/transpose_3dim/test_data.h"
#include "../TestData/transpose_3dim2/test_data.h"
#include "../TestData/transpose_chwn/test_data.h"
#include "../TestData/transpose_default/test_data.h"
#include "../TestData/transpose_matrix/test_data.h"
#include "../TestData/transpose_nchw/test_data.h"
#include "../TestData/transpose_ncwh/test_data.h"
#include "../TestData/transpose_nhcw/test_data.h"
#include "../TestData/transpose_nwhc/test_data.h"
#include "../TestData/transpose_wchn/test_data.h"
#include "../Utils/validate.h"

void transpose_default_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_DEFAULT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_DEFAULT_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_DEFAULT_OUT_DIM;

    const int8_t *input_data = transpose_default_input_tensor;
    const int8_t *const output_ref = transpose_default_output;
    const int32_t output_ref_size = TRANSPOSE_DEFAULT_SIZE;

    const uint32_t perm[TRANSPOSE_DEFAULT_PERM_SIZE] = TRANSPOSE_DEFAULT_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_DEFAULT_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_nhcw_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_NHCW_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_NHCW_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_NHCW_OUT_DIM;

    const int8_t *input_data = transpose_nhcw_input_tensor;
    const int8_t *const output_ref = transpose_nhcw_output;
    const int32_t output_ref_size = TRANSPOSE_NHCW_SIZE;

    const uint32_t perm[TRANSPOSE_NHCW_PERM_SIZE] = TRANSPOSE_NHCW_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_NHCW_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_wchn_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_WCHN_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_WCHN_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_WCHN_OUT_DIM;

    const int8_t *input_data = transpose_wchn_input_tensor;
    const int8_t *const output_ref = transpose_wchn_output;
    const int32_t output_ref_size = TRANSPOSE_WCHN_SIZE;

    const uint32_t perm[TRANSPOSE_WCHN_PERM_SIZE] = TRANSPOSE_WCHN_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_WCHN_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_nchw_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_NCHW_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_NCHW_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_NCHW_OUT_DIM;

    const int8_t *input_data = transpose_nchw_input_tensor;
    const int8_t *const output_ref = transpose_nchw_output;
    const int32_t output_ref_size = TRANSPOSE_NCHW_SIZE;

    const uint32_t perm[TRANSPOSE_NCHW_PERM_SIZE] = TRANSPOSE_NCHW_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_NCHW_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_chwn_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_CHWN_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_CHWN_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_CHWN_OUT_DIM;

    const int8_t *input_data = transpose_chwn_input_tensor;
    const int8_t *const output_ref = transpose_chwn_output;
    const int32_t output_ref_size = TRANSPOSE_CHWN_SIZE;

    const uint32_t perm[TRANSPOSE_CHWN_PERM_SIZE] = TRANSPOSE_CHWN_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_CHWN_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_matrix_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_MATRIX_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_MATRIX_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_MATRIX_OUT_DIM;

    const int8_t *input_data = transpose_matrix_input_tensor;
    const int8_t *const output_ref = transpose_matrix_output;
    const int32_t output_ref_size = TRANSPOSE_MATRIX_SIZE;

    const uint32_t perm[TRANSPOSE_MATRIX_PERM_SIZE] = TRANSPOSE_MATRIX_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_MATRIX_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_ncwh_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_NCWH_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_NCWH_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_NCWH_OUT_DIM;

    const int8_t *input_data = transpose_ncwh_input_tensor;
    const int8_t *const output_ref = transpose_ncwh_output;
    const int32_t output_ref_size = TRANSPOSE_NCWH_SIZE;

    const uint32_t perm[TRANSPOSE_NCWH_PERM_SIZE] = TRANSPOSE_NCWH_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_NCWH_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_nwhc_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_NWHC_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_NWHC_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_NWHC_OUT_DIM;

    const int8_t *input_data = transpose_nwhc_input_tensor;
    const int8_t *const output_ref = transpose_nwhc_output;
    const int32_t output_ref_size = TRANSPOSE_NWHC_SIZE;

    const uint32_t perm[TRANSPOSE_NWHC_PERM_SIZE] = TRANSPOSE_NWHC_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_NWHC_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_3dim_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_3DIM_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_3DIM_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_3DIM_OUT_DIM;

    const int8_t *input_data = transpose_3dim_input_tensor;
    const int8_t *const output_ref = transpose_3dim_output;
    const int32_t output_ref_size = TRANSPOSE_3DIM_SIZE;

    const uint32_t perm[TRANSPOSE_3DIM_PERM_SIZE] = TRANSPOSE_3DIM_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_3DIM_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

void transpose_3dim2_arm_transpose_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_3DIM2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_3DIM2_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_3DIM2_OUT_DIM;

    const int8_t *input_data = transpose_3dim2_input_tensor;
    const int8_t *const output_ref = transpose_3dim2_output;
    const int32_t output_ref_size = TRANSPOSE_3DIM2_SIZE;

    const uint32_t perm[TRANSPOSE_3DIM2_PERM_SIZE] = TRANSPOSE_3DIM2_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_3DIM2_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
}

// An output_dims that is not input_dims permuted by perm must be rejected before any write.
// see AmbiqAI/ns-cmsis-nn#443
void transpose_dims_mismatch_arm_transpose_s8(void)
{
    const int32_t buffer_size = 1 * 8 * 10 * 12;
    const int8_t poison = (int8_t)0x5a;
    static int8_t input_data[1 * 8 * 10 * 12];
    static int8_t output_data[1 * 8 * 10 * 12];

    const cmsis_nn_dims input_dims = {1, 8, 10, 12};
    const cmsis_nn_dims output_dims = {12, 8, 10, 1};
    const uint32_t perm[4] = {3, 2, 1, 0};
    const cmsis_nn_transpose_params transpose_params = {4, perm};

    for (int32_t i = 0; i < buffer_size; i++)
    {
        input_data[i] = (int8_t)(i & 0x7f);
        output_data[i] = poison;
    }

    arm_cmsis_nn_status result =
        arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    bool output_untouched = true;
    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}

// The 2-D path transposes unconditionally, so the identity permutation needs its own case.
// see AmbiqAI/ns-cmsis-nn#443
void transpose_2dim_identity_arm_transpose_s8(void)
{
    const int8_t input_data[6] = {1, 2, 3, 4, 5, 6};
    int8_t output_data[6] = {0};

    const cmsis_nn_dims input_dims = {2, 3, 1, 1};
    const cmsis_nn_dims output_dims = {2, 3, 1, 1};
    const uint32_t perm[2] = {0, 1};
    const cmsis_nn_transpose_params transpose_params = {2, perm};

    arm_cmsis_nn_status result =
        arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output_data, input_data, 6));
}

// A perm with a repeated axis is not a permutation; the dims agree, so only the repeat can reject it.
// see AmbiqAI/ns-cmsis-nn#443
void transpose_perm_duplicate_axis_arm_transpose_s8(void)
{
    const int32_t input_size = 2 * 2 * 3 * 4;
    const int32_t output_size = 2 * 2 * 2 * 3;
    const int8_t poison = (int8_t)0x5a;
    static int8_t input_data[2 * 2 * 3 * 4];
    static int8_t output_data[2 * 2 * 2 * 3];

    const cmsis_nn_dims input_dims = {2, 2, 3, 4};
    const cmsis_nn_dims output_dims = {2, 2, 2, 3};
    const uint32_t perm[4] = {0, 0, 1, 2};
    const cmsis_nn_transpose_params transpose_params = {4, perm};

    for (int32_t i = 0; i < input_size; i++)
    {
        input_data[i] = (int8_t)(i & 0x7f);
    }
    for (int32_t i = 0; i < output_size; i++)
    {
        output_data[i] = poison;
    }

    arm_cmsis_nn_status result =
        arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    bool output_untouched = true;
    for (int32_t i = 0; i < output_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}

// Only num_dims in [1, 4] is representable in cmsis_nn_dims. see AmbiqAI/ns-cmsis-nn#443
void transpose_num_dims_out_of_range_arm_transpose_s8(void)
{
    const int32_t buffer_size = 2 * 3 * 4 * 5;
    const int8_t poison = (int8_t)0x5a;
    static int8_t input_data[2 * 3 * 4 * 5];
    static int8_t output_data[2 * 3 * 4 * 5];

    const cmsis_nn_dims input_dims = {2, 3, 4, 5};
    const cmsis_nn_dims output_dims = {2, 3, 4, 5};
    const uint32_t perm[4] = {0, 1, 2, 3};
    const cmsis_nn_transpose_params too_few = {0, perm};
    const cmsis_nn_transpose_params too_many = {5, perm};

    for (int32_t i = 0; i < buffer_size; i++)
    {
        input_data[i] = (int8_t)(i & 0x7f);
        output_data[i] = poison;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &too_few));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &too_many));

    bool output_untouched = true;
    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}
// An extent of 0 agrees across the permutation, so only the extent check can reject it; without it the copy length
// wraps. see AmbiqAI/ns-cmsis-nn#443
void transpose_zero_extent_arm_transpose_s8(void)
{
    const int32_t buffer_size = 6;
    const int8_t poison = (int8_t)0x5a;
    int8_t input_data[6] = {1, 2, 3, 4, 5, 6};
    int8_t output_data[6];

    const cmsis_nn_dims input_dims = {2, 0, 1, 1};
    const cmsis_nn_dims output_dims = {0, 2, 1, 1};
    const uint32_t perm[2] = {1, 0};
    const cmsis_nn_transpose_params transpose_params = {2, perm};

    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_data[i] = poison;
    }

    arm_cmsis_nn_status result =
        arm_transpose_s8(input_data, output_data, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    bool output_untouched = true;
    for (int32_t i = 0; i < buffer_size; i++)
    {
        output_untouched = output_untouched && (output_data[i] == poison);
    }
    TEST_ASSERT_TRUE(output_untouched);
}
