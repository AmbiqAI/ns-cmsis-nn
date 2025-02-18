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
#include <stdio.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/transpose_s16_3dim/test_data.h"
#include "../TestData/transpose_s16_3dim2/test_data.h"
#include "../TestData/transpose_s16_chwn/test_data.h"
#include "../TestData/transpose_s16_default/test_data.h"
#include "../TestData/transpose_s16_matrix/test_data.h"
#include "../TestData/transpose_s16_nchw/test_data.h"
#include "../TestData/transpose_s16_ncwh/test_data.h"
#include "../TestData/transpose_s16_nhcw/test_data.h"
#include "../TestData/transpose_s16_nwhc/test_data.h"
#include "../TestData/transpose_s16_wchn/test_data.h"
#include "../Utils/validate.h"

void transpose_default_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_DEFAULT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_DEFAULT_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_DEFAULT_OUT_DIM;

    const int16_t *input_data = transpose_s16_default_input_tensor;
    const int16_t *const output_ref = transpose_s16_default_output;
    const int32_t output_ref_size = TRANSPOSE_S16_DEFAULT_SIZE;

    const uint32_t perm[TRANSPOSE_S16_DEFAULT_PERM_SIZE] = TRANSPOSE_S16_DEFAULT_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_DEFAULT_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_nhcw_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_NHCW_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_NHCW_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_NHCW_OUT_DIM;

    const int16_t *input_data = transpose_s16_nhcw_input_tensor;
    const int16_t *const output_ref = transpose_s16_nhcw_output;
    const int32_t output_ref_size = TRANSPOSE_S16_NHCW_SIZE;

    const uint32_t perm[TRANSPOSE_S16_NHCW_PERM_SIZE] = TRANSPOSE_S16_NHCW_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_NHCW_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_wchn_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_WCHN_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_WCHN_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_WCHN_OUT_DIM;

    const int16_t *input_data = transpose_s16_wchn_input_tensor;
    const int16_t *const output_ref = transpose_s16_wchn_output;
    const int32_t output_ref_size = TRANSPOSE_S16_WCHN_SIZE;

    const uint32_t perm[TRANSPOSE_S16_WCHN_PERM_SIZE] = TRANSPOSE_S16_WCHN_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_WCHN_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_nchw_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_NCHW_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_NCHW_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_NCHW_OUT_DIM;

    const int16_t *input_data = transpose_s16_nchw_input_tensor;
    const int16_t *const output_ref = transpose_s16_nchw_output;
    const int32_t output_ref_size = TRANSPOSE_S16_NCHW_SIZE;

    const uint32_t perm[TRANSPOSE_S16_NCHW_PERM_SIZE] = TRANSPOSE_S16_NCHW_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_NCHW_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_chwn_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_CHWN_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_CHWN_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_CHWN_OUT_DIM;

    const int16_t *input_data = transpose_s16_chwn_input_tensor;
    const int16_t *const output_ref = transpose_s16_chwn_output;
    const int32_t output_ref_size = TRANSPOSE_S16_CHWN_SIZE;

    const uint32_t perm[TRANSPOSE_S16_CHWN_PERM_SIZE] = TRANSPOSE_S16_CHWN_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_CHWN_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_matrix_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_MATRIX_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_MATRIX_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_MATRIX_OUT_DIM;

    const int16_t *input_data = transpose_s16_matrix_input_tensor;
    const int16_t *const output_ref = transpose_s16_matrix_output;
    const int32_t output_ref_size = TRANSPOSE_S16_MATRIX_SIZE;

    const uint32_t perm[TRANSPOSE_S16_MATRIX_PERM_SIZE] = TRANSPOSE_S16_MATRIX_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_MATRIX_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_ncwh_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_NCWH_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_NCWH_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_NCWH_OUT_DIM;

    const int16_t *input_data = transpose_s16_ncwh_input_tensor;
    const int16_t *const output_ref = transpose_s16_ncwh_output;
    const int32_t output_ref_size = TRANSPOSE_S16_NCWH_SIZE;

    const uint32_t perm[TRANSPOSE_S16_NCWH_PERM_SIZE] = TRANSPOSE_S16_NCWH_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_NCWH_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_nwhc_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_NWHC_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_NWHC_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_NWHC_OUT_DIM;

    const int16_t *input_data = transpose_s16_nwhc_input_tensor;
    const int16_t *const output_ref = transpose_s16_nwhc_output;
    const int32_t output_ref_size = TRANSPOSE_S16_NWHC_SIZE;

    const uint32_t perm[TRANSPOSE_S16_NWHC_PERM_SIZE] = TRANSPOSE_S16_NWHC_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_NWHC_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_3dim_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_3DIM_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_3DIM_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_3DIM_OUT_DIM;

    const int16_t *input_data = transpose_s16_3dim_input_tensor;
    const int16_t *const output_ref = transpose_s16_3dim_output;
    const int32_t output_ref_size = TRANSPOSE_S16_3DIM_SIZE;

    const uint32_t perm[TRANSPOSE_S16_3DIM_PERM_SIZE] = TRANSPOSE_S16_3DIM_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_3DIM_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}

void transpose_3dim2_arm_transpose_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output_data[TRANSPOSE_S16_3DIM2_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_S16_3DIM2_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_S16_3DIM2_OUT_DIM;

    const int16_t *input_data = transpose_s16_3dim2_input_tensor;
    const int16_t *const output_ref = transpose_s16_3dim2_output;
    const int32_t output_ref_size = TRANSPOSE_S16_3DIM2_SIZE;

    const uint32_t perm[TRANSPOSE_S16_3DIM2_PERM_SIZE] = TRANSPOSE_S16_3DIM2_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_S16_3DIM2_PERM_SIZE, perm};

    arm_cmsis_nn_status result = arm_transpose_s16(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output_data, output_ref, output_ref_size));
}
