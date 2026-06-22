/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/where_3d_sparse_s16/test_data.h"
#include "../TestData/where_all_true_2d_s16/test_data.h"
#include "../TestData/where_2d_s16/test_data.h"
#include "../Utils/validate.h"

void where_2d_s16_arm_where_s16(void)
{
    const int32_t shape[] = WHERE_2D_S16_SHAPE;
    const cmsis_nn_where_params params = {
        .rank = WHERE_2D_S16_RANK,
        .shape = shape,
    };
    const int32_t expected_output_len = (int32_t)(sizeof(where_2d_s16_output) / sizeof(where_2d_s16_output[0]));
    int64_t output[WHERE_2D_S16_COND_SIZE * WHERE_2D_S16_RANK] = {0};
    int32_t num_true = 0;

    const arm_cmsis_nn_status result = arm_where_s16(where_2d_s16_condition_input_tensor, &params, output, &num_true);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL(expected_output_len / WHERE_2D_S16_RANK, num_true);
    TEST_ASSERT_TRUE(validate_s64(output, where_2d_s16_output, expected_output_len));
}

void where_all_true_2d_s16_arm_where_s16(void)
{
    const int32_t shape[] = WHERE_ALL_TRUE_2D_S16_SHAPE;
    const cmsis_nn_where_params params = {
        .rank = WHERE_ALL_TRUE_2D_S16_RANK,
        .shape = shape,
    };
    const int32_t expected_output_len = (int32_t)(sizeof(where_all_true_2d_s16_output) / sizeof(where_all_true_2d_s16_output[0]));
    int64_t output[WHERE_ALL_TRUE_2D_S16_COND_SIZE * WHERE_ALL_TRUE_2D_S16_RANK] = {0};
    int32_t num_true = 0;

    const arm_cmsis_nn_status result = arm_where_s16(where_all_true_2d_s16_condition_input_tensor, &params, output, &num_true);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL(expected_output_len / WHERE_ALL_TRUE_2D_S16_RANK, num_true);
    TEST_ASSERT_TRUE(validate_s64(output, where_all_true_2d_s16_output, expected_output_len));
}

void where_3d_sparse_s16_arm_where_s16(void)
{
    const int32_t shape[] = WHERE_3D_SPARSE_S16_SHAPE;
    const cmsis_nn_where_params params = {
        .rank = WHERE_3D_SPARSE_S16_RANK,
        .shape = shape,
    };
    const int32_t expected_output_len = (int32_t)(sizeof(where_3d_sparse_s16_output) / sizeof(where_3d_sparse_s16_output[0]));
    int64_t output[WHERE_3D_SPARSE_S16_COND_SIZE * WHERE_3D_SPARSE_S16_RANK] = {0};
    int32_t num_true = 0;

    const arm_cmsis_nn_status result = arm_where_s16(where_3d_sparse_s16_condition_input_tensor, &params, output, &num_true);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL(expected_output_len / WHERE_3D_SPARSE_S16_RANK, num_true);
    TEST_ASSERT_TRUE(validate_s64(output, where_3d_sparse_s16_output, expected_output_len));
}