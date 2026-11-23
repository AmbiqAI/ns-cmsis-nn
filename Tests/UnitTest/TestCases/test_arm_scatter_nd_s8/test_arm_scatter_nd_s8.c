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

#include "../TestData/scatter_nd_1d_s8/test_data.h"
#include "../TestData/scatter_nd_points_2d_s8/test_data.h"
#include "../TestData/scatter_nd_rows_s8/test_data.h"
#include "../Utils/validate.h"

void scatter_nd_1d_s8_arm_scatter_nd_s8(void)
{
    const int32_t output_strides[] = SCATTER_ND_1D_S8_OUTPUT_STRIDES;
    const cmsis_nn_scatter_nd_params params = {
        .num_updates = SCATTER_ND_1D_S8_NUM_UPDATES,
        .index_depth = SCATTER_ND_1D_S8_INDEX_DEPTH,
        .slice_size = SCATTER_ND_1D_S8_SLICE_SIZE,
        .output_size = SCATTER_ND_1D_S8_OUTPUT_SIZE,
        .output_strides = output_strides,
    };
    int8_t output[SCATTER_ND_1D_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result =
        arm_scatter_nd_s8(scatter_nd_1d_s8_indices, scatter_nd_1d_s8_updates_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, scatter_nd_1d_s8_output, SCATTER_ND_1D_S8_OUTPUT_SIZE));
}

void scatter_nd_rows_s8_arm_scatter_nd_s8(void)
{
    const int32_t output_strides[] = SCATTER_ND_ROWS_S8_OUTPUT_STRIDES;
    const cmsis_nn_scatter_nd_params params = {
        .num_updates = SCATTER_ND_ROWS_S8_NUM_UPDATES,
        .index_depth = SCATTER_ND_ROWS_S8_INDEX_DEPTH,
        .slice_size = SCATTER_ND_ROWS_S8_SLICE_SIZE,
        .output_size = SCATTER_ND_ROWS_S8_OUTPUT_SIZE,
        .output_strides = output_strides,
    };
    int8_t output[SCATTER_ND_ROWS_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result =
        arm_scatter_nd_s8(scatter_nd_rows_s8_indices, scatter_nd_rows_s8_updates_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, scatter_nd_rows_s8_output, SCATTER_ND_ROWS_S8_OUTPUT_SIZE));
}

void scatter_nd_points_2d_s8_arm_scatter_nd_s8(void)
{
    const int32_t output_strides[] = SCATTER_ND_POINTS_2D_S8_OUTPUT_STRIDES;
    const cmsis_nn_scatter_nd_params params = {
        .num_updates = SCATTER_ND_POINTS_2D_S8_NUM_UPDATES,
        .index_depth = SCATTER_ND_POINTS_2D_S8_INDEX_DEPTH,
        .slice_size = SCATTER_ND_POINTS_2D_S8_SLICE_SIZE,
        .output_size = SCATTER_ND_POINTS_2D_S8_OUTPUT_SIZE,
        .output_strides = output_strides,
    };
    int8_t output[SCATTER_ND_POINTS_2D_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_scatter_nd_s8(scatter_nd_points_2d_s8_indices,
                                                         scatter_nd_points_2d_s8_updates_input_tensor,
                                                         &params,
                                                         output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, scatter_nd_points_2d_s8_output, SCATTER_ND_POINTS_2D_S8_OUTPUT_SIZE));
}

void scatter_nd_coordinate_bounds(void)
{
    const int32_t strides[] = {3, 1};
    const cmsis_nn_scatter_nd_params params = {
        .num_updates = 1, .index_depth = 2, .slice_size = 1, .output_size = 6, .output_strides = strides};
    const int8_t update = 7;
    int8_t output[6];
    const int32_t last[] = {1, 2};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_scatter_nd_s8(last, &update, &params, output));
    TEST_ASSERT_EQUAL_INT8(7, output[5]);

    const int32_t invalid[][2] = {{0, 3}, {1, -1}, {INT32_MAX, 0}, {1, INT32_MAX}};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_scatter_nd_s8(invalid[i], &update, &params, output));
    }
}

void scatter_nd_slice_bounds(void)
{
    const int32_t stride = 2;
    const int32_t indices[] = {1, 1};
    const int8_t updates[] = {1, 2, 3, 4};
    int8_t output[4];
    cmsis_nn_scatter_nd_params params = {
        .num_updates = 2, .index_depth = 1, .slice_size = 2, .output_size = 4, .output_strides = &stride};
    const int8_t expected[] = {0, 0, 4, 6};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_scatter_nd_s8(indices, updates, &params, output));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, 4);
    params.slice_size = 3;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_scatter_nd_s8(indices, updates, &params, output));
}

void scatter_nd_empty_slices(void)
{
    const int32_t strides[] = {0, 0};
    int32_t indices[] = {0, 1, 1, 0};
    const int8_t update = 7;
    int8_t output = 42;
    cmsis_nn_scatter_nd_params params = {
        .num_updates = 2, .index_depth = 2, .slice_size = 0, .output_size = 0, .output_strides = strides};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_scatter_nd_s8(indices, &update, &params, &output));
    indices[3] = -1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_scatter_nd_s8(indices, &update, &params, &output));
    indices[3] = 0;
    params.output_size = 1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_scatter_nd_s8(indices, &update, &params, &output));
    TEST_ASSERT_EQUAL_INT8(42, output);
}
