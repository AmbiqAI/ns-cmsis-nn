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

#include "../TestData/scatter_nd_1d_s16/test_data.h"
#include "../TestData/scatter_nd_points_2d_s16/test_data.h"
#include "../TestData/scatter_nd_rows_s16/test_data.h"
#include "../Utils/validate.h"

void scatter_nd_1d_s16_arm_scatter_nd_s16(void)
{
    const int32_t output_strides[] = SCATTER_ND_1D_S16_OUTPUT_STRIDES;
    const cmsis_nn_scatter_nd_params params = {
        .num_updates = SCATTER_ND_1D_S16_NUM_UPDATES,
        .index_depth = SCATTER_ND_1D_S16_INDEX_DEPTH,
        .slice_size = SCATTER_ND_1D_S16_SLICE_SIZE,
        .output_size = SCATTER_ND_1D_S16_OUTPUT_SIZE,
        .output_strides = output_strides,
    };
    int16_t output[SCATTER_ND_1D_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result =
        arm_scatter_nd_s16(scatter_nd_1d_s16_indices, scatter_nd_1d_s16_updates_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, scatter_nd_1d_s16_output, SCATTER_ND_1D_S16_OUTPUT_SIZE));
}

void scatter_nd_rows_s16_arm_scatter_nd_s16(void)
{
    const int32_t output_strides[] = SCATTER_ND_ROWS_S16_OUTPUT_STRIDES;
    const cmsis_nn_scatter_nd_params params = {
        .num_updates = SCATTER_ND_ROWS_S16_NUM_UPDATES,
        .index_depth = SCATTER_ND_ROWS_S16_INDEX_DEPTH,
        .slice_size = SCATTER_ND_ROWS_S16_SLICE_SIZE,
        .output_size = SCATTER_ND_ROWS_S16_OUTPUT_SIZE,
        .output_strides = output_strides,
    };
    int16_t output[SCATTER_ND_ROWS_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result =
        arm_scatter_nd_s16(scatter_nd_rows_s16_indices, scatter_nd_rows_s16_updates_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, scatter_nd_rows_s16_output, SCATTER_ND_ROWS_S16_OUTPUT_SIZE));
}

void scatter_nd_points_2d_s16_arm_scatter_nd_s16(void)
{
    const int32_t output_strides[] = SCATTER_ND_POINTS_2D_S16_OUTPUT_STRIDES;
    const cmsis_nn_scatter_nd_params params = {
        .num_updates = SCATTER_ND_POINTS_2D_S16_NUM_UPDATES,
        .index_depth = SCATTER_ND_POINTS_2D_S16_INDEX_DEPTH,
        .slice_size = SCATTER_ND_POINTS_2D_S16_SLICE_SIZE,
        .output_size = SCATTER_ND_POINTS_2D_S16_OUTPUT_SIZE,
        .output_strides = output_strides,
    };
    int16_t output[SCATTER_ND_POINTS_2D_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_scatter_nd_s16(scatter_nd_points_2d_s16_indices,
                                                          scatter_nd_points_2d_s16_updates_input_tensor,
                                                          &params,
                                                          output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, scatter_nd_points_2d_s16_output, SCATTER_ND_POINTS_2D_S16_OUTPUT_SIZE));
}