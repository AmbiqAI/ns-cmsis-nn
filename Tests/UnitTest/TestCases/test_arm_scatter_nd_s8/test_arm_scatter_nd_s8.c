/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "arm_nn_types.h"
#include "unity.h"

#include "../TestData/scatter_nd_1d_s8/test_data.h"

void scatter_nd_1d_arm_scatter_nd_s8(void)
{
    int8_t output[SCATTER_ND_1D_S8_OUTPUT_SIZE] = {0};

    const cmsis_nn_scatter_nd_params params = {
        .num_updates = SCATTER_ND_1D_S8_NUM_UPDATES,
        .index_depth = SCATTER_ND_1D_S8_INDEX_DEPTH,
        .slice_size = SCATTER_ND_1D_S8_SLICE_SIZE,
        .output_size = SCATTER_ND_1D_S8_OUTPUT_SIZE,
        .output_strides = scatter_nd_1d_s8_output_strides,
    };

    const arm_cmsis_nn_status result = arm_scatter_nd_s8(
        scatter_nd_1d_s8_indices, scatter_nd_1d_s8_updates, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT8_ARRAY(scatter_nd_1d_s8_output, output, SCATTER_ND_1D_S8_OUTPUT_SIZE);
}
