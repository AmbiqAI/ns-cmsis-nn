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

#include "../TestData/scatter_nd_1d_s16/test_data.h"

void scatter_nd_1d_arm_scatter_nd_s16(void)
{
    int16_t output[SCATTER_ND_1D_S16_OUTPUT_SIZE] = {0};

    const cmsis_nn_scatter_nd_params params = {
        .num_updates = SCATTER_ND_1D_S16_NUM_UPDATES,
        .index_depth = SCATTER_ND_1D_S16_INDEX_DEPTH,
        .slice_size = SCATTER_ND_1D_S16_SLICE_SIZE,
        .output_size = SCATTER_ND_1D_S16_OUTPUT_SIZE,
        .output_strides = scatter_nd_1d_s16_output_strides,
    };

    const arm_cmsis_nn_status result = arm_scatter_nd_s16(
        scatter_nd_1d_s16_indices, scatter_nd_1d_s16_updates, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(scatter_nd_1d_s16_output, output, SCATTER_ND_1D_S16_OUTPUT_SIZE);
}
