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

#include "../TestData/tile_2d_s16/test_data.h"

void tile_2d_arm_tile_s16(void)
{
    int16_t output[TILE_2D_S16_OUTPUT_SIZE] = {0};

    const cmsis_nn_tile_params params = {
        .rank = TILE_2D_S16_RANK,
        .input_shape = tile_2d_s16_input_shape,
        .multiples = tile_2d_s16_multiples,
    };

    const arm_cmsis_nn_status result = arm_tile_s16(tile_2d_s16_input, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(tile_2d_s16_output, output, TILE_2D_S16_OUTPUT_SIZE);
}
