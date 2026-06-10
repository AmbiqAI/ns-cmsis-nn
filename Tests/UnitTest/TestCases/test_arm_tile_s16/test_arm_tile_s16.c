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

#include "../TestData/tile_1d_repeat_s16/test_data.h"
#include "../TestData/tile_2d_s16/test_data.h"
#include "../TestData/tile_3d_s16/test_data.h"
#include "../Utils/validate.h"

void tile_2d_s16_arm_tile_s16(void)
{
    const int32_t input_shape[] = TILE_2D_S16_INPUT_SHAPE;
    const int32_t multiples[] = TILE_2D_S16_MULTIPLES;
    const cmsis_nn_tile_params params = {
        .rank = TILE_2D_S16_RANK,
        .input_shape = input_shape,
        .multiples = multiples,
    };
    int16_t output[TILE_2D_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_tile_s16(tile_2d_s16_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, tile_2d_s16_output, TILE_2D_S16_OUTPUT_SIZE));
}

void tile_1d_repeat_s16_arm_tile_s16(void)
{
    const int32_t input_shape[] = TILE_1D_REPEAT_S16_INPUT_SHAPE;
    const int32_t multiples[] = TILE_1D_REPEAT_S16_MULTIPLES;
    const cmsis_nn_tile_params params = {
        .rank = TILE_1D_REPEAT_S16_RANK,
        .input_shape = input_shape,
        .multiples = multiples,
    };
    int16_t output[TILE_1D_REPEAT_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_tile_s16(tile_1d_repeat_s16_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, tile_1d_repeat_s16_output, TILE_1D_REPEAT_S16_OUTPUT_SIZE));
}

void tile_3d_s16_arm_tile_s16(void)
{
    const int32_t input_shape[] = TILE_3D_S16_INPUT_SHAPE;
    const int32_t multiples[] = TILE_3D_S16_MULTIPLES;
    const cmsis_nn_tile_params params = {
        .rank = TILE_3D_S16_RANK,
        .input_shape = input_shape,
        .multiples = multiples,
    };
    int16_t output[TILE_3D_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_tile_s16(tile_3d_s16_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, tile_3d_s16_output, TILE_3D_S16_OUTPUT_SIZE));
}