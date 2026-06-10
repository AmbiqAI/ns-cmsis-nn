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

#include "../TestData/tile_1d_repeat_s8/test_data.h"
#include "../TestData/tile_2d_s8/test_data.h"
#include "../TestData/tile_3d_s8/test_data.h"
#include "../Utils/validate.h"

void tile_2d_s8_arm_tile_s8(void)
{
    const int32_t input_shape[] = TILE_2D_S8_INPUT_SHAPE;
    const int32_t multiples[] = TILE_2D_S8_MULTIPLES;
    const cmsis_nn_tile_params params = {
        .rank = TILE_2D_S8_RANK,
        .input_shape = input_shape,
        .multiples = multiples,
    };
    int8_t output[TILE_2D_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_tile_s8(tile_2d_s8_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, tile_2d_s8_output, TILE_2D_S8_OUTPUT_SIZE));
}

void tile_1d_repeat_s8_arm_tile_s8(void)
{
    const int32_t input_shape[] = TILE_1D_REPEAT_S8_INPUT_SHAPE;
    const int32_t multiples[] = TILE_1D_REPEAT_S8_MULTIPLES;
    const cmsis_nn_tile_params params = {
        .rank = TILE_1D_REPEAT_S8_RANK,
        .input_shape = input_shape,
        .multiples = multiples,
    };
    int8_t output[TILE_1D_REPEAT_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_tile_s8(tile_1d_repeat_s8_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, tile_1d_repeat_s8_output, TILE_1D_REPEAT_S8_OUTPUT_SIZE));
}

void tile_3d_s8_arm_tile_s8(void)
{
    const int32_t input_shape[] = TILE_3D_S8_INPUT_SHAPE;
    const int32_t multiples[] = TILE_3D_S8_MULTIPLES;
    const cmsis_nn_tile_params params = {
        .rank = TILE_3D_S8_RANK,
        .input_shape = input_shape,
        .multiples = multiples,
    };
    int8_t output[TILE_3D_S8_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_tile_s8(tile_3d_s8_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, tile_3d_s8_output, TILE_3D_S8_OUTPUT_SIZE));
}