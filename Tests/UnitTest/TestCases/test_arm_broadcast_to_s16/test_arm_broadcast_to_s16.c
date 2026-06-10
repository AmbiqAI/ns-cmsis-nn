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

#include "../TestData/broadcast_to_2d_s16/test_data.h"
#include "../TestData/broadcast_to_mid_dim_s16/test_data.h"
#include "../TestData/broadcast_to_scalar_cube_s16/test_data.h"
#include "../Utils/validate.h"

void broadcast_to_2d_s16_arm_broadcast_to_s16(void)
{
    const int32_t input_shape[] = BROADCAST_TO_2D_S16_INPUT_SHAPE;
    const int32_t output_shape[] = BROADCAST_TO_2D_S16_OUTPUT_SHAPE;
    const cmsis_nn_broadcast_to_params params = {
        .rank = BROADCAST_TO_2D_S16_RANK,
        .input_shape = input_shape,
        .output_shape = output_shape,
    };
    int16_t output[BROADCAST_TO_2D_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_broadcast_to_s16(broadcast_to_2d_s16_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, broadcast_to_2d_s16_output, BROADCAST_TO_2D_S16_OUTPUT_SIZE));
}

void broadcast_to_scalar_cube_s16_arm_broadcast_to_s16(void)
{
    const int32_t input_shape[] = BROADCAST_TO_SCALAR_CUBE_S16_INPUT_SHAPE;
    const int32_t output_shape[] = BROADCAST_TO_SCALAR_CUBE_S16_OUTPUT_SHAPE;
    const cmsis_nn_broadcast_to_params params = {
        .rank = BROADCAST_TO_SCALAR_CUBE_S16_RANK,
        .input_shape = input_shape,
        .output_shape = output_shape,
    };
    int16_t output[BROADCAST_TO_SCALAR_CUBE_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result =
        arm_broadcast_to_s16(broadcast_to_scalar_cube_s16_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, broadcast_to_scalar_cube_s16_output, BROADCAST_TO_SCALAR_CUBE_S16_OUTPUT_SIZE));
}

void broadcast_to_mid_dim_s16_arm_broadcast_to_s16(void)
{
    const int32_t input_shape[] = BROADCAST_TO_MID_DIM_S16_INPUT_SHAPE;
    const int32_t output_shape[] = BROADCAST_TO_MID_DIM_S16_OUTPUT_SHAPE;
    const cmsis_nn_broadcast_to_params params = {
        .rank = BROADCAST_TO_MID_DIM_S16_RANK,
        .input_shape = input_shape,
        .output_shape = output_shape,
    };
    int16_t output[BROADCAST_TO_MID_DIM_S16_OUTPUT_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_broadcast_to_s16(broadcast_to_mid_dim_s16_input_tensor, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, broadcast_to_mid_dim_s16_output, BROADCAST_TO_MID_DIM_S16_OUTPUT_SIZE));
}