/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "unity.h"
#include <arm_nnfunctions.h>

#include "../Utils/validate.h"

static const int32_t prelu_identity_multiplier = 1073741824;
static const int32_t prelu_identity_shift = 1;

void prelu_scalar_alpha_s16_arm_prelu_s16(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 1, 4};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 4};
    const int16_t input_data[] = {-4, -1, 2, 5};
    const int16_t alpha_data[] = {3};
    const int16_t expected[] = {-12, -3, 2, 5};
    int16_t output[4] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s16(&input_dims,
                      input_data,
                      &alpha_dims,
                      alpha_data,
                      0,
                      0,
                      0,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      &output_dims,
                      output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, expected, 4));
}

void prelu_per_channel_s16_arm_prelu_s16(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 1, 4};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 4};
    const cmsis_nn_dims output_dims = {1, 1, 1, 4};
    const int16_t input_data[] = {-4, -1, 2, 5};
    const int16_t alpha_data[] = {3, 2, 1, 4};
    const int16_t expected[] = {-12, -2, 2, 5};
    int16_t output[4] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s16(&input_dims,
                      input_data,
                      &alpha_dims,
                      alpha_data,
                      0,
                      0,
                      0,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      &output_dims,
                      output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, expected, 4));
}

void prelu_output_shape_mismatch_s16_arm_prelu_s16(void)
{
    const cmsis_nn_dims input_dims = {1, 1, 1, 2};
    const cmsis_nn_dims alpha_dims = {1, 1, 1, 2};
    const cmsis_nn_dims output_dims = {1, 1, 1, 3};
    const int16_t input_data[] = {-2, 4};
    const int16_t alpha_data[] = {2, 2};
    int16_t output[3] = {0};

    const arm_cmsis_nn_status result =
        arm_prelu_s16(&input_dims,
                      input_data,
                      &alpha_dims,
                      alpha_data,
                      0,
                      0,
                      0,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      prelu_identity_multiplier,
                      prelu_identity_shift,
                      &output_dims,
                      output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
}