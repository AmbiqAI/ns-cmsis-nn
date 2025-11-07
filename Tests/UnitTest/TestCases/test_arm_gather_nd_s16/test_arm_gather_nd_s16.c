/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "../Utils/validate.h"
#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

static const int16_t gather_nd_elements_params[4] = {0, 100, 200, 300};
static const int32_t gather_nd_elements_indices[6] = {0, 0, 0, 1, 1, 0};
static const int16_t gather_nd_elements_expect[3] = {0, 100, 200};

static const int16_t gather_nd_slices_params[8] = {0, 11, 22, 33, 44, 55, 66, 77};
static const int32_t gather_nd_slices_indices[4] = {0, 0, 1, 1};
static const int16_t gather_nd_slices_expect[4] = {0, 11, 66, 77};

void gather_nd_elements_arm_gather_nd_s16(void)
{
    const cmsis_nn_dims params_dims = {2, 2, 1, 1};
    const cmsis_nn_dims indices_dims = {3, 2, 1, 1};
    const cmsis_nn_dims output_dims = {3, 1, 1, 1};
    const cmsis_nn_gather_nd_params params = {
        .params_rank = 2,
        .indices_rank = 2,
        .batch_dims = 0};

    int16_t output[3] = {0};

    const arm_cmsis_nn_status result = arm_gather_nd_s16(
        gather_nd_elements_params,
        &params_dims,
        gather_nd_elements_indices,
        &indices_dims,
        &params,
        output,
        &output_dims);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, gather_nd_elements_expect, 3));
}

void gather_nd_slices_arm_gather_nd_s16(void)
{
    const cmsis_nn_dims params_dims = {2, 2, 2, 1};
    const cmsis_nn_dims indices_dims = {2, 2, 1, 1};
    const cmsis_nn_dims output_dims = {2, 2, 1, 1};
    const cmsis_nn_gather_nd_params params = {
        .params_rank = 3,
        .indices_rank = 2,
        .batch_dims = 0};

    int16_t output[4] = {0};

    const arm_cmsis_nn_status result = arm_gather_nd_s16(
        gather_nd_slices_params,
        &params_dims,
        gather_nd_slices_indices,
        &indices_dims,
        &params,
        output,
        &output_dims);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, gather_nd_slices_expect, 4));
}
