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

static const int16_t gather_axis_c_input[24] = {
    0,   5,   10,  15,  20,  25,  30,  35,
    40,  45,  50,  55,  60,  65,  70,  75,
    80,  85,  90,  95,  100, 105, 110, 115};

static const int32_t gather_axis_c_indices[3] = {3, 0, 2};
static const int16_t gather_axis_c_expect[18] = {
    15, 0, 10,
    35, 20, 30,
    55, 40, 50,
    75, 60, 70,
    95, 80, 90,
    115, 100, 110};

static const int16_t gather_batch_input[8] = {0, 5, 10, 15, 20, 25, 30, 35};
static const int32_t gather_batch_indices[4] = {1, 3, 0, 2};
static const int16_t gather_batch_expect[4] = {5, 15, 20, 30};

void gather_axis_c_arm_gather_s16(void)
{
    const cmsis_nn_dims input_dims = {1, 2, 3, 4};
    const cmsis_nn_dims indices_dims = {3, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 2, 3, 3};
    const cmsis_nn_gather_params params = {
        .axis = 3,
        .batch_dims = 0,
        .input_rank = 4,
        .coords_rank = 1};

    int16_t output[18] = {0};

    const arm_cmsis_nn_status result = arm_gather_s16(
        gather_axis_c_input,
        &input_dims,
        gather_axis_c_indices,
        &indices_dims,
        &params,
        output,
        &output_dims);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, gather_axis_c_expect, 18));
}

void gather_batch_dims_arm_gather_s16(void)
{
    const cmsis_nn_dims input_dims = {2, 1, 1, 4};
    const cmsis_nn_dims indices_dims = {2, 2, 1, 1};
    const cmsis_nn_dims output_dims = {2, 1, 1, 2};
    const cmsis_nn_gather_params params = {
        .axis = 3,
        .batch_dims = 1,
        .input_rank = 4,
        .coords_rank = 2};

    int16_t output[4] = {0};

    const arm_cmsis_nn_status result = arm_gather_s16(
        gather_batch_input,
        &input_dims,
        gather_batch_indices,
        &indices_dims,
        &params,
        output,
        &output_dims);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, gather_batch_expect, 4));
}
