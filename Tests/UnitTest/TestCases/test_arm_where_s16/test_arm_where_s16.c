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

#include "../TestData/where_2d_s16/test_data.h"

void where_2d_arm_where_s16(void)
{
    int32_t output[WHERE_2D_S16_NUM_TRUE * WHERE_2D_S16_RANK] = {0};
    int32_t num_true = 0;

    const cmsis_nn_where_params params = {
        .rank = WHERE_2D_S16_RANK,
        .shape = where_2d_s16_shape,
    };

    const arm_cmsis_nn_status result = arm_where_s16(where_2d_s16_condition, &params, output, &num_true);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT32(WHERE_2D_S16_NUM_TRUE, num_true);
    TEST_ASSERT_EQUAL_INT32_ARRAY(where_2d_s16_output, output, WHERE_2D_S16_NUM_TRUE * WHERE_2D_S16_RANK);
}
