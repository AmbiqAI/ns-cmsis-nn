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

#include "../Utils/validate.h"

#include "../TestData/gather_nd_point_s8/test_data.h"
#include "../TestData/gather_nd_slice_hw_s8/test_data.h"
#include "../TestData/gather_nd_batch_dims_s8/test_data.h"
#include "../TestData/gather_nd_depth_s8/test_data.h"
#include "../TestData/gather_nd_reduce_rank_s8/test_data.h"

#define DEFINE_GATHER_ND_TEST(TEST_UPPER, TEST_LOWER)                                                                  \
    void TEST_LOWER##_arm_gather_nd_s8(void)                                                                           \
    {                                                                                                                  \
        const cmsis_nn_dims params_dims = {TEST_UPPER##_PARAMS_N,                                                      \
                                           TEST_UPPER##_PARAMS_H,                                                      \
                                           TEST_UPPER##_PARAMS_W,                                                      \
                                           TEST_UPPER##_PARAMS_C};                                                     \
        const cmsis_nn_dims indices_dims = {TEST_UPPER##_INDICES_N,                                                    \
                                            TEST_UPPER##_INDICES_H,                                                    \
                                            TEST_UPPER##_INDICES_W,                                                    \
                                            TEST_UPPER##_INDICES_C};                                                   \
        const cmsis_nn_dims output_dims = {TEST_UPPER##_OUTPUT_N,                                                      \
                                           TEST_UPPER##_OUTPUT_H,                                                      \
                                           TEST_UPPER##_OUTPUT_W,                                                      \
                                           TEST_UPPER##_OUTPUT_C};                                                     \
        const cmsis_nn_gather_nd_params params = {.params_rank = TEST_UPPER##_PARAMS_RANK,                             \
                                                  .indices_rank = TEST_UPPER##_INDICES_RANK,                           \
                                                  .batch_dims = TEST_UPPER##_BATCH_DIMS};                              \
                                                                                                                       \
        int8_t output[TEST_UPPER##_OUTPUT_SIZE] = {0};                                                                 \
                                                                                                                       \
        const arm_cmsis_nn_status result =                                                                              \
            arm_gather_nd_s8(TEST_LOWER##_input_tensor,                                                                \
                              &params_dims,                                                                            \
                              TEST_LOWER##_indices_tensor,                                                             \
                              &indices_dims,                                                                            \
                              &params,                                                                                \
                              output,                                                                                 \
                              &output_dims);                                                                           \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);                                                               \
        TEST_ASSERT_TRUE(validate(output, TEST_LOWER##_output, TEST_UPPER##_OUTPUT_SIZE));                             \
    }

DEFINE_GATHER_ND_TEST(GATHER_ND_POINT_S8, gather_nd_point_s8)
DEFINE_GATHER_ND_TEST(GATHER_ND_SLICE_HW_S8, gather_nd_slice_hw_s8)
DEFINE_GATHER_ND_TEST(GATHER_ND_BATCH_DIMS_S8, gather_nd_batch_dims_s8)
DEFINE_GATHER_ND_TEST(GATHER_ND_DEPTH_S8, gather_nd_depth_s8)
DEFINE_GATHER_ND_TEST(GATHER_ND_REDUCE_RANK_S8, gather_nd_reduce_rank_s8)
