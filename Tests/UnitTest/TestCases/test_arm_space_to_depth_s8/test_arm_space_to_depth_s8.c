/*
 * SPDX-FileCopyrightText: Copyright (c) 2025
 * Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "unity.h"

#include <arm_nnfunctions.h>          // must declare arm_space_to_depth_s8(...)
#include <arm_nnsupportfunctions.h>

#include "../Utils/validate.h"         // validate_tol_s8

/* Per-test generated data */
#include "../TestData/space_to_depth_bs2_basic_s8/test_data.h"
#include "../TestData/space_to_depth_bs4_small_s8/test_data.h"
#include "../TestData/space_to_depth_bs2_batch_s8/test_data.h"
#include "../TestData/space_to_depth_bs2_channels_s8/test_data.h"
#include "../TestData/space_to_depth_bs2_rectangular_s8/test_data.h"
#include "../TestData/space_to_depth_min_spatial_s8/test_data.h"

#ifndef REPEAT_NUM
#define REPEAT_NUM (1)
#endif

/*
 * NAME_UP:  UPPER_SNAKE prefix used by generated #defines, e.g.
 *           SPACE_TO_DEPTH_BS2_BASIC_S8_*
 * name_lc:  lower_snake prefix used by generated arrays, e.g.
 *           space_to_depth_bs2_basic_s8_input_tensor / _output
 */
#define GEN_S2D_TEST(NAME_UP, name_lc)                                                        \
    void name_lc##_arm_space_to_depth_s8(void)                                                \
    {                                                                                         \
        /* Dimensions from generated macros */                                                \
        const int in_dim_arr[4]  = NAME_UP##_IN_DIM;                                          \
        const int out_dim_arr[4] = NAME_UP##_OUT_DIM;                                         \
                                                                                              \
        cmsis_nn_dims input_dims  = {in_dim_arr[0],  in_dim_arr[1],  in_dim_arr[2],  in_dim_arr[3]};  \
        cmsis_nn_dims output_dims = {out_dim_arr[0], out_dim_arr[1], out_dim_arr[2], out_dim_arr[3]}; \
                                                                                              \
        const int32_t block_size = NAME_UP##_BLOCK_SIZE;                                      \
        const int8_t *input_data = name_lc##_input_tensor;                                    \
                                                                                              \
        int8_t output[NAME_UP##_OUTPUT_SIZE];                                                 \
        arm_memset_s8(output, 0, (uint32_t)NAME_UP##_OUTPUT_SIZE);                            \
                                                                                              \
        const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;                            \
                                                                                              \
        for (int r = 0; r < REPEAT_NUM; ++r)                                                  \
        {                                                                                     \
            const arm_cmsis_nn_status res =                                                   \
                arm_space_to_depth_s8(                                                        \
                    input_data,                                                               \
                    &input_dims,                                                              \
                    block_size,                                                               \
                    output,                                                                   \
                    &output_dims);                                                            \
            TEST_ASSERT_EQUAL(expected, res);                                                 \
        }                                                                                     \
                                                                                              \
        /* Pure data rearrangement → exact match expected */                                  \
        TEST_ASSERT_TRUE(                                                                     \
            validate_tol_s8(output, name_lc##_output, NAME_UP##_OUTPUT_SIZE, 0));             \
    }

/* Instantiate tests for all six datasets */
GEN_S2D_TEST(SPACE_TO_DEPTH_BS2_BASIC_S8,        space_to_depth_bs2_basic_s8)
GEN_S2D_TEST(SPACE_TO_DEPTH_BS4_SMALL_S8,        space_to_depth_bs4_small_s8)
GEN_S2D_TEST(SPACE_TO_DEPTH_BS2_BATCH_S8,        space_to_depth_bs2_batch_s8)
GEN_S2D_TEST(SPACE_TO_DEPTH_BS2_CHANNELS_S8,     space_to_depth_bs2_channels_s8)
GEN_S2D_TEST(SPACE_TO_DEPTH_BS2_RECTANGULAR_S8,  space_to_depth_bs2_rectangular_s8)
GEN_S2D_TEST(SPACE_TO_DEPTH_MIN_SPATIAL_S8,      space_to_depth_min_spatial_s8)
