/*
 * SPDX-FileCopyrightText: Copyright (c) 2025
 * Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include "unity.h"

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>

#include "../Utils/validate.h"

/* Per-test generated data */
#include "../TestData/batch_to_space_nd_basic_2x2_s8/test_data.h"
#include "../TestData/batch_to_space_nd_nsquare_crops_s8/test_data.h"
#include "../TestData/batch_to_space_nd_batch_s8/test_data.h"
#include "../TestData/batch_to_space_nd_both_crops_s8/test_data.h"
#include "../TestData/batch_to_space_nd_bs1_identity_s8/test_data.h"

#ifndef REPEAT_NUM
#define REPEAT_NUM (1)
#endif

/* NAME_UP: upper-snake macro prefix (e.g. BATCH_TO_SPACE_ND_BASIC_2X2_S8)
 * name_lc: lower-snake array prefix (e.g. batch_to_space_nd_basic_2x2_s8)
 *
 * Expects these macros/arrays from each dataset header:
 *   NAME_UP_IN_DIM          -> {N,H,W,C}
 *   NAME_UP_OUT_DIM         -> {N,H,W,C}
 *   NAME_UP_BLOCK_SHAPE     -> {Hblock, Wblock}
 *   NAME_UP_CROPS           -> {top, left, bottom, right}
 *   NAME_UP_INPUT_SIZE, NAME_UP_OUTPUT_SIZE
 *   name_lc_input_tensor[], name_lc_output[]
 */
#define GEN_B2SND_TEST(NAME_UP, name_lc)                                                     \
    void name_lc##_arm_batch_to_space_nd_s8(void)                                            \
    {                                                                                        \
        const int32_t in_dim_arr[]      = NAME_UP##_IN_DIM;                                  \
        const int32_t out_dim_arr[]     = NAME_UP##_OUT_DIM;                                 \
        const int32_t blk_arr[]         = NAME_UP##_BLOCK_SHAPE;                             \
        const int32_t crop_arr[]        = NAME_UP##_CROPS;                                   \
                                                                                             \
        const cmsis_nn_dims input_dims  = { in_dim_arr[0],  in_dim_arr[1],                   \
                                            in_dim_arr[2],  in_dim_arr[3] };                 \
        const cmsis_nn_dims output_dims = { out_dim_arr[0], out_dim_arr[1],                  \
                                            out_dim_arr[2], out_dim_arr[3] };                \
        const cmsis_nn_tile block_shape = { .h = blk_arr[0], .w = blk_arr[1] };              \
        /* crop_dims packs {top,left,bottom,right} into {n,h,w,c} fields */                  \
        const cmsis_nn_dims crop_dims   = { crop_arr[0], crop_arr[1], crop_arr[2],           \
                                            crop_arr[3] };                                   \
                                                                                             \
        const int8_t *input_data = name_lc##_input_tensor;                                   \
        int8_t output[NAME_UP##_OUTPUT_SIZE];                                                \
        /* Not strictly required (kernel overwrites all), but keeps behavior deterministic */\
        arm_memset_s8(output, 0, (uint32_t)NAME_UP##_OUTPUT_SIZE);                           \
                                                                                             \
        for (int r = 0; r < REPEAT_NUM; ++r)                                                 \
        {                                                                                    \
            const arm_cmsis_nn_status res =                                                  \
                arm_batch_to_space_nd_s8(                                                    \
                    input_data, &input_dims, &block_shape, &crop_dims,                       \
                    output, &output_dims);                                                   \
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, res);                                    \
        }                                                                                    \
                                                                                             \
        TEST_ASSERT_TRUE(                                                                    \
            validate_tol_s8(output, name_lc##_output, NAME_UP##_OUTPUT_SIZE, 0));            \
    }

/* Instantiate tests */
GEN_B2SND_TEST(BATCH_TO_SPACE_ND_BASIC_2X2_S8,        batch_to_space_nd_basic_2x2_s8)
GEN_B2SND_TEST(BATCH_TO_SPACE_ND_NSQUARE_CROPS_S8,    batch_to_space_nd_nsquare_crops_s8)
GEN_B2SND_TEST(BATCH_TO_SPACE_ND_BATCH_S8,            batch_to_space_nd_batch_s8)
GEN_B2SND_TEST(BATCH_TO_SPACE_ND_BOTH_CROPS_S8,       batch_to_space_nd_both_crops_s8)
GEN_B2SND_TEST(BATCH_TO_SPACE_ND_BS1_IDENTITY_S8,     batch_to_space_nd_bs1_identity_s8)
