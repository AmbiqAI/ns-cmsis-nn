/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "unity.h"

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>

#include "../Utils/validate.h"

/* Per-test generated data */
#include "../TestData/space_to_batch_nd_basic_2x2_s8/test_data.h"
#include "../TestData/space_to_batch_nd_nsquare_pad_s8/test_data.h"
#include "../TestData/space_to_batch_nd_batch_s8/test_data.h"
#include "../TestData/space_to_batch_nd_both_pad_s8/test_data.h"
#include "../TestData/space_to_batch_nd_small_pad_s8/test_data.h"
#include "../TestData/space_to_batch_nd_bs1_identity_pad_s8/test_data.h"

#ifndef REPEAT_NUM
#define REPEAT_NUM (1)
#endif

/* NAME_UP: upper-snake macro prefix (e.g. SPACE_TO_BATCH_ND_BASIC_2X2_S8)
 * name_lc: lower-snake array prefix (e.g. space_to_batch_nd_basic_2x2_s8)
 *
 * Expects these macros/arrays from each dataset header:
 *   NAME_UP_IN_DIM          -> {N,H,W,C}
 *   NAME_UP_OUT_DIM         -> {N,H,W,C}
 *   NAME_UP_BLOCK_SHAPE     -> {Hblock, Wblock}
 *   NAME_UP_PADDINGS        -> {top, left, bottom, right}
 *   NAME_UP_INPUT_SIZE, NAME_UP_OUTPUT_SIZE
 *   NAME_UP_OUTPUT_OFFSET   -> int (output zero-point)
 *   name_lc_input_tensor[], name_lc_output[]
 */
#define GEN_S2BND_TEST(NAME_UP, name_lc)                                                       \
    void name_lc##_arm_space_to_batch_nd_s8(void)                                              \
    {                                                                                          \
        const int32_t in_dim_arr[]  = NAME_UP##_IN_DIM;                                        \
        const int32_t out_dim_arr[] = NAME_UP##_OUT_DIM;                                       \
        const int32_t blk_arr[]     = NAME_UP##_BLOCK_SHAPE;                                   \
        const int32_t pad_arr[]     = NAME_UP##_PADDINGS;                                      \
        const int32_t out_zp        = NAME_UP##_OUTPUT_OFFSET;                                 \
                                                                                               \
        const cmsis_nn_dims input_dims  = {in_dim_arr[0],  in_dim_arr[1],                      \
                                           in_dim_arr[2],  in_dim_arr[3]};                     \
        const cmsis_nn_dims output_dims = {out_dim_arr[0], out_dim_arr[1],                     \
                                           out_dim_arr[2], out_dim_arr[3]};                    \
        const cmsis_nn_tile block_shape = {.h = blk_arr[0], .w = blk_arr[1]};                  \
        /* pack {top,left,bottom,right} into {n,h,w,c} */                                      \
        const cmsis_nn_dims pad_dims    = {pad_arr[0], pad_arr[1], pad_arr[2], pad_arr[3]};    \
                                                                                               \
        const int8_t *input_data = name_lc##_input_tensor;                                     \
        int8_t output[NAME_UP##_OUTPUT_SIZE];                                                  \
                                                                                               \
        for (int r = 0; r < REPEAT_NUM; ++r)                                                   \
        {                                                                                      \
            const arm_cmsis_nn_status res =                                                    \
                arm_space_to_batch_nd_s8(                                                      \
                    input_data, &input_dims, &block_shape, &pad_dims,                          \
                    output, &output_dims, out_zp);                                             \
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, res);                                      \
        }                                                                                      \
                                                                                               \
        TEST_ASSERT_TRUE(                                                                      \
            validate_tol_s8(output, name_lc##_output, NAME_UP##_OUTPUT_SIZE, 0));              \
    }

/* Instantiate tests */
GEN_S2BND_TEST(SPACE_TO_BATCH_ND_BASIC_2X2_S8,        space_to_batch_nd_basic_2x2_s8)
GEN_S2BND_TEST(SPACE_TO_BATCH_ND_NSQUARE_PAD_S8,      space_to_batch_nd_nsquare_pad_s8)
GEN_S2BND_TEST(SPACE_TO_BATCH_ND_BATCH_S8,            space_to_batch_nd_batch_s8)
GEN_S2BND_TEST(SPACE_TO_BATCH_ND_BOTH_PAD_S8,         space_to_batch_nd_both_pad_s8)
GEN_S2BND_TEST(SPACE_TO_BATCH_ND_SMALL_PAD_S8,        space_to_batch_nd_small_pad_s8)
GEN_S2BND_TEST(SPACE_TO_BATCH_ND_BS1_IDENTITY_PAD_S8, space_to_batch_nd_bs1_identity_pad_s8)
