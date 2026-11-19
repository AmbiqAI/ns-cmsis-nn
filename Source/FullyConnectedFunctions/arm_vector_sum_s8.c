/*
 * SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_vector_sum_s8
 * Description:  Generic function for calculating vector sums
 *
 * $Date:        05 Sep 2024
 * $Revision:    V.3.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_vector_sum_s8_mve.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
/**
 *  @ingroup Public
 */

/**
 * @addtogroup FC
 * @{
 */

/*
 * S8 vector sum fuction in preparation for e.g. kernel sums in fully connected and matrix multiplication layer function
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_vector_sum_s8(int32_t *vector_sum_buf,
                                      const int32_t vector_cols,
                                      const int32_t vector_rows,
                                      const int8_t *vector_data,
                                      const int32_t lhs_offset,
                                      const int32_t rhs_offset,
                                      const int32_t *bias_data)
{

    if (bias_data)
    {
        memcpy(vector_sum_buf, bias_data, vector_rows * sizeof(int32_t));
    }
    else
    {
        memset(vector_sum_buf, 0, vector_rows * sizeof(int32_t));
    }

    if (lhs_offset)
    {
#if defined(ARM_MATH_MVEI)

        const int32_t row_loop_cnt = vector_rows / 5;
        for (int i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; i_row_loop_cnt++)
        {
            const arm_nn_vector_sum_s8x5_result sums = arm_nn_vector_sum_s8x5_mve(vector_data, vector_cols);
            int32_t vector_sum_0 = sums.sum[0];
            int32_t vector_sum_1 = sums.sum[1];
            int32_t vector_sum_2 = sums.sum[2];
            int32_t vector_sum_3 = sums.sum[3];
            int32_t vector_sum_4 = sums.sum[4];
            vector_data += 5 * vector_cols;

            if (rhs_offset)
            {
                vector_sum_0 = arm_nn_mac_wrap_s32_bits(vector_sum_0, vector_cols, rhs_offset);
                vector_sum_1 = arm_nn_mac_wrap_s32_bits(vector_sum_1, vector_cols, rhs_offset);
                vector_sum_2 = arm_nn_mac_wrap_s32_bits(vector_sum_2, vector_cols, rhs_offset);
                vector_sum_3 = arm_nn_mac_wrap_s32_bits(vector_sum_3, vector_cols, rhs_offset);
                vector_sum_4 = arm_nn_mac_wrap_s32_bits(vector_sum_4, vector_cols, rhs_offset);
            }

            vector_sum_buf[0] = arm_nn_mac_wrap_s32_bits(vector_sum_buf[0], vector_sum_0, lhs_offset);
            vector_sum_buf[1] = arm_nn_mac_wrap_s32_bits(vector_sum_buf[1], vector_sum_1, lhs_offset);
            vector_sum_buf[2] = arm_nn_mac_wrap_s32_bits(vector_sum_buf[2], vector_sum_2, lhs_offset);
            vector_sum_buf[3] = arm_nn_mac_wrap_s32_bits(vector_sum_buf[3], vector_sum_3, lhs_offset);
            vector_sum_buf[4] = arm_nn_mac_wrap_s32_bits(vector_sum_buf[4], vector_sum_4, lhs_offset);
            vector_sum_buf += 5;
        }
        const int32_t loop_cnt = vector_rows % 5;
        for (int i_row_loop_cnt = 0; i_row_loop_cnt < loop_cnt; i_row_loop_cnt++)
        {
            int32_t vector_sum_0 = arm_nn_vector_sum_s8x1_mve(vector_data, vector_cols);
            vector_data += vector_cols;
            if (rhs_offset)
            {
                vector_sum_0 = arm_nn_mac_wrap_s32_bits(vector_sum_0, vector_cols, rhs_offset);
            }

            vector_sum_buf[i_row_loop_cnt] =
                arm_nn_mac_wrap_s32_bits(vector_sum_buf[i_row_loop_cnt], vector_sum_0, lhs_offset);
        }
#else
    #if defined(__clang__)
        /* Keep one row reduction: duplicating it increases size and can worsen load scheduling. */
        #pragma clang loop unroll(disable)
    #endif
        for (int i = 0; i < vector_rows; i++)
        {
            uint32_t sum_bits = 0;
            for (int j = 0; j < vector_cols; j++)
            {
                sum_bits += (uint32_t)*vector_data++;
            }
            int32_t sum;
            memcpy(&sum, &sum_bits, sizeof(sum));
            if (rhs_offset)
            {
                sum = arm_nn_mac_wrap_s32_bits(sum, vector_cols, rhs_offset);
            }
            *vector_sum_buf = arm_nn_mac_wrap_s32_bits(*vector_sum_buf, sum, lhs_offset);
            vector_sum_buf++;
        }
#endif
    }

    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of FC group
 */
