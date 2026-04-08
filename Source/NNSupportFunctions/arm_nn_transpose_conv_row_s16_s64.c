/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_transpose_conv_row_s16_s64.c
 * Description:  Transpose convolution row helper for s16 input / s64 accumulator.
 *
 * $Date:        07 April 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/*
 * Row of s16 scalars multiplied with an s8 matrix and accumulated into
 * a rolling int64 scratch buffer.  Helper for arm_transpose_conv_s16.
 *
 * Refer header file for details.
 */
arm_cmsis_nn_status arm_nn_transpose_conv_row_s16_s64(const int16_t *lhs,
                                                      const int8_t *rhs,
                                                      int64_t *output_start,
                                                      const int32_t output_index,
                                                      const int32_t output_max,
                                                      const int32_t rhs_rows,
                                                      const int32_t rhs_cols,
                                                      const int32_t input_channels,
                                                      const int32_t output_channels,
                                                      const int32_t row_offset,
                                                      const int32_t input_x,
                                                      const int32_t stride_x,
                                                      const int32_t skip_rows_top,
                                                      const int32_t skip_rows_bottom)
{
    const int32_t skip_pre_elems  = skip_rows_top    * rhs_cols * input_channels;
    const int32_t skip_post_elems = skip_rows_bottom * rhs_cols * input_channels;
    const int32_t rhs_rows_count  = rhs_rows - skip_rows_top - skip_rows_bottom;

    /* ------------------------------------------------------------------ */
    /* Process 4 input pixels at a time                                    */
    /* ------------------------------------------------------------------ */
    int32_t input_count = input_x;
    for (; input_count > 3; input_count -= 4)
    {
        const int8_t *rhs_ptr = rhs;

        for (int32_t oc = 0; oc < output_channels; oc++)
        {
            rhs_ptr += skip_pre_elems;
            int32_t index = output_index;

            for (int32_t i_row = 0; i_row < rhs_rows_count; i_row++)
            {
                int64_t *output_ptr = output_start + index;

                for (int32_t i_col = 0; i_col < rhs_cols; i_col++)
                {
                    const int16_t *lhs_ptr = lhs;

                    int64_t result0 = 0;
                    int64_t result1 = 0;
                    int64_t result2 = 0;
                    int64_t result3 = 0;

                    for (int32_t ic = 0; ic < input_channels; ic++)
                    {
                        const int64_t rhs_val = (int64_t)*rhs_ptr++;

                        result0 += (int64_t) * lhs_ptr                             * rhs_val;
                        result1 += (int64_t) * (lhs_ptr + input_channels)          * rhs_val;
                        result2 += (int64_t) * (lhs_ptr + 2 * input_channels)      * rhs_val;
                        result3 += (int64_t) * (lhs_ptr + 3 * input_channels)      * rhs_val;
                        lhs_ptr++;
                    }

                    int64_t *out_temp   = output_ptr;
                    *output_ptr        += result0;
                    out_temp           += (int32_t)stride_x * output_channels;
                    *out_temp          += result1;
                    out_temp           += (int32_t)stride_x * output_channels;
                    *out_temp          += result2;
                    out_temp           += (int32_t)stride_x * output_channels;
                    *out_temp          += result3;

                    output_ptr += output_channels;
                }

                index = (index + row_offset) % output_max;
            }

            ++output_start;
            rhs_ptr += skip_post_elems;
        }

        output_start += ((int32_t)(4 * stride_x) - 1) * output_channels;
        lhs          += 4 * input_channels;
    }

    /* ------------------------------------------------------------------ */
    /* Tail: 2 input pixels                                                */
    /* ------------------------------------------------------------------ */
    if (input_count & 0b10)
    {
        const int8_t *rhs_ptr = rhs;

        for (int32_t oc = 0; oc < output_channels; oc++)
        {
            rhs_ptr += skip_pre_elems;
            int32_t index = output_index;

            for (int32_t i_row = 0; i_row < rhs_rows_count; i_row++)
            {
                int64_t *output_ptr = output_start + index;

                for (int32_t i_col = 0; i_col < rhs_cols; i_col++)
                {
                    const int16_t *lhs_ptr = lhs;

                    int64_t result0 = 0;
                    int64_t result1 = 0;

                    for (int32_t ic = 0; ic < input_channels; ic++)
                    {
                        const int64_t rhs_val = (int64_t)*rhs_ptr++;

                        result0 += (int64_t) * lhs_ptr                    * rhs_val;
                        result1 += (int64_t) * (lhs_ptr + input_channels) * rhs_val;
                        lhs_ptr++;
                    }

                    int64_t *out_temp  = output_ptr;
                    *output_ptr       += result0;
                    out_temp          += (int32_t)stride_x * output_channels;
                    *out_temp         += result1;

                    output_ptr += output_channels;
                }

                index = (index + row_offset) % output_max;
            }

            ++output_start;
            rhs_ptr += skip_post_elems;
        }

        output_start += ((int32_t)(2 * stride_x) - 1) * output_channels;
        lhs          += 2 * input_channels;
    }

    /* ------------------------------------------------------------------ */
    /* Tail: 1 input pixel                                                 */
    /* ------------------------------------------------------------------ */
    if (input_count & 0b1)
    {
        const int8_t *rhs_ptr = rhs;

        for (int32_t oc = 0; oc < output_channels; oc++)
        {
            rhs_ptr += skip_pre_elems;
            int32_t index = output_index;

            for (int32_t i_row = 0; i_row < rhs_rows_count; i_row++)
            {
                int64_t *output_ptr = output_start + index;

                for (int32_t i_col = 0; i_col < rhs_cols; i_col++)
                {
                    const int16_t *lhs_ptr = lhs;

                    int64_t result0 = 0;

                    for (int32_t ic = 0; ic < input_channels; ic++)
                    {
                        result0 += (int64_t)*lhs_ptr++ * (int64_t)*rhs_ptr++;
                    }

                    *output_ptr += result0;
                    output_ptr  += output_channels;
                }

                index = (index + row_offset) % output_max;
            }

            ++output_start;
            rhs_ptr += skip_post_elems;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of supportConvolution group
 */
