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
 * Title:        arm_transpose_conv_s16.c
 * Description:  s16 version of transposed convolution using symmetric quantization.
 *
 * $Date:        07 April 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Basic s16 transpose convolution function.
 *
 * Refer header file for details.
 *
 * Quantization scheme (int16x8):
 *   - Input:  int16, zero_point = 0 (symmetric)
 *   - Filter: int8
 *   - Bias:   int64 or int32 (via cmsis_nn_bias_data)
 *   - Output: int16, zero_point = 0 (symmetric)
 *
 * Rolling scratch-buffer algorithm (identical structure to arm_transpose_conv_s8):
 *   A circular int64 buffer of size (buf_y * buf_x) elements holds partial
 *   output sums.  Each input row scatters its contributions into the buffer
 *   via arm_nn_transpose_conv_row_s16_s64, and completed output rows are
 *   requantized and written to output_data.
 */
arm_cmsis_nn_status arm_transpose_conv_s16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_context *output_ctx,
                                           const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                           const cmsis_nn_per_channel_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const cmsis_nn_bias_data *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int16_t *output_data)
{
    (void)bias_dims;
    (void)output_ctx;

    if (ctx == NULL || ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t activation_min = transpose_conv_params->activation.min;
    const int32_t activation_max = transpose_conv_params->activation.max;

    const int32_t input_ch  = input_dims->c;
    const int32_t input_x   = input_dims->w;
    const int32_t input_y   = input_dims->h;

    const int32_t output_x  = output_dims->w;
    const int32_t output_y  = output_dims->h;
    const int32_t output_ch = output_dims->c;

    const int32_t filter_x  = filter_dims->w;
    const int32_t filter_y  = filter_dims->h;

    const int32_t pad_x     = transpose_conv_params->padding.w;
    const int32_t pad_y     = transpose_conv_params->padding.h;

    const int32_t stride_x  = transpose_conv_params->stride.w;
    const int32_t stride_y  = transpose_conv_params->stride.h;

    const int32_t *output_multiplier = quant_params->multiplier;
    const int32_t *output_shift      = quant_params->shift;

    /* Rolling int64 scratch buffer dimensions (in elements) */
    const int32_t buf_x_elements = ((input_x - 1) * stride_x + MAX(filter_x, stride_x));
    const int32_t buf_x          = buf_x_elements * output_ch;   /* elements per row */
    const int32_t buf_y          = MAX(filter_y, stride_y);
    const int32_t buf_size       = buf_y * buf_x;                 /* total int64 elements */

    int64_t *buf = (int64_t *)ctx->buf;

    int32_t batch_cnt        = input_dims->n;
    const int8_t  *filter    = filter_data;
    const int16_t *input     = input_data;
    int16_t       *output    = output_data;

    /* Pre-compute bias array properties */
    const bool     has_bias    = (bias_data != NULL) && (bias_data->data != NULL);
    const bool     bias_is_s32 = has_bias && bias_data->is_int32_bias;
    const int32_t *bias_s32    = bias_is_s32 ? (const int32_t *)bias_data->data : NULL;
    const int64_t *bias_s64    = (!bias_is_s32 && has_bias) ? (const int64_t *)bias_data->data : NULL;

    while (batch_cnt)
    {
        /* ---------------------------------------------------------------- */
        /* Reset rolling buffer with bias (or zeros)                        */
        /* ---------------------------------------------------------------- */
        if (has_bias)
        {
            for (int32_t pos = 0; pos < buf_x_elements * buf_y; pos++)
            {
                int64_t *dst = buf + pos * output_ch;
                if (bias_is_s32)
                {
                    for (int32_t z = 0; z < output_ch; z++)
                    {
                        dst[z] = (int64_t)bias_s32[z];
                    }
                }
                else
                {
                    arm_memcpy_s8((int8_t *)dst, (const int8_t *)bias_s64,
                                  (uint32_t)output_ch * sizeof(int64_t));
                }
            }
        }
        else
        {
            arm_memset_s8((int8_t *)buf, 0, (uint32_t)buf_size * sizeof(int64_t));
        }

        int32_t buf_row = 0;
        int32_t out_rows_written = 0;

        for (int32_t j = 0; j < input_y; j++)
        {
            const int32_t skip_rows_top =
                MAX(0, pad_y - j * stride_y);
            const int32_t skip_rows_bottom =
                MAX(0, (j * stride_y + filter_y) - (pad_y + output_y) - 1);

            /* Scatter contributions of one input row into rolling buffer */
            arm_nn_transpose_conv_row_s16_s64(input,
                                              filter,
                                              buf,
                                              buf_row,
                                              buf_size,
                                              filter_y,
                                              filter_x,
                                              input_ch,
                                              output_ch,
                                              buf_x,
                                              input_x,
                                              stride_x,
                                              skip_rows_top,
                                              skip_rows_bottom);
            input += input_ch * input_x;

            if (skip_rows_top == 0)
            {
                for (int32_t y = 0; y < stride_y && out_rows_written < output_y; y++)
                {
                    int64_t *buf_out = buf + buf_row;
                    buf_out         += output_ch * pad_x;   /* skip left-pad columns */

                    for (int32_t x = 0; x < output_x; x++)
                    {
                        const int32_t *mult_ptr  = output_multiplier;
                        const int32_t *shift_ptr = output_shift;

                        for (int32_t z = 0; z < output_ch; z++)
                        {
                            int64_t acc = *buf_out++;
                            const int32_t mult_val = *mult_ptr++;
                            int32_t reduced_mult = REDUCE_MULTIPLIER(mult_val);
                            int32_t result = arm_nn_requantize_s64(acc, reduced_mult, *shift_ptr++);
                            result = MAX(result, activation_min);
                            result = MIN(result, activation_max);
                            *output++ = (int16_t)result;
                        }
                    }

                    /* Reset this row of the rolling buffer back to bias */
                    if (has_bias)
                    {
                        for (int32_t pos = 0; pos < buf_x_elements; pos++)
                        {
                            int64_t *dst = buf + buf_row + pos * output_ch;
                            if (bias_is_s32)
                            {
                                for (int32_t z = 0; z < output_ch; z++)
                                {
                                    dst[z] = (int64_t)bias_s32[z];
                                }
                            }
                            else
                            {
                                arm_memcpy_s8((int8_t *)dst, (const int8_t *)bias_s64,
                                              (uint32_t)output_ch * sizeof(int64_t));
                            }
                        }
                    }
                    else
                    {
                        arm_memset_s8((int8_t *)(buf + buf_row), 0,
                                      (uint32_t)buf_x * sizeof(int64_t));
                    }

                    buf_row = (buf_row + buf_x) % buf_size;
                    out_rows_written++;
                }
            }
        }

        /* ---------------------------------------------------------------- */
        /* Write leftover rows (filter tail beyond last input row)          */
        /* ---------------------------------------------------------------- */
        for (int32_t y = 0; y < filter_y - stride_y; y++)
        {
            const int32_t out_row_abs = input_y * stride_y + y;
            if (out_row_abs >= pad_y && out_row_abs < pad_y + output_y)
            {
                int64_t *buf_out = buf + buf_row;
                buf_out         += output_ch * pad_x;

                for (int32_t x = 0; x < output_x; x++)
                {
                    const int32_t *mult_ptr  = output_multiplier;
                    const int32_t *shift_ptr = output_shift;

                    for (int32_t z = 0; z < output_ch; z++)
                    {
                        int64_t acc = *buf_out++;
                        const int32_t mult_val = *mult_ptr++;
                        int32_t reduced_mult = REDUCE_MULTIPLIER(mult_val);
                        int32_t result = arm_nn_requantize_s64(acc, reduced_mult, *shift_ptr++);
                        result = MAX(result, activation_min);
                        result = MIN(result, activation_max);
                        *output++ = (int16_t)result;
                    }
                }
            }

            buf_row = (buf_row + buf_x) % buf_size;
        }

        batch_cnt--;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
