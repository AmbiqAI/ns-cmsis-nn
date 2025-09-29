/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_depthwise_conv_s8_1d_valid_wrapper.c
 * Description:  Wrapper function for optimized s8 1D depthwise separable convolution function for
 *               height of 1, valid padding, stride of 1, and dilation of 1.
 *
 * $Date:        16 September 2025
 * $Revision:    V.3.5.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"


arm_cmsis_nn_status arm_depthwise_conv_s8_1d_valid_wrapper(const cmsis_nn_context *ctx,
    const cmsis_nn_context *weight_sum_ctx,
    const cmsis_nn_dw_conv_params *dw_conv_params,
    const cmsis_nn_per_channel_quant_params *quant_params,
    const cmsis_nn_dims *input_dims,
    const int8_t *input,
    const cmsis_nn_dims *filter_dims,
    const int8_t *kernel,
    const cmsis_nn_dims *bias_dims,
    const int32_t *bias,
    const cmsis_nn_dims *output_dims,
    int8_t *output)
{
    const int32_t input_ch = input_dims->c;
    const int32_t output_ch = output_dims->c;

    /* Check depth multiplier is 1 */
    if (input_ch != output_ch)
    {
    return ARM_CMSIS_NN_ARG_ERROR;
    }

#ifdef ARM_MATH_MVEI
    (void)bias_dims;
    (void)bias;
    (void)ctx;

    const int32_t input_x  = input_dims->w;
    const int32_t input_y  = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t pad_x    = dw_conv_params->padding.w;
    const int32_t pad_y    = dw_conv_params->padding.h;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t *output_shift = quant_params->shift;
    const int32_t *output_mult  = quant_params->multiplier;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t out_offset = dw_conv_params->output_offset;
    const int32_t in_offset  = dw_conv_params->input_offset;
    const int32_t act_min    = dw_conv_params->activation.min;
    const int32_t act_max    = dw_conv_params->activation.max;

    const bool is_1d_x =
    (input_y == 1) && (output_y == 1) && (kernel_y == 1);
    const bool is_1d_y =
    (input_x == 1) && (output_x == 1) && (kernel_x == 1);


    const bool is_1d_valid =
        (input_dims->n == 1) &&
        (pad_x == 0) && (pad_y == 0) &&
        (stride_x == 1) && (stride_y == 1) &&
        (dw_conv_params->dilation.w == 1) && (dw_conv_params->dilation.h == 1) &&
        (dw_conv_params->ch_mult == 1) &&
        (is_1d_x || is_1d_y);

    if (is_1d_x && is_1d_valid)
    {
        // Vectorize across channels and compute 4 outputs per call, no im2col.
        // Requirements:
        // - weight_sum_ctx->buf must be precomputed by arm_depthwise_convolve_weight_sum() in init()
        int32_t *weight_sum_buf = (int32_t *)weight_sum_ctx->buf;
        if (weight_sum_buf == NULL)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        const int32_t C = input_ch;          // channels
        const int32_t Wout = output_x;       // width outputs

        const int32_t ch_loop = (C + (CH_IN_BLOCK_MVE - 1)) / CH_IN_BLOCK_MVE;
        int32_t remaining_ch = C;

        for (int i_ch = 0; i_ch < ch_loop; i_ch++)
        {
            const int32_t block_offset = i_ch * CH_IN_BLOCK_MVE;
            const int32_t active_ch = MIN(CH_IN_BLOCK_MVE, remaining_ch);
            remaining_ch -= CH_IN_BLOCK_MVE;

            // Slide across width in tiles of 4 outputs
            int x0 = 0;
            for (; x0 + 3 < Wout; x0 += 4)
            {
                // Base pointers into input/output at column x0 and channel block
                const int8_t *lhs_base = input  + (x0 * C) + block_offset;
                int8_t       *out_base = output + (x0 * C) + block_offset;

                // Call 4-output sliding-window microkernel
                arm_cmsis_nn_status status = arm_depthwise_conv_s8_1d_valid(
                    weight_sum_buf + block_offset, 
                    lhs_base,
                    kernel + block_offset,
                    in_offset,
                    active_ch,
                    C,
                    output_shift + block_offset,
                    output_mult  + block_offset,
                    out_offset,
                    act_min,
                    act_max,
                    kernel_x,
                    NULL,             // unused (compensation folded in base)
                    out_base);

                if (status != ARM_CMSIS_NN_SUCCESS) return status;
            }

            // Tail: 0..3 leftover outputs use 1-output sliding microkernel
            for (; x0 < Wout; x0++)
            {
                const int8_t *lhs_col0 = input  + (x0 * C) + block_offset;
                int8_t       *out_col  = output + (x0 * C) + block_offset;

                arm_depthwise_conv_s8_1d_valid_1out(
                    weight_sum_buf + block_offset,
                    lhs_col0,
                    kernel + block_offset,
                    active_ch,
                    C,
                    output_shift + block_offset,
                    output_mult  + block_offset,
                    out_offset,
                    act_min,
                    act_max,
                    kernel_x,
                    out_col);
            }
        }

        return ARM_CMSIS_NN_SUCCESS;
    }
    else if (is_1d_y && is_1d_valid) {
        int32_t *weight_sum_buf = (int32_t *)weight_sum_ctx->buf;
        if (weight_sum_buf == NULL)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        const int32_t C = input_ch;          // channels
        const int32_t Hout = output_y;       // height outputs

        const int32_t ch_loop = (C + (CH_IN_BLOCK_MVE - 1)) / CH_IN_BLOCK_MVE;
        int32_t remaining_ch = C;

        for (int i_ch = 0; i_ch < ch_loop; i_ch++)
        {
            const int32_t block_offset = i_ch * CH_IN_BLOCK_MVE;
            const int32_t active_ch = MIN(CH_IN_BLOCK_MVE, remaining_ch);
            remaining_ch -= CH_IN_BLOCK_MVE;

            // Slide across height in tiles of 4 outputs
            int y0 = 0;
            for (; y0 + 3 < Hout; y0 += 4)
            {
                // Base pointers into input/output at column x0 and channel block
                const int8_t *lhs_base = input  + (y0 * C) + block_offset;
                int8_t       *out_base = output + (y0 * C) + block_offset;

                // Call 4-output sliding-window microkernel
                arm_cmsis_nn_status status = arm_depthwise_conv_s8_1d_valid(
                    weight_sum_buf + block_offset, 
                    lhs_base,
                    kernel + block_offset,
                    in_offset,
                    active_ch,
                    C,
                    output_shift + block_offset,
                    output_mult  + block_offset,
                    out_offset,
                    act_min,
                    act_max,
                    kernel_y,
                    NULL,             // unused (compensation folded in base)
                    out_base);

                if (status != ARM_CMSIS_NN_SUCCESS) return status;
            }

            // Tail: 0..3 leftover outputs use 1-output sliding microkernel
            for (; y0 < Hout; y0++)
            {
                const int8_t *lhs_col0 = input  + (y0 * C) + block_offset;
                int8_t       *out_col  = output + (y0 * C) + block_offset;

                arm_depthwise_conv_s8_1d_valid_1out(
                    weight_sum_buf + block_offset,
                    lhs_col0,
                    kernel + block_offset,
                    active_ch,
                    C,
                    output_shift + block_offset,
                    output_mult  + block_offset,
                    out_offset,
                    act_min,
                    act_max,
                    kernel_y,
                    out_col);
            }
        }

        return ARM_CMSIS_NN_SUCCESS;
    }
    else {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#else
    (void)weight_sum_ctx;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    return arm_depthwise_conv_s8(ctx,
                                dw_conv_params,
                                quant_params,
                                input_dims,
                                input,
                                filter_dims,
                                kernel,
                                bias_dims,
                                bias,
                                output_dims,
                                output);
#endif  // ARM_MATH_MVEI
}
