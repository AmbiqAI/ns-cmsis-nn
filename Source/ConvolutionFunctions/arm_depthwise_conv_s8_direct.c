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
 * Title:        arm_depthwise_conv_s8_opt.c
 * Description:  Optimized s8 depthwise separable convolution function for
 *               channel multiplier of 1.
 *
 * $Date:        22 March 2023
 * $Revision:    V.3.5.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"


/*******************************************************************************
* Depth-wise 3 × 3 convolution, stride = 1, padding = SAME, channels multiple-of-4
*   – input / weights  : int8_t
*   – bias             : int32_t
*   – output           : int8_t
*   – per-channel requant (mult, shift) supplied by CMSIS-NN
*   – no tail processing (ch % 4 == 0)
*   – no extra scratch memory   (ctx->buf == 0)
*
* Public entry point follows the usual CMSIS-NN signature so that the function
* can be dropped into arm_nnfunctions.c and called through
*     arm_depthwise_conv_s8()
*/
#if defined(ARM_MATH_MVEI)
arm_cmsis_nn_status arm_depthwise_conv_s8_direct(const cmsis_nn_context *ctx,
       const cmsis_nn_context                  *weight_sum_ctx,
       const cmsis_nn_dw_conv_params           *dw_conv_params,
       const cmsis_nn_per_channel_quant_params *quant_params,
       const cmsis_nn_dims                     *input_dims,
       const int8_t                            *input_data,
       const cmsis_nn_dims                     *filter_dims,
       const int8_t                            *filter_data,
       const cmsis_nn_dims                     *bias_dims,
       const int32_t                           *bias_data,
       const cmsis_nn_dims                     *output_dims,
       int8_t                                  *output_data)
{


    if ((input_dims->c & 0x3) != 0 || input_dims->c != output_dims->c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    (void)ctx;                            
    (void)bias_dims;
    (void)filter_dims;
    (void)bias_data;
    const int32_t *weight_sum_base = weight_sum_ctx->buf;
    const int32_t in_h = input_dims->h;
    const int32_t in_w = input_dims->w;
    const int32_t ch = input_dims->c;          
    const int32_t out_h = output_dims->h;
    const int32_t out_w = output_dims->w;    
    const int32_t input_offset = dw_conv_params->input_offset;  
    const int32_t output_offset = dw_conv_params->output_offset;
    const int32_t act_min = dw_conv_params->activation.min;
    const int32_t act_max = dw_conv_params->activation.max;
    const int32_t g_cnt = ch >> 2;         
    const int32_t minus_io = -input_offset; 
    for (int32_t g = 0; g < g_cnt; ++g)
    {
        const int8_t *kw = filter_data + 4 * g;           
        int32x4_t w00 = vldrbq_s32(kw + 0 * ch);
        int32x4_t w01 = vldrbq_s32(kw + 1 * ch);
        int32x4_t w02 = vldrbq_s32(kw + 2 * ch);
        int32x4_t w10 = vldrbq_s32(kw + 3 * ch);
        int32x4_t w11 = vldrbq_s32(kw + 4 * ch);
        int32x4_t w12 = vldrbq_s32(kw + 5 * ch);
        int32x4_t w20 = vldrbq_s32(kw + 6 * ch);
        int32x4_t w21 = vldrbq_s32(kw + 7 * ch);
        int32x4_t w22 = vldrbq_s32(kw + 8 * ch);
        int32x4_t bias4 = weight_sum_base ? vldrwq_s32(weight_sum_base + 4 * g) : vdupq_n_s32(0);

        const int32_t *mult_g  = quant_params->multiplier + 4 * g;
        const int32_t *shift_g = quant_params->shift      + 4 * g;
        for (int32_t oy = 0; oy < out_h; ++oy)
        {
            const int32_t iy0 = oy - 1;
            const int32_t iy1 = oy;
            const int32_t iy2 = oy + 1;
            const bool row0 = (iy0 >= 0);
            const bool row2 = (iy2 < in_h);
            for (int32_t ox = 0; ox < out_w; ++ox)
            {
                const int32_t ix0 = ox - 1;
                const int32_t ix2 = ox + 1;
                int32x4_t acc = bias4;
                if (row0) 
                {
                     int32x4_t v = (ix0 >= 0)
                        ? vldrbq_s32(input_data + ((iy0 * in_w + ix0) * ch) + 4 * g)
                        : vdupq_n_s32(minus_io);
                    acc = vaddq_s32(acc, vmulq_s32(v, w00));
                    v = vldrbq_s32(input_data + ((iy0 * in_w + ox) * ch) + 4 * g);
                    acc = vaddq_s32(acc, vmulq_s32(v, w01));
                    v = (ix2 < in_w)
                        ? vldrbq_s32(input_data + ((iy0 * in_w + ix2) * ch) + 4 * g)
                        : vdupq_n_s32(minus_io);
                    acc = vaddq_s32(acc, vmulq_s32(v, w02));
                }
                else
                {
                    int32x4_t pad = vdupq_n_s32(minus_io);
                    acc = vaddq_s32(acc, vmulq_s32(pad, w00));
                    acc = vaddq_s32(acc, vmulq_s32(pad, w01));
                    acc = vaddq_s32(acc, vmulq_s32(pad, w02));
                }
                {
                    int32x4_t v = (ix0 >= 0)
                        ? vldrbq_s32(input_data + ((iy1 * in_w + ix0) * ch) + 4 * g)
                        : vdupq_n_s32(minus_io);
                    acc = vaddq_s32(acc, vmulq_s32(v, w10));
                    v = vldrbq_s32(input_data + ((iy1 * in_w + ox) * ch) + 4 * g);
                    acc = vaddq_s32(acc, vmulq_s32(v, w11));
                    v = (ix2 < in_w)
                        ? vldrbq_s32(input_data + ((iy1 * in_w + ix2) * ch) + 4 * g)
                        : vdupq_n_s32(minus_io);
                    acc = vaddq_s32(acc, vmulq_s32(v, w12));
                }
                if (row2)
                {
                    int32x4_t v = (ix0 >= 0)
                        ? vldrbq_s32(input_data + ((iy2 * in_w + ix0) * ch) + 4 * g)
                        : vdupq_n_s32(minus_io);
                    acc = vaddq_s32(acc, vmulq_s32(v, w20));
                    v = vldrbq_s32(input_data + ((iy2 * in_w + ox) * ch) + 4 * g);
                    acc = vaddq_s32(acc, vmulq_s32(v, w21));
                    v = (ix2 < in_w)
                        ? vldrbq_s32(input_data + ((iy2 * in_w + ix2) * ch) + 4 * g)
                        : vdupq_n_s32(minus_io);
                    acc = vaddq_s32(acc, vmulq_s32(v, w22));
                }
                else
                {
                    int32x4_t pad = vdupq_n_s32(minus_io);
                    acc = vaddq_s32(acc, vmulq_s32(pad, w20));
                    acc = vaddq_s32(acc, vmulq_s32(pad, w21));
                    acc = vaddq_s32(acc, vmulq_s32(pad, w22));
                }
                int32x4_t mult  = vldrwq_s32(mult_g);
                int32x4_t shift = vldrwq_s32(shift_g);
                acc = arm_requantize_mve_32x4(acc, mult, shift);
                acc = vaddq_n_s32(acc, output_offset);
                acc = vmaxq_s32(acc, vdupq_n_s32(act_min));
                acc = vminq_s32(acc, vdupq_n_s32(act_max));
                vstrbq_s32(output_data + ((oy * out_w + ox) * ch) + 4 * g, acc);
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif
// 
/**
 * @} end of NNConv group
 */
