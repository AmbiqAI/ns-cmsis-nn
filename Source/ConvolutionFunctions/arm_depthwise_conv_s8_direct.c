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
#include <arm_acle.h>
#include "stdio.h"
#include <inttypes.h>
#ifndef CH_BLOCK
#define CH_BLOCK 1024          /* choose 4-lane multiples: 4, 8, 12, 16 …   */
#endif
#define PF_AHEAD 4
#ifndef __pld                       /* portable prefetch wrapper            */
#define __pld(addr) __builtin_prefetch((addr), 0, 3)
#endif
__STATIC_FORCEINLINE int32x4_t load_bias4(const int32_t *bias, int32_t idx)
{
   return bias ? vldrwq_s32(bias + idx) : vdupq_n_s32(0);
}
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
arm_cmsis_nn_status arm_depthwise_conv_s8_direct(const cmsis_nn_context                   *ctx,
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
    (void)ctx; (void)weight_sum_ctx; (void)bias_dims; (void)filter_dims;
    /* ------------------ basic sizes ----------------------------------- */
    const int32_t in_h  = input_dims->h;
    const int32_t in_w  = input_dims->w;
    const int32_t ch    = input_dims->c;
    const int32_t out_h = output_dims->h;
    const int32_t out_w = output_dims->w;
    /* channels must be divisible by CH_BLOCK */
    //    if (ch % CH_BLOCK)
    //        return ARM_CMSIS_NN_ARG_ERROR;
    const int32_t input_offset  = dw_conv_params->input_offset;
    const int32_t output_offset = dw_conv_params->output_offset;
    const int32_t act_min       = dw_conv_params->activation.min;
    const int32_t act_max       = dw_conv_params->activation.max;
    const int32_t minus_io      = -input_offset;
    int32_t eff_blk = CH_BLOCK;
    if(eff_blk > ch)
        eff_blk = ch;
    const int32_t vec_per_blk   = eff_blk / 4;       
    const int32_t blk_cnt = (ch + eff_blk - 1) / eff_blk;      
    for (int32_t blk = 0; blk < blk_cnt; ++blk)
    {
        // ---- software prefetch for future blocks 
        if (blk + PF_AHEAD < blk_cnt)
        {
            __pld(filter_data + eff_blk * (blk + PF_AHEAD));
            __pld(filter_data + eff_blk * (blk + PF_AHEAD) + 3 * ch);
        }
        // ---- per-vector (4-lane) inner loop 
        for (int32_t vi = 0; vi < vec_per_blk; ++vi)
        {
            const int32_t g = blk * vec_per_blk + vi;     
            // load 9 weight vectors (pixel-major order)
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
            // bias with folding (unchanged) 
            int32x4_t bias4 = load_bias4(bias_data, 4 * g);
            int32x4_t io_vec = vdupq_n_s32(input_offset);
            int32x4_t wsum1 = vaddq_s32(vaddq_s32(w00, w01), w02);
            int32x4_t wsum2 = vaddq_s32(vaddq_s32(w10, w11), w12);
            int32x4_t wsum3 = vaddq_s32(vaddq_s32(w20, w21), w22);
            int32x4_t wsum  = vaddq_s32(vaddq_s32(wsum1, wsum2), wsum3);
            bias4 = vaddq_s32(bias4, vmulq_s32(io_vec, wsum));
            const int32_t *mult_v  = quant_params->multiplier + 4 * g;
            const int32_t *shift_v = quant_params->shift      + 4 * g;
            // =============== sweep output rows 
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
                        int32x4_t v_data = (ix0 >= 0) ?
                            vldrbq_s32(input_data + ((iy0*in_w + ix0)*ch) + 4*g)
                            : vdupq_n_s32(minus_io);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w00));
                        v_data = vldrbq_s32(input_data + ((iy0*in_w + ox)*ch) + 4*g);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w01));
                        v_data = (ix2 < in_w) ?
                            vldrbq_s32(input_data + ((iy0*in_w + ix2)*ch) + 4*g)
                            : vdupq_n_s32(minus_io);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w02));
                    }
                    else
                    {
                        int32x4_t pad = vdupq_n_s32(minus_io);
                        acc = vaddq_s32(acc, vmulq_s32(pad, w00));
                        acc = vaddq_s32(acc, vmulq_s32(pad, w01));
                        acc = vaddq_s32(acc, vmulq_s32(pad, w02));
                    }
                    // ------------- row  0 
                    {
                        int32x4_t v_data = (ix0 >= 0) ?
                            vldrbq_s32(input_data + ((iy1*in_w + ix0)*ch) + 4*g)
                            : vdupq_n_s32(minus_io);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w10));
                        v_data = vldrbq_s32(input_data + ((iy1*in_w + ox)*ch) + 4*g);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w11));
                        v_data = (ix2 < in_w) ?
                            vldrbq_s32(input_data + ((iy1*in_w + ix2)*ch) + 4*g)
                            : vdupq_n_s32(minus_io);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w12));
                    }
                    
                    if (row2)
                    {
                        int32x4_t v_data = (ix0 >= 0) ?
                            vldrbq_s32(input_data + ((iy2*in_w + ix0)*ch) + 4*g)
                            : vdupq_n_s32(minus_io);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w20));
                        v_data = vldrbq_s32(input_data + ((iy2*in_w + ox)*ch) + 4*g);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w21));
                        v_data = (ix2 < in_w) ?
                            vldrbq_s32(input_data + ((iy2*in_w + ix2)*ch) + 4*g)
                            : vdupq_n_s32(minus_io);
                        acc = vaddq_s32(acc, vmulq_s32(v_data, w22));
                    }
                    else
                    {
                        int32x4_t pad = vdupq_n_s32(minus_io);
                        acc = vaddq_s32(acc, vmulq_s32(pad, w20));
                        acc = vaddq_s32(acc, vmulq_s32(pad, w21));
                        acc = vaddq_s32(acc, vmulq_s32(pad, w22));
                    }
                    // -------- requantise & store 
                    int32x4_t mult  = vldrwq_s32(mult_v);
                    int32x4_t shift = vldrwq_s32(shift_v);
                    mve_pred16_t p  = vctp32q(4);
                    acc = arm_requantize_mve_32x4(acc, mult, shift);
                    acc = vaddq_n_s32(acc, output_offset);
                    acc = vmaxq_s32(acc, vdupq_n_s32(act_min));
                    acc = vminq_s32(acc, vdupq_n_s32(act_max));
                    vstrbq_p_s32(output_data + ((oy*out_w + ox)*ch) + 4*g,
                                acc, p);
                }
            }
        }   
    }       
    return ARM_CMSIS_NN_SUCCESS;
    }

/**
 * @} end of NNConv group
 */
