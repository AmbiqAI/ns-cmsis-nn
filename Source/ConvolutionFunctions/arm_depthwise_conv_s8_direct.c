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
#include <stdio.h>
/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */
#include <inttypes.h>


#define CH_BLOCK      4
__STATIC_FORCEINLINE int32x4_t load_bias4(const int32_t *bias, int32_t idx)
{
   return (bias) ? vldrwq_s32(bias + idx) : vdupq_n_s32(0);
}

void print_vector(int32x4_t vector) {
    printf("[%" PRId32 ", %" PRId32 ", %" PRId32 ", %" PRId32 "]\n",
           vgetq_lane_s32(vector, 0),
           vgetq_lane_s32(vector, 1),
           vgetq_lane_s32(vector, 2),
           vgetq_lane_s32(vector, 3));
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
arm_cmsis_nn_status arm_depthwise_conv_s8_direct(const cmsis_nn_context     *ctx,
                                    const cmsis_nn_context                  *weight_sum_ctx,
                                    const cmsis_nn_dw_conv_params           *dw_params,
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
   (void)ctx;                            
   (void)bias_dims;
   (void)filter_dims;
   (void)weight_sum_ctx;
   
   const int32_t in_h  = input_dims->h;
   const int32_t in_w  = input_dims->w;
   const int32_t ch    = input_dims->c;       
   const int32_t out_h = output_dims->h;
   const int32_t out_w = output_dims->w;

   if (ch & (CH_BLOCK - 1))          
       return ARM_CMSIS_NN_ARG_ERROR;
   const int32_t input_offset           = dw_params->input_offset;  
   const int32_t output_offset          = dw_params->output_offset;
   const int32_t output_activation_min  = dw_params->activation.min;
   const int32_t output_activation_max  = dw_params->activation.max;
   const int32_t n_block = ch >> 2;    
   for (int32_t oy = 0; oy < out_h; ++oy)
   {
       for (int32_t ox = 0; ox < out_w; ++ox)
       {
           
           for (int32_t g = 0; g < n_block; ++g)
           {
               
               int32x4_t acc = load_bias4(bias_data, 4 * g);
            for (int32_t ky = 0; ky < 3; ++ky)
            {
                const int32_t iy = oy + ky - 1;               
                const bool    row_ok = (iy >= 0) && (iy < in_h);
                const int8_t *w_row = filter_data + (ky * 3 * ch) + 4 * g;
                for (int32_t kx = 0; kx < 3; ++kx)
                {
                    const int32_t ix = ox + kx - 1;
                    const bool    col_ok = (ix >= 0) && (ix < in_w);
                    const bool    in_ok  = row_ok && col_ok;
                    int32x4_t in_s32;
                    if (in_ok)
                    {
                        const int8_t *in_ptr = input_data + ((iy * in_w + ix) * ch) + 4 * g;
                        in_s32 = vldrbq_s32(in_ptr);
                        in_s32 = vaddq_n_s32(in_s32, input_offset);
                    }
                    else
                    {
                        in_s32 = vdupq_n_s32(0);
                    }
                    const int8_t *w_ptr = w_row + kx * ch;
                    int32x4_t     wt_s32 = vldrbq_s32(w_ptr);
                    int32x4_t prod = vmulq_s32(in_s32, wt_s32);
                    acc             = vaddq_s32(acc, prod);
            }
            }
               int32x4_t mult  = vldrwq_s32(quant_params->multiplier + 4 * g);
               int32x4_t shift = vldrwq_s32(quant_params->shift     + 4 * g);
               mve_pred16_t p = vctp32q(CH_BLOCK);
               acc = arm_requantize_mve_32x4(acc, mult, shift);
               acc = vaddq_n_s32(acc, output_offset);
               
               acc = vmaxq_s32(acc, vdupq_n_s32(output_activation_min));
               acc = vminq_s32(acc, vdupq_n_s32(output_activation_max));
               print_vector(acc);
               int8_t *out_p = output_data + ((oy * out_w + ox) * ch) + 4 * g;
               vstrbq_p_s32(out_p, acc, p);
               
           } 
       }
   }
   return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of NNConv group
 */
