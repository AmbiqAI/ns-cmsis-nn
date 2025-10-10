/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_depthwise_convolve_weight_sum.c
 * Description:  s8 version of convolution using symmetric quantization.
 *
 * $Date:        30 April 2025
 * $Revision:    V.4.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

arm_cmsis_nn_status arm_depthwise_convolve_weight_sum(int32_t* vector_sum_buf,
                                                      int8_t* scratch_buf,
                                                      const int8_t *rhs,
                                                      const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims,
                                                      const int32_t lhs_offset,
                                                      const int32_t *bias_data)
{
    if (vector_sum_buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    (void)scratch_buf;
    (void)dw_conv_params;
    (void)input_dims;

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    const int32_t kernel_elems = (int32_t)filter_dims->h * filter_dims->w;
    const int32_t out_ch       = output_dims->c;
    for (int32_t oc = 0; oc < out_ch; ++oc)
    {
        int32_t  ksum   = 0;
        const int8_t *kptr = rhs + oc;
        int32_t  remain = kernel_elems;
        while (remain >= 4)
        {
            uint32_t packed =
                (uint32_t)(uint8_t)kptr[0]            |
               ((uint32_t)(uint8_t)kptr[out_ch]  <<  8) |
               ((uint32_t)(uint8_t)kptr[2*out_ch] << 16)|
               ((uint32_t)(uint8_t)kptr[3*out_ch] << 24);

            ksum = SMLAD((int32_t)packed, 0x00010001, ksum);

            kptr   += 4 * out_ch;
            remain -= 4;
        }
        while (remain--)
        {
            ksum += *kptr;
            kptr += out_ch;
        }
        int32_t acc = ksum * lhs_offset;
        if (bias_data)
            acc += bias_data[oc];

        vector_sum_buf[oc] = acc;
    }
    return ARM_CMSIS_NN_SUCCESS;

#elif defined(ARM_MATH_MVEI)
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t output_channels = output_dims->c;

    if (bias_data)
        memcpy(vector_sum_buf, bias_data, output_channels * sizeof(int32_t));
    else
        memset(vector_sum_buf, 0,        output_channels * sizeof(int32_t));

    if (lhs_offset)
    {
        int32_t          ch_left   = output_channels;
        const int8_t    *rhs_base  = rhs;
        int32_t         *dst_base  = vector_sum_buf;
        const int32_t    row_x_col = kernel_x * kernel_y;

        for (int blk = 0; blk < (output_channels + 3) / 4; ++blk)
        {
            mve_pred16_t p       = vctp32q(ch_left);
            int32x4_t    ker_sum = vdupq_n_s32(0);
            const int8_t *rhs_it = rhs_base;

            for (int32_t k = 0; k < row_x_col; ++k)
            {
                int32x4_t w = vldrbq_z_s32(rhs_it, p);
                ker_sum      = vaddq_s32(ker_sum, w);
                rhs_it      += output_channels;
            }

            ker_sum = vmulq_n_s32(ker_sum, lhs_offset);
            if (ch_left >= 4)
            {
                vst1q_s32(dst_base, vaddq_s32(vld1q_s32(dst_base), ker_sum));
            }
            else
            {
                for (int i = 0; i < ch_left; ++i)
                    dst_base[i] += ker_sum[i];
            }

            /* Advance bases for next 4-ch block */
            rhs_base += 4;
            dst_base += 4;
            ch_left  -= 4;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;

#else
    const int32_t kernel_elems = (int32_t)filter_dims->h * filter_dims->w;
    const int32_t out_ch = output_dims->c;
    for (int32_t oc = 0; oc < out_ch; ++oc)
    {
        int32_t  ksum = 0;
        const int8_t *kptr = rhs + oc;
        for (int32_t k = 0; k < kernel_elems; ++k)
        {
            ksum += *kptr;
            kptr += out_ch;
        }
        int32_t acc = ksum * lhs_offset;
        if (bias_data)
            acc += bias_data[oc];
        vector_sum_buf[oc] = acc;
    }

    return ARM_CMSIS_NN_SUCCESS;
#endif 
}