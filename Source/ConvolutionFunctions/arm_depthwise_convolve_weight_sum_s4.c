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
 * Title:        arm_depthwise_convolve_weight_sum_s4.c
 * Description:  s4 version of depthwise convolution sum of weights calculation.
 *
 * $Date:        30 August 2025
 * $Revision:    V.4.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include <stdint.h>
#if defined(ARM_MATH_MVEI)
    #include "arm_mve.h"
#endif

arm_cmsis_nn_status arm_depthwise_weight_sum_s4(int32_t *vector_sum_buf,
                                                const int8_t *weights_s4,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims,
                                                const int32_t lhs_offset,
                                                const int32_t *bias_data)
{
    if (!vector_sum_buf || !weights_s4 || !input_dims || !filter_dims || !output_dims)
        return ARM_CMSIS_NN_ARG_ERROR;

    const uint32_t out_c = (uint32_t)output_dims->c;
    const uint32_t K_dw = (uint32_t)filter_dims->h * (uint32_t)filter_dims->w;
    if (out_c == 0 || K_dw == 0)
        return ARM_CMSIS_NN_ARG_ERROR;

    for (uint32_t oc = 0; oc < out_c; ++oc)
    {
        int32_t sum_w = 0;

#if defined(ARM_MATH_MVEI)
        if ((out_c & 1u) == 0u)
        {
            const uint32_t base_byte = (oc >> 1);
            const uint32_t stride_byte = (out_c >> 1);
            const uint8_t hi_sel = (uint8_t)(oc & 1u);

            uint32_t k = 0;
            for (; k + 16 <= K_dw; k += 16)
            {
                uint32_t b0 = base_byte + k * stride_byte;
                int32_t acc = 0;
                for (int i = 0; i < 16; ++i)
                {
                    uint8_t b = ((const uint8_t *)weights_s4)[b0 + (uint32_t)i * stride_byte];
                    uint8_t u4 = hi_sel ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0Fu);
                    acc += (int32_t)((int8_t)((u4 << 4)) >> 4);
                }
                sum_w += acc;
            }
            for (; k < K_dw; ++k)
            {
                const uint32_t byte_idx = base_byte + k * stride_byte;
                const uint8_t b = (uint8_t)weights_s4[byte_idx];
                const uint8_t u4 = hi_sel ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
                sum_w += (int32_t)((int8_t)((u4 << 4)) >> 4);
            }
        }
        else
        {
            uint32_t nib = oc;
            for (uint32_t k = 0; k < K_dw; ++k, nib += out_c)
            {
                const uint32_t byte_idx = nib >> 1;
                const uint8_t b = (uint8_t)weights_s4[byte_idx];
                const uint8_t u4 = (nib & 1u) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
                sum_w += (int32_t)((int8_t)((u4 << 4)) >> 4);
            }
        }
#elif defined(ARM_MATH_DSP)
        if ((out_c & 1u) == 0u)
        {
            const uint32_t base_byte = (oc >> 1);
            const uint32_t stride_byte = (out_c >> 1);
            const uint8_t hi_sel = (uint8_t)(oc & 1u);

            uint32_t k = 0;
            for (; k + 4 <= K_dw; k += 4)
            {
                const uint8_t b0 = (uint8_t)weights_s4[base_byte + (k + 0) * stride_byte];
                const uint8_t b1 = (uint8_t)weights_s4[base_byte + (k + 1) * stride_byte];
                const uint8_t b2 = (uint8_t)weights_s4[base_byte + (k + 2) * stride_byte];
                const uint8_t b3 = (uint8_t)weights_s4[base_byte + (k + 3) * stride_byte];

                const uint8_t u0 = hi_sel ? (uint8_t)(b0 >> 4) : (uint8_t)(b0 & 0x0F);
                const uint8_t u1 = hi_sel ? (uint8_t)(b1 >> 4) : (uint8_t)(b1 & 0x0F);
                const uint8_t u2 = hi_sel ? (uint8_t)(b2 >> 4) : (uint8_t)(b2 & 0x0F);
                const uint8_t u3 = hi_sel ? (uint8_t)(b3 >> 4) : (uint8_t)(b3 & 0x0F);

                sum_w += (int32_t)((int8_t)((u0 << 4)) >> 4);
                sum_w += (int32_t)((int8_t)((u1 << 4)) >> 4);
                sum_w += (int32_t)((int8_t)((u2 << 4)) >> 4);
                sum_w += (int32_t)((int8_t)((u3 << 4)) >> 4);
            }
            for (; k < K_dw; ++k)
            {
                const uint8_t b = (uint8_t)weights_s4[base_byte + k * stride_byte];
                const uint8_t u4 = hi_sel ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
                sum_w += (int32_t)((int8_t)((u4 << 4)) >> 4);
            }
        }
        else
        {
            uint32_t nib = oc;
            for (uint32_t k = 0; k < K_dw; ++k, nib += out_c)
            {
                const uint32_t byte_idx = nib >> 1;
                const uint8_t b = (uint8_t)weights_s4[byte_idx];
                const uint8_t u4 = (nib & 1u) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
                sum_w += (int32_t)((int8_t)((u4 << 4)) >> 4);
            }
        }
#else
        {
            uint32_t nib = oc;
            for (uint32_t k = 0; k < K_dw; ++k, nib += out_c)
            {
                const uint32_t byte_idx = nib >> 1;
                const uint8_t b = (uint8_t)weights_s4[byte_idx];
                const uint8_t u4 = (nib & 1u) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
                sum_w += (int32_t)((int8_t)((u4 << 4)) >> 4);
            }
        }
#endif

        const int32_t b = (bias_data) ? bias_data[oc] : 0;
        vector_sum_buf[oc] = b + lhs_offset * sum_w;
    }

    return ARM_CMSIS_NN_SUCCESS;
}
