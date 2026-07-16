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
 * Title:        arm_convolve_weight_sum_s4.c
 * Description:  s4 version of convolution sum of weights calculation
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
#if defined ARM_MATH_MVEI
    #include "arm_mve.h"
#endif

arm_cmsis_nn_status arm_convolve_weight_sum_s4(int32_t *vector_sum_buf,
                                               const int8_t *weights_s4,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               const int32_t lhs_offset,
                                               const int32_t *bias_data)
{
    if (!vector_sum_buf || !weights_s4 || !input_dims || !filter_dims || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const uint32_t out_c = (uint32_t)output_dims->c;
    const uint32_t in_c = (uint32_t)input_dims->c;
    const uint32_t filter_h = (uint32_t)filter_dims->h;
    const uint32_t filter_w = (uint32_t)filter_dims->w;

    const uint32_t K = filter_h * filter_w * in_c;

    uint32_t base_elem = 0;
    for (uint32_t o = 0; o < out_c; ++o, base_elem += K)
    {
        int32_t sum_w = 0;

        uint32_t consumed = 0;
        if ((base_elem & 1u) != 0u)
        {
            sum_w += (int32_t)s4_unpack_elem(weights_s4, base_elem);
            consumed = 1;
        }

        const uint8_t *p = (const uint8_t *)&weights_s4[(base_elem + consumed) >> 1];
        const uint32_t pairs = (K - consumed) >> 1;

#if defined(ARM_MATH_MVEI)

        uint32_t blk = pairs >> 4;
        for (uint32_t b = 0; b < blk; ++b)
        {
            uint8x16_t bytes = vld1q_u8(p);
            uint8x16_t lo_u = vandq_u8(bytes, vdupq_n_u8(0x0F));
            uint8x16_t hi_u = vshrq_n_u8(bytes, 4);

            int8x16_t lo_s = vshrq_n_s8(vshlq_n_s8((int8x16_t)lo_u, 4), 4);
            int8x16_t hi_s = vshrq_n_s8(vshlq_n_s8((int8x16_t)hi_u, 4), 4);

            int16x8_t lo_lo = vmovlbq_s8(lo_s);
            int16x8_t lo_hi = vmovltq_s8(lo_s);
            int16x8_t hi_lo = vmovlbq_s8(hi_s);
            int16x8_t hi_hi = vmovltq_s8(hi_s);

            int16x8_t acc16 = vaddq_s16(vaddq_s16(lo_lo, lo_hi), vaddq_s16(hi_lo, hi_hi));
            sum_w += vaddvq_s16(acc16);

            p += 16;
        }

        uint32_t rem = pairs & 15u;
        for (uint32_t i = 0; i < rem; ++i)
        {
            const uint8_t b8 = p[i];
            sum_w += (int32_t)s4_from_u4(b8 & 0x0Fu);
            sum_w += (int32_t)s4_from_u4((b8 >> 4) & 0x0Fu);
        }
#elif defined(ARM_MATH_DSP)
        uint32_t blk = pairs >> 2;
        for (uint32_t b = 0; b < blk; ++b)
        {
            uint8_t b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3];

            sum_w += (int32_t)s4_from_u4(b0 & 0x0Fu) + (int32_t)s4_from_u4((b0 >> 4) & 0x0Fu);
            sum_w += (int32_t)s4_from_u4(b1 & 0x0Fu) + (int32_t)s4_from_u4((b1 >> 4) & 0x0Fu);
            sum_w += (int32_t)s4_from_u4(b2 & 0x0Fu) + (int32_t)s4_from_u4((b2 >> 4) & 0x0Fu);
            sum_w += (int32_t)s4_from_u4(b3 & 0x0Fu) + (int32_t)s4_from_u4((b3 >> 4) & 0x0Fu);

            p += 4;
        }
        for (uint32_t i = (pairs & ~3u); i < pairs; ++i)
        {
            const uint8_t b8 = p[i - ((pairs & ~3u))];
            sum_w += (int32_t)s4_from_u4(b8 & 0x0Fu);
            sum_w += (int32_t)s4_from_u4((b8 >> 4) & 0x0Fu);
        }
#else
        for (uint32_t i = 0; i < pairs; ++i)
        {
            const uint8_t b8 = p[i];
            sum_w += (int32_t)s4_from_u4(b8 & 0x0Fu);
            sum_w += (int32_t)s4_from_u4((b8 >> 4) & 0x0Fu);
        }
#endif
        if (((K - consumed) & 1u) != 0u)
        {
            sum_w += (int32_t)s4_unpack_elem(weights_s4, base_elem + (K - 1));
        }

        const int32_t b = (bias_data) ? bias_data[o] : 0;
        vector_sum_buf[o] = b + lhs_offset * sum_w;
    }

    return ARM_CMSIS_NN_SUCCESS;
}
