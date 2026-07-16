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
 * Title:        arm_vector_sum_s4
 * Description:  Generic function for calculating int4-packed weight vector sums
 *
 * $Date:        05 Sep 2024
 * $Revision:    V.3.0.0
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
 * @addtogroup FC
 * @{
 */

/*
 * S4 vector sum function in preparation for e.g. kernel sums in fully connected and matrix multiplication layer
 * function
 *
 * Refer header file for details.
 *
 */
#include "arm_nn_types.h"
#include <stdint.h>
#if defined ARM_MATH_MVEI
    #include "arm_mve.h"
#endif

static inline int8_t s4_at(const int8_t *w, uint32_t nib_idx)
{
    const uint8_t b = (uint8_t)w[nib_idx >> 1];
    const uint8_t u = (nib_idx & 1u) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
    return s4_from_u4(u);
}

arm_cmsis_nn_status arm_vector_sum_s4(int32_t *vector_sum_buf,
                                      const int8_t *weights_s4,
                                      const cmsis_nn_dims *input_dims,
                                      const cmsis_nn_dims *output_dims,
                                      const int32_t lhs_offset,
                                      const int32_t *bias_data)
{
    if (!vector_sum_buf || !weights_s4 || !input_dims || !output_dims)
        return ARM_CMSIS_NN_ARG_ERROR;

    const uint32_t out_c = (uint32_t)output_dims->c;
    const uint32_t K = (uint32_t)input_dims->w * (uint32_t)input_dims->h * (uint32_t)input_dims->c;
    if (out_c == 0u || K == 0u)
        return ARM_CMSIS_NN_ARG_ERROR;

    uint32_t base_nib = 0;
    for (uint32_t oc = 0; oc < out_c; ++oc, base_nib += K)
    {
        int32_t sum_w = 0;

        uint32_t consumed = 0;
        if (base_nib & 1u)
        {
            sum_w += (int32_t)s4_at(weights_s4, base_nib);
            consumed = 1;
        }

        const uint8_t *p = (const uint8_t *)&weights_s4[(base_nib + consumed) >> 1];
        uint32_t pairs = (K - consumed) >> 1;

#if defined(ARM_MATH_MVEI)
        while (pairs >= 16)
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

            sum_w += vaddvq_s16(vaddq_s16(lo_lo, lo_hi));
            sum_w += vaddvq_s16(vaddq_s16(hi_lo, hi_hi));

            p += 16;
            pairs -= 16;
        }
#elif defined(ARM_MATH_DSP)
        while (pairs >= 4)
        {
            uint8_t b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3];
            sum_w += (int32_t)s4_from_u4(b0 & 0x0F) + (int32_t)s4_from_u4(b0 >> 4);
            sum_w += (int32_t)s4_from_u4(b1 & 0x0F) + (int32_t)s4_from_u4(b1 >> 4);
            sum_w += (int32_t)s4_from_u4(b2 & 0x0F) + (int32_t)s4_from_u4(b2 >> 4);
            sum_w += (int32_t)s4_from_u4(b3 & 0x0F) + (int32_t)s4_from_u4(b3 >> 4);
            p += 4;
            pairs -= 4;
        }
#endif

        for (uint32_t i = 0; i < pairs; ++i)
        {
            const uint8_t b = p[i];
            sum_w += (int32_t)s4_from_u4(b & 0x0F);
            sum_w += (int32_t)s4_from_u4(b >> 4);
        }

        if ((K - consumed) & 1u)
        {
            sum_w += (int32_t)s4_at(weights_s4, base_nib + (K - 1));
        }

        const int32_t b = (bias_data) ? bias_data[oc] : 0;
        vector_sum_buf[oc] = b + lhs_offset * sum_w;
    }

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of FC group
 */
