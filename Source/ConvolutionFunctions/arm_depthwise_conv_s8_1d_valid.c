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
 * Title:        arm_depthwise_conv_s8_1d_valid.c
 * Description:  Optimized s8 1D depthwise separable convolution function for
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

#if defined(ARM_MATH_MVEI)
arm_cmsis_nn_status arm_depthwise_conv_s8_1d_valid(const int32_t* weight_sum_buf,
        const int8_t *lhs,
        const int8_t *rhs,
        const int32_t input_offset,
        const int32_t active_ch,
        const int32_t total_ch,
        const int32_t *out_shift,
        const int32_t *out_mult,
        const int32_t out_offset,
        const int32_t activation_min,
        const int32_t activation_max,
        const uint16_t row_x_col,
        const int32_t *const output_bias,
        int8_t *out)
{

    /* Unused parameters in MVE implementation (handled by weight sum approach) */
    (void) input_offset;
    (void) output_bias;

    /* Pointer to pre-computed weight sums for optimization */
    const int32_t *weight_sum_base = weight_sum_buf;


    /* Process channels in groups of 4 using MVE vectorized operations.
    Each iteration processes 4 channels across 4 output positions. */
    int full = (active_ch & ~3);                 // multiple of 4
    int tail = (active_ch & 3);                     // remaining channels (0-3)
    // Full groups (no predicates)
    for (int off = 0; off < full; off += 4, out += 4) {
        int32x4_t base  = vldrwq_s32(weight_sum_base + off);
        int32x4_t multv = vldrwq_s32(out_mult + off);
        int32x4_t shftv = vldrwq_s32(out_shift + off);

        int32x4_t o0=base,o1=base,o2=base,o3=base;
        const int8_t *r = rhs + off;
        const int8_t *a0 = lhs + 0 * total_ch + off;
        const int8_t *a1 = lhs + 1 * total_ch + off;
        const int8_t *a2 = lhs + 2 * total_ch + off;
        const int8_t *a3 = lhs + 3 * total_ch + off;

        for (int k = 0; k < row_x_col; k++) {
            int32x4_t ker = vldrbq_s32(r);
            o0 = vaddq_s32(o0, vmulq_s32(vldrbq_s32(a0), ker));
            o1 = vaddq_s32(o1, vmulq_s32(vldrbq_s32(a1), ker));
            o2 = vaddq_s32(o2, vmulq_s32(vldrbq_s32(a2), ker));
            o3 = vaddq_s32(o3, vmulq_s32(vldrbq_s32(a3), ker));
            a0 += total_ch; a1 += total_ch; a2 += total_ch; a3 += total_ch;
            r  += total_ch;
        }

        // quant/clamp
        o0 = arm_requantize_mve_32x4(o0, multv, shftv);
        o1 = arm_requantize_mve_32x4(o1, multv, shftv);
        o2 = arm_requantize_mve_32x4(o2, multv, shftv);
        o3 = arm_requantize_mve_32x4(o3, multv, shftv);

        o0 = vaddq_n_s32(o0, out_offset);
        o1 = vaddq_n_s32(o1, out_offset);
        o2 = vaddq_n_s32(o2, out_offset);
        o3 = vaddq_n_s32(o3, out_offset);

        const int32x4_t vminv = vdupq_n_s32(activation_min);
        const int32x4_t vmaxv = vdupq_n_s32(activation_max);
        o0 = vminq_s32(vmaxq_s32(o0, vminv), vmaxv);
        o1 = vminq_s32(vmaxq_s32(o1, vminv), vmaxv);
        o2 = vminq_s32(vmaxq_s32(o2, vminv), vmaxv);
        o3 = vminq_s32(vmaxq_s32(o3, vminv), vmaxv);

        vstrbq_s32(out + 0 * total_ch, o0);
        vstrbq_s32(out + 1 * total_ch, o1);
        vstrbq_s32(out + 2 * total_ch, o2);
        vstrbq_s32(out + 3 * total_ch, o3);
    }

    // Tail case with predication
    if (tail) {
        int offset = full;
        mve_pred16_t p = vctp32q(tail);

        // per-channel state
        int32x4_t base  = vldrwq_z_s32(weight_sum_base + offset, p);
        int32x4_t mult  = vldrwq_z_s32(out_mult        + offset, p);
        int32x4_t shift = vldrwq_z_s32(out_shift       + offset, p);

        int32x4_t out_0 = base, out_1 = base, out_2 = base, out_3 = base;

        // pointers for this 4-lane group
        const int8_t *rhs_0 = rhs + offset;
        const int8_t *lhs_0 = lhs + 0 * total_ch + offset;
        const int8_t *lhs_1 = lhs + 1 * total_ch + offset;
        const int8_t *lhs_2 = lhs + 2 * total_ch + offset;
        const int8_t *lhs_3 = lhs + 3 * total_ch + offset;

        // inner taps (predicated loads)
        for (int k = 0; k < row_x_col; k++) {
            const int32x4_t ker = vldrbq_z_s32(rhs_0, p);
            out_0 = vaddq_s32(out_0, vmulq_s32(vldrbq_z_s32(lhs_0, p), ker));
            out_1 = vaddq_s32(out_1, vmulq_s32(vldrbq_z_s32(lhs_1, p), ker));
            out_2 = vaddq_s32(out_2, vmulq_s32(vldrbq_z_s32(lhs_2, p), ker));
            out_3 = vaddq_s32(out_3, vmulq_s32(vldrbq_z_s32(lhs_3, p), ker));
            lhs_0 += total_ch; lhs_1 += total_ch; lhs_2 += total_ch; lhs_3 += total_ch;
            rhs_0 += total_ch;
        }

        // requant + clamp + predicated stores
        out_0 = arm_requantize_mve_32x4(out_0, mult, shift);
        out_1 = arm_requantize_mve_32x4(out_1, mult, shift);
        out_2 = arm_requantize_mve_32x4(out_2, mult, shift);
        out_3 = arm_requantize_mve_32x4(out_3, mult, shift);

        const int32x4_t vout_off = vdupq_n_s32(out_offset);
        const int32x4_t vminv    = vdupq_n_s32(activation_min);
        const int32x4_t vmaxv    = vdupq_n_s32(activation_max);

        out_0 = vminq_s32(vmaxq_s32(vaddq_s32(out_0, vout_off), vminv), vmaxv);
        out_1 = vminq_s32(vmaxq_s32(vaddq_s32(out_1, vout_off), vminv), vmaxv);
        out_2 = vminq_s32(vmaxq_s32(vaddq_s32(out_2, vout_off), vminv), vmaxv);
        out_3 = vminq_s32(vmaxq_s32(vaddq_s32(out_3, vout_off), vminv), vmaxv);

        vstrbq_p_s32(out + 0 * total_ch, out_0, p);
        vstrbq_p_s32(out + 1 * total_ch, out_1, p);
        vstrbq_p_s32(out + 2 * total_ch, out_2, p);
        vstrbq_p_s32(out + 3 * total_ch, out_3, p);
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif