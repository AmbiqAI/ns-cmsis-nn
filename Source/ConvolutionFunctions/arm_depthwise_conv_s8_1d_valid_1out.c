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
 * Title:        arm_depthwise_conv_s8_1d_valid_1out.c
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
arm_cmsis_nn_status arm_depthwise_conv_s8_1d_valid_1out(
    const int32_t *weight_sum_base,
    const int8_t  *lhs_col0,
    const int8_t  *rhs,
    int32_t        active_ch,
    int32_t        total_ch,
    const int32_t *out_shift,
    const int32_t *out_mult, 
    int32_t        out_offset,
    int32_t        act_min,
    int32_t        act_max,
    int            kernel_x,
    int8_t        *out_col)
{
    uint32_t rem = active_ch;

    for (int off = 0; off < active_ch; off += 4, rem -= 4, out_col += 4)
    {
        mve_pred16_t p = vctp32q(rem);

        int32x4_t acc  = vldrwq_z_s32(weight_sum_base + off, p);
        int32x4_t mult = vldrwq_z_s32(out_mult         + off, p);
        int32x4_t shft = vldrwq_z_s32(out_shift        + off, p);

        // loop over taps
        const int8_t *x_t = lhs_col0 + off;
        const int8_t *w_t = rhs       + off;
        for (int t = 0; t < kernel_x; ++t) {
            int32x4_t xv = vldrbq_z_s32(x_t, p);
            int32x4_t wv = vldrbq_z_s32(w_t, p);
            acc = vaddq_s32(acc, vmulq_s32(xv, wv));
            x_t += total_ch; 
            w_t += total_ch;
        }

        acc = arm_requantize_mve_32x4(acc, mult, shft);
        acc = vaddq_n_s32(acc, out_offset);
        acc = vmaxq_s32(acc, vdupq_n_s32(act_min));
        acc = vminq_s32(acc, vdupq_n_s32(act_max));
        vstrbq_p_s32(out_col, acc, p);
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif