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

    /* Calculate number of 4-channel vector iterations needed */
    int32_t loop_count = (active_ch + 3) / 4;

    /* Track remaining channels to process in current iteration */
    uint32_t num_ch_to_process = active_ch;

    /* Pointer to pre-computed weight sums for optimization */
    const int32_t *weight_sum_base = weight_sum_buf;


    /* Process channels in groups of 4 using MVE vectorized operations.
    Each iteration processes 4 channels across 4 output positions. */

    for (int i_loop_cnt = 0, offset = 0; i_loop_cnt < loop_count;
    num_ch_to_process -= 4, offset += 4, out += 4, i_loop_cnt++)
    {
        mve_pred16_t p = vctp32q(num_ch_to_process);
        // per-channel state
        int32x4_t base  = vldrwq_z_s32(&weight_sum_base[offset], p);
        int32x4_t mult  = vldrwq_z_s32(&out_mult[offset],        p);
        int32x4_t shift  = vldrwq_z_s32(&out_shift[offset],       p);

        // Initialize 4 output accumulators
        int32x4_t out_0 = base;
        int32x4_t out_1 = base;
        int32x4_t out_2 = base;
        int32x4_t out_3 = base; 


        // Kernel weights pointer for current 4-channel group
        const int8_t *rhs_0 = rhs + offset;

        // Input data pointers for 4 output positions
        const int8_t *lhs_0 = lhs + 0 * total_ch + offset;
        const int8_t *lhs_1 = lhs + 1 * total_ch + offset;
        const int8_t *lhs_2 = lhs + 2 * total_ch + offset;
        const int8_t *lhs_3 = lhs + 3 * total_ch + offset;

        // Core convolution loop: for each kernel position, perform vectorized MACs across all 4 outputs
        for (int i_row_x_col = 0; i_row_x_col < row_x_col; i_row_x_col++)
        {
            // Load 4 weights for current kernel position and channel group
            const int32x4_t ker_0 = vldrbq_z_s32(rhs_0, p);

            // vectorized multiply-accumulate for 4 output positions
            int32x4_t ip_0 = vldrbq_z_s32(lhs_0, p);
            out_0 += vmulq_s32(ip_0, ker_0);

            int32x4_t ip_1 = vldrbq_z_s32(lhs_1, p);
            out_1 += vmulq_s32(ip_1, ker_0);

            int32x4_t ip_2 = vldrbq_z_s32(lhs_2, p);
            out_2 += vmulq_s32(ip_2, ker_0);

            int32x4_t ip_3 = vldrbq_z_s32(lhs_3, p); 
            out_3 += vmulq_s32(ip_3, ker_0);

            // Advance input data pointers to next kernel position
            lhs_0 += total_ch;
            lhs_1 += total_ch;
            lhs_2 += total_ch;
            lhs_3 += total_ch;

            // Advance kernel pointer to next kernel position 
            rhs_0 += total_ch;
        }

        // Quantization and clamping for all 4 output positions
        out_0 = arm_requantize_mve_32x4(out_0, mult, shift);
        out_0 = vaddq_n_s32(out_0, out_offset);
        out_0 = vmaxq_s32(out_0, vdupq_n_s32(activation_min));
        out_0 = vminq_s32(out_0, vdupq_n_s32(activation_max));
        vstrbq_p_s32(out, out_0, p);

        out_1 = arm_requantize_mve_32x4(out_1, mult, shift);
        out_1 = vaddq_n_s32(out_1, out_offset);
        out_1 = vmaxq_s32(out_1, vdupq_n_s32(activation_min));
        out_1 = vminq_s32(out_1, vdupq_n_s32(activation_max));
        vstrbq_p_s32(out + total_ch, out_1, p);

        out_2 = arm_requantize_mve_32x4(out_2, mult, shift);
        out_2 = vaddq_n_s32(out_2, out_offset);
        out_2 = vmaxq_s32(out_2, vdupq_n_s32(activation_min));
        out_2 = vminq_s32(out_2, vdupq_n_s32(activation_max));
        vstrbq_p_s32(out + 2 * total_ch, out_2, p);

        out_3 = arm_requantize_mve_32x4(out_3, mult, shift);
        out_3 = vaddq_n_s32(out_3, out_offset);
        out_3 = vmaxq_s32(out_3, vdupq_n_s32(activation_min));
        out_3 = vminq_s32(out_3, vdupq_n_s32(activation_max));
        vstrbq_p_s32(out + 3 * total_ch, out_3, p);
    }

    return ARM_CMSIS_NN_SUCCESS;
}
#endif