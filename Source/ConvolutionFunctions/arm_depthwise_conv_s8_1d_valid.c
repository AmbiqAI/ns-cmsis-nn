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
 * $Date:        22 March 2023
 * $Revision:    V.3.5.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if defined(ARM_MATH_MVEI)
static inline void arm_depthwise_conv_s8_1d_valid_1out(
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
}

arm_cmsis_nn_status arm_depthwise_conv_s8_1d_valid(const int32_t* weight_sum_buf,
        const int8_t *lhs,
        const int8_t *rhs
        const int32_t input_offset,
        const int32_t active_ch,
        const int32_t total_ch,
        const int32_t *out_shift,
        const int32_t *out_mult
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

//this wrapper requires 1D valid kernel
arm_cmsis_nn_status arm_depthwise_conv_s8_opt_1d_valid_wrapper(const cmsis_nn_context *ctx,
    const cmsis_nn_context *weight_sum_ctx,
    const cmsis_nn_dw_conv_params *dw_conv_params,
    const cmsis_nn_per_channel_quant_params *quant_params,
    const cmsis_nn_dims *input_dims,
    const int8_t *input,
    const cmsis_nn_dims *filter_dims,
    const int8_t *kernel,
    const cmsis_nn_dims *bias_dims,
    const int32_t *bias,
    const cmsis_nn_dims *output_dims,
    int8_t *output)
{
    const int32_t input_ch = input_dims->c;
    const int32_t output_ch = output_dims->c;

    /* Check depth multiplier is 1 */
    if (input_ch != output_ch)
    {
    return ARM_CMSIS_NN_ARG_ERROR;
    }

#ifdef ARM_MATH_MVEI
    (void)bias_dims;
    (void)bias;
    (void)ctx;

    const int32_t input_y  = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t pad_x    = dw_conv_params->padding.w;
    const int32_t pad_y    = dw_conv_params->padding.h;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t *output_shift = quant_params->shift;
    const int32_t *output_mult  = quant_params->multiplier;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t out_offset = dw_conv_params->output_offset;
    const int32_t in_offset  = dw_conv_params->input_offset;
    const int32_t act_min    = dw_conv_params->activation.min;
    const int32_t act_max    = dw_conv_params->activation.max;

    const bool is_1d_valid =
        (input_dims->n == 1) &&
        (input_y == 1) && (output_y == 1) &&
        (kernel_y == 1) &&
        (pad_x == 0) && (pad_y == 0) &&
        (stride_x == 1) && (stride_y == 1) &&
        (dw_conv_params->dilation.w == 1) && (dw_conv_params->dilation.h == 1) &&
        (dw_conv_params->ch_mult == 1);

    if (is_1d_valid)
    {
        // Vectorize across channels and compute 4 outputs per call, no im2col.
        // Requirements:
        // - weight_sum_ctx->buf must be precomputed by arm_depthwise_convolve_weight_sum() in init()
        int32_t *weight_sum_buf = (int32_t *)weight_sum_ctx->buf;
        if (weight_sum_buf == NULL)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        const int32_t C = input_ch;          // channels
        const int32_t Wout = output_x;       // width outputs

        const int32_t ch_loop = (C + (CH_IN_BLOCK_MVE - 1)) / CH_IN_BLOCK_MVE;
        int32_t remaining_ch = C;

        for (int i_ch = 0; i_ch < ch_loop; i_ch++)
        {
            const int32_t block_offset = i_ch * CH_IN_BLOCK_MVE;
            const int32_t active_ch = MIN(CH_IN_BLOCK_MVE, remaining_ch);
            remaining_ch -= CH_IN_BLOCK_MVE;

            // Slide across width in tiles of 4 outputs
            int x0 = 0;
            for (; x0 + 3 < Wout; x0 += 4)
            {
                // Base pointers into input/output at column x0 and channel block
                const int8_t *lhs_base = input  + (x0 * C) + block_offset;
                int8_t       *out_base = output + (x0 * C) + block_offset;

                // Call 4-output sliding-window microkernel
                arm_cmsis_nn_status status = arm_depthwise_conv_s8_1d_valid(
                    weight_sum_buf + block_offset, 
                    lhs_base,
                    kernel + block_offset,
                    in_offset,
                    active_ch,
                    C,
                    output_shift + block_offset,
                    output_mult  + block_offset,
                    out_offset,
                    act_min,
                    act_max,
                    kernel_x,
                    NULL,             // unused (compensation folded in base[])
                    out_base);

                if (status != ARM_CMSIS_NN_SUCCESS) return status;
            }

            // Tail: 0..3 leftover outputs use 1-output sliding microkernel
            for (; x0 < Wout; x0++)
            {
                const int8_t *lhs_col0 = input  + (x0 * C) + block_offset;
                int8_t       *out_col  = output + (x0 * C) + block_offset;

                arm_depthwise_conv_s8_1d_valid_1out(
                    weight_sum_buf + block_offset,
                    lhs_col0,
                    kernel + block_offset,
                    active_ch,
                    C,
                    output_shift + block_offset,
                    output_mult  + block_offset,
                    out_offset,
                    act_min,
                    act_max,
                    kernel_x,
                    out_col);
            }
        }

        return ARM_CMSIS_NN_SUCCESS;
    }
    else {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#else
    (void)weight_sum_ctx;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    return arm_depthwise_conv_s8(ctx,
                                dw_conv_params,
                                quant_params,
                                input_dims,
                                input,
                                filter_dims,
                                kernel,
                                bias_dims,
                                bias,
                                output_dims,
                                output);
#endif  // ARM_MATH_MVEI
}
