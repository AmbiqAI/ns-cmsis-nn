/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
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
 * Title:        arm_convolve_1x1_s8_fast.c
 * Description:  Fast s8 version of 1x1 convolution (non-square shape)
 *
 * $Date:        05 November 2024
 * $Revision:    V.3.6.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_FORCEINLINE int32x4_t arm_convolve_1x1_short_k_quant(int32x4_t res,
                                                              const int32_t sum,
                                                              const int32_t mult,
                                                              const int32_t shift,
                                                              const bool rshift_only,
                                                              const int32_t dst_offset,
                                                              const int32x4_t v_min,
                                                              const int32x4_t v_max)
{
    res = vaddq_n_s32(res, sum);
    res = rshift_only ? arm_requantize_mve_rshift(res, mult, shift) : arm_requantize_mve(res, mult, shift);
    res = vaddq_n_s32(res, dst_offset);
    res = vmaxq_s32(res, v_min);
    return vminq_s32(res, v_max);
}

/*
 * Pointwise path for an input depth of at most 16, where one int8 vector holds a whole dot product. Output
 * channels are the outer loop so the filter row stays in a register across all pixels, and four pixels are
 * reduced, requantized and stored per step. For an input depth of 8 two neighbouring pixels share one load: the
 * filter row sits in the low or the high half of an otherwise zero vector, which selects the pixel.
 */
static __attribute__((noinline)) void arm_convolve_1x1_s8_fast_short_k(const int32_t *weight_sum,
                                                                       const int8_t *lhs,
                                                                       const int8_t *rhs,
                                                                       int8_t *dst,
                                                                       const int32_t *dst_multipliers,
                                                                       const int32_t *dst_shifts,
                                                                       const int32_t lhs_rows,
                                                                       const int32_t rhs_rows,
                                                                       const int32_t rhs_cols,
                                                                       const int32_t dst_offset,
                                                                       const int32_t activation_min,
                                                                       const int32_t activation_max)
{
    const uint32x4_t scatter = vmulq_n_u32(vidupq_n_u32(0, 1), (uint32_t)rhs_rows);
    const int32x4_t v_min = vdupq_n_s32(activation_min);
    const int32x4_t v_max = vdupq_n_s32(activation_max);
    const int32_t row_step = 4 * rhs_rows;
    const int32_t tail = lhs_rows & 3;
    const int32_t body_rows = lhs_rows - tail;

    for (int32_t i_ch = 0; i_ch < rhs_rows; i_ch++)
    {
        const int32_t sum = weight_sum[i_ch];
        const int32_t mult = dst_multipliers[i_ch];
        const int32_t shift = dst_shifts[i_ch];
        const bool rshift_only = arm_nn_requantize_rshift_only(&mult, &shift, 1);
        const int8_t *ip = lhs;
        int8_t *out = dst + i_ch;
        int32_t acc[4];

        if (rhs_cols == 8)
        {
            int8_t w_buf[24] = {0};
            arm_memcpy_s8(&w_buf[8], rhs + i_ch * 8, 8);
            const int8x16_t w_hi = vldrbq_s8(&w_buf[0]);
            const int8x16_t w_lo = vldrbq_s8(&w_buf[8]);

            for (int32_t i_row = 0; i_row < body_rows; i_row += 4)
            {
                const int8x16_t x01 = vldrbq_s8(ip);
                const int8x16_t x23 = vldrbq_s8(ip + 16);
                ip += 32;
                acc[0] = vmladavq_s8(x01, w_lo);
                acc[1] = vmladavq_s8(x01, w_hi);
                acc[2] = vmladavq_s8(x23, w_lo);
                acc[3] = vmladavq_s8(x23, w_hi);
                const int32x4_t res = arm_convolve_1x1_short_k_quant(
                    vldrwq_s32(acc), sum, mult, shift, rshift_only, dst_offset, v_min, v_max);
                vstrbq_scatter_offset_s32(out, scatter, res);
                out += row_step;
            }
            if (tail)
            {
                const int8x16_t x01 = vldrbq_z_s8(ip, vctp8q((uint32_t)tail * 8));
                const int8x16_t x23 = vldrbq_z_s8(ip + 16, vctp8q(tail > 2 ? 8 : 0));
                acc[0] = vmladavq_s8(x01, w_lo);
                acc[1] = vmladavq_s8(x01, w_hi);
                acc[2] = vmladavq_s8(x23, w_lo);
                acc[3] = vmladavq_s8(x23, w_hi);
                const int32x4_t res = arm_convolve_1x1_short_k_quant(
                    vldrwq_s32(acc), sum, mult, shift, rshift_only, dst_offset, v_min, v_max);
                vstrbq_scatter_offset_p_s32(out, scatter, res, vctp32q((uint32_t)tail));
            }
        }
        else
        {
            const mve_pred16_t p_k = vctp8q((uint32_t)rhs_cols);
            const int8x16_t w = vldrbq_z_s8(rhs + i_ch * rhs_cols, p_k);

            for (int32_t i_row = 0; i_row < body_rows; i_row += 4)
            {
                acc[0] = vmladavq_s8(vldrbq_z_s8(ip, p_k), w);
                acc[1] = vmladavq_s8(vldrbq_z_s8(ip + rhs_cols, p_k), w);
                acc[2] = vmladavq_s8(vldrbq_z_s8(ip + 2 * rhs_cols, p_k), w);
                acc[3] = vmladavq_s8(vldrbq_z_s8(ip + 3 * rhs_cols, p_k), w);
                ip += 4 * rhs_cols;
                const int32x4_t res = arm_convolve_1x1_short_k_quant(
                    vldrwq_s32(acc), sum, mult, shift, rshift_only, dst_offset, v_min, v_max);
                vstrbq_scatter_offset_s32(out, scatter, res);
                out += row_step;
            }
            if (tail)
            {
                for (int32_t i = 0; i < 4; i++)
                {
                    acc[i] = i < tail ? vmladavq_s8(vldrbq_z_s8(ip + i * rhs_cols, p_k), w) : 0;
                }
                const int32x4_t res = arm_convolve_1x1_short_k_quant(
                    vldrwq_s32(acc), sum, mult, shift, rshift_only, dst_offset, v_min, v_max);
                vstrbq_scatter_offset_p_s32(out, scatter, res, vctp32q((uint32_t)tail));
            }
        }
    }
}
#endif

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Fast s8 version for 1x1 convolution (non-square shape)
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_convolve_1x1_s8_fast(const cmsis_nn_context *ctx,
                                             const cmsis_nn_context *weight_sum_ctx,
                                             const cmsis_nn_conv_params *conv_params,
                                             const cmsis_nn_per_channel_quant_params *quant_params,
                                             const cmsis_nn_dims *input_dims,
                                             const int8_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const int8_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const int32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             int8_t *output_data)
{
    if (conv_params->padding.w != 0 || conv_params->padding.h != 0 || conv_params->stride.w != 1 ||
        conv_params->stride.h != 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEI)
    /* Only the MVE path of arm_nn_mat_mult_nt_t_s8() reads the per-channel weight sums. Diagnose a
       missing buffer here rather than dereferencing NULL and silently returning garbage output. */
    if (weight_sum_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#endif

    (void)filter_dims;
    (void)bias_dims;

    const int32_t rhs_cols = input_dims->c;
    const int32_t rhs_rows = output_dims->c;
    int32_t lhs_rows = input_dims->w * input_dims->h * input_dims->n;

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI) && defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
    if (ctx->buf != NULL) /* Fall back to non buffered version if no additional memory buffer provided */
    {
        const int32_t batch = input_dims->n;
        const int32_t output_h = output_dims->h;
        const int32_t output_w = output_dims->w;
        const int32_t input_inc = input_dims->w * rhs_cols;

        for (int i_batch = 0; i_batch < batch; i_batch++)
        {
            const int32_t output_ch = output_dims->c;
            const int8_t *ip = input_data;
            int16_t *buffer_a = (int16_t *)ctx->buf;
            int16_t *im2col_buf = (int16_t *)ctx->buf;
            int8_t *out = output_data;
            lhs_rows = 0;

            for (int i_out_y = 0; i_out_y < output_h; i_out_y++, ip += input_inc)
            {
                for (int32_t k_x = 0, i_out_x = 0; i_out_x < output_w; i_out_x++, k_x += rhs_cols)
                {
                    arm_s8_to_s16_unordered_with_offset(ip + k_x, im2col_buf, rhs_cols, conv_params->input_offset);
                    im2col_buf += rhs_cols;
                    lhs_rows++;
                    if (lhs_rows == 2)
                    {
                        out = arm_nn_mat_mult_kernel_s8_s16(filter_data,
                                                            buffer_a,
                                                            output_ch,
                                                            quant_params->shift,
                                                            quant_params->multiplier,
                                                            conv_params->output_offset,
                                                            conv_params->activation.min,
                                                            conv_params->activation.max,
                                                            rhs_cols,
                                                            rhs_cols,
                                                            bias_data,
                                                            out);
                        im2col_buf = buffer_a;
                        lhs_rows = 0;
                    }
                }
                if (out == NULL)
                {
                    return ARM_CMSIS_NN_NO_IMPL_ERROR;
                }
            }

            /* Handle left over columns */
            if (lhs_rows != 0)
            {
                const int8_t *ker_a = filter_data;
                for (int i = 0; i < output_ch; i++)
                {
                    /* Load the accumulator with bias first */
                    int32_t sum = 0;
                    if (bias_data)
                    {
                        sum = bias_data[i];
                    }
                    const int16_t *ip_as_col = buffer_a;

                    /* 4 multiply and accumulates are done in one loop. */
                    uint16_t col_count = rhs_cols >> 2;
                    while (col_count)
                    {
                        int32_t ker_a1, ker_a2;
                        int32_t ip_b1, ip_b2;
                        ker_a = read_and_pad_reordered(ker_a, &ker_a1, &ker_a2);
                        ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
                        sum = SMLAD(ker_a1, ip_b1, sum);
                        ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
                        sum = SMLAD(ker_a2, ip_b2, sum);
                        col_count--;
                    }

                    /* Handle left over mac */
                    col_count = rhs_cols & 0x3;
                    while (col_count)
                    {
                        int8_t ker_a1 = *ker_a++;
                        int16_t ip_b1 = *ip_as_col++;
                        sum += ker_a1 * ip_b1;
                        col_count--;
                    }
                    sum = arm_nn_requantize(sum, quant_params->multiplier[i], quant_params->shift[i]);
                    sum += conv_params->output_offset;
                    sum = ARM_NN_MAX(sum, conv_params->activation.min);
                    sum = ARM_NN_MIN(sum, conv_params->activation.max);
                    *out++ = (int8_t)sum;
                }
            }
            /* Advance to the next batch */
            input_data += (input_dims->w * input_dims->h * rhs_cols);
            output_data += (output_w * output_h * output_ch);
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
#else
    (void)ctx;
#endif

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (rhs_cols <= 16 && lhs_rows >= 8)
    {
        arm_convolve_1x1_s8_fast_short_k(weight_sum_ctx->buf,
                                         input_data,
                                         filter_data,
                                         output_data,
                                         quant_params->multiplier,
                                         quant_params->shift,
                                         lhs_rows,
                                         rhs_rows,
                                         rhs_cols,
                                         conv_params->output_offset,
                                         conv_params->activation.min,
                                         conv_params->activation.max);
        return ARM_CMSIS_NN_SUCCESS;
    }
#endif

    arm_nn_mat_mult_nt_t_s8(weight_sum_ctx->buf,
                            input_data,
                            filter_data,
                            bias_data,
                            output_data,
                            quant_params->multiplier,
                            quant_params->shift,
                            lhs_rows,
                            rhs_rows,
                            rhs_cols,
                            conv_params->input_offset,
                            conv_params->output_offset,
                            conv_params->activation.min,
                            conv_params->activation.max,
                            rhs_rows,
                            rhs_cols);

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
