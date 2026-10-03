/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_1x1_s8_short_k.c
 * Description:  Direct-entry s8 1x1 convolution for input depths of 1 to 16.
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
 * @addtogroup NNConv
 * @{
 */

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
 * One output channel over all pixels: four pixels are reduced, requantized and stored per step. For an input depth of
 * 8 two neighbouring pixels share one load: the filter row sits in the low or the high half of an otherwise zero
 * vector, which selects the pixel. rshift_only is a constant at both call sites, so each requantization form gets
 * its own loop.
 */
__STATIC_FORCEINLINE void arm_convolve_1x1_s8_short_k_channel(const int8_t *ip,
                                                              const int8_t *w_row,
                                                              int8_t *dst,
                                                              const int32_t out_first,
                                                              const int32_t row_step,
                                                              const int32_t body_rows,
                                                              const int32_t tail,
                                                              const int32_t rhs_cols,
                                                              const uint32x4_t scatter,
                                                              const int32_t sum,
                                                              const int32_t mult,
                                                              const int32_t shift,
                                                              const bool rshift_only,
                                                              const int32_t dst_offset,
                                                              const int32x4_t v_min,
                                                              const int32x4_t v_max)
{
    /* The store base stays at the channel's first output and the lane offsets advance instead, so that stepping past
     * the last block forms no pointer */
    int8_t *const out = dst + out_first;
    uint32x4_t offs = scatter;
    int32_t acc[4];

    if (rhs_cols == 8)
    {
        int8_t w_buf[24] = {0};
        arm_memcpy_s8(&w_buf[8], w_row, 8);
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
            vstrbq_scatter_offset_s32(out, offs, res);
            offs = vaddq_n_u32(offs, (uint32_t)row_step);
        }
        if (tail)
        {
            const int8x16_t x01 = vldrbq_z_s8(ip, vctp8q((uint32_t)tail * 8));
            /* With fewer than three rows left the second load is fully predicated off; its base stays inside the
             * input so that no pointer is formed past its end. */
            const int8x16_t x23 = vldrbq_z_s8(tail > 2 ? ip + 16 : ip, vctp8q(tail > 2 ? 8 : 0));
            acc[0] = vmladavq_s8(x01, w_lo);
            acc[1] = vmladavq_s8(x01, w_hi);
            acc[2] = vmladavq_s8(x23, w_lo);
            acc[3] = vmladavq_s8(x23, w_hi);
            const int32x4_t res = arm_convolve_1x1_short_k_quant(
                vldrwq_s32(acc), sum, mult, shift, rshift_only, dst_offset, v_min, v_max);
            vstrbq_scatter_offset_p_s32(out, offs, res, vctp32q((uint32_t)tail));
        }
    }
    else
    {
        const mve_pred16_t p_k = vctp8q((uint32_t)rhs_cols);
        const int8x16_t w = vldrbq_z_s8(w_row, p_k);

        for (int32_t i_row = 0; i_row < body_rows; i_row += 4)
        {
            acc[0] = vmladavq_s8(vldrbq_z_s8(ip, p_k), w);
            acc[1] = vmladavq_s8(vldrbq_z_s8(ip + rhs_cols, p_k), w);
            acc[2] = vmladavq_s8(vldrbq_z_s8(ip + 2 * rhs_cols, p_k), w);
            acc[3] = vmladavq_s8(vldrbq_z_s8(ip + 3 * rhs_cols, p_k), w);
            ip += 4 * rhs_cols;
            const int32x4_t res = arm_convolve_1x1_short_k_quant(
                vldrwq_s32(acc), sum, mult, shift, rshift_only, dst_offset, v_min, v_max);
            vstrbq_scatter_offset_s32(out, offs, res);
            offs = vaddq_n_u32(offs, (uint32_t)row_step);
        }
        if (tail)
        {
            for (int32_t i = 0; i < 4; i++)
            {
                acc[i] = i < tail ? vmladavq_s8(vldrbq_z_s8(ip + i * rhs_cols, p_k), w) : 0;
            }
            const int32x4_t res = arm_convolve_1x1_short_k_quant(
                vldrwq_s32(acc), sum, mult, shift, rshift_only, dst_offset, v_min, v_max);
            vstrbq_scatter_offset_p_s32(out, offs, res, vctp32q((uint32_t)tail));
        }
    }
}

/*
 * Pointwise path for an input depth of at most 16, where one int8 vector holds a whole dot product. Output
 * channels are the outer loop so the filter row stays in a register across all pixels.
 */
static void arm_convolve_1x1_s8_short_k_kernel(const int32_t *weight_sum,
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
        int32_t mult = dst_multipliers[i_ch];
        int32_t shift = dst_shifts[i_ch];
        const int8_t *w_row = rhs + i_ch * rhs_cols;
        if (arm_nn_requantize_rshift_only(&mult, &shift, 1))
        {
            arm_convolve_1x1_s8_short_k_channel(lhs,
                                                w_row,
                                                dst,
                                                i_ch,
                                                row_step,
                                                body_rows,
                                                tail,
                                                rhs_cols,
                                                scatter,
                                                weight_sum[i_ch],
                                                mult,
                                                shift,
                                                true,
                                                dst_offset,
                                                v_min,
                                                v_max);
        }
        else
        {
            arm_convolve_1x1_s8_short_k_channel(lhs,
                                                w_row,
                                                dst,
                                                i_ch,
                                                row_step,
                                                body_rows,
                                                tail,
                                                rhs_cols,
                                                scatter,
                                                weight_sum[i_ch],
                                                mult,
                                                shift,
                                                false,
                                                dst_offset,
                                                v_min,
                                                v_max);
        }
    }
}
#endif

/*
 * 1x1 convolution for an input depth of 1 to 16, in the gate of arm_nn_is_convolve_s8_1x1_short_k().
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_convolve_1x1_s8_short_k(const cmsis_nn_context *ctx,
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
    (void)ctx;
    (void)bias_dims;
    (void)bias_data;

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (weight_sum_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!arm_nn_is_convolve_s8_1x1_short_k(conv_params, input_dims, filter_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    /* The pixel and output element counts are int32 indices in the kernel */
    const int64_t rows = (int64_t)input_dims->n * input_dims->h;
    if (rows > INT32_MAX || rows * input_dims->w > INT32_MAX || rows * input_dims->w * output_dims->c > INT32_MAX)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    /* The bias is folded into the weight sums */
    arm_convolve_1x1_s8_short_k_kernel(weight_sum_ctx->buf,
                                       input_data,
                                       filter_data,
                                       output_data,
                                       quant_params->multiplier,
                                       quant_params->shift,
                                       (int32_t)(rows * input_dims->w),
                                       output_dims->c,
                                       input_dims->c,
                                       conv_params->output_offset,
                                       conv_params->activation.min,
                                       conv_params->activation.max);
    return ARM_CMSIS_NN_SUCCESS;
#else
    (void)weight_sum_ctx;
    (void)conv_params;
    (void)quant_params;
    (void)input_dims;
    (void)input_data;
    (void)filter_dims;
    (void)filter_data;
    (void)output_dims;
    (void)output_data;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}

/**
 * @} end of NNConv group
 */
