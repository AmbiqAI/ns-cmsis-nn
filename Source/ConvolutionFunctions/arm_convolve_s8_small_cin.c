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
 * Title:        arm_convolve_s8_small_cin.c
 * Description:  Direct-entry s8 convolution for input depths of 1 to 3.
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

/*
 * Input depth 1 to 3 (first layers), in the gate of arm_nn_is_convolve_s8_small_cin(): one group, unit dilation, no
 * upscale, a kernel row (kernel_x * input_ch) of at most 16 bytes, at most 48 filter values per output channel and a
 * multiple of 4 output channels.
 *
 * im2col writes each kernel row with one predicated vector store, taking padding lanes from a splat of -input_offset,
 * into columns of col_len = 16 * nk bytes, so four columns fill the arm_convolve_s8_get_buffer_size() buffer exactly.
 * The columns are multiplied four output channels at a time, with the per-channel weight sums, multipliers and
 * shifts applied as vectors and the four results stored contiguously. The integer sums and the requantization are
 * those of arm_nn_mat_mult_nt_t_s8(), so the output is the same bit for bit.
 */

/* lhs_rows im2col columns of col_len = 16 * nk bytes against output_ch filters, four output channels per step. With
   whole_head, the first three filters' last chunks are loaded whole: past rhs_cols they read the following filters,
   which multiply the zeroed column tail, and they stay inside filter_data only while 16 * nk <= 2 * rhs_cols. That
   holds for nk 2 and 3 (rhs_cols of at least 17 and 33) and for nk 1 from 8 values; below that every chunk load is
   predicated. The fourth filter's last chunk is always predicated. */
__STATIC_FORCEINLINE void arm_convolve_s8_small_cin_gemm_nk(const int8_t *lhs,
                                                            const int32_t lhs_rows,
                                                            const int8_t *filter_data,
                                                            const int32_t *weight_sum,
                                                            const int32_t *output_mult,
                                                            const int32_t *output_shift,
                                                            int8_t *out,
                                                            const int32_t output_ch,
                                                            const int32_t rhs_cols,
                                                            const int32_t out_offset,
                                                            const int32_t act_min,
                                                            const int32_t act_max,
                                                            const int32_t nk,
                                                            const int32_t whole_head)
{
    const int32_t col_len = nk * 16;
    const mve_pred16_t p_last = vctp8q((uint32_t)(rhs_cols - (nk - 1) * 16));
    for (int32_t i_ch = 0; i_ch < output_ch; i_ch += 4)
    {
        const int32x4_t wsum = vldrwq_s32(weight_sum + i_ch);
        const int32x4_t mult = vldrwq_s32(output_mult + i_ch);
        const int32x4_t shift = vldrwq_s32(output_shift + i_ch);
        const int8_t *w0 = filter_data + i_ch * rhs_cols;
        const int8_t *w1 = w0 + rhs_cols;
        const int8_t *w2 = w1 + rhs_cols;
        const int8_t *w3 = w2 + rhs_cols;
        for (int32_t i_row = 0; i_row < lhs_rows; i_row++)
        {
            const int8_t *a = lhs + i_row * col_len;
            int32_t acc0, acc1, acc2, acc3;
            if (nk == 1)
            {
                const int8x16_t a0 = vldrbq_s8(a);
                acc0 = vmladavq_s8(a0, whole_head ? vldrbq_s8(w0) : vldrbq_z_s8(w0, p_last));
                acc1 = vmladavq_s8(a0, whole_head ? vldrbq_s8(w1) : vldrbq_z_s8(w1, p_last));
                acc2 = vmladavq_s8(a0, whole_head ? vldrbq_s8(w2) : vldrbq_z_s8(w2, p_last));
                acc3 = vmladavq_s8(a0, vldrbq_z_s8(w3, p_last));
            }
            else if (nk == 2)
            {
                const int8x16_t a0 = vldrbq_s8(a);
                const int8x16_t a1 = vldrbq_s8(a + 16);
                acc0 = vmladavaq_s8(vmladavq_s8(a0, vldrbq_s8(w0)), a1, vldrbq_s8(w0 + 16));
                acc1 = vmladavaq_s8(vmladavq_s8(a0, vldrbq_s8(w1)), a1, vldrbq_s8(w1 + 16));
                acc2 = vmladavaq_s8(vmladavq_s8(a0, vldrbq_s8(w2)), a1, vldrbq_s8(w2 + 16));
                acc3 = vmladavaq_s8(vmladavq_s8(a0, vldrbq_s8(w3)), a1, vldrbq_z_s8(w3 + 16, p_last));
            }
            else
            {
                const int8x16_t a0 = vldrbq_s8(a);
                const int8x16_t a1 = vldrbq_s8(a + 16);
                const int8x16_t a2 = vldrbq_s8(a + 32);
                acc0 = vmladavaq_s8(
                    vmladavaq_s8(vmladavq_s8(a0, vldrbq_s8(w0)), a1, vldrbq_s8(w0 + 16)), a2, vldrbq_s8(w0 + 32));
                acc1 = vmladavaq_s8(
                    vmladavaq_s8(vmladavq_s8(a0, vldrbq_s8(w1)), a1, vldrbq_s8(w1 + 16)), a2, vldrbq_s8(w1 + 32));
                acc2 = vmladavaq_s8(
                    vmladavaq_s8(vmladavq_s8(a0, vldrbq_s8(w2)), a1, vldrbq_s8(w2 + 16)), a2, vldrbq_s8(w2 + 32));
                acc3 = vmladavaq_s8(vmladavaq_s8(vmladavq_s8(a0, vldrbq_s8(w3)), a1, vldrbq_s8(w3 + 16)),
                                    a2,
                                    vldrbq_z_s8(w3 + 32, p_last));
            }
            int32x4_t res = vdupq_n_s32(acc0);
            res = vsetq_lane_s32(acc1, res, 1);
            res = vsetq_lane_s32(acc2, res, 2);
            res = vsetq_lane_s32(acc3, res, 3);
            res = vaddq_s32(res, wsum);
            res = arm_requantize_mve_32x4(res, mult, shift);
            res = vaddq_n_s32(res, out_offset);
            res = vmaxq_s32(res, vdupq_n_s32(act_min));
            res = vminq_s32(res, vdupq_n_s32(act_max));
            vstrbq_s32(out + i_row * output_ch + i_ch, res);
        }
    }
}

/* Out of line, with a run-time row count, so that each K-chunk count has one rolled copy of the loop. GCC would
   otherwise clone it per call site (interprocedural constant propagation), hence noipa there. */
    #if defined(__GNUC__) && !defined(__clang__)
        #define ARM_CONVOLVE_S8_SMALL_CIN_GEMM_ATTR __attribute__((noinline, noipa))
    #else
        #define ARM_CONVOLVE_S8_SMALL_CIN_GEMM_ATTR __attribute__((noinline))
    #endif
static ARM_CONVOLVE_S8_SMALL_CIN_GEMM_ATTR void
arm_convolve_s8_small_cin_gemm(const int8_t *lhs,
                               const int32_t lhs_rows,
                               const int8_t *filter_data,
                               const int32_t *weight_sum,
                               const cmsis_nn_per_channel_quant_params *quant_params,
                               int8_t *out,
                               const int32_t output_ch,
                               const int32_t rhs_cols,
                               const cmsis_nn_conv_params *conv_params)
{
    const int32_t *mult = quant_params->multiplier;
    const int32_t *shift = quant_params->shift;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t act_min = conv_params->activation.min;
    const int32_t act_max = conv_params->activation.max;
    if (rhs_cols < 8)
    {
        arm_convolve_s8_small_cin_gemm_nk(lhs,
                                          lhs_rows,
                                          filter_data,
                                          weight_sum,
                                          mult,
                                          shift,
                                          out,
                                          output_ch,
                                          rhs_cols,
                                          out_offset,
                                          act_min,
                                          act_max,
                                          1,
                                          0);
    }
    else if (rhs_cols <= 16)
    {
        arm_convolve_s8_small_cin_gemm_nk(lhs,
                                          lhs_rows,
                                          filter_data,
                                          weight_sum,
                                          mult,
                                          shift,
                                          out,
                                          output_ch,
                                          rhs_cols,
                                          out_offset,
                                          act_min,
                                          act_max,
                                          1,
                                          1);
    }
    else if (rhs_cols <= 32)
    {
        arm_convolve_s8_small_cin_gemm_nk(lhs,
                                          lhs_rows,
                                          filter_data,
                                          weight_sum,
                                          mult,
                                          shift,
                                          out,
                                          output_ch,
                                          rhs_cols,
                                          out_offset,
                                          act_min,
                                          act_max,
                                          2,
                                          1);
    }
    else
    {
        arm_convolve_s8_small_cin_gemm_nk(lhs,
                                          lhs_rows,
                                          filter_data,
                                          weight_sum,
                                          mult,
                                          shift,
                                          out,
                                          output_ch,
                                          rhs_cols,
                                          out_offset,
                                          act_min,
                                          act_max,
                                          3,
                                          1);
    }
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_s8_small_cin_kernel(const cmsis_nn_context *ctx,
                                 const cmsis_nn_context *weight_sum_ctx,
                                 const cmsis_nn_conv_params *conv_params,
                                 const cmsis_nn_per_channel_quant_params *quant_params,
                                 const cmsis_nn_dims *input_dims,
                                 const int8_t *input_data,
                                 const cmsis_nn_dims *filter_dims,
                                 const int8_t *filter_data,
                                 const cmsis_nn_dims *output_dims,
                                 int8_t *output_data)
{
    if (ctx->buf == NULL || weight_sum_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int8_t *buf = (int8_t *)ctx->buf;
    const int32_t *weight_sum = (const int32_t *)weight_sum_ctx->buf;

    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t input_ch = input_dims->c;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;
    const int32_t input_offset = conv_params->input_offset;

    const int32_t rhs_cols = kernel_x * kernel_y * input_ch;
    const int32_t col_len = ((rhs_cols + 15) / 16) * 16;
    const int32_t row_len = kernel_x * input_ch;
    const int32_t in_row_stride = input_x * input_ch;
    const mve_pred16_t p_row = vctp8q((uint32_t)row_len);
    const int8x16_t pad_val = vdupq_n_s8((int8_t)-input_offset);
    /* Zero the column bytes past rhs_cols: the GEMM reads them against the next filter's first values. */
    {
        const mve_pred16_t p_tail = (mve_pred16_t)~vctp8q((uint32_t)(rhs_cols - (col_len - 16)));
        for (int32_t i = 1; i <= 4; i++)
        {
            vstrbq_p_s8(buf + i * col_len - 16, vdupq_n_s8(0), p_tail);
        }
    }

    for (int32_t i_batch = 0; i_batch < input_dims->n; i_batch++)
    {
        int8_t *out = output_data;
        int32_t lhs_rows = 0;
        int8_t *col = buf;
        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            const int32_t base_y = stride_y * i_out_y - pad_y;
            /* Kernel rows [ky_lo, ky_hi) fall inside the input; the rest are padding. */
            const int32_t ky_lo = base_y < 0 ? (-base_y < kernel_y ? -base_y : kernel_y) : 0;
            const int32_t ky_hi_in = input_y - base_y;
            const int32_t ky_hi = ky_hi_in < kernel_y ? (ky_hi_in > ky_lo ? ky_hi_in : ky_lo) : kernel_y;
            const int32_t ky_valid = ky_hi - ky_lo;
            /* Integer address arithmetic: a row may start before the input, and masked lanes are not accessed. */
            uintptr_t src_x = (uintptr_t)input_data + (uintptr_t)(intptr_t)((base_y + ky_lo) * in_row_stride) -
                (uintptr_t)(intptr_t)(pad_x * input_ch);
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_x = stride_x * i_out_x - pad_x;
                mve_pred16_t p_valid = p_row;
                int32_t x_edge = 0;
                if (base_x < 0 || base_x + kernel_x > input_x)
                {
                    const int32_t lo = base_x < 0 ? -base_x * input_ch : 0;
                    const int32_t hi_x = input_x - base_x;
                    const int32_t hi = hi_x < kernel_x ? hi_x * input_ch : row_len;
                    p_valid =
                        (mve_pred16_t)(vctp8q((uint32_t)(hi > 0 ? hi : 0)) & ~vctp8q((uint32_t)(lo < 16 ? lo : 16)));
                    x_edge = 1;
                }
                const int8_t *src_int = (const int8_t *)src_x;
                if (!x_edge && ky_valid == kernel_y && kernel_y == 3)
                {
                    /* Interior pixel of a 3-row kernel: three row copies, no bounds work. */
                    vstrbq_p_s8(col, vldrbq_z_s8(src_int, p_row), p_row);
                    vstrbq_p_s8(col + row_len, vldrbq_z_s8(src_int + in_row_stride, p_row), p_row);
                    vstrbq_p_s8(col + 2 * row_len, vldrbq_z_s8(src_int + 2 * in_row_stride, p_row), p_row);
                }
                else if (!x_edge && ky_valid == kernel_y)
                {
                    int8_t *dst = col;
                    for (int32_t i = 0; i < kernel_y; i++)
                    {
                        vstrbq_p_s8(dst, vldrbq_z_s8(src_int, p_row), p_row);
                        src_int += in_row_stride;
                        dst += row_len;
                    }
                }
                else
                {
                    int8_t *dst = col;
                    for (int32_t i = 0; i < ky_lo; i++)
                    {
                        vstrbq_p_s8(dst, pad_val, p_row);
                        dst += row_len;
                    }
                    const int8_t *src = (const int8_t *)src_x;
                    for (int32_t i = 0; i < ky_valid; i++)
                    {
                        vstrbq_p_s8(dst, vpselq_s8(vldrbq_z_s8(src, p_valid), pad_val, p_valid), p_row);
                        src += in_row_stride;
                        dst += row_len;
                    }
                    for (int32_t i = ky_hi; i < kernel_y; i++)
                    {
                        vstrbq_p_s8(dst, pad_val, p_row);
                        dst += row_len;
                    }
                }
                src_x += (uintptr_t)(stride_x * input_ch);
                col += col_len;
                lhs_rows++;
                if (lhs_rows == 4)
                {
                    arm_convolve_s8_small_cin_gemm(
                        buf, 4, filter_data, weight_sum, quant_params, out, output_ch, rhs_cols, conv_params);
                    out += 4 * output_ch;
                    lhs_rows = 0;
                    col = buf;
                }
            }
        }
        if (lhs_rows != 0)
        {
            arm_convolve_s8_small_cin_gemm(
                buf, lhs_rows, filter_data, weight_sum, quant_params, out, output_ch, rhs_cols, conv_params);
        }
        input_data += input_x * input_y * input_ch;
        output_data += output_x * output_y * output_ch;
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif

/*
 * s8 convolution for input depths of 1 to 3.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_convolve_s8_small_cin(const cmsis_nn_context *ctx,
                                              const cmsis_nn_context *weight_sum_ctx,
                                              const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const cmsis_nn_dims *input_dims,
                                              const int8_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const int8_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const int32_t *bias_data,
                                              const cmsis_nn_dims *upscale_dims,
                                              const cmsis_nn_dims *output_dims,
                                              int8_t *output_data)
{
    (void)bias_dims;
    (void)bias_data;

    /* The argument checks of arm_convolve_s8() */
    if (ctx->buf == NULL || arm_nn_convolve_s8_groups_invalid(input_dims, filter_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#if defined(ARM_MATH_MVEI)
    if (weight_sum_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#endif

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (!arm_nn_is_convolve_s8_small_cin(conv_params, input_dims, filter_dims, output_dims, upscale_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    /* The bias is folded into the weight sums */
    return arm_convolve_s8_small_cin_kernel(ctx,
                                            weight_sum_ctx,
                                            conv_params,
                                            quant_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            output_dims,
                                            output_data);
#else
    (void)weight_sum_ctx;
    (void)conv_params;
    (void)quant_params;
    (void)input_data;
    (void)filter_data;
    (void)upscale_dims;
    (void)output_data;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}

/**
 * @} end of NNConv group
 */
