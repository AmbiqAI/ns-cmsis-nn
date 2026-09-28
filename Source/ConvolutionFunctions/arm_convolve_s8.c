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
 * Title:        arm_convolve_s8.c
 * Description:  s8 version of convolution using symmetric quantization.
 *
 * $Date:        27 September 2026
 * $Revision:    V.4.1.0
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
 * Input depth 1 to 3 (first layers): one group, unit dilation, no upscale, a kernel row (kernel_x * input_ch) of at
 * most 16 bytes, at most 48 filter values per output channel and a multiple of 4 output channels.
 *
 * im2col writes each kernel row with one predicated vector store, taking padding lanes from a splat of -input_offset,
 * into columns of col_len = 16 * nk bytes, so four columns fill the arm_convolve_s8_get_buffer_size() buffer exactly.
 * The columns are multiplied four output channels at a time, with the per-channel weight sums, multipliers and
 * shifts applied as vectors and the four results stored contiguously. The integer sums and the requantization are
 * those of arm_nn_mat_mult_nt_t_s8(), so the output is the same bit for bit.
 */
__STATIC_FORCEINLINE int arm_convolve_s8_is_small_cin(const cmsis_nn_conv_params *conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims,
                                                      const cmsis_nn_dims *upscale_dims)
{
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t input_ch = input_dims->c;
    return (upscale_dims == NULL) && (filter_dims->c == input_ch) && (input_ch >= 1) &&
        (conv_params->dilation.w == 1) && (conv_params->dilation.h == 1) && (kernel_x >= 1) && (kernel_y >= 1) &&
        (kernel_x * input_ch <= 16) && (kernel_x * kernel_y * input_ch <= 48) && (output_dims->c > 0) &&
        ((output_dims->c & 3) == 0);
}

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
arm_convolve_s8_small_cin(const cmsis_nn_context *ctx,
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

/*
 * 3x3 kernel over a 16-deep input with unit stride and dilation, one group and no upscale. A pixel's 144-value patch
 * is three 48-byte kernel rows. For a patch inside the input those rows are read in place; a patch that crosses the
 * border is first copied to an im2col slot with padding. Four pixels are then multiplied against every filter as in
 * arm_nn_mat_mult_nt_t_s8(), with the same requantization, so the output is the same bit for bit.
 */
__STATIC_FORCEINLINE int arm_convolve_s8_is_3x3_c16_s1(const cmsis_nn_conv_params *conv_params,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *upscale_dims)
{
    return (upscale_dims == NULL) && (filter_dims->c == 16) && (filter_dims->w == 3) && (filter_dims->h == 3) &&
        (conv_params->stride.w == 1) && (conv_params->stride.h == 1) && (conv_params->dilation.w == 1) &&
        (conv_params->dilation.h == 1);
}

/* Output channels for four pixels whose kernel rows start at seg[ky][pixel]; only the first n_pix are stored. */
static __attribute__((noinline)) void arm_convolve_s8_3x3_c16_block(const int8_t *seg[3][4],
                                                                    const int32_t n_pix,
                                                                    const int8_t *rhs,
                                                                    const int32_t *weight_sum,
                                                                    int8_t *dst,
                                                                    const int32_t *dst_multipliers,
                                                                    const int32_t *dst_shifts,
                                                                    const int32_t rhs_rows,
                                                                    const int32_t dst_offset,
                                                                    const int32_t activation_min,
                                                                    const int32_t activation_max)
{
    const uint32x4_t scatter = vmulq_n_u32(vidupq_n_u32(0, 1), (uint32_t)rhs_rows);
    const mve_pred16_t p_store = vctp32q((uint32_t)n_pix);
    const int32x4_t v_min = vdupq_n_s32(activation_min);
    const int32x4_t v_max = vdupq_n_s32(activation_max);
    for (int32_t i = 0; i < rhs_rows; i++)
    {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        const int8_t *w = rhs + i * 144;
        for (int32_t ky = 0; ky < 3; ky++)
        {
            /* Volatile so the patch loads stay inside the channel loop instead of being hoisted and spilled. */
            __ASM volatile("   vldrb.8         q0, [%[w]], #16        \n"
                           "   vldrb.8         q1, [%[x0]]            \n"
                           "   vmladava.s8     %[a0], q0, q1          \n"
                           "   vldrb.8         q2, [%[x1]]            \n"
                           "   vmladava.s8     %[a1], q0, q2          \n"
                           "   vldrb.8         q3, [%[x2]]            \n"
                           "   vmladava.s8     %[a2], q0, q3          \n"
                           "   vldrb.8         q4, [%[x3]]            \n"
                           "   vmladava.s8     %[a3], q0, q4          \n"
                           "   vldrb.8         q0, [%[w]], #16        \n"
                           "   vldrb.8         q1, [%[x0], #16]       \n"
                           "   vmladava.s8     %[a0], q0, q1          \n"
                           "   vldrb.8         q2, [%[x1], #16]       \n"
                           "   vmladava.s8     %[a1], q0, q2          \n"
                           "   vldrb.8         q3, [%[x2], #16]       \n"
                           "   vmladava.s8     %[a2], q0, q3          \n"
                           "   vldrb.8         q4, [%[x3], #16]       \n"
                           "   vmladava.s8     %[a3], q0, q4          \n"
                           "   vldrb.8         q0, [%[w]], #16        \n"
                           "   vldrb.8         q1, [%[x0], #32]       \n"
                           "   vmladava.s8     %[a0], q0, q1          \n"
                           "   vldrb.8         q2, [%[x1], #32]       \n"
                           "   vmladava.s8     %[a1], q0, q2          \n"
                           "   vldrb.8         q3, [%[x2], #32]       \n"
                           "   vmladava.s8     %[a2], q0, q3          \n"
                           "   vldrb.8         q4, [%[x3], #32]       \n"
                           "   vmladava.s8     %[a3], q0, q4          \n"
                           : [w] "+r"(w), [a0] "+Te"(a0), [a1] "+Te"(a1), [a2] "+Te"(a2), [a3] "+Te"(a3)
                           : [x0] "r"(seg[ky][0]), [x1] "r"(seg[ky][1]), [x2] "r"(seg[ky][2]), [x3] "r"(seg[ky][3])
                           : "q0", "q1", "q2", "q3", "q4", "memory");
        }
        int32x4_t res = {a0, a1, a2, a3};
        res = vaddq_n_s32(res, weight_sum[i]);
        res = arm_requantize_mve(res, dst_multipliers[i], dst_shifts[i]);
        res = vaddq_n_s32(res, dst_offset);
        res = vminq_s32(vmaxq_s32(res, v_min), v_max);
        vstrbq_scatter_offset_p_s32(dst + i, scatter, res, p_store);
    }
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_convolve_s8_3x3_c16_s1(const cmsis_nn_context *ctx,
                           const cmsis_nn_context *weight_sum_ctx,
                           const cmsis_nn_conv_params *conv_params,
                           const cmsis_nn_per_channel_quant_params *quant_params,
                           const cmsis_nn_dims *input_dims,
                           const int8_t *input_data,
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
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t in_row_stride = input_x * 16;
    const int8x16_t pad_val = vdupq_n_s8((int8_t)-conv_params->input_offset);

    for (int32_t i_batch = 0; i_batch < input_dims->n; i_batch++)
    {
        const int8_t *seg[3][4];
        int32_t n_pix = 0;
        int8_t *out = output_data;
        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            const int32_t base_y = i_out_y - pad_y;
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_x = i_out_x - pad_x;
                if (base_x >= 0 && base_x + 3 <= input_x && base_y >= 0 && base_y + 3 <= input_y)
                {
                    const int8_t *p = input_data + base_y * in_row_stride + base_x * 16;
                    seg[0][n_pix] = p;
                    seg[1][n_pix] = p + in_row_stride;
                    seg[2][n_pix] = p + 2 * in_row_stride;
                }
                else
                {
                    int8_t *slot = buf + n_pix * 144;
                    int8_t *dst = slot;
                    for (int32_t ky = 0; ky < 3; ky++)
                    {
                        const int32_t k_y = base_y + ky;
                        for (int32_t kx = 0; kx < 3; kx++)
                        {
                            const int32_t k_x = base_x + kx;
                            int8x16_t v = pad_val;
                            if (k_y >= 0 && k_y < input_y && k_x >= 0 && k_x < input_x)
                            {
                                v = vldrbq_s8(input_data + k_y * in_row_stride + k_x * 16);
                            }
                            vstrbq_s8(dst, v);
                            dst += 16;
                        }
                    }
                    seg[0][n_pix] = slot;
                    seg[1][n_pix] = slot + 48;
                    seg[2][n_pix] = slot + 96;
                }
                n_pix++;
                if (n_pix == 4)
                {
                    arm_convolve_s8_3x3_c16_block(seg,
                                                  4,
                                                  filter_data,
                                                  weight_sum,
                                                  out,
                                                  quant_params->multiplier,
                                                  quant_params->shift,
                                                  output_ch,
                                                  conv_params->output_offset,
                                                  conv_params->activation.min,
                                                  conv_params->activation.max);
                    out += 4 * output_ch;
                    n_pix = 0;
                }
            }
        }
        if (n_pix != 0)
        {
            /* Unused lanes repeat the first pixel so every load stays in bounds; they are not stored. */
            for (int32_t j = n_pix; j < 4; j++)
            {
                seg[0][j] = seg[0][0];
                seg[1][j] = seg[1][0];
                seg[2][j] = seg[2][0];
            }
            arm_convolve_s8_3x3_c16_block(seg,
                                          n_pix,
                                          filter_data,
                                          weight_sum,
                                          out,
                                          quant_params->multiplier,
                                          quant_params->shift,
                                          output_ch,
                                          conv_params->output_offset,
                                          conv_params->activation.min,
                                          conv_params->activation.max);
        }
        input_data += input_x * input_y * 16;
        output_data += output_x * output_y * output_ch;
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif

/*
 * With the low-depth paths, arm_convolve_s8() checks the shape first and the general path below is
 * arm_convolve_s8_generic(). It keeps external linkage and stays out of line so that it compiles to the same code as
 * the general path had inside arm_convolve_s8(): a static function may have its unused parameters removed or be
 * specialized for its one caller. Without MVE the general path is arm_convolve_s8() itself.
 */
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    #define ARM_CONVOLVE_S8_GENERIC __attribute__((noinline)) arm_cmsis_nn_status arm_convolve_s8_generic
ARM_CONVOLVE_S8_GENERIC(const cmsis_nn_context *ctx,
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
                        int8_t *output_data);
#else
    #define ARM_CONVOLVE_S8_GENERIC arm_cmsis_nn_status arm_convolve_s8
#endif

/*
 * Basic s8 convolution function.
 *
 * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
 * are multiples of 4 or atleast greater than 4.
 *
 */

ARM_CONVOLVE_S8_GENERIC(const cmsis_nn_context *ctx,
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

    if (ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int16_t *buffer_a = (int16_t *)ctx->buf;

    const int32_t input_batches = input_dims->n;
    const uint16_t input_x = input_dims->w;
    const uint16_t input_y = input_dims->h;
    const uint16_t input_ch = input_dims->c;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t kernel_ch = filter_dims->c;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_y = output_dims->h;
    const uint16_t output_ch = output_dims->c;

    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t pad_y = conv_params->padding.h;
    const uint16_t stride_x = conv_params->stride.w;
    const uint16_t stride_y = conv_params->stride.h;
    const int32_t dilation_x = conv_params->dilation.w;
    const int32_t dilation_y = conv_params->dilation.h;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    const int32_t input_offset = conv_params->input_offset;

    const int32_t groups = input_ch / kernel_ch;
    const int32_t rhs_cols = kernel_x * kernel_y * kernel_ch;
    const int32_t output_ch_per_group = output_ch / groups;

    const int32_t *output_mult = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    if (input_ch % groups != 0 || output_ch % groups != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // For upscale_dims == 2, the actual index of the input data is the index of the upscaled input divided by two. In
    // the ordinary case, there is no difference. The division is implemented as a rshift for optimization purposes.
    uint32_t y_rshift = 0;
    uint32_t x_rshift = 0;

    if (upscale_dims)
    {
        y_rshift = upscale_dims->h == 2 ? 1 : 0;
        x_rshift = upscale_dims->w == 2 ? 1 : 0;
    }

    const int32_t input_x_rshifted = input_x >> x_rshift;
    const int32_t input_y_rshifted = input_y >> y_rshift;

    const int32_t remainder = rhs_cols % 4;
    const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;

#if defined(ARM_MATH_MVEI)
    /* Hoisted out of the batch loop below: the check is loop-invariant, and leaving it inside meant an
       input_dims->n of zero or less skipped it and returned ARM_CMSIS_NN_SUCCESS with a NULL buffer undiagnosed. */
    if (weight_sum_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#endif

    for (int i_batch = 0; i_batch < input_batches; i_batch++)
    {

#if defined(ARM_MATH_MVEI)
        const int32_t aligned_rhs_cols_offset = aligned_rhs_cols - rhs_cols;

        /* Generate up to four columns from the input tensor a GEMM computation */
        int8_t *im2col_buf = (int8_t *)buffer_a;
        const int32_t *weight_sum_data_ptr = weight_sum_ctx->buf;
#else
        (void)weight_sum_ctx;
        /* Use as a ping-pong buffer for unordered elements */
        int8_t *im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
        int16_t *im2col_buf_start_s16 = buffer_a;
#endif
        int32_t lhs_rows = 0;

        const int8_t *filter_data_ptr = &filter_data[0];
        const int32_t *bias_data_ptr = &bias_data[0];
        const int32_t *output_mult_ptr = &output_mult[0];
        const int32_t *output_shift_ptr = &output_shift[0];

        /* This part implements the im2col function */
        for (int32_t i_group = 0; i_group < groups; i_group++)
        {
            int8_t *out = output_data + i_group * output_ch_per_group;
            for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
            {
                for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
                {
                    const int32_t base_idx_x = stride_x * i_out_x - pad_x;
                    const int32_t base_idx_y = stride_y * i_out_y - pad_y;

                    if (y_rshift == 1 || x_rshift == 1)
                    {
                        // Fill complete buf with -input_offset
                        arm_memset_s8(
                            im2col_buf, (int8_t)-input_offset, sizeof(int8_t) * kernel_ch * kernel_x * kernel_y);
                        for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
                        {
                            const int32_t k_y = base_idx_y + dilation_y * i_ker_y;

                            //  Don't copy data when padding, or for every second row if stride_y == 2
                            if ((k_y < 0 || k_y >= input_y) || (k_y % 2 && y_rshift == 1))
                            {
                                im2col_buf += kernel_ch * kernel_x;
                            }
                            else
                            {
                                const int32_t k_y_rshifted = k_y >> y_rshift;
                                for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                                {
                                    const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                                    // Don't copy data when padding, or for every second element if stride_x == 2
                                    if ((k_x >= 0 && k_x < input_x) && ((k_x % 2 == 0) || x_rshift == 0))
                                    {
                                        const int32_t k_x_rshifted = k_x >> x_rshift;
                                        arm_memcpy_s8(im2col_buf,
                                                      input_data +
                                                          (k_y_rshifted * input_x_rshifted + k_x_rshifted) * input_ch,
                                                      sizeof(int8_t) * kernel_ch);
                                    }
                                    im2col_buf += kernel_ch;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
                        {
                            for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                            {
                                const int32_t k_y = base_idx_y + dilation_y * i_ker_y;
                                const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                                if (k_y < 0 || k_y >= input_y || k_x < 0 || k_x >= input_x)
                                {
                                    arm_memset_s8(im2col_buf, (int8_t)-input_offset, sizeof(int8_t) * kernel_ch);
                                }
                                else
                                {
                                    arm_memcpy_s8(im2col_buf,
                                                  input_data + (k_y * input_x + k_x) * input_ch + i_group * kernel_ch,
                                                  sizeof(int8_t) * kernel_ch);
                                }
                                im2col_buf += kernel_ch;
                            }
                        }
                    }
                    lhs_rows++;

#if defined(ARM_MATH_MVEI)
                    im2col_buf += aligned_rhs_cols_offset;

                    /* Computation is filed for every 4 columns */
                    if (lhs_rows == 4)
                    {
                        arm_nn_mat_mult_nt_t_s8(weight_sum_data_ptr,
                                                (int8_t *)buffer_a,
                                                filter_data_ptr,
                                                bias_data_ptr,
                                                out,
                                                output_mult_ptr,
                                                output_shift_ptr,
                                                lhs_rows,
                                                output_ch_per_group,
                                                rhs_cols,
                                                input_offset,
                                                out_offset,
                                                out_activation_min,
                                                out_activation_max,
                                                output_ch,
                                                aligned_rhs_cols);

                        out += lhs_rows * output_ch;

                        lhs_rows = 0;
                        im2col_buf = (int8_t *)buffer_a;
                    }
#else
    #if defined(ARM_MATH_DSP)
                    /* Copy one column with input offset and no ordering */
                    arm_s8_to_s16_unordered_with_offset(
                        im2col_buf - rhs_cols, im2col_buf_start_s16, rhs_cols, (int16_t)input_offset);
    #else

                    arm_q7_to_q15_with_offset(
                        im2col_buf - rhs_cols, im2col_buf_start_s16, rhs_cols, (int16_t)input_offset);

    #endif
                    im2col_buf_start_s16 += aligned_rhs_cols;

                    if (lhs_rows == 2)
                    {
                        if (groups > 1)
                        {
                            out = arm_nn_mat_mult_kernel_row_offset_s8_s16(filter_data_ptr,
                                                                           buffer_a,
                                                                           output_ch_per_group,
                                                                           output_shift_ptr,
                                                                           output_mult_ptr,
                                                                           out_offset,
                                                                           out_activation_min,
                                                                           out_activation_max,
                                                                           rhs_cols,
                                                                           aligned_rhs_cols,
                                                                           bias_data_ptr,
                                                                           output_ch,
                                                                           out);
                        }
                        else
                        {
                            out = arm_nn_mat_mult_kernel_s8_s16(filter_data_ptr,
                                                                buffer_a,
                                                                output_ch_per_group,
                                                                output_shift_ptr,
                                                                output_mult_ptr,
                                                                out_offset,
                                                                out_activation_min,
                                                                out_activation_max,
                                                                rhs_cols,
                                                                aligned_rhs_cols,
                                                                bias_data_ptr,
                                                                out);
                        }

                        /* counter reset */
                        im2col_buf_start_s16 = buffer_a;
                        im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
                        lhs_rows = 0;
                    }
#endif
                }
            }

            if (out == NULL)
            {
                return ARM_CMSIS_NN_NO_IMPL_ERROR;
            }

            /* Handle left over columns */
            if (lhs_rows != 0)
            {
#if defined(ARM_MATH_MVEI)
                arm_nn_mat_mult_nt_t_s8(weight_sum_data_ptr,
                                        (int8_t *)buffer_a,
                                        filter_data_ptr,
                                        bias_data_ptr,
                                        out,
                                        output_mult_ptr,
                                        output_shift_ptr,
                                        lhs_rows,
                                        output_ch_per_group,
                                        rhs_cols,
                                        input_offset,
                                        out_offset,
                                        out_activation_min,
                                        out_activation_max,
                                        output_ch,
                                        aligned_rhs_cols);

                out += lhs_rows * output_ch;
                lhs_rows = 0;
                im2col_buf = (int8_t *)buffer_a;
#else // #if defined(ARM_MATH_MVEI)

                const int8_t *ker_a = filter_data_ptr;
                int i;

                for (i = 0; i < output_ch_per_group; i++)
                {
                    /* Load the accumulator with bias first */
                    int32_t sum = 0;
                    if (bias_data_ptr)
                    {
                        sum = bias_data_ptr[i];
                    }

                    const int16_t *ip_as_col = buffer_a;

    #if defined(ARM_MATH_DSP)
                    /* 4 multiply and accumulates are done in one loop. */
                    uint16_t col_count = rhs_cols / 4;
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
    #else
                    uint16_t col_count = rhs_cols;

    #endif
                    while (col_count)
                    {
                        int8_t ker_a1 = *ker_a++;
                        int16_t ip_b1 = *ip_as_col++;

                        sum += ker_a1 * ip_b1;
                        col_count--;
                    }

                    sum = arm_nn_requantize(sum, output_mult_ptr[i], output_shift_ptr[i]);
                    sum += out_offset;
                    sum = ARM_NN_MAX(sum, out_activation_min);
                    sum = ARM_NN_MIN(sum, out_activation_max);
                    *out++ = (int8_t)sum;
                }

                im2col_buf_start_s16 = buffer_a;
                im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
                lhs_rows = 0;
#endif // #if defined(ARM_MATH_MVEI)
            }
#if defined(ARM_MATH_MVEI)
            weight_sum_data_ptr += output_ch_per_group;
#endif
            filter_data_ptr += output_ch_per_group * rhs_cols;
            bias_data_ptr += output_ch_per_group;
            output_mult_ptr += output_ch_per_group;
            output_shift_ptr += output_ch_per_group;
        }
        /* Advance to the next batch */
        input_data += (input_x_rshifted * input_y_rshifted * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
arm_cmsis_nn_status arm_convolve_s8(const cmsis_nn_context *ctx,
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
    /* Cheapest rejection first: most layers have neither input depth. */
    const int32_t input_ch = input_dims->c;
    if (input_ch <= 3)
    {
        if (arm_convolve_s8_is_small_cin(conv_params, input_dims, filter_dims, output_dims, upscale_dims))
        {
            return arm_convolve_s8_small_cin(ctx,
                                             weight_sum_ctx,
                                             conv_params,
                                             quant_params,
                                             input_dims,
                                             input_data,
                                             filter_dims,
                                             filter_data,
                                             output_dims,
                                             output_data);
        }
    }
    else if (input_ch == 16)
    {
        if (arm_convolve_s8_is_3x3_c16_s1(conv_params, filter_dims, upscale_dims))
        {
            return arm_convolve_s8_3x3_c16_s1(ctx,
                                              weight_sum_ctx,
                                              conv_params,
                                              quant_params,
                                              input_dims,
                                              input_data,
                                              filter_data,
                                              output_dims,
                                              output_data);
        }
    }
    return arm_convolve_s8_generic(ctx,
                                   weight_sum_ctx,
                                   conv_params,
                                   quant_params,
                                   input_dims,
                                   input_data,
                                   filter_dims,
                                   filter_data,
                                   bias_dims,
                                   bias_data,
                                   upscale_dims,
                                   output_dims,
                                   output_data);
}
#endif

/**
 * @} end of NNConv group
 */
