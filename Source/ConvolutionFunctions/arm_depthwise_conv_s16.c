/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
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
 * Title:        arm_depthwise_conv_s16.c
 * Description:  s16 version of depthwise convolution.
 *
 * $Date:        8 October 2026
 * $Revision:    V.2.1.0
 *
 * Target Processor:  Cortex-M CPUs
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

static void depthwise_conv_s16_generic_s16(const int16_t *input,
                                           const uint16_t input_batches,
                                           const uint16_t input_x,
                                           const uint16_t input_y,
                                           const uint16_t input_ch,
                                           const int8_t *kernel,
                                           const uint16_t ch_mult,
                                           const uint16_t kernel_x,
                                           const uint16_t kernel_y,
                                           const uint16_t pad_x,
                                           const uint16_t pad_y,
                                           const uint16_t stride_x,
                                           const uint16_t stride_y,
                                           const int64_t *bias,
                                           int16_t *output,
                                           const int32_t *output_shift,
                                           const int32_t *output_mult,
                                           const uint16_t output_x,
                                           const uint16_t output_y,
                                           const int32_t output_activation_min,
                                           const int32_t output_activation_max,
                                           const uint16_t dilation_x,
                                           const uint16_t dilation_y)

{
    for (int i_batch = 0; i_batch < input_batches; i_batch++)
    {
        for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
            for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;
                for (int i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                {
                    for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                    {
                        const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;

                        const int32_t reduced_multiplier = REDUCE_MULTIPLIER(output_mult[idx_out_ch]);
                        int64_t acc_0 = 0;

                        int ker_y_start;
                        int ker_x_start;
                        int ker_y_end;
                        int ker_x_end;

                        if (dilation_x > 1)
                        {
                            const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                            ker_x_start = ARM_NN_MAX(0, start_x_max);
                            const int32_t end_min_x = (input_x - base_idx_x + dilation_x - 1) / dilation_x;
                            ker_x_end = ARM_NN_MIN(kernel_x, end_min_x);
                        }
                        else
                        {
                            ker_x_start = ARM_NN_MAX(0, -base_idx_x);
                            ker_x_end = ARM_NN_MIN(kernel_x, input_x - base_idx_x);
                        }

                        if (dilation_y > 1)
                        {
                            const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                            ker_y_start = ARM_NN_MAX(0, start_y_max);
                            const int32_t end_min_y = (input_y - base_idx_y + dilation_y - 1) / dilation_y;
                            ker_y_end = ARM_NN_MIN(kernel_y, end_min_y);
                        }
                        else
                        {
                            ker_y_start = ARM_NN_MAX(0, -base_idx_y);
                            ker_y_end = ARM_NN_MIN(kernel_y, input_y - base_idx_y);
                        }

                        if (bias)
                        {
                            acc_0 = bias[idx_out_ch];
                        }

                        for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                        {
                            const int32_t idx_y = base_idx_y + dilation_y * i_ker_y;
                            for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                            {
                                const int32_t idx_x = base_idx_x + dilation_x * i_ker_x;
                                int32_t idx_0 = (idx_y * input_x + idx_x) * input_ch + i_input_ch;
                                int32_t ker_idx_0 = (i_ker_y * kernel_x + i_ker_x) * (input_ch * ch_mult) + idx_out_ch;

                                acc_0 += input[idx_0] * kernel[ker_idx_0];
                            }
                        }

                        /* Requantize and clamp output to provided range */
                        int32_t result = arm_nn_requantize_s64(acc_0, reduced_multiplier, output_shift[idx_out_ch]);
                        result = ARM_NN_MAX(result, output_activation_min);
                        result = ARM_NN_MIN(result, output_activation_max);
                        *output++ = (int16_t)result;
                    }
                }
            }
        }
        /* Advance to the next batch */
        input += (input_x * input_y * input_ch);
    }
}

#if defined(ARM_MATH_MVEI)
/* Exact s64 requant state in s32 lanes */
typedef struct
{
    int32x4_t mult;  // reduced multiplier << 16
    int32x4_t left;  // pre-shift, non-negative
    int32x4_t right; // rounding right shift, negative
    int32x4_t bias;  // bias, proven to fit s32
} dw_s16_rq;

/**
 * @brief           Whether dw_s16_rq_apply() matches arm_nn_requantize_s64() for all channels.
 * @param[in]       bias    Per-channel s64 bias, or NULL
 * @param[in]       shift   Per-channel shifts
 * @param[in]       num_ch  Number of channels
 * @param[in]       taps    Upper bound of s16 x s8 products summed per output
 *
 * @return          true when bias + sum, pre-shifted, fits s32 for every channel.
 */
static bool dw_s16_rq_ok(const int64_t *bias, const int32_t *shift, const int32_t num_ch, const int32_t taps)
{
    int32_t s_min = 0;
    int32_t s_max = -31;
    for (int32_t i = 0; i < num_ch; i++)
    {
        s_min = ARM_NN_MIN(s_min, shift[i]);
        s_max = ARM_NN_MAX(s_max, shift[i]);
    }
    uint64_t b_max = 0;
    if (bias)
    {
        for (int32_t i = 0; i < num_ch; i++)
        {
            const uint64_t mag = bias[i] < 0 ? 0 - (uint64_t)bias[i] : (uint64_t)bias[i];
            b_max = ARM_NN_MAX(b_max, mag);
        }
    }
    const uint64_t acc_max = (uint64_t)taps << 22;
    if (s_min < -31 || s_max > 8 || b_max > INT32_MAX)
    {
        return false;
    }
    if (b_max + acc_max < ((uint64_t)1 << (31 - ARM_NN_MAX(s_max + 2, 0))))
    {
        return true;
    }

    /* Exact check, one channel at a time */
    for (int32_t i = 0; i < num_ch; i++)
    {
        const uint64_t mag = !bias ? 0 : (bias[i] < 0 ? 0 - (uint64_t)bias[i] : (uint64_t)bias[i]);
        if (mag + acc_max >= ((uint64_t)1 << (31 - ARM_NN_MAX(shift[i] + 2, 0))))
        {
            return false;
        }
    }
    return true;
}

/* Load requant state for four channels */
__STATIC_FORCEINLINE dw_s16_rq dw_s16_rq_load(const int32_t *mult,
                                              const int32_t *shift,
                                              const int64_t *bias,
                                              const mve_pred16_t p)
{
    dw_s16_rq rq;
    const int32x4_t m = vldrwq_z_s32(mult, p);
    const int32x4_t s = vldrwq_z_s32(shift, p);
    const int32x4_t reduced = vshrq_n_s32(vaddq_n_s32(m, 1 << 15), 16);
    rq.mult = vshlq_n_s32(vpselq_s32(vdupq_n_s32(0x7FFF), reduced, vcmpgeq_n_s32(m, 0x7FFF0000)), 16);
    rq.left = vmaxq_s32(vaddq_n_s32(s, 2), vdupq_n_s32(0));
    rq.right = vminq_s32(vaddq_n_s32(s, 1), vdupq_n_s32(-1));
    rq.bias = bias ? vldrwq_gather_shifted_offset_z_s32((const int32_t *)bias, vidupq_n_u32(0, 2), p) : vdupq_n_s32(0);
    return rq;
}

/* Requantize, then clamp to activation range */
__STATIC_FORCEINLINE int32x4_t dw_s16_rq_apply(int32x4_t acc,
                                               const dw_s16_rq *rq,
                                               const int32_t act_min,
                                               const int32_t act_max)
{
    acc = vmulhq_s32(vshlq_s32(acc, rq->left), rq->mult);
    acc = vrshlq_s32(acc, rq->right);
    acc = vmaxq_s32(acc, vdupq_n_s32(act_min));
    return vminq_s32(acc, vdupq_n_s32(act_max));
}

/* Input read modes per channel block */
    #define DW_S16_IN_VEC 0
    #define DW_S16_IN_DUP 1
    #define DW_S16_IN_GATHER 2
    /* Predicated forms, fewer than four channels */
    #define DW_S16_IN_VEC_P 3
    #define DW_S16_IN_GATHER_P 4

/* Load weights for four output channels */
__STATIC_FORCEINLINE int32x4_t dw_s16_ker(const int8_t *rhs, const int32_t mode, const mve_pred16_t p)
{
    if (mode == DW_S16_IN_VEC_P || mode == DW_S16_IN_GATHER_P)
    {
        return vldrbq_z_s32(rhs, p);
    }
    return vldrbq_s32(rhs);
}

/* One tap for four output channels */
__STATIC_FORCEINLINE int32x4_t dw_s16_mac(int32x4_t acc,
                                          const int16_t *ip,
                                          const int32x4_t ker,
                                          const int32_t mode,
                                          const uint32x4_t offs,
                                          const mve_pred16_t p)
{
    if (mode == DW_S16_IN_VEC)
    {
        return vaddq_s32(acc, vmulq_s32(vldrhq_s32(ip), ker));
    }
    if (mode == DW_S16_IN_VEC_P)
    {
        return vaddq_s32(acc, vmulq_s32(vldrhq_z_s32(ip, p), ker));
    }
    if (mode == DW_S16_IN_DUP)
    {
        return vmlaq_n_s32(acc, ker, *ip);
    }
    if (mode == DW_S16_IN_GATHER)
    {
        return vaddq_s32(acc, vmulq_s32(vldrhq_gather_shifted_offset_s32(ip, offs), ker));
    }
    return vaddq_s32(acc, vmulq_s32(vldrhq_gather_shifted_offset_z_s32(ip, offs, p), ker));
}

/* Sum taps for one pixel */
__STATIC_FORCEINLINE int32x4_t dw_s16_taps1(int32x4_t acc,
                                            const int16_t *ip,
                                            const int8_t *rhs,
                                            const int32_t mode,
                                            const uint32x4_t offs,
                                            const mve_pred16_t p,
                                            const int32_t n_y,
                                            const int32_t n_x,
                                            const int32_t in_col,
                                            const int32_t in_row,
                                            const int32_t ker_col,
                                            const int32_t ker_row)
{
    for (int32_t ky = 0; ky < n_y; ky++, ip += in_row, rhs += ker_row)
    {
        const int16_t *ipx = ip;
        const int8_t *rhsx = rhs;
        for (int32_t kx = 0; kx < n_x; kx++, ipx += in_col, rhsx += ker_col)
        {
            acc = dw_s16_mac(acc, ipx, dw_s16_ker(rhsx, mode, p), mode, offs, p);
        }
    }
    return acc;
}

/* Sum taps for four pixels */
__STATIC_FORCEINLINE void dw_s16_taps4(int32x4_t *acc,
                                       const int16_t *ip_0,
                                       const int16_t *ip_1,
                                       const int16_t *ip_2,
                                       const int16_t *ip_3,
                                       const int8_t *rhs,
                                       const int32_t mode,
                                       const uint32x4_t offs,
                                       const mve_pred16_t p,
                                       const int32_t n_y,
                                       const int32_t n_x,
                                       const int32_t in_col,
                                       const int32_t in_row,
                                       const int32_t ker_col,
                                       const int32_t ker_row)
{
    int32x4_t acc_0 = acc[0];
    int32x4_t acc_1 = acc[1];
    int32x4_t acc_2 = acc[2];
    int32x4_t acc_3 = acc[3];
    const int32_t row_step = in_row - n_x * in_col;
    const int32_t ker_step = ker_row - n_x * ker_col;
    for (int32_t ky = 0; ky < n_y; ky++)
    {
        for (int32_t kx = 0; kx < n_x; kx++)
        {
            const int32x4_t ker = dw_s16_ker(rhs, mode, p);
            acc_0 = dw_s16_mac(acc_0, ip_0, ker, mode, offs, p);
            acc_1 = dw_s16_mac(acc_1, ip_1, ker, mode, offs, p);
            acc_2 = dw_s16_mac(acc_2, ip_2, ker, mode, offs, p);
            acc_3 = dw_s16_mac(acc_3, ip_3, ker, mode, offs, p);
            ip_0 += in_col;
            ip_1 += in_col;
            ip_2 += in_col;
            ip_3 += in_col;
            rhs += ker_col;
        }
        ip_0 += row_step;
        ip_1 += row_step;
        ip_2 += row_step;
        ip_3 += row_step;
        rhs += ker_step;
    }
    acc[0] = acc_0;
    acc[1] = acc_1;
    acc[2] = acc_2;
    acc[3] = acc_3;
}

/* Exact s64 requant, one lane each */
static int32x4_t dw_s16_rq_s64(const int32x4_t acc,
                               const int64_t *bias,
                               const int32_t *mult,
                               const int32_t *shift,
                               const int32_t lanes,
                               const int32_t act_min,
                               const int32_t act_max)
{
    int32_t res[4];
    vst1q_s32(res, acc);
    for (int32_t i = 0; i < lanes; i++)
    {
        const int64_t val = (int64_t)res[i] + (bias ? bias[i] : 0);
        const int32_t r = arm_nn_requantize_s64(val, REDUCE_MULTIPLIER(mult[i]), shift[i]);
        res[i] = ARM_NN_MIN(ARM_NN_MAX(r, act_min), act_max);
    }
    return vld1q_s32(res);
}

/* Requantize one pixel of four channels */
__STATIC_FORCEINLINE int32x4_t dw_s16_out(const int32x4_t acc,
                                          const dw_s16_rq *rq,
                                          const int32_t slow,
                                          const int64_t *bias,
                                          const int32_t *mult,
                                          const int32_t *shift,
                                          const int32_t lanes,
                                          const int32_t act_min,
                                          const int32_t act_max)
{
    if (slow)
    {
        return dw_s16_rq_s64(acc, bias, mult, shift, lanes, act_min, act_max);
    }
    return dw_s16_rq_apply(acc, rq, act_min, act_max);
}

/* Channel blocks outer, pixels inner */
static void dw_s16_mve(const int16_t *input,
                       const int32_t input_batches,
                       const int32_t input_x,
                       const int32_t input_y,
                       const int32_t input_ch,
                       const int8_t *kernel,
                       const int32_t ch_mult,
                       const int32_t kernel_x,
                       const int32_t kernel_y,
                       const int32_t pad_x,
                       const int32_t pad_y,
                       const int32_t stride_x,
                       const int32_t stride_y,
                       const int64_t *bias,
                       int16_t *output,
                       const int32_t *output_shift,
                       const int32_t *output_mult,
                       const int32_t output_x,
                       const int32_t output_y,
                       const int32_t act_min,
                       const int32_t act_max,
                       const int32_t dil_x,
                       const int32_t dil_y,
                       const int32_t slow)
{
    const int32_t output_ch = input_ch * ch_mult;
    const int32_t mode = ch_mult == 1 ? DW_S16_IN_VEC : (ch_mult % 4 == 0 ? DW_S16_IN_DUP : DW_S16_IN_GATHER);
    const int32_t in_col = dil_x * input_ch;
    const int32_t in_row = dil_y * input_x * input_ch;
    const int32_t ker_col = output_ch;
    const int32_t ker_row = kernel_x * output_ch;
    const int32_t span_x = (kernel_x - 1) * dil_x;
    const int32_t span_y = (kernel_y - 1) * dil_y;
    const int32_t px_step = stride_x * input_ch;

    /* Output x range needing no clipping */
    const int32_t ox_lo = ARM_NN_MIN((pad_x + stride_x - 1) / stride_x, output_x);
    const int32_t hi_num = input_x - 1 - span_x + pad_x;
    const int32_t ox_hi = hi_num < 0 ? 0 : ARM_NN_MIN(hi_num / stride_x + 1, output_x);

    /* Under four channels needs lane predicates */
    const int32_t tail = output_ch < 4;
    const int32_t mode_1 = !tail ? mode : (mode == DW_S16_IN_VEC ? DW_S16_IN_VEC_P : DW_S16_IN_GATHER_P);

    for (int32_t oc_i = 0; oc_i < output_ch; oc_i += 4)
    {
        /* Last block overlaps, rewriting equal outputs */
        const int32_t oc = tail ? 0 : ARM_NN_MIN(oc_i, output_ch - 4);
        const mve_pred16_t p = vctp32q((uint32_t)(output_ch - oc));
        const int32_t lanes = ARM_NN_MIN(4, output_ch - oc);
        const int64_t *bias_oc = bias ? bias + oc : NULL;
        const int32_t *mult_oc = output_mult + oc;
        const int32_t *shift_oc = output_shift + oc;
        /* Slow mode adds bias in s64 */
        const dw_s16_rq rq = dw_s16_rq_load(mult_oc, shift_oc, slow ? NULL : bias_oc, p);
        uint32_t offs_buf[4] = {0, 0, 0, 0};
        if (mode == DW_S16_IN_GATHER)
        {
            /* Clamp keeps tail lanes in range */
            for (int32_t i = 0; i < 4; i++)
            {
                offs_buf[i] = (uint32_t)(ARM_NN_MIN(oc + i, output_ch - 1) / ch_mult);
            }
        }
        const uint32x4_t offs = vld1q_u32(offs_buf);
        const int32_t in_off = mode == DW_S16_IN_VEC ? oc : (mode == DW_S16_IN_DUP ? oc / ch_mult : 0);
        const int16_t *in_batch = input + in_off;
        int16_t *out = output + oc;

        for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
        {
            for (int32_t oy = 0, base_y = -pad_y; oy < output_y; oy++, base_y += stride_y)
            {
                const int32_t ky_start = base_y >= 0 ? 0 : ARM_NN_MAX(0, (-base_y + dil_y - 1) / dil_y);
                const int32_t ky_end =
                    base_y + span_y < input_y ? kernel_y : ARM_NN_MIN(kernel_y, (input_y - base_y + dil_y - 1) / dil_y);
                const int32_t n_y = ARM_NN_MAX(ky_end - ky_start, 0);
                const int32_t in_y = (base_y + ky_start * dil_y) * input_x;
                const int8_t *ker_y = kernel + ky_start * ker_row + oc;

                for (int32_t ox = 0; ox < output_x;)
                {
                    const int32_t base_x = ox * stride_x - pad_x;
                    const int32_t n = ARM_NN_MIN(4, ox_hi - ox);
                    if (!tail && !slow && ox >= ox_lo && n > 1)
                    {
                        /* Two to four unclipped pixels */
                        int32x4_t acc[4] = {rq.bias, rq.bias, rq.bias, rq.bias};
                        const int16_t *ip_0 = in_batch + (in_y + base_x) * input_ch;
                        const int16_t *ip_1 = ip_0 + px_step;
                        const int16_t *ip_2 = n > 2 ? ip_0 + 2 * px_step : ip_0;
                        const int16_t *ip_3 = n > 3 ? ip_0 + 3 * px_step : ip_0;
                        if (mode == DW_S16_IN_VEC)
                        {
                            dw_s16_taps4(acc,
                                         ip_0,
                                         ip_1,
                                         ip_2,
                                         ip_3,
                                         ker_y,
                                         DW_S16_IN_VEC,
                                         offs,
                                         p,
                                         n_y,
                                         kernel_x,
                                         in_col,
                                         in_row,
                                         ker_col,
                                         ker_row);
                        }
                        else if (mode == DW_S16_IN_DUP)
                        {
                            dw_s16_taps4(acc,
                                         ip_0,
                                         ip_1,
                                         ip_2,
                                         ip_3,
                                         ker_y,
                                         DW_S16_IN_DUP,
                                         offs,
                                         p,
                                         n_y,
                                         kernel_x,
                                         in_col,
                                         in_row,
                                         ker_col,
                                         ker_row);
                        }
                        else
                        {
                            dw_s16_taps4(acc,
                                         ip_0,
                                         ip_1,
                                         ip_2,
                                         ip_3,
                                         ker_y,
                                         DW_S16_IN_GATHER,
                                         offs,
                                         p,
                                         n_y,
                                         kernel_x,
                                         in_col,
                                         in_row,
                                         ker_col,
                                         ker_row);
                        }
                        vstrhq_s32(out, dw_s16_rq_apply(acc[0], &rq, act_min, act_max));
                        vstrhq_s32(out + output_ch, dw_s16_rq_apply(acc[1], &rq, act_min, act_max));
                        if (n > 2)
                        {
                            vstrhq_s32(out + 2 * output_ch, dw_s16_rq_apply(acc[2], &rq, act_min, act_max));
                        }
                        if (n > 3)
                        {
                            vstrhq_s32(out + 3 * output_ch, dw_s16_rq_apply(acc[3], &rq, act_min, act_max));
                        }
                        out += n * output_ch;
                        ox += n;
                        continue;
                    }

                    const int32_t kx_start = base_x >= 0 ? 0 : ARM_NN_MAX(0, (-base_x + dil_x - 1) / dil_x);
                    const int32_t kx_end = base_x + span_x < input_x
                        ? kernel_x
                        : ARM_NN_MIN(kernel_x, (input_x - base_x + dil_x - 1) / dil_x);
                    const int32_t n_x = ARM_NN_MAX(kx_end - kx_start, 0);
                    const int16_t *ip = in_batch + ((in_y + base_x) * input_ch + kx_start * in_col);
                    const int8_t *rhs = ker_y + kx_start * ker_col;
                    int32x4_t acc = rq.bias;
                    if (mode_1 == DW_S16_IN_VEC)
                    {
                        acc = dw_s16_taps1(
                            acc, ip, rhs, DW_S16_IN_VEC, offs, p, n_y, n_x, in_col, in_row, ker_col, ker_row);
                    }
                    else if (mode_1 == DW_S16_IN_DUP)
                    {
                        acc = dw_s16_taps1(
                            acc, ip, rhs, DW_S16_IN_DUP, offs, p, n_y, n_x, in_col, in_row, ker_col, ker_row);
                    }
                    else if (mode_1 == DW_S16_IN_GATHER)
                    {
                        acc = dw_s16_taps1(
                            acc, ip, rhs, DW_S16_IN_GATHER, offs, p, n_y, n_x, in_col, in_row, ker_col, ker_row);
                    }
                    else if (mode_1 == DW_S16_IN_VEC_P)
                    {
                        acc = dw_s16_taps1(
                            acc, ip, rhs, DW_S16_IN_VEC_P, offs, p, n_y, n_x, in_col, in_row, ker_col, ker_row);
                    }
                    else
                    {
                        acc = dw_s16_taps1(
                            acc, ip, rhs, DW_S16_IN_GATHER_P, offs, p, n_y, n_x, in_col, in_row, ker_col, ker_row);
                    }
                    vstrhq_p_s32(
                        out, dw_s16_out(acc, &rq, slow, bias_oc, mult_oc, shift_oc, lanes, act_min, act_max), p);
                    out += output_ch;
                    ox++;
                }
            }
            in_batch += input_x * input_y * input_ch;
        }
    }
}

/*
 * MVE s16 depthwise convolution, any channel multiplier.
 *
 * Refer header file for details.
 */
bool arm_nn_depthwise_conv_s16_mve(const cmsis_nn_dw_conv_params *dw_conv_params,
                                   const cmsis_nn_per_channel_quant_params *quant_params,
                                   const cmsis_nn_dims *input_dims,
                                   const int16_t *input,
                                   const cmsis_nn_dims *filter_dims,
                                   const int8_t *kernel,
                                   const int64_t *bias,
                                   const cmsis_nn_dims *output_dims,
                                   int16_t *output)
{
    const int32_t taps = filter_dims->w * filter_dims->h;
    /* s32 sums need fewer than 512 taps */
    if (taps >= MAX_COL_COUNT)
    {
        return false;
    }
    const int32_t slow = !dw_s16_rq_ok(bias, quant_params->shift, input_dims->c * dw_conv_params->ch_mult, taps);
    dw_s16_mve(input,
               input_dims->n,
               input_dims->w,
               input_dims->h,
               input_dims->c,
               kernel,
               dw_conv_params->ch_mult,
               filter_dims->w,
               filter_dims->h,
               dw_conv_params->padding.w,
               dw_conv_params->padding.h,
               dw_conv_params->stride.w,
               dw_conv_params->stride.h,
               bias,
               output,
               quant_params->shift,
               quant_params->multiplier,
               output_dims->w,
               output_dims->h,
               dw_conv_params->activation.min,
               dw_conv_params->activation.max,
               dw_conv_params->dilation.w,
               dw_conv_params->dilation.h,
               slow);
    return true;
}
#endif

/*
 *  Basic s16 depthwise convolution function.
 *
 *  Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_s16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params *dw_conv_params,
                                           const cmsis_nn_per_channel_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int16_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const int64_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           int16_t *output)
{
    const uint16_t dilation_x = dw_conv_params->dilation.w;
    const uint16_t dilation_y = dw_conv_params->dilation.h;

    (void)bias_dims;
    (void)ctx;

#if defined(ARM_MATH_MVEI)
    if (arm_nn_depthwise_conv_s16_mve(
            dw_conv_params, quant_params, input_dims, input, filter_dims, kernel, bias, output_dims, output))
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
#endif

    depthwise_conv_s16_generic_s16(input,
                                   input_dims->n,
                                   input_dims->w,
                                   input_dims->h,
                                   input_dims->c,
                                   kernel,
                                   dw_conv_params->ch_mult,
                                   filter_dims->w,
                                   filter_dims->h,
                                   dw_conv_params->padding.w,
                                   dw_conv_params->padding.h,
                                   dw_conv_params->stride.w,
                                   dw_conv_params->stride.h,
                                   bias,
                                   output,
                                   quant_params->shift,
                                   quant_params->multiplier,
                                   output_dims->w,
                                   output_dims->h,
                                   dw_conv_params->activation.min,
                                   dw_conv_params->activation.max,
                                   dilation_x,
                                   dilation_y);

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
