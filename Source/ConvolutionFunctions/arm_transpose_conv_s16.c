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
 * Title:        arm_transpose_conv_s16.c
 * Description:  s16 transpose convolution with int8 weights and int64 bias.
 *
 * $Date:        8 October 2026
 * $Revision:    V.1.1.0
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
/* Requantize with a reduced multiplier. */
static int16_t tconv_requant_mve(const int64_t acc,
                                 const int32_t reduced,
                                 const int32_t total_shift,
                                 const int32_t act_min,
                                 const int32_t act_max)
{
    /* Rounding right shift, then saturate to int32. */
    const int64_t result = sqrshrl(acc * reduced, total_shift);
    int32_t val = (int32_t)(sqshll(result, 32) >> 32);

    val = ARM_NN_MAX(val, act_min);
    val = ARM_NN_MIN(val, act_max);
    return (int16_t)val;
}

/* Requantize four int32 sums and store. */
__STATIC_FORCEINLINE void tconv_s16_store4(const int32_t s0,
                                           const int32_t s1,
                                           const int32_t s2,
                                           const int32_t s3,
                                           const int32x4_t bias,
                                           const int32x4_t mult,
                                           const int32x4_t act_min,
                                           const int32x4_t act_max,
                                           const mve_pred16_t pred,
                                           int16_t *out)
{
    int32x4_t res = vdupq_n_s32(s0);
    res = vsetq_lane_s32(s1, res, 1);
    res = vsetq_lane_s32(s2, res, 2);
    res = vsetq_lane_s32(s3, res, 3);
    res = vqrdmulhq_s32(vaddq_s32(res, bias), mult);
    res = vmaxq_s32(res, act_min);
    res = vminq_s32(res, act_max);
    vstrhq_p_s32(out, res, pred);
}

/* Dot one pixel, or two if next. */
static void tconv_s16_pixel(const int16_t *src,
                            const int32_t next,
                            const int32_t row_step,
                            const int32_t n_rows,
                            const int32_t row8,
                            const int8_t *w,
                            const int32_t *mult_buf,
                            const int32_t *bias_buf,
                            const int32_t group_ch,
                            const cmsis_nn_activation *act,
                            int16_t *out,
                            const int32_t out_next)
{
    const int32x4_t act_min = vdupq_n_s32(act->min);
    const int32x4_t act_max = vdupq_n_s32(act->max);

    for (int32_t oc = 0; oc < group_ch; oc += 4)
    {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        int32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        for (int32_t j = 0; j < n_rows; j++)
        {
            const int16_t *cp = src + j * row_step;

            /* Weights hold 8 values per channel. */
            if (next == 0)
            {
                for (int32_t n = row8 >> 3; n > 0; n--)
                {
                    const int16x8_t x = vldrhq_s16(cp);
                    cp += 8;
                    a0 = vmladavaq_s16(a0, x, vldrbq_s16(w));
                    a1 = vmladavaq_s16(a1, x, vldrbq_s16(w + 8));
                    a2 = vmladavaq_s16(a2, x, vldrbq_s16(w + 16));
                    a3 = vmladavaq_s16(a3, x, vldrbq_s16(w + 24));
                    w += 32;
                }
                continue;
            }

            /* Second pixel shares each weight load. */
            const int16_t *cq = cp + next;
            for (int32_t n = row8 >> 3; n > 0; n--)
            {
                const int16x8_t x = vldrhq_s16(cp);
                const int16x8_t y = vldrhq_s16(cq);
                cp += 8;
                cq += 8;
                int16x8_t wv = vldrbq_s16(w);
                a0 = vmladavaq_s16(a0, x, wv);
                b0 = vmladavaq_s16(b0, y, wv);
                wv = vldrbq_s16(w + 8);
                a1 = vmladavaq_s16(a1, x, wv);
                b1 = vmladavaq_s16(b1, y, wv);
                wv = vldrbq_s16(w + 16);
                a2 = vmladavaq_s16(a2, x, wv);
                b2 = vmladavaq_s16(b2, y, wv);
                wv = vldrbq_s16(w + 24);
                a3 = vmladavaq_s16(a3, x, wv);
                b3 = vmladavaq_s16(b3, y, wv);
                w += 32;
            }
        }

        /* One pixel: the a store overwrites. */
        const int32x4_t bias = vldrwq_s32(bias_buf + oc);
        const int32x4_t mult = vldrwq_s32(mult_buf + oc);
        const mve_pred16_t pred = vctp32q((uint32_t)(group_ch - oc));
        tconv_s16_store4(b0, b1, b2, b3, bias, mult, act_min, act_max, pred, out + out_next + oc);
        tconv_s16_store4(a0, a1, a2, a3, bias, mult, act_min, act_max, pred, out + oc);
    }
}
#else
/* TFLite int64 requantization, then activation clamp. */
static int16_t tconv_requant_s16(const int64_t acc,
                                 const int32_t multiplier,
                                 const int32_t shift,
                                 const int32_t act_min,
                                 const int32_t act_max)
{
    const int64_t reduced = REDUCE_MULTIPLIER(multiplier);
    const int32_t total_shift = 15 - shift;
    int64_t result = (acc * reduced + ((int64_t)1 << (total_shift - 1))) >> total_shift;

    result = ARM_NN_MAX(result, (int64_t)act_min);
    result = ARM_NN_MIN(result, (int64_t)act_max);
    return (int16_t)result;
}
#endif

/*
 * Basic s16 transpose convolution function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_transpose_conv_s16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_context *output_ctx,
                                           const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                           const cmsis_nn_per_channel_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int16_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const int64_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           int16_t *output_data)
{
    (void)output_ctx;
    (void)bias_dims;

    const int32_t stride_x = transpose_conv_params->stride.w;
    const int32_t stride_y = transpose_conv_params->stride.h;

    if (stride_x <= 0 || stride_y <= 0 || transpose_conv_params->dilation.w != 1 ||
        transpose_conv_params->dilation.h != 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batches = input_dims->n;
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t input_ch = input_dims->c;

    const int32_t filter_y = filter_dims->h;
    const int32_t filter_x = filter_dims->w;

    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_ch = output_dims->c;

    const int32_t pad_x = transpose_conv_params->padding.w;
    const int32_t pad_y = transpose_conv_params->padding.h;

    const int32_t act_min = transpose_conv_params->activation.min;
    const int32_t act_max = transpose_conv_params->activation.max;

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const int32_t taps_y = filter_y / stride_y + (filter_y % stride_y != 0);
    const int32_t taps_x = filter_x / stride_x + (filter_x % stride_x != 0);
    const int32_t batch_size = input_y * input_x * input_ch;
    const int32_t in_size = batches * batch_size;

    /* int32 path needs depth in [1, 255]. */
    int32_t group_pad = 0;
    if (ctx != NULL && ctx->buf != NULL && (uint64_t)((int64_t)taps_y * taps_x * input_ch - 1) < 255)
    {
        group_pad = 4;
        for (int32_t oc = 0; oc < output_ch; oc++)
        {
            const int64_t bias = bias_data != NULL ? bias_data[oc] : 0;
            const int32_t shift = quant_params->shift[oc];
            if (shift < -16 || shift > 0 || bias < -(1LL << 30) || bias > (1LL << 30))
            {
                group_pad = 0;
            }
        }
    }

    /* Channels per group that fit. */
    const int32_t col_max = group_pad != 0 ? taps_y * ((taps_x * input_ch + 7) & ~7) : 0;
    if (group_pad != 0)
    {
        group_pad = ((ctx->size - 32 - col_max * 2) / (8 + col_max)) & ~3;
        group_pad = ARM_NN_MIN(group_pad, (output_ch + 3) & ~3);
    }

    if (group_pad >= 4)
    {
        /* Scratch: bias, multipliers, column, weights. */
        int32_t *bias_buf = (int32_t *)(((uintptr_t)ctx->buf + 3) & ~(uintptr_t)3);
        int32_t *mult_buf = bias_buf + group_pad;
        int16_t *col = (int16_t *)(mult_buf + group_pad);
        int8_t *w_buf = (int8_t *)(col + col_max);

        /* Zero once: tail lanes stay defined. */
        arm_memset_s8((int8_t *)bias_buf, 0, group_pad * (8 + col_max) + col_max * 2);

        /* Each stride phase uses a fixed tap grid. */
        for (int32_t py = 0; py < stride_y; py++)
        {
            int32_t oy_start = (py - pad_y) % stride_y;
            if (oy_start < 0)
            {
                oy_start += stride_y;
            }
            if (oy_start >= output_y)
            {
                continue;
            }
            const int32_t qy_start = (oy_start + pad_y - py) / stride_y;
            const int32_t nty = py < filter_y ? (filter_y - py - 1) / stride_y + 1 : 0;

            for (int32_t px = 0; px < stride_x; px++)
            {
                int32_t ox_start = (px - pad_x) % stride_x;
                if (ox_start < 0)
                {
                    ox_start += stride_x;
                }
                if (ox_start >= output_x)
                {
                    continue;
                }
                const int32_t qx_start = (ox_start + pad_x - px) / stride_x;
                const int32_t ntx = px < filter_x ? (filter_x - px - 1) / stride_x + 1 : 0;

                /* Rows of taps, each padded to 8. */
                const int32_t row_len = ntx * input_ch;
                const int32_t row8 = (row_len + 7) & ~7;
                const int32_t len8 = nty * row8;

                for (int32_t oc_start = 0; oc_start < output_ch; oc_start += group_pad)
                {
                    const int32_t group_ch = ARM_NN_MIN(group_pad, output_ch - oc_start);

                    for (int32_t g = 0; g < group_ch; g++)
                    {
                        const int32_t oc = oc_start + g;
                        const int32_t reduced = REDUCE_MULTIPLIER(quant_params->multiplier[oc]);
                        bias_buf[g] = bias_data != NULL ? (int32_t)bias_data[oc] : 0;
                        /* Doubling high multiply does the shift. */
                        mult_buf[g] = (int32_t)((uint32_t)reduced << (16 + quant_params->shift[oc]));

                        /* Pack phase weights, kx reversed. */
                        int8_t *tmp = (int8_t *)col;
                        for (int32_t j = 0; j < nty; j++)
                        {
                            const int32_t ky = py + j * stride_y;
                            for (int32_t i = 0; i < ntx; i++)
                            {
                                const int32_t kx = px + (ntx - 1 - i) * stride_x;
                                arm_memcpy_s8(
                                    tmp, filter_data + ((oc * filter_y + ky) * filter_x + kx) * input_ch, input_ch);
                                tmp += input_ch;
                            }
                            arm_memset_s8(tmp, 0, row8 - row_len);
                            tmp += row8 - row_len;
                        }

                        /* Interleave 8-value runs of 4 channels. */
                        int8_t *dst = w_buf + (g >> 2) * 4 * len8 + (g & 3) * 8;
                        for (int32_t k = 0; k < len8; k += 8)
                        {
                            memcpy(dst, (const int8_t *)col + k, 8);
                            dst += 32;
                        }
                    }

                    for (int32_t b = 0; b < batches; b++)
                    {
                        const int16_t *batch_in = input_data + b * batch_size;
                        int16_t *batch_out = output_data + b * output_y * output_x * output_ch + oc_start;

                        for (int32_t oy = oy_start, qy = qy_start; oy < output_y; oy += stride_y, qy++)
                        {
                            for (int32_t ox = ox_start, qx = qx_start; ox < output_x; ox += stride_x, qx++)
                            {
                                /* Offsets keep pointers in range. */
                                const int32_t ix_first = qx - ntx + 1;
                                const int32_t src_off = (qy * input_x + ix_first) * input_ch;
                                const int32_t end_off = b * batch_size + src_off + row8;
                                const int32_t inside = qy - nty + 1 >= 0 && qy < input_y && ix_first >= 0;
                                const int16_t *src = col;
                                int32_t row_step = row8;
                                int32_t next = 0;

                                if (inside && ix_first + ntx <= input_x && end_off <= in_size)
                                {
                                    /* Interior: read input rows in place. */
                                    src = batch_in + src_off;
                                    row_step = -input_x * input_ch;
                                    if (ox + stride_x < output_x && ix_first + ntx < input_x &&
                                        end_off + input_ch <= in_size)
                                    {
                                        /* Two interior pixels at once. */
                                        next = input_ch;
                                    }
                                }
                                else
                                {
                                    /* Gather valid taps, zero the rest. */
                                    const int32_t i_lo = ARM_NN_MAX(0, -ix_first);
                                    const int32_t i_hi = ARM_NN_MIN(ntx, input_x - ix_first);
                                    int16_t *c = col;
                                    for (int32_t j = 0; j < nty; j++)
                                    {
                                        const int32_t iy = qy - j;
                                        arm_memset_s8((int8_t *)c, 0, row_len * 2);
                                        if (iy >= 0 && iy < input_y && i_lo < i_hi)
                                        {
                                            arm_memcpy_s8((int8_t *)(c + i_lo * input_ch),
                                                          (const int8_t *)(batch_in +
                                                                           (iy * input_x + ix_first + i_lo) * input_ch),
                                                          (i_hi - i_lo) * input_ch * 2);
                                        }
                                        c += row8;
                                    }
                                }

                                tconv_s16_pixel(src,
                                                next,
                                                row_step,
                                                nty,
                                                row8,
                                                w_buf,
                                                mult_buf,
                                                bias_buf,
                                                group_ch,
                                                &transpose_conv_params->activation,
                                                batch_out + (oy * output_x + ox) * output_ch,
                                                next != 0 ? stride_x * output_ch : 0);
                                if (next != 0)
                                {
                                    ox += stride_x;
                                    qx++;
                                }
                            }
                        }
                    }
                }
            }
        }
        return ARM_CMSIS_NN_SUCCESS;
    }

    /* No scratch: walk the valid taps per output. */
    int16_t *out = output_data;

    for (int32_t b = 0; b < batches; b++)
    {
        const int16_t *batch_in = input_data + b * batch_size;

        for (int32_t oy = 0; oy < output_y; oy++)
        {
            /* Valid ky: ky = oy + pad_y - iy * stride_y. */
            const int32_t ry = oy + pad_y;
            const int32_t ky_hi = ARM_NN_MIN(filter_y - 1, ry);
            int32_t ky_lo = ARM_NN_MAX(0, ry - (input_y - 1) * stride_y);
            if (ry >= ky_lo)
            {
                ky_lo += (ry - ky_lo) % stride_y;
            }
            const int32_t iy_lo = (ry - ky_lo) / stride_y;

            for (int32_t ox = 0; ox < output_x; ox++)
            {
                const int32_t rx = ox + pad_x;
                const int32_t kx_hi = ARM_NN_MIN(filter_x - 1, rx);
                int32_t kx_lo = ARM_NN_MAX(0, rx - (input_x - 1) * stride_x);
                if (rx >= kx_lo)
                {
                    kx_lo += (rx - kx_lo) % stride_x;
                }
                const int32_t ix_lo = (rx - kx_lo) / stride_x;

                for (int32_t oc = 0; oc < output_ch; oc++)
                {
                    int64_t acc = bias_data != NULL ? bias_data[oc] : 0;

                    for (int32_t ky = ky_lo, iy = iy_lo; ky <= ky_hi; ky += stride_y, iy--)
                    {
                        for (int32_t kx = kx_lo, ix = ix_lo; kx <= kx_hi; kx += stride_x, ix--)
                        {
                            const int16_t *ip = batch_in + (iy * input_x + ix) * input_ch;
                            const int8_t *wp = filter_data + ((oc * filter_y + ky) * filter_x + kx) * input_ch;
                            for (int32_t i = 0; i < input_ch; i += 8)
                            {
                                const mve_pred16_t p = vctp16q((uint32_t)(input_ch - i));
                                acc = vmlaldavaq_s16(acc, vldrhq_z_s16(ip + i, p), vldrbq_z_s16(wp + i, p));
                            }
                        }
                    }

                    *out++ = tconv_requant_mve(acc,
                                               REDUCE_MULTIPLIER(quant_params->multiplier[oc]),
                                               15 - quant_params->shift[oc],
                                               act_min,
                                               act_max);
                }
            }
        }
    }
#else
    (void)ctx;
    int16_t *out = output_data;

    for (int32_t b = 0; b < batches; b++)
    {
        const int16_t *batch_in = input_data + (int64_t)b * input_y * input_x * input_ch;

        for (int32_t oy = 0; oy < output_y; oy++)
        {
            for (int32_t ox = 0; ox < output_x; ox++)
            {
                for (int32_t oc = 0; oc < output_ch; oc++)
                {
                    int64_t acc = 0;

                    for (int32_t ky = 0; ky < filter_y; ky++)
                    {
                        /* Find the input row hitting oy. */
                        const int32_t ty = oy + pad_y - ky;
                        if (ty < 0 || ty % stride_y != 0 || ty / stride_y >= input_y)
                        {
                            continue;
                        }
                        const int32_t iy = ty / stride_y;

                        for (int32_t kx = 0; kx < filter_x; kx++)
                        {
                            const int32_t tx = ox + pad_x - kx;
                            if (tx < 0 || tx % stride_x != 0 || tx / stride_x >= input_x)
                            {
                                continue;
                            }
                            const int32_t ix = tx / stride_x;

                            const int16_t *in = batch_in + ((int64_t)iy * input_x + ix) * input_ch;
                            const int8_t *w = filter_data + (((int64_t)oc * filter_y + ky) * filter_x + kx) * input_ch;

                            for (int32_t ic = 0; ic < input_ch; ic++)
                            {
                                acc += (int32_t)in[ic] * (int32_t)w[ic];
                            }
                        }
                    }

                    if (bias_data)
                    {
                        acc += bias_data[oc];
                    }

                    *out++ =
                        tconv_requant_s16(acc, quant_params->multiplier[oc], quant_params->shift[oc], act_min, act_max);
                }
            }
        }
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
