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
 * Title:        arm_nn_depthwise_conv_s8_planar.c
 * Description:  s8 depthwise convolution, channel multiplier 1, stride 1, vectorized across output pixels of one
 *               channel plane instead of across channels.
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/* Plane bytes past the last row that a block or a 16-lane tap load may read; those lanes are never stored. */
#define DW_PLANAR_SLACK (32)

#if defined(ARM_MATH_MVEI)

/* Output pixels computed per block: four int32x4 accumulators. */
    #define DW_PLANAR_BLOCK (16)

/* Fills n plane bytes: [0, i_start) and [i_end, n) take -input_offset, [i_start, i_end) are gathered from src at
   an element stride of step bytes. A padded tap then contributes (-input_offset) * w, which the weight sum cancels
   exactly as the im2col path does. */
static void dw_planar_fill(int8_t *dst,
                           const int32_t n,
                           const int32_t i_start,
                           const int32_t i_end,
                           const int8_t *src,
                           const int32_t step,
                           const int8_t pad_val)
{
    const int32_t count = i_end - i_start;
    if (i_start > 0)
    {
        arm_memset_s8(dst, pad_val, (uint32_t)i_start);
    }

    int8_t *out = dst + i_start;
    if (step == 1)
    {
        arm_memcpy_s8(out, src, (uint32_t)count);
    }
    else if (step <= 17)
    {
        const uint8x16_t offs = vmulq_n_u8(vidupq_u8((uint32_t)0, 1), (uint8_t)step);
        for (int32_t i = 0; i < count; i += 16)
        {
            const mve_pred16_t p = vctp8q((uint32_t)(count - i));
            vstrbq_p_s8(out, vldrbq_gather_offset_z_s8(src, offs, p), p);
            src += 16 * step;
            out += 16;
        }
    }
    else
    {
        const uint16x8_t offs = vmulq_n_u16(vidupq_u16((uint32_t)0, 1), (uint16_t)step);
        for (int32_t i = 0; i < count; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(count - i));
            vstrbq_p_s16(out, vldrbq_gather_offset_z_s16(src, offs, p), p);
            src += 8 * step;
            out += 8;
        }
    }

    if (i_end < n)
    {
        arm_memset_s8(dst + i_end, pad_val, (uint32_t)(n - i_end));
    }
}

static inline int32x4_t dw_planar_requantize(int32x4_t acc,
                                             const int32_t mult,
                                             const int32_t shift,
                                             const int32_t out_offset,
                                             const int32_t act_min,
                                             const int32_t act_max)
{
    acc = arm_requantize_mve(acc, mult, shift);
    acc = vaddq_n_s32(acc, out_offset);
    acc = vmaxq_s32(acc, vdupq_n_s32(act_min));
    return vminq_s32(acc, vdupq_n_s32(act_max));
}

static inline void
dw_planar_store(int8_t *out, const uint32x4_t offs, const int32x4_t v, const int32_t out_step, int32_t rem)
{
    if (out_step == 1)
    {
        vstrbq_p_s32(out, v, vctp32q((uint32_t)rem));
    }
    else
    {
        vstrbq_scatter_offset_p_s32(out, offs, v, vctp32q((uint32_t)rem));
    }
}

#endif

/* A 1xk kernel of 5..16 taps runs as one dot product per output pixel over a plane split into dilation phases, so the
   taps of every output are contiguous. */
static bool dw_planar_use_dot(const cmsis_nn_dw_conv_params *dw_conv_params,
                              const cmsis_nn_dims *input_dims,
                              const cmsis_nn_dims *filter_dims,
                              const cmsis_nn_dims *output_dims)
{
    return output_dims->h == 1 && input_dims->h == 1 && filter_dims->h == 1 && dw_conv_params->padding.h == 0 &&
        filter_dims->w >= 5 && filter_dims->w <= 16;
}

/*
 * Plane size of the planar path. The rule is plain C so it evaluates the same on every build.
 *
 * Refer header file for details.
 *
 */
int32_t arm_nn_depthwise_conv_s8_planar_bytes(const cmsis_nn_dw_conv_params *dw_conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *output_dims)
{
    if (input_dims->c != output_dims->c || input_dims->n != 1 || dw_conv_params->ch_mult != 1 ||
        dw_conv_params->stride.w != 1 || dw_conv_params->stride.h != 1 || dw_conv_params->dilation.h != 1 ||
        dw_conv_params->dilation.w < 1 || input_dims->c < 1 || output_dims->w < 1 || output_dims->h < 1 ||
        filter_dims->w < 1 || filter_dims->h < 1)
    {
        return -1;
    }
    /* The channel-vectorized path amortizes its per-tap copy over the channels, so it stays ahead for wide channel
       counts and for narrow planes that leave most of a 16-pixel block idle. */
    const int32_t ch = input_dims->c;
    const int32_t dilation_x = dw_conv_params->dilation.w;
    /* On Apollo510 the channel-vectorized path is within 1.15x at C = 40 and at C = 32 with dilation 8. The bound on
       C * dilation also keeps the gather offsets of dw_planar_fill() (step * 15 or step * 7) inside their lane type. */
    if (ch > 32 || dilation_x > 128 / ch)
    {
        return -1;
    }
    /* Sizes are formed in 64 bits and rejected past INT32_MAX, each side before the product so it cannot overflow; the
       caller also rejects any plane above the scratch. */
    const int64_t plane_w = (int64_t)output_dims->w + (int64_t)(filter_dims->w - 1) * dilation_x;
    if (plane_w > INT32_MAX)
    {
        return -1;
    }
    int64_t bytes;
    if (dw_planar_use_dot(dw_conv_params, input_dims, filter_dims, output_dims))
    {
        /* Past 16 channels the per-phase fill and the per-channel loop cost more than they save on short phases. */
        if (output_dims->w < 8 || (ch > 16 && output_dims->w < 24 * dilation_x))
        {
            return -1;
        }
        bytes = ((plane_w + dilation_x - 1) / dilation_x) * dilation_x + DW_PLANAR_SLACK;
        return bytes > INT32_MAX ? -1 : (int32_t)bytes;
    }
    const bool profitable = output_dims->w >= 8 &&
        ((output_dims->h == 1) ? (ch <= 16 || (ch <= 32 && output_dims->w >= 32))
                               : (ch <= 8 || (ch <= 16 && output_dims->w >= 16)));
    if (!profitable)
    {
        return -1;
    }
    const int64_t plane_h = (int64_t)output_dims->h + filter_dims->h - 1;
    if (plane_h > INT32_MAX)
    {
        return -1;
    }
    bytes = plane_w * plane_h + DW_PLANAR_SLACK;
    return bytes > INT32_MAX ? -1 : (int32_t)bytes;
}

#if defined(ARM_MATH_MVEI)
static void dw_planar_dot_1xk(const int8_t *input,
                              int8_t *output,
                              int8_t *plane,
                              const int8_t *kernel,
                              const int32_t *weight_sum,
                              const cmsis_nn_per_channel_quant_params *quant_params,
                              const int32_t ch,
                              const int32_t input_x,
                              const int32_t output_x,
                              const int32_t kernel_x,
                              const int32_t pad_x,
                              const int32_t dilation_x,
                              const int32_t out_offset,
                              const int32_t act_min,
                              const int32_t act_max,
                              const int8_t pad_val)
{
    const int32_t plane_w = output_x + (kernel_x - 1) * dilation_x;
    const int32_t phase_len = (plane_w + dilation_x - 1) / dilation_x;
    const int32_t out_step = dilation_x * ch;
    const uint32x4_t out_offs = vmulq_n_u32(vidupq_u32((uint32_t)0, 1), (uint32_t)out_step);

    arm_memset_s8(plane + dilation_x * phase_len, pad_val, DW_PLANAR_SLACK);

    for (int32_t i_ch = 0; i_ch < ch; i_ch++)
    {
        /* Phase r holds plane columns r, r + dilation_x, r + 2 * dilation_x, ... */
        for (int32_t r = 0; r < dilation_x; r++)
        {
            const int32_t n = (plane_w - r + dilation_x - 1) / dilation_x;
            const int32_t lo = pad_x - r;
            const int32_t hi = pad_x + input_x - r;
            const int32_t i_start = ARM_NN_MIN(n, lo > 0 ? (lo + dilation_x - 1) / dilation_x : 0);
            const int32_t i_end = ARM_NN_MAX(i_start, ARM_NN_MIN(n, hi > 0 ? (hi + dilation_x - 1) / dilation_x : 0));
            dw_planar_fill(plane + r * phase_len,
                           n,
                           i_start,
                           i_end,
                           input + (r + i_start * dilation_x - pad_x) * ch + i_ch,
                           out_step,
                           pad_val);
        }

        int8_t taps[16] = {0};
        for (int32_t k = 0; k < kernel_x; k++)
        {
            taps[k] = kernel[k * ch + i_ch];
        }
        /* Lanes past the last tap have a zero weight, so the bytes they read never reach the sum. */
        const int8x16_t w_vec = vldrbq_s8(taps);
        const int32_t acc_init = weight_sum[i_ch];
        const int32_t mult = quant_params->multiplier[i_ch];
        const int32_t shift = quant_params->shift[i_ch];

        for (int32_t r = 0; r < dilation_x && r < output_x; r++)
        {
            const int32_t n_out = (output_x - r + dilation_x - 1) / dilation_x;
            const int8_t *phase = plane + r * phase_len;
            int8_t *out = output + r * ch + i_ch;
            for (int32_t j = 0; j < n_out; j += 4)
            {
                const int8_t *src = phase + j;
                int32x4_t acc = vdupq_n_s32(0);
                acc = vsetq_lane_s32(vmladavaq_s8(acc_init, vldrbq_s8(src), w_vec), acc, 0);
                acc = vsetq_lane_s32(vmladavaq_s8(acc_init, vldrbq_s8(src + 1), w_vec), acc, 1);
                acc = vsetq_lane_s32(vmladavaq_s8(acc_init, vldrbq_s8(src + 2), w_vec), acc, 2);
                acc = vsetq_lane_s32(vmladavaq_s8(acc_init, vldrbq_s8(src + 3), w_vec), acc, 3);
                dw_planar_store(out + j * out_step,
                                out_offs,
                                dw_planar_requantize(acc, mult, shift, out_offset, act_min, act_max),
                                out_step,
                                n_out - j);
            }
        }
    }
}
#endif

/*
 * s8 depthwise convolution vectorized across the output pixels of one channel plane.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_depthwise_conv_s8_planar(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_context *weight_sum_ctx,
                                                    const cmsis_nn_dw_conv_params *dw_conv_params,
                                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const int8_t *input,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const int8_t *kernel,
                                                    const cmsis_nn_dims *output_dims,
                                                    int8_t *output)
{
#if defined(ARM_MATH_MVEI)
    const int32_t plane_bytes =
        arm_nn_depthwise_conv_s8_planar_bytes(dw_conv_params, input_dims, filter_dims, output_dims);
    if (plane_bytes < 0 || ctx == NULL || ctx->buf == NULL || plane_bytes > ctx->size || weight_sum_ctx == NULL ||
        weight_sum_ctx->buf == NULL || plane_bytes > arm_depthwise_conv_s8_opt_get_buffer_size(input_dims, filter_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    const int32_t ch = input_dims->c;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t dilation_x = dw_conv_params->dilation.w;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t out_offset = dw_conv_params->output_offset;
    const int32_t act_min = dw_conv_params->activation.min;
    const int32_t act_max = dw_conv_params->activation.max;
    const int8_t pad_val = (int8_t)(-dw_conv_params->input_offset);
    const int32_t plane_w = output_x + (kernel_x - 1) * dilation_x;
    const int32_t plane_h = output_y + kernel_y - 1;
    const int32_t *weight_sum = (const int32_t *)weight_sum_ctx->buf;
    int8_t *plane = (int8_t *)ctx->buf;

    if (dw_planar_use_dot(dw_conv_params, input_dims, filter_dims, output_dims))
    {
        dw_planar_dot_1xk(input,
                          output,
                          plane,
                          kernel,
                          weight_sum,
                          quant_params,
                          ch,
                          input_x,
                          output_x,
                          kernel_x,
                          pad_x,
                          dilation_x,
                          out_offset,
                          act_min,
                          act_max,
                          pad_val);
        return ARM_CMSIS_NN_SUCCESS;
    }

    arm_memset_s8(plane + plane_w * plane_h, pad_val, DW_PLANAR_SLACK);
    const uint32x4_t out_offs = vmulq_n_u32(vidupq_u32((uint32_t)0, 1), (uint32_t)ch);
    const int32_t block_step = 4 * ch;

    for (int32_t i_ch = 0; i_ch < ch; i_ch++)
    {
        for (int32_t py = 0; py < plane_h; py++)
        {
            const int32_t iy = py - pad_y;
            int8_t *row = plane + py * plane_w;
            if (iy < 0 || iy >= input_y)
            {
                arm_memset_s8(row, pad_val, (uint32_t)plane_w);
            }
            else
            {
                const int32_t x_start = ARM_NN_MAX(0, ARM_NN_MIN(pad_x, plane_w));
                const int32_t x_end = ARM_NN_MAX(x_start, ARM_NN_MIN(pad_x + input_x, plane_w));
                dw_planar_fill(
                    row, plane_w, x_start, x_end, input + (iy * input_x + x_start - pad_x) * ch + i_ch, ch, pad_val);
            }
        }

        const int32_t acc_init = weight_sum[i_ch];
        const int32_t mult = quant_params->multiplier[i_ch];
        const int32_t shift = quant_params->shift[i_ch];
        const int8_t *ker = kernel + i_ch;

        for (int32_t oy = 0; oy < output_y; oy++)
        {
            const int8_t *plane_row = plane + oy * plane_w;
            int8_t *out_row = output + oy * output_x * ch + i_ch;

            for (int32_t ox = 0; ox < output_x; ox += DW_PLANAR_BLOCK)
            {
                int32x4_t acc_0 = vdupq_n_s32(acc_init);
                int32x4_t acc_1 = acc_0;
                int32x4_t acc_2 = acc_0;
                int32x4_t acc_3 = acc_0;

                const int8_t *ker_ptr = ker;
                for (int32_t ky = 0; ky < kernel_y; ky++)
                {
                    const int8_t *src = plane_row + ky * plane_w + ox;
                    for (int32_t kx = 0; kx < kernel_x; kx++)
                    {
                        const int32_t w = *ker_ptr;
                        ker_ptr += ch;
                        acc_0 = vmlaq_n_s32(acc_0, vldrbq_s32(src), w);
                        acc_1 = vmlaq_n_s32(acc_1, vldrbq_s32(src + 4), w);
                        acc_2 = vmlaq_n_s32(acc_2, vldrbq_s32(src + 8), w);
                        acc_3 = vmlaq_n_s32(acc_3, vldrbq_s32(src + 12), w);
                        src += dilation_x;
                    }
                }

                const int32_t rem = output_x - ox;
                int8_t *out = out_row + ox * ch;
                dw_planar_store(
                    out, out_offs, dw_planar_requantize(acc_0, mult, shift, out_offset, act_min, act_max), ch, rem);
                if (rem > 4)
                {
                    dw_planar_store(out + block_step,
                                    out_offs,
                                    dw_planar_requantize(acc_1, mult, shift, out_offset, act_min, act_max),
                                    ch,
                                    rem - 4);
                }
                if (rem > 8)
                {
                    dw_planar_store(out + 2 * block_step,
                                    out_offs,
                                    dw_planar_requantize(acc_2, mult, shift, out_offset, act_min, act_max),
                                    ch,
                                    rem - 8);
                }
                if (rem > 12)
                {
                    dw_planar_store(out + 3 * block_step,
                                    out_offs,
                                    dw_planar_requantize(acc_3, mult, shift, out_offset, act_min, act_max),
                                    ch,
                                    rem - 12);
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
#else
    (void)ctx;
    (void)weight_sum_ctx;
    (void)dw_conv_params;
    (void)quant_params;
    (void)input_dims;
    (void)input;
    (void)filter_dims;
    (void)kernel;
    (void)output_dims;
    (void)output;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}

/**
 * @} end of Doxygen group
 */
