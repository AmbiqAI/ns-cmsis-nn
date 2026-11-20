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
 * Title:        arm_nn_depthwise_conv_s8_direct.c
 * Description:  s8 depthwise convolution, channel multiplier 1, few channels. Reads the input in place instead of
 *               building im2col columns, four output pixels at a time.
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

#if defined(ARM_MATH_MVEI)

/* Layer constants */
typedef struct
{
    const int8_t *input;
    const int8_t *kernel;
    const int32_t *weight_sum;
    const int32_t *mult;
    const int32_t *shift;
    int32_t ch;
    int32_t input_x;
    int32_t input_y;
    int32_t kernel_x;
    int32_t kernel_y;
    int32_t stride_x;
    int32_t dilation_x;
    int32_t pad_val;
    int32_t out_offset;
    int32_t act_min;
    int32_t act_max;
} dw_direct_args;

/* Per-group values shared by all channels */
typedef struct
{
    const int8_t *in;
    const int8_t *ker;
    int32_t rows;
    int32_t row_adj;
    int32_t pad_lo;
    int32_t pad_hi;
    int32_t o1;
    int32_t o2;
    int32_t o3;
    int32_t num_px;
} dw_direct_group;

/* Loads four channels */
__STATIC_FORCEINLINE int32x4_t dw_load(const int8_t *src, const mve_pred16_t p, const bool tail)
{
    return tail ? vldrbq_z_s32(src, p) : vldrbq_s32(src);
}

/* Loads four channel words */
__STATIC_FORCEINLINE int32x4_t dw_load_w(const int32_t *src, const mve_pred16_t p, const bool tail)
{
    return tail ? vldrwq_z_s32(src, p) : vldrwq_s32(src);
}

/* Requantizes and stores four channels */
__STATIC_FORCEINLINE void dw_store(int8_t *out,
                                   int32x4_t acc,
                                   const int32x4_t mult,
                                   const int32x4_t shift,
                                   const dw_direct_args *a,
                                   const mve_pred16_t p,
                                   const bool tail)
{
    acc = arm_requantize_mve_32x4(acc, mult, shift);
    acc = vaddq_n_s32(acc, a->out_offset);
    acc = vminq_s32(vmaxq_s32(acc, vdupq_n_s32(a->act_min)), vdupq_n_s32(a->act_max));
    if (tail)
    {
        vstrbq_p_s32(out, acc, p);
    }
    else
    {
        vstrbq_s32(out, acc);
    }
}

/* Adds padded taps to the accumulator */
__STATIC_FORCEINLINE int32x4_t dw_pad_taps(int32x4_t acc,
                                           const int8_t *ker,
                                           const int32_t taps,
                                           const dw_direct_args *a,
                                           const mve_pred16_t p,
                                           const bool tail)
{
    for (int32_t i = 0; i < taps; i++)
    {
        acc = vmlaq_n_s32(acc, dw_load(ker, p, tail), a->pad_val);
        ker += a->ch;
    }
    return acc;
}

/* Four interior pixels, one channel block */
__STATIC_FORCEINLINE void dw_px4_block(const dw_direct_args *a,
                                       const dw_direct_group *g,
                                       const int32_t c,
                                       int8_t *out,
                                       const mve_pred16_t p,
                                       const bool tail)
{
    const int32_t ch = a->ch;
    const int32_t row_taps = a->kernel_x;
    const int32_t tap_step = a->dilation_x * ch;

    /* Padded rows are shared by all pixels */
    int32x4_t acc_0 = dw_load_w(a->weight_sum + c, p, tail);
    if (g->pad_lo + g->pad_hi > 0)
    {
        acc_0 = dw_pad_taps(acc_0, a->kernel + c, g->pad_lo, a, p, tail);
        acc_0 = dw_pad_taps(acc_0, a->kernel + c + (a->kernel_y * row_taps - g->pad_hi) * ch, g->pad_hi, a, p, tail);
    }
    int32x4_t acc_1 = acc_0;
    int32x4_t acc_2 = acc_0;
    int32x4_t acc_3 = acc_0;

    const int8_t *ker = g->ker + c;
    const int8_t *p0 = g->in + c;
    const int8_t *p1 = p0 + g->o1;
    const int8_t *p2 = p0 + g->o2;
    const int8_t *p3 = p0 + g->o3;
    for (int32_t r = 0; r < g->rows; r++)
    {
        for (int32_t kx = 0; kx < row_taps; kx++)
        {
            const int32x4_t w = dw_load(ker, p, tail);
            ker += ch;
            acc_0 += vmulq_s32(dw_load(p0, p, tail), w);
            p0 += tap_step;
            acc_1 += vmulq_s32(dw_load(p1, p, tail), w);
            p1 += tap_step;
            acc_2 += vmulq_s32(dw_load(p2, p, tail), w);
            p2 += tap_step;
            acc_3 += vmulq_s32(dw_load(p3, p, tail), w);
            p3 += tap_step;
        }
        p0 += g->row_adj;
        p1 += g->row_adj;
        p2 += g->row_adj;
        p3 += g->row_adj;
    }

    const int32x4_t mult = dw_load_w(a->mult + c, p, tail);
    const int32x4_t shift = dw_load_w(a->shift + c, p, tail);
    out += c;
    dw_store(out, acc_0, mult, shift, a, p, tail);
    if (g->num_px > 1)
    {
        dw_store(out + ch, acc_1, mult, shift, a, p, tail);
    }
    if (g->num_px > 2)
    {
        dw_store(out + 2 * ch, acc_2, mult, shift, a, p, tail);
    }
    if (g->num_px > 3)
    {
        dw_store(out + 3 * ch, acc_3, mult, shift, a, p, tail);
    }
}

/* Up to four interior pixels */
static void dw_px4(const dw_direct_args *args,
                   const int32_t base_y,
                   const int32_t base_x,
                   const int32_t ky_lo,
                   const int32_t ky_hi,
                   const int32_t num_px,
                   int8_t *out)
{
    /* Local copy, so stores cannot alias it */
    const dw_direct_args a = *args;
    const int32_t ch = a.ch;
    const int32_t px_step = a.stride_x * ch;
    const int32_t in_row = a.input_x * ch;
    dw_direct_group g;
    g.rows = ky_hi - ky_lo;
    g.in = g.rows > 0 ? a.input + (base_y + ky_lo) * in_row + base_x * ch : a.input;
    g.ker = a.kernel + ky_lo * a.kernel_x * ch;
    g.row_adj = in_row - a.kernel_x * a.dilation_x * ch;
    g.pad_lo = ky_lo * a.kernel_x;
    g.pad_hi = (a.kernel_y - ky_hi) * a.kernel_x;
    g.o1 = ARM_NN_MIN(1, num_px - 1) * px_step;
    g.o2 = ARM_NN_MIN(2, num_px - 1) * px_step;
    g.o3 = ARM_NN_MIN(3, num_px - 1) * px_step;
    g.num_px = num_px;

    const int32_t full_ch = ch & ~0x3;
    int32_t c = 0;
    for (; c < full_ch; c += 4)
    {
        dw_px4_block(&a, &g, c, out, 0xFFFF, false);
    }
    if (c < ch)
    {
        dw_px4_block(&a, &g, c, out, vctp32q((uint32_t)(ch - c)), true);
    }
}

/* One border pixel, one channel block */
__STATIC_FORCEINLINE void dw_px1_block(const dw_direct_args *a,
                                       const int32_t c,
                                       const int32_t base_y,
                                       const int32_t base_x,
                                       int8_t *out,
                                       const mve_pred16_t p,
                                       const bool tail)
{
    const int32_t ch = a->ch;
    const int8_t *ker = a->kernel + c;
    int32x4_t acc = dw_load_w(a->weight_sum + c, p, tail);

    for (int32_t ky = 0; ky < a->kernel_y; ky++)
    {
        const int32_t iy = base_y + ky;
        const bool row_ok = iy >= 0 && iy < a->input_y;
        for (int32_t kx = 0, ix = base_x; kx < a->kernel_x; kx++, ix += a->dilation_x)
        {
            const int32x4_t w = dw_load(ker, p, tail);
            ker += ch;
            if (row_ok && ix >= 0 && ix < a->input_x)
            {
                acc += vmulq_s32(dw_load(a->input + (iy * a->input_x + ix) * ch + c, p, tail), w);
            }
            else
            {
                acc = vmlaq_n_s32(acc, w, a->pad_val);
            }
        }
    }

    dw_store(out + c, acc, dw_load_w(a->mult + c, p, tail), dw_load_w(a->shift + c, p, tail), a, p, tail);
}

/* One pixel with padded columns */
static void dw_px1(const dw_direct_args *args, const int32_t base_y, const int32_t base_x, int8_t *out)
{
    const dw_direct_args a = *args;
    const int32_t full_ch = a.ch & ~0x3;
    int32_t c = 0;
    for (; c < full_ch; c += 4)
    {
        dw_px1_block(&a, c, base_y, base_x, out, 0xFFFF, false);
    }
    if (c < a.ch)
    {
        dw_px1_block(&a, c, base_y, base_x, out, vctp32q((uint32_t)(a.ch - c)), true);
    }
}

#endif

/*
 * s8 depthwise convolution that reads the input in place.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_depthwise_conv_s8_direct(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                    const cmsis_nn_per_channel_quant_params *quant_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const int8_t *input,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const int8_t *kernel,
                                                    const int32_t *weight_sum,
                                                    const cmsis_nn_dims *output_dims,
                                                    int8_t *output)
{
#if defined(ARM_MATH_MVEI)
    const int32_t ch = input_dims->c;
    const int32_t input_x = input_dims->w;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_x = dw_conv_params->padding.w;
    const int32_t stride_x = dw_conv_params->stride.w;
    const int32_t dilation_x = dw_conv_params->dilation.w;
    const int32_t output_x = output_dims->w;

    if (ch > DW_DIRECT_MAX_CH || ch < 1 || output_dims->c != ch || dw_conv_params->dilation.h != 1 || dilation_x < 1 ||
        stride_x < 1 || dw_conv_params->stride.h < 1 || weight_sum == NULL)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    /* Outputs whose taps all lie inside the row */
    const int32_t last_x = input_x - 1 - (kernel_x - 1) * dilation_x + pad_x;
    const int32_t ox_lo = ARM_NN_MIN(ARM_NN_MAX(0, (pad_x + stride_x - 1) / stride_x), output_x);
    const int32_t ox_hi = ARM_NN_MAX(ox_lo, last_x < 0 ? 0 : ARM_NN_MIN(last_x / stride_x + 1, output_x));

    /* Border pixels are slow here; im2col wins */
    if (ox_hi - ox_lo < 4)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    const int32_t input_y = input_dims->h;
    const int32_t kernel_y = filter_dims->h;
    const int32_t pad_y = dw_conv_params->padding.h;
    const int32_t stride_y = dw_conv_params->stride.h;
    const int32_t output_y = output_dims->h;
    const dw_direct_args args = {input,
                                 kernel,
                                 weight_sum,
                                 quant_params->multiplier,
                                 quant_params->shift,
                                 ch,
                                 input_x,
                                 input_y,
                                 kernel_x,
                                 kernel_y,
                                 stride_x,
                                 dilation_x,
                                 (int8_t)-dw_conv_params->input_offset,
                                 dw_conv_params->output_offset,
                                 dw_conv_params->activation.min,
                                 dw_conv_params->activation.max};

    for (int32_t i_out_y = 0, base_y = -pad_y; i_out_y < output_y; i_out_y++, base_y += stride_y)
    {
        const int32_t ky_lo = ARM_NN_MIN(ARM_NN_MAX(0, -base_y), kernel_y);
        const int32_t ky_hi = ARM_NN_MAX(ky_lo, ARM_NN_MIN(kernel_y, input_y - base_y));
        int8_t *out_row = output + i_out_y * output_x * ch;

        for (int32_t i_out_x = 0; i_out_x < ox_lo; i_out_x++)
        {
            dw_px1(&args, base_y, i_out_x * stride_x - pad_x, out_row + i_out_x * ch);
        }
        for (int32_t i_out_x = ox_lo; i_out_x < ox_hi; i_out_x += 4)
        {
            dw_px4(&args,
                   base_y,
                   i_out_x * stride_x - pad_x,
                   ky_lo,
                   ky_hi,
                   ARM_NN_MIN(4, ox_hi - i_out_x),
                   out_row + i_out_x * ch);
        }
        for (int32_t i_out_x = ox_hi; i_out_x < output_x; i_out_x++)
        {
            dw_px1(&args, base_y, i_out_x * stride_x - pad_x, out_row + i_out_x * ch);
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
#else
    (void)dw_conv_params;
    (void)quant_params;
    (void)input_dims;
    (void)input;
    (void)filter_dims;
    (void)kernel;
    (void)weight_sum;
    (void)output_dims;
    (void)output;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}

/**
 * @} end of Doxygen group
 */
