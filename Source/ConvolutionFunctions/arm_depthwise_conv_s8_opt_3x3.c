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
 * Title:        arm_depthwise_conv_s8_opt_3x3.c
 * Description:  Direct-entry s8 3x3 depthwise convolution (channel multiplier 1) that reads the input in place.
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

#if defined(ARM_MATH_MVEI)

/* A block is three output rows at one output column; each 4-channel group loads its nine weight vectors once for the
   three rows and reads the input rows in place. Parameters are repacked per 4-channel group as [9 taps x 4 weights][ws]
   (52 bytes), in three variants: all taps, left edge and right edge. An edge variant shifts the window one column
   inward (so every load stays inside the row), sets the weights of the padded column to zero and cancels that column's
   input_offset term in ws. Rows outside the input read a pad row of -input_offset, which the ws term cancels as in the
   im2col path of arm_depthwise_conv_s8_opt. */
    #define DW3_GRP (52)
    #define DW3_NB (3)
    #define DW3_MAX_ROWS ((DW3_NB - 1) * 2 + 3)
/* Channels packed and processed per pass; bounds the scratch use for wide layers. */
    #define DW3_CB (64)

/* Pins the accumulators and orders memory at each tap, so the compiler neither hoists the next taps' loads nor sinks
   their multiplies; either one keeps more vectors live than the eight Q registers hold. */
    #define DW3_PIN() __asm volatile("" : "+w"(a0), "+w"(a1), "+w"(a2)::"memory")

typedef struct
{
    const int8_t *pk_full;
    const int8_t *pk_left;
    const int8_t *pk_right;
    const int32_t *qm;
    const int8_t *input;
    const int8_t *pad_row;
    int8_t *output;
    int32_t n_ch;
    int32_t ch;
    int32_t input_x;
    int32_t input_y;
    int32_t output_x;
    int32_t output_y;
    int32_t sx;
    int32_t sy;
    int32_t pad_x;
    int32_t pad_y;
    int32_t out_offset;
    int32_t act_min;
    int32_t act_max;
} dw3_layer;

typedef void (*dw3_run_fn)(const dw3_layer *L);

__STATIC_FORCEINLINE int32x4_t dw3_requant(const int32x4_t acc,
                                           const int32x4_t mult,
                                           const int32x4_t shift,
                                           const int32_t out_offset,
                                           const int32_t act_min,
                                           const int32_t act_max)
{
    int32x4_t v = arm_requantize_mve_32x4(acc, mult, shift);
    v = vaddq_n_s32(v, out_offset);
    v = vmaxq_s32(v, vdupq_n_s32(act_min));
    return vminq_s32(v, vdupq_n_s32(act_max));
}

/* nb (3 or 1) output rows at one output column, any channel count. rows[j * sy + ky] is the centre-column pointer of
   the input row that output row j reads for kernel row ky. The row pointers walk the three kernel columns (+ch, +ch,
   then on to the next group), so no load needs a column offset register. */
__STATIC_FORCEINLINE void dw3_block_any(const dw3_layer *L,
                                        const int8_t *pk,
                                        const int8_t *const *rows,
                                        int8_t *out,
                                        const int32_t out_stride,
                                        const int32_t sy,
                                        const int32_t nb)
{
    /* Locals, not struct reads: the pin's memory clobber would reload every field on each tap. */
    const int32_t *qm = L->qm;
    const int32_t n_ch = L->n_ch;
    const int32_t ch = L->ch;
    const int32_t out_offset = L->out_offset;
    const int32_t act_min = L->act_min;
    const int32_t act_max = L->act_max;
    const int32_t n_rows = (nb - 1) * sy + 3;
    const int32_t back = 4 - 2 * ch;
    const int8_t *r[DW3_MAX_ROWS];
    for (int32_t k = 0; k < n_rows; k++)
    {
        r[k] = rows[k] - ch;
    }
    for (int32_t g = 0; g < n_ch; g += 4)
    {
        const int32x4_t base = vldrwq_s32((const int32_t *)(const void *)(pk + 36));
        int32x4_t a0 = base;
        int32x4_t a1 = base;
        int32x4_t a2 = base;
        for (int32_t kx = 0; kx < 3; kx++)
        {
            for (int32_t ky = 0; ky < 3; ky++)
            {
                const int32x4_t wv = vldrbq_s32(pk + 4 * (ky * 3 + kx));
                a0 = vaddq_s32(a0, vmulq_s32(vldrbq_s32(r[ky]), wv));
                if (nb > 1)
                {
                    a1 = vaddq_s32(a1, vmulq_s32(vldrbq_s32(r[1 * sy + ky]), wv));
                    a2 = vaddq_s32(a2, vmulq_s32(vldrbq_s32(r[2 * sy + ky]), wv));
                }
                DW3_PIN();
            }
            const int32_t step = kx < 2 ? ch : back;
            for (int32_t k = 0; k < n_rows; k++)
            {
                /* Keeps the pointers as three in-place steps instead of three separately formed addresses. */
                r[k] += step;
                __asm volatile("" : "+r"(r[k]));
            }
        }
        const int32x4_t mult = vldrwq_s32(qm);
        const int32x4_t shift = vldrwq_s32(qm + 4);
        vstrbq_s32(out, dw3_requant(a0, mult, shift, out_offset, act_min, act_max));
        if (nb > 1)
        {
            vstrbq_s32(out + out_stride, dw3_requant(a1, mult, shift, out_offset, act_min, act_max));
            vstrbq_s32(out + 2 * out_stride, dw3_requant(a2, mult, shift, out_offset, act_min, act_max));
        }
        pk += DW3_GRP;
        qm += 8;
        out += 4;
    }
}

/* Three output rows at one output column for a compile-time channel count, so every load uses an immediate column
   offset. skip names the zero-weight tap column of an edge variant (1: column 2 of the left variant, 2: column 0 of
   the right variant), which is left out; 0 computes all nine taps. */
__STATIC_FORCEINLINE void dw3_block_fixed(const dw3_layer *L,
                                          const int8_t *pk,
                                          const int8_t *const *rows,
                                          int8_t *out,
                                          const int32_t out_stride,
                                          const int32_t ch,
                                          const int32_t sy,
                                          const int32_t skip)
{
    const int32_t *qm = L->qm;
    const int32_t n_ch = L->n_ch;
    const int32_t out_offset = L->out_offset;
    const int32_t act_min = L->act_min;
    const int32_t act_max = L->act_max;
    const int32_t n_rows = (DW3_NB - 1) * sy + 3;
    const int8_t *r[DW3_MAX_ROWS];
    for (int32_t k = 0; k < n_rows; k++)
    {
        r[k] = rows[k];
    }
    for (int32_t g = 0; g < n_ch; g += 4)
    {
        const int32x4_t base = vldrwq_s32((const int32_t *)(const void *)(pk + 36));
        int32x4_t a0 = base;
        int32x4_t a1 = base;
        int32x4_t a2 = base;
        for (int32_t ky = 0; ky < 3; ky++)
        {
            for (int32_t kx = 0; kx < 3; kx++)
            {
                if ((skip == 1 && kx == 2) || (skip == 2 && kx == 0))
                {
                    continue;
                }
                const int32x4_t wv = vldrbq_s32(pk + 4 * (ky * 3 + kx));
                a0 = vaddq_s32(a0, vmulq_s32(vldrbq_s32(r[ky] + (kx - 1) * ch), wv));
                a1 = vaddq_s32(a1, vmulq_s32(vldrbq_s32(r[1 * sy + ky] + (kx - 1) * ch), wv));
                a2 = vaddq_s32(a2, vmulq_s32(vldrbq_s32(r[2 * sy + ky] + (kx - 1) * ch), wv));
                DW3_PIN();
            }
        }
        const int32x4_t mult = vldrwq_s32(qm);
        const int32x4_t shift = vldrwq_s32(qm + 4);
        vstrbq_s32(out, dw3_requant(a0, mult, shift, out_offset, act_min, act_max));
        vstrbq_s32(out + out_stride, dw3_requant(a1, mult, shift, out_offset, act_min, act_max));
        vstrbq_s32(out + 2 * out_stride, dw3_requant(a2, mult, shift, out_offset, act_min, act_max));
        pk += DW3_GRP;
        qm += 8;
        out += 4;
        for (int32_t k = 0; k < n_rows; k++)
        {
            r[k] += 4;
        }
    }
}

__STATIC_FORCEINLINE int32_t dw3_col_mask(const int32_t ix0, const int32_t input_x)
{
    int32_t m = 0;
    for (int32_t kx = 0; kx < 3; kx++)
    {
        if (ix0 + kx >= 0 && ix0 + kx < input_x)
        {
            m |= 1 << kx;
        }
    }
    return m;
}

__STATIC_FORCEINLINE void dw3_rows(const dw3_layer *L,
                                   const int8_t **rows,
                                   const int8_t *in_c,
                                   const int8_t *pad_c,
                                   const int32_t row_bytes,
                                   const int32_t iy0,
                                   const int32_t n)
{
    for (int32_t k = 0; k < n; k++)
    {
        const int32_t iy = iy0 + k;
        rows[k] = (iy >= 0 && iy < L->input_y) ? in_c + iy * row_bytes : pad_c;
    }
}

/* One output row (output_y % 3 tail), shared by every instance. */
__attribute__((noinline)) static void
dw3_tail(const dw3_layer *L, const int8_t *pk, const int8_t *const *rows, int8_t *out)
{
    dw3_block_any(L, pk, rows, out, 0, 1, 1);
}

/* One output column at vertical stride sy. fixed_ch != 0 selects the immediate-offset block for that channel count,
   with skip as in dw3_block_fixed; fixed_ch == 0 runs the runtime channel count. */
__STATIC_FORCEINLINE void dw3_column(const dw3_layer *L,
                                     const int8_t *pkx,
                                     const int32_t centre,
                                     const int32_t x,
                                     const int32_t sy,
                                     const int32_t fixed_ch,
                                     const int32_t skip)
{
    const int32_t ch = fixed_ch ? fixed_ch : L->ch;
    const int32_t row_bytes = L->input_x * ch;
    const int32_t out_stride = L->output_x * ch;
    const int8_t *in_c = L->input + centre * ch;
    const int8_t *pad_c = L->pad_row + centre * ch;
    int8_t *out = L->output + x * ch;
    const int8_t *rows[DW3_MAX_ROWS];
    int32_t y0 = 0;
    for (; y0 + DW3_NB <= L->output_y; y0 += DW3_NB)
    {
        int8_t *o = out + y0 * out_stride;
        dw3_rows(L, rows, in_c, pad_c, row_bytes, y0 * sy - L->pad_y, (DW3_NB - 1) * sy + 3);
        if (fixed_ch)
        {
            dw3_block_fixed(L, pkx, rows, o, out_stride, fixed_ch, sy, skip);
        }
        else
        {
            dw3_block_any(L, pkx, rows, o, out_stride, sy, DW3_NB);
        }
    }
    for (; y0 < L->output_y; y0++)
    {
        dw3_rows(L, rows, in_c, pad_c, row_bytes, y0 * sy - L->pad_y, 3);
        dw3_tail(L, pkx, rows, out + y0 * out_stride);
    }
}

/* Whole channel pass. edge_spec builds separate edge-column code that skips the zero-weight taps. */
__STATIC_FORCEINLINE void dw3_run(const dw3_layer *L, const int32_t sy, const int32_t fixed_ch, const int32_t edge_spec)
{
    for (int32_t x = 0; x < L->output_x; x++)
    {
        const int32_t ix0 = x * L->sx - L->pad_x;
        const int32_t m = dw3_col_mask(ix0, L->input_x);
        if (edge_spec && m == 6)
        {
            dw3_column(L, L->pk_left, ix0 + 2, x, sy, fixed_ch, 1);
        }
        else if (edge_spec && m == 3)
        {
            dw3_column(L, L->pk_right, ix0, x, sy, fixed_ch, 2);
        }
        else
        {
            /* An edge column reads the variant whose window is shifted one column inward. */
            const int8_t *pkx = m == 7 ? L->pk_full : (m == 6 ? L->pk_left : L->pk_right);
            const int32_t centre = m == 7 ? ix0 + 1 : (m == 6 ? ix0 + 2 : ix0);
            dw3_column(L, pkx, centre, x, sy, fixed_ch, 0);
        }
    }
}

/* Separate code per vertical stride keeps every row pointer in a register. */
static void dw3_run_any(const dw3_layer *L)
{
    if (L->sy == 1)
    {
        dw3_run(L, 1, 0, 0);
    }
    else
    {
        dw3_run(L, 2, 0, 0);
    }
}

static void dw3_run_c64_s1(const dw3_layer *L) { dw3_run(L, 1, 64, 1); }

/* Scratch bytes: three parameter variants and the multiplier/shift pairs of one channel pass, and one pad row. */
static int64_t dw3_scratch_bytes(const cmsis_nn_dims *input_dims)
{
    return (DW3_CB / 4) * (3 * DW3_GRP + 32) + (int64_t)input_dims->w * input_dims->c + 16;
}

/* Gate, parameter packing and channel passes shared by every entry; run computes one pass. */
__attribute__((noinline)) static arm_cmsis_nn_status dw3_s8(const cmsis_nn_context *ctx,
                                                            const cmsis_nn_context *weight_sum_ctx,
                                                            const cmsis_nn_dw_conv_params *p,
                                                            const cmsis_nn_per_channel_quant_params *q,
                                                            const cmsis_nn_dims *input_dims,
                                                            const int8_t *input,
                                                            const cmsis_nn_dims *filter_dims,
                                                            const int8_t *kernel,
                                                            const cmsis_nn_dims *output_dims,
                                                            int8_t *output,
                                                            const dw3_run_fn run)
{
    const int32_t ch = input_dims->c;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t sx = p->stride.w;
    const int32_t sy = p->stride.h;
    const int32_t pad_x = p->padding.w;
    const int32_t pad_y = p->padding.h;

    if (filter_dims->w != 3 || filter_dims->h != 3 || p->dilation.w != 1 || p->dilation.h != 1 || input_dims->n != 1 ||
        ch != output_dims->c || ch < 16 || ch > 2048 || (ch & 3) != 0 || sx < 1 || sx > 2 || sy < 1 || sy > 2 ||
        pad_x < 0 || pad_x > 1 || pad_y < 0 || pad_y > 1 || input_x < 3 || input_y < 1 || output_x < 1 ||
        output_y < DW3_NB || input_x > 4096 || input_y > 4096 || output_x > 4096 || output_y > 4096 ||
        output_x * output_y < 16 || (int64_t)input_x * input_y * ch > INT32_MAX ||
        (int64_t)output_x * output_y * ch > INT32_MAX || ctx == NULL || ctx->buf == NULL || weight_sum_ctx == NULL ||
        weight_sum_ctx->buf == NULL || dw3_scratch_bytes(input_dims) > ctx->size ||
        (output_x - 1) * sx - pad_x + 1 >= input_x)
    {
        /* With pad_x <= 1 and input_x >= 3, a last window centre inside the row makes every column's tap mask full,
           left edge or right edge. */
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    /* 16-byte aligned: [full | left | right] parameter variants and multiplier/shift pairs of one pass, pad row. */
    int8_t *pk = (int8_t *)(((uintptr_t)ctx->buf + 15) & ~(uintptr_t)15);
    const int32_t cb_groups = DW3_CB / 4;
    int8_t *pk_full = pk;
    int8_t *pk_left = pk + cb_groups * DW3_GRP;
    int8_t *pk_right = pk + 2 * cb_groups * DW3_GRP;
    int32_t *qm = (int32_t *)(void *)(pk + 3 * cb_groups * DW3_GRP);
    int8_t *pad_row = (int8_t *)(qm + 8 * cb_groups);
    const int32_t pad_val = -p->input_offset;
    const int32x4_t zero = vdupq_n_s32(0);
    arm_memset_s8(pad_row, (int8_t)pad_val, (uint32_t)(input_x * ch));

    for (int32_t c0 = 0; c0 < ch; c0 += DW3_CB)
    {
        const int32_t n_ch = ARM_NN_MIN(DW3_CB, ch - c0);
        const int32_t *ws = (const int32_t *)weight_sum_ctx->buf + c0;
        for (int32_t g = 0; g < n_ch / 4; g++)
        {
            const int8_t *kw = kernel + c0 + g * 4;
            int8_t *f = pk_full + g * DW3_GRP;
            int8_t *l = pk_left + g * DW3_GRP;
            int8_t *rt = pk_right + g * DW3_GRP;
            int32x4_t col0 = zero;
            int32x4_t col2 = zero;
            for (int32_t ky = 0; ky < 3; ky++)
            {
                const int32x4_t w0 = vldrbq_s32(kw + (ky * 3 + 0) * ch);
                const int32x4_t w1 = vldrbq_s32(kw + (ky * 3 + 1) * ch);
                const int32x4_t w2 = vldrbq_s32(kw + (ky * 3 + 2) * ch);
                col0 = vaddq_s32(col0, w0);
                col2 = vaddq_s32(col2, w2);
                vstrbq_s32(f + (ky * 3 + 0) * 4, w0);
                vstrbq_s32(f + (ky * 3 + 1) * 4, w1);
                vstrbq_s32(f + (ky * 3 + 2) * 4, w2);
                vstrbq_s32(l + (ky * 3 + 0) * 4, w1);
                vstrbq_s32(l + (ky * 3 + 1) * 4, w2);
                vstrbq_s32(l + (ky * 3 + 2) * 4, zero);
                vstrbq_s32(rt + (ky * 3 + 0) * 4, zero);
                vstrbq_s32(rt + (ky * 3 + 1) * 4, w0);
                vstrbq_s32(rt + (ky * 3 + 2) * 4, w1);
            }
            const int32x4_t wsv = vldrwq_s32(ws + g * 4);
            vstrwq_s32((int32_t *)(void *)(f + 36), wsv);
            vstrwq_s32((int32_t *)(void *)(l + 36), vmlaq_n_s32(wsv, col0, pad_val));
            vstrwq_s32((int32_t *)(void *)(rt + 36), vmlaq_n_s32(wsv, col2, pad_val));
            vstrwq_s32(qm + g * 8, vldrwq_s32(q->multiplier + c0 + g * 4));
            vstrwq_s32(qm + g * 8 + 4, vldrwq_s32(q->shift + c0 + g * 4));
        }

        const dw3_layer L = {pk_full,
                             pk_left,
                             pk_right,
                             qm,
                             input + c0,
                             pad_row,
                             output + c0,
                             n_ch,
                             ch,
                             input_x,
                             input_y,
                             output_x,
                             output_y,
                             sx,
                             sy,
                             pad_x,
                             pad_y,
                             p->output_offset,
                             p->activation.min,
                             p->activation.max};
        run(&L);
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif

/*
 * Direct-entry s8 3x3 depthwise convolution, any channel count in the gate.
 *
 * Refer to header file for details.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_s8_opt_3x3(const cmsis_nn_context *ctx,
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
    (void)bias_dims;
    (void)bias;
#if defined(ARM_MATH_MVEI)
    return dw3_s8(ctx,
                  weight_sum_ctx,
                  dw_conv_params,
                  quant_params,
                  input_dims,
                  input,
                  filter_dims,
                  kernel,
                  output_dims,
                  output,
                  dw3_run_any);
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

/*
 * Direct-entry s8 3x3 depthwise convolution for 64 channels and a vertical stride of 1.
 *
 * Refer to header file for details.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_s8_opt_3x3_c64_s1(const cmsis_nn_context *ctx,
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
    (void)bias_dims;
    (void)bias;
#if defined(ARM_MATH_MVEI)
    if (input_dims->c != 64 || dw_conv_params->stride.h != 1)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    return dw3_s8(ctx,
                  weight_sum_ctx,
                  dw_conv_params,
                  quant_params,
                  input_dims,
                  input,
                  filter_dims,
                  kernel,
                  output_dims,
                  output,
                  dw3_run_c64_s1);
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
 * @} end of NNConv group
 */
