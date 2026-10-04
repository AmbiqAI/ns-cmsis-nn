/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_depthwise_conv_f16.c
 * Description:  Convolution: depthwise (float16)
 *
 * $Date:        23 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nn_types.h"

#if ARM_NN_ENABLE_F16

    #include "Internal/arm_depthwise_conv_opt_common.h"
    #include "Internal/arm_depthwise_conv_opt_f16.h"
    #include "Internal/arm_nn_activation_flt.h"
    #include "arm_nnfunctions.h"
    #include "arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Direct ch_mult == 1 NHWC depthwise kernel (#448). Lanes are channels: every tap is one contiguous vector
 * load of the input pixel and one of the [kh][kw][C] filter, accumulated in registers from the bias in
 * (ky, kx) order, one FMA per tap -- the tap order of the routes this replaced. The MVE leg accumulates in
 * float16 lanes (#446); the scalar leg in float32 with one rounding at the store (#449). Padding is the tap
 * window (no scratch, nothing read out of range); stride and dilation are generic. The channel tail is
 * one straight-line predicated vector; the tap loops carry no vctp.
 */
typedef struct
{
    size_t in_row_step; /* dilation_y * input_x * ch, elements */
    size_t in_tap_step; /* dilation_x * ch */
    size_t in_px_step;  /* stride_x * ch */
    size_t w_row_step;  /* kernel_x * ch */
    size_t ch;          /* channels (also the weight tap step) */
    int32_t taps_y;     /* in-range tap rows for this output row */
    int32_t taps_x;     /* in-range tap columns for every pixel of this run */
    float16_t act_min;
    float16_t act_max;
} arm_depthwise_direct_run_f16;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
/* Bias substitute for a NULL bias: two vectors of zeros (a two-vector block reads bias_c + 8). */
static const float16_t arm_depthwise_direct_zero_bias_f16[16] =
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

/*
 * `npx` consecutive output pixels of one channel block. nvec == 2: two vectors per pixel, npx <= 2 (acc0 / acc1 are
 * pixel 0, acc2 / acc3 pixel 1). nvec == 1: one vector per pixel, npx <= 4 (acc0..acc3 are pixels 0..3). `pred`
 * predicates the last vector of every pixel with `p`. Every weight load is shared by the pixels; the accumulators
 * and the tap temporaries are the only live Q registers in the tap loop.
 */
__STATIC_FORCEINLINE float16x8_t arm_depthwise_direct_load_f16(const float16_t *src, const int32_t pred, mve_pred16_t p)
{
    return pred ? vld1q_z(src, p) : vld1q(src);
}

__STATIC_FORCEINLINE void
arm_depthwise_direct_store_f16(float16_t *dst, float16x8_t v, const int32_t pred, mve_pred16_t p)
{
    if (pred)
    {
        vst1q_p(dst, v, p);
    }
    else
    {
        vst1q(dst, v);
    }
}

__STATIC_FORCEINLINE void arm_depthwise_direct_taps_mve_f16(const float16_t *__RESTRICT in_tap,
                                                            const float16_t *__RESTRICT w_tap,
                                                            const float16_t *__RESTRICT bias_c,
                                                            float16_t *__RESTRICT out_c,
                                                            const arm_depthwise_direct_run_f16 *__RESTRICT run,
                                                            const int32_t npx,
                                                            const int32_t nvec,
                                                            const int32_t pred,
                                                            mve_pred16_t p,
                                                            const int32_t block)
{
    const size_t in_tap_step = run->in_tap_step;
    const size_t in_px_step = run->in_px_step;
    const size_t w_tap_step = run->ch;
    const int32_t taps_x = run->taps_x;
    float16x8_t acc0;
    float16x8_t acc1;
    float16x8_t acc2;
    float16x8_t acc3;

    if (nvec == 2)
    {
        acc0 = vld1q(bias_c);
        acc1 = arm_depthwise_direct_load_f16(bias_c + 8, pred, p);
        acc2 = acc0;
        acc3 = acc1;
    }
    else
    {
        acc0 = arm_depthwise_direct_load_f16(bias_c, pred, p);
        acc1 = acc0;
        acc2 = acc0;
        acc3 = acc0;
    }

    /* Folding (#586): with more than `block` in-range taps, each float16 partial covers at most `block` taps (row by
     * row, column by column) before it is widened into per-lane float32 accumulators; one rounding at the end. */
    const bool fold = run->taps_y * taps_x > block;
    float32x4_t vsum[4][2];
    if (fold)
    {
        /* Set by the first fold; zeroed only so that no compiler sees a read before it. */
        for (int32_t i = 0; i < 4; ++i)
        {
            vsum[i][0] = vdupq_n_f32(0.0f);
            vsum[i][1] = vdupq_n_f32(0.0f);
        }
    }
    bool first = true;
    int32_t n_taps = 0;

    for (int32_t ky = 0; ky < run->taps_y; ++ky)
    {
        const float16_t *in0 = in_tap + (size_t)ky * run->in_row_step;
        const float16_t *in1 = in0 + in_px_step;
        const float16_t *in2 = in1 + in_px_step;
        const float16_t *in3 = in2 + in_px_step;
        const float16_t *w = w_tap + (size_t)ky * run->w_row_step;
        for (int32_t kx = 0; kx < taps_x; ++kx)
        {
            if (fold)
            {
                if (n_taps == block)
                {
                    arm_nn_f16_fold_lanes_f32(&vsum[0][0], &vsum[0][1], acc0, first);
                    arm_nn_f16_fold_lanes_f32(&vsum[1][0], &vsum[1][1], acc1, first);
                    arm_nn_f16_fold_lanes_f32(&vsum[2][0], &vsum[2][1], acc2, first);
                    arm_nn_f16_fold_lanes_f32(&vsum[3][0], &vsum[3][1], acc3, first);
                    first = false;
                    acc0 = vdupq_n_f16((float16_t)0.0f);
                    acc1 = acc0;
                    acc2 = acc0;
                    acc3 = acc0;
                    n_taps = 0;
                }
                ++n_taps;
            }
            if (nvec == 2)
            {
                const float16x8_t w0 = vld1q(w);
                const float16x8_t w1 = arm_depthwise_direct_load_f16(w + 8, pred, p);
                acc0 = vfmaq(acc0, vld1q(in0), w0);
                acc1 = vfmaq(acc1, arm_depthwise_direct_load_f16(in0 + 8, pred, p), w1);
                if (npx >= 2)
                {
                    acc2 = vfmaq(acc2, vld1q(in1), w0);
                    acc3 = vfmaq(acc3, arm_depthwise_direct_load_f16(in1 + 8, pred, p), w1);
                }
            }
            else
            {
                const float16x8_t w0 = arm_depthwise_direct_load_f16(w, pred, p);
                acc0 = vfmaq(acc0, arm_depthwise_direct_load_f16(in0, pred, p), w0);
                if (npx >= 2)
                {
                    acc1 = vfmaq(acc1, arm_depthwise_direct_load_f16(in1, pred, p), w0);
                }
                if (npx >= 3)
                {
                    acc2 = vfmaq(acc2, arm_depthwise_direct_load_f16(in2, pred, p), w0);
                }
                if (npx >= 4)
                {
                    acc3 = vfmaq(acc3, arm_depthwise_direct_load_f16(in3, pred, p), w0);
                }
            }
            in0 += in_tap_step;
            in1 += in_tap_step;
            in2 += in_tap_step;
            in3 += in_tap_step;
            w += w_tap_step;
        }
    }

    if (fold && !first)
    {
        arm_nn_f16_fold_lanes_f32(&vsum[0][0], &vsum[0][1], acc0, false);
        arm_nn_f16_fold_lanes_f32(&vsum[1][0], &vsum[1][1], acc1, false);
        arm_nn_f16_fold_lanes_f32(&vsum[2][0], &vsum[2][1], acc2, false);
        arm_nn_f16_fold_lanes_f32(&vsum[3][0], &vsum[3][1], acc3, false);
        acc0 = arm_nn_f16_narrow_lanes_f32(vsum[0][0], vsum[0][1]);
        acc1 = arm_nn_f16_narrow_lanes_f32(vsum[1][0], vsum[1][1]);
        acc2 = arm_nn_f16_narrow_lanes_f32(vsum[2][0], vsum[2][1]);
        acc3 = arm_nn_f16_narrow_lanes_f32(vsum[3][0], vsum[3][1]);
    }

    /* Same clamp as arm_nn_vector_clamp_f16 (NaN resolves to a bound on MVE, as before). */
    const float16x8_t vmin = vdupq_n_f16(run->act_min);
    const float16x8_t vmax = vdupq_n_f16(run->act_max);
    acc0 = arm_nn_clamp_mve_f16(acc0, vmin, vmax);
    acc1 = arm_nn_clamp_mve_f16(acc1, vmin, vmax);
    acc2 = arm_nn_clamp_mve_f16(acc2, vmin, vmax);
    acc3 = arm_nn_clamp_mve_f16(acc3, vmin, vmax);
    if (nvec == 2)
    {
        vst1q(out_c, acc0);
        arm_depthwise_direct_store_f16(out_c + 8, acc1, pred, p);
        if (npx >= 2)
        {
            vst1q(out_c + run->ch, acc2);
            arm_depthwise_direct_store_f16(out_c + run->ch + 8, acc3, pred, p);
        }
    }
    else
    {
        arm_depthwise_direct_store_f16(out_c, acc0, pred, p);
        if (npx >= 2)
        {
            arm_depthwise_direct_store_f16(out_c + run->ch, acc1, pred, p);
        }
        if (npx >= 3)
        {
            arm_depthwise_direct_store_f16(out_c + 2 * run->ch, acc2, pred, p);
        }
        if (npx >= 4)
        {
            arm_depthwise_direct_store_f16(out_c + 3 * run->ch, acc3, pred, p);
        }
    }
}

/* One channel block over `n_px` consecutive output pixels sharing a tap window: quads (one vector per pixel) or
 * pairs (two), then the remainder. */
__STATIC_FORCEINLINE void arm_depthwise_direct_block_mve_f16(const float16_t *in_tap,
                                                             const float16_t *w_tap,
                                                             const float16_t *bias_c,
                                                             float16_t *out_c,
                                                             int32_t n_px,
                                                             const arm_depthwise_direct_run_f16 *run,
                                                             const int32_t nvec,
                                                             const int32_t pred,
                                                             mve_pred16_t p,
                                                             const int32_t block)
{
    const size_t in_px_step = run->in_px_step;
    const size_t out_px_step = run->ch;
    int32_t px = 0;
    if (nvec == 1)
    {
        for (; px + 4 <= n_px; px += 4)
        {
            arm_depthwise_direct_taps_mve_f16(in_tap, w_tap, bias_c, out_c, run, 4, 1, pred, p, block);
            in_tap += 4U * in_px_step;
            out_c += 4U * out_px_step;
        }
    }
    for (; px + 2 <= n_px; px += 2)
    {
        arm_depthwise_direct_taps_mve_f16(in_tap, w_tap, bias_c, out_c, run, 2, nvec, pred, p, block);
        in_tap += 2U * in_px_step;
        out_c += 2U * out_px_step;
    }
    if (px < n_px)
    {
        arm_depthwise_direct_taps_mve_f16(in_tap, w_tap, bias_c, out_c, run, 1, nvec, pred, p, block);
    }
}

/* All channel blocks of `n_px` pixels sharing one tap window: pairs of full vectors, then one tail block. */
__STATIC_FORCEINLINE void arm_depthwise_direct_run_mve_f16_body(const float16_t *__RESTRICT in_tap,
                                                                const float16_t *__RESTRICT w_tap,
                                                                const float16_t *__RESTRICT bias,
                                                                float16_t *__RESTRICT out_px,
                                                                int32_t n_px,
                                                                const arm_depthwise_direct_run_f16 *__RESTRICT run,
                                                                const int32_t block)
{
    const int32_t ch = (int32_t)run->ch;
    const int32_t ch_pairs = ch & ~15;
    const int32_t ch_rem = ch - ch_pairs;
    const mve_pred16_t p_tail = vctp16q((uint32_t)(ch_rem & 7));
    const float16_t *bias_c = bias;
    const size_t bias_step = bias ? 16U : 0U;
    int32_t c = 0;

    if (!bias)
    {
        bias_c = arm_depthwise_direct_zero_bias_f16;
    }

    for (; c < ch_pairs; c += 16)
    {
        arm_depthwise_direct_block_mve_f16(in_tap + c, w_tap + c, bias_c, out_px + c, n_px, run, 2, 0, p_tail, block);
        bias_c += bias_step;
    }
    if (ch_rem > 8)
    {
        arm_depthwise_direct_block_mve_f16(in_tap + c, w_tap + c, bias_c, out_px + c, n_px, run, 2, 1, p_tail, block);
    }
    else if (ch_rem == 8)
    {
        arm_depthwise_direct_block_mve_f16(in_tap + c, w_tap + c, bias_c, out_px + c, n_px, run, 1, 0, p_tail, block);
    }
    else if (ch_rem > 0)
    {
        arm_depthwise_direct_block_mve_f16(in_tap + c, w_tap + c, bias_c, out_px + c, n_px, run, 1, 1, p_tail, block);
    }
}

static void __attribute__((noinline))
arm_depthwise_direct_run_mve_fold_f16(const float16_t *__RESTRICT in_tap,
                                      const float16_t *__RESTRICT w_tap,
                                      const float16_t *__RESTRICT bias,
                                      float16_t *__RESTRICT out_px,
                                      int32_t n_px,
                                      const arm_depthwise_direct_run_f16 *__RESTRICT run)
{
    arm_depthwise_direct_run_mve_f16_body(in_tap, w_tap, bias, out_px, n_px, run, ARM_NN_F16_ACC_BLOCK);
}

static void __attribute__((noinline))
arm_depthwise_direct_run_mve_acc16_f16(const float16_t *__RESTRICT in_tap,
                                       const float16_t *__RESTRICT w_tap,
                                       const float16_t *__RESTRICT bias,
                                       float16_t *__RESTRICT out_px,
                                       int32_t n_px,
                                       const arm_depthwise_direct_run_f16 *__RESTRICT run)
{
    arm_depthwise_direct_run_mve_f16_body(in_tap, w_tap, bias, out_px, n_px, run, ARM_NN_F16_ACC_BLOCK_NONE);
}
    #else
/* Scalar twin: same tap order, float32 accumulation, one f16 rounding at the store (#449, #457), then the same
 * NaN-propagating clamp as arm_nn_vector_clamp_f16's scalar leg. */
static void arm_depthwise_direct_run_scalar_f16(const float16_t *in_tap,
                                                const float16_t *w_tap,
                                                const float16_t *bias,
                                                float16_t *out_px,
                                                int32_t n_px,
                                                const arm_depthwise_direct_run_f16 *run)
{
    const size_t ch = run->ch;

    for (int32_t px = 0; px < n_px; ++px)
    {
        for (size_t c = 0; c < ch; ++c)
        {
            float32_t acc = bias ? (float32_t)bias[c] : 0.0f;
            for (int32_t ky = 0; ky < run->taps_y; ++ky)
            {
                const float16_t *in = in_tap + c + (size_t)ky * run->in_row_step;
                const float16_t *w = w_tap + c + (size_t)ky * run->w_row_step;
                for (int32_t kx = 0; kx < run->taps_x; ++kx)
                {
                    acc += (float32_t)in[(size_t)kx * run->in_tap_step] * (float32_t)w[(size_t)kx * ch];
                }
            }
            out_px[c] = arm_nn_clamp_scalar_f16((float16_t)acc, run->act_min, run->act_max);
        }
        in_tap += run->in_px_step;
        out_px += ch;
    }
}
    #endif

/* First / one-past-last in-range tap index of a padded edge (main's generic-route formulas, dilation-aware). */
__STATIC_FORCEINLINE int32_t arm_depthwise_direct_tap_start(int32_t base_idx, int32_t dilation)
{
    if (dilation > 1)
    {
        return ARM_NN_MAX(0, (-base_idx + dilation - 1) / dilation);
    }
    return ARM_NN_MAX(0, -base_idx);
}

__STATIC_FORCEINLINE int32_t arm_depthwise_direct_tap_end(int32_t base_idx,
                                                          int32_t dilation,
                                                          int32_t kernel_size,
                                                          int32_t input_size)
{
    if (dilation > 1)
    {
        return ARM_NN_MIN(kernel_size, (input_size - base_idx + dilation - 1) / dilation);
    }
    return ARM_NN_MIN(kernel_size, input_size - base_idx);
}

__STATIC_FORCEINLINE void arm_depthwise_conv_nhwc_direct_chmult1_f16(const float16_t *input,
                                                                     int32_t input_batches,
                                                                     int32_t input_x,
                                                                     int32_t input_y,
                                                                     int32_t input_ch,
                                                                     const float16_t *kernel,
                                                                     int32_t kernel_x,
                                                                     int32_t kernel_y,
                                                                     int32_t pad_x,
                                                                     int32_t pad_y,
                                                                     int32_t stride_x,
                                                                     int32_t stride_y,
                                                                     int32_t dilation_x,
                                                                     int32_t dilation_y,
                                                                     const float16_t *bias,
                                                                     float16_t *output,
                                                                     int32_t output_x,
                                                                     int32_t output_y,
                                                                     float16_t output_activation_min,
                                                                     float16_t output_activation_max,
                                                                     const bool acc16)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_DW_DIRECT_RUN_F16                                                                                  \
            (acc16 ? arm_depthwise_direct_run_mve_acc16_f16 : arm_depthwise_direct_run_mve_fold_f16)
    #else
        #define ARM_DW_DIRECT_RUN_F16 arm_depthwise_direct_run_scalar_f16
    (void)acc16;
    #endif
    const size_t ch = (size_t)input_ch;
    const size_t in_row_elems = (size_t)input_x * ch;
    const size_t in_batch_stride = (size_t)input_y * in_row_elems;
    const size_t out_batch_stride = (size_t)output_x * (size_t)output_y * ch;
    arm_depthwise_direct_run_f16 run;
    run.in_row_step = (size_t)dilation_y * in_row_elems;
    run.in_tap_step = (size_t)dilation_x * ch;
    run.in_px_step = (size_t)stride_x * ch;
    run.w_row_step = (size_t)kernel_x * ch;
    run.ch = ch;
    run.act_min = output_activation_min;
    run.act_max = output_activation_max;

    /* Output columns whose every tap column is in range: [x_lo, x_hi). The rest take the per-pixel window. */
    int32_t x_lo = output_x;
    int32_t x_hi = output_x;
    if (stride_x > 0)
    {
        const int32_t hi_num = input_x - 1 + pad_x - (kernel_x - 1) * dilation_x;
        x_lo = (pad_x > 0) ? (pad_x + stride_x - 1) / stride_x : 0;
        x_hi = (hi_num >= 0) ? (hi_num / stride_x) + 1 : 0;
        x_lo = ARM_NN_MIN(x_lo, output_x);
        x_hi = ARM_NN_MIN(x_hi, output_x);
        x_hi = ARM_NN_MAX(x_hi, x_lo);
    }

    for (int32_t i_batch = 0; i_batch < input_batches; ++i_batch)
    {
        const float16_t *input_b = input + (size_t)i_batch * in_batch_stride;
        float16_t *output_b = output + (size_t)i_batch * out_batch_stride;

        for (int32_t i_out_y = 0; i_out_y < output_y; ++i_out_y)
        {
            const int32_t base_idx_y = (i_out_y * stride_y) - pad_y;
            const int32_t ker_y_start = arm_depthwise_direct_tap_start(base_idx_y, dilation_y);
            const int32_t ker_y_end = arm_depthwise_direct_tap_end(base_idx_y, dilation_y, kernel_y, input_y);
            /* Row pointers of the first in-range tap row; left at the tensor base (never read) when taps_y == 0. */
            const float16_t *in_row = input_b;
            const float16_t *w_row = kernel;
            run.taps_y = ARM_NN_MAX(0, ker_y_end - ker_y_start);
            if (run.taps_y > 0)
            {
                in_row = input_b + (size_t)(base_idx_y + dilation_y * ker_y_start) * in_row_elems;
                w_row = kernel + (size_t)ker_y_start * run.w_row_step;
            }
            float16_t *out_px = output_b + (size_t)i_out_y * (size_t)output_x * ch;

            for (int32_t i_out_x = 0; i_out_x < output_x; ++i_out_x)
            {
                const int32_t base_idx_x = (i_out_x * stride_x) - pad_x;
                int32_t n_px = 1;
                const float16_t *in_tap = input_b;
                const float16_t *w_tap = kernel;
                if (i_out_x == x_lo && x_hi > x_lo)
                {
                    /* Interior run: every tap column in range, one call for the whole run. */
                    n_px = x_hi - x_lo;
                    run.taps_x = kernel_x;
                    in_tap = in_row + (size_t)base_idx_x * ch;
                    w_tap = w_row;
                }
                else
                {
                    const int32_t ker_x_start = arm_depthwise_direct_tap_start(base_idx_x, dilation_x);
                    const int32_t ker_x_end = arm_depthwise_direct_tap_end(base_idx_x, dilation_x, kernel_x, input_x);
                    run.taps_x = ARM_NN_MAX(0, ker_x_end - ker_x_start);
                    if (run.taps_x > 0)
                    {
                        in_tap = in_row + (size_t)(base_idx_x + dilation_x * ker_x_start) * ch;
                        w_tap = w_row + (size_t)ker_x_start * ch;
                    }
                }
                if (run.taps_y == 0)
                {
                    /* Fully padded row (padding wider than the kernel): bias only, nothing read. */
                    run.taps_x = 0;
                }
                ARM_DW_DIRECT_RUN_F16(in_tap, w_tap, bias, out_px, n_px, &run);
                out_px += (size_t)n_px * ch;
                i_out_x += n_px - 1;
            }
        }
    }
    #undef ARM_DW_DIRECT_RUN_F16
}

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE bool arm_depthwise_conv_nhwc_convert_to_conv_f16(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                                 const cmsis_nn_dims *input_dims,
                                                                 const cmsis_nn_dims *output_dims)
{
    return dw_conv_params && input_dims && output_dims && arm_nn_dw_f16_is_cin1(input_dims, output_dims);
}

__STATIC_INLINE void arm_depthwise_pack_conv_kernel_nt_n_f16(const float16_t *kernel,
                                                             int32_t output_c,
                                                             int32_t kernel_h,
                                                             int32_t kernel_w,
                                                             arm_nn_dw_kernel_layout_f16 kernel_layout,
                                                             float16_t *packed_kernel)
{
    const int32_t kernel_elems = kernel_h * kernel_w;
    const int32_t block_cols = 8;
    const int32_t packed_output_c = ARM_NN_ROUND_UP(output_c, block_cols);

    for (int32_t n_base = 0; n_base < packed_output_c; n_base += block_cols)
    {
        for (int32_t k = 0; k < kernel_elems; ++k)
        {
            float16_t *dst = packed_kernel + (size_t)n_base * kernel_elems + (size_t)k * block_cols;
            for (int32_t lane = 0; lane < block_cols; ++lane)
            {
                const int32_t oc = n_base + lane;
                if (oc < output_c)
                {
                    dst[lane] = (kernel_layout == ARM_NN_DW_KERNEL_CK) ? kernel[(size_t)oc * kernel_elems + k]
                                                                       : kernel[(size_t)k * output_c + oc];
                }
                else
                {
                    dst[lane] = (float16_t)0.0f;
                }
            }
        }
    }
}

__STATIC_FORCEINLINE arm_cmsis_nn_status
arm_depthwise_conv_nhwc_to_conv_packed_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *packed_kernel,
                                           const float16_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output,
                                           const bool acc16)
{
    if (!ctx || !ctx->buf || ctx->size <= 0 || !dw_conv_params || !input_dims || !input || !packed_kernel ||
        !output_dims || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const int32_t patch_len = kernel_h * kernel_w;
    const int32_t output_positions = output_h * output_w;

    if (input_dims->c != 1 || output_c <= 0 || kernel_h <= 0 || kernel_w <= 0 || patch_len <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t stride_h = dw_conv_params->stride.h;
    const int32_t stride_w = dw_conv_params->stride.w;
    const int32_t pad_h = dw_conv_params->padding.h;
    const int32_t pad_w = dw_conv_params->padding.w;
    const int32_t dil_h = dw_conv_params->dilation.h;
    const int32_t dil_w = dw_conv_params->dilation.w;
    const size_t row_bytes = (size_t)patch_len * sizeof(float16_t);
    const int32_t max_rows = (int32_t)((size_t)ctx->size / row_bytes);

    if (max_rows <= 0)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    /*
     * The converted depthwise path always has input_c == 1, so each patch row
     * is simply the kernel footprint scalars. Packing that directly here avoids
     * the general NHWC patch helper overhead.
     */
    float16_t *lhs_buffer = (float16_t *)ctx->buf;
    const int32_t in_batch_stride = input_h * input_w;
    const int32_t out_batch_stride = output_h * output_w * output_c;

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input + (size_t)b * in_batch_stride;
        float16_t *output_b = output + (size_t)b * out_batch_stride;

        for (int32_t pos = 0; pos < output_positions; pos += max_rows)
        {
            const int32_t rows = ((output_positions - pos) < max_rows) ? (output_positions - pos) : max_rows;

            for (int32_t r = 0; r < rows; ++r)
            {
                const int32_t out_pos = pos + r;
                const int32_t out_y = out_pos / output_w;
                const int32_t out_x = out_pos - out_y * output_w;
                const int32_t base_idx_y = (out_y * stride_h) - pad_h;
                const int32_t base_idx_x = (out_x * stride_w) - pad_w;
                float16_t *patch_row = lhs_buffer + (size_t)r * patch_len;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const int32_t idx_y = base_idx_y + dil_h * ky;
                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const int32_t idx_x = base_idx_x + dil_w * kx;
                        *patch_row++ = ((uint32_t)idx_y < (uint32_t)input_h && (uint32_t)idx_x < (uint32_t)input_w)
                            ? input_b[idx_y * input_w + idx_x]
                            : (float16_t)0.0f;
                    }
                }
            }

            arm_cmsis_nn_status status = (acc16 ? arm_nn_mat_mult_nt_n_packed_f16_acc16
                                                : arm_nn_mat_mult_nt_n_packed_f16)(lhs_buffer,
                                                                                   packed_kernel,
                                                                                   bias,
                                                                                   output_b + (size_t)pos * output_c,
                                                                                   rows,
                                                                                   output_c,
                                                                                   patch_len,
                                                                                   output_c,
                                                                                   dw_conv_params->activation.min,
                                                                                   dw_conv_params->activation.max);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

__STATIC_FORCEINLINE arm_cmsis_nn_status
arm_depthwise_conv_nhwc_to_conv_f16(const cmsis_nn_context *ctx,
                                    const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                    const cmsis_nn_dims *input_dims,
                                    const float16_t *input,
                                    const cmsis_nn_dims *filter_dims,
                                    const float16_t *kernel,
                                    const cmsis_nn_dims *bias_dims,
                                    const float16_t *bias,
                                    const cmsis_nn_dims *output_dims,
                                    float16_t *output,
                                    arm_nn_dw_kernel_layout_f16 kernel_layout,
                                    const bool acc16)
{
    /* The packed filter and one patch row, bounded in 64 bits so that no filter can wrap the size_t sums below */
    if (!ctx || !ctx->buf || (int64_t)ctx->size < arm_nn_dw_f16_cin1_min_ctx_size(filter_dims, output_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    const int32_t output_c = output_dims->c;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const size_t kernel_row_elems = (size_t)kernel_h * (size_t)kernel_w;
    const size_t packed_output_c = (size_t)ARM_NN_ROUND_UP(output_c, 8);
    const size_t kernel_bytes = packed_output_c * kernel_row_elems * sizeof(float16_t);
    uint8_t *ctx_bytes = (uint8_t *)ctx->buf;
    const float16_t *conv_kernel;
    size_t scratch_offset = 0U;

    if ((size_t)ctx->size < kernel_bytes)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    /*
     * The converted conv path stores the filter in packed NTxN scratch memory
     * and then consumes it as float16_t data. Require float16_t-aligned scratch
     * here so the typed view is explicit and safe.
     */
    if (((uintptr_t)ctx->buf & (sizeof(float16_t) - 1U)) != 0U)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (kernel_layout != ARM_NN_DW_KERNEL_CK && kernel_layout != ARM_NN_DW_KERNEL_KC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    float16_t *packed_kernel = (float16_t *)ctx->buf;
    arm_depthwise_pack_conv_kernel_nt_n_f16(kernel, output_c, kernel_h, kernel_w, kernel_layout, packed_kernel);
    conv_kernel = packed_kernel;
    scratch_offset = kernel_bytes;

    cmsis_nn_context conv_ctx = {0};
    if ((size_t)ctx->size > scratch_offset)
    {
        conv_ctx.buf = ctx_bytes + scratch_offset;
        conv_ctx.size = ctx->size - (int32_t)scratch_offset;
    }

    const cmsis_nn_conv_params_f16 conv_params = {.stride = dw_conv_params->stride,
                                                  .padding = dw_conv_params->padding,
                                                  .dilation = dw_conv_params->dilation,
                                                  .activation = dw_conv_params->activation,
                                                  .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims conv_filter_dims = {.n = output_c, .h = kernel_h, .w = kernel_w, .c = input_dims->c};
    (void)bias_dims;
    (void)conv_filter_dims;
    (void)conv_params;

    return arm_depthwise_conv_nhwc_to_conv_packed_f16(
        &conv_ctx, dw_conv_params, input_dims, input, filter_dims, conv_kernel, bias, output_dims, output, acc16);
}
    #endif

__STATIC_FORCEINLINE void arm_depthwise_conv_f16_generic(const float16_t *input,
                                                         const int32_t input_batches,
                                                         const int32_t input_x,
                                                         const int32_t input_y,
                                                         const int32_t input_ch,
                                                         const float16_t *kernel,
                                                         const int32_t ch_mult,
                                                         const int32_t kernel_x,
                                                         const int32_t kernel_y,
                                                         const int32_t pad_x,
                                                         const int32_t pad_y,
                                                         const int32_t stride_x,
                                                         const int32_t stride_y,
                                                         const float16_t *bias,
                                                         float16_t *output,
                                                         const int32_t output_x,
                                                         const int32_t output_y,
                                                         const float16_t output_activation_min,
                                                         const float16_t output_activation_max,
                                                         const int32_t dilation_x,
                                                         const int32_t dilation_y,
                                                         const cmsis_nn_dw_conv_params_f16 *params,
                                                         arm_nn_dw_kernel_layout_f16 kernel_layout,
                                                         const bool acc16)
{
    (void)params;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* Folding (#586), MVE builds: with more than `block` in-range taps, each float16 partial covers at most `block`
     * taps before it is widened into a float32 accumulator; one rounding at the end. */
    const int32_t block = acc16 ? ARM_NN_F16_ACC_BLOCK_NONE : ARM_NN_F16_ACC_BLOCK;
    #else
    /* Scalar legs accumulate the whole window, bias included, in float32 and round to float16 once at the store,
     * as arm_depthwise_direct_run_scalar_f16 does (#449). Both entries agree there. */
    (void)acc16;
    #endif
    const int32_t output_ch = input_ch * ch_mult;
    const int32_t in_batch_stride = input_x * input_y * input_ch;
    const int32_t out_batch_stride = output_x * output_y * output_ch;

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        const float16_t *input_b = input + i_batch * in_batch_stride;
        float16_t *output_b = output + i_batch * out_batch_stride;

        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            const int32_t base_idx_y = (i_out_y * stride_y) - pad_y;
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_idx_x = (i_out_x * stride_x) - pad_x;
                for (int32_t i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                {
                    for (int32_t i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                    {
                        const int32_t idx_out_ch = i_ch_mult + i_input_ch * ch_mult;
                        _Float16 acc_0 = (_Float16)0;

                        int32_t ker_y_start;
                        int32_t ker_x_start;
                        int32_t ker_y_end;
                        int32_t ker_x_end;

                        if (dilation_x > 1)
                        {
                            const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                            ker_x_start = (start_x_max > 0) ? start_x_max : 0;
                            const int32_t end_min_x = (input_x - base_idx_x + dilation_x - 1) / dilation_x;
                            ker_x_end = (kernel_x < end_min_x) ? kernel_x : end_min_x;
                        }
                        else
                        {
                            ker_x_start = (base_idx_x < 0) ? -base_idx_x : 0;
                            const int32_t end_min_x = input_x - base_idx_x;
                            ker_x_end = (kernel_x < end_min_x) ? kernel_x : end_min_x;
                        }

                        if (dilation_y > 1)
                        {
                            const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                            ker_y_start = (start_y_max > 0) ? start_y_max : 0;
                            const int32_t end_min_y = (input_y - base_idx_y + dilation_y - 1) / dilation_y;
                            ker_y_end = (kernel_y < end_min_y) ? kernel_y : end_min_y;
                        }
                        else
                        {
                            ker_y_start = (base_idx_y < 0) ? -base_idx_y : 0;
                            const int32_t end_min_y = input_y - base_idx_y;
                            ker_y_end = (kernel_y < end_min_y) ? kernel_y : end_min_y;
                        }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                        if (bias)
                        {
                            acc_0 = (_Float16)bias[idx_out_ch];
                        }

                        const bool fold = (ker_y_end - ker_y_start) * (ker_x_end - ker_x_start) > block;
                        float32_t acc32 = 0.0f;
                        bool first = true;
                        int32_t n_taps = 0;
    #else
                        float32_t acc32 = bias ? (float32_t)bias[idx_out_ch] : 0.0f;
    #endif
                        for (int32_t i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                        {
                            const int32_t idx_y = base_idx_y + dilation_y * i_ker_y;
                            for (int32_t i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                            {
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                                if (fold)
                                {
                                    if (n_taps == block)
                                    {
                                        acc32 = first ? (float32_t)acc_0 : acc32 + (float32_t)acc_0;
                                        first = false;
                                        acc_0 = (_Float16)0;
                                        n_taps = 0;
                                    }
                                    ++n_taps;
                                }
    #endif
                                const int32_t idx_x = base_idx_x + dilation_x * i_ker_x;
                                const int32_t idx_0 =
                                    arm_depthwise_conv_input_index_nhwc(idx_x, idx_y, i_input_ch, input_x, input_ch);
                                int32_t ker_idx_0;
                                if (kernel_layout == ARM_NN_DW_KERNEL_CK)
                                {
                                    ker_idx_0 = idx_out_ch * (kernel_x * kernel_y) + i_ker_y * kernel_x + i_ker_x;
                                }
                                else
                                {
                                    ker_idx_0 = (i_ker_y * kernel_x + i_ker_x) * (output_ch) + idx_out_ch;
                                }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                                acc_0 += (_Float16)input_b[idx_0] * (_Float16)kernel[ker_idx_0];
    #else
                                acc32 += (float32_t)input_b[idx_0] * (float32_t)kernel[ker_idx_0];
    #endif
                            }
                        }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                        if (fold && !first)
                        {
                            acc_0 = (_Float16)(acc32 + (float32_t)acc_0);
                        }
    #else
                        acc_0 = (_Float16)acc32;
    #endif
                        acc_0 =
                            arm_nn_clamp_f16h(acc_0, (_Float16)output_activation_max, (_Float16)output_activation_min);
                        const int32_t out_idx =
                            arm_depthwise_conv_output_index_nhwc(i_out_x, i_out_y, idx_out_ch, output_x, output_ch);
                        output_b[out_idx] = (float16_t)acc_0;
                    }
                }
            }
        }
    }
}

static arm_cmsis_nn_status arm_depthwise_conv_f16_validate(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const cmsis_nn_dims *output_dims,
                                                           const float16_t *input,
                                                           const float16_t *kernel,
                                                           const float16_t *output,
                                                           arm_nn_dw_kernel_layout_f16 forced_kernel_layout,
                                                           arm_nn_dw_kernel_layout_f16 *kernel_layout)
{
    if (!dw_conv_params || !input_dims || !filter_dims || !output_dims || !input || !kernel || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    *kernel_layout = forced_kernel_layout;

    return ARM_CMSIS_NN_SUCCESS;
}

__STATIC_FORCEINLINE arm_cmsis_nn_status
arm_depthwise_conv_nhwc_dispatch_f16(const cmsis_nn_context *ctx,
                                     const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                     const cmsis_nn_dims *input_dims,
                                     const float16_t *input,
                                     const cmsis_nn_dims *filter_dims,
                                     const float16_t *kernel,
                                     const cmsis_nn_dims *bias_dims,
                                     const float16_t *bias,
                                     const cmsis_nn_dims *output_dims,
                                     float16_t *output,
                                     arm_nn_dw_kernel_layout_f16 kernel_layout,
                                     const bool acc16)
{
    /* Read by the table and the to-conv route only; neither is compiled on every leg. */
    (void)ctx;

    #ifndef NN_DISABLE_SPECIALIZATION
    /* First try the exact-shape NHWC specializations (1D-k3, 2x5). */
    ARM_DW_DISPATCH(arm_dw_spec_nhwc_f16,
                    ARM_DW_ARRAY_SIZE(arm_dw_spec_nhwc_f16),
                    ctx,
                    dw_conv_params,
                    input_dims,
                    input,
                    filter_dims,
                    kernel,
                    bias_dims,
                    bias,
                    output_dims,
                    output,
                    kernel_layout);
    #endif

    /* ch_mult == 1: the direct channel-vectorized kernel, any stride/dilation/padding/batch, no scratch (#448).
     * The sizer asks for none here either (#625). */
    if (dw_conv_params->ch_mult == 1)
    {
        arm_depthwise_conv_nhwc_direct_chmult1_f16(input,
                                                   input_dims->n,
                                                   input_dims->w,
                                                   input_dims->h,
                                                   input_dims->c,
                                                   kernel,
                                                   filter_dims->w,
                                                   filter_dims->h,
                                                   dw_conv_params->padding.w,
                                                   dw_conv_params->padding.h,
                                                   dw_conv_params->stride.w,
                                                   dw_conv_params->stride.h,
                                                   dw_conv_params->dilation.w,
                                                   dw_conv_params->dilation.h,
                                                   bias,
                                                   output,
                                                   output_dims->w,
                                                   output_dims->h,
                                                   dw_conv_params->activation.min,
                                                   dw_conv_params->activation.max,
                                                   acc16);
        return ARM_CMSIS_NN_SUCCESS;
    }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* For one-input-channel cases on MVE, reusing the float convolution wrapper can be faster. */
    if (arm_depthwise_conv_nhwc_convert_to_conv_f16(dw_conv_params, input_dims, output_dims))
    {
        arm_cmsis_nn_status conv_status = arm_depthwise_conv_nhwc_to_conv_f16(ctx,
                                                                              dw_conv_params,
                                                                              input_dims,
                                                                              input,
                                                                              filter_dims,
                                                                              kernel,
                                                                              bias_dims,
                                                                              bias,
                                                                              output_dims,
                                                                              output,
                                                                              kernel_layout,
                                                                              acc16);
        if (conv_status == ARM_CMSIS_NN_SUCCESS)
        {
            return conv_status;
        }
    }
    #endif

    /* Generic NHWC fallback for remaining channel-multiplier, dilation, or kernel-layout cases. */
    arm_depthwise_conv_f16_generic(input,
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
                                   output_dims->w,
                                   output_dims->h,
                                   dw_conv_params->activation.min,
                                   dw_conv_params->activation.max,
                                   dw_conv_params->dilation.w,
                                   dw_conv_params->dilation.h,
                                   dw_conv_params,
                                   kernel_layout,
                                   acc16);

    return ARM_CMSIS_NN_SUCCESS;
}

/* Shared body; `acc16` is a constant at every call site (see the two instantiations below). */
__STATIC_FORCEINLINE arm_cmsis_nn_status
arm_depthwise_nhwc_conv_f16_body(const cmsis_nn_context *ctx,
                                 const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                 const cmsis_nn_dims *input_dims,
                                 const float16_t *input,
                                 const cmsis_nn_dims *filter_dims,
                                 const float16_t *kernel,
                                 const cmsis_nn_dims *bias_dims,
                                 const float16_t *bias,
                                 const cmsis_nn_dims *output_dims,
                                 float16_t *output,
                                 const bool acc16)
{
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    arm_cmsis_nn_status status = arm_depthwise_conv_f16_validate(dw_conv_params,
                                                                 input_dims,
                                                                 filter_dims,
                                                                 output_dims,
                                                                 input,
                                                                 kernel,
                                                                 output,
                                                                 ARM_NN_DW_KERNEL_KC,
                                                                 &kernel_layout);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return status;
    }

    return arm_depthwise_conv_nhwc_dispatch_f16(ctx,
                                                dw_conv_params,
                                                input_dims,
                                                input,
                                                filter_dims,
                                                kernel,
                                                bias_dims,
                                                bias,
                                                output_dims,
                                                output,
                                                kernel_layout,
                                                acc16);
}

/* One out-of-line instantiation per variant, each called by that variant's two public entries. */
static __attribute__((noinline)) arm_cmsis_nn_status
arm_depthwise_nhwc_conv_f16_fold(const cmsis_nn_context *ctx,
                                 const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                 const cmsis_nn_dims *input_dims,
                                 const float16_t *input,
                                 const cmsis_nn_dims *filter_dims,
                                 const float16_t *kernel,
                                 const cmsis_nn_dims *bias_dims,
                                 const float16_t *bias,
                                 const cmsis_nn_dims *output_dims,
                                 float16_t *output)
{
    return arm_depthwise_nhwc_conv_f16_body(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output, false);
}

static __attribute__((noinline)) arm_cmsis_nn_status
arm_depthwise_nhwc_conv_f16_acc16_impl(const cmsis_nn_context *ctx,
                                       const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                       const cmsis_nn_dims *input_dims,
                                       const float16_t *input,
                                       const cmsis_nn_dims *filter_dims,
                                       const float16_t *kernel,
                                       const cmsis_nn_dims *bias_dims,
                                       const float16_t *bias,
                                       const cmsis_nn_dims *output_dims,
                                       float16_t *output)
{
    return arm_depthwise_nhwc_conv_f16_body(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output, true);
}

arm_cmsis_nn_status arm_depthwise_nhwc_conv_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *kernel,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output)
{
    return arm_depthwise_nhwc_conv_f16_fold(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status arm_depthwise_nhwc_conv_f16_acc16(const cmsis_nn_context *ctx,
                                                      const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const float16_t *input,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const float16_t *kernel,
                                                      const cmsis_nn_dims *bias_dims,
                                                      const float16_t *bias,
                                                      const cmsis_nn_dims *output_dims,
                                                      float16_t *output)
{
    return arm_depthwise_nhwc_conv_f16_acc16_impl(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status arm_depthwise_conv_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output,
                                           arm_nn_tensor_layout layout)
{
    if (!dw_conv_params || layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_depthwise_nhwc_conv_f16(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status arm_depthwise_conv_f16_acc16(const cmsis_nn_context *ctx,
                                                 const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const float16_t *input,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const float16_t *kernel,
                                                 const cmsis_nn_dims *bias_dims,
                                                 const float16_t *bias,
                                                 const cmsis_nn_dims *output_dims,
                                                 float16_t *output,
                                                 arm_nn_tensor_layout layout)
{
    if (!dw_conv_params || layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_depthwise_nhwc_conv_f16_acc16(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status arm_depthwise_conv_wrapper_f16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float16_t *input,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float16_t *kernel,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float16_t *bias,
                                                   const cmsis_nn_dims *output_dims,
                                                   float16_t *output)
{
    return arm_depthwise_nhwc_conv_f16_fold(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status arm_depthwise_conv_wrapper_f16_acc16(const cmsis_nn_context *ctx,
                                                         const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const float16_t *input,
                                                         const cmsis_nn_dims *filter_dims,
                                                         const float16_t *kernel,
                                                         const cmsis_nn_dims *bias_dims,
                                                         const float16_t *bias,
                                                         const cmsis_nn_dims *output_dims,
                                                         float16_t *output)
{
    return arm_depthwise_nhwc_conv_f16_acc16_impl(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status arm_depthwise_conv_1d_k3_nhwc_f16(const cmsis_nn_context *ctx,
                                                      const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const float16_t *input,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const float16_t *kernel,
                                                      const cmsis_nn_dims *bias_dims,
                                                      const float16_t *bias,
                                                      const cmsis_nn_dims *output_dims,
                                                      float16_t *output)
{
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    if (arm_depthwise_conv_f16_validate(dw_conv_params,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        input,
                                        kernel,
                                        output,
                                        ARM_NN_DW_KERNEL_KC,
                                        &kernel_layout) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!arm_nn_dw_f16_is_1d_k3(dw_conv_params, input_dims, filter_dims, output_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    return arm_dw_spec_k3_1d_nhwc_f16_call(ctx,
                                           dw_conv_params,
                                           input_dims,
                                           input,
                                           filter_dims,
                                           kernel,
                                           bias_dims,
                                           bias,
                                           output_dims,
                                           output,
                                           kernel_layout);
}

arm_cmsis_nn_status arm_depthwise_conv_2x5_nhwc_f16(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const float16_t *input,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const float16_t *kernel,
                                                    const cmsis_nn_dims *bias_dims,
                                                    const float16_t *bias,
                                                    const cmsis_nn_dims *output_dims,
                                                    float16_t *output)
{
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    if (arm_depthwise_conv_f16_validate(dw_conv_params,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        input,
                                        kernel,
                                        output,
                                        ARM_NN_DW_KERNEL_KC,
                                        &kernel_layout) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!arm_nn_dw_f16_is_2x5(dw_conv_params, input_dims, filter_dims, output_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    return arm_dw_spec_2x5_nhwc_f16_call(ctx,
                                         dw_conv_params,
                                         input_dims,
                                         input,
                                         filter_dims,
                                         kernel,
                                         bias_dims,
                                         bias,
                                         output_dims,
                                         output,
                                         kernel_layout);
}

arm_cmsis_nn_status arm_depthwise_conv_direct_nhwc_f16(const cmsis_nn_context *ctx,
                                                       const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const float16_t *input,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const float16_t *kernel,
                                                       const cmsis_nn_dims *bias_dims,
                                                       const float16_t *bias,
                                                       const cmsis_nn_dims *output_dims,
                                                       float16_t *output)
{
    (void)ctx;
    (void)bias_dims;
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    if (arm_depthwise_conv_f16_validate(dw_conv_params,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        input,
                                        kernel,
                                        output,
                                        ARM_NN_DW_KERNEL_KC,
                                        &kernel_layout) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    (void)kernel_layout;
    if (dw_conv_params->ch_mult != 1)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    arm_depthwise_conv_nhwc_direct_chmult1_f16(input,
                                               input_dims->n,
                                               input_dims->w,
                                               input_dims->h,
                                               input_dims->c,
                                               kernel,
                                               filter_dims->w,
                                               filter_dims->h,
                                               dw_conv_params->padding.w,
                                               dw_conv_params->padding.h,
                                               dw_conv_params->stride.w,
                                               dw_conv_params->stride.h,
                                               dw_conv_params->dilation.w,
                                               dw_conv_params->dilation.h,
                                               bias,
                                               output,
                                               output_dims->w,
                                               output_dims->h,
                                               dw_conv_params->activation.min,
                                               dw_conv_params->activation.max,
                                               false);
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_depthwise_conv_direct_nhwc_f16_acc16(const cmsis_nn_context *ctx,
                                                             const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                             const cmsis_nn_dims *input_dims,
                                                             const float16_t *input,
                                                             const cmsis_nn_dims *filter_dims,
                                                             const float16_t *kernel,
                                                             const cmsis_nn_dims *bias_dims,
                                                             const float16_t *bias,
                                                             const cmsis_nn_dims *output_dims,
                                                             float16_t *output)
{
    (void)ctx;
    (void)bias_dims;
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    if (arm_depthwise_conv_f16_validate(dw_conv_params,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        input,
                                        kernel,
                                        output,
                                        ARM_NN_DW_KERNEL_KC,
                                        &kernel_layout) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    (void)kernel_layout;
    if (dw_conv_params->ch_mult != 1)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    arm_depthwise_conv_nhwc_direct_chmult1_f16(input,
                                               input_dims->n,
                                               input_dims->w,
                                               input_dims->h,
                                               input_dims->c,
                                               kernel,
                                               filter_dims->w,
                                               filter_dims->h,
                                               dw_conv_params->padding.w,
                                               dw_conv_params->padding.h,
                                               dw_conv_params->stride.w,
                                               dw_conv_params->stride.h,
                                               dw_conv_params->dilation.w,
                                               dw_conv_params->dilation.h,
                                               bias,
                                               output,
                                               output_dims->w,
                                               output_dims->h,
                                               dw_conv_params->activation.min,
                                               dw_conv_params->activation.max,
                                               true);
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_depthwise_conv_cin1_nhwc_f16(const cmsis_nn_context *ctx,
                                                     const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const float16_t *input,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const float16_t *kernel,
                                                     const cmsis_nn_dims *bias_dims,
                                                     const float16_t *bias,
                                                     const cmsis_nn_dims *output_dims,
                                                     float16_t *output)
{
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    if (arm_depthwise_conv_f16_validate(dw_conv_params,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        input,
                                        kernel,
                                        output,
                                        ARM_NN_DW_KERNEL_KC,
                                        &kernel_layout) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!arm_nn_dw_f16_is_cin1(input_dims, output_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const arm_cmsis_nn_status status = arm_depthwise_conv_nhwc_to_conv_f16(ctx,
                                                                           dw_conv_params,
                                                                           input_dims,
                                                                           input,
                                                                           filter_dims,
                                                                           kernel,
                                                                           bias_dims,
                                                                           bias,
                                                                           output_dims,
                                                                           output,
                                                                           kernel_layout,
                                                                           false);
    /* The router falls back to the generic route when ctx is too small; the entry reports it */
    return status == ARM_CMSIS_NN_NO_IMPL_ERROR ? ARM_CMSIS_NN_ARG_ERROR : status;
    #else
    (void)ctx;
    (void)bias_dims;
    (void)bias;
    (void)kernel_layout;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
    #endif
}

arm_cmsis_nn_status arm_depthwise_conv_cin1_nhwc_f16_acc16(const cmsis_nn_context *ctx,
                                                           const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const float16_t *input,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const float16_t *kernel,
                                                           const cmsis_nn_dims *bias_dims,
                                                           const float16_t *bias,
                                                           const cmsis_nn_dims *output_dims,
                                                           float16_t *output)
{
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    if (arm_depthwise_conv_f16_validate(dw_conv_params,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        input,
                                        kernel,
                                        output,
                                        ARM_NN_DW_KERNEL_KC,
                                        &kernel_layout) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!arm_nn_dw_f16_is_cin1(input_dims, output_dims))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const arm_cmsis_nn_status status = arm_depthwise_conv_nhwc_to_conv_f16(ctx,
                                                                           dw_conv_params,
                                                                           input_dims,
                                                                           input,
                                                                           filter_dims,
                                                                           kernel,
                                                                           bias_dims,
                                                                           bias,
                                                                           output_dims,
                                                                           output,
                                                                           kernel_layout,
                                                                           true);
    /* The router falls back to the generic route when ctx is too small; the entry reports it */
    return status == ARM_CMSIS_NN_NO_IMPL_ERROR ? ARM_CMSIS_NN_ARG_ERROR : status;
    #else
    (void)ctx;
    (void)bias_dims;
    (void)bias;
    (void)kernel_layout;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
    #endif
}

arm_cmsis_nn_status arm_depthwise_conv_generic_nhwc_f16(const cmsis_nn_context *ctx,
                                                        const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                        const cmsis_nn_dims *input_dims,
                                                        const float16_t *input,
                                                        const cmsis_nn_dims *filter_dims,
                                                        const float16_t *kernel,
                                                        const cmsis_nn_dims *bias_dims,
                                                        const float16_t *bias,
                                                        const cmsis_nn_dims *output_dims,
                                                        float16_t *output)
{
    (void)ctx;
    (void)bias_dims;
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    if (arm_depthwise_conv_f16_validate(dw_conv_params,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        input,
                                        kernel,
                                        output,
                                        ARM_NN_DW_KERNEL_KC,
                                        &kernel_layout) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_depthwise_conv_f16_generic(input,
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
                                   output_dims->w,
                                   output_dims->h,
                                   dw_conv_params->activation.min,
                                   dw_conv_params->activation.max,
                                   dw_conv_params->dilation.w,
                                   dw_conv_params->dilation.h,
                                   dw_conv_params,
                                   kernel_layout,
                                   false);
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_depthwise_conv_generic_nhwc_f16_acc16(const cmsis_nn_context *ctx,
                                                              const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                              const cmsis_nn_dims *input_dims,
                                                              const float16_t *input,
                                                              const cmsis_nn_dims *filter_dims,
                                                              const float16_t *kernel,
                                                              const cmsis_nn_dims *bias_dims,
                                                              const float16_t *bias,
                                                              const cmsis_nn_dims *output_dims,
                                                              float16_t *output)
{
    (void)ctx;
    (void)bias_dims;
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    if (arm_depthwise_conv_f16_validate(dw_conv_params,
                                        input_dims,
                                        filter_dims,
                                        output_dims,
                                        input,
                                        kernel,
                                        output,
                                        ARM_NN_DW_KERNEL_KC,
                                        &kernel_layout) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_depthwise_conv_f16_generic(input,
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
                                   output_dims->w,
                                   output_dims->h,
                                   dw_conv_params->activation.min,
                                   dw_conv_params->activation.max,
                                   dw_conv_params->dilation.w,
                                   dw_conv_params->dilation.h,
                                   dw_conv_params,
                                   kernel_layout,
                                   true);
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */

#endif /* ARM_NN_ENABLE_F16 */
