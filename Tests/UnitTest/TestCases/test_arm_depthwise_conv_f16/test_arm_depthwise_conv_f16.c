/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

// Ambiq-specific: not in upstream ARM-software/CMSIS-NN, which has no float depthwise convolution (#448).
// Float16 depthwise convolution: every ch_mult == 1 shape takes the direct channel-vectorized kernel (#448),
// ch_mult > 1 the generic route. Each case is checked against an in-test float64 reference with guard bytes
// on the output and on a scratch buffer sized exactly by the sizer. The MVE leg accumulates in float16 lanes,
// the scalar leg in float32 with one rounding at the store (#449), so the tolerance is per tap count.

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unity.h>

// Deterministic data in [-1, 1) on a 1/32 grid (exact in float16 as well).
static float16_t dw_f16_value(int32_t i, int32_t seed)
{
    return (float16_t)((float32_t)(((i * 37 + seed * 11) % 64) - 32) / 32.0f);
}

// NHWC depthwise reference in float64: filter [1][kh][kw][C * ch_mult], output channel ic * ch_mult + m.
static void dw_f16_reference(const cmsis_nn_dw_conv_params_f16 *dp,
                             const cmsis_nn_dims *in,
                             const float16_t *x,
                             const cmsis_nn_dims *flt,
                             const float16_t *w,
                             const float16_t *bias,
                             const cmsis_nn_dims *out,
                             double *y)
{
    const int32_t out_c = in->c * dp->ch_mult;
    for (int32_t b = 0; b < out->n; b++)
    {
        for (int32_t oy = 0; oy < out->h; oy++)
        {
            for (int32_t ox = 0; ox < out->w; ox++)
            {
                for (int32_t oc = 0; oc < out_c; oc++)
                {
                    const int32_t ic = oc / dp->ch_mult;
                    double acc = bias ? (double)(float32_t)bias[oc] : 0.0;
                    for (int32_t ky = 0; ky < flt->h; ky++)
                    {
                        const int32_t iy = oy * dp->stride.h - dp->padding.h + ky * dp->dilation.h;
                        if (iy < 0 || iy >= in->h)
                        {
                            continue;
                        }
                        for (int32_t kx = 0; kx < flt->w; kx++)
                        {
                            const int32_t ix = ox * dp->stride.w - dp->padding.w + kx * dp->dilation.w;
                            if (ix < 0 || ix >= in->w)
                            {
                                continue;
                            }
                            acc += (double)(float32_t)x[((b * in->h + iy) * in->w + ix) * in->c + ic] *
                                (double)(float32_t)w[(ky * flt->w + kx) * out_c + oc];
                        }
                    }
                    acc = fmin(fmax(acc, (double)(float32_t)dp->activation.min), (double)(float32_t)dp->activation.max);
                    y[((b * out->h + oy) * out->w + ox) * out_c + oc] = acc;
                }
            }
        }
    }
}

#define DW_GUARD_BYTES 64
#define DW_GUARD_FILL 0xA5

static void *dw_alloc_guarded(size_t size)
{
    uint8_t *p = (uint8_t *)malloc(size + DW_GUARD_BYTES);
    if (p != NULL)
    {
        memset(p, 0x3C, size);
        memset(p + size, DW_GUARD_FILL, DW_GUARD_BYTES);
    }
    return p;
}

static void dw_assert_guard_intact(const void *p, size_t size, const char *what)
{
    const uint8_t *guard = (const uint8_t *)p + size;
    for (size_t i = 0; i < DW_GUARD_BYTES; i++)
    {
        TEST_ASSERT_EQUAL_HEX8_MESSAGE(DW_GUARD_FILL, guard[i], what);
    }
}

static void dw_f16_params(cmsis_nn_dw_conv_params_f16 *dp,
                          int32_t ch_mult,
                          int32_t kh_stride,
                          int32_t kw_stride,
                          int32_t pad_h,
                          int32_t pad_w,
                          int32_t dil_h,
                          int32_t dil_w)
{
    memset(dp, 0, sizeof(*dp));
    dp->ch_mult = ch_mult;
    dp->stride.h = kh_stride;
    dp->stride.w = kw_stride;
    dp->padding.h = pad_h;
    dp->padding.w = pad_w;
    dp->dilation.h = dil_h;
    dp->dilation.w = dil_w;
    dp->activation.min = (float16_t)-1.0e4f;
    dp->activation.max = (float16_t)1.0e4f;
}

static int32_t dw_out_size(int32_t in, int32_t k, int32_t stride, int32_t pad, int32_t dil)
{
    return (in + pad - dil * (k - 1) - 1) / stride + 1;
}

// Run arm_depthwise_conv_wrapper_f16 (scratch sized exactly by the sizer when use_ctx, else no ctx), check
// the guards, and compare every output with the float64 reference within `tol`.
static void dw_f16_check(const cmsis_nn_dw_conv_params_f16 *dp,
                         const cmsis_nn_dims *in,
                         const float16_t *x,
                         const cmsis_nn_dims *flt,
                         const float16_t *w,
                         const float16_t *bias,
                         const cmsis_nn_dims *out,
                         int32_t use_ctx,
                         float32_t tol)
{
    const int32_t out_size = out->n * out->h * out->w * out->c;
    const size_t out_bytes = (size_t)out_size * sizeof(float16_t);
    const cmsis_nn_dims bias_dims = {1, 1, 1, out->c};
    double *ref = (double *)malloc((size_t)out_size * sizeof(double));
    float16_t *y = (float16_t *)dw_alloc_guarded(out_bytes);
    cmsis_nn_context ctx = {NULL, 0};
    TEST_ASSERT_NOT_NULL(ref);
    TEST_ASSERT_NOT_NULL(y);

    if (use_ctx)
    {
        const int32_t size = arm_depthwise_conv_wrapper_f16_get_buffer_size(dp, in, flt, out);
        TEST_ASSERT_EQUAL(size, arm_depthwise_conv_f16_get_buffer_size(dp, in, flt, out, ARM_NN_LAYOUT_NHWC));
        if (size > 0)
        {
            ctx.buf = dw_alloc_guarded((size_t)size);
            ctx.size = size;
            TEST_ASSERT_NOT_NULL(ctx.buf);
        }
    }

    dw_f16_reference(dp, in, x, flt, w, bias, out, ref);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_f16(&ctx, dp, in, x, flt, w, &bias_dims, bias, out, y));
    if (ctx.buf)
    {
        dw_assert_guard_intact(ctx.buf, (size_t)ctx.size, "kernel wrote past the sizer-sized scratch buffer");
    }
    dw_assert_guard_intact(y, out_bytes, "kernel wrote past the output buffer");
    for (int32_t i = 0; i < out_size; i++)
    {
        TEST_ASSERT_FLOAT_WITHIN(tol, (float32_t)ref[i], (float32_t)y[i]);
    }

    free(ctx.buf);
    free(y);
    free(ref);
}

// Fill x / w / bias for a layer and run the check with and without scratch. `bias` may be skipped (NULL).
static void dw_f16_layer(int32_t batch,
                         int32_t in_h,
                         int32_t in_w,
                         int32_t in_c,
                         int32_t kh,
                         int32_t kw,
                         const cmsis_nn_dw_conv_params_f16 *dp,
                         int32_t with_bias,
                         int32_t seed)
{
    const int32_t out_c = in_c * dp->ch_mult;
    const cmsis_nn_dims in = {batch, in_h, in_w, in_c};
    const cmsis_nn_dims flt = {1, kh, kw, out_c};
    const cmsis_nn_dims out = {batch,
                               dw_out_size(in_h, kh, dp->stride.h, dp->padding.h, dp->dilation.h),
                               dw_out_size(in_w, kw, dp->stride.w, dp->padding.w, dp->dilation.w),
                               out_c};
    const int32_t x_size = batch * in_h * in_w * in_c;
    const int32_t w_size = kh * kw * out_c;
    float16_t *x = (float16_t *)malloc((size_t)x_size * sizeof(float16_t));
    float16_t *w = (float16_t *)malloc((size_t)w_size * sizeof(float16_t));
    float16_t *bias = (float16_t *)malloc((size_t)out_c * sizeof(float16_t));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(w);
    TEST_ASSERT_NOT_NULL(bias);
    TEST_ASSERT_TRUE(out.h > 0 && out.w > 0);

    for (int32_t i = 0; i < x_size; i++)
    {
        x[i] = dw_f16_value(i, seed);
    }
    for (int32_t i = 0; i < w_size; i++)
    {
        w[i] = dw_f16_value(i, seed + 1);
    }
    for (int32_t i = 0; i < out_c; i++)
    {
        bias[i] = dw_f16_value(i, seed + 2);
    }

    // Float16 lanes on MVE: half an ulp of a partial sum below 32 (0.0078) per tap, plus the final rounding.
    const float32_t tol = 0.008f * (float32_t)(kh * kw) + 0.01f;
    dw_f16_check(dp, &in, x, &flt, w, with_bias ? bias : NULL, &out, 1, tol);
    dw_f16_check(dp, &in, x, &flt, w, with_bias ? bias : NULL, &out, 0, tol);

    free(bias);
    free(w);
    free(x);
}

// The KWS DEPTHWISE_CONV_2D layer of #448: [1,16,16,8] 3x3 stride 2, TFLite SAME (pad_before 0, one implicit
// pad row/column after), fused RELU. With the sizer's scratch (which the route no longer reads) and without.
void depthwise_conv_kws_layer_f16(void)
{
    cmsis_nn_dw_conv_params_f16 dp;
    dw_f16_params(&dp, 1, 2, 2, 0, 0, 1, 1);
    dp.activation.min = (float16_t)0.0f;
    dp.activation.max = (float16_t)65504.0f;
    dw_f16_layer(1, 16, 16, 8, 3, 3, &dp, 1, 1);
    dw_f16_layer(1, 16, 16, 8, 3, 3, &dp, 0, 4);
}

// arm_depthwise_conv_f16 with ARM_NN_LAYOUT_NHWC is bit-identical to the wrapper; any other layout is rejected
// before the kernel runs.
void depthwise_conv_layout_gate_f16(void)
{
    const cmsis_nn_dims in = {1, 16, 16, 8};
    const cmsis_nn_dims flt = {1, 3, 3, 8};
    const cmsis_nn_dims out = {1, 8, 8, 8};
    const cmsis_nn_dims bias_dims = {1, 1, 1, 8};
    static float16_t x[16 * 16 * 8];
    static float16_t w[72];
    static float16_t bias[8];
    static float16_t y_wrapper[8 * 8 * 8];
    static float16_t y_entry[8 * 8 * 8];
    cmsis_nn_dw_conv_params_f16 dp;
    cmsis_nn_context ctx = {NULL, 0};

    for (int32_t i = 0; i < 16 * 16 * 8; i++)
    {
        x[i] = dw_f16_value(i, 7);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = dw_f16_value(i, 8);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = dw_f16_value(i, 9);
    }
    dw_f16_params(&dp, 1, 2, 2, 0, 0, 1, 1);
    dp.activation.min = (float16_t)0.0f;
    dp.activation.max = (float16_t)6.0f;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_f16(&ctx, &dp, &in, x, &flt, w, &bias_dims, bias, &out, y_wrapper));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_depthwise_conv_f16(&ctx, &dp, &in, x, &flt, w, &bias_dims, bias, &out, y_entry, ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL_MEMORY(y_wrapper, y_entry, sizeof(y_wrapper));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_depthwise_conv_f16(&ctx, &dp, &in, x, &flt, w, &bias_dims, bias, &out, y_entry, (arm_nn_tensor_layout)1));
}

// Channel counts around the 8-lane f16 vector: pairs of vectors, one full vector, and every tail length, with a
// bias and without one (the NULL-bias substitute must cover a two-vector block, #448).
void depthwise_conv_channel_sweep_f16(void)
{
    const int32_t channels[] = {1, 3, 4, 5, 8, 9, 16, 17};
    cmsis_nn_dw_conv_params_f16 dp;
    dw_f16_params(&dp, 1, 1, 1, 1, 1, 1, 1);
    for (size_t i = 0; i < sizeof(channels) / sizeof(channels[0]); i++)
    {
        dw_f16_layer(1, 6, 7, channels[i], 3, 3, &dp, 1, 10 + (int32_t)i);
        dw_f16_layer(1, 6, 7, channels[i], 3, 3, &dp, 0, 100 + (int32_t)i);
    }
}

// Kernels 3x3 / 5x5 / 2x3 at strides 1 / 2 / 3, SAME-style padding, tail channel count.
void depthwise_conv_kernel_stride_sweep_f16(void)
{
    const int32_t kernels[][2] = {{3, 3}, {5, 5}, {2, 3}};
    cmsis_nn_dw_conv_params_f16 dp;
    for (size_t k = 0; k < 3; k++)
    {
        for (int32_t stride = 1; stride <= 3; stride++)
        {
            const int32_t kh = kernels[k][0];
            const int32_t kw = kernels[k][1];
            dw_f16_params(&dp, 1, stride, stride, kh / 2, kw / 2, 1, 1);
            dw_f16_layer(1, 9, 8, 5, kh, kw, &dp, 1, 20 + (int32_t)k * 3 + stride);
            dw_f16_layer(1, 9, 8, 8, kh, kw, &dp, 1, 30 + (int32_t)k * 3 + stride);
        }
    }
}

// VALID (no padding) 5x5 at stride 3, and stride 2 with an asymmetric SAME pad on a 2x3 kernel.
void depthwise_conv_valid_and_asymmetric_f16(void)
{
    cmsis_nn_dw_conv_params_f16 dp;
    dw_f16_params(&dp, 1, 3, 3, 0, 0, 1, 1);
    dw_f16_layer(1, 16, 16, 8, 5, 5, &dp, 1, 40);
    dw_f16_params(&dp, 1, 2, 1, 0, 2, 1, 1);
    dw_f16_layer(1, 10, 9, 5, 2, 3, &dp, 1, 41);
}

// Padding wider than the kernel: border output pixels see no input tap at all (bias only), interior columns
// start several pixels in. Stride 1 and stride 2, and a pad wide enough that whole rows are empty.
void depthwise_conv_pad_wider_than_kernel_f16(void)
{
    cmsis_nn_dw_conv_params_f16 dp;
    dw_f16_params(&dp, 1, 1, 1, 4, 4, 1, 1);
    dw_f16_layer(1, 5, 6, 8, 3, 3, &dp, 1, 50);
    dw_f16_layer(1, 5, 6, 5, 3, 3, &dp, 0, 51);
    dw_f16_params(&dp, 1, 2, 2, 5, 5, 1, 1);
    dw_f16_layer(1, 7, 7, 9, 3, 3, &dp, 1, 52);
    dw_f16_params(&dp, 1, 1, 1, 6, 1, 1, 1);
    dw_f16_layer(1, 4, 8, 4, 3, 3, &dp, 1, 53);
}

// Dilation 2 (previously the scalar generic route): SAME pad 2 at stride 1, and pad 0 at stride 2.
void depthwise_conv_dilation_f16(void)
{
    cmsis_nn_dw_conv_params_f16 dp;
    dw_f16_params(&dp, 1, 1, 1, 2, 2, 2, 2);
    dw_f16_layer(1, 9, 10, 9, 3, 3, &dp, 1, 60);
    dw_f16_params(&dp, 1, 2, 2, 0, 0, 2, 2);
    dw_f16_layer(1, 12, 11, 5, 3, 3, &dp, 1, 61);
    dw_f16_params(&dp, 1, 1, 2, 3, 1, 2, 1);
    dw_f16_layer(1, 8, 9, 4, 3, 2, &dp, 0, 62);
}

// Batch 2 through the KWS geometry and through a SAME stride-1 layer.
void depthwise_conv_batch2_f16(void)
{
    cmsis_nn_dw_conv_params_f16 dp;
    dw_f16_params(&dp, 1, 2, 2, 0, 0, 1, 1);
    dw_f16_layer(2, 16, 16, 8, 3, 3, &dp, 1, 70);
    dw_f16_params(&dp, 1, 1, 1, 1, 1, 1, 1);
    dw_f16_layer(2, 5, 6, 5, 3, 3, &dp, 1, 71);
}

// ch_mult 2 stays on the generic route (the direct kernel is ch_mult == 1 only).
void depthwise_conv_ch_mult2_f16(void)
{
    cmsis_nn_dw_conv_params_f16 dp;
    dw_f16_params(&dp, 2, 1, 1, 1, 1, 1, 1);
    dw_f16_layer(1, 6, 7, 3, 3, 3, &dp, 1, 80);
    dw_f16_params(&dp, 2, 2, 2, 0, 0, 1, 1);
    dw_f16_layer(1, 16, 16, 8, 3, 3, &dp, 1, 81);
}

// Bit-pattern classification, immune to -ffinite-math-only folding.
static int32_t dw_f16_is_nan(float16_t v)
{
    uint16_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7FFFu) > 0x7C00u;
}

// Non-finite inputs on the KWS layer, main's route semantics kept (#448): +Inf / -Inf taps clamp to the
// activation bounds on every leg; a NaN tap resolves to activation.min on the MVE leg (vmaxnm/vminnm suppress
// NaN, as arm_nn_vector_clamp_f16 did) and propagates on the scalar leg (arm_nn_clamp_scalar_f16, as
// before). Pixels whose window does not contain the special tap are unaffected.
void depthwise_conv_nonfinite_inputs_f16(void)
{
    const cmsis_nn_dims in = {1, 16, 16, 8};
    const cmsis_nn_dims flt = {1, 3, 3, 8};
    const cmsis_nn_dims out = {1, 8, 8, 8};
    const cmsis_nn_dims bias_dims = {1, 1, 1, 8};
    static float16_t x[16 * 16 * 8];
    static float16_t w[72];
    static float16_t bias[8];
    static float16_t y_clean[8 * 8 * 8];
    static float16_t y[8 * 8 * 8];
    cmsis_nn_dw_conv_params_f16 dp;
    cmsis_nn_context ctx = {NULL, 0};
    const float16_t inf = (float16_t)INFINITY;
    const float16_t nan_v = (float16_t)NAN;

    for (int32_t i = 0; i < 16 * 16 * 8; i++)
    {
        x[i] = dw_f16_value(i, 90);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = dw_f16_value(i, 91);
        if ((float32_t)w[i] == 0.0f)
        {
            w[i] = (float16_t)0.03125f;
        }
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = dw_f16_value(i, 92);
    }
    dw_f16_params(&dp, 1, 2, 2, 0, 0, 1, 1);
    dp.activation.min = (float16_t)-6.0f;
    dp.activation.max = (float16_t)6.0f;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_f16(&ctx, &dp, &in, x, &flt, w, &bias_dims, bias, &out, y_clean));

    // Input pixel (5, 7) channel 3 is a tap of output (2, 3) only (stride 2: rows 4..6, cols 6..8).
    const int32_t px = (5 * 16 + 7) * 8 + 3;
    const int32_t hit = (2 * 8 + 3) * 8 + 3;
    const float16_t w_hit = w[(1 * 3 + 1) * 8 + 3];
    const float16_t specials[2] = {inf, (float16_t)-INFINITY};
    for (int32_t s = 0; s < 2; s++)
    {
        x[px] = specials[s];
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_depthwise_conv_wrapper_f16(&ctx, &dp, &in, x, &flt, w, &bias_dims, bias, &out, y));
        const float32_t expect = (((float32_t)specials[s] > 0.0f) == ((float32_t)w_hit > 0.0f)) ? 6.0f : -6.0f;
        TEST_ASSERT_EQUAL_FLOAT(expect, (float32_t)y[hit]);
        y[hit] = y_clean[hit];
        TEST_ASSERT_EQUAL_MEMORY(y_clean, y, sizeof(y));
    }

    x[px] = nan_v;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_f16(&ctx, &dp, &in, x, &flt, w, &bias_dims, bias, &out, y));
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    TEST_ASSERT_FALSE(dw_f16_is_nan(y[hit]));
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, (float32_t)y[hit]);
#else
    TEST_ASSERT_TRUE(dw_f16_is_nan(y[hit]));
#endif
    y[hit] = y_clean[hit];
    TEST_ASSERT_EQUAL_MEMORY(y_clean, y, sizeof(y));
    x[px] = dw_f16_value(px, 90);

    // A NaN weight poisons every pixel whose window contains that tap in range; the rest are unaffected.
    w[(0 * 3 + 0) * 8 + 5] = nan_v;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_f16(&ctx, &dp, &in, x, &flt, w, &bias_dims, bias, &out, y));
    for (int32_t oy = 0; oy < 8; oy++)
    {
        for (int32_t ox = 0; ox < 8; ox++)
        {
            for (int32_t c = 0; c < 8; c++)
            {
                const int32_t idx = (oy * 8 + ox) * 8 + c;
                if (c == 5)
                {
                    // Tap (0, 0) is always in range at pad 0.
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                    TEST_ASSERT_FALSE(dw_f16_is_nan(y[idx]));
                    TEST_ASSERT_EQUAL_FLOAT(-6.0f, (float32_t)y[idx]);
#else
                    TEST_ASSERT_TRUE(dw_f16_is_nan(y[idx]));
#endif
                }
                else
                {
                    TEST_ASSERT_EQUAL_FLOAT((float32_t)y_clean[idx], (float32_t)y[idx]);
                }
            }
        }
    }
}
