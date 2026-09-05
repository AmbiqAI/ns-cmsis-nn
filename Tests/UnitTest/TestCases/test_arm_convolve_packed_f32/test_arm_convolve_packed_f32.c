/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unity.h>

#include "../Common/float_packed_test_utils.h"

// Deterministic data in [-1, 1) that survives a float16 round trip.
static float32_t conv_f32_value(int32_t i, int32_t seed)
{
    return (float32_t)((float32_t)(((i * 37 + seed * 11) % 64) - 32) / 32.0f);
}

// Plain NHWC convolution against OHWI weights, the reference every dispatch path must agree with.
static void conv_f32_reference(const cmsis_nn_conv_params_f32 *cp,
                               const cmsis_nn_dims *in,
                               const float32_t *x,
                               const cmsis_nn_dims *flt,
                               const float32_t *w,
                               const float32_t *bias,
                               const cmsis_nn_dims *out,
                               float32_t *y)
{
    for (int32_t b = 0; b < out->n; b++)
    {
        for (int32_t oy = 0; oy < out->h; oy++)
        {
            for (int32_t ox = 0; ox < out->w; ox++)
            {
                for (int32_t oc = 0; oc < out->c; oc++)
                {
                    float32_t acc = bias ? (float32_t)bias[oc] : 0.0f;
                    for (int32_t ky = 0; ky < flt->h; ky++)
                    {
                        const int32_t iy = oy * cp->stride.h - cp->padding.h + ky * cp->dilation.h;
                        if (iy < 0 || iy >= in->h)
                        {
                            continue;
                        }
                        for (int32_t kx = 0; kx < flt->w; kx++)
                        {
                            const int32_t ix = ox * cp->stride.w - cp->padding.w + kx * cp->dilation.w;
                            if (ix < 0 || ix >= in->w)
                            {
                                continue;
                            }
                            for (int32_t ic = 0; ic < in->c; ic++)
                            {
                                acc += (float32_t)x[((b * in->h + iy) * in->w + ix) * in->c + ic] *
                                    (float32_t)w[((oc * flt->h + ky) * flt->w + kx) * in->c + ic];
                            }
                        }
                    }
                    acc = fminf(fmaxf(acc, (float32_t)cp->activation.min), (float32_t)cp->activation.max);
                    y[((b * out->h + oy) * out->w + ox) * out->c + oc] = acc;
                }
            }
        }
    }
}

// Guard region appended to every buffer the kernel may write: the scratch sized exactly by the sizer and the
// output. The 1xN pack-rows helper used to zero-fill past the scratch when padding.w exceeded the kernel
// width while still producing correct values, so checking the values alone would not pin that fix.
#define CONV_GUARD_BYTES 64
#define CONV_GUARD_FILL 0xA5

// malloc `size` bytes followed by CONV_GUARD_BYTES of sentinel.
static void *conv_alloc_guarded(size_t size)
{
    uint8_t *p = (uint8_t *)malloc(size + CONV_GUARD_BYTES);
    if (p != NULL)
    {
        memset(p + size, CONV_GUARD_FILL, CONV_GUARD_BYTES);
    }
    return p;
}

// Fail if any byte of the guard region that follows the first `size` bytes of `p` has been overwritten.
static void conv_assert_guard_intact(const void *p, size_t size, const char *what)
{
    const uint8_t *guard = (const uint8_t *)p + size;
    for (size_t i = 0; i < CONV_GUARD_BYTES; i++)
    {
        TEST_ASSERT_EQUAL_HEX8_MESSAGE(CONV_GUARD_FILL, guard[i], what);
    }
}

// Run arm_convolve_wrapper_f32 on a layer with ctx sized exactly by the sizer (or no ctx at all), check that
// neither the scratch nor the output guard was touched, and compare every output element against the reference.
static void conv_f32_check(const cmsis_nn_conv_params_f32 *cp,
                           const cmsis_nn_dims *in,
                           const float32_t *x,
                           const cmsis_nn_dims *flt,
                           const float32_t *w_kernel,
                           const float32_t *w_ohwi,
                           const float32_t *bias,
                           const cmsis_nn_dims *out,
                           int32_t use_ctx)
{
    const int32_t out_size = out->n * out->h * out->w * out->c;
    const size_t out_bytes = (size_t)out_size * sizeof(float32_t);
    float32_t *ref = (float32_t *)malloc((size_t)out_size * sizeof(float32_t));
    float32_t *y = (float32_t *)conv_alloc_guarded(out_bytes);
    cmsis_nn_context ctx = {NULL, 0};
    TEST_ASSERT_NOT_NULL(ref);
    TEST_ASSERT_NOT_NULL(y);

    if (use_ctx)
    {
        const int32_t size = arm_convolve_wrapper_f32_get_buffer_size(cp, in, flt, out);
        TEST_ASSERT_TRUE(size > 0);
        ctx.buf = conv_alloc_guarded((size_t)size);
        ctx.size = size;
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }

    conv_f32_reference(cp, in, x, flt, w_ohwi, bias, out, ref);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_wrapper_f32(&ctx, cp, in, x, flt, w_kernel, NULL, bias, out, y));
    if (use_ctx)
    {
        conv_assert_guard_intact(ctx.buf, (size_t)ctx.size, "kernel wrote past the sizer-sized scratch buffer");
    }
    conv_assert_guard_intact(y, out_bytes, "kernel wrote past the output buffer");
    for (int32_t i = 0; i < out_size; i++)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, ref[i], (float32_t)y[i]);
    }

    free(ctx.buf);
    free(y);
    free(ref);
}

static void conv_f32_params(cmsis_nn_conv_params_f32 *cp, int32_t pad_h, int32_t pad_w, int32_t packed)
{
    memset(cp, 0, sizeof(*cp));
    cp->stride.h = 1;
    cp->stride.w = 1;
    cp->padding.h = pad_h;
    cp->padding.w = pad_w;
    cp->dilation.h = 1;
    cp->dilation.w = 1;
    cp->activation.min = (float32_t)-1.0e4f;
    cp->activation.max = (float32_t)1.0e4f;
    cp->weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
}

// 3x3, in_c = 4, out_c = 8 on a 4x4 input (in_c = 4 is one full vector, so the direct small-C kernel does not
// claim it on MVE): with scratch this takes the patch-GEMM path, without it the generic fallback. Both must honour
// NT_N_PACKED; the fallback used to read packed weights as OHWI.
void convolve_packed_3x3_f32(void)
{
    const cmsis_nn_dims in = {1, 4, 4, 4};
    const cmsis_nn_dims flt = {8, 3, 3, 4};
    const cmsis_nn_dims out = {1, 4, 4, 8};
    float32_t x[64];
    float32_t w[288];
    float32_t bias[8];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 64; i++)
    {
        x[i] = conv_f32_value(i, 1);
    }
    for (int32_t i = 0; i < 288; i++)
    {
        w[i] = conv_f32_value(i, 2);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f32_value(i, 3);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 8, 36);

    conv_f32_params(&cp, 1, 1, 0);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f32_params(&cp, 1, 1, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// 1xN with batch 2 (so the batch-1 k3 specialization does not claim it) and out_c = 5, which is not a
// whole packed block: the 1xN path used to hand packed weights to the OHWI matmul.
void convolve_packed_1xn_f32(void)
{
    const cmsis_nn_dims in = {2, 1, 10, 4};
    const cmsis_nn_dims flt = {5, 1, 3, 4};
    const cmsis_nn_dims out = {2, 1, 10, 5};
    float32_t x[80];
    float32_t w[60];
    float32_t bias[5];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 80; i++)
    {
        x[i] = conv_f32_value(i, 4);
    }
    for (int32_t i = 0; i < 60; i++)
    {
        w[i] = conv_f32_value(i, 5);
    }
    for (int32_t i = 0; i < 5; i++)
    {
        bias[i] = conv_f32_value(i, 6);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 5, 12);

    conv_f32_params(&cp, 0, 1, 0);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_params(&cp, 0, 1, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// padding.w larger than the kernel width on the 1xN path, with scratch sized exactly by the sizer: the
// fully padded positions used to zero-fill more than one patch row and write past the buffer.
void convolve_1xn_pad_wider_than_kernel_f32(void)
{
    const cmsis_nn_dims in = {1, 1, 10, 4};
    const cmsis_nn_dims flt = {2, 1, 3, 4};
    const cmsis_nn_dims out = {1, 1, 18, 2};
    float32_t x[40];
    float32_t w[24];
    float32_t bias[2];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 40; i++)
    {
        x[i] = conv_f32_value(i, 7);
    }
    for (int32_t i = 0; i < 24; i++)
    {
        w[i] = conv_f32_value(i, 8);
    }
    for (int32_t i = 0; i < 2; i++)
    {
        bias[i] = conv_f32_value(i, 9);
    }

    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 2, 12);

    conv_f32_params(&cp, 0, 5, 0);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_params(&cp, 0, 5, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);

    free(w_packed);
}

// Stride-2 3x3 with a single input channel (patch length 9). Route: on MVE the direct small-C kernel (in_c < 4),
// with or without scratch; on non-MVE builds patch-GEMM with scratch (no patch-length floor since #417) and the
// generic fallback without.
void convolve_small_k_3x3_s2_f32(void)
{
    const cmsis_nn_dims in = {1, 32, 32, 1};
    const cmsis_nn_dims flt = {8, 3, 3, 1};
    const cmsis_nn_dims out = {1, 16, 16, 8};
    static float32_t x[1024];
    float32_t w[72];
    float32_t bias[8];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 1024; i++)
    {
        x[i] = conv_f32_value(i, 10);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = conv_f32_value(i, 11);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f32_value(i, 12);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 8, 9);

    conv_f32_params(&cp, 1, 1, 0);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f32_params(&cp, 1, 1, 1);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Patch length 18 with only 5 filters (in_c = 2, below MIN_OC). Route: on MVE the direct small-C kernel, whose
// last output-channel group is partial (5 = 4 + 1); on non-MVE builds the generic fallback with or without
// scratch, reading the partial packed block (4-lane blocks, 5 live) lane-wise.
void convolve_small_k_few_filters_f32(void)
{
    const cmsis_nn_dims in = {1, 6, 6, 2};
    const cmsis_nn_dims flt = {5, 3, 3, 2};
    const cmsis_nn_dims out = {1, 6, 6, 5};
    float32_t x[72];
    float32_t w[90];
    float32_t bias[5];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 72; i++)
    {
        x[i] = conv_f32_value(i, 13);
    }
    for (int32_t i = 0; i < 90; i++)
    {
        w[i] = conv_f32_value(i, 14);
    }
    for (int32_t i = 0; i < 5; i++)
    {
        bias[i] = conv_f32_value(i, 15);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 5, 18);

    conv_f32_params(&cp, 1, 1, 0);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f32_params(&cp, 1, 1, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// 5x5 single channel (patch length 25). Route: on MVE the direct small-C kernel; on non-MVE builds patch-GEMM
// with scratch (as before #417, the floor was 16) and the generic fallback without.
void convolve_5x5_single_channel_f32(void)
{
    const cmsis_nn_dims in = {1, 12, 12, 1};
    const cmsis_nn_dims flt = {8, 5, 5, 1};
    const cmsis_nn_dims out = {1, 12, 12, 8};
    float32_t x[144];
    float32_t w[200];
    float32_t bias[8];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 144; i++)
    {
        x[i] = conv_f32_value(i, 16);
    }
    for (int32_t i = 0; i < 200; i++)
    {
        w[i] = conv_f32_value(i, 17);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f32_value(i, 18);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 8, 25);

    conv_f32_params(&cp, 2, 2, 0);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f32_params(&cp, 2, 2, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Direct small-C kernel (3 input channels, output_w = 9, not a whole lane group): 3x3 with dilation 2 and padding 2,
// six filters so the last output-channel group is partial. OHWI and NT_N_PACKED, with a (sizer-sized, untouched)
// scratch and without one. #417
void convolve_small_c_dilated_f32(void)
{
    const cmsis_nn_dims in = {1, 7, 9, 3};
    const cmsis_nn_dims flt = {6, 3, 3, 3};
    const cmsis_nn_dims out = {1, 7, 9, 6};
    float32_t x[189];
    float32_t w[162];
    float32_t bias[6];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 189; i++)
    {
        x[i] = conv_f32_value(i, 19);
    }
    for (int32_t i = 0; i < 162; i++)
    {
        w[i] = conv_f32_value(i, 20);
    }
    for (int32_t i = 0; i < 6; i++)
    {
        bias[i] = conv_f32_value(i, 21);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 6, 27);

    conv_f32_params(&cp, 2, 2, 0);
    cp.dilation.h = 2;
    cp.dilation.w = 2;
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f32_params(&cp, 2, 2, 1);
    cp.dilation.h = 2;
    cp.dilation.w = 2;
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Batch 2 through the direct small-C kernel: single channel, stride 2, padding 1, output_w = 5.
void convolve_small_c_batch2_f32(void)
{
    const cmsis_nn_dims in = {2, 9, 9, 1};
    const cmsis_nn_dims flt = {8, 3, 3, 1};
    const cmsis_nn_dims out = {2, 5, 5, 8};
    float32_t x[162];
    float32_t w[72];
    float32_t bias[8];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 162; i++)
    {
        x[i] = conv_f32_value(i, 22);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = conv_f32_value(i, 23);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f32_value(i, 24);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 8, 9);

    conv_f32_params(&cp, 1, 1, 0);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f32_params(&cp, 1, 1, 1);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// in_c = 5, one full vector or more, so the direct small-C kernel never claims this on MVE. Route on every build:
// patch-GEMM with scratch (out_c = 13 >= MIN_OC, 36 positions), the generic fallback without. 13 filters make the
// last packed block partial (13 = 3 * 4 + 1), so the fallback's predicated last-block load and the OHWI accumulator are
// both pinned here, OHWI and NT_N_PACKED. The patch is 45 long and these paths accumulate in f32, so the data
// are dyadic (inputs k/8, weights k/8): every product is a multiple of 1/64 and every partial sum is exact,
// which keeps the comparison independent of summation order.
static float32_t conv_f32_dyadic_input(int32_t i, int32_t seed)
{
    return (float32_t)((float32_t)(((i * 37 + seed * 11) % 16) - 8) / 8.0f);
}

static float32_t conv_f32_dyadic_weight(int32_t i, int32_t seed)
{
    return (float32_t)((float32_t)(((i * 53 + seed * 7) % 8) - 4) / 8.0f);
}

void convolve_full_c_partial_block_f32(void)
{
    const cmsis_nn_dims in = {1, 6, 6, 5};
    const cmsis_nn_dims flt = {13, 3, 3, 5};
    const cmsis_nn_dims out = {1, 6, 6, 13};
    float32_t x[180];
    static float32_t w[585];
    float32_t bias[13];
    cmsis_nn_conv_params_f32 cp;

    for (int32_t i = 0; i < 180; i++)
    {
        x[i] = conv_f32_dyadic_input(i, 25);
    }
    for (int32_t i = 0; i < 585; i++)
    {
        w[i] = conv_f32_dyadic_weight(i, 26);
    }
    for (int32_t i = 0; i < 13; i++)
    {
        bias[i] = conv_f32_dyadic_input(i, 27);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 13, 45);

    conv_f32_params(&cp, 1, 1, 0);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f32_params(&cp, 1, 1, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Contiguous interior loads must not read past the input tensor. Single-channel 3x9 input, 3x3, stride 2,
// no padding, output_w = 4 (one full lane group): the last tap column of the group starts at x = 2 and a vld2q
// there reads 2 * LANES elements, one past the row -- and this is the last row of the only batch, so past the
// tensor. The input is an exact-size heap allocation so host ASan reports such a read; on the FVP the values
// still have to match. #417
void convolve_small_c_no_overread_f32(void)
{
    const cmsis_nn_dims in = {1, 3, 9, 1};
    const cmsis_nn_dims flt = {8, 3, 3, 1};
    const cmsis_nn_dims out = {1, 1, 4, 8};
    float32_t *x = (float32_t *)malloc(27 * sizeof(float32_t));
    float32_t w[72];
    float32_t bias[8];
    cmsis_nn_conv_params_f32 cp;

    TEST_ASSERT_NOT_NULL(x);
    for (int32_t i = 0; i < 27; i++)
    {
        x[i] = conv_f32_value(i, 28);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = conv_f32_value(i, 29);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f32_value(i, 30);
    }
    float32_t *w_packed = pack_rhs_nt_n_from_nt_t_f32(w, 8, 9);

    conv_f32_params(&cp, 0, 0, 0);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f32_params(&cp, 0, 0, 1);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f32_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
    free(x);
}
