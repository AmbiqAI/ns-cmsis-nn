/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unity.h>

#include "../Common/float_packed_test_utils.h"

// Deterministic data in [-1, 1) that survives a float16 round trip.
static float16_t conv_f16_value(int32_t i, int32_t seed)
{
    return (float16_t)((float32_t)(((i * 37 + seed * 11) % 64) - 32) / 32.0f);
}

// Plain NHWC convolution against OHWI weights, the reference every dispatch path must agree with.
static void conv_f16_reference(const cmsis_nn_conv_params_f16 *cp,
                               const cmsis_nn_dims *in,
                               const float16_t *x,
                               const cmsis_nn_dims *flt,
                               const float16_t *w,
                               const float16_t *bias,
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

// Run arm_convolve_wrapper_f16 on a layer with ctx sized exactly by the sizer (or no ctx at all), check that
// neither the scratch nor the output guard was touched, and compare every output element against the reference.
static void conv_f16_check(const cmsis_nn_conv_params_f16 *cp,
                           const cmsis_nn_dims *in,
                           const float16_t *x,
                           const cmsis_nn_dims *flt,
                           const float16_t *w_kernel,
                           const float16_t *w_ohwi,
                           const float16_t *bias,
                           const cmsis_nn_dims *out,
                           int32_t use_ctx)
{
    const int32_t out_size = out->n * out->h * out->w * out->c;
    const size_t out_bytes = (size_t)out_size * sizeof(float16_t);
    float32_t *ref = (float32_t *)malloc((size_t)out_size * sizeof(float32_t));
    float16_t *y = (float16_t *)conv_alloc_guarded(out_bytes);
    cmsis_nn_context ctx = {NULL, 0};
    TEST_ASSERT_NOT_NULL(ref);
    TEST_ASSERT_NOT_NULL(y);

    if (use_ctx)
    {
        const int32_t size = arm_convolve_wrapper_f16_get_buffer_size(cp, in, flt, out);
        TEST_ASSERT_TRUE(size > 0);
        ctx.buf = conv_alloc_guarded((size_t)size);
        ctx.size = size;
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }

    conv_f16_reference(cp, in, x, flt, w_ohwi, bias, out, ref);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_wrapper_f16(&ctx, cp, in, x, flt, w_kernel, NULL, bias, out, y));
    if (use_ctx)
    {
        conv_assert_guard_intact(ctx.buf, (size_t)ctx.size, "kernel wrote past the sizer-sized scratch buffer");
    }
    conv_assert_guard_intact(y, out_bytes, "kernel wrote past the output buffer");
    for (int32_t i = 0; i < out_size; i++)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.5e-2f, ref[i], (float32_t)y[i]);
    }

    free(ctx.buf);
    free(y);
    free(ref);
}

static void conv_f16_params(cmsis_nn_conv_params_f16 *cp, int32_t pad_h, int32_t pad_w, int32_t packed)
{
    memset(cp, 0, sizeof(*cp));
    cp->stride.h = 1;
    cp->stride.w = 1;
    cp->padding.h = pad_h;
    cp->padding.w = pad_w;
    cp->dilation.h = 1;
    cp->dilation.w = 1;
    cp->activation.min = (float16_t)-1.0e4f;
    cp->activation.max = (float16_t)1.0e4f;
    cp->weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
}

// 3x3, in_c = 4, out_c = 12 on a 4x4 input (so the second packed block is exercised). Route: on MVE the direct
// small-C kernel (in_c < 8) with or without scratch; on non-MVE builds patch-GEMM with scratch and the generic
// fallback without. Both must honour NT_N_PACKED; the fallback used to read packed weights as OHWI.
void convolve_packed_3x3_f16(void)
{
    const cmsis_nn_dims in = {1, 4, 4, 4};
    const cmsis_nn_dims flt = {12, 3, 3, 4};
    const cmsis_nn_dims out = {1, 4, 4, 12};
    float16_t x[64];
    float16_t w[432];
    float16_t bias[12];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 64; i++)
    {
        x[i] = conv_f16_value(i, 1);
    }
    for (int32_t i = 0; i < 432; i++)
    {
        w[i] = conv_f16_value(i, 2);
    }
    for (int32_t i = 0; i < 12; i++)
    {
        bias[i] = conv_f16_value(i, 3);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 12, 36);

    conv_f16_params(&cp, 1, 1, 0);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 1, 1, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// 1xN with batch 2 (so the batch-1 k3 specialization does not claim it) and out_c = 5, which is not a
// whole packed block: the 1xN path used to hand packed weights to the OHWI matmul.
void convolve_packed_1xn_f16(void)
{
    const cmsis_nn_dims in = {2, 1, 10, 4};
    const cmsis_nn_dims flt = {5, 1, 3, 4};
    const cmsis_nn_dims out = {2, 1, 10, 5};
    float16_t x[80];
    float16_t w[60];
    float16_t bias[5];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 80; i++)
    {
        x[i] = conv_f16_value(i, 4);
    }
    for (int32_t i = 0; i < 60; i++)
    {
        w[i] = conv_f16_value(i, 5);
    }
    for (int32_t i = 0; i < 5; i++)
    {
        bias[i] = conv_f16_value(i, 6);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 5, 12);

    conv_f16_params(&cp, 0, 1, 0);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_params(&cp, 0, 1, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// padding.w larger than the kernel width on the 1xN path, with scratch sized exactly by the sizer: the
// fully padded positions used to zero-fill more than one patch row and write past the buffer.
void convolve_1xn_pad_wider_than_kernel_f16(void)
{
    const cmsis_nn_dims in = {1, 1, 10, 4};
    const cmsis_nn_dims flt = {2, 1, 3, 4};
    const cmsis_nn_dims out = {1, 1, 18, 2};
    float16_t x[40];
    float16_t w[24];
    float16_t bias[2];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 40; i++)
    {
        x[i] = conv_f16_value(i, 7);
    }
    for (int32_t i = 0; i < 24; i++)
    {
        w[i] = conv_f16_value(i, 8);
    }
    for (int32_t i = 0; i < 2; i++)
    {
        bias[i] = conv_f16_value(i, 9);
    }

    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 2, 12);

    conv_f16_params(&cp, 0, 5, 0);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_params(&cp, 0, 5, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);

    free(w_packed);
}

// Rebuild a float16 from bits held in a volatile so the NaN input is created at run time. The harness
// builds this TU with -fno-finite-math-only (Tests/UnitTest/CMakeLists.txt); the staging is
// defense-in-depth for a standalone -Ofast build, where the implied -ffinite-math-only licenses
// constant-folding arithmetic on a compile-time NaN so the kernel is never handed one at all.
static float16_t conv_f16_from_bits(volatile const uint16_t *bits)
{
    const uint16_t b = *bits;
    float16_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

// Bit-pattern NaN test: all-ones exponent, non-zero mantissa. isnan() would also work under the
// harness flags; the bit-pattern check survives a standalone -Ofast build of this TU.
static inline bool conv_f16_bits_are_nan(float16_t x)
{
    uint16_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (uint16_t)(bits & 0x7FFFu) > 0x7C00u;
}

// NaN through arm_nn_mat_mult_nt_n_packed_f16's output clamp, the packed f16 matmul entry the packed
// conv paths land on. On the scalar (non-MVE) build path the clamp is the bit-classified clamp of #380,
// so the NaN row must come back NaN at every optimization level; the MVE path clamps with
// vmaxnmq/vminnmq and no NaN restore, so there the NaN resolves to the lower clamp bound instead
// (documented on the declaration). The finite row must clamp normally on both paths.
void convolve_packed_matmul_nan_f16(void)
{
    volatile uint16_t nan_bits = 0x7E00u;
    const float16_t nan = conv_f16_from_bits(&nan_bits);
    // lhs: 2 rows, K = 1. Row 0 is NaN, row 1 is finite and overflows the upper bound.
    const float16_t lhs[2] = {nan, (float16_t)8.0f};
    // rhs_packed: one block of 8 packed lanes for K = 1; lanes 0 and 1 are live (rhs_rows = 2).
    float16_t rhs_packed[8] = {(float16_t)1.0f, (float16_t)1.0f};
    float16_t dst[4] = {0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mat_mult_nt_n_packed_f16(lhs, rhs_packed, NULL, dst, 2, 2, 1, 2, (float16_t)-6.0f, (float16_t)6.0f));

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, (float32_t)dst[0]);
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, (float32_t)dst[1]);
#else
    TEST_ASSERT_TRUE_MESSAGE(conv_f16_bits_are_nan(dst[0]), "Expected NaN through the scalar clamp");
    TEST_ASSERT_TRUE_MESSAGE(conv_f16_bits_are_nan(dst[1]), "Expected NaN through the scalar clamp");
#endif
    TEST_ASSERT_EQUAL_FLOAT(6.0f, (float32_t)dst[2]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, (float32_t)dst[3]);
}

// Stride-2 3x3 with a single input channel (patch length 9). Route: on MVE the direct small-C kernel (in_c < 8),
// with or without scratch; on non-MVE builds patch-GEMM with scratch (no patch-length floor since #417) and the
// generic fallback without.
void convolve_small_k_3x3_s2_f16(void)
{
    const cmsis_nn_dims in = {1, 32, 32, 1};
    const cmsis_nn_dims flt = {8, 3, 3, 1};
    const cmsis_nn_dims out = {1, 16, 16, 8};
    static float16_t x[1024];
    float16_t w[72];
    float16_t bias[8];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 1024; i++)
    {
        x[i] = conv_f16_value(i, 10);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = conv_f16_value(i, 11);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f16_value(i, 12);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 8, 9);

    conv_f16_params(&cp, 1, 1, 0);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 1, 1, 1);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Patch length 18 with only 5 filters (in_c = 2, below MIN_OC). Route: on MVE the direct small-C kernel, whose
// last output-channel group is partial (5 = 2 + 2 + 1); on non-MVE builds the generic fallback with or without
// scratch, reading the partial packed block (8-lane blocks, 5 live) lane-wise.
void convolve_small_k_few_filters_f16(void)
{
    const cmsis_nn_dims in = {1, 6, 6, 2};
    const cmsis_nn_dims flt = {5, 3, 3, 2};
    const cmsis_nn_dims out = {1, 6, 6, 5};
    float16_t x[72];
    float16_t w[90];
    float16_t bias[5];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 72; i++)
    {
        x[i] = conv_f16_value(i, 13);
    }
    for (int32_t i = 0; i < 90; i++)
    {
        w[i] = conv_f16_value(i, 14);
    }
    for (int32_t i = 0; i < 5; i++)
    {
        bias[i] = conv_f16_value(i, 15);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 5, 18);

    conv_f16_params(&cp, 1, 1, 0);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 1, 1, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// 5x5 single channel (patch length 25). Route: on MVE the direct small-C kernel; on non-MVE builds patch-GEMM
// with scratch (as before #417, the floor was 16) and the generic fallback without.
void convolve_5x5_single_channel_f16(void)
{
    const cmsis_nn_dims in = {1, 12, 12, 1};
    const cmsis_nn_dims flt = {8, 5, 5, 1};
    const cmsis_nn_dims out = {1, 12, 12, 8};
    float16_t x[144];
    float16_t w[200];
    float16_t bias[8];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 144; i++)
    {
        x[i] = conv_f16_value(i, 16);
    }
    for (int32_t i = 0; i < 200; i++)
    {
        w[i] = conv_f16_value(i, 17);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f16_value(i, 18);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 8, 25);

    conv_f16_params(&cp, 2, 2, 0);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 2, 2, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Direct small-C kernel (5 input channels, output_w = 11, not a whole lane group): 3x3 with dilation 2 and padding 2,
// six filters so the last output-channel group is partial. OHWI and NT_N_PACKED, with a (sizer-sized, untouched)
// scratch and without one. #417
void convolve_small_c_dilated_f16(void)
{
    const cmsis_nn_dims in = {1, 7, 11, 5};
    const cmsis_nn_dims flt = {6, 3, 3, 5};
    const cmsis_nn_dims out = {1, 7, 11, 6};
    float16_t x[385];
    float16_t w[270];
    float16_t bias[6];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 385; i++)
    {
        x[i] = conv_f16_value(i, 19);
    }
    for (int32_t i = 0; i < 270; i++)
    {
        w[i] = conv_f16_value(i, 20);
    }
    for (int32_t i = 0; i < 6; i++)
    {
        bias[i] = conv_f16_value(i, 21);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 6, 45);

    conv_f16_params(&cp, 2, 2, 0);
    cp.dilation.h = 2;
    cp.dilation.w = 2;
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 2, 2, 1);
    cp.dilation.h = 2;
    cp.dilation.w = 2;
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Batch 2 through the direct small-C kernel: single channel, stride 2, padding 1, output_w = 5.
void convolve_small_c_batch2_f16(void)
{
    const cmsis_nn_dims in = {2, 9, 9, 1};
    const cmsis_nn_dims flt = {8, 3, 3, 1};
    const cmsis_nn_dims out = {2, 5, 5, 8};
    float16_t x[162];
    float16_t w[72];
    float16_t bias[8];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 162; i++)
    {
        x[i] = conv_f16_value(i, 22);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = conv_f16_value(i, 23);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f16_value(i, 24);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 8, 9);

    conv_f16_params(&cp, 1, 1, 0);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 1, 1, 1);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// in_c = 9, one full vector or more, so the direct small-C kernel never claims this on MVE. Route on every build:
// patch-GEMM with scratch (out_c = 13 >= MIN_OC, 36 positions), the generic fallback without. 13 filters make the
// last packed block partial (13 = 8 + 5), so the fallback's predicated last-block load and the OHWI accumulator are
// both pinned here, OHWI and NT_N_PACKED. The patch is 81 long and these paths accumulate in f16, so the data
// are dyadic (inputs k/8, weights k/8): every product is a multiple of 1/64 and every partial sum is exact,
// which keeps the comparison independent of summation order.
static float16_t conv_f16_dyadic_input(int32_t i, int32_t seed)
{
    return (float16_t)((float32_t)(((i * 37 + seed * 11) % 16) - 8) / 8.0f);
}

static float16_t conv_f16_dyadic_weight(int32_t i, int32_t seed)
{
    return (float16_t)((float32_t)(((i * 53 + seed * 7) % 8) - 4) / 8.0f);
}

// 1xN with stride 2, batch 2, against the same layer at stride 1: output x of the strided layer reads the same patch
// as output 2x of the unit-stride one and takes the same route, so the two must match bit for bit. Covers the
// no-padding rows read in place through the contiguous-K matmul, which step through the input by stride * in_c.
static void conv_1xn_stride2_case_f16(int32_t in_c, int32_t kw, int32_t out_c, int32_t acc16)
{
    const int32_t in_w = 14;
    const int32_t pad = kw / 2;
    const int32_t out_w1 = in_w + 2 * pad - kw + 1;
    const int32_t out_w2 = (in_w + 2 * pad - kw) / 2 + 1;
    const cmsis_nn_dims in = {2, 1, in_w, in_c};
    const cmsis_nn_dims flt = {out_c, 1, kw, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims out1 = {2, 1, out_w1, out_c};
    const cmsis_nn_dims out2 = {2, 1, out_w2, out_c};
    float16_t *x = (float16_t *)malloc((size_t)2 * in_w * in_c * sizeof(float16_t));
    float16_t *w = (float16_t *)malloc((size_t)out_c * kw * in_c * sizeof(float16_t));
    float16_t *bias = (float16_t *)malloc((size_t)out_c * sizeof(float16_t));
    float16_t *y1 = (float16_t *)malloc((size_t)2 * out_w1 * out_c * sizeof(float16_t));
    float16_t *y2 = (float16_t *)malloc((size_t)2 * out_w2 * out_c * sizeof(float16_t));
    cmsis_nn_conv_params_f16 p;
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(w);
    TEST_ASSERT_NOT_NULL(bias);
    TEST_ASSERT_NOT_NULL(y1);
    TEST_ASSERT_NOT_NULL(y2);

    for (int32_t i = 0; i < 2 * in_w * in_c; i++)
    {
        x[i] = (float16_t)((float)(((i * 37 + 5) % 61) - 30) / 17.0f);
    }
    for (int32_t i = 0; i < out_c * kw * in_c; i++)
    {
        w[i] = (float16_t)((float)(((i * 53 + 7) % 59) - 29) / 113.0f);
    }
    for (int32_t i = 0; i < out_c; i++)
    {
        bias[i] = (float16_t)((float)(i % 7 - 3) / 5.0f);
    }
    memset(&p, 0, sizeof(p));
    p.stride.h = 1;
    p.stride.w = 1;
    p.padding.w = pad;
    p.dilation.h = 1;
    p.dilation.w = 1;
    p.activation.min = (float16_t)-6.0e4f;
    p.activation.max = (float16_t)6.0e4f;
    p.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;
    int32_t size = arm_convolve_1_x_n_f16_get_buffer_size(&p, &in, &flt, &out1, ARM_NN_LAYOUT_NHWC);
    cmsis_nn_context ctx1 = {malloc((size_t)size), size};
    TEST_ASSERT_NOT_NULL(ctx1.buf);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      (acc16 ? arm_convolve_1_x_n_f16_acc16 : arm_convolve_1_x_n_f16)(
                          &ctx1, &p, &in, x, &flt, w, &bias_dims, bias, &out1, y1, ARM_NN_LAYOUT_NHWC));
    p.stride.w = 2;
    size = arm_convolve_1_x_n_f16_get_buffer_size(&p, &in, &flt, &out2, ARM_NN_LAYOUT_NHWC);
    cmsis_nn_context ctx2 = {malloc((size_t)size), size};
    TEST_ASSERT_NOT_NULL(ctx2.buf);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      (acc16 ? arm_convolve_1_x_n_f16_acc16 : arm_convolve_1_x_n_f16)(
                          &ctx2, &p, &in, x, &flt, w, &bias_dims, bias, &out2, y2, ARM_NN_LAYOUT_NHWC));
    for (int32_t b = 0; b < 2; b++)
    {
        for (int32_t ox = 0; ox < out_w2; ox++)
        {
            for (int32_t oc = 0; oc < out_c; oc++)
            {
                uint16_t got;
                uint16_t want;
                memcpy(&got, &y2[(b * out_w2 + ox) * out_c + oc], sizeof(got));
                memcpy(&want, &y1[(b * out_w1 + 2 * ox) * out_c + oc], sizeof(want));
                TEST_ASSERT_EQUAL_HEX32((uint32_t)want, (uint32_t)got);
            }
        }
    }
    free(ctx1.buf);
    free(ctx2.buf);
    free(y2);
    free(y1);
    free(bias);
    free(w);
    free(x);
}

void convolve_1xn_stride2_f16(void)
{
    for (int32_t acc16 = 0; acc16 < 2; acc16++)
    {
        conv_1xn_stride2_case_f16(16, 5, 5, acc16);  /* K = 80, 5 channels: in place */
        conv_1xn_stride2_case_f16(32, 7, 8, acc16);  /* K = 224: in place */
        conv_1xn_stride2_case_f16(40, 7, 13, acc16); /* K = 280: in place */
        conv_1xn_stride2_case_f16(8, 7, 8, acc16);   /* K = 56: strided kernel */
    }
}

// 1xN, batch 2: each batch must match a batch-1 call on that batch's own input, bit for bit. Non-periodic data, so a
// row that reads the wrong batch changes the output.
static void conv_1xn_batch2_case_f16(int32_t in_c, int32_t kw, int32_t out_c)
{
    const int32_t in_w = 14;
    const int32_t pad = kw / 2;
    const cmsis_nn_dims in2 = {2, 1, in_w, in_c};
    const cmsis_nn_dims in1 = {1, 1, in_w, in_c};
    const cmsis_nn_dims flt = {out_c, 1, kw, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims out2 = {2, 1, in_w, out_c};
    const cmsis_nn_dims out1 = {1, 1, in_w, out_c};
    const int32_t in_size = in_w * in_c;
    const int32_t out_size = in_w * out_c;
    float16_t *x = (float16_t *)malloc((size_t)2 * in_size * sizeof(float16_t));
    float16_t *w = (float16_t *)malloc((size_t)out_c * kw * in_c * sizeof(float16_t));
    float16_t *bias = (float16_t *)malloc((size_t)out_c * sizeof(float16_t));
    float16_t *y2 = (float16_t *)malloc((size_t)2 * out_size * sizeof(float16_t));
    float16_t *y1 = (float16_t *)malloc((size_t)out_size * sizeof(float16_t));
    cmsis_nn_conv_params_f16 p;
    uint32_t seed = 61u;
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(w);
    TEST_ASSERT_NOT_NULL(bias);
    TEST_ASSERT_NOT_NULL(y2);
    TEST_ASSERT_NOT_NULL(y1);

    for (int32_t i = 0; i < 2 * in_size; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        x[i] = (float16_t)((float)((int32_t)(seed >> 9) % 2001 - 1000) / 1000.0f);
    }
    for (int32_t i = 0; i < out_c * kw * in_c; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        w[i] = (float16_t)((float)((int32_t)(seed >> 9) % 2001 - 1000) / 4000.0f);
    }
    for (int32_t i = 0; i < out_c; i++)
    {
        bias[i] = (float16_t)((float)(i % 7 - 3) / 5.0f);
    }
    memset(&p, 0, sizeof(p));
    p.stride.h = 1;
    p.stride.w = 1;
    p.padding.w = pad;
    p.dilation.h = 1;
    p.dilation.w = 1;
    p.activation.min = (float16_t)-6.0e4f;
    p.activation.max = (float16_t)6.0e4f;
    p.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;
    const int32_t size = arm_convolve_1_x_n_f16_get_buffer_size(&p, &in2, &flt, &out2, ARM_NN_LAYOUT_NHWC);
    cmsis_nn_context ctx = {malloc((size_t)size), size};
    TEST_ASSERT_NOT_NULL(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_1_x_n_f16(&ctx, &p, &in2, x, &flt, w,
                                             &bias_dims, bias, &out2, y2, ARM_NN_LAYOUT_NHWC));
    for (int32_t b = 0; b < 2; b++)
    {
        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_SUCCESS,
            arm_convolve_1_x_n_f16(&ctx, &p, &in1, x + b * in_size, &flt, w,
                                   &bias_dims, bias, &out1, y1, ARM_NN_LAYOUT_NHWC));
        for (int32_t i = 0; i < out_size; i++)
        {
            uint16_t got;
            uint16_t want;
            memcpy(&got, &y2[b * out_size + i], sizeof(got));
            memcpy(&want, &y1[i], sizeof(want));
            TEST_ASSERT_EQUAL_HEX32((uint32_t)want, (uint32_t)got);
        }
    }
    free(ctx.buf);
    free(y1);
    free(y2);
    free(bias);
    free(w);
    free(x);
}

void convolve_1xn_batch2_f16(void)
{
    conv_1xn_batch2_case_f16(16, 5, 5);  /* K = 80, 5 channels: in place */
    conv_1xn_batch2_case_f16(32, 7, 8);  /* K = 224: in place */
    conv_1xn_batch2_case_f16(8, 7, 8);   /* K = 56: strided kernel */
}

void convolve_full_c_partial_block_f16(void)
{
    const cmsis_nn_dims in = {1, 6, 6, 9};
    const cmsis_nn_dims flt = {13, 3, 3, 9};
    const cmsis_nn_dims out = {1, 6, 6, 13};
    float16_t x[324];
    static float16_t w[1053];
    float16_t bias[13];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 324; i++)
    {
        x[i] = conv_f16_dyadic_input(i, 25);
    }
    for (int32_t i = 0; i < 1053; i++)
    {
        w[i] = conv_f16_dyadic_weight(i, 26);
    }
    for (int32_t i = 0; i < 13; i++)
    {
        bias[i] = conv_f16_dyadic_input(i, 27);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 13, 81);

    conv_f16_params(&cp, 1, 1, 0);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 1, 1, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Contiguous interior loads must not read past the input tensor. Single-channel 3x17 input, 3x3, stride 2,
// no padding, output_w = 8 (one full lane group): the last tap column of the group starts at x = 2 and a vld2q
// there reads 2 * LANES elements, one past the row -- and this is the last row of the only batch, so past the
// tensor. The input is an exact-size heap allocation so host ASan reports such a read; on the FVP the values
// still have to match. #417
void convolve_small_c_no_overread_f16(void)
{
    const cmsis_nn_dims in = {1, 3, 17, 1};
    const cmsis_nn_dims flt = {8, 3, 3, 1};
    const cmsis_nn_dims out = {1, 1, 8, 8};
    float16_t *x = (float16_t *)malloc(51 * sizeof(float16_t));
    float16_t w[72];
    float16_t bias[8];
    cmsis_nn_conv_params_f16 cp;

    TEST_ASSERT_NOT_NULL(x);
    for (int32_t i = 0; i < 51; i++)
    {
        x[i] = conv_f16_value(i, 28);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = conv_f16_value(i, 29);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f16_value(i, 30);
    }
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 8, 9);

    conv_f16_params(&cp, 0, 0, 0);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 0, 0, 1);
    cp.stride.h = 2;
    cp.stride.w = 2;
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
    free(x);
}

// Non-finite weights at taps that land in the horizontal padding. The input is 8 wide (one lane group, pad 1,
// stride 1), so at kx = 0 lane 0 is padded and at kx = 2 lane 7 is; oc 2 carries +Inf at (ky 1, kx 0) and oc 5
// -Inf at (ky 1, kx 2). The reference skips padded taps, so the left column of oc 2 and the right column of oc 5
// must stay finite (0 * Inf must never reach an accumulator), while every other position sees the Inf through a
// real input and clamps to the activation bound in both. Even and odd padded lanes so both widened halves of the
// f16 path are covered. Inputs are never zero so no NaN can arise elsewhere. #417
void convolve_small_c_inf_weight_in_padding_f16(void)
{
    const cmsis_nn_dims in = {1, 4, 8, 1};
    const cmsis_nn_dims flt = {8, 3, 3, 1};
    const cmsis_nn_dims out = {1, 4, 8, 8};
    float16_t x[32];
    float16_t w[72];
    float16_t bias[8];
    cmsis_nn_conv_params_f16 cp;

    for (int32_t i = 0; i < 32; i++)
    {
        x[i] = (float16_t)((float32_t)((i % 7) + 1) / 8.0f);
    }
    for (int32_t i = 0; i < 72; i++)
    {
        w[i] = conv_f16_value(i, 31);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        bias[i] = conv_f16_value(i, 32);
    }
    w[(2 * 3 + 1) * 3 + 0] = (float16_t)INFINITY;
    w[(5 * 3 + 1) * 3 + 2] = -(float16_t)INFINITY;
    float16_t *w_packed = pack_rhs_nt_n_from_nt_t_f16(w, 8, 9);

    /* No scratch on purpose: on MVE this shape takes the direct small-C kernel either way, but on scalar builds
     * scratch selects patch-GEMM, whose zero-filled patches multiply padded taps by the weight (0 * Inf = NaN,
     * folded to the clamp bound) -- long-standing behaviour of that route, not what this case pins. Without
     * scratch the scalar route is the generic fallback, which skips padded taps like the direct kernel. */
    conv_f16_params(&cp, 1, 1, 0);
    conv_f16_check(&cp, &in, x, &flt, w, w, bias, &out, 0);
    conv_f16_params(&cp, 1, 1, 1);
    conv_f16_check(&cp, &in, x, &flt, w_packed, w, bias, &out, 0);

    free(w_packed);
}

// Long chains on the scalar leg (#465), which accumulates bias and products in float32 and rounds to float16 once.
// The bias is minus the float16-rounded chain at output 0, so it cancels most of the sum. With x = 1 and
// w = 1 + 2^-10 over just over 2048 taps every partial sum is exact in float32 in any order, and the result must be
// exactly the chain's fractional excess (2^-8 for 2052 taps): a float16 accumulator, a bias added after rounding (0)
// or a dropped bias all miss it. Positive non-integer data in [0.5, 1.5) must stay within the float32 accumulation
// bound of a float64 reference plus 1.5 float16 ulp. Routes: the k=3 and k=5 conv1d packed specializations and the
// 1xN OHWI no-padding region. The MVE leg accumulates in float16 lanes by design, so the case applies to non-MVE
// builds and ARM_MATH_AUTOVECTORIZE.
#if !defined(ARM_MATH_MVE_FLOAT16) || defined(ARM_MATH_AUTOVECTORIZE)
static void conv_f16_long_chain(int32_t in_w, int32_t in_c, int32_t kw, int32_t packed, int32_t exact)
{
    const int32_t out_c = 2;
    const cmsis_nn_dims in = {1, 1, in_w, in_c};
    const cmsis_nn_dims flt = {out_c, 1, kw, in_c};
    const cmsis_nn_dims out = {1, 1, in_w - kw + 1, out_c};
    const int32_t x_n = in_w * in_c;
    const int32_t w_n = out_c * kw * in_c;
    const int32_t y_n = out.w * out_c;
    float16_t *x = (float16_t *)malloc((size_t)x_n * sizeof(float16_t));
    float16_t *w = (float16_t *)malloc((size_t)w_n * sizeof(float16_t));
    float16_t *y = (float16_t *)malloc((size_t)y_n * sizeof(float16_t));
    cmsis_nn_conv_params_f16 cp;
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(w);
    TEST_ASSERT_NOT_NULL(y);
    for (int32_t i = 0; i < x_n; i++)
    {
        x[i] = exact ? (float16_t)1.0f : (float16_t)(0.5f + (float32_t)((i * 37) % 64) / 64.0f);
    }
    for (int32_t i = 0; i < w_n; i++)
    {
        w[i] = exact ? (float16_t)(1.0f + 1.0f / 1024.0f) : (float16_t)(0.5f + (float32_t)((i * 53 + 11) % 64) / 64.0f);
    }
    float16_t *w_kernel = packed ? pack_rhs_nt_n_from_nt_t_f16(w, out_c, kw * in_c) : w;
    TEST_ASSERT_NOT_NULL(w_kernel);

    conv_f16_params(&cp, 0, 0, packed);
    const int32_t size = arm_convolve_wrapper_f16_get_buffer_size(&cp, &in, &flt, &out);
    TEST_ASSERT_TRUE(size >= 0);
    cmsis_nn_context ctx = {size > 0 ? malloc((size_t)size) : NULL, size};
    TEST_ASSERT_TRUE(size == 0 || ctx.buf != NULL);
    const int32_t taps = kw * in_c;
    float16_t bias[2];
    for (int32_t oc = 0; oc < out_c; oc++)
    {
        double s = 0.0;
        for (int32_t k = 0; k < taps; k++)
        {
            s += (double)x[k] * (double)w[oc * taps + k];
        }
        bias[oc] = (float16_t)(-s);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_wrapper_f16(&ctx, &cp, &in, x, &flt, w_kernel, NULL, bias, &out, y));
    for (int32_t o = 0; o < out.w; o++)
    {
        for (int32_t oc = 0; oc < out_c; oc++)
        {
            double r = (double)bias[oc];
            double sum_abs = fabs(r);
            for (int32_t k = 0; k < taps; k++)
            {
                const double prod = (double)x[o * in_c + k] * (double)w[oc * taps + k];
                r += prod;
                sum_abs += fabs(prod);
            }
            const float32_t got = (float32_t)y[o * out_c + oc];
            if (exact)
            {
                TEST_ASSERT_EQUAL_FLOAT((float32_t)r, got);
            }
            else
            {
                int exp2;
                (void)frexp(r, &exp2);
                const double tol = (double)(taps + 1) * ldexp(1.0, -24) * sum_abs + 1.5 * ldexp(1.0, exp2 - 11);
                TEST_ASSERT_FLOAT_WITHIN((float32_t)tol, (float32_t)r, got);
            }
        }
    }

    free(ctx.buf);
    if (packed)
    {
        free(w_kernel);
    }
    free(y);
    free(w);
    free(x);
}
#endif

void convolve_scalar_f32_accumulation_f16(void)
{
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    TEST_IGNORE_MESSAGE("MVE leg accumulates in float16 lanes");
#else
    for (int32_t exact = 1; exact >= 0; exact--)
    {
        conv_f16_long_chain(4, 684, 3, 1, exact); /* k=3 packed: 2052 taps */
        conv_f16_long_chain(6, 412, 5, 1, exact); /* k=5 packed: 2060 taps */
        conv_f16_long_chain(8, 294, 7, 0, exact); /* 1xN OHWI no-padding region: 2058 taps */
    }
#endif
}

/* Direct entries (#674): for one layer on each route of the router, the entry a caller picks with the route
 * predicates in dispatch order gives the router's output byte for byte, for both filter formats; the other
 * format's entry and an entry of another route decline with ARM_CMSIS_NN_NO_IMPL_ERROR and write nothing; the
 * packed-patch GEMM entry reports a missing ctx. */
typedef arm_cmsis_nn_status (*conv_entry_f16)(const cmsis_nn_context *,
                                         const cmsis_nn_conv_params_f16 *,
                                         const cmsis_nn_dims *,
                                         const float16_t *,
                                         const cmsis_nn_dims *,
                                         const float16_t *,
                                         const cmsis_nn_dims *,
                                         const float16_t *,
                                         const cmsis_nn_dims *,
                                         float16_t *);

enum
{
    ROUTE_1X1_F16,
    ROUTE_1XN_F16,
    ROUTE_K5_F16,
    ROUTE_K3_F16,
    ROUTE_SMALL_C_F16,
    ROUTE_PATCH_F16,
    ROUTE_DIRECT_F16,
    ROUTE_COUNT_F16
};

/* [route][accumulation][format: 0 OHWI, 1 NT_N_PACKED] */
static const conv_entry_f16 conv_entries_f16[ROUTE_COUNT_F16][2][2] = {
    {{arm_convolve_1x1_nhwc_ohwi_f16, arm_convolve_1x1_nhwc_packed_f16}, {arm_convolve_1x1_nhwc_ohwi_f16_acc16, arm_convolve_1x1_nhwc_packed_f16_acc16}},
    {{arm_convolve_1_x_n_nhwc_ohwi_f16, arm_convolve_1_x_n_nhwc_packed_f16}, {arm_convolve_1_x_n_nhwc_ohwi_f16_acc16, arm_convolve_1_x_n_nhwc_packed_f16_acc16}},
    {{arm_convolve_1d_k5_nhwc_ohwi_f16, arm_convolve_1d_k5_nhwc_packed_f16}, {arm_convolve_1d_k5_nhwc_ohwi_f16_acc16, arm_convolve_1d_k5_nhwc_packed_f16_acc16}},
    {{arm_convolve_1d_k3_nhwc_ohwi_f16, arm_convolve_1d_k3_nhwc_packed_f16}, {arm_convolve_1d_k3_nhwc_ohwi_f16_acc16, arm_convolve_1d_k3_nhwc_packed_f16_acc16}},
    {{arm_convolve_small_c_nhwc_f16, arm_convolve_small_c_nhwc_f16}, {arm_convolve_small_c_nhwc_f16, arm_convolve_small_c_nhwc_f16}},
    {{arm_convolve_patch_gemm_nhwc_ohwi_f16, arm_convolve_patch_gemm_nhwc_packed_f16}, {arm_convolve_patch_gemm_nhwc_ohwi_f16_acc16, arm_convolve_patch_gemm_nhwc_packed_f16_acc16}},
    {{arm_convolve_direct_nhwc_ohwi_f16, arm_convolve_direct_nhwc_packed_f16}, {arm_convolve_direct_nhwc_ohwi_f16_acc16, arm_convolve_direct_nhwc_packed_f16_acc16}},
};
static const conv_entry_f16 conv_routers_f16[2] = {arm_convolve_nhwc_f16, arm_convolve_nhwc_f16_acc16};

/* The router's choice, from the shared route predicates in its dispatch order */
static int32_t conv_route_f16(const cmsis_nn_context *ctx,
                             const cmsis_nn_conv_params_f16 *cp,
                             const cmsis_nn_dims *in,
                             const cmsis_nn_dims *flt,
                             const cmsis_nn_dims *out)
{
    const int32_t row = flt->h * flt->w * in->c * (int32_t)sizeof(float16_t);
    if (arm_nn_conv_flt_is_1x1(&cp->padding, flt))
    {
        return ROUTE_1X1_F16;
    }
#ifndef NN_DISABLE_SPECIALIZATION
    const bool k5 = arm_nn_conv_flt_is_1d_k(&cp->stride, &cp->padding, &cp->dilation, in, flt, out, 5);
    const bool k3 = arm_nn_conv_flt_is_1d_k(&cp->stride, &cp->padding, &cp->dilation, in, flt, out, 3);
#else
    const bool k5 = false;
    const bool k3 = false;
#endif
    if (arm_nn_conv_flt_is_1xn(&cp->stride, &cp->padding, &cp->dilation, in, flt, out) && !k5 && !k3 && ctx->buf &&
        ctx->size >= arm_convolve_1_x_n_f16_get_buffer_size(cp, in, flt, out, ARM_NN_LAYOUT_NHWC))
    {
        return ROUTE_1XN_F16;
    }
    if (k5)
    {
        return ROUTE_K5_F16;
    }
    if (k3)
    {
        return ROUTE_K3_F16;
    }
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (arm_nn_conv_f16_is_small_c(&cp->stride, &cp->padding, &cp->dilation, in, flt, out))
    {
        return ROUTE_SMALL_C_F16;
    }
#endif
    if (arm_nn_conv_flt_is_patch_gemm(out) && ctx->buf && ctx->size >= row)
    {
        return ROUTE_PATCH_F16;
    }
    return ROUTE_DIRECT_F16;
}

static void conv_direct_entries_case_f16(int32_t n,
                                        int32_t h,
                                        int32_t w,
                                        int32_t c,
                                        int32_t kh,
                                        int32_t kw,
                                        int32_t oc,
                                        int32_t pad,
                                        int32_t expected_route)
{
    const cmsis_nn_dims in = {n, h, w, c};
    const cmsis_nn_dims flt = {oc, kh, kw, c};
    const cmsis_nn_dims out = {n, h + 2 * pad - kh + 1, w + 2 * pad - kw + 1, oc};
    const int32_t in_size = n * h * w * c;
    const int32_t w_size = (oc + 7) / 8 * 8 * kh * kw * c;
    const int32_t out_size = out.n * out.h * out.w * oc;
    float16_t *x = malloc((size_t)in_size * sizeof(float16_t));
    float16_t *wt = malloc((size_t)w_size * sizeof(float16_t));
    float16_t *bias = malloc((size_t)oc * sizeof(float16_t));
    float16_t *ref = malloc((size_t)out_size * sizeof(float16_t));
    float16_t *got = malloc((size_t)out_size * sizeof(float16_t));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(wt);
    TEST_ASSERT_NOT_NULL(bias);
    TEST_ASSERT_NOT_NULL(ref);
    TEST_ASSERT_NOT_NULL(got);
    for (int32_t i = 0; i < in_size; i++)
    {
        x[i] = (float16_t)((float32_t)(((i * 29 + 7) % 97) - 48) / 37.0f);
    }
    for (int32_t i = 0; i < w_size; i++)
    {
        wt[i] = (float16_t)((float32_t)(((i * 31 + 3) % 89) - 44) / 41.0f);
    }
    for (int32_t i = 0; i < oc; i++)
    {
        bias[i] = (float16_t)((float32_t)(i - 3) / 8.0f);
    }
    const cmsis_nn_dims bias_dims = {1, 1, 1, oc};

    for (int32_t packed = 0; packed < 2; packed++)
    {
        cmsis_nn_conv_params_f16 cp;
        memset(&cp, 0, sizeof(cp));
        cp.stride.h = 1;
        cp.stride.w = 1;
        cp.padding.h = kh > 1 ? pad : 0;
        cp.padding.w = pad;
        cp.dilation.h = 1;
        cp.dilation.w = 1;
        cp.activation.min = (float16_t)-2.0f;
        cp.activation.max = (float16_t)2.0f;
        cp.weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED : ARM_NN_WEIGHT_FORMAT_STANDARD;
        const int32_t buf_size = arm_convolve_f16_get_buffer_size(&cp, &in, &flt, &out, ARM_NN_LAYOUT_NHWC);
        void *buf = buf_size > 0 ? malloc((size_t)buf_size) : NULL;
        const cmsis_nn_context ctx = {buf, buf_size};
        const cmsis_nn_context no_ctx = {NULL, 0};
        const int32_t route = conv_route_f16(&ctx, &cp, &in, &flt, &out);
        TEST_ASSERT_EQUAL(expected_route, route);

        for (int32_t acc = 0; acc < 2; acc++)
        {
            memset(ref, 0x55, (size_t)out_size * sizeof(float16_t));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                              conv_routers_f16[acc](&ctx, &cp, &in, x, &flt, wt, &bias_dims, bias, &out, ref));
            memset(got, 0x55, (size_t)out_size * sizeof(float16_t));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                              conv_entries_f16[route][acc][packed](&ctx, &cp, &in, x, &flt, wt, &bias_dims, bias, &out, got));
            TEST_ASSERT_EQUAL_MEMORY(ref, got, (size_t)out_size * sizeof(float16_t));

            /* The other format's entry declines, except small-C, which takes both */
            if (route != ROUTE_SMALL_C_F16)
            {
                memset(got, 0x55, (size_t)out_size * sizeof(float16_t));
                TEST_ASSERT_EQUAL(
                    ARM_CMSIS_NN_NO_IMPL_ERROR,
                    conv_entries_f16[route][acc][1 - packed](&ctx, &cp, &in, x, &flt, wt, &bias_dims, bias, &out, got));
                for (int32_t i = 0; i < out_size * (int32_t)sizeof(float16_t); i++)
                {
                    TEST_ASSERT_EQUAL_HEX8(0x55, ((const uint8_t *)got)[i]);
                }
            }
            /* The conv1d entries decline every shape outside their gate */
            if (!arm_nn_conv_flt_is_1d_k(&cp.stride, &cp.padding, &cp.dilation, &in, &flt, &out, 5))
            {
                TEST_ASSERT_EQUAL(
                    ARM_CMSIS_NN_NO_IMPL_ERROR,
                    conv_entries_f16[ROUTE_K5_F16][acc][packed](&ctx, &cp, &in, x, &flt, wt, &bias_dims, bias, &out, got));
            }
            if (!arm_nn_conv_flt_is_1d_k(&cp.stride, &cp.padding, &cp.dilation, &in, &flt, &out, 3))
            {
                TEST_ASSERT_EQUAL(
                    ARM_CMSIS_NN_NO_IMPL_ERROR,
                    conv_entries_f16[ROUTE_K3_F16][acc][packed](&ctx, &cp, &in, x, &flt, wt, &bias_dims, bias, &out, got));
            }
            /* An out-of-range filter format is neither entry's */
            if (route != ROUTE_SMALL_C_F16 && packed == 0)
            {
                cmsis_nn_conv_params_f16 odd = cp;
                odd.weight_format = (arm_nn_weight_format_flt)2;
                TEST_ASSERT_EQUAL(
                    ARM_CMSIS_NN_NO_IMPL_ERROR,
                    conv_entries_f16[route][acc][0](&ctx, &odd, &in, x, &flt, wt, &bias_dims, bias, &out, got));
            }
            /* A negative scratch size is an argument error */
            const cmsis_nn_context negative = {buf != NULL ? buf : got, -1};
            TEST_ASSERT_EQUAL(
                ARM_CMSIS_NN_ARG_ERROR,
                conv_entries_f16[ROUTE_PATCH_F16][acc][packed](&negative, &cp, &in, x, &flt, wt, &bias_dims, bias, &out, got));
            /* Packed-patch GEMM without scratch is an argument error, not a silent fallback */
            TEST_ASSERT_EQUAL(
                ARM_CMSIS_NN_ARG_ERROR,
                conv_entries_f16[ROUTE_PATCH_F16][acc][packed](&no_ctx, &cp, &in, x, &flt, wt, &bias_dims, bias, &out, got));
        }
        free(buf);
    }
#if !(defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE))
    {
        cmsis_nn_conv_params_f16 cp;
        memset(&cp, 0, sizeof(cp));
        cp.stride.h = 1;
        cp.stride.w = 1;
        cp.dilation.h = 1;
        cp.dilation.w = 1;
        const cmsis_nn_context none = {NULL, 0};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                          arm_convolve_small_c_nhwc_f16(&none, &cp, &in, x, &flt, wt, &bias_dims, bias, &out, got));
    }
#endif
    free(x);
    free(wt);
    free(bias);
    free(ref);
    free(got);
}

void convolve_direct_entries_f16(void)
{
    conv_direct_entries_case_f16(1, 4, 4, 8, 1, 1, 8, 0, ROUTE_1X1_F16);
    conv_direct_entries_case_f16(1, 1, 16, 4, 1, 4, 8, 0, ROUTE_1XN_F16);
    /* More than 32 taps per output (40 and 48), so float16 folding differs from float16 lanes */
    conv_direct_entries_case_f16(1, 4, 4, 40, 1, 1, 8, 0, ROUTE_1X1_F16);
    conv_direct_entries_case_f16(1, 1, 16, 12, 1, 4, 8, 0, ROUTE_1XN_F16);
#ifndef NN_DISABLE_SPECIALIZATION
    conv_direct_entries_case_f16(1, 1, 20, 4, 1, 5, 6, 0, ROUTE_K5_F16);
    /* 40 taps per output */
    conv_direct_entries_case_f16(1, 1, 20, 8, 1, 5, 6, 0, ROUTE_K5_F16);
    conv_direct_entries_case_f16(1, 1, 20, 4, 1, 3, 9, 0, ROUTE_K3_F16);
    /* 48 taps per output: float16 folding differs from float16 lanes */
    conv_direct_entries_case_f16(1, 1, 20, 16, 1, 3, 9, 0, ROUTE_K3_F16);
#else
    conv_direct_entries_case_f16(1, 1, 20, 4, 1, 5, 6, 0, ROUTE_1XN_F16);
    conv_direct_entries_case_f16(1, 1, 20, 4, 1, 3, 9, 0, ROUTE_1XN_F16);
#endif
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    conv_direct_entries_case_f16(1, 6, 6, 7, 3, 3, 8, 1, ROUTE_SMALL_C_F16);
#else
    conv_direct_entries_case_f16(1, 6, 6, 7, 3, 3, 8, 1, ROUTE_PATCH_F16);
#endif
    conv_direct_entries_case_f16(2, 6, 6, 8, 3, 3, 9, 1, ROUTE_PATCH_F16);
    conv_direct_entries_case_f16(1, 5, 5, 8, 3, 3, 4, 1, ROUTE_DIRECT_F16);
}

/* Each term of the shared route predicates and the float16 small-C predicate against a fixed answer: one layer inside
 * each gate, then that layer with one term changed. A wrong term would move the router, its entries and the
 * route-following test above together, so only fixed answers pin it. */
typedef struct
{
    cmsis_nn_tile stride, padding, dilation;
    cmsis_nn_dims in, flt, out;
} conv_shape_flt;

#define CONV_SHAPE(s) &(s).stride, &(s).padding, &(s).dilation, &(s).in, &(s).flt, &(s).out
#define EXPECT_TERM(base, field, value, expected, call)                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        conv_shape_flt s_ = (base);                                                                                    \
        s_.field = (value);                                                                                            \
        TEST_ASSERT_EQUAL_MESSAGE((expected), (call), #call ": " #field " = " #value);                                 \
    } while (0)

void convolve_route_predicates_f16(void)
{
    const conv_shape_flt pw = {{1, 1}, {0, 0}, {1, 1}, {1, 4, 4, 8}, {8, 1, 1, 8}, {1, 4, 4, 8}};
    TEST_ASSERT_TRUE(arm_nn_conv_flt_is_1x1(&pw.padding, &pw.flt));
    EXPECT_TERM(pw, flt.h, 2, false, arm_nn_conv_flt_is_1x1(&s_.padding, &s_.flt));
    EXPECT_TERM(pw, flt.w, 2, false, arm_nn_conv_flt_is_1x1(&s_.padding, &s_.flt));
    EXPECT_TERM(pw, padding.h, 1, false, arm_nn_conv_flt_is_1x1(&s_.padding, &s_.flt));
    EXPECT_TERM(pw, padding.w, 1, false, arm_nn_conv_flt_is_1x1(&s_.padding, &s_.flt));

    const conv_shape_flt k3 = {{1, 1}, {0, 0}, {1, 1}, {1, 1, 20, 4}, {6, 1, 3, 4}, {1, 1, 18, 6}};
    TEST_ASSERT_TRUE(arm_nn_conv_flt_is_1d_k(CONV_SHAPE(k3), 3));
    TEST_ASSERT_FALSE(arm_nn_conv_flt_is_1d_k(CONV_SHAPE(k3), 5));
    EXPECT_TERM(k3, in.n, 2, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, in.h, 2, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, out.h, 2, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, flt.h, 2, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, flt.w, 4, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, stride.h, 2, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, stride.w, 2, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, padding.h, 1, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, padding.w, 1, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, dilation.h, 2, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));
    EXPECT_TERM(k3, dilation.w, 2, false, arm_nn_conv_flt_is_1d_k(CONV_SHAPE(s_), 3));

    const conv_shape_flt row = {{1, 1}, {0, 0}, {1, 1}, {1, 1, 16, 4}, {8, 1, 4, 4}, {1, 1, 13, 8}};
    TEST_ASSERT_TRUE(arm_nn_conv_flt_is_1xn(CONV_SHAPE(row)));
    EXPECT_TERM(row, in.n, 2, true, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, stride.w, 2, true, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, padding.w, 1, true, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, in.h, 2, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, out.h, 2, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, flt.h, 2, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, flt.w, 1, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, stride.h, 2, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, stride.w, 0, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, padding.h, 1, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, dilation.h, 2, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));
    EXPECT_TERM(row, dilation.w, 2, false, arm_nn_conv_flt_is_1xn(CONV_SHAPE(s_)));

    /* Patch GEMM: 8 output channels and 8 output positions at least */
    const conv_shape_flt gemm = {{1, 1}, {1, 1}, {1, 1}, {1, 1, 8, 4}, {8, 3, 3, 4}, {1, 1, 8, 8}};
    TEST_ASSERT_TRUE(arm_nn_conv_flt_is_patch_gemm(&gemm.out));
    EXPECT_TERM(gemm, out.c, 7, false, arm_nn_conv_flt_is_patch_gemm(&s_.out));
    EXPECT_TERM(gemm, out.w, 7, false, arm_nn_conv_flt_is_patch_gemm(&s_.out));
    EXPECT_TERM(gemm, out.h, 2, true, arm_nn_conv_flt_is_patch_gemm(&s_.out));

    /* Small C: 1 to 7 input channels, and every 16-bit gather and scatter offset in range */
    const conv_shape_flt sc = {{1, 1}, {1, 1}, {1, 1}, {1, 6, 6, 7}, {8, 3, 3, 7}, {1, 6, 6, 8}};
    TEST_ASSERT_TRUE(arm_nn_conv_f16_is_small_c(CONV_SHAPE(sc)));
    EXPECT_TERM(sc, in.c, 8, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(sc, in.c, 0, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(sc, out.c, 0, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(sc, out.w, 0, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(sc, flt.w, INT32_MIN, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    /* Gather reach W + pad + 7 * stride + (KW - 1) * dilation = 65,535 columns of one channel, then one more */
    const conv_shape_flt edge = {{1, 1}, {0, 0}, {1, 1}, {1, 1, 65528, 1}, {8, 1, 1, 1}, {1, 1, 65528, 8}};
    TEST_ASSERT_TRUE(arm_nn_conv_f16_is_small_c(CONV_SHAPE(edge)));
    EXPECT_TERM(edge, in.w, 65529, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(edge, padding.w, 1, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(edge, stride.w, 2, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(edge, flt.w, 2, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(edge, in.c, 2, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    /* Scatter: 8 positions of C_OUT channels; 8 x 8,191 = 65,528 fits, 8 x 8,192 does not */
    EXPECT_TERM(edge, out.c, 8191, true, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
    EXPECT_TERM(edge, out.c, 8192, false, arm_nn_conv_f16_is_small_c(CONV_SHAPE(s_)));
}
