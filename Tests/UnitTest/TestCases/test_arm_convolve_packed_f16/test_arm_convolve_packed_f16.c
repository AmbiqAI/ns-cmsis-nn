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

// 3x3, in_c = 4, out_c = 12 on a 4x4 input (so the second packed block is exercised): with scratch this takes the
// patch-GEMM path, without it the generic fallback. Both must honour NT_N_PACKED; the fallback used to read packed
// weights as OHWI.
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
static bool conv_f16_bits_are_nan(float16_t x)
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
