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
 * Title:        arm_nnsupportfunctions_flt.h
 * Description:  Floating-point support API extensions for CMSIS-NN
 *
 * $Date:        17 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NNSUPPORTFUNCTIONS_FLT_H
#define ARM_NNSUPPORTFUNCTIONS_FLT_H

#include "Internal/arm_conv_opt_common.h"
#include "Internal/arm_nn_compiler.h"
#include "Internal/arm_nn_vcvt_f16.h"
#include "arm_nn_types_flt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Floating-point internal support APIs. */

/**
 * @addtogroup groupSupport
 * @{
 */

#if ARM_NN_FLOAT_API_ENABLED && defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
/**
 * @brief Reduce a float32 MVE vector with addition.
 *
 * Gated on MVE availability rather than ARM_NN_ENABLE_F32: float32x4_t is a
 * hardware register type, and float16 kernels that accumulate in float32
 * (e.g. arm_reduce_sum_f16) need this helper in F16-only builds.
 *
 * @param[in] v Vector to reduce.
 * @return Sum of the four lanes of @p v.
 */
__STATIC_INLINE float32_t arm_nn_vec_reduce_add_f32(float32x4_t v)
{
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}
#endif

#if ARM_NN_ENABLE_F32

/**
 * @brief Polynomial coefficients used by the float32 MVE exp approximation.
 */
extern const float32_t arm_nn_exp_poly_coeffs_f32[8];

/**
 * @brief LUT for `2^(i/256)` used by the float32 LUT softmax approximation.
 *
 * Stores 257 samples for `i = 0..256` so interpolation can safely read
 * `lut[idx + 1]` while indexing the 256 fractional segments.
 */
extern const float32_t arm_nn_exp2_lut_f32[257];

/**
 * @brief Floor of @p x as an int32_t.
 *
 * Precondition: @p x must already be reduced to the int32_t range and must not
 * be NaN -- the float-to-int conversion below is undefined otherwise. The only
 * caller, arm_nn_softmax_exp_lut_f32(), guarantees this by clamping its input
 * to [-80, 80] (NaN included, see there) before scaling by log2(e), which
 * bounds @p x to +/-116.
 *
 * @param[in] x Value to floor.
 * @return Largest int32_t not greater than @p x.
 */
__STATIC_INLINE int32_t arm_nn_softmax_floor_to_int_f32(float32_t x)
{
    const int32_t n = (int32_t)x;
    return (x < (float32_t)n) ? (n - 1) : n;
}

/**
 * @brief Reinterpret a 32-bit pattern as a float32.
 *
 * @param[in] bits IEEE-754 binary32 bit pattern.
 * @return The float32 value with the bit pattern @p bits.
 */
__STATIC_INLINE float32_t arm_nn_softmax_fp32_from_bits(uint32_t bits)
{
    union
    {
        uint32_t u;
        float32_t f;
    } cvt;
    cvt.u = bits;
    return cvt.f;
}

/**
 * @brief Compute `2^n` as a float32 by building the exponent field directly.
 *
 * @param[in] n Integer exponent. Clamped to the normal float32 exponent range `[-126, 127]`.
 * @return `2^n` as a float32.
 */
__STATIC_INLINE float32_t arm_nn_softmax_exp2i_f32(int32_t n)
{
    const int32_t float32_min_normal_exponent = -126;
    const int32_t float32_max_finite_exponent = 127;
    const int32_t float32_exponent_bias = 127;
    const int32_t float32_mantissa_bits = 23;

    n = ARM_NN_CLAMP(n, float32_max_finite_exponent, float32_min_normal_exponent);
    return arm_nn_softmax_fp32_from_bits((uint32_t)(n + float32_exponent_bias) << float32_mantissa_bits);
}

/**
 * @brief Taylor/Estrin exp approximation for float32 softmax helpers.
 *
 * The polynomial is evaluated on r in [-ln2/2, ln2/2].
 * Coefficients come from the Maclaurin series of exp(r):
 *   exp(r) ~= 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6!
 * Grouped via Estrin to reduce dependency depth:
 *   p = (1 + r) + r^2*(1/2 + r/6) + r^4*(1/24 + r/120) + r^6*(1/720)
 *
 * Range reduction follows:
 *   x = n * ln(2) + r,  exp(x) = exp(r) * 2^n
 *
 * @param[in] x Exponent argument. Clamped to `[-80, 80]` before evaluation.
 * @return Approximation of `exp(x)`.
 */
__STATIC_INLINE float32_t arm_nn_softmax_exp_taylor_f32(float32_t x)
{
    const float32_t max_value = 80.0f;
    const float32_t min_value = -80.0f;
    const float32_t log2e = 1.44269504088896341f;
    const float32_t ln2 = 0.69314718055994531f;

    /* Same clamp as arm_nn_softmax_exp_lut_f32(); see there for why both the
     * order and the strictness of these compares are load-bearing. */
    x = (x < max_value) ? x : max_value;
    x = (x > min_value) ? x : min_value;

    const float32_t t = x * log2e;
    const int32_t n = (t >= 0.0f) ? (int32_t)(t + 0.5f) : (int32_t)(t - 0.5f);
    const float32_t r = x - (float32_t)n * ln2;

    const float32_t r2 = r * r;
    const float32_t r4 = r2 * r2;
    const float32_t r6 = r4 * r2;

    const float32_t t0 = 1.0f + r;
    const float32_t t1 = 0.5f + (1.0f / 6.0f) * r;
    const float32_t t2 = (1.0f / 24.0f) + (1.0f / 120.0f) * r;
    const float32_t t3 = (1.0f / 720.0f);

    return (t0 + t1 * r2 + t2 * r4 + t3 * r6) * arm_nn_softmax_exp2i_f32(n);
}

/**
 * @brief LUT-based exp approximation for float32 softmax helpers.
 *
 * Splits `x * log2(e)` into an integer part handled by arm_nn_softmax_exp2i_f32() and a fractional part
 * interpolated linearly from `arm_nn_exp2_lut_f32`.
 *
 * @param[in] x Exponent argument. Clamped to `[-80, 80]` before evaluation; NaN is flushed to `80`.
 * @return Approximation of `exp(x)`.
 */
__STATIC_INLINE float32_t arm_nn_softmax_exp_lut_f32(float32_t x)
{
    const float32_t max_value = 80.0f;
    const float32_t min_value = -80.0f;
    const float32_t log2e = 1.44269504088896341f;
    const float32_t exp2_lut_segments = 256.0f;
    const int32_t exp2_lut_max_index = 255;

    /* Ordered compares, so NaN fails the first test and is flushed to
     * max_value. This is the same result ARM_NN_CLAMP() produces (ARM_NN_MIN() runs first
     * and returns max_value for NaN), written explicitly because the rest of
     * this function depends on it: it is what keeps NaN out of the two
     * float-to-int conversions below, where it would be undefined behaviour.
     *
     * The min-then-max ORDER is load-bearing and must not be "simplified" into
     * a single nested conditional. Under -ffinite-math-only -- which the
     * default library build enables, since CMSIS_OPTIMIZATION_LEVEL is -Ofast
     * -- the compiler is free to contract each statement into VMINNM/VMAXNM,
     * and those are IEEE minNum/maxNum, which return the *numeric* operand
     * against a NaN. Applying the max bound first therefore sends NaN to
     * max_value; applying the min bound first would send it to min_value
     * instead. ARM_NN_CLAMP() expands to ARM_NN_MAX(ARM_NN_MIN(x, hi), lo), i.e. min-bound-first,
     * so this order is what reproduces the long-standing behaviour bit for
     * bit in every optimisation mode.
     *
     * The compares are STRICT for the same reason, so the saturating arm owns
     * the exact boundary exactly as ARM_NN_MIN()/ARM_NN_MAX() do. With `<=` the pass-through
     * arm keeps x at x == max_value, and the compiler then propagates a
     * runtime value where base propagates the literal 80.0f -- enough to
     * change constant folding downstream and shift the Taylor result by tens
     * of ULP right at the boundary. NaN routing is unaffected: NaN < max_value
     * is false either way, so NaN still lands on max_value.
     *
     * Defined consequence: exp(NaN) == exp(80), and hence
     * arm_nn_sigmoid_scalar_f32(NaN) == 1.0f. NaN is not a supported input to
     * the softmax/sigmoid kernels; this only pins down what happens if one
     * arrives. */
    x = (x < max_value) ? x : max_value;
    x = (x > min_value) ? x : min_value;

    const float32_t t = x * log2e;
    const int32_t n = arm_nn_softmax_floor_to_int_f32(t);
    const float32_t f = t - (float32_t)n;
    const float32_t idx_f = f * exp2_lut_segments;
    int32_t idx = (int32_t)idx_f;

    if (idx < 0)
    {
        idx = 0;
    }
    else if (idx > exp2_lut_max_index)
    {
        idx = exp2_lut_max_index;
    }

    const float32_t frac = idx_f - (float32_t)idx;
    const float32_t y0 = arm_nn_exp2_lut_f32[idx];
    const float32_t y1 = arm_nn_exp2_lut_f32[idx + 1];
    return (y0 + (y1 - y0) * frac) * arm_nn_softmax_exp2i_f32(n);
}

/**
 * @brief Scalar exp approximation used by the float32 softmax paths.
 *
 * Dispatches to arm_nn_softmax_exp_taylor_f32() when `ARM_NN_USE_EXP_TAYLOR` is defined and to
 * arm_nn_softmax_exp_lut_f32() otherwise.
 *
 * @param[in] x Exponent argument.
 * @return Approximation of `exp(x)`.
 */
__STATIC_INLINE float32_t arm_nn_softmax_exp_scalar_f32(float32_t x)
{
    #if defined(ARM_NN_USE_EXP_TAYLOR)
    return arm_nn_softmax_exp_taylor_f32(x);
    #else
    return arm_nn_softmax_exp_lut_f32(x);
    #endif
}

#endif

#if ARM_NN_ENABLE_F32

/**
 * @brief LUT for tanh(x) sampled over `x in [0, 6]` for float32 helpers.
 *
 * Stores 385 samples so interpolation can safely read `lut[idx + 1]` while
 * indexing the 384 fractional segments across the interval. The grid spacing
 * (`6/384 == 1/64`) matches the earlier 257-entry `[0, 4]` table, so entries
 * `0..256` are bit-identical to it and the index multiplier is unchanged.
 * Generated by `scripts/gen_tanh_lut_f32.py`.
 */
extern const float32_t arm_nn_tanh_lut_f32[385];

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
/**
 * @brief MVE float32 exp approximation used by float softmax paths.
 *
 * @param[in] x Vector of exponent arguments.
 * @return Per-lane approximation of `exp(x)`. Lanes that would underflow are flushed to zero.
 */
__STATIC_INLINE float32x4_t arm_nn_vexpq_poly_mve_f32(float32x4_t x)
{
    const int32x4_t m = vcvtq_s32_f32(vmulq(x, 1.4426950408f));
    const float32x4_t val = vfmsq(x, vcvtq_f32_s32(m), vdupq_n_f32(0.6931471805f));

    const float32x4_t a = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f32[4]), val, arm_nn_exp_poly_coeffs_f32[0]);
    const float32x4_t b = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f32[6]), val, arm_nn_exp_poly_coeffs_f32[2]);
    const float32x4_t c = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f32[5]), val, arm_nn_exp_poly_coeffs_f32[1]);
    const float32x4_t d = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f32[7]), val, arm_nn_exp_poly_coeffs_f32[3]);
    const float32x4_t x2 = vmulq(val, val);
    const float32x4_t x4 = vmulq(x2, x2);
    float32x4_t poly = vfmaq(vfmaq(a, b, x2), vfmaq(c, d, x2), x4);

    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vdupq_m(poly, 0.0f, vcmpltq(m, -126));
    return poly;
}
    #endif

/**
 * @brief Copy a float32 vector.
 * @param[out] dst        Destination buffer.
 * @param[in]  src        Source buffer.
 * @param[in]  block_size Number of elements to copy.
 */
__STATIC_FORCEINLINE void
arm_memcpy_f32(float32_t *__RESTRICT dst, const float32_t *__RESTRICT src, uint32_t block_size)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    __asm volatile("   wlstp.32                lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vldrw.32                q0, [%[in]], #16           \n"
                   "   vstrw.32                q0, [%[out]], #16          \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(src), [out] "+r"(dst)
                   : [cnt] "r"(block_size)
                   : "q0", "memory", "r14");
    #else
    __builtin_memcpy(dst, src, (size_t)block_size * sizeof(float32_t));
    #endif
}

/**
 * @brief Set a float32 vector to a constant value.
 * @param[out] dst        Destination buffer.
 * @param[in]  val        Fill value.
 * @param[in]  block_size Number of elements to write.
 */
__STATIC_FORCEINLINE void arm_memset_f32(float32_t *__RESTRICT dst, const float32_t val, uint32_t block_size)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t vec = vdupq_n_f32(val);
    uint32_t i = 0;
    for (; i + 4U <= block_size; i += 4U)
    {
        vst1q(dst + i, vec);
    }
    if (i < block_size)
    {
        const mve_pred16_t p = vctp32q(block_size - i);
        vst1q_p(dst + i, vec, p);
    }
    #else
    for (uint32_t i = 0; i < block_size; ++i)
    {
        dst[i] = val;
    }
    #endif
}

/**
 * @brief Specialized NHWC depthwise 1D kernel for `k=3`, `ch_mult=1` (float32).
 *
 * @param[in]  x_nhwc Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c   Number of input (and output) channels.
 * @param[in]  in_w   Input width. Currently unused by the kernel.
 * @param[in]  kernel Depthwise weights with shape `[3][in_c]`.
 * @param[in]  b      Optional bias vector of `in_c` elements. May be NULL.
 * @param[out] out    Output row in NHWC layout with shape `[out_w][in_c]`.
 * @param[in]  out_w  Output width. Output position `ow` reads input positions `ow..ow+2`.
 */
void arm_nn_depthwise_conv1d_k3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                         int32_t in_c,
                                         int32_t in_w,
                                         const float32_t *__RESTRICT kernel,
                                         const float32_t *__RESTRICT b,
                                         float32_t *__RESTRICT out,
                                         int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=5` (float32).
 *
 * @param[in]  x_nhwc Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c   Number of input channels.
 * @param[in]  in_w   Input width. Currently unused by the kernel.
 * @param[in]  kernel Weights with shape `[out_c][5][in_c]`.
 * @param[in]  b      Optional bias vector of `out_c` elements. May be NULL.
 * @param[out] out    Output row in NHWC layout with shape `[out_w][out_c]`.
 * @param[in]  out_c  Number of output channels.
 * @param[in]  out_w  Output width. Output position `ow` reads input positions `ow..ow+4`.
 */
void arm_nn_conv1d_k5_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float32_t *__RESTRICT kernel,
                               const float32_t *__RESTRICT b,
                               float32_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=5` (float32, packed weights).
 *
 * The packed kernel uses the same `NTxN` RHS layout as
 * `arm_nn_mat_mult_nt_n_packed_f32`, i.e. `[(5 * in_c)][out_c_block_of_4]`.
 *
 * @param[in]  x_nhwc        Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c          Number of input channels.
 * @param[in]  in_w          Input width. Currently unused by the kernel.
 * @param[in]  kernel_packed Weights packed in output-channel blocks of 4 as described above.
 * @param[in]  b             Optional bias vector of `out_c` elements. May be NULL.
 * @param[out] out           Output row in NHWC layout with shape `[out_w][out_c]`.
 * @param[in]  out_c         Number of output channels.
 * @param[in]  out_w         Output width. Output position `ow` reads input positions `ow..ow+4`.
 */
void arm_nn_conv1d_k5_packed_f32(const float32_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float32_t *__RESTRICT kernel_packed,
                                 const float32_t *__RESTRICT b,
                                 float32_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=3` (float32).
 *
 * @param[in]  x_nhwc Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c   Number of input channels.
 * @param[in]  in_w   Input width. Currently unused by the kernel.
 * @param[in]  kernel Weights with shape `[out_c][3][in_c]`.
 * @param[in]  b      Optional bias vector of `out_c` elements. May be NULL.
 * @param[out] out    Output row in NHWC layout with shape `[out_w][out_c]`.
 * @param[in]  out_c  Number of output channels.
 * @param[in]  out_w  Output width. Output position `ow` reads input positions `ow..ow+2`.
 */
void arm_nn_conv1d_k3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float32_t *__RESTRICT kernel,
                               const float32_t *__RESTRICT b,
                               float32_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=3` (float32, packed weights).
 *
 * The packed kernel uses the same `NTxN` RHS layout as
 * `arm_nn_mat_mult_nt_n_packed_f32`, i.e. `[(3 * in_c)][out_c_block_of_4]`.
 *
 * @param[in]  x_nhwc        Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c          Number of input channels.
 * @param[in]  in_w          Input width. Currently unused by the kernel.
 * @param[in]  kernel_packed Weights packed in output-channel blocks of 4 as described above.
 * @param[in]  b             Optional bias vector of `out_c` elements. May be NULL.
 * @param[out] out           Output row in NHWC layout with shape `[out_w][out_c]`.
 * @param[in]  out_c         Number of output channels.
 * @param[in]  out_w         Output width. Output position `ow` reads input positions `ow..ow+2`.
 */
void arm_nn_conv1d_k3_packed_f32(const float32_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float32_t *__RESTRICT kernel_packed,
                                 const float32_t *__RESTRICT b,
                                 float32_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=3`, `s=3` (float32).
 *
 * @param[in]  x_nhwc Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c   Number of channels.
 * @param[in]  in_w   Input width. Currently unused by the kernel.
 * @param[out] out    Output row in NHWC layout with shape `[out_w][in_c]`.
 * @param[in]  out_w  Output width. Output position `ow` reads input positions `3*ow..3*ow+2`.
 */
void arm_nn_maxpool1d_k3s3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float32_t *__RESTRICT out,
                                    int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=2`, `s=2` without output clamp (float32).
 *
 * @param[in]  x_nhwc Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c   Number of channels.
 * @param[in]  in_w   Input width. Currently unused by the kernel.
 * @param[out] out    Output row in NHWC layout with shape `[out_w][in_c]`.
 * @param[in]  out_w  Output width. Output position `ow` reads input positions `2*ow..2*ow+1`.
 */
void arm_nn_maxpool1d_k2s2_nhwc_noclip_f32(const float32_t *__RESTRICT x_nhwc,
                                           int32_t in_c,
                                           int32_t in_w,
                                           float32_t *__RESTRICT out,
                                           int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=2`, `s=2` with clamp (float32).
 *
 * @param[in]  x_nhwc  Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c    Number of channels.
 * @param[in]  in_w    Input width. Currently unused by the kernel.
 * @param[out] out     Output row in NHWC layout with shape `[out_w][in_c]`.
 * @param[in]  out_w   Output width. Output position `ow` reads input positions `2*ow..2*ow+1`.
 * @param[in]  act_min Lower clamp bound applied to @p out.
 * @param[in]  act_max Upper clamp bound applied to @p out.
 */
void arm_nn_maxpool1d_k2s2_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float32_t *__RESTRICT out,
                                    int32_t out_w,
                                    float32_t act_min,
                                    float32_t act_max);

/**
 * @brief Matrix multiply with non-transposed lhs and transposed rhs rows (float32).
 *
 * @param[in]  lhs                Left-hand matrix stored row-major.
 * @param[in]  rhs                Right-hand matrix stored row-major, one row per output channel.
 * @param[in]  bias               Optional bias vector.
 * @param[out] dst                Output matrix.
 * @param[in]  lhs_rows           Number of rows in @p lhs.
 * @param[in]  rhs_rows           Number of rows in @p rhs.
 * @param[in]  rhs_cols           Number of columns in @p rhs.
 * @param[in]  row_address_offset Output row stride, expressed in elements.
 * @param[in]  activation_min     Lower clamp bound.
 * @param[in]  activation_max     Upper clamp bound.
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_f32(const float32_t *__RESTRICT lhs,
                                             const float32_t *__RESTRICT rhs,
                                             const float32_t *__RESTRICT bias,
                                             float32_t *__RESTRICT dst,
                                             int32_t lhs_rows,
                                             int32_t rhs_rows,
                                             int32_t rhs_cols,
                                             int32_t row_address_offset,
                                             float32_t activation_min,
                                             float32_t activation_max);

/**
 * @brief Matrix multiply with non-transposed lhs and packed non-transposed rhs (float32).
 *
 * @param[in]  lhs                Left-hand matrix stored row-major with logical shape `[lhs_rows, rhs_cols]`.
 * @param[in]  rhs_packed         Right-hand matrix with logical shape `[rhs_cols, rhs_rows]`, packed in column blocks
 *                                of 4. The final block uses the same packed stride and inactive tail lanes are ignored.
 * @param[in]  bias               Optional bias vector.
 * @param[out] dst                Output matrix.
 * @param[in]  lhs_rows           Number of rows in @p lhs.
 * @param[in]  rhs_rows           Number of logical output columns in the unpacked rhs matrix.
 * @param[in]  rhs_cols           Shared reduction dimension `K`.
 * @param[in]  row_address_offset Output row stride, expressed in elements.
 * @param[in]  activation_min     Lower clamp bound.
 * @param[in]  activation_max     Upper clamp bound.
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_n_packed_f32(const float32_t *__RESTRICT lhs,
                                                    const float32_t *__RESTRICT rhs_packed,
                                                    const float32_t *__RESTRICT bias,
                                                    float32_t *__RESTRICT dst,
                                                    int32_t lhs_rows,
                                                    int32_t rhs_rows,
                                                    int32_t rhs_cols,
                                                    int32_t row_address_offset,
                                                    float32_t activation_min,
                                                    float32_t activation_max);

/**
 * @brief Pack a single convolution patch into one row of a contiguous float32 patch matrix.
 *
 * Developers familiar with im2row/im2col terminology can think of this as packing one output patch into one row.
 *
 * @param[in]  input      Input tensor for one batch in NHWC layout with shape `[in_h][in_w][in_c]`.
 * @param[in]  in_h       Input height.
 * @param[in]  in_w       Input width.
 * @param[in]  in_c       Number of input channels.
 * @param[in]  kernel_h   Kernel height.
 * @param[in]  kernel_w   Kernel width.
 * @param[in]  stride_h   Vertical stride.
 * @param[in]  stride_w   Horizontal stride.
 * @param[in]  pad_h      Top padding.
 * @param[in]  pad_w      Left padding.
 * @param[in]  dilation_h Vertical dilation.
 * @param[in]  dilation_w Horizontal dilation.
 * @param[in]  out_y      Output row index of the patch to pack.
 * @param[in]  out_x      Output column index of the patch to pack.
 * @param[in]  pad_value  Value written for taps that fall outside the input.
 * @param[out] patch_row  Destination row of `kernel_h * kernel_w * in_c` elements, ordered
 *                        `[kernel_h][kernel_w][in_c]`.
 */
void arm_nn_pack_conv_patch_f32(const float32_t *__RESTRICT input,
                                int32_t in_h,
                                int32_t in_w,
                                int32_t in_c,
                                int32_t kernel_h,
                                int32_t kernel_w,
                                int32_t stride_h,
                                int32_t stride_w,
                                int32_t pad_h,
                                int32_t pad_w,
                                int32_t dilation_h,
                                int32_t dilation_w,
                                int32_t out_y,
                                int32_t out_x,
                                float32_t pad_value,
                                float32_t *__RESTRICT patch_row);

/**
 * @brief Specialized softmax helper for a single float32 row of length 2.
 *
 * @param[in]  in   Pointer to two contiguous float32 input values.
 * @param[out] out  Pointer to two contiguous float32 output values.
 */
void arm_nn_softmax_1x2_f32(const float32_t *in, float32_t *out);

#endif /* ARM_NN_ENABLE_F32 */

#if ARM_NN_ENABLE_F16

    /**
     * @brief Blockwise float16 accumulation on the MVE legs (AmbiqAI/ns-cmsis-nn#586).
     *
     * A float16 accumulator lane sums at most ARM_NN_F16_ACC_BLOCK taps, in the kernel's tap order, before its partial
     * is widened exactly and added into a float32 accumulator; the float32 sum rounds to float16 once. The `_acc16`
     * entries instantiate the same kernel bodies with ARM_NN_F16_ACC_BLOCK_NONE, which never folds.
     */
    #define ARM_NN_F16_ACC_BLOCK (32)
    #define ARM_NN_F16_ACC_BLOCK_NONE (INT32_MAX)

/**
 * @brief Polynomial coefficients used by the float16 MVE exp approximation.
 *
 * The float16 MVE helper evaluates the polynomial in widened float32 lanes,
 * but it uses a dedicated coefficient table to keep the float16 path isolated
 * from the float32 feature gate and softmax support stack.
 */
extern const float32_t arm_nn_exp_poly_coeffs_f16[8];

/**
 * @brief Quantized binary16 LUT for `2^(i/256)` used by float16 helpers.
 *
 * Stores 257 samples for `i = 0..256` so interpolation can safely read
 * `lut[idx + 1]` while indexing the 256 fractional segments.
 */
extern const uint16_t arm_nn_exp2_lut_f16[257];

/**
 * @brief Quantized binary16 LUT for tanh(x) with `x in [0, 4]`.
 *
 * Stores 257 samples so interpolation can safely read `lut[idx + 1]` while
 * indexing the 256 fractional segments across the interval.
 */
extern const uint16_t arm_nn_tanh_lut_f16[257];

/**
 * @brief Reinterpret a 16-bit pattern as a float16.
 *
 * @param[in] bits IEEE-754 binary16 bit pattern.
 * @return The float16 value with the bit pattern @p bits.
 */
__STATIC_INLINE float16_t arm_nn_softmax_fp16_from_bits(uint16_t bits)
{
    union
    {
        uint16_t u;
        float16_t f;
    } cvt;
    cvt.u = bits;
    return cvt.f;
}

/**
 * @brief Floor of @p x as an int32_t.
 *
 * @param[in] x Value to floor. Must be finite and within the int32_t range.
 * @return Largest int32_t not greater than @p x.
 */
__STATIC_INLINE int32_t arm_nn_softmax_floor_to_int_f16(float16_t x)
{
    const float32_t x_f32 = (float32_t)x;
    const int32_t n = (int32_t)x_f32;
    return (x_f32 < (float32_t)n) ? (n - 1) : n;
}

/**
 * @brief Compute `2^n` as a float16 by building the exponent field directly.
 *
 * @param[in] n Integer exponent. Clamped to the normal float16 exponent range `[-14, 15]`.
 * @return `2^n` as a float16.
 */
__STATIC_INLINE float16_t arm_nn_softmax_exp2i_f16(int32_t n)
{
    const int32_t float16_min_normal_exponent = -14;
    const int32_t float16_max_finite_exponent = 15;
    const int32_t float16_exponent_bias = 15;
    const int32_t float16_mantissa_bits = 10;

    n = ARM_NN_CLAMP(n, float16_max_finite_exponent, float16_min_normal_exponent);
    return arm_nn_softmax_fp16_from_bits((uint16_t)((n + float16_exponent_bias) << float16_mantissa_bits));
}

/**
 * @brief Taylor/Estrin exp approximation for float16 softmax helpers.
 *
 * The evaluation uses float32 intermediates to keep the approximation stable,
 * but it is fully independent from the float32 softmax support tables.
 *
 * @param[in] x Exponent argument. Clamped to `[-80, 80]` before evaluation.
 * @return Approximation of `exp(x)`.
 */
__STATIC_INLINE float16_t arm_nn_softmax_exp_taylor_f16(float16_t x)
{
    const float32_t max_value = 80.0f;
    const float32_t min_value = -80.0f;
    const float32_t log2e = 1.44269504088896341f;
    const float32_t ln2 = 0.69314718055994531f;

    float32_t x_f32 = (float32_t)x;
    x_f32 = ARM_NN_CLAMP(x_f32, max_value, min_value);

    const float32_t t = x_f32 * log2e;
    const int32_t n = (t >= 0.0f) ? (int32_t)(t + 0.5f) : (int32_t)(t - 0.5f);
    const float32_t r = x_f32 - (float32_t)n * ln2;

    const float32_t r2 = r * r;
    const float32_t r4 = r2 * r2;
    const float32_t r6 = r4 * r2;

    const float32_t t0 = 1.0f + r;
    const float32_t t1 = 0.5f + (1.0f / 6.0f) * r;
    const float32_t t2 = (1.0f / 24.0f) + (1.0f / 120.0f) * r;
    const float32_t t3 = (1.0f / 720.0f);

    const float32_t poly = t0 + t1 * r2 + t2 * r4 + t3 * r6;
    return (float16_t)(poly * (float32_t)arm_nn_softmax_exp2i_f16(n));
}

/**
 * @brief LUT-based exp approximation for float16 softmax helpers.
 *
 * Splits `x * log2(e)` into an integer part handled by arm_nn_softmax_exp2i_f16() and a fractional part
 * interpolated linearly from `arm_nn_exp2_lut_f16`, using float32 intermediates.
 *
 * @param[in] x Exponent argument. Clamped to `[-80, 80]` before evaluation.
 * @return Approximation of `exp(x)`.
 */
__STATIC_INLINE float16_t arm_nn_softmax_exp_lut_f16(float16_t x)
{
    const float32_t max_value = 80.0f;
    const float32_t min_value = -80.0f;
    const float32_t log2e = 1.44269504088896341f;
    const float32_t exp2_lut_segments = 256.0f;
    const int32_t exp2_lut_max_index = 255;

    float32_t x_f32 = (float32_t)x;
    x_f32 = ARM_NN_CLAMP(x_f32, max_value, min_value);

    const float32_t t = x_f32 * log2e;
    const int32_t n = arm_nn_softmax_floor_to_int_f16((float16_t)t);
    const float32_t f = t - (float32_t)n;
    const float32_t idx_f = f * exp2_lut_segments;
    int32_t idx = (int32_t)idx_f;

    if (idx < 0)
    {
        idx = 0;
    }
    else if (idx > exp2_lut_max_index)
    {
        idx = exp2_lut_max_index;
    }

    const float32_t frac = idx_f - (float32_t)idx;
    const float32_t y0 = (float32_t)arm_nn_softmax_fp16_from_bits(arm_nn_exp2_lut_f16[idx]);
    const float32_t y1 = (float32_t)arm_nn_softmax_fp16_from_bits(arm_nn_exp2_lut_f16[idx + 1]);
    return (float16_t)((y0 + (y1 - y0) * frac) * (float32_t)arm_nn_softmax_exp2i_f16(n));
}

/**
 * @brief Scalar exp approximation used by the float16 softmax paths.
 *
 * Dispatches to arm_nn_softmax_exp_taylor_f16() when `ARM_NN_USE_EXP_TAYLOR` is defined and to
 * arm_nn_softmax_exp_lut_f16() otherwise.
 *
 * @param[in] x Exponent argument.
 * @return Approximation of `exp(x)`.
 */
__STATIC_INLINE float16_t arm_nn_softmax_exp_scalar_f16(float16_t x)
{
    #if defined(ARM_NN_USE_EXP_TAYLOR)
    return arm_nn_softmax_exp_taylor_f16(x);
    #else
    return arm_nn_softmax_exp_lut_f16(x);
    #endif
}

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
/**
 * @brief Reduce a float16 MVE vector with addition.
 *
 * @param[in] in Vector to reduce.
 * @return Sum of the eight lanes of @p in.
 */
__STATIC_INLINE float16_t arm_nn_vec_reduce_add_f16(float16x8_t in)
{
    float16x8_t tmp_vec = (float16x8_t)vrev32q_s16((int16x8_t)in);
    in = vaddq(tmp_vec, in);
    tmp_vec = (float16x8_t)vrev64q_s32((int32x4_t)in);
    in = vaddq(tmp_vec, in);
    return (float16_t)((_Float16)vgetq_lane_f16(in, 0) + (_Float16)vgetq_lane_f16(in, 4));
}

/**
 * @brief Fold a float16 lane partial into four float32 pair accumulators (reduction kernels).
 *
 * Lanes 2j and 2j+1 are widened exactly and added in float32; the pair sum is added to accumulator lane j, or sets
 * it on the first partial (AmbiqAI/ns-cmsis-nn#586).
 *
 * @param[in,out] acc   Float32 pair accumulators.
 * @param[in]     in    Float16 partial sums.
 * @param[in]     first True for the first partial: the accumulator is set rather than added to.
 */
__STATIC_FORCEINLINE void arm_nn_f16_fold_pairs_f32(float32x4_t *acc, float16x8_t in, const bool first)
{
    float32x4_t pairs = vaddq(arm_nn_vcvtbq_f32_f16(in), arm_nn_vcvttq_f32_f16(in));
    /* Opaque to the optimizer: -ffast-math must not re-pair `acc + (even + odd)` as `(acc + even) + odd`. */
    __asm__("" : "+w"(pairs));
    *acc = first ? pairs : vaddq(*acc, pairs);
}

/**
 * @brief Sum four float32 pair accumulators once: (0+1) + (2+3), each addition rounded to float32.
 *
 * With arm_nn_f16_fold_pairs_f32 this pairs the lanes of a single partial as arm_nn_vec_reduce_add_f16 does:
 * ((0+1) + (2+3)) + ((4+5) + (6+7)) (AmbiqAI/ns-cmsis-nn#586).
 *
 * @param[in] acc Float32 pair accumulators.
 * @return Float32 sum of the four accumulator lanes.
 */
__STATIC_FORCEINLINE float32_t arm_nn_f16_pairs_sum_f32(float32x4_t acc)
{
    acc = vaddq(acc, vrev64q(acc));
    float32_t total = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 2);
    /* Opaque to the optimizer: under -ffast-math the caller's `bias + total` must not be reassociated into the
     * lane sum, or the float32 rounding order would depend on the compiler. */
    __asm__("" : "+t"(total));
    return total;
}

/**
 * @brief arm_nn_f16_pairs_sum_f32 for four accumulators at once.
 *
 * @param[in] acc0 Float32 pair accumulators of output 0.
 * @param[in] acc1 Float32 pair accumulators of output 1.
 * @param[in] acc2 Float32 pair accumulators of output 2.
 * @param[in] acc3 Float32 pair accumulators of output 3.
 * @return Lane j holds the sum of accumulator j, rounded exactly as arm_nn_f16_pairs_sum_f32 rounds it.
 */
__STATIC_FORCEINLINE float32x4_t arm_nn_f16_pairs_sum4_f32(float32x4_t acc0,
                                                           float32x4_t acc1,
                                                           float32x4_t acc2,
                                                           float32x4_t acc3)
{
    /* De-interleaving load: vector i of `v` holds lane i of every accumulator. */
    float32_t t[16];
    vst1q(t, acc0);
    vst1q(t + 4, acc1);
    vst1q(t + 8, acc2);
    vst1q(t + 12, acc3);
    const float32x4x4_t v = vld4q_f32(t);
    float32x4_t s01 = vaddq(v.val[0], v.val[1]);
    float32x4_t s23 = vaddq(v.val[2], v.val[3]);
    /* Opaque to the optimizer, as in arm_nn_f16_pairs_sum_f32: -ffast-math must not re-pair the four sums. */
    __asm__("" : "+w"(s01), "+w"(s23));
    float32x4_t total = vaddq(s01, s23);
    __asm__("" : "+w"(total));
    return total;
}

/**
 * @brief Sum the eight lanes of a float16 partial in float32.
 *
 * Every lane widens exactly; the lanes then sum ((0+1) + (2+3)) + ((4+5) + (6+7)) in float32, as
 * arm_nn_f16_fold_pairs_f32 then arm_nn_f16_pairs_sum_f32 (AmbiqAI/ns-cmsis-nn#586).
 *
 * @param[in] in Float16 partial sums.
 * @return Float32 sum of the eight lanes of @p in.
 */
__STATIC_FORCEINLINE float32_t arm_nn_vec_reduce_add_f16_to_f32(float16x8_t in)
{
    return arm_nn_f16_pairs_sum_f32(vaddq(arm_nn_vcvtbq_f32_f16(in), arm_nn_vcvttq_f32_f16(in)));
}

/**
 * @brief Fold a float16 lane partial into per-lane float32 accumulators.
 *
 * @param[in,out] acc_even Float32 accumulators of lanes 0, 2, 4, 6.
 * @param[in,out] acc_odd  Float32 accumulators of lanes 1, 3, 5, 7.
 * @param[in]     in       Float16 partial sums; widened exactly.
 * @param[in]     first    True for the first partial: the accumulators are set rather than added to.
 */
__STATIC_FORCEINLINE void
arm_nn_f16_fold_lanes_f32(float32x4_t *acc_even, float32x4_t *acc_odd, float16x8_t in, const bool first)
{
    if (first)
    {
        *acc_even = arm_nn_vcvtbq_f32_f16(in);
        *acc_odd = arm_nn_vcvttq_f32_f16(in);
    }
    else
    {
        *acc_even = vaddq(*acc_even, arm_nn_vcvtbq_f32_f16(in));
        *acc_odd = vaddq(*acc_odd, arm_nn_vcvttq_f32_f16(in));
    }
}

/**
 * @brief Round per-lane float32 accumulators to float16 once.
 *
 * @param[in] acc_even Float32 accumulators of lanes 0, 2, 4, 6.
 * @param[in] acc_odd  Float32 accumulators of lanes 1, 3, 5, 7.
 * @return The eight lanes rounded to float16.
 */
__STATIC_FORCEINLINE float16x8_t arm_nn_f16_narrow_lanes_f32(float32x4_t acc_even, float32x4_t acc_odd)
{
    /* Both half-lanes are written by the two narrowing converts, so any seed will do. */
    const float16x8_t lo = arm_nn_vcvtbq_f16_f32(vreinterpretq_f16_f32(acc_even), acc_even);
    return arm_nn_vcvttq_f16_f32(lo, acc_odd);
}

/**
 * @brief MVE float16 exp approximation used by float softmax paths.
 *
 * @param[in] x Vector of exponent arguments.
 * @return Per-lane approximation of `exp(x)`. Lanes that would underflow are flushed to zero.
 */
__STATIC_INLINE float16x8_t arm_nn_vexpq_poly_mve_f16(float16x8_t x)
{
    const float32x4_t x_lo = arm_nn_vcvtbq_f32_f16(x);
    const float32x4_t x_hi = arm_nn_vcvttq_f32_f16(x);
    const int32x4_t m_lo = vcvtq_s32_f32(vmulq(x_lo, 1.4426950408f));
    const int32x4_t m_hi = vcvtq_s32_f32(vmulq(x_hi, 1.4426950408f));
    const float32x4_t val_lo = vfmsq(x_lo, vcvtq_f32_s32(m_lo), vdupq_n_f32(0.6931471805f));
    const float32x4_t val_hi = vfmsq(x_hi, vcvtq_f32_s32(m_hi), vdupq_n_f32(0.6931471805f));

    const float32x4_t a_lo = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[4]), val_lo, arm_nn_exp_poly_coeffs_f16[0]);
    const float32x4_t b_lo = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[6]), val_lo, arm_nn_exp_poly_coeffs_f16[2]);
    const float32x4_t c_lo = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[5]), val_lo, arm_nn_exp_poly_coeffs_f16[1]);
    const float32x4_t d_lo = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[7]), val_lo, arm_nn_exp_poly_coeffs_f16[3]);
    const float32x4_t a_hi = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[4]), val_hi, arm_nn_exp_poly_coeffs_f16[0]);
    const float32x4_t b_hi = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[6]), val_hi, arm_nn_exp_poly_coeffs_f16[2]);
    const float32x4_t c_hi = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[5]), val_hi, arm_nn_exp_poly_coeffs_f16[1]);
    const float32x4_t d_hi = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[7]), val_hi, arm_nn_exp_poly_coeffs_f16[3]);
    const float32x4_t x2_lo = vmulq(val_lo, val_lo);
    const float32x4_t x2_hi = vmulq(val_hi, val_hi);
    const float32x4_t x4_lo = vmulq(x2_lo, x2_lo);
    const float32x4_t x4_hi = vmulq(x2_hi, x2_hi);
    float32x4_t y_lo = vfmaq(vfmaq(a_lo, b_lo, x2_lo), vfmaq(c_lo, d_lo, x2_lo), x4_lo);
    float32x4_t y_hi = vfmaq(vfmaq(a_hi, b_hi, x2_hi), vfmaq(c_hi, d_hi, x2_hi), x4_hi);

    y_lo = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(y_lo), vqshlq_n_s32(m_lo, 23)));
    y_hi = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(y_hi), vqshlq_n_s32(m_hi, 23)));
    y_lo = vdupq_m(y_lo, 0.0f, vcmpltq(m_lo, -126));
    y_hi = vdupq_m(y_hi, 0.0f, vcmpltq(m_hi, -126));

    float16x8_t y = vdupq_n_f16((float16_t)0.0f);
    y = arm_nn_vcvtbq_f16_f32(y, y_lo);
    y = arm_nn_vcvttq_f16_f32(y, y_hi);
    return y;
}
    #endif

/**
 * @brief Copy a float16 vector.
 * @param[out] dst        Destination buffer.
 * @param[in]  src        Source buffer.
 * @param[in]  block_size Number of elements to copy.
 */
__STATIC_FORCEINLINE void
arm_memcpy_f16(float16_t *__RESTRICT dst, const float16_t *__RESTRICT src, uint32_t block_size)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    __asm volatile("   wlstp.16                lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vldrh.16                q0, [%[in]], #16           \n"
                   "   vstrh.16                q0, [%[out]], #16          \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(src), [out] "+r"(dst)
                   : [cnt] "r"(block_size)
                   : "q0", "memory", "r14");
    #else
    __builtin_memcpy(dst, src, (size_t)block_size * sizeof(float16_t));
    #endif
}

/**
 * @brief Set a float16 vector to a constant value.
 * @param[out] dst        Destination buffer.
 * @param[in]  val        Fill value.
 * @param[in]  block_size Number of elements to write.
 */
__STATIC_FORCEINLINE void arm_memset_f16(float16_t *__RESTRICT dst, const float16_t val, uint32_t block_size)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t vec = vdupq_n_f16(val);
    uint32_t i = 0;
    for (; i + 8U <= block_size; i += 8U)
    {
        vst1q(dst + i, vec);
    }
    if (i < block_size)
    {
        const mve_pred16_t p = vctp16q(block_size - i);
        vst1q_p(dst + i, vec, p);
    }
    #else
    for (uint32_t i = 0; i < block_size; ++i)
    {
        dst[i] = val;
    }
    #endif
}

/**
 * @brief Specialized NHWC depthwise `2x5` kernel (float16).
 *
 * @param[in]  x_nhwc  Input tensor in NHWC layout with shape `[batches][2][in_w][in_c]`.
 * @param[in]  batches Number of batches.
 * @param[in]  in_c    Number of input channels.
 * @param[in]  in_w    Input width.
 * @param[in]  ch_mult Channel multiplier; the output has `in_c * ch_mult` channels.
 * @param[in]  kernel  Depthwise weights with shape `[2][5][in_c * ch_mult]`.
 * @param[in]  b       Optional bias vector of `in_c * ch_mult` elements. May be NULL.
 * @param[out] out     Output tensor in NHWC layout with shape `[batches][1][out_w][in_c * ch_mult]`.
 * @param[in]  out_w   Output width. Output position `ow` reads input columns `ow..ow+4`.
 * @param[in]  act_min Lower clamp bound applied to @p out.
 * @param[in]  act_max Upper clamp bound applied to @p out.
 */
void arm_nn_depthwise_conv2x5_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                       int32_t batches,
                                       int32_t in_c,
                                       int32_t in_w,
                                       int32_t ch_mult,
                                       const float16_t *__RESTRICT kernel,
                                       const float16_t *__RESTRICT b,
                                       float16_t *__RESTRICT out,
                                       int32_t out_w,
                                       float16_t act_min,
                                       float16_t act_max);

/**
 * @copydoc arm_nn_depthwise_conv1d_k3_nhwc_f32
 */
void arm_nn_depthwise_conv1d_k3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                         int32_t in_c,
                                         int32_t in_w,
                                         const float16_t *__RESTRICT kernel,
                                         const float16_t *__RESTRICT b,
                                         float16_t *__RESTRICT out,
                                         int32_t out_w);

/**
 * @copydoc arm_nn_conv1d_k5_nhwc_f32
 *
 * @note MVE leg: blockwise float16 accumulation (AmbiqAI/ns-cmsis-nn#586). Input channel c feeds lane c % 8 with
 *       5 taps per channel step; above 32 taps per output, a lane's float16 partial covers at most 6 channel
 *       steps; each block's lanes are folded into float32 pair accumulators (arm_nn_f16_fold_pairs_f32), which
 *       are summed once (arm_nn_f16_pairs_sum_f32), the bias is added in float32 and the total rounds to float16
 *       once. Up to 32 taps: float16 lanes, a float16 reduction and the bias added
 *       in float16, as before, in the order the compiler gives them (it may reorder them under -ffast-math); only
 *       the fold's order is fixed. The scalar leg accumulates in float32 (#449, #465).
 */
void arm_nn_conv1d_k5_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float16_t *__RESTRICT kernel,
                               const float16_t *__RESTRICT b,
                               float16_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w);

/**
 * @copydoc arm_nn_conv1d_k5_nhwc_f16
 *
 * @note Every MVE accumulator lane stays in float16 (no blockwise fold, AmbiqAI/ns-cmsis-nn#586); the scalar leg is
 *       the same as arm_nn_conv1d_k5_nhwc_f16.
 */
void arm_nn_conv1d_k5_nhwc_f16_acc16(const float16_t *__RESTRICT x_nhwc,
                                     int32_t in_c,
                                     int32_t in_w,
                                     const float16_t *__RESTRICT kernel,
                                     const float16_t *__RESTRICT b,
                                     float16_t *__RESTRICT out,
                                     int32_t out_c,
                                     int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=5` (float16, packed weights).
 *
 * The packed kernel uses the same `NTxN` RHS layout as
 * `arm_nn_mat_mult_nt_n_packed_f16`, i.e. `[(5 * in_c)][out_c_block_of_8]`.
 *
 * @param[in]  x_nhwc        Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c          Number of input channels.
 * @param[in]  in_w          Input width. Currently unused by the kernel.
 * @param[in]  kernel_packed Weights packed in output-channel blocks of 8 as described above.
 * @param[in]  b             Optional bias vector of `out_c` elements. May be NULL.
 * @param[out] out           Output row in NHWC layout with shape `[out_w][out_c]`.
 * @param[in]  out_c         Number of output channels.
 * @param[in]  out_w         Output width. Output position `ow` reads input positions `ow..ow+4`.
 *
 * @note Accumulation width per leg. Scalar leg (non-MVE builds and ARM_MATH_AUTOVECTORIZE): bias and products
 *       accumulate in float32 and round to float16 once at the store (AmbiqAI/ns-cmsis-nn#449, #465). MVE leg:
 *       blockwise (#586): a lane's float16 partial covers at most 6 input channels (30 taps, bias first)
 *       before it is widened into a float32 accumulator; one rounding at the store. Up to 32 taps
 *       (in_c <= 6) this is the float16-lane result.
 */
void arm_nn_conv1d_k5_packed_f16(const float16_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float16_t *__RESTRICT kernel_packed,
                                 const float16_t *__RESTRICT b,
                                 float16_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w);

/**
 * @copydoc arm_nn_conv1d_k5_packed_f16
 *
 * @note Every MVE accumulator lane stays in float16 (no blockwise fold, AmbiqAI/ns-cmsis-nn#586); the scalar leg is
 *       the same as arm_nn_conv1d_k5_packed_f16.
 */
void arm_nn_conv1d_k5_packed_f16_acc16(const float16_t *__RESTRICT x_nhwc,
                                       int32_t in_c,
                                       int32_t in_w,
                                       const float16_t *__RESTRICT kernel_packed,
                                       const float16_t *__RESTRICT b,
                                       float16_t *__RESTRICT out,
                                       int32_t out_c,
                                       int32_t out_w);

/**
 * @copydoc arm_nn_conv1d_k3_nhwc_f32
 *
 * @note MVE leg: blockwise float16 accumulation (AmbiqAI/ns-cmsis-nn#586). Input channel c feeds lane c % 8 with
 *       3 taps per channel step; above 32 taps per output, a lane's float16 partial covers at most 10 channel
 *       steps; each block's lanes are folded into float32 pair accumulators (arm_nn_f16_fold_pairs_f32), which
 *       are summed once (arm_nn_f16_pairs_sum_f32), the bias is added in float32 and the total rounds to float16
 *       once. Up to 32 taps: float16 lanes, a float16 reduction and the bias added
 *       in float16, as before, in the order the compiler gives them (it may reorder them under -ffast-math); only
 *       the fold's order is fixed. The scalar leg accumulates in float32 (#449, #465).
 */
void arm_nn_conv1d_k3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float16_t *__RESTRICT kernel,
                               const float16_t *__RESTRICT b,
                               float16_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w);

/**
 * @copydoc arm_nn_conv1d_k3_nhwc_f16
 *
 * @note Every MVE accumulator lane stays in float16 (no blockwise fold, AmbiqAI/ns-cmsis-nn#586); the scalar leg is
 *       the same as arm_nn_conv1d_k3_nhwc_f16.
 */
void arm_nn_conv1d_k3_nhwc_f16_acc16(const float16_t *__RESTRICT x_nhwc,
                                     int32_t in_c,
                                     int32_t in_w,
                                     const float16_t *__RESTRICT kernel,
                                     const float16_t *__RESTRICT b,
                                     float16_t *__RESTRICT out,
                                     int32_t out_c,
                                     int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=3` (float16, packed weights).
 *
 * The packed kernel uses the same `NTxN` RHS layout as
 * `arm_nn_mat_mult_nt_n_packed_f16`, i.e. `[(3 * in_c)][out_c_block_of_8]`.
 *
 * @param[in]  x_nhwc        Input row in NHWC layout with shape `[in_w][in_c]`.
 * @param[in]  in_c          Number of input channels.
 * @param[in]  in_w          Input width. Currently unused by the kernel.
 * @param[in]  kernel_packed Weights packed in output-channel blocks of 8 as described above.
 * @param[in]  b             Optional bias vector of `out_c` elements. May be NULL.
 * @param[out] out           Output row in NHWC layout with shape `[out_w][out_c]`.
 * @param[in]  out_c         Number of output channels.
 * @param[in]  out_w         Output width. Output position `ow` reads input positions `ow..ow+2`.
 *
 * @note Accumulation width per leg. Scalar leg (non-MVE builds and ARM_MATH_AUTOVECTORIZE): bias and products
 *       accumulate in float32 and round to float16 once at the store (AmbiqAI/ns-cmsis-nn#449, #465). MVE leg:
 *       blockwise (#586): a lane's float16 partial covers at most 10 input channels (30 taps, bias first)
 *       before it is widened into a float32 accumulator; one rounding at the store. Up to 32 taps
 *       (in_c <= 10) this is the float16-lane result.
 */
void arm_nn_conv1d_k3_packed_f16(const float16_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float16_t *__RESTRICT kernel_packed,
                                 const float16_t *__RESTRICT b,
                                 float16_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w);

/**
 * @copydoc arm_nn_conv1d_k3_packed_f16
 *
 * @note Every MVE accumulator lane stays in float16 (no blockwise fold, AmbiqAI/ns-cmsis-nn#586); the scalar leg is
 *       the same as arm_nn_conv1d_k3_packed_f16.
 */
void arm_nn_conv1d_k3_packed_f16_acc16(const float16_t *__RESTRICT x_nhwc,
                                       int32_t in_c,
                                       int32_t in_w,
                                       const float16_t *__RESTRICT kernel_packed,
                                       const float16_t *__RESTRICT b,
                                       float16_t *__RESTRICT out,
                                       int32_t out_c,
                                       int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=3`, `s=3` (float16).
 *
 * @copydetails arm_nn_maxpool1d_k3s3_nhwc_f32
 */
void arm_nn_maxpool1d_k3s3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float16_t *__RESTRICT out,
                                    int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=2`, `s=2` without output clamp (float16).
 *
 * @copydetails arm_nn_maxpool1d_k2s2_nhwc_noclip_f32
 */
void arm_nn_maxpool1d_k2s2_nhwc_noclip_f16(const float16_t *__RESTRICT x_nhwc,
                                           int32_t in_c,
                                           int32_t in_w,
                                           float16_t *__RESTRICT out,
                                           int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=2`, `s=2` with clamp (float16).
 *
 * @copydetails arm_nn_maxpool1d_k2s2_nhwc_f32
 */
void arm_nn_maxpool1d_k2s2_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float16_t *__RESTRICT out,
                                    int32_t out_w,
                                    float16_t act_min,
                                    float16_t act_max);

/**
 * @copydoc arm_nn_mat_mult_nt_t_f32
 *
 * @note Accumulation width per leg. MVE legs accumulate blockwise (AmbiqAI/ns-cmsis-nn#586, superseding the
 * float16-lane choice of #417 / #446 for the MVE legs). Up to rhs_cols 32 nothing changes: per-k float16 lanes on the
 * gather path (rhs_cols below the contiguous-K threshold), lane-partial sums then one float16 reduction plus the bias
 * in float16 at rhs_cols 32. Above 32, on the contiguous-K path and the remainder rows, each lane (element k goes to
 * lane k % 8) sums at most 32 of its own taps (256 elements) in float16; each block's lanes are then widened and lanes
 * 2j and 2j+1 added in float32 into pair accumulator j (set by the first block, added to by later ones); the four pair
 * accumulators are summed once as (0+1) + (2+3), so a single block sums ((0+1) + (2+3)) + ((4+5) + (6+7)), the bias is
 * added in float32 and the total rounds to float16 once before the clamp. The float16 reduction up to rhs_cols 32 is
 * ordered by the compiler, which may reorder it under -ffast-math; only the fold's order is fixed.
 * arm_nn_mat_mult_nt_t_f16_acc16 keeps the float16 lanes and float16 reduction throughout. The scalar leg (non-MVE
 * builds and ARM_MATH_AUTOVECTORIZE) accumulates bias and every product in float32 and rounds to float16 once before
 * the clamp (AmbiqAI/ns-cmsis-nn#449, #457).
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_f16(const float16_t *__RESTRICT lhs,
                                             const float16_t *__RESTRICT rhs,
                                             const float16_t *__RESTRICT bias,
                                             float16_t *__RESTRICT dst,
                                             int32_t lhs_rows,
                                             int32_t rhs_rows,
                                             int32_t rhs_cols,
                                             int32_t row_address_offset,
                                             float16_t activation_min,
                                             float16_t activation_max);

/**
 * @brief arm_nn_mat_mult_nt_t_f16 with every MVE accumulator lane in float16 (no blockwise fold).
 *
 * Same arguments, return codes and scalar leg as arm_nn_mat_mult_nt_t_f16; see its accumulation note.
 *
 * @param[in]  lhs                Left-hand matrix, row-major `[lhs_rows, rhs_cols]`.
 * @param[in]  rhs                Right-hand matrix, row-major `[rhs_rows, rhs_cols]` (transposed operand).
 * @param[in]  bias               Optional bias vector of `rhs_rows` elements.
 * @param[out] dst                Output matrix.
 * @param[in]  lhs_rows           Number of rows in @p lhs.
 * @param[in]  rhs_rows           Number of rows in @p rhs.
 * @param[in]  rhs_cols           Shared reduction dimension `K`.
 * @param[in]  row_address_offset Output row stride, expressed in elements.
 * @param[in]  activation_min     Lower clamp bound.
 * @param[in]  activation_max     Upper clamp bound.
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_f16_acc16(const float16_t *__RESTRICT lhs,
                                                   const float16_t *__RESTRICT rhs,
                                                   const float16_t *__RESTRICT bias,
                                                   float16_t *__RESTRICT dst,
                                                   int32_t lhs_rows,
                                                   int32_t rhs_rows,
                                                   int32_t rhs_cols,
                                                   int32_t row_address_offset,
                                                   float16_t activation_min,
                                                   float16_t activation_max);

/**
 * @brief Matrix multiply with non-transposed lhs and packed non-transposed rhs (float16).
 *
 * @param[in]  lhs                Left-hand matrix stored row-major with logical shape `[lhs_rows, rhs_cols]`.
 * @param[in]  rhs_packed         Right-hand matrix with logical shape `[rhs_cols, rhs_rows]`, packed in column blocks
 *                                of 8. The final block uses the same packed stride and inactive tail lanes are ignored.
 * @param[in]  bias               Optional bias vector.
 * @param[out] dst                Output matrix.
 * @param[in]  lhs_rows           Number of rows in @p lhs.
 * @param[in]  rhs_rows           Number of logical output columns in the unpacked rhs matrix.
 * @param[in]  rhs_cols           Shared reduction dimension `K`.
 * @param[in]  row_address_offset Output row stride, expressed in elements.
 * @param[in]  activation_min     Lower clamp bound.
 * @param[in]  activation_max     Upper clamp bound.
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 *
 * @note On non-MVE builds the output clamp is the bit-classified scalar clamp of #380, so a NaN accumulator
 *       (a NaN in @p lhs, @p rhs_packed or @p bias) propagates to @p dst at every optimization level
 *       on the gated toolchains,
 *       including the shipped -Ofast. On MVE builds the clamp is vmaxnmq/vminnmq with no NaN restore, so a
 *       NaN resolves to a clamp bound there instead.
 *
 * @note Accumulation width per leg: the MVE leg accumulates blockwise (AmbiqAI/ns-cmsis-nn#586): one
 *       lane per output column, per-k, the bias opening the first block; every 32 k the float16
 *       partial is widened exactly into per-lane float32 accumulators, which round to float16 once
 *       before the clamp (rhs_cols up to 32: exactly the float16-lane result;
 *       arm_nn_mat_mult_nt_n_packed_f16_acc16 keeps float16 lanes throughout). The scalar leg (non-MVE builds and
 * ARM_MATH_AUTOVECTORIZE) accumulates bias and every product in float32 and rounds to float16 once before the clamp
 *       (AmbiqAI/ns-cmsis-nn#449, #457).
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_n_packed_f16(const float16_t *__RESTRICT lhs,
                                                    const float16_t *__RESTRICT rhs_packed,
                                                    const float16_t *__RESTRICT bias,
                                                    float16_t *__RESTRICT dst,
                                                    int32_t lhs_rows,
                                                    int32_t rhs_rows,
                                                    int32_t rhs_cols,
                                                    int32_t row_address_offset,
                                                    float16_t activation_min,
                                                    float16_t activation_max);

/**
 * @brief arm_nn_mat_mult_nt_n_packed_f16 with every MVE accumulator lane in float16 (no blockwise fold).
 *
 * Same arguments, return codes and scalar leg as arm_nn_mat_mult_nt_n_packed_f16; see its accumulation note.
 *
 * @param[in]  lhs                Left-hand matrix stored row-major with logical shape `[lhs_rows, rhs_cols]`.
 * @param[in]  rhs_packed         Right-hand matrix packed in column blocks of 8.
 * @param[in]  bias               Optional bias vector.
 * @param[out] dst                Output matrix.
 * @param[in]  lhs_rows           Number of rows in @p lhs.
 * @param[in]  rhs_rows           Number of logical output columns in the unpacked rhs matrix.
 * @param[in]  rhs_cols           Shared reduction dimension `K`.
 * @param[in]  row_address_offset Output row stride, expressed in elements.
 * @param[in]  activation_min     Lower clamp bound.
 * @param[in]  activation_max     Upper clamp bound.
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_n_packed_f16_acc16(const float16_t *__RESTRICT lhs,
                                                          const float16_t *__RESTRICT rhs_packed,
                                                          const float16_t *__RESTRICT bias,
                                                          float16_t *__RESTRICT dst,
                                                          int32_t lhs_rows,
                                                          int32_t rhs_rows,
                                                          int32_t rhs_cols,
                                                          int32_t row_address_offset,
                                                          float16_t activation_min,
                                                          float16_t activation_max);

/**
 * @brief Update LSTM function for an iteration step using float16 input, output and state.
 *
 * @param[in]   data_in                         Data input pointer.
 * @param[in]   hidden_in                       Hidden state / recurrent input pointer. May be NULL for the first step.
 * @param[out]  hidden_out                      Hidden state / recurrent output pointer.
 * @param[in]   params                          Struct containing all information about the LSTM operator.
 * @param[in,out] buffers                       Struct containing pointers to mutable cell-state storage.
 * @param[in]   batch_offset                    Number of timesteps between consecutive batches.
 * @return                                      ARM_CMSIS_NN_SUCCESS on success, or ARM_CMSIS_NN_ARG_ERROR on
 *                                              invalid arguments (NULL data_in/hidden_out/params/buffers or
 *                                              buffers->cell_state, batch_offset <= 0).
 */
arm_cmsis_nn_status arm_nn_lstm_step_f16(const float16_t *data_in,
                                         const float16_t *hidden_in,
                                         float16_t *hidden_out,
                                         const cmsis_nn_lstm_params_f16 *params,
                                         cmsis_nn_lstm_context_f16 *buffers,
                                         const int32_t batch_offset);

/**
 * @brief Update GRU function for a single iteration step using float16 data.
 *
 * @param[in]   data_in       Data input pointer for this time step.
 * @param[in]   hidden_in     Recurrent input pointer. NULL for the first step (h_prev = 0).
 * @param[out]  hidden_out    Hidden-state output pointer for this time step.
 * @param[in]   params        Struct describing the GRU operator.
 * @param[in,out] buffers   Scratch buffers. temp1 (>= hidden_size) is required when reset_after == 0.
 * @param[in]   batch_offset  Number of timesteps between consecutive batches.
 * @return                    ARM_CMSIS_NN_SUCCESS on success, or ARM_CMSIS_NN_ARG_ERROR on
 *                            invalid arguments (NULL data_in/hidden_out/params, batch_offset <= 0,
 *                            or missing temp1 when reset_after == 0).
 */
arm_cmsis_nn_status arm_nn_gru_step_f16(const float16_t *data_in,
                                        const float16_t *hidden_in,
                                        float16_t *hidden_out,
                                        const cmsis_nn_gru_params_f16 *params,
                                        cmsis_nn_gru_context_f16 *buffers,
                                        const int32_t batch_offset);

/**
 * @copydoc arm_nn_pack_conv_patch_f32
 */
void arm_nn_pack_conv_patch_f16(const float16_t *__RESTRICT input,
                                int32_t in_h,
                                int32_t in_w,
                                int32_t in_c,
                                int32_t kernel_h,
                                int32_t kernel_w,
                                int32_t stride_h,
                                int32_t stride_w,
                                int32_t pad_h,
                                int32_t pad_w,
                                int32_t dilation_h,
                                int32_t dilation_w,
                                int32_t out_y,
                                int32_t out_x,
                                float16_t pad_value,
                                float16_t *__RESTRICT patch_row);

/**
 * @brief Specialized softmax helper for a single float16 row of length 2.
 *
 * @param[in]  in   Pointer to two contiguous float16 input values.
 * @param[out] out  Pointer to two contiguous float16 output values.
 */
void arm_nn_softmax_1x2_f16(const float16_t *in, float16_t *out);

#endif /* ARM_NN_ENABLE_F16 */

#if ARM_NN_ENABLE_F32

/**
 * @brief Update LSTM function for an iteration step using float32 input, output and state.
 *
 * @param[in]   data_in                         Data input pointer.
 * @param[in]   hidden_in                       Hidden state / recurrent input pointer. May be NULL for the first step.
 * @param[out]  hidden_out                      Hidden state / recurrent output pointer.
 * @param[in]   params                          Struct containing all information about the LSTM operator.
 * @param[in,out] buffers                       Struct containing pointers to mutable cell-state storage.
 * @param[in]   batch_offset                    Number of timesteps between consecutive batches.
 * @return                                      ARM_CMSIS_NN_SUCCESS on success, or ARM_CMSIS_NN_ARG_ERROR on
 *                                              invalid arguments (NULL data_in/hidden_out/params/buffers or
 *                                              buffers->cell_state, batch_offset <= 0).
 */
arm_cmsis_nn_status arm_nn_lstm_step_f32(const float32_t *data_in,
                                         const float32_t *hidden_in,
                                         float32_t *hidden_out,
                                         const cmsis_nn_lstm_params_f32 *params,
                                         cmsis_nn_lstm_context_f32 *buffers,
                                         const int32_t batch_offset);

/**
 * @brief Update GRU function for a single iteration step using float32 data.
 *
 * @param[in]   data_in       Data input pointer for this time step.
 * @param[in]   hidden_in     Recurrent input pointer. NULL for the first step (h_prev = 0).
 * @param[out]  hidden_out    Hidden-state output pointer for this time step.
 * @param[in]   params        Struct describing the GRU operator.
 * @param[in,out] buffers   Scratch buffers. temp1 (>= hidden_size) is required when reset_after == 0.
 * @param[in]   batch_offset  Number of timesteps between consecutive batches.
 * @return                    ARM_CMSIS_NN_SUCCESS on success, or ARM_CMSIS_NN_ARG_ERROR on
 *                            invalid arguments (NULL data_in/hidden_out/params, batch_offset <= 0,
 *                            or missing temp1 when reset_after == 0).
 */
arm_cmsis_nn_status arm_nn_gru_step_f32(const float32_t *data_in,
                                        const float32_t *hidden_in,
                                        float32_t *hidden_out,
                                        const cmsis_nn_gru_params_f32 *params,
                                        cmsis_nn_gru_context_f32 *buffers,
                                        const int32_t batch_offset);

#endif /* ARM_NN_ENABLE_F32 */

/**
 * @defgroup floatConvRoutes Float convolution route predicates
 * The rules by which arm_convolve_f16(), arm_convolve_f16_acc16() and arm_convolve_f32() choose a route, in their
 * dispatch order: 1x1, 1xN, conv1d k5, conv1d k3, small-C (MVE float builds), packed-patch GEMM (when ctx holds one
 * patch row), direct. The routers, their buffer-size queries and the direct entries of each route share these, so
 * that a caller selecting the entry per layer ahead of time takes the route the router takes.
 * @{
 */

/** Lanes of the float16 small-C kernel: the route takes fewer input channels than this. */
#define ARM_NN_CONV_SMALL_C_F16_LANES (8)
/** Lanes of the float32 small-C kernel: the route takes fewer input channels than this. */
#define ARM_NN_CONV_SMALL_C_F32_LANES (4)

/**
 * @brief The 1x1 route: a 1x1 filter with no padding.
 * @param[in] padding     Spatial zero-padding
 * @param[in] filter_dims Filter dimensions [C_OUT, KH, KW, C_IN]
 * @return true if the router takes the 1x1 route
 */
__STATIC_INLINE bool arm_nn_conv_flt_is_1x1(const cmsis_nn_tile *padding, const cmsis_nn_dims *filter_dims)
{
    return filter_dims->h == 1 && filter_dims->w == 1 && padding->h == 0 && padding->w == 0;
}

/**
 * @brief A conv1d with a 1 x k filter: batch 1, input and output height 1, unit stride and dilation, no padding.
 * @param[in] stride      Spatial stride
 * @param[in] padding     Spatial zero-padding
 * @param[in] dilation    Spatial dilation
 * @param[in] input_dims  Input dimensions [N, H, W, C_IN]
 * @param[in] filter_dims Filter dimensions [C_OUT, KH, KW, C_IN]
 * @param[in] output_dims Output dimensions [N, H, W, C_OUT]
 * @param[in] k           Filter width
 * @return true for the conv1d k5 route with k = 5 and the conv1d k3 route with k = 3
 */
__STATIC_INLINE bool arm_nn_conv_flt_is_1d_k(const cmsis_nn_tile *stride,
                                             const cmsis_nn_tile *padding,
                                             const cmsis_nn_tile *dilation,
                                             const cmsis_nn_dims *input_dims,
                                             const cmsis_nn_dims *filter_dims,
                                             const cmsis_nn_dims *output_dims,
                                             const int32_t k)
{
    return input_dims->n == 1 && input_dims->h == 1 && output_dims->h == 1 && filter_dims->h == 1 &&
        filter_dims->w == k && stride->h == 1 && stride->w == 1 && padding->h == 0 && padding->w == 0 &&
        dilation->h == 1 && dilation->w == 1;
}

/**
 * @brief The shape part of the 1xN route: input, output and filter height 1, a filter wider than 1, unit stride and
 *        dilation in height, unit dilation in width, a positive width stride and no height padding. Unless
 *        NN_DISABLE_SPECIALIZATION is defined, the routers take the conv1d routes for their shapes first. The router
 *        also needs ctx to hold arm_convolve_1_x_n_f16_get_buffer_size() (or _f32).
 * @param[in] stride      Spatial stride
 * @param[in] padding     Spatial zero-padding
 * @param[in] dilation    Spatial dilation
 * @param[in] input_dims  Input dimensions [N, H, W, C_IN]
 * @param[in] filter_dims Filter dimensions [C_OUT, KH, KW, C_IN]
 * @param[in] output_dims Output dimensions [N, H, W, C_OUT]
 * @return true if the shape takes the 1xN route when ctx is large enough and no conv1d route claims it
 */
__STATIC_INLINE bool arm_nn_conv_flt_is_1xn(const cmsis_nn_tile *stride,
                                            const cmsis_nn_tile *padding,
                                            const cmsis_nn_tile *dilation,
                                            const cmsis_nn_dims *input_dims,
                                            const cmsis_nn_dims *filter_dims,
                                            const cmsis_nn_dims *output_dims)
{
    return input_dims->h == 1 && output_dims->h == 1 && filter_dims->h == 1 && filter_dims->w > 1 && stride->h == 1 &&
        stride->w > 0 && padding->h == 0 && dilation->h == 1 && dilation->w == 1;
}

/**
 * @brief The float16 small-C route: fewer input channels than ARM_NN_CONV_SMALL_C_F16_LANES, a positive output
 *        depth and width, and every gather and scatter offset within 16 bits.
 * @param[in] stride      Spatial stride
 * @param[in] padding     Spatial zero-padding
 * @param[in] dilation    Spatial dilation
 * @param[in] input_dims  Input dimensions [N, H, W, C_IN]
 * @param[in] filter_dims Filter dimensions [C_OUT, KH, KW, C_IN]
 * @param[in] output_dims Output dimensions [N, H, W, C_OUT]
 * @return true if the float16 routers take the small-C route on MVE float builds
 */
__STATIC_INLINE bool arm_nn_conv_f16_is_small_c(const cmsis_nn_tile *stride,
                                                const cmsis_nn_tile *padding,
                                                const cmsis_nn_tile *dilation,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims)
{
    if (input_dims->c <= 0 || input_dims->c >= ARM_NN_CONV_SMALL_C_F16_LANES || output_dims->c <= 0 ||
        output_dims->w <= 0)
    {
        return false;
    }
    /* u16 gather/scatter offsets are relative to one input row / one output position group, and every
     * reachable column (including the padded ones, which wrap) must stay inside the offset type. */
    const size_t reach = (size_t)input_dims->w + (size_t)padding->w +
        (size_t)(ARM_NN_CONV_SMALL_C_F16_LANES - 1) * (size_t)stride->w +
        (size_t)(filter_dims->w - 1) * (size_t)dilation->w;
    return reach * (size_t)input_dims->c <= (size_t)UINT16_MAX &&
        (size_t)ARM_NN_CONV_SMALL_C_F16_LANES * (size_t)output_dims->c <= (size_t)UINT16_MAX;
}

/**
 * @brief The float32 small-C route: fewer input channels than ARM_NN_CONV_SMALL_C_F32_LANES and a positive output
 *        depth and width.
 * @param[in] input_dims  Input dimensions [N, H, W, C_IN]
 * @param[in] output_dims Output dimensions [N, H, W, C_OUT]
 * @return true if arm_convolve_f32() takes the small-C route on MVE float builds
 */
__STATIC_INLINE bool arm_nn_conv_f32_is_small_c(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *output_dims)
{
    return input_dims->c > 0 && input_dims->c < ARM_NN_CONV_SMALL_C_F32_LANES && output_dims->c > 0 &&
        output_dims->w > 0;
}

/**
 * @brief The shape part of the packed-patch GEMM route: at least ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_OC output
 *        channels and ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_POS output positions per batch (the float32 thresholds
 *        are equal). The router also needs ctx to hold one patch row (KH x KW x C_IN elements).
 * @param[in] output_dims Output dimensions [N, H, W, C_OUT]
 * @return true if the shape takes the packed-patch GEMM route when ctx is large enough
 */
__STATIC_INLINE bool arm_nn_conv_flt_is_patch_gemm(const cmsis_nn_dims *output_dims)
{
    return output_dims->c >= ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_OC &&
        (int64_t)output_dims->h * output_dims->w >= ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_POS;
}

/**
 * @} end of floatConvRoutes group
 */

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* ARM_NNSUPPORTFUNCTIONS_FLT_H */
