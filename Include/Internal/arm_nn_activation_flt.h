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
 * Title:        arm_nn_activation_flt.h
 * Description:  Internal floating-point activation helper utilities
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_ACTIVATION_FLT_H
#define ARM_NN_ACTIVATION_FLT_H

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

    /*
     * Shared geometry of arm_nn_tanh_lut384_f32, used by both the scalar and the
     * MVE tanh helpers so the two legs cannot drift apart.
     *
     * The table samples tanh(x) on a uniform grid over |x| <= ARM_NN_TANH_F32_XMAX
     * with ARM_NN_TANH_F32_LUT_SEGMENTS interpolation segments (and one extra entry
     * so lut[idx + 1] is always in range). Grid spacing is 6/384 == 1/64, the same
     * spacing the earlier [0, 4] table used, so the index multiplier
     * (SEGMENTS / XMAX == 64) is unchanged and every output for |x| <= 4 is
     * bit-identical to the previous table.
     */
    #define ARM_NN_TANH_F32_XMAX (6.0f)
    #define ARM_NN_TANH_F32_LUT_SEGMENTS (384)
    #define ARM_NN_TANH_F32_LUT_MAX_IDX (ARM_NN_TANH_F32_LUT_SEGMENTS - 1)

__STATIC_INLINE float32_t arm_nn_hardswish_scalar_f32(float32_t x)
{
    float32_t t = x * (1.0f / 6.0f) + 0.5f;
    t = CLAMP(t, 1.0f, 0.0f);
    return x * t;
}

/*
 * Tanh deliberately uses different scalar approximations by dtype.
 * Float32 keeps a LUT + linear interpolation path to preserve tighter
 * accuracy over a wider input range, while float16 uses a compact rational
 * approximation because half precision does not benefit as much from a
 * larger table and the lower-order form is sufficient for its target error.
 *
 * Accuracy: max interpolation error is 2.35e-5 inside the table window
 * (|x| < ARM_NN_TANH_F32_XMAX); at or outside the boundary the helper
 * saturates to +/-1.0, a step of 1 - tanh(6) ~= 1.2e-5. Every result for
 * |x| <= 4 is bit-identical to the earlier 257-entry [0, 4] table.
 *
 * NaN contract, and why the two legs differ. NOTE this contract holds only in
 * builds WITHOUT -ffinite-math-only. The default library build uses -Ofast
 * (CMSIS_OPTIMIZATION_LEVEL in the top-level CMakeLists.txt), which sets
 * __FINITE_MATH_ONLY__ and lets the compiler delete the NaN test below
 * outright -- there, NaN input is simply outside the language contract. It
 * remains memory-safe: the conversion saturates and NaN propagates through
 * frac, so a NaN still comes out, but that is an observation about today's
 * codegen, not a guarantee.
 *   - Scalar (this helper) propagates NaN. The saturation test is written as
 *     !(ax < xmax) so NaN, which compares unordered, takes the cold branch;
 *     that also keeps NaN away from the float->int conversion below, which
 *     would otherwise be undefined behaviour. A quiet NaN passes through with
 *     its payload and sign intact; a signalling NaN is quieted on return, so
 *     its invalid-operation exception is raised here rather than being handed
 *     to the caller.
 *   - MVE (arm_nn_vtanh_lut_direct_mve_f32) does not. vminnmq is IEEE minNum,
 *     which returns the numeric operand when the other is a quiet NaN, so a
 *     qNaN lane is replaced by xmax and interpolates to tanh(xmax) ~=
 *     0.9999877 -- and always with a NEGATIVE sign, giving -0.9999877
 *     (0xbf7fff32). That sign is not a typo and not what C `x < 0.0f`
 *     semantics would suggest: Armv8.1-M defines VCMP `lt` as the logical
 *     inverse of `ge`, so it is TRUE for unordered operands, and the
 *     vnegq_m negation predicate therefore fires on every NaN lane. A
 *     signalling NaN is quieted and returned by minNum, and is then negated
 *     the same way, so an sNaN lane yields a negated default qNaN
 *     (0xffc00000). Restoring NaN in general would cost an extra compare and
 *     select in the vector loop body, which this helper's callers (LSTM/GRU
 *     step kernels) run per element. NaN is not a supported input to these
 *     kernels, so the divergence is accepted rather than paid for. Finite
 *     inputs, including |x| == xmax, agree exactly across legs.
 */
__STATIC_INLINE float32_t arm_nn_tanh_scalar_ref_f32(float32_t x)
{
    float32_t ax = (x < 0.0f) ? -x : x;
    const float32_t xmax = ARM_NN_TANH_F32_XMAX;

    if (!(ax < xmax))
    {
        /* NaN (unordered) lands here too; propagate it rather than saturating.
         * The + 0.0f is what quiets a signalling NaN, matching what the old
         * code did incidentally by running the sNaN through the interpolation
         * arithmetic. Returning x bare would hand an sNaN straight back and
         * defer its invalid-operation exception to whatever the caller does
         * next. IEEE addition propagates a quiet NaN operand unchanged, so
         * qNaN payload and sign still pass through untouched. */
        if (ax != ax)
        {
            return x + 0.0f;
        }
        return (x < 0.0f) ? -1.0f : 1.0f;
    }

    /* No index clamp: the strict test above leaves 0 <= ax < xmax, and the
     * scale is an exact power of two, so 0 <= t < SEGMENTS and idx is in
     * [0, SEGMENTS - 1] by construction -- lut[idx + 1] stays in range. The
     * previous [0, 4] table needed a clamp because its test admitted ax ==
     * xmax; keeping one here would cost three instructions per call in the hot
     * path, because 383 (unlike 255) is not a `usat` saturation boundary. */
    const float32_t t = ax * ((float32_t)ARM_NN_TANH_F32_LUT_SEGMENTS / xmax);
    const int32_t idx = (int32_t)t;
    const float32_t frac = t - (float32_t)idx;
    const float32_t y0 = arm_nn_tanh_lut384_f32[idx];
    const float32_t y1 = arm_nn_tanh_lut384_f32[idx + 1];
    const float32_t y = y0 + (y1 - y0) * frac;
    return (x < 0.0f) ? -y : y;
}

/*
 * Keep the float32 sigmoid on the non-positive exp domain used by softmax.
 * This mirrors the float16 helper and avoids needlessly evaluating exp()
 * on large positive arguments.
 */
__STATIC_INLINE float32_t arm_nn_sigmoid_scalar_f32(float32_t x)
{
    if (x >= 0.0f)
    {
        const float32_t e = arm_nn_softmax_exp_scalar_f32(-x);
        return 1.0f / (1.0f + e);
    }

    const float32_t e = arm_nn_softmax_exp_scalar_f32(x);
    return e / (1.0f + e);
}

__STATIC_INLINE float32_t arm_nn_apply_activation_type_f32(float32_t x,
                                                           arm_nn_activation_type_flt type,
                                                           float32_t act_param)
{
    switch (type)
    {
    case ARM_NN_FLT_ACT_NONE:
        return x;
    case ARM_NN_FLT_ACT_SIGMOID:
        return arm_nn_sigmoid_scalar_f32(x);
    case ARM_NN_FLT_ACT_RELU:
        return (x < 0.0f) ? 0.0f : x;
    case ARM_NN_FLT_ACT_RELU6: {
        const float32_t y = (x < 0.0f) ? 0.0f : x;
        return (y > 6.0f) ? 6.0f : y;
    }
    case ARM_NN_FLT_ACT_TANH:
        return arm_nn_tanh_scalar_ref_f32(x);
    case ARM_NN_FLT_ACT_HARDSWISH:
        return arm_nn_hardswish_scalar_f32(x);
    case ARM_NN_FLT_ACT_LEAKY_RELU:
        return (x >= 0.0f) ? x : (act_param * x);
    default:
        return x;
    }
}

__STATIC_INLINE float32_t arm_nn_clamp_scalar_f32(float32_t x, float32_t min_v, float32_t max_v)
{
    // Comparisons are false for NaN, so NaN propagates (TFLite semantics)
    return (x < min_v) ? min_v : ((x > max_v) ? max_v : x);
}

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE float32x4_t arm_nn_clamp_mve_f32(float32x4_t x, float32x4_t min_v, float32x4_t max_v)
{
    x = vmaxnmq(x, min_v);
    x = vminnmq(x, max_v);
    return x;
}

__STATIC_INLINE float32x4_t arm_nn_clamp_propagate_nan_mve_f32(float32x4_t x, float32x4_t min_v, float32x4_t max_v)
{
    // vmaxnmq/vminnmq are IEEE maxNum/minNum and suppress NaN, so restore
    // NaN lanes afterwards to match the scalar path and TFLite
    const mve_pred16_t nan_p = vcmpneq(x, x);
    float32x4_t y = vmaxnmq(x, min_v);
    y = vminnmq(y, max_v);
    return vpselq(x, y, nan_p);
}

/*
 * Vector twin of arm_nn_tanh_scalar_ref_f32, sharing arm_nn_tanh_lut384_f32 and
 * the ARM_NN_TANH_F32_* geometry above so the two legs stay in step.
 *
 * Finite inputs agree with the scalar leg exactly, including at |x| == xmax:
 * the predicate below is >=, matching the scalar helper's !(ax < xmax), so
 * both legs saturate at the boundary rather than one interpolating to
 * lut[SEGMENTS] there. NaN is the one place the legs differ: a qNaN lane comes
 * out as -tanh(xmax) and an sNaN lane as a negated default qNaN, both NEGATIVE
 * because the vnegq_m predicate below uses VCMP `lt`, which Armv8.1-M defines
 * as !ge and is therefore true for unordered operands. See the scalar helper's
 * comment for the full split and why the divergence is accepted.
 */
__STATIC_INLINE float32x4_t arm_nn_vtanh_lut_direct_mve_f32(float32x4_t x)
{
    float32x4_t ax = vabsq(x);
    /* Splat once and compare vector-to-vector. The scalar-operand form of
     * vcmpgeq() would force xmax into a GP register on every iteration,
     * because 0x40c00000 (6.0f) is not a Thumb-2 modified immediate and so
     * becomes a literal-pool load inside the loop. Against a vector operand
     * the constant is materialised once (literal-pool load plus vdup) and
     * hoisted above the `dls`, leaving the loop body free of it. */
    const float32x4_t vmax = vdupq_n_f32(ARM_NN_TANH_F32_XMAX);
    const mve_pred16_t sat_p = vcmpgeq(ax, vmax);
    ax = vminnmq(ax, vmax);
    const float32x4_t t = vmulq(ax, (float32_t)ARM_NN_TANH_F32_LUT_SEGMENTS / ARM_NN_TANH_F32_XMAX);
    uint32x4_t idx = vcvtmq_u32_f32(t);
    idx = vminq(idx, vdupq_n_u32((uint32_t)ARM_NN_TANH_F32_LUT_MAX_IDX));
    const float32x4_t frac = vsubq(t, vcvtq_f32_u32(idx));
    const float32x4_t y0 = vldrwq_gather_shifted_offset((const float32_t *)arm_nn_tanh_lut384_f32, idx);
    const float32x4_t y1 = vldrwq_gather_shifted_offset((const float32_t *)arm_nn_tanh_lut384_f32, vaddq(idx, (uint32_t)1U));
    float32x4_t y = vfmaq(y0, vsubq(y1, y0), frac);
    y = vpselq(vdupq_n_f32(1.0f), y, sat_p);
    return vnegq_m(y, y, vcmpltq(x, 0.0f));
}

__STATIC_INLINE float32x4_t arm_nn_vhardswish_mve_f32(float32x4_t x)
{
    float32x4_t t = vfmaq(vdupq_n_f32(0.5f), x, (1.0f / 6.0f));
    t = vmaxnmq(t, vdupq_n_f32(0.0f));
    t = vminnmq(t, vdupq_n_f32(1.0f));
    return vmulq(x, t);
}
    #endif

__STATIC_INLINE void
arm_nn_vector_clamp_f32(float32_t *data, int32_t block_size, float32_t activation_min, float32_t activation_max)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t vmin = vdupq_n_f32(activation_min);
    const float32x4_t vmax = vdupq_n_f32(activation_max);
    for (int32_t i = 0; i < block_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
        float32x4_t v = vld1q_z(data + i, p);
        v = arm_nn_clamp_mve_f32(v, vmin, vmax);
        vst1q_p(data + i, v, p);
    }
    #else
    for (int32_t i = 0; i < block_size; ++i)
    {
        data[i] = CLAMP(data[i], activation_max, activation_min);
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F32 */

#if ARM_NN_ENABLE_F16

__STATIC_INLINE float16_t arm_nn_clamp_scalar_f16(float16_t x, float16_t min_v, float16_t max_v)
{
    /* NaN propagates (TFLite semantics), matching arm_nn_clamp_scalar_f32 and
     * the MVE path in arm_nn_clamp_propagate_nan_mve_f16(). Bounds are assumed
     * ordered; inverted bounds are unspecified. */
    const _Float16 y = arm_nn_clamp_propagate_nan_f16h((_Float16)x, (_Float16)min_v, (_Float16)max_v);
    return (float16_t)y;
}

__STATIC_INLINE float16_t arm_nn_hardswish_scalar_f16(float16_t x)
{
    _Float16 t = (_Float16)x * (_Float16)(1.0f / 6.0f) + (_Float16)0.5f;
    t = arm_nn_clamp_f16h(t, (_Float16)1.0f, (_Float16)0.0f);
    return (float16_t)((_Float16)x * t);
}

__STATIC_INLINE float16_t arm_nn_tanh_scalar_ref_f16(float16_t x)
{
    const _Float16 ax = arm_nn_abs_f16h((_Float16)x);
    if (ax > (_Float16)arm_nn_tanh_approx_coeffs_f16[0])
    {
        /* Select the saturation sign in float32: SFmode conditional moves are
         * unaffected by GCC PR target/118460 (HFmode only). */
        const float32_t sign = ((float32_t)(_Float16)x < 0.0f) ? -1.0f : 1.0f;
        return (float16_t)(_Float16)sign;
    }

    const _Float16 x2 = (_Float16)x * (_Float16)x;
    const _Float16 num = (_Float16)x * ((_Float16)arm_nn_tanh_approx_coeffs_f16[1] + x2);
    const _Float16 den = (_Float16)arm_nn_tanh_approx_coeffs_f16[1] + (_Float16)arm_nn_tanh_approx_coeffs_f16[2] * x2;
    return (float16_t)(num / den);
}

/*
 * Keep the float16 sigmoid on the non-positive exp domain used by softmax.
 * This avoids relying on the positive-side exp
 */
__STATIC_INLINE float16_t arm_nn_sigmoid_scalar_f16(float16_t x)
{
    if ((_Float16)x >= (_Float16)0.0f)
    {
        const float32_t e = (float32_t)arm_nn_softmax_exp_scalar_f16((float16_t)(-(_Float16)x));
        return (float16_t)(1.0f / (1.0f + e));
    }

    const float32_t e = (float32_t)arm_nn_softmax_exp_scalar_f16((float16_t)(_Float16)x);
    return (float16_t)(e / (1.0f + e));
}

__STATIC_INLINE float16_t arm_nn_apply_activation_type_f16(float16_t x,
                                                           arm_nn_activation_type_flt type,
                                                           float16_t act_param)
{
    switch (type)
    {
    case ARM_NN_FLT_ACT_NONE:
        return x;
    case ARM_NN_FLT_ACT_SIGMOID:
        return arm_nn_sigmoid_scalar_f16(x);
    case ARM_NN_FLT_ACT_RELU:
        /* NaN propagates, matching the f32 path and the TFLite reference. */
        return (float16_t)arm_nn_propagate_nan_f16h((_Float16)x, arm_nn_max_f16h((_Float16)x, (_Float16)0.0f));
    case ARM_NN_FLT_ACT_RELU6:
        return (float16_t)arm_nn_clamp_propagate_nan_f16h((_Float16)x, (_Float16)0.0f, (_Float16)6.0f);
    case ARM_NN_FLT_ACT_TANH:
        return arm_nn_tanh_scalar_ref_f16(x);
    case ARM_NN_FLT_ACT_HARDSWISH:
        return arm_nn_hardswish_scalar_f16(x);
    case ARM_NN_FLT_ACT_LEAKY_RELU: {
        /* max(x, 0) + alpha * min(x, 0) avoids a scalar _Float16 conditional select (GCC PR target/118460).
         * vmaxnm/vminnm suppress NaN, so restore it to match the f32 path. */
        const _Float16 pos = arm_nn_max_f16h((_Float16)x, (_Float16)0.0f);
        const _Float16 neg = arm_nn_min_f16h((_Float16)x, (_Float16)0.0f);
        return (float16_t)arm_nn_propagate_nan_f16h((_Float16)x, pos + (_Float16)act_param * neg);
    }
    default:
        return x;
    }
}

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE float16x8_t arm_nn_clamp_mve_f16(float16x8_t x, float16x8_t min_v, float16x8_t max_v)
{
    x = vmaxnmq(x, min_v);
    x = vminnmq(x, max_v);
    return x;
}

__STATIC_INLINE float16x8_t arm_nn_clamp_propagate_nan_mve_f16(float16x8_t x, float16x8_t min_v, float16x8_t max_v)
{
    // vmaxnmq/vminnmq are IEEE maxNum/minNum and suppress NaN, so restore
    // NaN lanes afterwards to match the scalar path and TFLite
    const mve_pred16_t nan_p = vcmpneq(x, x);
    float16x8_t y = vmaxnmq(x, min_v);
    y = vminnmq(y, max_v);
    return vpselq(x, y, nan_p);
}

__STATIC_INLINE float16x8_t arm_nn_vtanh_lut_direct_mve_f16(float16x8_t x)
{
    float16x8_t ax = vabsq(x);
    const mve_pred16_t sat_p = vcmpgtq(ax, (float16_t)4.0f);
    ax = vminnmq(ax, vdupq_n_f16((float16_t)4.0f));
    const uint16_t xmax = 4U;
    const uint16_t lut_tbl_max_idx = 256U;
    const float16x8_t t = vmulq(ax, (float16_t)(lut_tbl_max_idx / xmax));
    uint16x8_t idx = vcvtmq_u16_f16(t);
    idx = vminq(idx, vdupq_n_u16(255U));
    const float16x8_t frac = vsubq(t, vcvtq_f16_u16(idx));
    const float16x8_t y0 = vldrhq_gather_shifted_offset((const float16_t *)arm_nn_tanh_lut256_f16, idx);
    const float16x8_t y1 = vldrhq_gather_shifted_offset((const float16_t *)arm_nn_tanh_lut256_f16, vaddq(idx, (uint16_t)1U));
    float16x8_t y = vfmaq(y0, vsubq(y1, y0), frac);
    y = vpselq(vdupq_n_f16((float16_t)1.0f), y, sat_p);
    return vnegq_m(y, y, vcmpltq(x, (float16_t)0.0f));
}

__STATIC_INLINE float16x8_t arm_nn_vhardswish_mve_f16(float16x8_t x)
{
    float16x8_t t = vfmaq(vdupq_n_f16((float16_t)0.5f), x, (float16_t)(1.0f / 6.0f));
    t = vmaxnmq(t, vdupq_n_f16((float16_t)0.0f));
    t = vminnmq(t, vdupq_n_f16((float16_t)1.0f));
    return vmulq(x, t);
}
    #endif

__STATIC_INLINE void
arm_nn_vector_clamp_f16(float16_t *data, int32_t block_size, float16_t activation_min, float16_t activation_max)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t vmin = vdupq_n_f16(activation_min);
    const float16x8_t vmax = vdupq_n_f16(activation_max);
    for (int32_t i = 0; i < block_size; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(block_size - i));
        float16x8_t v = vld1q_z(data + i, p);
        v = arm_nn_clamp_mve_f16(v, vmin, vmax);
        vst1q_p(data + i, v, p);
    }
    #else
    for (int32_t i = 0; i < block_size; ++i)
    {
        data[i] = arm_nn_clamp_scalar_f16(data[i], activation_min, activation_max);
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F16 */

#endif /* ARM_NN_ACTIVATION_FLT_H */
