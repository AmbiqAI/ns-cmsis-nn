/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_exp_flt.h
 * Description:  Scalar body of the float exponential kernels
 *
 * $Date:        26 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_EXP_FLT_H
#define ARM_NN_EXP_FLT_H

#include "arm_nnsupportfunctions.h"

#include <math.h>
#include <string.h>

#if ARM_NN_FLOAT_API_ENABLED

/*
 * exp(x) in float32 for arm_nn_exp_f32 and arm_nn_exp_f16. Declared under
 * ARM_NN_FLOAT_API_ENABLED, not ARM_NN_ENABLE_F32, so the float16 entry can call
 * it in a build without the float32 kernels (AGENTS.md, "Cross-dtype helpers").
 *
 * The value is heliaML's heliaml_reference_exp_f32 (helia-ml 4633a47c,
 * Include/heliaml/internal/heliaml_kernels.h), bit for bit on every non-NaN input:
 *
 *   x NaN             x with the quiet bit set, sign and payload kept
 *   x < -103.97f      +0 (-inf included)
 *   x > 88.7228317f   +inf (0x42B17217 is the largest input with a finite result)
 *   otherwise         2^n * p(r)
 *
 * n = x log2(e) rounded half away from zero: fmaf adds +-0.5 in one rounding and the
 * cast truncates, whatever the rounding mode. r = x - n ln2 by Cody-Waite against a
 * two-part ln2 whose high part has nine trailing zero bits, so n * kLn2Hi is exact;
 * |r| <= 0.35. p is a degree-6 minimax polynomial in Horner form, fmaf at every step,
 * with its constant and linear terms 1, so exp(0) is exactly 1.
 *
 * Every classification is on the integer bit pattern, so -ffinite-math-only cannot
 * fold it and FPSCR.DN cannot change a NaN's payload. -0 takes the -0.5 that the
 * sign bit selects where heliaML's `x < 0.0f` takes +0.5; n is 0 either way.
 *
 * The scaling by 2^n holds its bits by construction, whatever the compiler, the
 * optimisation level or the contraction setting: n is in [-150, 128], so 2^n is not
 * always a float32. heliaML writes it as (p * 2^(n/2)) * 2^(n - n/2), whose first
 * product is exact; AppleClang at -O3 -ffast-math reassociates that to
 * p * (2^(n/2) * 2^(n - n/2)), which overflows at n = 128 and underflows at n = -150
 * (90 581 inputs differ). Here the first, exact step is an integer add of n/2 to p's
 * exponent field -- p * 2^(n/2) is a normal float for every n reached, since p is in
 * [0.70, 1.42] and n/2 in [-75, 64] -- and the only float operation left is one
 * multiply by the normal 2^(n - n/2). Nothing remains for an optimiser to reorder, and
 * the result, subnormal or not, is rounded once. The shift is of the uint32_t cast,
 * so a negative n/2 wraps to the same bits without the undefined left shift of a
 * negative int.
 *
 * FPSCR: the rounding mode is neither set nor read; a subnormal result follows FZ
 * (with FZ set it is +0); a subnormal input reads as zero under FZ, and its result is
 * 1.0f either way. The cumulative exception flags are not part of the contract: an
 * exponential raises inexact, overflow and underflow by nature.
 */
__STATIC_FORCEINLINE uint32_t arm_nn_exp_bits_f32(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__STATIC_FORCEINLINE float32_t arm_nn_exp_from_bits_f32(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/** Bits of -103.97f: a negative input whose bits exceed this is below the zero threshold. */
    #define ARM_NN_EXP_F32_ZERO_BITS UINT32_C(0xC2CFF0A4)
/** Bits of 88.7228317f: a positive input whose bits exceed this overflows. */
    #define ARM_NN_EXP_F32_INF_BITS UINT32_C(0x42B17217)

__STATIC_FORCEINLINE float32_t arm_nn_exp_scalar_f32(float32_t x)
{
    const float32_t log2e = 1.44269502f;
    const float32_t ln2_hi = 0.693145752f;
    const float32_t ln2_lo = 1.42860677e-06f;
    const float32_t c2 = 0.49999994f;
    const float32_t c3 = 0.166664213f;
    const float32_t c4 = 0.0416680574f;
    const float32_t c5 = 0.00837495551f;
    const float32_t c6 = 0.00138427224f;
    const uint32_t bits = arm_nn_exp_bits_f32(x);

    if ((bits & UINT32_C(0x7FFFFFFF)) > UINT32_C(0x7F800000))
    {
        return arm_nn_exp_from_bits_f32(bits | UINT32_C(0x00400000));
    }
    if (bits > ARM_NN_EXP_F32_ZERO_BITS)
    {
        return 0.0f;
    }
    if ((int32_t)bits > (int32_t)ARM_NN_EXP_F32_INF_BITS)
    {
        return arm_nn_exp_from_bits_f32(UINT32_C(0x7F800000));
    }

    const float32_t round_half = arm_nn_exp_from_bits_f32(UINT32_C(0x3F000000) | (bits & UINT32_C(0x80000000)));
    const int32_t n = (int32_t)fmaf(x, log2e, round_half);
    const float32_t nf = (float32_t)n;
    float32_t r = fmaf(nf, -ln2_hi, x);
    r = fmaf(nf, -ln2_lo, r);
    float32_t p = fmaf(c6, r, c5);
    p = fmaf(p, r, c4);
    p = fmaf(p, r, c3);
    p = fmaf(p, r, c2);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);

    const int32_t half = n / 2;
    const float32_t scaled = arm_nn_exp_from_bits_f32(arm_nn_exp_bits_f32(p) + ((uint32_t)half << 23));
    const float32_t second = arm_nn_exp_from_bits_f32((uint32_t)(n - half + 127) << 23);
    return scaled * second;
}

#endif /* ARM_NN_FLOAT_API_ENABLED */

#endif /* ARM_NN_EXP_FLT_H */
