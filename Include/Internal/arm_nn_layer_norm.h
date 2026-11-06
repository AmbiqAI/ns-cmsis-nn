/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      NS-CMSIS NN Library
 * Title:        arm_nn_layer_norm.h
 * Description:  Shared integer machinery of arm_layer_norm_s8 / arm_layer_norm_s16
 *
 * The integer algorithm is defined by Tests/UnitTest/layer_norm_reference.py; the scalar, DSP and MVE paths of
 * both kernels match it bit for bit (in either requantization rounding mode: it uses its own operations). Per row
 * of N values x:
 *
 *   S = sum(x), Q = sum(x * x), D = N * Q - S * S + epsilon       (exact; D = N^2 * (variance + eps))
 *   (m, L) = arm_nn_layer_norm_row_multiplier(D)                    (2^20 / sqrt(D) ~= m * 2^L / 2^31)
 *   z = qrdmulh((N << L) * x - (S << L), m)                         (normalized value, 20 fraction bits)
 *   y = clamp(rshr(qrdmulh(z, gamma[c]) + bias[c], output_shift))
 *
 * qrdmulh is VQRDMULH (round-half-up (a * b) / 2^31) and rshr is VRSHL by a negative shift (round half up).
 *
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_LAYER_NORM_H
#define ARM_NN_LAYER_NORM_H

#include "arm_nn_tables.h"
#include "arm_nnsupportfunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Fraction bits of the normalized value z. |z| <= sqrt(N - 1) * 2^20, which with the supported row sizes keeps
 *  z, and the left-shifted operand of its requantization, in int32. */
#define ARM_NN_LAYER_NORM_FRAC_BITS (20)
/** Largest row sizes: s8 keeps sum(x * x) in int32, s16 keeps N * x - S in int32. */
#define ARM_NN_LAYER_NORM_MAX_ROW_SIZE_S8 (65536)
#define ARM_NN_LAYER_NORM_MAX_ROW_SIZE_S16 (32767)
/** Largest epsilon term: keeps D below 2^62 for every supported row. */
#define ARM_NN_LAYER_NORM_MAX_EPSILON ((int64_t)((1ULL << 62) - (1ULL << 60) - 1ULL))

/** Range of norm_frac_bits for arm_layer_norm_s8_fast. The export picks 15 - bit_length(isqrt(N - 1)), the most
 *  fraction bits that keep |z| <= sqrt(N - 1) * 2^bits in int16: 15 for N = 1, 6 for N = 65536. */
#define ARM_NN_LAYER_NORM_FAST_MIN_FRAC_BITS (6)
#define ARM_NN_LAYER_NORM_FAST_MAX_FRAC_BITS (15)

/**
 * @brief           Row multiplier for integer layer norm, for z with @p frac_bits fraction bits.
 * @param[in]       d          N^2 * (variance + eps) of the row, in [0, 2^62)
 * @param[in]       frac_bits  Fraction bits of z, in [1, 20]
 * @param[out]      mult       Q31 multiplier, in [0, INT32_MAX]
 * @param[out]      lshift     Left shift, in [0, frac_bits]
 *
 * 2^frac_bits / sqrt(d) ~= *mult * 2^(*lshift) / 2^31. d is normalized by an even shift to dn = d * 4^a in
 * [2^60, 2^62); x = dn >> 30 is X = dn / 2^60 as Q30. Y ~= 1/sqrt(X) as Q31 starts from layer_norm_inv_sqrt_seed and
 * takes three truncating Newton steps Y <- Y * (3 - X * Y^2) / 2, so 2^frac_bits / sqrt(d) =
 * Y * 2^(frac_bits + a - 30) / 2^31. A negative exponent is folded into Y with a rounding shift, so the per-element
 * path needs no shift after its multiply. Integer only, about 40 cycles. d == 0 (a constant row with epsilon 0)
 * gives a zero multiplier, so the row normalizes to 0.
 */
__STATIC_FORCEINLINE void
arm_nn_layer_norm_row_multiplier_frac(const uint64_t d, const int32_t frac_bits, int32_t *mult, int32_t *lshift)
{
    if (d == 0)
    {
        *mult = 0;
        *lshift = 0;
        return;
    }
    const uint32_t hi = (uint32_t)(d >> 32);
    const int32_t top_bit = hi ? 63 - (int32_t)CLZ(hi) : 31 - (int32_t)CLZ((uint32_t)d);
    const int32_t a = (61 - top_bit) >> 1;
    const uint32_t x = (uint32_t)((d << (2 * a)) >> 30);
    uint32_t y = layer_norm_inv_sqrt_seed[(x >> 27) - 8];
    for (int32_t i = 0; i < 3; i++)
    {
        /* y * (3 * 2^31 - xy2) >> 32 == y + ((y * (2^31 - xy2)) >> 32) exactly, and every operand fits in 32 bits:
         * each step is three 32 x 32 -> 64-bit multiplies. y stays below 2^31 and xy2 within 13% of 2^31. */
        const uint32_t y2 = (uint32_t)(((uint64_t)y * y) >> 31);
        const uint32_t xy2 = (uint32_t)(((uint64_t)x * y2) >> 30);
        const int32_t delta = (int32_t)((1U << 31) - xy2);
        y = (uint32_t)((int64_t)y + (((int64_t)y * delta) >> 32));
    }
    if (y > (uint32_t)INT32_MAX)
    {
        y = INT32_MAX;
    }
    const int32_t shift = frac_bits + a - 30;
    if (shift < 0)
    {
        y = (uint32_t)(((uint64_t)y + (1ULL << (-shift - 1))) >> -shift);
        *lshift = 0;
    }
    else
    {
        *lshift = shift;
    }
    *mult = (int32_t)y;
}

/**
 * @brief           Row multiplier for arm_layer_norm_s8 / arm_layer_norm_s16: z with ARM_NN_LAYER_NORM_FRAC_BITS (20)
 *                  fraction bits.
 * @param[in]       d       N^2 * (variance + eps) of the row, in [0, 2^62)
 * @param[out]      mult    Q31 multiplier, in [0, INT32_MAX]
 * @param[out]      lshift  Left shift, in [0, 20]
 *
 * See arm_nn_layer_norm_row_multiplier_frac(). With 20 fraction bits the relative error is below 2^-20 and the
 * multiplier keeps at least 21 significant bits.
 */
__STATIC_FORCEINLINE void arm_nn_layer_norm_row_multiplier(const uint64_t d, int32_t *mult, int32_t *lshift)
{
    arm_nn_layer_norm_row_multiplier_frac(d, ARM_NN_LAYER_NORM_FRAC_BITS, mult, lshift);
}

/** Sum and sum of squares of one int8 row; exact in int32 for n <= ARM_NN_LAYER_NORM_MAX_ROW_SIZE_S8. */
__STATIC_FORCEINLINE void
arm_nn_layer_norm_row_stats_s8(const int8_t *x, const int32_t n, int32_t *sum_out, int32_t *sumsq_out)
{
    int32_t sum = 0;
    int32_t sumsq = 0;
    int32_t i = 0;
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (; i <= n - 16; i += 16)
    {
        const int8x16_t v = vld1q_s8(x + i);
        sum = vaddvaq_s8(sum, v);
        sumsq = vmladavaq_s8(sumsq, v, v);
    }
    if (i < n)
    {
        const int8x16_t v = vldrbq_z_s8(x + i, vctp8q((uint32_t)(n - i))); /* inactive lanes load 0 */
        sum = vaddvaq_s8(sum, v);
        sumsq = vmladavaq_s8(sumsq, v, v);
    }
#else
    #if defined(ARM_MATH_DSP)
    const int8_t *ptr = x;
    for (; i <= n - 4; i += 4)
    {
        const int32_t word = arm_nn_read_s8x4_ia(&ptr);
        const int32_t even = (int32_t)SXTB16((uint32_t)word);
        const int32_t odd = (int32_t)SXTB16_RORn((uint32_t)word, 8);
        sumsq = (int32_t)SMLAD((uint32_t)even, (uint32_t)even, (uint32_t)sumsq);
        sumsq = (int32_t)SMLAD((uint32_t)odd, (uint32_t)odd, (uint32_t)sumsq);
        sum = (int32_t)SMLAD((uint32_t)even, 0x00010001U, (uint32_t)sum);
        sum = (int32_t)SMLAD((uint32_t)odd, 0x00010001U, (uint32_t)sum);
    }
    #endif
    for (; i < n; i++)
    {
        const int32_t v = x[i];
        sum += v;
        sumsq += v * v;
    }
#endif
    *sum_out = sum;
    *sumsq_out = sumsq;
}

/** VQRDMULH.S32 on scalars: round-half-up (a * b) / 2^31 (the operands are never both INT32_MIN). */
__STATIC_FORCEINLINE int32_t arm_nn_layer_norm_qrdmulh(const int32_t a, const int32_t b)
{
    return (int32_t)(((int64_t)a * b + (1LL << 30)) >> 31);
}

/** VQRDMULH.S16 on scalars: round-half-up (a * b) / 2^15, saturated to int16. */
__STATIC_FORCEINLINE int16_t arm_nn_layer_norm_qrdmulh_s16(const int16_t a, const int16_t b)
{
    const int32_t v = ((int32_t)a * b + (1 << 14)) >> 15;
    return (int16_t)(v > INT16_MAX ? INT16_MAX : v);
}

/** VRSHL.S32 by -shift on a scalar: round-half-up v / 2^shift, shift in [1, 31]. */
__STATIC_FORCEINLINE int32_t arm_nn_layer_norm_rshr(const int32_t v, const int32_t shift)
{
    return (int32_t)(((int64_t)v + (1LL << (shift - 1))) >> shift);
}

#ifdef __cplusplus
}
#endif

#endif /* ARM_NN_LAYER_NORM_H */
