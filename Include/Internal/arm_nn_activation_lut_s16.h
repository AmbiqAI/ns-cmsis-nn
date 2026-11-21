/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_ACTIVATION_LUT_S16_H
#define ARM_NN_ACTIVATION_LUT_S16_H

#include "arm_nnsupportfunctions.h"

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
/* Each active lane reads table[index] and table[index + 1]. */
__STATIC_FORCEINLINE mve_pred16_t arm_nn_activation_lut_predicate_s16(uint32x4_t index, mve_pred16_t active)
{
    return vcmphiq_m_u32(vdupq_n_u32(255), index, active);
}
#endif

/* The autovectorize configuration leaves explicit MVE disabled. On cores with
 * DSP and unaligned little-endian loads, SMLAD interpolates two signed knots. */
#if defined(ARM_MATH_MVEI) && defined(ARM_MATH_AUTOVECTORIZE) && defined(ARM_MATH_DSP) &&                              \
    defined(__ARM_FEATURE_UNALIGNED) && !defined(__ARM_BIG_ENDIAN)
    #define ARM_NN_ACTIVATION_S16_DSP_LUT
extern const int16_t arm_nn_tanh_knots_s16[513];
extern const int16_t arm_nn_logistic_knots_s16[513];

__STATIC_FORCEINLINE int16_t arm_nn_activation_interpolate_s16(int32_t scaled, bool is_tanh)
{
    const int32_t shift = is_tanh ? 8 : 9;
    const int32_t step = 1 << shift;
    const int32_t index = __ssat(scaled >> shift, 9);
    const uint32_t fraction = (uint32_t)scaled & (step - 1);
    const int16_t *table = is_tanh ? arm_nn_tanh_knots_s16 : arm_nn_logistic_knots_s16;
    uint32_t knots;
    #if defined(__GNUC__) || defined(__clang__)
    __builtin_memcpy(&knots, table + 256 + index, sizeof(knots));
    #else
    memcpy(&knots, table + 256 + index, sizeof(knots));
    #endif
    const uint32_t weights = (fraction << 16) | (step - fraction);
    /* Negative ties round away from zero. LOGISTIC is centered at Q15 0.5. */
    const int32_t bias = (is_tanh ? 128 : (16384 << 10)) + (scaled >> 31);
    const int32_t sum = (int32_t)SMLAD(knots, weights, (uint32_t)bias);
    return (int16_t)(sum >> (is_tanh ? 8 : 10));
}

/* Call with a constant zero shift to remove the per-element rounding and shift. */
__STATIC_FORCEINLINE void arm_nn_activation_dsp_loop_s16(const int16_t *input,
                                                         const int16_t *end,
                                                         int16_t *output,
                                                         int32_t multiplier,
                                                         int32_t shift,
                                                         int32_t round,
                                                         bool is_tanh)
{
    while (input < end)
    {
        const int32_t scaled = arm_nn_mac_wrap_s32_bits(round, *input++, multiplier) >> shift;
        *output++ = arm_nn_activation_interpolate_s16(scaled, is_tanh);
    }
}

__STATIC_FORCEINLINE void arm_nn_activation_dsp_s16(const int16_t *input,
                                                    int16_t *output,
                                                    int32_t size,
                                                    int32_t multiplier,
                                                    int32_t shift,
                                                    int32_t round,
                                                    bool is_tanh)
{
    if (size > 0)
    {
        const int16_t *end = input + size;
        if (shift == 0)
        {
            arm_nn_activation_dsp_loop_s16(input, end, output, multiplier, 0, 0, is_tanh);
        }
        else
        {
            arm_nn_activation_dsp_loop_s16(input, end, output, multiplier, shift, round, is_tanh);
        }
    }
}
#endif

#endif /* ARM_NN_ACTIVATION_LUT_S16_H */
