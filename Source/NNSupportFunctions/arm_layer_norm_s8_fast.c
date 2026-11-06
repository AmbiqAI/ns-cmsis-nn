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
 * Title:        arm_layer_norm_s8_fast.c
 * Description:  Integer layer normalization over the innermost dimension, s8, with a 16-bit affine step
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_layer_norm.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Normalization
 * @{
 */

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
/* Eight outputs, clamped and ready for VSTRB.16. The input is widened into even and odd 32-bit lanes, normalized
 * there, and narrowed back into the matching 16-bit lanes, so the 16-bit vector holds the values in order.
 *
 * VQMOVNB writes only the even lanes of its destination and VQMOVNT only the odd ones, so each narrowing starts
 * from a register that is dead by then (the input bytes, then the bias), which lets the compiler narrow in place
 * instead of copying. With the full int8 range (full_range, a constant at each call) the clamp is a saturating
 * narrow to int8, whose results sit in the even bytes that VSTRB.16 stores; otherwise it is VMAX/VMIN. */
__STATIC_FORCEINLINE int16x8_t layer_norm_fast_out_mve(const int16x8_t x,
                                                       const int32x4_t v_n,
                                                       const int32_t neg_s,
                                                       const int32_t m,
                                                       const int16x8_t gamma,
                                                       const int16x8_t bias,
                                                       const int16x8_t v_rshift,
                                                       const int32_t full_range,
                                                       const int16x8_t v_min,
                                                       const int16x8_t v_max)
{
    const int32x4_t z_even = vqrdmulhq_n_s32(vmlasq_n_s32(vmovlbq_s16(x), v_n, neg_s), m);
    const int32x4_t z_odd = vqrdmulhq_n_s32(vmlasq_n_s32(vmovltq_s16(x), v_n, neg_s), m);
    const int16x8_t z = vqmovntq_s32(vqmovnbq_s32(x, z_even), z_odd);
    const int16x8_t o = vrshlq_s16(vqaddq_s16(vqrdmulhq_s16(z, gamma), bias), v_rshift);
    if (full_range)
    {
        return vreinterpretq_s16_s8(vqmovnbq_s16(vreinterpretq_s8_s16(bias), o));
    }
    return vminq_s16(vmaxq_s16(o, v_min), v_max);
}

/* One row's output pass; full_range is a compile-time constant at each call, so each gets its own loop. */
__STATIC_FORCEINLINE void layer_norm_fast_row_mve(const int8_t *x,
                                                  int8_t *y,
                                                  const int32_t n,
                                                  const int32x4_t v_n,
                                                  const int32_t neg_s,
                                                  const int32_t m,
                                                  const int16_t *gamma,
                                                  const int16_t *bias,
                                                  const int16x8_t v_rshift,
                                                  const int32_t full_range,
                                                  const int16x8_t v_min,
                                                  const int16x8_t v_max)
{
    int32_t c = 0;
    for (; c <= n - 16; c += 16)
    {
        vstrbq_s16(y + c,
                   layer_norm_fast_out_mve(vldrbq_s16(x + c),
                                           v_n,
                                           neg_s,
                                           m,
                                           vld1q_s16(gamma + c),
                                           vld1q_s16(bias + c),
                                           v_rshift,
                                           full_range,
                                           v_min,
                                           v_max));
        vstrbq_s16(y + c + 8,
                   layer_norm_fast_out_mve(vldrbq_s16(x + c + 8),
                                           v_n,
                                           neg_s,
                                           m,
                                           vld1q_s16(gamma + c + 8),
                                           vld1q_s16(bias + c + 8),
                                           v_rshift,
                                           full_range,
                                           v_min,
                                           v_max));
    }
    for (; c < n; c += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(n - c));
        vstrbq_p_s16(y + c,
                     layer_norm_fast_out_mve(vldrbq_z_s16(x + c, p),
                                             v_n,
                                             neg_s,
                                             m,
                                             vldrhq_z_s16(gamma + c, p),
                                             vldrhq_z_s16(bias + c, p),
                                             v_rshift,
                                             full_range,
                                             v_min,
                                             v_max),
                     p);
    }
}
#endif

/*
 * s8 integer layer normalization with a 16-bit affine step; see arm_layer_norm_s8_fast() in arm_nnfunctions.h and
 * Tests/UnitTest/layer_norm_reference.py (layer_norm_int_fast) for the definition.
 */
arm_cmsis_nn_status arm_layer_norm_s8_fast(const cmsis_nn_layer_norm_fast_params *params,
                                           const int32_t rows,
                                           const int32_t row_size,
                                           const int8_t *input,
                                           const int16_t *gamma,
                                           const int16_t *bias,
                                           int8_t *output)
{
    if (!params || !input || !gamma || !bias || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (rows < 0 || row_size < 1 || row_size > ARM_NN_LAYER_NORM_MAX_ROW_SIZE_S8 || params->epsilon < 0 ||
        params->epsilon > ARM_NN_LAYER_NORM_MAX_EPSILON ||
        params->norm_frac_bits < ARM_NN_LAYER_NORM_FAST_MIN_FRAC_BITS ||
        params->norm_frac_bits > ARM_NN_LAYER_NORM_FAST_MAX_FRAC_BITS || params->output_shift < 1 ||
        params->output_shift > 7 || params->activation.min < INT8_MIN || params->activation.max > INT8_MAX ||
        params->activation.min > params->activation.max)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t n = row_size;
    const int32_t frac_bits = params->norm_frac_bits;
    const int32_t out_shift = params->output_shift;
    const int32_t act_min = params->activation.min;
    const int32_t act_max = params->activation.max;

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const int16x8_t v_rshift = vdupq_n_s16((int16_t)-out_shift);
    const int16x8_t v_min = vdupq_n_s16((int16_t)act_min);
    const int16x8_t v_max = vdupq_n_s16((int16_t)act_max);
#endif

    for (int32_t r = 0; r < rows; r++)
    {
        const int8_t *x = input + (size_t)r * (size_t)n;
        int8_t *y = output + (size_t)r * (size_t)n;

        int32_t sum;
        int32_t sumsq;
        arm_nn_layer_norm_row_stats_s8(x, n, &sum, &sumsq);
        int32_t m;
        int32_t lshift;
        arm_nn_layer_norm_row_multiplier_frac(
            (uint64_t)((int64_t)n * sumsq - (int64_t)sum * sum + params->epsilon), frac_bits, &m, &lshift);
        /* (N << L) * x - (S << L) in wrapping arithmetic: exact, because the true value fits in int32. */
        const int32_t n_l = (int32_t)((uint32_t)n << lshift);
        const int32_t neg_s_l = (int32_t)(0U - ((uint32_t)sum << lshift));

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
        const int32x4_t v_n = vdupq_n_s32(n_l);
        if (act_min == INT8_MIN && act_max == INT8_MAX)
        {
            layer_norm_fast_row_mve(x, y, n, v_n, neg_s_l, m, gamma, bias, v_rshift, 1, v_min, v_max);
        }
        else
        {
            layer_norm_fast_row_mve(x, y, n, v_n, neg_s_l, m, gamma, bias, v_rshift, 0, v_min, v_max);
        }
#else
        for (int32_t c = 0; c < n; c++)
        {
            const int32_t t = (int32_t)((uint32_t)n_l * (uint32_t)(int32_t)x[c] + (uint32_t)neg_s_l);
            const int16_t z = (int16_t)ARM_NN_CLAMP(arm_nn_layer_norm_qrdmulh(t, m), INT16_MAX, INT16_MIN);
            const int32_t acc =
                ARM_NN_CLAMP((int32_t)arm_nn_layer_norm_qrdmulh_s16(z, gamma[c]) + bias[c], INT16_MAX, INT16_MIN);
            y[c] = (int8_t)ARM_NN_CLAMP(arm_nn_layer_norm_rshr(acc, out_shift), act_max, act_min);
        }
#endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Normalization group
 */
