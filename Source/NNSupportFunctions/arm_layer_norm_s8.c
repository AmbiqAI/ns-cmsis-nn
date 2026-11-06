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
 * Title:        arm_layer_norm_s8.c
 * Description:  Integer layer normalization over the innermost dimension, s8
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

/* Sum and sum of squares of one row; exact in int32 for row_size <= ARM_NN_LAYER_NORM_MAX_ROW_SIZE_S8. */
static void layer_norm_row_stats_s8(const int8_t *x, const int32_t n, int32_t *sum_out, int32_t *sumsq_out)
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

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
/* Four outputs, before the clamp: rshr(qrdmulh(qrdmulh(N' * x - S', m), gamma) + bias, output_shift). */
__STATIC_FORCEINLINE int32x4_t layer_norm_out_mve(const int32x4_t x,
                                                  const int32x4_t v_n,
                                                  const int32_t neg_s,
                                                  const int32_t m,
                                                  const int32x4_t gamma,
                                                  const int32x4_t bias,
                                                  const int32x4_t v_rshift)
{
    const int32x4_t z = vqrdmulhq_n_s32(vmlasq_n_s32(x, v_n, neg_s), m);
    return vrshlq_s32(vaddq_s32(vqrdmulhq_s32(z, gamma), bias), v_rshift);
}
#endif

/*
 * s8 integer layer normalization; see arm_layer_norm_s8() in arm_nnfunctions.h and
 * Tests/UnitTest/layer_norm_reference.py for the definition.
 */
arm_cmsis_nn_status arm_layer_norm_s8(const cmsis_nn_layer_norm_params *params,
                                      const int32_t rows,
                                      const int32_t row_size,
                                      const int8_t *input,
                                      const int32_t *gamma,
                                      const int32_t *bias,
                                      int8_t *output)
{
    if (!params || !input || !gamma || !bias || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (rows < 0 || row_size < 1 || row_size > ARM_NN_LAYER_NORM_MAX_ROW_SIZE_S8 || params->epsilon < 0 ||
        params->epsilon > ARM_NN_LAYER_NORM_MAX_EPSILON || params->output_shift < 1 || params->output_shift > 31 ||
        params->activation.min > params->activation.max)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t n = row_size;
    const int32_t out_shift = params->output_shift;
    const int32_t act_min = params->activation.min;
    const int32_t act_max = params->activation.max;

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const int32x4_t v_rshift = vdupq_n_s32(-out_shift);
    const int32x4_t v_min = vdupq_n_s32(act_min);
    const int32x4_t v_max = vdupq_n_s32(act_max);
#endif

    for (int32_t r = 0; r < rows; r++)
    {
        const int8_t *x = input + (size_t)r * (size_t)n;
        int8_t *y = output + (size_t)r * (size_t)n;

        int32_t sum;
        int32_t sumsq;
        layer_norm_row_stats_s8(x, n, &sum, &sumsq);
        int32_t m;
        int32_t lshift;
        arm_nn_layer_norm_row_multiplier(
            (uint64_t)((int64_t)n * sumsq - (int64_t)sum * sum + params->epsilon), &m, &lshift);
        /* (N << L) * x - (S << L) in wrapping arithmetic: exact, because the true value fits in int32. */
        const int32_t n_l = (int32_t)((uint32_t)n << lshift);
        const int32_t neg_s_l = (int32_t)(0U - ((uint32_t)sum << lshift));

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
        const int32x4_t v_n = vdupq_n_s32(n_l);
        int32_t c = 0;
        for (; c <= n - 16; c += 16)
        {
            const int32x4_t o0 = layer_norm_out_mve(
                vldrbq_s32(x + c), v_n, neg_s_l, m, vld1q_s32(gamma + c), vld1q_s32(bias + c), v_rshift);
            const int32x4_t o1 = layer_norm_out_mve(
                vldrbq_s32(x + c + 4), v_n, neg_s_l, m, vld1q_s32(gamma + c + 4), vld1q_s32(bias + c + 4), v_rshift);
            const int32x4_t o2 = layer_norm_out_mve(
                vldrbq_s32(x + c + 8), v_n, neg_s_l, m, vld1q_s32(gamma + c + 8), vld1q_s32(bias + c + 8), v_rshift);
            const int32x4_t o3 = layer_norm_out_mve(
                vldrbq_s32(x + c + 12), v_n, neg_s_l, m, vld1q_s32(gamma + c + 12), vld1q_s32(bias + c + 12), v_rshift);
            vstrbq_s32(y + c, vminq_s32(vmaxq_s32(o0, v_min), v_max));
            vstrbq_s32(y + c + 4, vminq_s32(vmaxq_s32(o1, v_min), v_max));
            vstrbq_s32(y + c + 8, vminq_s32(vmaxq_s32(o2, v_min), v_max));
            vstrbq_s32(y + c + 12, vminq_s32(vmaxq_s32(o3, v_min), v_max));
        }
        for (; c < n; c += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(n - c));
            const int32x4_t o = layer_norm_out_mve(vldrbq_z_s32(x + c, p),
                                                   v_n,
                                                   neg_s_l,
                                                   m,
                                                   vldrwq_z_s32(gamma + c, p),
                                                   vldrwq_z_s32(bias + c, p),
                                                   v_rshift);
            vstrbq_p_s32(y + c, vminq_s32(vmaxq_s32(o, v_min), v_max), p);
        }
#else
        for (int32_t c = 0; c < n; c++)
        {
            const int32_t t = (int32_t)((uint32_t)n_l * (uint32_t)(int32_t)x[c] + (uint32_t)neg_s_l);
            const int32_t z = arm_nn_layer_norm_qrdmulh(t, m);
            const int32_t acc = arm_nn_layer_norm_qrdmulh(z, gamma[c]) + bias[c];
            y[c] = (int8_t)ARM_NN_CLAMP(arm_nn_layer_norm_rshr(acc, out_shift), act_max, act_min);
        }
#endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Normalization group
 */
