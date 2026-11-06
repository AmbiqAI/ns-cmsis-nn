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
 * Title:        arm_layer_norm_f32.c
 * Description:  Float32 layer normalization over the innermost dimension
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "arm_nnsupportfunctions_flt.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#if ARM_NN_ENABLE_F32

/**
 * @ingroup Public
 */

/**
 * @addtogroup Normalization
 * @{
 */

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
/* sum(x) over n values: two accumulators (eight values per iteration), then one predicated tail. */
__STATIC_FORCEINLINE float32_t layer_norm_sum_f32(const float32_t *x, const int32_t n)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int32_t c = 0;
    for (; c <= n - 8; c += 8)
    {
        acc0 = vaddq_f32(acc0, vld1q_f32(x + c));
        acc1 = vaddq_f32(acc1, vld1q_f32(x + c + 4));
    }
    for (; c < n; c += 4)
    {
        acc0 = vaddq_f32(acc0, vld1q_z_f32(x + c, vctp32q((uint32_t)(n - c))));
    }
    return arm_nn_vec_reduce_add_f32(vaddq_f32(acc0, acc1));
}

/* sum((x - mean)^2) over n values; inactive tail lanes are kept out of the sum by the predicate. */
__STATIC_FORCEINLINE float32_t layer_norm_sum_sq_f32(const float32_t *x, const int32_t n, const float32_t mean)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int32_t c = 0;
    for (; c <= n - 8; c += 8)
    {
        const float32x4_t d0 = vsubq_n_f32(vld1q_f32(x + c), mean);
        const float32x4_t d1 = vsubq_n_f32(vld1q_f32(x + c + 4), mean);
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
    }
    for (; c < n; c += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(n - c));
        const float32x4_t d = vsubq_n_f32(vld1q_z_f32(x + c, p), mean);
        acc0 = vfmaq_m_f32(acc0, d, d, p);
    }
    return arm_nn_vec_reduce_add_f32(vaddq_f32(acc0, acc1));
}
    #endif

/* Two passes for the statistics (mean, then the mean of (x - mean)^2), which avoids the cancellation of
 * E[x^2] - E[x]^2. The output is ((x - mean) * inv_std) * gamma + beta, subtracting the mean first: folding it into
 * x * inv_std - mean * inv_std saves an instruction but loses precision when |mean| is large next to the standard
 * deviation. The output loop is chosen up front for the affine parameters given. */
arm_cmsis_nn_status arm_layer_norm_f32(const int32_t rows,
                                       const int32_t row_size,
                                       const float32_t epsilon,
                                       const float32_t *input,
                                       const float32_t *gamma,
                                       const float32_t *beta,
                                       float32_t *output)
{
    /* Checked on the bits: under -ffast-math a NaN can pass a floating-point comparison. */
    uint32_t eps_bits;
    memcpy(&eps_bits, &epsilon, sizeof(eps_bits));
    const bool eps_invalid = (eps_bits & 0x7FFFFFFFU) >= 0x7F800000U /* inf or NaN */ ||
        ((eps_bits & 0x80000000U) && (eps_bits & 0x7FFFFFFFU)) /* below zero */;
    if (!input || !output || rows < 0 || row_size < 1 || eps_invalid)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t n = row_size;
    const float32_t inv_n = 1.0f / (float32_t)n;

    for (int32_t r = 0; r < rows; r++)
    {
        const float32_t *x = input + (size_t)r * (size_t)n;
        float32_t *y = output + (size_t)r * (size_t)n;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        const float32_t mean = layer_norm_sum_f32(x, n) * inv_n;
        const float32_t inv_std = 1.0f / sqrtf(layer_norm_sum_sq_f32(x, n, mean) * inv_n + epsilon);
        const float32x4_t v_inv = vdupq_n_f32(inv_std);
        int32_t c = 0;
        if (gamma && beta)
        {
            for (; c <= n - 8; c += 8)
            {
                const float32x4_t t0 = vmulq_f32(vsubq_n_f32(vld1q_f32(x + c), mean), v_inv);
                const float32x4_t t1 = vmulq_f32(vsubq_n_f32(vld1q_f32(x + c + 4), mean), v_inv);
                vst1q_f32(y + c, vfmaq_f32(vld1q_f32(beta + c), t0, vld1q_f32(gamma + c)));
                vst1q_f32(y + c + 4, vfmaq_f32(vld1q_f32(beta + c + 4), t1, vld1q_f32(gamma + c + 4)));
            }
            for (; c < n; c += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(n - c));
                const float32x4_t t = vmulq_f32(vsubq_n_f32(vld1q_z_f32(x + c, p), mean), v_inv);
                vst1q_p_f32(y + c, vfmaq_f32(vld1q_z_f32(beta + c, p), t, vld1q_z_f32(gamma + c, p)), p);
            }
        }
        else if (gamma)
        {
            for (; c <= n - 8; c += 8)
            {
                const float32x4_t t0 = vmulq_f32(vsubq_n_f32(vld1q_f32(x + c), mean), v_inv);
                const float32x4_t t1 = vmulq_f32(vsubq_n_f32(vld1q_f32(x + c + 4), mean), v_inv);
                vst1q_f32(y + c, vmulq_f32(t0, vld1q_f32(gamma + c)));
                vst1q_f32(y + c + 4, vmulq_f32(t1, vld1q_f32(gamma + c + 4)));
            }
            for (; c < n; c += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(n - c));
                const float32x4_t t = vmulq_f32(vsubq_n_f32(vld1q_z_f32(x + c, p), mean), v_inv);
                vst1q_p_f32(y + c, vmulq_f32(t, vld1q_z_f32(gamma + c, p)), p);
            }
        }
        else
        {
            for (; c <= n - 8; c += 8)
            {
                float32x4_t t0 = vmulq_f32(vsubq_n_f32(vld1q_f32(x + c), mean), v_inv);
                float32x4_t t1 = vmulq_f32(vsubq_n_f32(vld1q_f32(x + c + 4), mean), v_inv);
                if (beta)
                {
                    t0 = vaddq_f32(t0, vld1q_f32(beta + c));
                    t1 = vaddq_f32(t1, vld1q_f32(beta + c + 4));
                }
                vst1q_f32(y + c, t0);
                vst1q_f32(y + c + 4, t1);
            }
            for (; c < n; c += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(n - c));
                float32x4_t t = vmulq_f32(vsubq_n_f32(vld1q_z_f32(x + c, p), mean), v_inv);
                if (beta)
                {
                    t = vaddq_f32(t, vld1q_z_f32(beta + c, p));
                }
                vst1q_p_f32(y + c, t, p);
            }
        }
    #else
        float32_t sum = 0.0f;
        for (int32_t c = 0; c < n; c++)
        {
            sum += x[c];
        }
        const float32_t mean = sum * inv_n;
        float32_t sq = 0.0f;
        for (int32_t c = 0; c < n; c++)
        {
            const float32_t d = x[c] - mean;
            sq += d * d;
        }
        const float32_t inv_std = 1.0f / sqrtf(sq * inv_n + epsilon);
        for (int32_t c = 0; c < n; c++)
        {
            const float32_t t = (x[c] - mean) * inv_std;
            y[c] = t * (gamma ? gamma[c] : 1.0f) + (beta ? beta[c] : 0.0f);
        }
    #endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Normalization group
 */

#endif /* ARM_NN_ENABLE_F32 */
