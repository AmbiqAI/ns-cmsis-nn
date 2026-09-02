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
 * Title:        arm_hard_swish_f32.c
 * Description:  Hard swish activation function for float32 data
 *
 * $Date:        02 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"

#if ARM_NN_ENABLE_F32

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Acti
 * @{
 */

/*
 * float32 hard swish: out[i] = x * relu6(x + 3) / 6, evaluated as
 * x * clamp(fma(x, 1/6, 0.5), 0, 1).
 *
 * The gate t = fma(x, 1/6, 0.5) is a correctly rounded fused multiply-add on
 * both legs -- __builtin_fmaf on the scalar leg, vfmaq on the MVE leg -- so
 * the two legs round identically everywhere, not just where the clamp
 * saturates. The clamp-to-[0, 1] evaluation (rather than the relu6 form's
 * clamp-to-[0, 6] with a final * (1/6)) is what makes the saturated regions
 * exact: for x >= 3 the gate is exactly 1 (fma(x, 1/6f, 0.5) >= 1 there
 * because 1/6f rounds up), so x * 1 returns x bit-exactly, and for x <= -3
 * the gate is exactly 0, so the output is a zero (of negative sign, as IEEE
 * negative * +0.0). A final multiply by 1/6f instead would perturb the
 * identity region by an ulp.
 *
 * NaN and Inf: see the header. NaN propagation rides the final multiply
 * x * t (a NaN x makes the product NaN whatever the gate resolved to), not a
 * compare-and-select, so it survives the -ffinite-math-only implied by the
 * shipped -Ofast without any integer-domain restore. -Inf gates to 0 and
 * (-Inf) * 0 is NaN by IEEE 754; +Inf gates to 1 and returns +Inf.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_hard_swish_f32(const float32_t *input, float32_t *output, int32_t size)
{
    if (!input || !output || size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vone = vdupq_n_f32(1.0f);
    for (int32_t i = 0; i < size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(size - i));
        const float32x4_t vx = vld1q_z(&input[i], p);
        float32x4_t vt = vfmaq(vhalf, vx, vdupq_n_f32(1.0f / 6.0f));
        vt = vmaxnmq(vt, vzero);
        vt = vminnmq(vt, vone);
        vst1q_p(&output[i], vmulq(vx, vt), p);
    }
    #else
    for (int32_t i = 0; i < size; ++i)
    {
        const float32_t x = input[i];
        // Correctly rounded fma keeps this leg bit-identical to the MVE
        // vfmaq leg at every optimization level; see the note above.
        float32_t t = __builtin_fmaf(x, 1.0f / 6.0f, 0.5f);
        t = CLAMP(t, 1.0f, 0.0f);
        output[i] = x * t;
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Acti group
 */

#endif /* ARM_NN_ENABLE_F32 */
