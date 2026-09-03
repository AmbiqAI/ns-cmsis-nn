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
 * Title:        arm_hard_swish_f16.c
 * Description:  Hard swish activation function for float16 data
 *
 * $Date:        02 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"

#if ARM_NN_ENABLE_F16

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Acti
 * @{
 */

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
// One 8-lane block: widen the even (bottom) and odd (top) half-lanes to
// float32, compute the gate and product there, and narrow each product
// exactly once. All lanes are computed unpredicated; the caller decides
// which results reach memory.
static inline float16x8_t arm_hard_swish_block_f16(float16x8_t vx)
{
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vone = vdupq_n_f32(1.0f);
    const float32x4_t vsixth = vdupq_n_f32(1.0f / 6.0f);

    const float32x4_t xb = vcvtbq_f32_f16(vx);
    const float32x4_t xt = vcvttq_f32_f16(vx);

    float32x4_t tb = vfmaq(vhalf, xb, vsixth);
    tb = vmaxnmq(tb, vzero);
    tb = vminnmq(tb, vone);

    float32x4_t tt = vfmaq(vhalf, xt, vsixth);
    tt = vmaxnmq(tt, vzero);
    tt = vminnmq(tt, vone);

    float16x8_t vy = vcvtbq_f16_f32(vdupq_n_f16((float16_t)0.0f), vmulq(xb, tb));
    return vcvttq_f16_f32(vy, vmulq(xt, tt));
}
    #endif

/*
 * float16 hard swish: out[i] = x * relu6(x + 3) / 6, computed in float32 and
 * rounded to float16 once. Each element is widened exactly to float32, the
 * gate and the product are evaluated in float32 with the same operation
 * sequence as arm_hard_swish_f32 (see the leg-agreement and NaN/Inf notes
 * there), and only the final product is narrowed. The single narrowing is
 * the whole point: a native-f16 evaluation rounds the gate and the product
 * separately and lands an ulp off in the curved region for some inputs.
 *
 * Both legs share that shape -- the scalar leg widens with a conversion and
 * uses __builtin_fmaf, the MVE leg widens with vcvtbq/vcvttq_f32_f16 and uses
 * vfmaq -- so they narrow identical float32 products and agree bit-exactly
 * on every numeric input (NaN payloads may differ between legs; header).
 * No scalar _Float16 arithmetic or selects are involved, so GCC PR
 * target/118460 (HFmode conditional moves) has nothing to bite on.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_hard_swish_f16(const float16_t *input, float16_t *output, int32_t size)
{
    if (!input || !output || size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    // Full 8-lane blocks run unpredicated; at most one predicated block
    // handles the tail, OUTSIDE any loop (the arm_memset_f16 shape). This is
    // deliberate and load-bearing, not a style choice: a vctp16q loop around
    // this widening body miscompiles under GCC's implicit-tail-predication
    // conversion (observed with Arm GNU Toolchain 14.2.Rel1 at -Ofast, which
    // turned it into dlstp.16/letp). In an architecturally tail-predicated
    // loop, predication is byte-granular across EVERY vector instruction in
    // the body regardless of its element size, so on a partial tail the
    // .f32-width widen/fma/mul/narrow ops here get the upper bytes of their
    // 32-bit lanes masked and produce garbage for any size not a multiple of
    // 8 (caught on FVP Corstone-300). With no vctp in a loop there is
    // nothing for the dlstp conversion to convert.
    int32_t i = 0;
    for (; i <= size - 8; i += 8)
    {
        const float16x8_t vx = vld1q(&input[i]);
        vst1q(&output[i], arm_hard_swish_block_f16(vx));
    }
    if (i < size)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(size - i));
        const float16x8_t vx = vld1q_z(&input[i], p);
        vst1q_p(&output[i], arm_hard_swish_block_f16(vx), p);
    }
    #else
    for (int32_t i = 0; i < size; ++i)
    {
        const float32_t x = (float32_t)input[i];
        // Correctly rounded fma keeps this leg bit-identical to the MVE
        // vfmaq leg at every optimization level; see the note above.
        float32_t t = __builtin_fmaf(x, 1.0f / 6.0f, 0.5f);
        t = CLAMP(t, 1.0f, 0.0f);
        output[i] = (float16_t)(x * t);
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Acti group
 */

#endif /* ARM_NN_ENABLE_F16 */
