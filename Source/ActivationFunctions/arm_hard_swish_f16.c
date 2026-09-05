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
        #define ARM_NN_HARD_SWISH_F16_MVE 1
    #else
        #define ARM_NN_HARD_SWISH_F16_MVE 0
    #endif

    #if ARM_NN_HARD_SWISH_F16_MVE
// One 8-lane block, evaluated entirely in float16: no half<->single
// conversion, so nothing here rides on the assembler's Q-register VCVTB/VCVTT
// encoding (see AmbiqAI/ns-cmsis-nn#427). The gate is scaled before the
// product, not after: 6 * float16(1/6) rounds to exactly 1.0, so the
// multiplier stays in [0, 1] and |out| <= |in| for every input, whereas
// (x * relu6(x + 3)) / 6 overflows float16 for large x. All lanes are
// computed unpredicated; the caller decides which results reach memory.
static inline float16x8_t arm_hard_swish_block_f16(float16x8_t vx)
{
    const float16x8_t vzero = vdupq_n_f16((float16_t)0.0f);
    const float16x8_t vthree = vdupq_n_f16((float16_t)3.0f);
    const float16x8_t vsix = vdupq_n_f16((float16_t)6.0f);
    const float16x8_t vsixth = vdupq_n_f16((float16_t)(1.0f / 6.0f));

    float16x8_t t = vaddq(vx, vthree);
    t = vmaxnmq(t, vzero);
    t = vminnmq(t, vsix);

    return vmulq(vx, vmulq(t, vsixth));
}
    #endif

/*
 * float16 hard swish: out[i] = x * relu6(x + 3) / 6.
 *
 * The scalar leg widens each element to float32, evaluates the gate and the
 * product there with the same operation sequence as arm_hard_swish_f32 (see
 * the leg-agreement and NaN/Inf notes there) and narrows the product once.
 * The MVE leg evaluates the same expression in float16 throughout. The
 * widened form is the more accurate of the two -- it is single-rounded, where
 * the float16 form rounds the gate and the product separately -- but over all
 * 63488 finite float16 inputs the float16 form stays inside 1e-3 + 1e-3*|ref|
 * of it, the combined reading the tests apply, with 70.2% of that band the
 * worst case; and it drops both the half<->single conversions and the
 * duplicated half-vector arithmetic they force. See AmbiqAI/ns-cmsis-nn#427. The two
 * legs are therefore no longer bit-identical to each other; the saturated
 * regions and the NaN/Inf behaviour still match (header).
 *
 * No scalar _Float16 arithmetic or selects are involved on either leg, so GCC
 * PR target/118460 (HFmode conditional moves) has nothing to bite on.
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

    #if ARM_NN_HARD_SWISH_F16_MVE
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
        t = ARM_NN_CLAMP(t, 1.0f, 0.0f);
        output[i] = (float16_t)(x * t);
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Acti group
 */

#endif /* ARM_NN_ENABLE_F16 */
