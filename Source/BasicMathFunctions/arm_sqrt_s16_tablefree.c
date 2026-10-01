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
 * Title:        arm_sqrt_s16_tablefree
 * Description:  Elementwise square root (int16) without a lookup table
 *
 * $Date:        1 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
// Four lanes of the table-free square root, all unpredicated. This is the
// vector form of arm_nn_sqrt_s16_tablefree_element(): the same float32
// operations in the same order, VFMAS where the scalar code calls fmaf, so the
// two agree bit for bit. The lane value must already be clamped to >= 0.
//
//   z  = x * 2^-14 * scale              VCVT with a fixed-point immediate, then one multiply
//   r0 = bits(MAGIC - bits(z) >> 1)     ~ 1/sqrt(z), the float bit-hack seed
//   r1 = r0 * (z * r0 * r0 + K0)        Newton step; K0 < 0 makes r1 negative
//   y  = (z * r1) * (z * r1 * r1 + K1)  Newton step folded into z * r2 = 2^7 sqrt(z)
//
// The sign flips cancel over the two steps, the last step returns z * r2
// directly (which saves the multiply that would form r2), and the chain's 2^7
// output scale is cancelled by the 2^-14 in the conversion. A lane with z = 0
// stays finite: every product with z is 0 and r0 * K0 fits in float32, so
// y = +0 and the conversion returns 0 without any special case.
static inline int32x4_t arm_sqrt_tablefree_quad_s32(const int32x4_t x, const uint32x4_t magic, const float scale)
{
    const float32x4_t z = vmulq_n_f32(vcvtq_n_f32_s32(x, ARM_NN_SQRT_S16_TABLEFREE_SHIFT), scale);
    const uint32x4_t half_bits = vshrq_n_u32(vreinterpretq_u32_f32(z), 1);
    const float32x4_t r0 = vreinterpretq_f32_u32(vsubq_u32(magic, half_bits));

    const float32x4_t u0 = vmulq_f32(z, r0);
    const float32x4_t t0 = vfmasq_n_f32(u0, r0, ARM_NN_SQRT_S16_TABLEFREE_K0);
    const float32x4_t r1 = vmulq_f32(r0, t0);

    const float32x4_t u1 = vmulq_f32(z, r1);
    // VFMAS overwrites its first operand; r1 is dead here and u1 is not.
    const float32x4_t t1 = vfmasq_n_f32(r1, u1, ARM_NN_SQRT_S16_TABLEFREE_K1);
    const float32x4_t y = vmulq_f32(u1, t1);

    // VCVT truncates toward zero and saturates at INT32_MAX; the caller's
    // VQMOVN then saturates to 32767, so any y >= 32767 lands on 32767 exactly
    // as the scalar element function does.
    return vcvtq_s32_f32(y);
}

// One 8-lane block: clamp negatives to 0, widen to two int32 quads, run the
// float chain on each, and narrow back with saturation. Unpredicated, as
// arm_sqrt_block_s16 in arm_sqrt_s16.c; the caller predicates the load and
// store only.
static inline int16x8_t arm_sqrt_tablefree_block_s16(const int16x8_t val, const uint32x4_t magic, const float scale)
{
    const int16x8_t x = vmaxq_s16(val, vdupq_n_s16(0));
    const int32x4_t lo = arm_sqrt_tablefree_quad_s32(vmovlbq_s16(x), magic, scale);
    const int32x4_t hi = arm_sqrt_tablefree_quad_s32(vmovltq_s16(x), magic, scale);

    // The inactive lanes of the first narrow are overwritten by the second,
    // so any register serves as its inactive operand.
    return vqmovntq_s32(vqmovnbq_s32(vreinterpretq_s16_s32(lo), lo), hi);
}
#endif

/*
 * s16 elementwise square root without a lookup table.
 *
 * Each element is sqrt(x * scale) from a float32 reciprocal square root seed
 * and two Newton steps, truncated and saturated; see
 * arm_nn_sqrt_s16_tablefree_element() for the exact chain and the constants.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status
arm_sqrt_s16_tablefree(const int16_t *input, const cmsis_nn_dims *input_dims, int16_t *output, const float scale)
{
    const int32_t block_size = input_dims->n * input_dims->h * input_dims->w * input_dims->c;

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    // Same loop shape as arm_sqrt_s16 and for the same reason: full blocks
    // unpredicated, at most one predicated block outside the loop, so GCC has
    // no vctp loop to turn into dlstp/letp around a body that mixes 16-bit
    // and 32-bit lanes.
    const uint32x4_t magic = vdupq_n_u32(ARM_NN_SQRT_S16_TABLEFREE_MAGIC);
    int32_t i = 0;
    for (; i <= block_size - 8; i += 8)
    {
        const int16x8_t val = vldrhq_s16(&input[i]);
        vstrhq_s16(&output[i], arm_sqrt_tablefree_block_s16(val, magic, scale));
    }
    if (i < block_size)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(block_size - i));
        const int16x8_t val = vldrhq_z_s16(&input[i], p);
        vstrhq_p_s16(&output[i], arm_sqrt_tablefree_block_s16(val, magic, scale), p);
    }
#else
    for (int32_t i = 0; i < block_size; ++i)
    {
        output[i] = arm_nn_sqrt_s16_tablefree_element(input[i], scale);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
