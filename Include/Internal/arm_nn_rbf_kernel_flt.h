/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_RBF_KERNEL_FLT_H
#define ARM_NN_RBF_KERNEL_FLT_H

#include "Internal/arm_nn_exp_flt.h"
#include "arm_nnfunctions.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if ARM_NN_ENABLE_F32

    #define ARM_NN_RBF_BLOCK_ROWS (4)

    /* The radial basis function kernel of a float32 reference table against one input row, in one stage:
     *
     *   out[s] = exp(neg_gamma * sum_k (row_s[k] - input[k])^2),  s = 0 to rows - 1,
     *
     * a block of ARM_NN_RBF_BLOCK_ROWS rows at a time, each block's squared distances held in registers, scaled
     * and passed to the library's exponential, with no buffer of one value per row. The table is row-major, row s at
     * references[s * features].
     *
     * Each lane is one sequential sum in feature order, k = 0 to features - 1, from +0.0f, with d = row[k] - input[k]
     * rounded once:
     *
     *   by default        acc = fma(d, d, acc), one fused multiply-add a term rounded once: vfma in the Helium leg,
     *                     fmaf in the scalar leg, so both legs are the same arithmetic whatever the compiler's
     *                     contraction. fmaf is an instruction where the target has a fused multiply-add (the
     *                     Cortex-M4F and Cortex-M55 FPUs) and the build lets the compiler expand it.
     *   ARM_NN_RBF_KERNEL_MUL_ADD defined (to any value) in the translation unit:
     *                     acc = acc + d * d, vmul then vadd in the Helium leg. Its rounding is fixed only where the
     *                     compiler does not contract a multiply and an add (ISO C dialects of Arm GNU, or
     *                     -ffp-contract=off); elsewhere the compiler may fuse some or all of the terms.
     *
     * ARM_NN_RBF_KERNEL_FUSED is 1 for the first form and 0 for the second, so a caller can tell which one a
     * translation unit compiled. arm_nn_rbf_kernel_f32 has the form of the library's build.
     *
     * The value is then acc * neg_gamma, one multiply, and exp of that: arm_nn_exp_scalar_f32 in the scalar leg,
     * arm_nn_exp_mve_f32 in the Helium leg, unchanged (Internal/arm_nn_exp_flt.h). The stage inherits the
     * exponential's edges: a NaN argument returns that NaN with its quiet bit set, an argument below -103.97f (-inf
     * included) +0, above 88.7228317f (+inf included) +inf. A result below 2^-126 is subnormal in the scalar leg (+0
     * under FPSCR.FZ) and +0 in the Helium leg, which flushes it whatever FPSCR holds; every other result is the same
     * bits in both legs. A NaN that the Helium arithmetic produces is its default NaN.
     *
     * The leg is the build's: Helium where ARM_MATH_MVEF is defined without ARM_MATH_AUTOVECTORIZE, scalar otherwise.
     * Both legs have one loop over the features, a feature a trip, so there is no feature remainder; the row remainder
     * of the last block (lanes below 4) is inside each body: a lane past
     * `lanes` computes the value of the last row of the block again and is not written, the Helium store predicated
     * by vctp32q(lanes). The bodies compile under the caller's floating-point options; under fast-math the compiler
     * may reorder the scale into the sum or the exponential, and a lane then need not equal the sequence above.
     *
     * What the caller holds (not checked by the bodies): non-NULL arrays at the alignment of float32_t,
     * 1 <= lanes <= 4, rows >= 1, features >= 1, and `out` not overlapping the table or the input. A caller that passes
     * the extents as literals lets the compiler fold the lane clamps. arm_nn_rbf_kernel_f32 checks them. */

    #if defined(ARM_NN_RBF_KERNEL_MUL_ADD)
        #define ARM_NN_RBF_KERNEL_FUSED (0)
    #else
        #define ARM_NN_RBF_KERNEL_FUSED (1)
    #endif

    #if !defined(ARM_MATH_MVEF) || defined(ARM_MATH_AUTOVECTORIZE)
/* The scalar leg: four chains, lane j reading row j' of the block with j' = j below `lanes` and lanes - 1 from there,
 * so a remainder lane reads the last row of the block. All four sums are written to acc. */
static inline void arm_nn_rbf_sq_distance_lanes_f32(const float32_t *rows,
                                                    const float32_t *input,
                                                    int32_t features,
                                                    int32_t lanes,
                                                    float32_t acc[ARM_NN_RBF_BLOCK_ROWS])
{
    const float32_t *const r0 = rows;
    const float32_t *const r1 = rows + (ptrdiff_t)(lanes > 1 ? 1 : lanes - 1) * features;
    const float32_t *const r2 = rows + (ptrdiff_t)(lanes > 2 ? 2 : lanes - 1) * features;
    const float32_t *const r3 = rows + (ptrdiff_t)(lanes > 3 ? 3 : lanes - 1) * features;
    float32_t acc0 = 0.0f;
    float32_t acc1 = 0.0f;
    float32_t acc2 = 0.0f;
    float32_t acc3 = 0.0f;
    for (int32_t k = 0; k < features; ++k)
    {
        const float32_t x = input[k];
        const float32_t d0 = r0[k] - x;
        const float32_t d1 = r1[k] - x;
        const float32_t d2 = r2[k] - x;
        const float32_t d3 = r3[k] - x;
        #if ARM_NN_RBF_KERNEL_FUSED
        acc0 = fmaf(d0, d0, acc0);
        acc1 = fmaf(d1, d1, acc1);
        acc2 = fmaf(d2, d2, acc2);
        acc3 = fmaf(d3, d3, acc3);
        #else
        acc0 = acc0 + d0 * d0;
        acc1 = acc1 + d1 * d1;
        acc2 = acc2 + d2 * d2;
        acc3 = acc3 + d3 * d3;
        #endif
    }
    acc[0] = acc0;
    acc[1] = acc1;
    acc[2] = acc2;
    acc[3] = acc3;
}
    #else
/* The Helium leg: lane j gathers row j' of the block, so no lane reads past the block's
 * last row and the loop needs no predicate. */
static inline float32x4_t
arm_nn_rbf_sq_distance_mve_f32(const float32_t *rows, const float32_t *input, int32_t features, int32_t lanes)
{
    const uint32x4_t lane = vminq_u32(vidupq_n_u32(0u, 1), vdupq_n_u32((uint32_t)(lanes - 1)));
    const uint32x4_t offsets = vmulq_n_u32(lane, (uint32_t)features);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int32_t k = 0; k < features; ++k)
    {
        const float32x4_t d = vsubq_n_f32(vldrwq_gather_shifted_offset_f32(rows + k, offsets), input[k]);
        #if ARM_NN_RBF_KERNEL_FUSED
        acc = vfmaq_f32(acc, d, d);
        #else
        acc = vaddq_f32(acc, vmulq_f32(d, d));
        #endif
    }
    return acc;
}
    #endif

/* The squared distances of `lanes` rows of a row-major table, row j at rows[j * features], to `input`: the sums the
 * kernel block scales, written to out[0] to out[lanes - 1]. */
static inline void arm_nn_rbf_sq_distance_block_f32(const float32_t *rows,
                                                    const float32_t *input,
                                                    int32_t features,
                                                    int32_t lanes,
                                                    float32_t *out)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    vstrwq_p_f32(out, arm_nn_rbf_sq_distance_mve_f32(rows, input, features, lanes), vctp32q((uint32_t)lanes));
    #else
    float32_t acc[ARM_NN_RBF_BLOCK_ROWS];
    arm_nn_rbf_sq_distance_lanes_f32(rows, input, features, lanes, acc);
    /* One store a lane, not a loop the compiler may turn into a call to memcpy. */
    out[0] = acc[0];
    if (lanes > 1)
    {
        out[1] = acc[1];
    }
    if (lanes > 2)
    {
        out[2] = acc[2];
    }
    if (lanes > 3)
    {
        out[3] = acc[3];
    }
    #endif
}

/* The kernel values of `lanes` rows of a row-major table, row j at rows[j * features], against `input`:
 * out[j] = exp(neg_gamma * squared distance), written to out[0] to out[lanes - 1]. */
static inline void arm_nn_rbf_kernel_block_f32(const float32_t *rows,
                                               const float32_t *input,
                                               float32_t neg_gamma,
                                               int32_t features,
                                               int32_t lanes,
                                               float32_t *out)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t scaled = vmulq_n_f32(arm_nn_rbf_sq_distance_mve_f32(rows, input, features, lanes), neg_gamma);
    vstrwq_p_f32(out, arm_nn_exp_mve_f32(scaled), vctp32q((uint32_t)lanes));
    #else
    float32_t acc[ARM_NN_RBF_BLOCK_ROWS];
    arm_nn_rbf_sq_distance_lanes_f32(rows, input, features, lanes, acc);
    for (int32_t j = 0; j < lanes; ++j)
    {
        out[j] = arm_nn_exp_scalar_f32(acc[j] * neg_gamma);
    }
    #endif
}

/* The stage: the kernel values of every row of a row-major table of `rows` rows, out[0] to out[rows - 1], a block of
 * four rows at a time with the remainder in the last block. Nothing else is written. */
static inline void arm_nn_rbf_kernel_body_f32(const float32_t *input,
                                              const float32_t *references,
                                              int32_t rows,
                                              int32_t features,
                                              float32_t neg_gamma,
                                              float32_t *out)
{
    for (int32_t first = 0; first < rows; first += ARM_NN_RBF_BLOCK_ROWS)
    {
        const int32_t left = rows - first;
        arm_nn_rbf_kernel_block_f32(references + (ptrdiff_t)first * features,
                                    input,
                                    neg_gamma,
                                    features,
                                    left < ARM_NN_RBF_BLOCK_ROWS ? left : ARM_NN_RBF_BLOCK_ROWS,
                                    out + first);
    }
}

#endif /* ARM_NN_ENABLE_F32 */
#endif /* ARM_NN_RBF_KERNEL_FLT_H */
