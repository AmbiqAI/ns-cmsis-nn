/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* The references the rbf kernel bodies are held to, one row at a time: the squared distance as a sequential sum in
 * feature order from +0.0f with every rounding point a volatile, so no build (fast-math included) reorders, contracts
 * or folds it; then the scale as one multiply; then the library's own exponential, arm_nn_exp_scalar_f32 (or, for a
 * Helium lane, arm_nn_exp_mve_f32 on the same argument). */

#ifndef RBF_KERNEL_REFERENCE_H
#define RBF_KERNEL_REFERENCE_H

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdint.h>

#include "../../../../Include/Internal/arm_nn_exp_flt.h"

/* From +0.0f, acc = fma(d, d, acc) with d = row[k] - input[k]: each term rounded once with its sum. */
static inline float32_t rk_reference_sum_fused(const float32_t *row, const float32_t *input, int32_t features)
{
    volatile float32_t acc = 0.0f;
    for (int32_t k = 0; k < features; ++k)
    {
        volatile float32_t d = row[k] - input[k];
        acc = fmaf(d, d, acc);
    }
    return acc;
}

/* From +0.0f, acc = acc + round(d * d): the square rounded before its sum. */
static inline float32_t rk_reference_sum_unfused(const float32_t *row, const float32_t *input, int32_t features)
{
    volatile float32_t acc = 0.0f;
    for (int32_t k = 0; k < features; ++k)
    {
        volatile float32_t d = row[k] - input[k];
        volatile float32_t square = d * d;
        acc = acc + square;
    }
    return acc;
}

/* The argument of the exponential: the sum times neg_gamma, one rounding. */
static inline float32_t rk_reference_argument(float32_t sum, float32_t neg_gamma)
{
    volatile float32_t argument = sum * neg_gamma;
    return argument;
}

/* The exponential of the scalar leg. */
static inline float32_t rk_reference_exp_scalar(float32_t argument) { return arm_nn_exp_scalar_f32(argument); }

/* The exponential of the build's leg: the Helium body on one lane where it is compiled, the scalar body otherwise. */
static inline float32_t rk_reference_exp_leg(float32_t argument)
{
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    return vgetq_lane_f32(arm_nn_exp_mve_f32(vdupq_n_f32(argument)), 0);
#else
    return arm_nn_exp_scalar_f32(argument);
#endif
}

#endif /* RBF_KERNEL_REFERENCE_H */
