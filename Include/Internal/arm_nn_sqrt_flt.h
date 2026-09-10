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
 * Title:        arm_nn_sqrt_flt.h
 * Description:  Shared special-value contract of the float sqrt / rsqrt kernels
 *
 * $Date:        6 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_SQRT_FLT_H
#define ARM_NN_SQRT_FLT_H

#include "arm_nnsupportfunctions.h"

#include <stdbool.h>
#include <string.h>

/*
 * arm_nn_sqrt_f32 / arm_rsqrt_f32 / arm_nn_sqrt_f16 / arm_rsqrt_f16 decide the
 * non-positive-finite cases on the integer bit pattern, so the outcome is the
 * same on every leg and at every optimization level (-ffinite-math-only and
 * FPSCR.DN cannot change it). One table for all four kernels (#295):
 *
 *   input        sqrt      rsqrt
 *   +0 / -0      +0 / -0   +Inf / -Inf
 *   NaN          the same NaN with the quiet bit set (sign and payload kept)
 *   +Inf         +Inf      +0
 *   negative     default quiet NaN (-Inf included)
 *
 * Positive finite inputs (subnormals included) fall through to the value path.
 */

/*
 * GCC's x86 backend replaces sqrtf with an RSQRTSS Newton approximation when
 * -ffinite-math-only and -funsafe-math-optimizations are both on (-Ofast),
 * which is not correctly rounded. The Arm backend has no such expansion, so
 * this only pins the non-Arm host build, which is a test vehicle (#295).
 */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__arm__) && !defined(__aarch64__)
    #define ARM_NN_SQRT_EXACT_FN __attribute__((optimize("no-unsafe-math-optimizations")))
#else
    #define ARM_NN_SQRT_EXACT_FN
#endif

#if ARM_NN_ENABLE_F32

__STATIC_FORCEINLINE uint32_t arm_nn_f32_to_bits(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__STATIC_FORCEINLINE float32_t arm_nn_f32_from_bits(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Returns true and writes *out_bits when in_bits is not a positive finite value. */
__STATIC_FORCEINLINE bool arm_nn_sqrt_special_f32(uint32_t in_bits, bool reciprocal, uint32_t *out_bits)
{
    const uint32_t magnitude = in_bits & UINT32_C(0x7FFFFFFF);

    if (magnitude == 0)
    {
        *out_bits = reciprocal ? (in_bits | UINT32_C(0x7F800000)) : in_bits;
        return true;
    }
    if (magnitude > UINT32_C(0x7F800000))
    {
        *out_bits = in_bits | UINT32_C(0x00400000);
        return true;
    }
    if (in_bits == UINT32_C(0x7F800000))
    {
        *out_bits = reciprocal ? UINT32_C(0) : in_bits;
        return true;
    }
    if (in_bits & UINT32_C(0x80000000))
    {
        *out_bits = UINT32_C(0x7FC00000);
        return true;
    }
    return false;
}

#endif /* ARM_NN_ENABLE_F32 */

#if ARM_NN_ENABLE_F16

__STATIC_FORCEINLINE uint16_t arm_nn_f16_to_bits(float16_t value)
{
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__STATIC_FORCEINLINE float16_t arm_nn_f16_from_bits(uint16_t bits)
{
    float16_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Returns true and writes *out_bits when in_bits is not a positive finite value. */
__STATIC_FORCEINLINE bool arm_nn_sqrt_special_f16(uint16_t in_bits, bool reciprocal, uint16_t *out_bits)
{
    const uint16_t magnitude = in_bits & UINT16_C(0x7FFF);

    if (magnitude == 0)
    {
        *out_bits = reciprocal ? (uint16_t)(in_bits | UINT16_C(0x7C00)) : in_bits;
        return true;
    }
    if (magnitude > UINT16_C(0x7C00))
    {
        *out_bits = (uint16_t)(in_bits | UINT16_C(0x0200));
        return true;
    }
    if (in_bits == UINT16_C(0x7C00))
    {
        *out_bits = reciprocal ? UINT16_C(0) : in_bits;
        return true;
    }
    if (in_bits & UINT16_C(0x8000))
    {
        *out_bits = UINT16_C(0x7E00);
        return true;
    }
    return false;
}

#endif /* ARM_NN_ENABLE_F16 */

#endif /* ARM_NN_SQRT_FLT_H */
