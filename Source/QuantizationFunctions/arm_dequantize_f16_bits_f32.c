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
 * Title:        arm_dequantize_f16_bits_f32.c
 * Description:  Widen float16, given as raw IEEE bits, to float32, bit-exact
 *
 * $Date:        5 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include <string.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Quantization
 * @{
 */

#if ARM_NN_ENABLE_F16 && defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    #define ARM_NN_DEQUANTIZE_MVE
#elif defined(__ARM_FP) && (__ARM_FP & 4) && defined(__ARM_ARCH_PROFILE) && (__ARM_ARCH_PROFILE == 'M') &&             \
    !defined(__ARM_BIG_ENDIAN)
    /* Every M-profile FPU converts half to single (VCVTB), whatever __ARM_FP says about half precision */
    #define ARM_NN_DEQUANTIZE_FPU
#endif

#if !defined(ARM_NN_DEQUANTIZE_MVE) && !defined(ARM_NN_DEQUANTIZE_FPU)
/*
 * Integer widening of one half. Exact for every input class, raises no FP
 * flag, and keeps a NaN's sign, quiet bit and payload (a C cast would quiet
 * a signaling NaN). Half subnormals are normal singles: renormalize.
 */
static inline uint32_t arm_nn_f16_bits_to_f32_bits(const uint32_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0x1Fu)
    {
        return sign | 0x7F800000u | (mant << 13);
    }
    if (exp != 0u)
    {
        return sign | ((exp + 112u) << 23) | (mant << 13);
    }
    if (mant == 0u)
    {
        return sign;
    }
    uint32_t shift = 0u;
    while ((mant & 0x400u) == 0u)
    {
        mant <<= 1;
        shift++;
    }
    return sign | ((113u - shift) << 23) | ((mant & 0x3FFu) << 13);
}
#endif

#if defined(ARM_NN_DEQUANTIZE_FPU)
/*
 * One half: the FPU converts every non-NaN half exactly. A NaN is rebuilt from its bits, because the conversion
 * would quiet a signaling NaN and, with FPSCR.DN set, drop the payload.
 */
static inline float arm_nn_dequantize_one_f32(const uint32_t h)
{
    float out;
    if ((h & 0x7FFFu) <= 0x7C00u)
    {
        float in;
        memcpy(&in, &h, sizeof(in));
        __asm("vcvtb.f32.f16 %0, %1" : "=t"(out) : "t"(in));
    }
    else
    {
        const uint32_t nan = ((h & 0x8000u) << 16) | 0x7F800000u | ((h & 0x3FFu) << 13);
        memcpy(&out, &nan, sizeof(out));
    }
    return out;
}
#endif

#if defined(ARM_NN_DEQUANTIZE_MVE)
/*
 * Vector VCVT always returns the default NaN (Armv8.1-M StandardFPSCRValue,
 * DN=1). Rebuild NaN lanes from the half's bits so the payload survives:
 * `bits` holds the halves `conv` was converted from, one per 32-bit lane.
 * Out of line: keeps its constants out of the hot loop's registers.
 */
static __attribute__((noinline)) float32x4_t arm_nn_dequantize_nan_lanes_f32(const uint32x4_t bits,
                                                                             const float32x4_t conv)
{
    const uint32x4_t sign = vshlq_n_u32(vandq(bits, vdupq_n_u32(0x8000u)), 16);
    const uint32x4_t mant = vshlq_n_u32(vandq(bits, vdupq_n_u32(0x3FFu)), 13);
    const uint32x4_t nan = vorrq(vorrq(sign, mant), vdupq_n_u32(0x7F800000u));
    const mve_pred16_t nan_p = vcmphiq_n_u32(vandq(bits, vdupq_n_u32(0x7FFFu)), 0x7C00u);
    return vpselq(vreinterpretq_f32_u32(nan), conv, nan_p);
}

/* NaN test on the raw halves (|h| > 0x7C00), one predicate bit per 32-bit lane. */
static inline mve_pred16_t arm_nn_dequantize_nan_p(const uint32x4_t bits)
{
    return vcmphiq_n_u32(vandq(bits, vdupq_n_u32(0x7FFFu)), 0x7C00u);
}
#endif

arm_cmsis_nn_status arm_dequantize_f16_bits_f32(const uint16_t *input, float *output, const int32_t block_size)
{
    if (block_size < 0 || ((input == NULL || output == NULL) && block_size != 0))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#if defined(ARM_NN_DEQUANTIZE_MVE)
    /*
     * Widening loads keep element order: half j of a 4-block lands in the low
     * half of lane j, which is what the bottom-half VCVT converts. A plain
     * 8-lane load would leave VCVTB/VCVTT with the even and odd halves.
     */
    int32_t i = 0;
    for (; i + 8 <= block_size; i += 8)
    {
        const uint32x4_t w0 = vldrhq_u32(input + i);
        const uint32x4_t w1 = vldrhq_u32(input + i + 4);
        float32x4_t f0 = arm_nn_vcvtbq_f32_f16(vreinterpretq_f16_u32(w0));
        float32x4_t f1 = arm_nn_vcvtbq_f32_f16(vreinterpretq_f16_u32(w1));
        if ((arm_nn_dequantize_nan_p(w0) | arm_nn_dequantize_nan_p(w1)) != 0u)
        {
            f0 = arm_nn_dequantize_nan_lanes_f32(w0, f0);
            f1 = arm_nn_dequantize_nan_lanes_f32(w1, f1);
        }
        vst1q(output + i, f0);
        vst1q(output + i + 4, f1);
    }
    for (; i < block_size; i += 4)
    {
        /* Inactive lanes load as zero and never trip the NaN test. */
        const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
        const uint32x4_t w = vldrhq_z_u32(input + i, p);
        float32x4_t f = arm_nn_vcvtbq_f32_f16(vreinterpretq_f16_u32(w));
        if (arm_nn_dequantize_nan_p(w) != 0u)
        {
            f = arm_nn_dequantize_nan_lanes_f32(w, f);
        }
        vst1q_p(output + i, f, p);
    }
#elif defined(ARM_NN_DEQUANTIZE_FPU)
    int32_t i = 0;
    for (; i + 2 <= block_size; i += 2)
    {
        /* Two halves per word, read through memcpy: arm_dequantize_f16_f32() passes float16_t storage */
        uint32_t pair;
        memcpy(&pair, &input[i], sizeof(pair));
        /* A half is a NaN when its magnitude exceeds 0x7C00; adding 0x3FF carries such a lane into its top bit */
        if ((((pair & 0x7FFF7FFFu) + 0x03FF03FFu) & 0x80008000u) == 0u)
        {
            float in;
            memcpy(&in, &pair, sizeof(in));
            __asm("vcvtb.f32.f16 %0, %2\n\t"
                  "vcvtt.f32.f16 %1, %2"
                  : "=&t"(output[i]), "=&t"(output[i + 1])
                  : "t"(in));
        }
        else
        {
            output[i] = arm_nn_dequantize_one_f32((uint16_t)pair);
            output[i + 1] = arm_nn_dequantize_one_f32((uint16_t)(pair >> 16));
        }
    }
    if (i < block_size)
    {
        uint16_t h;
        memcpy(&h, &input[i], sizeof(h));
        output[i] = arm_nn_dequantize_one_f32(h);
    }
#else
    for (int32_t i = 0; i < block_size; i++)
    {
        /* Read through memcpy: arm_dequantize_f16_f32() passes float16_t storage */
        uint16_t h;
        uint32_t f;
        memcpy(&h, &input[i], sizeof(h));
        f = arm_nn_f16_bits_to_f32_bits(h);
        memcpy(&output[i], &f, sizeof(f));
    }
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Quantization group
 */
