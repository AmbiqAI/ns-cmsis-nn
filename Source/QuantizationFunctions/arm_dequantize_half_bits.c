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
 * Title:        arm_dequantize_half_bits.c
 * Description:  Widen float16, given as raw IEEE bits, to float32 as the hardware conversion does
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
 * Integer widening of one half, with the result of the FPU's VCVTB at reset FPSCR: exact for every finite half and
 * Inf, and a NaN keeps its sign and payload and comes back quiet. Raises no FP flag. Half subnormals are normal
 * singles: renormalize.
 */
static inline uint32_t arm_nn_f16_bits_to_f32_bits(const uint32_t h)
{
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0x1Fu)
    {
        return sign | 0x7F800000u | (mant << 13) | (mant != 0u ? 0x00400000u : 0u);
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

arm_cmsis_nn_status arm_dequantize_f16_bits_f32(const uint16_t *input, float *output, const int32_t block_size)
{
    if (block_size < 0 || ((input == NULL || output == NULL) && block_size != 0))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
#if defined(ARM_NN_DEQUANTIZE_MVE)
    /*
     * A widening load puts half j of a 4-block in the low half of lane j, which is what the bottom-half VCVT
     * converts, so the output keeps element order.
     */
    const uint16_t *in = input;
    float *out = output;
    for (int32_t n = block_size >> 2; n > 0; n--)
    {
        vst1q(out, arm_nn_vcvtbq_f32_f16(vreinterpretq_f16_u32(vldrhq_u32(in))));
        in += 4;
        out += 4;
    }
    if ((block_size & 3) != 0)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(block_size & 3));
        vst1q_p(out, arm_nn_vcvtbq_f32_f16(vreinterpretq_f16_u32(vldrhq_z_u32(in, p))), p);
    }
#elif defined(ARM_NN_DEQUANTIZE_FPU)
    int32_t i = 0;
    for (; i + 2 <= block_size; i += 2)
    {
        /* Two halves per word, read through memcpy: arm_dequantize_f16_f32() passes float16_t storage */
        float in;
        memcpy(&in, &input[i], sizeof(in));
        __asm("vcvtb.f32.f16 %0, %2\n\t"
              "vcvtt.f32.f16 %1, %2"
              : "=&t"(output[i]), "=&t"(output[i + 1])
              : "t"(in));
    }
    if (i < block_size)
    {
        uint32_t h = 0u;
        float in;
        memcpy(&h, &input[i], sizeof(uint16_t));
        memcpy(&in, &h, sizeof(in));
        __asm("vcvtb.f32.f16 %0, %1" : "=t"(output[i]) : "t"(in));
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
