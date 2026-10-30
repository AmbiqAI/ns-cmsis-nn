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
 * Title:        arm_nn_exp_f16
 * Description:  Elementwise exponential for float16 tensors
 *
 * $Date:        26 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_exp_flt.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

arm_cmsis_nn_status arm_nn_exp_f16(const float16_t *input, float16_t *output, int32_t block_size)
{
    if (!input || !output || block_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Each element is read before its output is written, so output may alias input. A NaN
    // is decided on its float16 bits, before any widening, so FPSCR.DN cannot drop its payload.
    #if defined(ARM_NN_EXP_MVE_F32) && defined(ARM_MATH_MVE_FLOAT16)
    // Eight halves per block, widened to two float32 vectors (even and odd lanes), run
    // through the float32 Helium body and narrowed once. Tail-predicated, no epilogue.
    const int32_t blocks = block_size / 8 + ((block_size & 7) != 0);
    for (int32_t b = 0; b < blocks; ++b)
    {
        const int32_t i = b * 8;
        const mve_pred16_t p = vctp16q((uint32_t)(block_size - i));
        const float16x8_t x = vld1q_z(&input[i], p);
        const uint16x8_t bits = vreinterpretq_u16_f16(x);
        const float32x4_t even = arm_nn_exp_mve_f32(arm_nn_vcvtbq_f32_f16(x));
        const float32x4_t odd = arm_nn_exp_mve_f32(arm_nn_vcvttq_f32_f16(x));
        float16x8_t y = arm_nn_vcvtbq_f16_f32(x, even);
        y = arm_nn_vcvttq_f16_f32(y, odd);
        const mve_pred16_t nan = vcmphiq_n_u16(vandq_u16(bits, vdupq_n_u16(UINT16_C(0x7FFF))), UINT16_C(0x7C00));
        y = vpselq_f16(vreinterpretq_f16_u16(vorrq_u16(bits, vdupq_n_u16(UINT16_C(0x0200)))), y, nan);
        vstrhq_p(&output[i], y, p);
    }
    #else
    for (int32_t i = 0; i < block_size; ++i)
    {
        uint16_t bits;
        memcpy(&bits, &input[i], sizeof(bits));
        if ((bits & UINT16_C(0x7FFF)) > UINT16_C(0x7C00))
        {
            bits = (uint16_t)(bits | UINT16_C(0x0200));
            memcpy(&output[i], &bits, sizeof(bits));
            continue;
        }
        output[i] = (float16_t)arm_nn_exp_scalar_f32((float32_t)input[i]);
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupElementwise group
 */

#endif /* ARM_NN_ENABLE_F16 */
