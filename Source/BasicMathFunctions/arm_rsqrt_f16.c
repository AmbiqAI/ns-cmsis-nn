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
 * Title:        arm_rsqrt_f16
 * Description:  Elementwise reciprocal square root for float16 tensors
 *
 * $Date:        21 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_sqrt_flt.h"

#if ARM_NN_ENABLE_F16

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

ARM_NN_SQRT_EXACT_FN
arm_cmsis_nn_status arm_rsqrt_f16(const float16_t *input, float16_t *output, int32_t block_size)
{
    if (!input || !output || block_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Scalar value path on every toolchain: Helium has no vector square root.
    // armclang and ATfE vectorize the classification below into an MVE loop and
    // agree bit-for-bit. float32 evaluate, round once (#295).
    for (int32_t i = 0; i < block_size; ++i)
    {
    #if !defined(__ARM_FP16_FORMAT_ALTERNATIVE)
        uint16_t special_bits;
        if (arm_nn_sqrt_special_f16(arm_nn_f16_to_bits(input[i]), true, &special_bits))
        {
            output[i] = arm_nn_f16_from_bits(special_bits);
            continue;
        }
    #endif
        output[i] = (float16_t)(1.0f / __builtin_sqrtf((float32_t)input[i]));
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupElementwise group
 */

#endif /* ARM_NN_ENABLE_F16 */
