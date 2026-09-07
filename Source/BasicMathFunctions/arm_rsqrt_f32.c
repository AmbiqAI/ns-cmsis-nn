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
 * Title:        arm_rsqrt_f32
 * Description:  Elementwise reciprocal square root for float32 tensors
 *
 * $Date:        6 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_sqrt_flt.h"

#if ARM_NN_ENABLE_F32

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

ARM_NN_SQRT_EXACT_FN
arm_cmsis_nn_status arm_rsqrt_f32(const float32_t *input, float32_t *output, int32_t block_size)
{
    if (!input || !output || block_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Scalar value path on every toolchain: Helium has no vector square root.
    // Subnormal inputs follow FPSCR.FZ; see the header for the contract (#295).
    for (int32_t i = 0; i < block_size; ++i)
    {
        uint32_t special_bits;
        if (arm_nn_sqrt_special_f32(arm_nn_f32_to_bits(input[i]), true, &special_bits))
        {
            output[i] = arm_nn_f32_from_bits(special_bits);
            continue;
        }
        output[i] = 1.0f / __builtin_sqrtf(input[i]);
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupElementwise group
 */

#endif /* ARM_NN_ENABLE_F32 */
