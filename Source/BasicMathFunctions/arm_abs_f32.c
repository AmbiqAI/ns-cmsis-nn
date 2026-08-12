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
 * Title:        arm_abs_f32.c
 * Description:  Elementwise absolute value for float32 tensors
 *
 * $Date:        12 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

arm_cmsis_nn_status arm_abs_f32(const float32_t *input, float32_t *output, int32_t block_size)
{
    if (!input || !output || block_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t i = 0; i < block_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
        const float32x4_t va = vld1q_z(&input[i], p);
        vstrwq_p(&output[i], vabsq_f32(va), p);
    }
#else
    for (int32_t i = 0; i < block_size; ++i)
    {
        // __builtin_fabsf clears the sign bit even for -0.0 and -NaN,
        // keeping the scalar path bit-exact with the MVE vabsq path.
        output[i] = __builtin_fabsf(input[i]);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of groupElementwise group
 */
