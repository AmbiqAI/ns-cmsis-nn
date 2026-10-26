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
 * Title:        arm_nn_exp_f32
 * Description:  Elementwise exponential for float32 tensors
 *
 * $Date:        26 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "Internal/arm_nn_exp_flt.h"

#if ARM_NN_ENABLE_F32

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

arm_cmsis_nn_status arm_nn_exp_f32(const float32_t *input, float32_t *output, int32_t block_size)
{
    if (!input || !output || block_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Each element is read before its output is written, so output may alias input.
    #if defined(ARM_NN_EXP_MVE_F32)
    // Tail-predicated: the last block loads, computes and stores only its live lanes.
    const int32_t blocks = block_size / 4 + ((block_size & 3) != 0);
    for (int32_t b = 0; b < blocks; ++b)
    {
        const int32_t i = b * 4;
        const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
        vstrwq_p(&output[i], arm_nn_exp_mve_f32(vld1q_z(&input[i], p)), p);
    }
    #else
    for (int32_t i = 0; i < block_size; ++i)
    {
        output[i] = arm_nn_exp_scalar_f32(input[i]);
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupElementwise group
 */

#endif /* ARM_NN_ENABLE_F32 */
