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
    (void)input;
    (void)output;
    (void)block_size;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
}

/**
 * @} end of groupElementwise group
 */

#endif /* ARM_NN_ENABLE_F32 */
