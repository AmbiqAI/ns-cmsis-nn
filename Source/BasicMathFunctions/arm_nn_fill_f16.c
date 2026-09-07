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
 * Title:        arm_nn_fill_f16.c
 * Description:  Fill a float16_t vector with one value
 *
 * $Date:        6 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 *  @ingroup Public
 */

/**
 * @addtogroup BasicMath
 * @{
 */

arm_cmsis_nn_status arm_nn_fill_f16(const float16_t value, float16_t *output, const int32_t block_size)
{
    if (block_size < 0 || (output == NULL && block_size != 0))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    /* arm_memset_f16 splats the value's bits (vdup / plain stores): a NaN fill lands bit-exact. */
    arm_memset_f16(output, value, (uint32_t)block_size);
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of BasicMath group
 */

#endif /* ARM_NN_ENABLE_F16 */
