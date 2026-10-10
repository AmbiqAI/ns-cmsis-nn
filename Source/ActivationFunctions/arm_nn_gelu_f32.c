/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

arm_cmsis_nn_status arm_nn_gelu_f32(const float32_t *input, float32_t *output, int32_t size)
{
    if (size < 0 || (size > 0 && (!input || !output)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int32_t i = 0; i < size; ++i)
    {
        output[i] = arm_nn_gelu_scalar_f32(input[i]);
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F32 */
