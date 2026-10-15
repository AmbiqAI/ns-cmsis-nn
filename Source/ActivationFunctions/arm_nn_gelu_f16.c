/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

arm_cmsis_nn_status arm_nn_gelu_f16(const float16_t *input, float16_t *output, int32_t size)
{
    if (size < 0 || (size > 0 && (!input || !output)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int32_t i = 0; i < size; ++i)
    {
        /* The float32 exact expression, rounded to float16 once. */
        output[i] = (float16_t)arm_nn_gelu_scalar_f32((float32_t)input[i]);
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F16 */
