/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include <math.h>

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
        const float32_t x = (float32_t)input[i];
        output[i] = (float16_t)(0.5f * x * erfcf(x * -0x1.6a09e6p-1f));
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F16 */
