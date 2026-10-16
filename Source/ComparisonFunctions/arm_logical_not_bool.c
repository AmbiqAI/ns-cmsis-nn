/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_logical_common.h"
#include "arm_nnfunctions.h"

arm_cmsis_nn_status arm_logical_not_bool(const bool *input, bool *output, int32_t count)
{
    if (!input || !output || count < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    uint32_t i = 0;
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (; i < (uint32_t)count; i += 16)
    {
        const mve_pred16_t p = vctp8q((uint32_t)count - i);
        const uint8x16_t value = vldrbq_z_u8((const uint8_t *)input + i, p);
        vstrbq_p_u8((uint8_t *)output + i, veorq_u8(value, vdupq_n_u8(1)), p);
    }
#else
    #if defined(ARM_MATH_DSP)
    for (; i + 4 <= (uint32_t)count; i += 4)
    {
        uint32_t value;
        memcpy(&value, input + i, sizeof(value));
        value ^= 0x01010101U;
        memcpy(output + i, &value, sizeof(value));
    }
    #endif
    for (; i < (uint32_t)count; ++i)
    {
        output[i] = !input[i];
    }
#endif
    return ARM_CMSIS_NN_SUCCESS;
}
