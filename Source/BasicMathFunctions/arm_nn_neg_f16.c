/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnfunctions.h"
#include <string.h>

#if ARM_NN_ENABLE_F16

/**
 * @ingroup groupElementwise
 */
arm_cmsis_nn_status arm_nn_neg_f16(const float16_t *input, float16_t *output, int32_t count)
{
    if (count < 0 || count > INT32_MAX / (int32_t)sizeof(float16_t) || (count > 0 && (!input || !output)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const uint16x8_t sign = vdupq_n_u16(0x8000u);
    uint32_t i = 0;
    for (int32_t remaining = count; remaining > 0; remaining -= 8, i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)remaining);
        const uint16x8_t value = vreinterpretq_u16_f16(vld1q_z(input + i, p));
        vstrhq_p(output + i, vreinterpretq_f16_u16(veorq(value, sign)), p);
    }
    #else
    for (int32_t i = 0; i < count; ++i)
    {
        // Bitwise negation preserves signaling NaNs and subnormals under every FP mode.
        uint16_t value;
        memcpy(&value, input + i, sizeof(value));
        value ^= 0x8000u;
        memcpy(output + i, &value, sizeof(value));
    }
    #endif
    return ARM_CMSIS_NN_SUCCESS;
}

#endif
