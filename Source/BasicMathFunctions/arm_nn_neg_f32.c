/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnfunctions.h"
#include <string.h>

#if ARM_NN_ENABLE_F32

/**
 * @ingroup groupElementwise
 */
arm_cmsis_nn_status arm_nn_neg_f32(const float32_t *input, float32_t *output, int32_t count)
{
    if (count < 0 || count > INT32_MAX / (int32_t)sizeof(float32_t) || (count > 0 && (!input || !output)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const uint32x4_t sign = vdupq_n_u32(0x80000000u);
    uint32_t i = 0;
    for (int32_t remaining = count; remaining > 0; remaining -= 4, i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)remaining);
        const uint32x4_t value = vreinterpretq_u32_f32(vld1q_z(input + i, p));
        vstrwq_p(output + i, vreinterpretq_f32_u32(veorq(value, sign)), p);
    }
    #else
    for (int32_t i = 0; i < count; ++i)
    {
        // Bitwise negation preserves signaling NaNs and subnormals under every FP mode.
        uint32_t value;
        memcpy(&value, input + i, sizeof(value));
        value ^= 0x80000000u;
        memcpy(output + i, &value, sizeof(value));
    }
    #endif
    return ARM_CMSIS_NN_SUCCESS;
}

#endif
