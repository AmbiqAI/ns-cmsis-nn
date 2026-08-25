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
 * Title:        arm_rsqrt_f16
 * Description:  Elementwise reciprocal square root for float16 tensors
 *
 * $Date:        21 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

#include <string.h>

#if ARM_NN_ENABLE_F16

    #if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
static inline _Float16 arm_nn_sqrt_f16h(_Float16 value)
{
    _Float16 result;
    __asm__("vsqrt.f16 %0, %1" : "=t"(result) : "t"(value));
    return result;
}
    #endif

static inline uint16_t arm_nn_f16_to_bits(float16_t value)
{
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline float16_t arm_nn_f16_from_bits(uint16_t bits)
{
    float16_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

arm_cmsis_nn_status arm_rsqrt_f16(const float16_t *input, float16_t *output, int32_t block_size)
{
    if (!input || !output || block_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int32_t i = 0; i < block_size; ++i)
    {
        const uint16_t input_bits = arm_nn_f16_to_bits(input[i]);
        const uint16_t magnitude = input_bits & UINT16_C(0x7FFF);

        /* Classify by representation so -Ofast cannot discard IEEE special cases. */
        if (magnitude == 0)
        {
            output[i] = arm_nn_f16_from_bits((input_bits & UINT16_C(0x8000)) | UINT16_C(0x7C00));
            continue;
        }
        if (magnitude > UINT16_C(0x7C00))
        {
            output[i] = arm_nn_f16_from_bits(input_bits | UINT16_C(0x0200));
            continue;
        }
        if (input_bits == UINT16_C(0x7C00))
        {
            output[i] = arm_nn_f16_from_bits(0);
            continue;
        }
        if (input_bits & UINT16_C(0x8000))
        {
            output[i] = arm_nn_f16_from_bits(UINT16_C(0x7E00));
            continue;
        }

    #if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
        const _Float16 root = arm_nn_sqrt_f16h((_Float16)input[i]);
        output[i] = (float16_t)((_Float16)1.0f / root);
    #else
        const float32_t value = (float32_t)input[i];
        output[i] = (float16_t)(1.0f / __builtin_sqrtf(value));
    #endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupElementwise group
 */

#endif /* ARM_NN_ENABLE_F16 */
