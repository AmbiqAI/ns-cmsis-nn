/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

arm_cmsis_nn_status arm_elementwise_squared_difference_f32(const float32_t *input_1_vect,
                                                           const float32_t *input_2_vect,
                                                           float32_t *output,
                                                           int32_t block_size)
{
    if (!input_1_vect || !input_2_vect || !output || block_size < 1 ||
        block_size > INT32_MAX / (int32_t)sizeof(float32_t))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t i = 0; i < block_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
        const float32x4_t a = vld1q_z(&input_1_vect[i], p);
        const float32x4_t b = vld1q_z(&input_2_vect[i], p);
        const float32x4_t difference = vsubq(a, b);
        vstrwq_p(&output[i], vmulq(difference, difference), p);
    }
    #else
    for (int32_t i = 0; i < block_size; ++i)
    {
        const float32_t difference = input_1_vect[i] - input_2_vect[i];
        output[i] = difference * difference;
    }
    #endif
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F32 */
