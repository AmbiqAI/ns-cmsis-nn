/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "Internal/arm_nn_div_flt.h"

#if ARM_NN_ENABLE_F32

/* MVE has no vector divide. Use scalar division on all targets; the compiler
 * selects hardware VDIV where available. Keep one loop for flat and broadcast
 * runs, avoiding copies at every branch of the broadcast traversal. */
static __attribute__((noinline)) void div_f32(const float32_t *a,
                                              const float32_t *b,
                                              float32_t *out,
                                              int32_t count,
                                              int32_t stride_a,
                                              int32_t stride_b,
                                              float32_t activation_min,
                                              float32_t activation_max)
{
    for (int32_t i = 0; i < count; ++i)
    {
        const float32_t quotient = (float32_t)*a / (float32_t)*b;
        out[i] = arm_nn_clamp_scalar_f32((float32_t)quotient, activation_min, activation_max);
        a += stride_a;
        b += stride_b;
    }
}

arm_cmsis_nn_status arm_elementwise_div_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size)
{
    if (!input_1_vect || !input_2_vect || !output || block_size < 1 ||
        block_size > INT32_MAX / (int32_t)sizeof(float32_t))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    div_f32(input_1_vect, input_2_vect, output, block_size, 1, 1, out_activation_min, out_activation_max);
    return ARM_CMSIS_NN_SUCCESS;
}

    #define DIV_FULL(a, b, o, n) div_f32((a), (b), (o), (n), 1, 1, out_activation_min, out_activation_max)
    #define DIV_SCALAR_1(s, v, o, n) div_f32((s), (v), (o), (n), 0, 1, out_activation_min, out_activation_max)
    #define DIV_SCALAR_2(s, v, o, n) div_f32((v), (s), (o), (n), 1, 0, out_activation_min, out_activation_max)

arm_cmsis_nn_status arm_elementwise_div_broadcast_f32(const float32_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float32_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float32_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float32_t out_activation_min,
                                                      float32_t out_activation_max)
{
    if (!input_1_data || !input_2_data || !output_data ||
        !arm_nn_div_dims_valid(input_1_dims, input_2_dims, output_dims, sizeof(float32_t)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    ARM_NN_BROADCAST_WALK_NHWC(float32_t,
                               float32_t,
                               input_1_data,
                               input_1_dims,
                               input_2_data,
                               input_2_dims,
                               output_data,
                               output_dims,
                               DIV_FULL,
                               DIV_SCALAR_1,
                               DIV_SCALAR_2);
    return ARM_CMSIS_NN_SUCCESS;
}

    #undef DIV_FULL
    #undef DIV_SCALAR_1
    #undef DIV_SCALAR_2

#endif /* ARM_NN_ENABLE_F32 */
