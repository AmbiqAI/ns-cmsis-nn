/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */
#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

/* The walker calls this loop from several branches. Keep one copy in flash. */
static __attribute__((noinline)) void
arm_squared_difference_row_f16(const float16_t scalar, const float16_t *vec, float16_t *restrict out, int32_t count)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t b = vdupq_n_f16(scalar);
    for (int32_t i = 0; i < count; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(count - i));
        const float16x8_t diff = vsubq(vld1q_z(vec + i, p), b);
        vstrhq_p(out + i, vmulq(diff, diff), p);
    }
    #else

    for (int32_t i = 0; i < count; ++i)
    {
        const _Float16 diff = (_Float16)vec[i] - (_Float16)scalar;
        out[i] = (float16_t)(diff * diff);
    }
    #endif
}

    #define SQUARED_DIFF_ROW(s, v, o, n) arm_squared_difference_row_f16(*(s), (v), (o), (n))

arm_cmsis_nn_status arm_elementwise_squared_difference_broadcast_f16(const float16_t *input_1_data,
                                                                     const cmsis_nn_dims *input_1_dims,
                                                                     const float16_t *input_2_data,
                                                                     const cmsis_nn_dims *input_2_dims,
                                                                     float16_t *output_data,
                                                                     const cmsis_nn_dims *output_dims)
{
    if (!input_1_data || !input_2_data || !output_data || !input_1_dims || !input_2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input_1_dims, input_2_dims, output_dims) ||
        !arm_nn_broadcast_bytes_fit(output_dims, sizeof(float16_t)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    ARM_NN_BROADCAST_WALK_SYMMETRIC_NHWC(float16_t,
                                         input_1_data,
                                         input_1_dims,
                                         input_2_data,
                                         input_2_dims,
                                         output_data,
                                         output_dims,
                                         arm_elementwise_squared_difference_f16,
                                         SQUARED_DIFF_ROW);
    return ARM_CMSIS_NN_SUCCESS;
}

    #undef SQUARED_DIFF_ROW
#endif
