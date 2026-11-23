/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */
#include "Internal/arm_nn_broadcast_walk.h"
#include "Internal/arm_nn_fp_control.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

/* The walker calls this loop from several branches. Keep one copy in flash. */
static __attribute__((noinline)) void arm_squared_difference_row_f32(const float32_t scalar,
                                                                     const float32_t *vec,
                                                                     float32_t *restrict out,
                                                                     int32_t count,
                                                                     int flush)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t b = vdupq_n_f32(scalar);
    uint32x4_t smallest = vdupq_n_u32(UINT32_MAX);
    for (int32_t i = 0; i < count; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(count - i));
        const float32x4_t diff = vsubq(vld1q_z(vec + i, p), b);
        vstrwq_p(out + i, vmulq(diff, diff), p);
        // Same zero-excluding underflow test as arm_elementwise_squared_difference_f32.
        const uint32x4_t key = vsubq_n_u32(vshlq_n_u32(vreinterpretq_u32_f32(diff), 1), 1);
        smallest = vminq_m(smallest, smallest, key, p);
    }
    if (!flush && vminvq(UINT32_MAX, smallest) < 0x3fffffffu)
    {
        // With gradual underflow, replay the row on the scalar FPU. Volatile forces scalar rounding.
        for (int32_t i = 0; i < count; ++i)
        {
            volatile float32_t diff = vec[i] - scalar;
            out[i] = diff * diff;
        }
    }
    #else
    (void)flush;
    for (int32_t i = 0; i < count; ++i)
    {
        const float32_t diff = vec[i] - scalar;
        out[i] = (float32_t)(diff * diff);
    }
    #endif
}

    #define SQUARED_DIFF_ROW(s, v, o, n) arm_squared_difference_row_f32(*(s), (v), (o), (n), flush)

arm_cmsis_nn_status arm_elementwise_squared_difference_broadcast_f32(const float32_t *input_1_data,
                                                                     const cmsis_nn_dims *input_1_dims,
                                                                     const float32_t *input_2_data,
                                                                     const cmsis_nn_dims *input_2_dims,
                                                                     float32_t *output_data,
                                                                     const cmsis_nn_dims *output_dims)
{
    if (!input_1_data || !input_2_data || !output_data || !input_1_dims || !input_2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input_1_dims, input_2_dims, output_dims) ||
        !arm_nn_broadcast_bytes_fit(output_dims, sizeof(float32_t)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) && defined(ARM_NN_FP_CONTROL_AVAILABLE)
    const int flush = (arm_nn_get_fp_control() & (1u << 24)) != 0;
    #else
    const int flush = 0;
    #endif

    ARM_NN_BROADCAST_WALK_SYMMETRIC_NHWC(float32_t,
                                         input_1_data,
                                         input_1_dims,
                                         input_2_data,
                                         input_2_dims,
                                         output_data,
                                         output_dims,
                                         arm_elementwise_squared_difference_f32,
                                         SQUARED_DIFF_ROW);
    return ARM_CMSIS_NN_SUCCESS;
}

    #undef SQUARED_DIFF_ROW
#endif
