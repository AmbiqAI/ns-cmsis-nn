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
 * Title:        arm_elementwise_mul_broadcast_f32.c
 * Description:  Elementwise multiply for float32 tensors with NHWC broadcasting
 *
 * $Date:        5 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 * @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

/* vec * scalar: the arithmetic and clamp of arm_elementwise_mul_f32 with the second operand
 * held in a register, so the result is bit-identical to the flat kernel over a materialised
 * broadcast operand (#415). */
static void arm_elementwise_mul_scalar_f32(const float32_t *vec,
                                           float32_t scalar,
                                           float32_t *output,
                                           float32_t out_activation_min,
                                           float32_t out_activation_max,
                                           int32_t n)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t vs = vdupq_n_f32(scalar);
    const float32x4_t vmin = vdupq_n_f32(out_activation_min);
    const float32x4_t vmax = vdupq_n_f32(out_activation_max);
    /* Full blocks unpredicated, one predicated tail outside the loop, so there is no
     * vctp in a loop for the compiler's dlstp/letp conversion to act on (#427). */
    int32_t i = 0;
    for (; i <= n - 4; i += 4)
    {
        const float32x4_t va = vld1q(&vec[i]);
        vst1q(&output[i], arm_nn_clamp_propagate_nan_mve_f32(vmulq(va, vs), vmin, vmax));
    }
    if (i < n)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(n - i));
        const float32x4_t va = vld1q_z(&vec[i], p);
        vstrwq_p(&output[i], arm_nn_clamp_propagate_nan_mve_f32(vmulq(va, vs), vmin, vmax), p);
    }
    #else
    for (int32_t i = 0; i < n; ++i)
    {
        const float32_t v = vec[i] * scalar;
        output[i] = arm_nn_clamp_scalar_f32(v, out_activation_min, out_activation_max);
    }
    #endif
}

    /* Kernel adapters for ARM_NN_BROADCAST_WALK_NHWC; the clamp bounds are the enclosing function's
     * parameters. SCALAR_1 broadcasts one element of input 1 against a run of input 2, SCALAR_2 one
     * element of input 2 against a run of input 1. */
    #define ARM_MUL_BCAST_F32_FULL(a, b, o, n)                                                                         \
        (void)arm_elementwise_mul_f32((a), (b), (o), out_activation_min, out_activation_max, (n))
    #define ARM_MUL_BCAST_F32_SCALAR_1(s, v, o, n)                                                                     \
        arm_elementwise_mul_scalar_f32((v), *(s), (o), out_activation_min, out_activation_max, (n))
    #define ARM_MUL_BCAST_F32_SCALAR_2(s, v, o, n)                                                                     \
        arm_elementwise_mul_scalar_f32((v), *(s), (o), out_activation_min, out_activation_max, (n))

/*
 * float32 elementwise multiply with NHWC broadcasting.
 *
 * Refer header file for details.
 */
arm_cmsis_nn_status arm_elementwise_mul_broadcast_f32(const float32_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float32_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float32_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float32_t out_activation_min,
                                                      float32_t out_activation_max)
{
    if (!input_1_data || !input_2_data || !output_data || !input_1_dims || !input_2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input_1_dims, input_2_dims, output_dims))
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
                               ARM_MUL_BCAST_F32_FULL,
                               ARM_MUL_BCAST_F32_SCALAR_1,
                               ARM_MUL_BCAST_F32_SCALAR_2);

    return ARM_CMSIS_NN_SUCCESS;
}

    #undef ARM_MUL_BCAST_F32_FULL
    #undef ARM_MUL_BCAST_F32_SCALAR_1
    #undef ARM_MUL_BCAST_F32_SCALAR_2

/**
 * @} end of groupElementwise group
 */

#endif /* ARM_NN_ENABLE_F32 */
