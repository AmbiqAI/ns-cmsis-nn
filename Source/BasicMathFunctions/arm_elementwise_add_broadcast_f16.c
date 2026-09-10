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
 * Title:        arm_elementwise_add_broadcast_f16.c
 * Description:  Elementwise add for float16 tensors with NHWC broadcasting
 *
 * $Date:        5 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

/* vec + scalar: the arithmetic and clamp of arm_elementwise_add_f16 with the second operand
 * held in a register, so the result is bit-identical to the flat kernel over a materialised
 * broadcast operand (#415). */
static void arm_elementwise_add_scalar_f16(const float16_t *vec,
                                           float16_t scalar,
                                           float16_t *output,
                                           float16_t out_activation_min,
                                           float16_t out_activation_max,
                                           int32_t n)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t vs = vdupq_n_f16(scalar);
    const float16x8_t vmin = vdupq_n_f16(out_activation_min);
    const float16x8_t vmax = vdupq_n_f16(out_activation_max);
    /* Full blocks unpredicated, one predicated tail outside the loop, so there is no
     * vctp in a loop for the compiler's dlstp/letp conversion to act on (#427). */
    int32_t i = 0;
    for (; i <= n - 8; i += 8)
    {
        const float16x8_t va = vld1q(&vec[i]);
        vst1q(&output[i], arm_nn_clamp_propagate_nan_mve_f16(vaddq(va, vs), vmin, vmax));
    }
    if (i < n)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(n - i));
        const float16x8_t va = vld1q_z(&vec[i], p);
        vstrhq_p(&output[i], arm_nn_clamp_propagate_nan_mve_f16(vaddq(va, vs), vmin, vmax), p);
    }
    #else
    for (int32_t i = 0; i < n; ++i)
    {
        const _Float16 v = (_Float16)vec[i] + (_Float16)scalar;
        output[i] = arm_nn_clamp_scalar_f16((float16_t)v, out_activation_min, out_activation_max);
    }
    #endif
}

    /* Kernel adapters for ARM_NN_BROADCAST_WALK_NHWC; the clamp bounds are the enclosing function's
     * parameters. SCALAR_1 broadcasts one element of input 1 against a run of input 2, SCALAR_2 one
     * element of input 2 against a run of input 1. */
    #define ARM_ADD_BCAST_F16_FULL(a, b, o, n)                                                                         \
        (void)arm_elementwise_add_f16((a), (b), (o), out_activation_min, out_activation_max, (n))
    #define ARM_ADD_BCAST_F16_SCALAR_1(s, v, o, n)                                                                     \
        arm_elementwise_add_scalar_f16((v), *(s), (o), out_activation_min, out_activation_max, (n))
    #define ARM_ADD_BCAST_F16_SCALAR_2(s, v, o, n)                                                                     \
        arm_elementwise_add_scalar_f16((v), *(s), (o), out_activation_min, out_activation_max, (n))

/*
 * float16 elementwise add with NHWC broadcasting.
 *
 * Refer header file for details.
 */
arm_cmsis_nn_status arm_elementwise_add_broadcast_f16(const float16_t *input_1_data,
                                                      const cmsis_nn_dims *input_1_dims,
                                                      const float16_t *input_2_data,
                                                      const cmsis_nn_dims *input_2_dims,
                                                      float16_t *output_data,
                                                      const cmsis_nn_dims *output_dims,
                                                      float16_t out_activation_min,
                                                      float16_t out_activation_max)
{
    if (!input_1_data || !input_2_data || !output_data || !input_1_dims || !input_2_dims || !output_dims ||
        !arm_nn_broadcast_dims_valid(input_1_dims, input_2_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    ARM_NN_BROADCAST_WALK_NHWC(float16_t,
                               float16_t,
                               input_1_data,
                               input_1_dims,
                               input_2_data,
                               input_2_dims,
                               output_data,
                               output_dims,
                               ARM_ADD_BCAST_F16_FULL,
                               ARM_ADD_BCAST_F16_SCALAR_1,
                               ARM_ADD_BCAST_F16_SCALAR_2);

    return ARM_CMSIS_NN_SUCCESS;
}

    #undef ARM_ADD_BCAST_F16_FULL
    #undef ARM_ADD_BCAST_F16_SCALAR_1
    #undef ARM_ADD_BCAST_F16_SCALAR_2

/**
 * @} end of groupElementwise group
 */

#endif /* ARM_NN_ENABLE_F16 */
