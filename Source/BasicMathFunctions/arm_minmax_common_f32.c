/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_minmax_common_f32.c
 * Description:  Shared float32 min/max helper implementation
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nn_types.h"

#if ARM_NN_ENABLE_F32

    #include "Internal/arm_minmax_f32_common.h"
    #include "Internal/arm_nn_broadcast_walk.h"

    /* Scalar-path helper: only referenced from the #else branches of the loops
     * below, so keep it out of MVE builds (-Wunused-function, issue #246). */
    #if !defined(ARM_MATH_MVEF) || defined(ARM_MATH_AUTOVECTORIZE)
static float32_t arm_minmax_select_f32(float32_t a, float32_t b, int32_t select_max)
{
    return (a >= b) ? (select_max ? a : b) : (select_max ? b : a);
}
    #endif

static arm_cmsis_nn_status arm_minmax_no_broadcast_f32(const float32_t *input_1,
                                                       const float32_t *input_2,
                                                       float32_t *output,
                                                       int32_t flat_size,
                                                       int32_t select_max)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (select_max)
    {
        for (int32_t i = 0; i < flat_size; i += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(flat_size - i));
            const float32x4_t v_in1 = vld1q_z(input_1 + i, p);
            const float32x4_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vmaxnmq(v_in1, v_in2), p);
        }
    }
    else
    {
        for (int32_t i = 0; i < flat_size; i += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(flat_size - i));
            const float32x4_t v_in1 = vld1q_z(input_1 + i, p);
            const float32x4_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vminnmq(v_in1, v_in2), p);
        }
    }
    #else
    for (int32_t i = 0; i < flat_size; ++i)
    {
        output[i] = arm_minmax_select_f32(input_1[i], input_2[i], select_max);
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_minmax_scalar_f32(const float32_t *input_1,
                                                 const float32_t *input_2,
                                                 float32_t *output,
                                                 int32_t flat_size,
                                                 int32_t select_max)
{
    const float32_t in1 = *input_1;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t v_in1 = vdupq_n_f32(in1);
    if (select_max)
    {
        for (int32_t i = 0; i < flat_size; i += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(flat_size - i));
            const float32x4_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vmaxnmq(v_in1, v_in2), p);
        }
    }
    else
    {
        for (int32_t i = 0; i < flat_size; i += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(flat_size - i));
            const float32x4_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vminnmq(v_in1, v_in2), p);
        }
    }
    #else
    for (int32_t i = 0; i < flat_size; ++i)
    {
        output[i] = arm_minmax_select_f32(in1, input_2[i], select_max);
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

    /* Kernel adapters for ARM_NN_BROADCAST_WALK_NHWC; select_max is the enclosing function's parameter. */
    #define ARM_MINMAX_FULL_F32(a, b, o, n) arm_minmax_no_broadcast_f32((a), (b), (o), (n), select_max)
    #define ARM_MINMAX_SCALAR_F32(s, v, o, n) arm_minmax_scalar_f32((s), (v), (o), (n), select_max)

arm_cmsis_nn_status arm_minmax_f32_impl(const cmsis_nn_context *ctx,
                                        const float32_t *input_1_data,
                                        const cmsis_nn_dims *input_1_dims,
                                        const float32_t *input_2_data,
                                        const cmsis_nn_dims *input_2_dims,
                                        float32_t *output_data,
                                        const cmsis_nn_dims *output_dims,
                                        int32_t select_max)
{
    (void)ctx;
    if (!input_1_data || !input_2_data || !output_data || !input_1_dims || !input_2_dims || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (!arm_nn_broadcast_dims_valid(input_1_dims, input_2_dims, output_dims))
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
                               ARM_MINMAX_FULL_F32,
                               ARM_MINMAX_SCALAR_F32,
                               ARM_MINMAX_SCALAR_F32);

    return ARM_CMSIS_NN_SUCCESS;
}

    #undef ARM_MINMAX_FULL_F32
    #undef ARM_MINMAX_SCALAR_F32

#endif /* ARM_NN_ENABLE_F32 */
