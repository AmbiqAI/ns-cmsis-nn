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
 * Title:        arm_minmax_common_f16.c
 * Description:  Shared float16 min/max helper implementation
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nn_types.h"

#include <string.h>

#if ARM_NN_ENABLE_F16

    #include "Internal/arm_minmax_f16_common.h"
    #include "Internal/arm_nn_broadcast_walk.h"

    /* Scalar-path helper: only referenced from the #else branches of the loops
     * below, so keep it out of MVE builds (-Wunused-function, issue #246). */
    #if !defined(ARM_MATH_MVE_FLOAT16) || defined(ARM_MATH_AUTOVECTORIZE)
static float16_t arm_minmax_select_f16(float16_t a, float16_t b, int32_t select_max)
{
    /* Bitwise select (the arm_nn_propagate_nan_f16h idiom): a _Float16 ternary
     * here becomes an HFmode conditional move that ICEs GCC 14.3 at -O3
     * (PR target/118460) — even when written as a float32 select, since GCC
     * narrows a select of exactly-round-tripped halves back to HFmode. The
     * mask form has no FP select at all, and returns the chosen operand's
     * exact bits. Kept self-contained instead of using the arm_nn_*_f16h
     * helpers because this file must also compile with ARM_NN_ENABLE_F16=0,
     * where those helpers are gated out.
     *
     * The a_ge_b comparison gives this the same tie-break as the float32 twin:
     * a tie between zeros of opposite sign resolves by operand position, where
     * vmaxnmq / vminnmq below resolve it by sign (issue #316). Left as is for
     * the reasons set out on arm_minmax_select_f32(). An fmaxf() / fminf() form
     * would also have to clear the mask-select rule in AGENTS.md: written as a
     * float32 select narrowed back to _Float16 it compiles with the pinned
     * 14.2.Rel1 toolchain for cortex-m55 with ARM_MATH_AUTOVECTORIZE at -Ofast
     * and at -O3 with -fno-finite-math-only, emitting vmaxnm.f32 / vminnm.f32
     * around vcvtb pairs, but that is the shape AGENTS.md and #344 record GCC
     * 14.3 ICEing on, and 14.3.Rel1 is gated on every pull request. The zero tie
     * and NaN are documented as unspecified in arm_nnfunctions_flt.h. */
    uint16_t a_bits, b_bits;
    memcpy(&a_bits, &a, sizeof(a_bits));
    memcpy(&b_bits, &b, sizeof(b_bits));

    const int32_t a_ge_b = ((float32_t)a >= (float32_t)b);
    const uint16_t pick_a = (uint16_t)(0U - (uint32_t)(a_ge_b == (select_max != 0)));
    const uint16_t r_bits = (uint16_t)((a_bits & pick_a) | (b_bits & (uint16_t)~pick_a));

    float16_t r;
    memcpy(&r, &r_bits, sizeof(r));
    return r;
}
    #endif

static arm_cmsis_nn_status arm_minmax_no_broadcast_f16(const float16_t *input_1,
                                                       const float16_t *input_2,
                                                       float16_t *output,
                                                       int32_t flat_size,
                                                       int32_t select_max)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (select_max)
    {
        for (int32_t i = 0; i < flat_size; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(flat_size - i));
            const float16x8_t v_in1 = vld1q_z(input_1 + i, p);
            const float16x8_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vmaxnmq(v_in1, v_in2), p);
        }
    }
    else
    {
        for (int32_t i = 0; i < flat_size; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(flat_size - i));
            const float16x8_t v_in1 = vld1q_z(input_1 + i, p);
            const float16x8_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vminnmq(v_in1, v_in2), p);
        }
    }
    #else
    for (int32_t i = 0; i < flat_size; ++i)
    {
        output[i] = arm_minmax_select_f16(input_1[i], input_2[i], select_max);
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_minmax_scalar_f16(const float16_t *input_1,
                                                 const float16_t *input_2,
                                                 float16_t *output,
                                                 int32_t flat_size,
                                                 int32_t select_max)
{
    const float16_t in1 = *input_1;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t v_in1 = vdupq_n_f16(in1);
    if (select_max)
    {
        for (int32_t i = 0; i < flat_size; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(flat_size - i));
            const float16x8_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vmaxnmq(v_in1, v_in2), p);
        }
    }
    else
    {
        for (int32_t i = 0; i < flat_size; i += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(flat_size - i));
            const float16x8_t v_in2 = vld1q_z(input_2 + i, p);
            vst1q_p(output + i, vminnmq(v_in1, v_in2), p);
        }
    }
    #else
    for (int32_t i = 0; i < flat_size; ++i)
    {
        output[i] = arm_minmax_select_f16(in1, input_2[i], select_max);
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

    /* Kernel adapters for ARM_NN_BROADCAST_WALK_NHWC; select_max is the enclosing function's parameter. */
    #define ARM_MINMAX_FULL_F16(a, b, o, n) arm_minmax_no_broadcast_f16((a), (b), (o), (n), select_max)
    #define ARM_MINMAX_SCALAR_F16(s, v, o, n) arm_minmax_scalar_f16((s), (v), (o), (n), select_max)

arm_cmsis_nn_status arm_minmax_f16_impl(const cmsis_nn_context *ctx,
                                        const float16_t *input_1_data,
                                        const cmsis_nn_dims *input_1_dims,
                                        const float16_t *input_2_data,
                                        const cmsis_nn_dims *input_2_dims,
                                        float16_t *output_data,
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

    ARM_NN_BROADCAST_WALK_NHWC(float16_t,
                               float16_t,
                               input_1_data,
                               input_1_dims,
                               input_2_data,
                               input_2_dims,
                               output_data,
                               output_dims,
                               ARM_MINMAX_FULL_F16,
                               ARM_MINMAX_SCALAR_F16,
                               ARM_MINMAX_SCALAR_F16);

    return ARM_CMSIS_NN_SUCCESS;
}

    #undef ARM_MINMAX_FULL_F16
    #undef ARM_MINMAX_SCALAR_F16

#endif /* ARM_NN_ENABLE_F16 */
