/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
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
 * Title:        arm_nn_gru_step_f32.c
 * Description:  Update GRU function for a single iteration step (float32)
 *
 * $Date:        14 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportLSTM
 * @{
 */

#if ARM_NN_ENABLE_F32

/*
 * Dot product of two float32 vectors. Mirrors the reduction used by the GRU
 * float16 step helper.
 */
__STATIC_INLINE float32_t arm_nn_gru_dot_f32(const float32_t *lhs, const float32_t *rhs, int32_t count)
{
    float32_t acc = 0.0f;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    float32x4_t vacc = vdupq_n_f32(0.0f);

    for (int32_t i = 0; i < count; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(count - i));
        vacc = vfmaq(vacc, vld1q_z(lhs + i, p), vld1q_z(rhs + i, p));
    }

    acc += arm_nn_vec_reduce_add_f32(vacc);
    #else
    for (int32_t i = 0; i < count; ++i)
    {
        acc += lhs[i] * rhs[i];
    }
    #endif

    return acc;
}

/*
 * Input projection for gate at hidden index h:  (W . x) + input_bias[h].
 */
__STATIC_INLINE float32_t arm_nn_gru_input_proj_f32(const cmsis_nn_gru_gate_f32 *gate,
                                                    const float32_t *input,
                                                    int32_t input_size,
                                                    int32_t h)
{
    float32_t acc = gate->input_bias ? gate->input_bias[h] : 0.0f;
    if (gate->input_weights)
    {
        const float32_t *w = gate->input_weights + (size_t)h * (size_t)input_size;
        acc += arm_nn_gru_dot_f32(input, w, input_size);
    }
    return acc;
}

/*
 * Recurrent projection for gate at hidden index h:  (U . h_prev) + hidden_bias[h].
 * When hidden is NULL (first step), the U . h_prev term is zero but the bias
 * still contributes.
 */
__STATIC_INLINE float32_t arm_nn_gru_hidden_proj_f32(const cmsis_nn_gru_gate_f32 *gate,
                                                     const float32_t *hidden,
                                                     int32_t hidden_size,
                                                     int32_t h)
{
    float32_t acc = gate->hidden_bias ? gate->hidden_bias[h] : 0.0f;
    if (hidden && gate->hidden_weights)
    {
        const float32_t *w = gate->hidden_weights + (size_t)h * (size_t)hidden_size;
        acc += arm_nn_gru_dot_f32(hidden, w, hidden_size);
    }
    return acc;
}

arm_cmsis_nn_status arm_nn_gru_step_f32(const float32_t *data_in,
                                        const float32_t *hidden_in,
                                        float32_t *hidden_out,
                                        const cmsis_nn_gru_params_f32 *params,
                                        cmsis_nn_gru_context_f32 *buffers,
                                        const int32_t batch_offset)
{
    if (!data_in || !hidden_out || !params || batch_offset <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = params->batch_size;
    const int32_t input_size = params->input_size;
    const int32_t hidden_size = params->hidden_size;
    const int32_t reset_after = params->reset_after;

    const cmsis_nn_gru_gate_f32 *zg = &params->update_gate;
    const cmsis_nn_gru_gate_f32 *rg = &params->reset_gate;
    const cmsis_nn_gru_gate_f32 *ng = &params->candidate_gate;

    float32_t *reset_buf = NULL;
    if (!reset_after)
    {
        // The pre-reset formulation needs the reset gate for all hidden units
        // before the candidate recurrent matmul, so a scratch vector is required.
        if (!buffers || !buffers->temp1)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        reset_buf = buffers->temp1;
    }

    for (int32_t b = 0; b < batch; b++)
    {
        const float32_t *x = data_in + (size_t)b * (size_t)batch_offset * (size_t)input_size;
        const float32_t *h_prev =
            hidden_in ? (hidden_in + (size_t)b * (size_t)batch_offset * (size_t)hidden_size) : NULL;
        float32_t *h_out = hidden_out + (size_t)b * (size_t)batch_offset * (size_t)hidden_size;

        if (!reset_after)
        {
            for (int32_t h = 0; h < hidden_size; h++)
            {
                const float32_t r_pre = arm_nn_gru_input_proj_f32(rg, x, input_size, h) +
                    arm_nn_gru_hidden_proj_f32(rg, h_prev, hidden_size, h);
                reset_buf[h] = arm_nn_sigmoid_scalar_f32(r_pre);
            }
        }

        for (int32_t h = 0; h < hidden_size; h++)
        {
            const float32_t z_pre = arm_nn_gru_input_proj_f32(zg, x, input_size, h) +
                arm_nn_gru_hidden_proj_f32(zg, h_prev, hidden_size, h);
            const float32_t z = arm_nn_sigmoid_scalar_f32(z_pre);

            float32_t cand_pre;
            if (reset_after)
            {
                // n = tanh( Wn.x + b_in + r * (Un.h_prev + b_hn) )
                const float32_t r_pre = arm_nn_gru_input_proj_f32(rg, x, input_size, h) +
                    arm_nn_gru_hidden_proj_f32(rg, h_prev, hidden_size, h);
                const float32_t r = arm_nn_sigmoid_scalar_f32(r_pre);
                const float32_t xh = arm_nn_gru_input_proj_f32(ng, x, input_size, h);
                const float32_t hh = arm_nn_gru_hidden_proj_f32(ng, h_prev, hidden_size, h);
                cand_pre = xh + r * hh;
            }
            else
            {
                // n = tanh( Wn.x + b_in + Un.(r . h_prev) + b_hn )
                float32_t hh = ng->hidden_bias ? ng->hidden_bias[h] : 0.0f;
                if (h_prev && ng->hidden_weights)
                {
                    const float32_t *w = ng->hidden_weights + (size_t)h * (size_t)hidden_size;
                    float32_t s = 0.0f;
                    for (int32_t k = 0; k < hidden_size; k++)
                    {
                        s += w[k] * (reset_buf[k] * h_prev[k]);
                    }
                    hh += s;
                }
                const float32_t xh = arm_nn_gru_input_proj_f32(ng, x, input_size, h);
                cand_pre = xh + hh;
            }

            const float32_t cand = arm_nn_tanh_scalar_ref_f32(cand_pre);
            const float32_t h_prev_h = h_prev ? h_prev[h] : 0.0f;
            h_out[h] = z * h_prev_h + (1.0f - z) * cand;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F32 */

/** @} end of supportLSTM group */
