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
 * Title:        arm_nn_gru_step_f16.c
 * Description:  Update GRU function for a single iteration step (float16)
 *
 * $Date:        22 July 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture with FP16 support
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

#if ARM_NN_ENABLE_F16

/*
 * Dot product of two float16 vectors, returning a _Float16 accumulator.
 * Mirrors the reduction used by the LSTM float16 step helper.
 */
__STATIC_INLINE _Float16 arm_nn_gru_dot_f16(const float16_t *lhs, const float16_t *rhs, int32_t count)
{
    _Float16 acc = (_Float16)0.0f;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);

    for (int32_t i = 0; i < count; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(count - i));
        vacc = vfmaq(vacc, vld1q_z(lhs + i, p), vld1q_z(rhs + i, p));
    }

    acc += (_Float16)arm_nn_vec_reduce_add_f16(vacc);
    #else
    for (int32_t i = 0; i < count; ++i)
    {
        acc += (_Float16)lhs[i] * (_Float16)rhs[i];
    }
    #endif

    return acc;
}

/*
 * Input projection for gate at hidden index h:  (W . x) + input_bias[h].
 */
__STATIC_INLINE _Float16
arm_nn_gru_input_proj_f16(const cmsis_nn_gru_gate_f16 *gate, const float16_t *input, int32_t input_size, int32_t h)
{
    _Float16 acc = gate->input_bias ? (_Float16)gate->input_bias[h] : (_Float16)0.0f;
    if (gate->input_weights)
    {
        const float16_t *w = gate->input_weights + (size_t)h * (size_t)input_size;
        acc += arm_nn_gru_dot_f16(input, w, input_size);
    }
    return acc;
}

/*
 * Recurrent projection for gate at hidden index h:  (U . h_prev) + hidden_bias[h].
 * When hidden is NULL (first step), the U . h_prev term is zero but the bias
 * still contributes.
 */
__STATIC_INLINE _Float16
arm_nn_gru_hidden_proj_f16(const cmsis_nn_gru_gate_f16 *gate, const float16_t *hidden, int32_t hidden_size, int32_t h)
{
    _Float16 acc = gate->hidden_bias ? (_Float16)gate->hidden_bias[h] : (_Float16)0.0f;
    if (hidden && gate->hidden_weights)
    {
        const float16_t *w = gate->hidden_weights + (size_t)h * (size_t)hidden_size;
        acc += arm_nn_gru_dot_f16(hidden, w, hidden_size);
    }
    return acc;
}

arm_cmsis_nn_status arm_nn_gru_step_f16(const float16_t *data_in,
                                        const float16_t *hidden_in,
                                        float16_t *hidden_out,
                                        const cmsis_nn_gru_params_f16 *params,
                                        cmsis_nn_gru_context_f16 *buffers,
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

    const cmsis_nn_gru_gate_f16 *zg = &params->update_gate;
    const cmsis_nn_gru_gate_f16 *rg = &params->reset_gate;
    const cmsis_nn_gru_gate_f16 *ng = &params->candidate_gate;

    float16_t *reset_buf = NULL;
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
        const float16_t *x = data_in + (size_t)b * (size_t)batch_offset * (size_t)input_size;
        const float16_t *h_prev =
            hidden_in ? (hidden_in + (size_t)b * (size_t)batch_offset * (size_t)hidden_size) : NULL;
        float16_t *h_out = hidden_out + (size_t)b * (size_t)batch_offset * (size_t)hidden_size;

        if (!reset_after)
        {
            for (int32_t h = 0; h < hidden_size; h++)
            {
                const _Float16 r_pre = (_Float16)(arm_nn_gru_input_proj_f16(rg, x, input_size, h) +
                                                  arm_nn_gru_hidden_proj_f16(rg, h_prev, hidden_size, h));
                reset_buf[h] = arm_nn_sigmoid_scalar_f16((float16_t)r_pre);
            }
        }

        for (int32_t h = 0; h < hidden_size; h++)
        {
            const _Float16 z_pre = (_Float16)(arm_nn_gru_input_proj_f16(zg, x, input_size, h) +
                                              arm_nn_gru_hidden_proj_f16(zg, h_prev, hidden_size, h));
            const _Float16 z = (_Float16)arm_nn_sigmoid_scalar_f16((float16_t)z_pre);

            _Float16 cand_pre;
            if (reset_after)
            {
                // n = tanh( Wn.x + b_in + r * (Un.h_prev + b_hn) )
                const _Float16 r_pre = (_Float16)(arm_nn_gru_input_proj_f16(rg, x, input_size, h) +
                                                  arm_nn_gru_hidden_proj_f16(rg, h_prev, hidden_size, h));
                const _Float16 r = (_Float16)arm_nn_sigmoid_scalar_f16((float16_t)r_pre);
                const _Float16 xh = arm_nn_gru_input_proj_f16(ng, x, input_size, h);
                const _Float16 hh = arm_nn_gru_hidden_proj_f16(ng, h_prev, hidden_size, h);
                cand_pre = (_Float16)(xh + r * hh);
            }
            else
            {
                // n = tanh( Wn.x + b_in + Un.(r . h_prev) + b_hn )
                _Float16 hh = ng->hidden_bias ? (_Float16)ng->hidden_bias[h] : (_Float16)0.0f;
                if (h_prev && ng->hidden_weights)
                {
                    const float16_t *w = ng->hidden_weights + (size_t)h * (size_t)hidden_size;
                    _Float16 s = (_Float16)0.0f;
                    for (int32_t k = 0; k < hidden_size; k++)
                    {
                        s += (_Float16)w[k] * ((_Float16)reset_buf[k] * (_Float16)h_prev[k]);
                    }
                    hh += s;
                }
                const _Float16 xh = arm_nn_gru_input_proj_f16(ng, x, input_size, h);
                cand_pre = (_Float16)(xh + hh);
            }

            const _Float16 cand = (_Float16)arm_nn_tanh_scalar_ref_f16((float16_t)cand_pre);
            const _Float16 h_prev_h = h_prev ? (_Float16)h_prev[h] : (_Float16)0.0f;
            h_out[h] = (float16_t)(z * h_prev_h + ((_Float16)1.0f - z) * cand);
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F16 */

/** @} end of supportLSTM group */
