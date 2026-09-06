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

/*
 * Update gate z = sigmoid(Wz.x + b_iz + Uz.h_prev + b_hz) for hidden index h.
 */
__STATIC_INLINE float32_t arm_nn_gru_update_f32(const cmsis_nn_gru_params_f32 *params,
                                                const float32_t *x,
                                                const float32_t *h_prev,
                                                int32_t h)
{
    const cmsis_nn_gru_gate_f32 *zg = &params->update_gate;
    const float32_t z_pre = arm_nn_gru_input_proj_f32(zg, x, params->input_size, h) +
        arm_nn_gru_hidden_proj_f32(zg, h_prev, params->hidden_size, h);
    return arm_nn_sigmoid_scalar_f32(z_pre);
}

/*
 * Candidate pre-activation for hidden index h. On the pre-reset path reset_buf holds r . h_prev, or NULL when
 * that recurrent term is dead (#251).
 */
__STATIC_INLINE float32_t arm_nn_gru_candidate_pre_f32(const cmsis_nn_gru_params_f32 *params,
                                                       const float32_t *x,
                                                       const float32_t *h_prev,
                                                       const float32_t *reset_buf,
                                                       int32_t h)
{
    const cmsis_nn_gru_gate_f32 *rg = &params->reset_gate;
    const cmsis_nn_gru_gate_f32 *ng = &params->candidate_gate;
    const int32_t input_size = params->input_size;
    const int32_t hidden_size = params->hidden_size;
    const float32_t xh = arm_nn_gru_input_proj_f32(ng, x, input_size, h);

    if (params->reset_after)
    {
        // n = tanh( Wn.x + b_in + r * (Un.h_prev + b_hn) )
        const float32_t r_pre =
            arm_nn_gru_input_proj_f32(rg, x, input_size, h) + arm_nn_gru_hidden_proj_f32(rg, h_prev, hidden_size, h);
        const float32_t r = arm_nn_sigmoid_scalar_f32(r_pre);
        const float32_t hh = arm_nn_gru_hidden_proj_f32(ng, h_prev, hidden_size, h);
        return xh + r * hh;
    }

    // n = tanh( Wn.x + b_in + Un.(r . h_prev) + b_hn )
    float32_t hh = ng->hidden_bias ? ng->hidden_bias[h] : 0.0f;
    if (reset_buf)
    {
        const float32_t *w = ng->hidden_weights + (size_t)h * (size_t)hidden_size;
        float32_t s = 0.0f;
        for (int32_t k = 0; k < hidden_size; k++)
        {
            s += w[k] * reset_buf[k];
        }
        hh += s;
    }
    return xh + hh;
}

    #if !defined(ARM_MATH_MVEF) || defined(ARM_MATH_AUTOVECTORIZE)
/*
 * h = z * h_prev + (1 - z) * n, fused as fma(z, h_prev, (1 - z) * n). The MVE block below mirrors this with
 * vfmaq(vmulq(1 - z, n), z, h_prev) so the two legs round identically (#251, #315).
 */
__STATIC_INLINE float32_t arm_nn_gru_combine_f32(float32_t z, float32_t h_prev, float32_t cand)
{
        #if defined(__clang__)
            // Under fast-math clang rewrites (1 - z) * n as n - z*n (vfms), a different rounding; pin it off (#251).
            #pragma clang fp contract(off) reassociate(off)
        #endif
    return fmaf(z, h_prev, (1.0f - z) * cand);
}

    #endif

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
/*
 * One block of up to 4 hidden units: sigmoid per lane (scalar), candidate tanh and the combine as vector ops.
 * The last partial block is predicated rather than scalar so every unit takes the same arithmetic (#315).
 */
__STATIC_INLINE void arm_nn_gru_block_f32(const cmsis_nn_gru_params_f32 *params,
                                          const float32_t *x,
                                          const float32_t *h_prev,
                                          const float32_t *reset_buf,
                                          float32_t *h_out,
                                          int32_t h,
                                          int32_t lanes)
{
    float32_t z_lane[4] = {0};
    float32_t cand_pre_lane[4] = {0};

    for (int32_t lane = 0; lane < lanes; lane++)
    {
        z_lane[lane] = arm_nn_gru_update_f32(params, x, h_prev, h + lane);
        cand_pre_lane[lane] = arm_nn_gru_candidate_pre_f32(params, x, h_prev, reset_buf, h + lane);
    }

    const float32x4_t vz = vld1q(z_lane);
    const float32x4_t vcand_pre = vld1q(cand_pre_lane);
    // The vector tanh maps a NaN lane to a finite value; the scalar leg propagates it (#251). Classify NaN in the
    // integer domain (arm_nn_clamp_propagate_nan_mve_f32 style, immune to -ffinite-math-only) before the tanh and
    // select any NaN afterwards: MVE arithmetic returns the default NaN whatever the source, and an all-ones splat
    // is one immediate move with no live range across the tanh.
    const mve_pred16_t nan_p = vcmphiq_n_u32(vshlq_n_u32(vreinterpretq_u32_f32(vcand_pre), 1), 0xFF000000u);
    const float32x4_t vcand =
        vpselq(vreinterpretq_f32_u32(vdupq_n_u32(0xFFFFFFFFu)), arm_nn_vtanh_lut_direct_mve_f32(vcand_pre), nan_p);
    const float32x4_t vp = vmulq(vsubq(vdupq_n_f32(1.0f), vz), vcand);

    if (lanes == 4)
    {
        const float32x4_t vh_prev = h_prev ? vld1q(h_prev + h) : vdupq_n_f32(0.0f);
        vst1q(h_out + h, vfmaq(vp, vz, vh_prev));
    }
    else
    {
        const mve_pred16_t p = vctp32q((uint32_t)lanes);
        const float32x4_t vh_prev = h_prev ? vld1q_z(h_prev + h, p) : vdupq_n_f32(0.0f);
        vst1q_p(h_out + h, vfmaq(vp, vz, vh_prev), p);
    }
}
    #endif

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

    const cmsis_nn_gru_gate_f32 *rg = &params->reset_gate;
    const cmsis_nn_gru_gate_f32 *ng = &params->candidate_gate;

    float32_t *stage = NULL;
    if (!reset_after)
    {
        // The pre-reset formulation needs the reset gate for all hidden units
        // before the candidate recurrent matmul, so a scratch vector is required.
        if (!buffers || !buffers->temp1)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        stage = buffers->temp1;
    }

    for (int32_t b = 0; b < batch; b++)
    {
        const float32_t *x = data_in + (size_t)b * (size_t)batch_offset * (size_t)input_size;
        const float32_t *h_prev =
            hidden_in ? (hidden_in + (size_t)b * (size_t)batch_offset * (size_t)hidden_size) : NULL;
        float32_t *h_out = hidden_out + (size_t)b * (size_t)batch_offset * (size_t)hidden_size;

        // Pre-reset: the candidate recurrent term is Un.(r . h_prev). Stage r . h_prev once per batch so the
        // n^2 candidate loop reads a single operand; skipped when that term is dead (#251).
        const float32_t *reset_buf = NULL;
        if (!reset_after && h_prev && ng->hidden_weights)
        {
            for (int32_t h = 0; h < hidden_size; h++)
            {
                const float32_t r_pre = arm_nn_gru_input_proj_f32(rg, x, input_size, h) +
                    arm_nn_gru_hidden_proj_f32(rg, h_prev, hidden_size, h);
                stage[h] = arm_nn_sigmoid_scalar_f32(r_pre) * h_prev[h];
            }
            reset_buf = stage;
        }

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t h = 0;
        for (; h + 4 <= hidden_size; h += 4)
        {
            arm_nn_gru_block_f32(params, x, h_prev, reset_buf, h_out, h, 4);
        }
        if (h < hidden_size)
        {
            arm_nn_gru_block_f32(params, x, h_prev, reset_buf, h_out, h, hidden_size - h);
        }
    #else
        for (int32_t h = 0; h < hidden_size; h++)
        {
            const float32_t z = arm_nn_gru_update_f32(params, x, h_prev, h);
            const float32_t cand =
                arm_nn_tanh_scalar_ref_f32(arm_nn_gru_candidate_pre_f32(params, x, h_prev, reset_buf, h));
            const float32_t h_prev_h = h_prev ? h_prev[h] : 0.0f;
            h_out[h] = arm_nn_gru_combine_f32(z, h_prev_h, cand);
        }
    #endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F32 */

/** @} end of supportLSTM group */
