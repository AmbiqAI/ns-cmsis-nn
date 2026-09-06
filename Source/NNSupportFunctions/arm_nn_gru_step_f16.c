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

/*
 * Update gate z = sigmoid(Wz.x + b_iz + Uz.h_prev + b_hz) for hidden index h.
 */
__STATIC_INLINE float16_t arm_nn_gru_update_f16(const cmsis_nn_gru_params_f16 *params,
                                                const float16_t *x,
                                                const float16_t *h_prev,
                                                int32_t h)
{
    const cmsis_nn_gru_gate_f16 *zg = &params->update_gate;
    const _Float16 z_pre = (_Float16)(arm_nn_gru_input_proj_f16(zg, x, params->input_size, h) +
                                      arm_nn_gru_hidden_proj_f16(zg, h_prev, params->hidden_size, h));
    return arm_nn_sigmoid_scalar_f16((float16_t)z_pre);
}

/*
 * Candidate pre-activation for hidden index h. On the pre-reset path reset_buf holds r . h_prev, or NULL when
 * that recurrent term is dead (#251).
 */
__STATIC_INLINE _Float16 arm_nn_gru_candidate_pre_f16(const cmsis_nn_gru_params_f16 *params,
                                                      const float16_t *x,
                                                      const float16_t *h_prev,
                                                      const float16_t *reset_buf,
                                                      int32_t h)
{
    const cmsis_nn_gru_gate_f16 *rg = &params->reset_gate;
    const cmsis_nn_gru_gate_f16 *ng = &params->candidate_gate;
    const int32_t input_size = params->input_size;
    const int32_t hidden_size = params->hidden_size;
    const _Float16 xh = arm_nn_gru_input_proj_f16(ng, x, input_size, h);

    if (params->reset_after)
    {
        // n = tanh( Wn.x + b_in + r * (Un.h_prev + b_hn) )
        const _Float16 r_pre = (_Float16)(arm_nn_gru_input_proj_f16(rg, x, input_size, h) +
                                          arm_nn_gru_hidden_proj_f16(rg, h_prev, hidden_size, h));
        const _Float16 r = (_Float16)arm_nn_sigmoid_scalar_f16((float16_t)r_pre);
        const _Float16 hh = arm_nn_gru_hidden_proj_f16(ng, h_prev, hidden_size, h);
        return (_Float16)(xh + r * hh);
    }

    // n = tanh( Wn.x + b_in + Un.(r . h_prev) + b_hn )
    _Float16 hh = ng->hidden_bias ? (_Float16)ng->hidden_bias[h] : (_Float16)0.0f;
    if (reset_buf)
    {
        const float16_t *w = ng->hidden_weights + (size_t)h * (size_t)hidden_size;
        _Float16 s = (_Float16)0.0f;
        for (int32_t k = 0; k < hidden_size; k++)
        {
            s += (_Float16)w[k] * (_Float16)reset_buf[k];
        }
        hh += s;
    }
    return (_Float16)(xh + hh);
}

    #if !defined(ARM_MATH_MVE_FLOAT16) || defined(ARM_MATH_AUTOVECTORIZE)
/*
 * h = z * h_prev + (1 - z) * n, fused as fma(z, h_prev, (1 - z) * n). The MVE block below mirrors this with
 * vfmaq(vmulq(1 - z, n), z, h_prev) so the two legs round identically (#251, #315).
 */
__STATIC_INLINE _Float16 arm_nn_gru_combine_f16(_Float16 z, _Float16 h_prev, _Float16 cand)
{
        #if defined(__clang__)
            // Under fast-math clang may rewrite (1 - z) * n as n - z*n (vfms), a different rounding. The pragma holds
            // on ATfE; armclang ignores it at -Ofast (scalar leg only, no shipped M55 leg takes this path, #251).
            #pragma clang fp contract(off) reassociate(off)
        #endif
    const _Float16 p = ((_Float16)1.0f - z) * cand;
        #if defined(__GNUC__) && !defined(__clang__) && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    return __builtin_fmaf16(z, h_prev, p);
        #else
    // clang lowers __builtin_fmaf16 to an fmaf16 libcall that no libc defines (#251). The double fma is exact
    // in the product and rounds the sum once at 53 bits; rounding that to half is innocuous (53 >= 2*22 + 2),
    // so the result equals vfma.f16.
    return (_Float16)fma((double)z, (double)h_prev, (double)p);
        #endif
}

    #endif

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
/*
 * One block of up to 8 hidden units: sigmoid per lane (scalar), candidate tanh and the combine as vector ops.
 * The last partial block is predicated rather than scalar so every unit takes the same arithmetic (#315).
 */
__STATIC_INLINE void arm_nn_gru_block_f16(const cmsis_nn_gru_params_f16 *params,
                                          const float16_t *x,
                                          const float16_t *h_prev,
                                          const float16_t *reset_buf,
                                          float16_t *h_out,
                                          int32_t h,
                                          int32_t lanes)
{
    float16_t z_lane[8] = {0};
    float16_t cand_pre_lane[8] = {0};

    for (int32_t lane = 0; lane < lanes; lane++)
    {
        z_lane[lane] = arm_nn_gru_update_f16(params, x, h_prev, h + lane);
        cand_pre_lane[lane] = (float16_t)arm_nn_gru_candidate_pre_f16(params, x, h_prev, reset_buf, h + lane);
    }

    const float16x8_t vz = vld1q(z_lane);
    const float16x8_t vcand_pre = vld1q(cand_pre_lane);
    // The vector tanh maps a NaN lane to a finite value; the scalar leg propagates it (#251). Classify NaN in the
    // integer domain (arm_nn_clamp_propagate_nan_mve_f16 style, immune to -ffinite-math-only) before the tanh and
    // select any NaN afterwards: MVE arithmetic returns the default NaN whatever the source, and an all-ones splat
    // is one immediate move with no live range across the tanh.
    const mve_pred16_t nan_p = vcmphiq_n_u16(vshlq_n_u16(vreinterpretq_u16_f16(vcand_pre), 1), 0xF800);
    const float16x8_t vcand =
        vpselq(vreinterpretq_f16_u16(vdupq_n_u16(0xFFFF)), arm_nn_vtanh_lut_direct_mve_f16(vcand_pre), nan_p);
    const float16x8_t vp = vmulq(vsubq(vdupq_n_f16((float16_t)1.0f), vz), vcand);

    if (lanes == 8)
    {
        const float16x8_t vh_prev = h_prev ? vld1q(h_prev + h) : vdupq_n_f16((float16_t)0.0f);
        vst1q(h_out + h, vfmaq(vp, vz, vh_prev));
    }
    else
    {
        const mve_pred16_t p = vctp16q((uint32_t)lanes);
        const float16x8_t vh_prev = h_prev ? vld1q_z(h_prev + h, p) : vdupq_n_f16((float16_t)0.0f);
        vst1q_p(h_out + h, vfmaq(vp, vz, vh_prev), p);
    }
}
    #endif

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

    const cmsis_nn_gru_gate_f16 *rg = &params->reset_gate;
    const cmsis_nn_gru_gate_f16 *ng = &params->candidate_gate;

    float16_t *stage = NULL;
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
        const float16_t *x = data_in + (size_t)b * (size_t)batch_offset * (size_t)input_size;
        const float16_t *h_prev =
            hidden_in ? (hidden_in + (size_t)b * (size_t)batch_offset * (size_t)hidden_size) : NULL;
        float16_t *h_out = hidden_out + (size_t)b * (size_t)batch_offset * (size_t)hidden_size;

        // Pre-reset: the candidate recurrent term is Un.(r . h_prev). Stage r . h_prev once per batch so the
        // n^2 candidate loop reads a single operand; skipped when that term is dead (#251).
        const float16_t *reset_buf = NULL;
        if (!reset_after && h_prev && ng->hidden_weights)
        {
            for (int32_t h = 0; h < hidden_size; h++)
            {
                const _Float16 r_pre = (_Float16)(arm_nn_gru_input_proj_f16(rg, x, input_size, h) +
                                                  arm_nn_gru_hidden_proj_f16(rg, h_prev, hidden_size, h));
                stage[h] = (float16_t)((_Float16)arm_nn_sigmoid_scalar_f16((float16_t)r_pre) * (_Float16)h_prev[h]);
            }
            reset_buf = stage;
        }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t h = 0;
        for (; h + 8 <= hidden_size; h += 8)
        {
            arm_nn_gru_block_f16(params, x, h_prev, reset_buf, h_out, h, 8);
        }
        if (h < hidden_size)
        {
            arm_nn_gru_block_f16(params, x, h_prev, reset_buf, h_out, h, hidden_size - h);
        }
    #else
        for (int32_t h = 0; h < hidden_size; h++)
        {
            const _Float16 z = (_Float16)arm_nn_gru_update_f16(params, x, h_prev, h);
            const _Float16 cand = (_Float16)arm_nn_tanh_scalar_ref_f16(
                (float16_t)arm_nn_gru_candidate_pre_f16(params, x, h_prev, reset_buf, h));
            const _Float16 h_prev_h = h_prev ? (_Float16)h_prev[h] : (_Float16)0.0f;
            h_out[h] = (float16_t)arm_nn_gru_combine_f16(z, h_prev_h, cand);
        }
    #endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_F16 */

/** @} end of supportLSTM group */
