/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_svdf_f32.c
 * Description:  SVDF layer (float32)
 *
 * $Date:        23 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 * @ingroup Public
 */

/**
 * @addtogroup SVDF
 * @{
 */

__STATIC_INLINE float32_t clamp_and_activate(float32_t x, const cmsis_nn_activation_f32 *act)
{
    return CLAMP(x, act->max, act->min);
}

__STATIC_INLINE float32_t arm_nn_svdf_dot_f32(const float32_t *lhs, const float32_t *rhs, int32_t count)
{
    float32_t sum = 0.0f;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    float32x4_t v_acc = vdupq_n_f32(0.0f);
    int32_t i = 0;
    for (; i + 4 <= count; i += 4)
    {
        v_acc = vfmaq(v_acc, vld1q(lhs + i), vld1q(rhs + i));
    }
    sum = arm_nn_vec_reduce_add_f32(v_acc);
    for (; i < count; ++i)
    {
        sum += lhs[i] * rhs[i];
    }
    #else
    for (int32_t i = 0; i < count; ++i)
    {
        sum += lhs[i] * rhs[i];
    }
    #endif

    return sum;
}

arm_cmsis_nn_status arm_svdf_f32(const cmsis_nn_context *ctx,
                                 const cmsis_nn_context *input_ctx,
                                 const cmsis_nn_context *output_ctx,
                                 const cmsis_nn_svdf_params_f32 *svdf_params,
                                 const cmsis_nn_dims *input_dims,
                                 const float32_t *input_data,
                                 const cmsis_nn_dims *state_dims,
                                 float32_t *state_data,
                                 const cmsis_nn_dims *weights_feature_dims,
                                 const float32_t *weights_feature_data,
                                 const cmsis_nn_dims *weights_time_dims,
                                 const float32_t *weights_time_data,
                                 const cmsis_nn_dims *bias_dims,
                                 const float32_t *bias_data,
                                 const cmsis_nn_dims *output_dims,
                                 float32_t *output_data)
{
    (void)ctx;
    (void)state_dims;
    (void)bias_dims;
    (void)output_dims;

    if (!svdf_params || !input_dims || !weights_feature_dims || !weights_time_dims || !input_data || !state_data ||
        !weights_feature_data || !weights_time_data || !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = svdf_params->rank;
    const int32_t input_batches = input_dims->n;
    const int32_t input_height = input_dims->h;
    const int32_t feature_batches = weights_feature_dims->n;
    const int32_t time_batches = weights_time_dims->h;
    const int32_t unit_count = feature_batches / rank;

    float32_t *buffer_a = (float32_t *)(input_ctx ? input_ctx->buf : NULL);
    float32_t *buffer_b = (float32_t *)(output_ctx ? output_ctx->buf : NULL);
    if (!buffer_a || !buffer_b)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Opt-in staging size checks, entry only. A caller that populates ctx->size has an undersized buffer
       diagnosed here instead of overflowed further down; leaving size at 0 opts out, which is what TFLite Micro
       and derivatives do (they allocate from the arena but never propagate the figure). The size query is only
       reached when the caller opted in, and nothing below this point inspects ctx->size, so no loop pays for it. */
    if (input_ctx->size != 0)
    {
        const int32_t required_bytes = arm_svdf_f32_input_ctx_get_buffer_size(input_dims, weights_feature_dims);
        if ((required_bytes < 0) || (input_ctx->size < required_bytes))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    if (output_ctx->size != 0)
    {
        const int32_t required_bytes =
            arm_svdf_f32_output_ctx_get_buffer_size(svdf_params, input_dims, weights_feature_dims);
        if ((required_bytes < 0) || (output_ctx->size < required_bytes))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        float32_t *state_b = state_data + (i_batch * feature_batches * time_batches);
        for (int32_t i_feat = 0; i_feat < feature_batches; i_feat++)
        {
            float32_t *state_f = state_b + i_feat * time_batches;
            if (time_batches > 1)
            {
                for (int32_t t = 0; t < time_batches - 1; ++t)
                {
                    state_f[t] = state_f[t + 1];
                }
            }
        }
    }

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        const float32_t *input = input_data + i_batch * input_height;
        float32_t *state_b = state_data + (i_batch * feature_batches * time_batches);
        for (int32_t i_feat = 0; i_feat < feature_batches; i_feat++)
        {
            const float32_t *w_feat = weights_feature_data + i_feat * input_height;
            float32_t acc = arm_nn_svdf_dot_f32(input, w_feat, input_height);
            acc = clamp_and_activate(acc, &svdf_params->input_activation);
            state_b[i_feat * time_batches + (time_batches - 1)] = acc;
        }
    }

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        float32_t *state_b = state_data + (i_batch * feature_batches * time_batches);
        float32_t *out_a = buffer_a + i_batch * feature_batches;
        for (int32_t i_feat = 0; i_feat < feature_batches; i_feat++)
        {
            const float32_t *w_time = weights_time_data + i_feat * time_batches;
            const float32_t *s_time = state_b + i_feat * time_batches;
            out_a[i_feat] = arm_nn_svdf_dot_f32(w_time, s_time, time_batches);
        }
    }

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        float32_t *out_b = buffer_b + i_batch * unit_count;
        float32_t *ptr_a = buffer_a + i_batch * feature_batches;
        if (bias_data)
        {
            if (unit_count == feature_batches)
            {
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                for (int32_t j = 0; j < feature_batches; j += 4)
                {
                    const mve_pred16_t p = vctp32q((uint32_t)(feature_batches - j));
                    const float32x4_t v_a = vld1q_z(ptr_a + j, p);
                    const float32x4_t v_b = vld1q_z(bias_data + j, p);
                    vst1q_p(out_b + j, vaddq(v_a, v_b), p);
                }
    #else
                for (int32_t j = 0; j < feature_batches; j++)
                {
                    out_b[j] = ptr_a[j] + bias_data[j];
                }
    #endif
            }
            else
            {
                for (int32_t i = 0; i < unit_count; i++)
                {
                    float32_t sum = bias_data[i];
                    for (int32_t j = 0; j < rank; j++)
                    {
                        sum += *ptr_a++;
                    }
                    out_b[i] = sum;
                }
            }
        }
        else
        {
            for (int32_t i = 0; i < unit_count; i++)
            {
                float32_t sum = 0.0f;
                for (int32_t j = 0; j < rank; j++)
                {
                    sum += *ptr_a++;
                }
                out_b[i] = sum;
            }
        }
    }

    const int32_t output_count = input_batches * unit_count;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t v_act_min = vdupq_n_f32(svdf_params->output_activation.min);
    const float32x4_t v_act_max = vdupq_n_f32(svdf_params->output_activation.max);
    int32_t i = 0;
    for (; i < output_count; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(output_count - i));
        float32x4_t v = vld1q_z(buffer_b + i, p);
        v = arm_nn_clamp_mve_f32(v, v_act_min, v_act_max);
        vst1q_p(output_data + i, v, p);
    }
    #else
    for (int32_t i = 0; i < output_count; i++)
    {
        float32_t v = buffer_b[i];
        v = clamp_and_activate(v, &svdf_params->output_activation);
        output_data[i] = v;
    }
    #endif

    return ARM_CMSIS_NN_SUCCESS;
}

/*
 * Staging-buffer queries for arm_svdf_f32(). These live beside the kernel rather than in
 * arm_get_buffer_size_f32.c so that the numbers and the stores they describe cannot drift apart.
 *
 * Sentinel note: these follow the SVDF family's -1-on-invalid convention (arm_nn_size_mul(), as used by
 * arm_svdf_s8_get_buffer_size()), NOT the 0-on-out-of-range convention of the float convolution sizers in
 * Source/NNSupportFunctions/arm_get_buffer_size_f32.c.
 *
 * The reason is the guard at the top of arm_svdf_f32(): ctx->size == 0 is the opt-out signal for that guard. A
 * 0-on-overflow sizer would hand blind codegen doing `ctx.buf = alloc(sz); ctx.size = sz;` a non-NULL zero-byte
 * allocation paired with size 0, the guard would opt out, and the kernel would write the full overflowing extent
 * into it. With -1, alloc((size_t)-1) fails, buf comes back NULL, and the existing NULL check catches it. -1
 * fails safe here; 0 does not.
 *
 * Note that 0 is still a legitimate *return* for a degenerate shape - input_dims->n == 0, or a rank larger than
 * weights_feature_dims->n, which truncates unit_count to 0. It is only never used to report an out-of-range one.
 */

int32_t arm_svdf_f32_input_ctx_get_buffer_size(const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *weights_feature_dims)
{
    if (!input_dims || !weights_feature_dims)
    {
        return -1;
    }

    /* One float32_t per (input batch, feature batch): the time-weight block above stores exactly once per
       iteration of the i_batch / i_feat loops through `out_a = buffer_a + i_batch * feature_batches`, so the
       highest element written is input_dims->n * weights_feature_dims->n - 1. Folded one factor at a time so the
       accumulator stays bounded; see arm_nn_size_mul(). */
    int64_t required_bytes = arm_nn_size_mul(1, input_dims->n);
    required_bytes = arm_nn_size_mul(required_bytes, weights_feature_dims->n);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(float32_t));

    return (int32_t)required_bytes;
}

int32_t arm_svdf_f32_output_ctx_get_buffer_size(const cmsis_nn_svdf_params_f32 *svdf_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *weights_feature_dims)
{
    if (!svdf_params || !input_dims || !weights_feature_dims)
    {
        return -1;
    }

    /* rank is a divisor, so it is validated before use rather than folded through arm_nn_size_mul(). */
    if ((svdf_params->rank <= 0) || (weights_feature_dims->n < 0))
    {
        return -1;
    }

    /* One float32_t per (input batch, output unit). Every branch of the bias/no-bias block above writes through
       `out_b = buffer_b + i_batch * unit_count` for indices [0, unit_count); the branch that indexes by
       feature_batches instead is guarded by `unit_count == feature_batches` and so covers the same extent. The
       MVE clamp tail loads buffer_b under a vctp32q() predicate and therefore adds no round-up. The division
       truncates, exactly as the kernel's `feature_batches / rank` does. */
    const int32_t unit_count = weights_feature_dims->n / svdf_params->rank;

    int64_t required_bytes = arm_nn_size_mul(1, input_dims->n);
    required_bytes = arm_nn_size_mul(required_bytes, unit_count);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(float32_t));

    return (int32_t)required_bytes;
}

/**
 * @} end of SVDF group
 */

#endif /* ARM_NN_ENABLE_F32 */
