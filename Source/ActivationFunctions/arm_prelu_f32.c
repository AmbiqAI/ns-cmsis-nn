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
 * Title:        arm_prelu_f32.c
 * Description:  Parametric ReLU function for float32 data
 *
 * $Date:        12 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"

#if ARM_NN_ENABLE_F32

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Acti
 * @{
 */

// out[i] = input[i] >= 0 ? input[i] : input[i] * alpha[i]
static void arm_prelu_vec_f32(const float32_t *input, const float32_t *alpha, float32_t *output, int32_t flat_size)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t i = 0; i < flat_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(flat_size - i));
        const float32x4_t vx = vld1q_z(&input[i], p);
        const float32x4_t va = vld1q_z(&alpha[i], p);
        const float32x4_t vneg = vmulq(vx, va);
        const mve_pred16_t pos = vcmpgeq_n_f32(vx, 0.0f);
        vst1q_p(&output[i], vpselq(vx, vneg, pos), p);
    }
    #else
    for (int32_t i = 0; i < flat_size; ++i)
    {
        const float32_t x = input[i];
        output[i] = (x >= 0.0f) ? x : x * alpha[i];
    }
    #endif
}

// out[i] = input[i] >= 0 ? input[i] : input[i] * alpha_value
static void
arm_prelu_alpha_scalar_f32(const float32_t *input, float32_t alpha_value, float32_t *output, int32_t flat_size)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t va = vdupq_n_f32(alpha_value);
    for (int32_t i = 0; i < flat_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(flat_size - i));
        const float32x4_t vx = vld1q_z(&input[i], p);
        const float32x4_t vneg = vmulq(vx, va);
        const mve_pred16_t pos = vcmpgeq_n_f32(vx, 0.0f);
        vst1q_p(&output[i], vpselq(vx, vneg, pos), p);
    }
    #else
    for (int32_t i = 0; i < flat_size; ++i)
    {
        const float32_t x = input[i];
        output[i] = (x >= 0.0f) ? x : x * alpha_value;
    }
    #endif
}

/*
 * float32 PReLU. Alpha broadcasts onto the input (TensorFlow Lite
 * semantics): each alpha dimension must equal the matching input
 * dimension or 1.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_prelu_f32(const cmsis_nn_dims *input_dims,
                                  const float32_t *input,
                                  const cmsis_nn_dims *alpha_dims,
                                  const float32_t *alpha,
                                  const cmsis_nn_dims *output_dims,
                                  float32_t *output)
{
    if (!input_dims || !input || !alpha_dims || !alpha || !output_dims || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t in_n = input_dims->n, in_h = input_dims->h, in_w = input_dims->w, in_c = input_dims->c;
    const int32_t al_n = alpha_dims->n, al_h = alpha_dims->h, al_w = alpha_dims->w, al_c = alpha_dims->c;

    // PReLU is elementwise (input_dims == output_dims)
    if (in_n != output_dims->n || in_h != output_dims->h || in_w != output_dims->w || in_c != output_dims->c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Alpha broadcasts onto input only: every alpha dim is the input dim or 1
    if ((al_n != in_n && al_n != 1) || (al_h != in_h && al_h != 1) || (al_w != in_w && al_w != 1) ||
        (al_c != in_c && al_c != 1))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t flat_size = in_n * in_h * in_w * in_c;
    const int32_t alpha_size = al_n * al_h * al_w * al_c;

    // 1) Identical shapes: one flat pass
    if (al_n == in_n && al_h == in_h && al_w == in_w && al_c == in_c)
    {
        arm_prelu_vec_f32(input, alpha, output, flat_size);
        return ARM_CMSIS_NN_SUCCESS;
    }

    // 2) Scalar alpha
    if (alpha_size == 1)
    {
        arm_prelu_alpha_scalar_f32(input, alpha[0], output, flat_size);
        return ARM_CMSIS_NN_SUCCESS;
    }

    // 3) Per-channel alpha (1,1,1,C): reuse the same alpha row per pixel
    if (al_n == 1 && al_h == 1 && al_w == 1 && al_c == in_c)
    {
        const int32_t pixels = in_n * in_h * in_w;
        for (int32_t pix = 0; pix < pixels; pix++)
        {
            arm_prelu_vec_f32(&input[pix * in_c], alpha, &output[pix * in_c], in_c);
        }
        return ARM_CMSIS_NN_SUCCESS;
    }

    // 4) General broadcast: scalar walk with clamped alpha indices
    {
        int32_t out_idx = 0;
        for (int32_t n = 0; n < in_n; n++)
        {
            const int32_t a_n = (al_n == 1) ? 0 : n;
            for (int32_t h = 0; h < in_h; h++)
            {
                const int32_t a_h = (al_h == 1) ? 0 : h;
                for (int32_t w = 0; w < in_w; w++)
                {
                    const int32_t a_w = (al_w == 1) ? 0 : w;
                    const int32_t a_base = ((a_n * al_h + a_h) * al_w + a_w) * al_c;
                    for (int32_t c = 0; c < in_c; c++)
                    {
                        const float32_t x = input[out_idx];
                        const float32_t a = alpha[a_base + ((al_c == 1) ? 0 : c)];
                        output[out_idx] = (x >= 0.0f) ? x : x * a;
                        out_idx++;
                    }
                }
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Acti group
 */

#endif /* ARM_NN_ENABLE_F32 */
