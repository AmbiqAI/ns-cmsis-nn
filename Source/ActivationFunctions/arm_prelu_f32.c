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

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
/*
 * MVE f32 compares and multiplies treat a subnormal as zero whatever FPSCR.FZ holds, so a lane with a negative,
 * nonzero x is redone on the scalar FPU when x is subnormal (the compare sees -0), or when alpha is nonzero and it
 * or the product is zero or subnormal. The other lanes are exact as they are; a zero alpha with a normal x gives an
 * exact zero. The lanes are tested as integers loaded as integers: from a float load, LLVM turns the bit tests into
 * the same flushing compares. Refs #853, #873.
 */
static inline mve_pred16_t arm_prelu_redo_f32(const uint32x4_t x, const uint32x4_t a, const float32x4_t product)
{
    const uint32x4_t exponent = vdupq_n_u32(0x7F800000u);
    const mve_pred16_t tiny_alpha_or_product =
        vcmpeqq_n_u32(vandq(a, exponent), 0) | vcmpeqq_n_u32(vandq(vreinterpretq_u32_f32(product), exponent), 0);
    const mve_pred16_t flushed =
        vcmpeqq_n_u32(vandq(x, exponent), 0) | (tiny_alpha_or_product & vcmpneq_n_u32(vshlq_n_u32(a, 1), 0));
    return flushed & vcmpltq_n_s32(vreinterpretq_s32_u32(x), 0) & vcmpneq_n_u32(vshlq_n_u32(x, 1), 0);
}

// The scalar FPU's product, kept out of reach of the vectorizer
static inline float32_t arm_prelu_scalar_mul_f32(const float32_t x, const float32_t a)
{
    float32_t r;
    __asm("vmul.f32 %0, %1, %2" : "=t"(r) : "t"(x), "t"(a));
    return r;
}
    #endif

// out[i] = input[i] >= 0 ? input[i] : input[i] * alpha[i]
static void arm_prelu_vec_f32(const float32_t *input, const float32_t *alpha, float32_t *output, int32_t flat_size)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t i = 0; i < flat_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(flat_size - i));
        const uint32x4_t xb = vld1q_z((const uint32_t *)&input[i], p);
        const uint32x4_t ab = vld1q_z((const uint32_t *)&alpha[i], p);
        const float32x4_t vx = vreinterpretq_f32_u32(xb);
        const float32x4_t vneg = vmulq(vx, vreinterpretq_f32_u32(ab));
        const mve_pred16_t pos = vcmpgeq_n_f32(vx, 0.0f);
        vst1q_p(&output[i], vpselq(vx, vneg, pos), p);
        const mve_pred16_t redo = arm_prelu_redo_f32(xb, ab, vneg) & p;
        if (redo)
        {
            // From the loaded lanes: output may be input or alpha, already overwritten by the store above
            float32_t xs[4], as[4];
            vst1q(xs, vx);
            vst1q(as, vreinterpretq_f32_u32(ab));
            for (int32_t j = 0; j < 4; ++j)
            {
                if (redo & (1u << (4 * j)))
                {
                    output[i + j] = arm_prelu_scalar_mul_f32(xs[j], as[j]);
                }
            }
        }
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
    uint32_t alpha_bits;
    memcpy(&alpha_bits, &alpha_value, sizeof(alpha_bits));
    const uint32x4_t ab = vdupq_n_u32(alpha_bits);
    const float32x4_t va = vreinterpretq_f32_u32(ab);
    for (int32_t i = 0; i < flat_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(flat_size - i));
        const uint32x4_t xb = vld1q_z((const uint32_t *)&input[i], p);
        const float32x4_t vx = vreinterpretq_f32_u32(xb);
        const float32x4_t vneg = vmulq(vx, va);
        const mve_pred16_t pos = vcmpgeq_n_f32(vx, 0.0f);
        vst1q_p(&output[i], vpselq(vx, vneg, pos), p);
        const mve_pred16_t redo = arm_prelu_redo_f32(xb, ab, vneg) & p;
        if (redo)
        {
            // From the loaded lanes: output may be input, already overwritten by the store above
            float32_t xs[4];
            vst1q(xs, vx);
            for (int32_t j = 0; j < 4; ++j)
            {
                if (redo & (1u << (4 * j)))
                {
                    output[i + j] = arm_prelu_scalar_mul_f32(xs[j], alpha_value);
                }
            }
        }
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

    // 4) General broadcast: one alpha row (al_c == in_c) or value (al_c == 1) per pixel, clamped alpha indices
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
                    if (al_c == 1)
                    {
                        arm_prelu_alpha_scalar_f32(&input[out_idx], alpha[a_base], &output[out_idx], in_c);
                    }
                    else
                    {
                        arm_prelu_vec_f32(&input[out_idx], &alpha[a_base], &output[out_idx], in_c);
                    }
                    out_idx += in_c;
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
