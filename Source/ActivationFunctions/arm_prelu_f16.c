/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_prelu_f16.c
 * Description:  Parametric ReLU for float16 tensors
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup Acti
 * @{
 */

static void prelu_elementwise_f16(const float16_t *input, const float16_t *alpha, float16_t *output, int32_t size)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t index = 0; index < size; index += 8)
    {
        const mve_pred16_t tail_predicate = vctp16q((uint32_t)(size - index));
        const float16x8_t input_vector = vld1q_z(input + index, tail_predicate);
        const float16x8_t alpha_vector = vld1q_z(alpha + index, tail_predicate);
        const mve_pred16_t negative_predicate = vcmpltq(input_vector, (float16_t)0.0f);
        const float16x8_t result = vpselq(vmulq(input_vector, alpha_vector), input_vector, negative_predicate);
        vst1q_p(output + index, result, tail_predicate);
    }
    #else
    for (int32_t index = 0; index < size; ++index)
    {
        const _Float16 input_value = (_Float16)input[index];
        output[index] = input_value < (_Float16)0.0f ? (float16_t)(input_value * (_Float16)alpha[index]) : input[index];
    }
    #endif
}

static void prelu_scalar_alpha_f16(const float16_t *input, const float16_t alpha, float16_t *output, int32_t size)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t alpha_vector = vdupq_n_f16(alpha);
    for (int32_t index = 0; index < size; index += 8)
    {
        const mve_pred16_t tail_predicate = vctp16q((uint32_t)(size - index));
        const float16x8_t input_vector = vld1q_z(input + index, tail_predicate);
        const mve_pred16_t negative_predicate = vcmpltq(input_vector, (float16_t)0.0f);
        const float16x8_t result = vpselq(vmulq(input_vector, alpha_vector), input_vector, negative_predicate);
        vst1q_p(output + index, result, tail_predicate);
    }
    #else
    const _Float16 alpha_value = (_Float16)alpha;
    for (int32_t index = 0; index < size; ++index)
    {
        const _Float16 input_value = (_Float16)input[index];
        output[index] = input_value < (_Float16)0.0f ? (float16_t)(input_value * alpha_value) : input[index];
    }
    #endif
}

/* Refer header file for details. */
arm_cmsis_nn_status arm_prelu_f16(const cmsis_nn_dims *input_dims,
                                  const float16_t *input,
                                  const cmsis_nn_dims *alpha_dims,
                                  const float16_t *alpha,
                                  const cmsis_nn_dims *output_dims,
                                  float16_t *output)
{
    if (!input_dims || !input || !alpha_dims || !alpha || !output_dims || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (input_dims->n <= 0 || input_dims->h <= 0 || input_dims->w <= 0 || input_dims->c <= 0 ||
        input_dims->n != output_dims->n || input_dims->h != output_dims->h || input_dims->w != output_dims->w ||
        input_dims->c != output_dims->c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if ((alpha_dims->n != 1 && alpha_dims->n != input_dims->n) ||
        (alpha_dims->h != 1 && alpha_dims->h != input_dims->h) ||
        (alpha_dims->w != 1 && alpha_dims->w != input_dims->w) ||
        (alpha_dims->c != 1 && alpha_dims->c != input_dims->c))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t input_size = input_dims->n * input_dims->h * input_dims->w * input_dims->c;
    const int32_t alpha_size = alpha_dims->n * alpha_dims->h * alpha_dims->w * alpha_dims->c;

    if (alpha_size == 1)
    {
        prelu_scalar_alpha_f16(input, alpha[0], output, input_size);
        return ARM_CMSIS_NN_SUCCESS;
    }

    if (alpha_dims->n == input_dims->n && alpha_dims->h == input_dims->h && alpha_dims->w == input_dims->w &&
        alpha_dims->c == input_dims->c)
    {
        prelu_elementwise_f16(input, alpha, output, input_size);
        return ARM_CMSIS_NN_SUCCESS;
    }

    if (alpha_dims->n == 1 && alpha_dims->h == 1 && alpha_dims->w == 1 && alpha_dims->c == input_dims->c)
    {
        const int32_t outer_size = input_dims->n * input_dims->h * input_dims->w;
        for (int32_t outer_index = 0; outer_index < outer_size; ++outer_index)
        {
            prelu_elementwise_f16(input, alpha, output, input_dims->c);
            input += input_dims->c;
            output += input_dims->c;
        }
        return ARM_CMSIS_NN_SUCCESS;
    }

    int32_t output_index = 0;
    for (int32_t batch = 0; batch < input_dims->n; ++batch)
    {
        const int32_t alpha_batch = alpha_dims->n == 1 ? 0 : batch;
        for (int32_t height = 0; height < input_dims->h; ++height)
        {
            const int32_t alpha_height = alpha_dims->h == 1 ? 0 : height;
            for (int32_t width = 0; width < input_dims->w; ++width)
            {
                const int32_t alpha_width = alpha_dims->w == 1 ? 0 : width;
                for (int32_t channel = 0; channel < input_dims->c; ++channel)
                {
                    const int32_t alpha_channel = alpha_dims->c == 1 ? 0 : channel;
                    const int32_t alpha_index =
                        ((alpha_batch * alpha_dims->h + alpha_height) * alpha_dims->w + alpha_width) * alpha_dims->c +
                        alpha_channel;
                    const _Float16 input_value = (_Float16)input[output_index];
                    output[output_index] = input_value < (_Float16)0.0f
                        ? (float16_t)(input_value * (_Float16)alpha[alpha_index])
                        : input[output_index];
                    ++output_index;
                }
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Acti group
 */

#endif /* ARM_NN_ENABLE_F16 */
