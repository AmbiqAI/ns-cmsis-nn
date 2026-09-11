/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full
 * license text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_f16_group_ch_mult_1.c
 * Description:  float16 grouped convolution for filter channel one and
 *               one output channel per group
 *
 * $Date:        10 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_FORCEINLINE float32_t arm_convolve_group_ch_mult_1_widened_dot(float16x8_t lhs, float16x8_t rhs)
{
    const float32x4_t product_lo = vmulq(arm_nn_vcvtbq_f32_f16(lhs), arm_nn_vcvtbq_f32_f16(rhs));
    const float32x4_t product_hi = vmulq(arm_nn_vcvttq_f32_f16(lhs), arm_nn_vcvttq_f32_f16(rhs));
    return arm_nn_vec_reduce_add_f32(vaddq(product_lo, product_hi));
}

/* Match the established MVE convolution clamp: maxNum with the lower bound first makes NaN resolve to min. */
__STATIC_FORCEINLINE _Float16
arm_convolve_group_ch_mult_1_clamp_mve_compatible(_Float16 value, _Float16 activation_min, _Float16 activation_max)
{
    return arm_nn_min_f16h(arm_nn_max_f16h(value, activation_min), activation_max);
}

__STATIC_FORCEINLINE bool arm_convolve_group_ch_mult_1_gather_offset(int32_t kernel_y,
                                                                     int32_t kernel_x,
                                                                     int32_t dilation_y,
                                                                     int32_t dilation_x,
                                                                     int32_t input_x,
                                                                     int32_t input_ch,
                                                                     uint16_t *offset)
{
    const uint64_t limit = ARM_NN_MVE_F16_GATHER_OFFSET_MAX;
    uint64_t value = (uint64_t)(uint32_t)kernel_y * (uint32_t)dilation_y;
    if (value > limit / (uint32_t)input_x)
    {
        return false;
    }
    value *= (uint32_t)input_x;

    const uint64_t column = (uint64_t)(uint32_t)kernel_x * (uint32_t)dilation_x;
    if (column > limit || value > limit - column)
    {
        return false;
    }
    value += column;

    if (value > limit / (uint32_t)input_ch)
    {
        return false;
    }
    *offset = (uint16_t)(value * (uint32_t)input_ch);
    return true;
}

__STATIC_FORCEINLINE bool arm_convolve_group_ch_mult_1_pointer_steps(int32_t input_x,
                                                                     int32_t input_ch,
                                                                     int32_t output_x,
                                                                     int32_t output_y,
                                                                     int32_t stride_x,
                                                                     int32_t stride_y,
                                                                     size_t *column_step,
                                                                     size_t *row_step)
{
    *column_step = 0;
    *row_step = 0;

    if (output_x > 1)
    {
        const uint64_t step = (uint64_t)(uint32_t)input_ch * (uint32_t)stride_x;
        if (step > SIZE_MAX)
        {
            return false;
        }
        *column_step = (size_t)step;
    }

    if (output_y > 1)
    {
        const uint64_t next_row = (uint64_t)(uint32_t)stride_y * (uint32_t)input_x;
        const uint64_t current_column = (uint64_t)(uint32_t)(output_x - 1) * (uint32_t)stride_x;
        if (next_row < current_column)
        {
            return false;
        }
        const uint64_t spatial_step = next_row - current_column;
        if (spatial_step > SIZE_MAX / (uint32_t)input_ch)
        {
            return false;
        }
        *row_step = (size_t)(spatial_step * (uint32_t)input_ch);
    }
    return true;
}
    #endif

arm_cmsis_nn_status arm_convolve_f16_group_ch_mult_1(const cmsis_nn_context *ctx,
                                                     const cmsis_nn_conv_params_f16 *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const float16_t *input_data,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const float16_t *filter_data,
                                                     const cmsis_nn_dims *bias_dims,
                                                     const float16_t *bias_data,
                                                     const cmsis_nn_dims *output_dims,
                                                     float16_t *output_data)
{
    (void)ctx;
    (void)bias_dims;

    if (!conv_params || !input_dims || !input_data || !filter_dims || !filter_data || !output_dims || !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t input_batches = input_dims->n;
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t input_ch = input_dims->c;
    const int32_t output_batches = output_dims->n;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_ch = output_dims->c;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y = conv_params->padding.h;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t stride_y = conv_params->stride.h;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t dilation_y = conv_params->dilation.h;
    const int32_t dilation_x = conv_params->dilation.w;
    const _Float16 activation_min = (_Float16)conv_params->activation.min;
    const _Float16 activation_max = (_Float16)conv_params->activation.max;

    if (filter_dims->c != 1 || filter_dims->n != output_ch || input_ch <= 0 || input_ch != output_ch ||
        input_batches < 0 || input_x <= 0 || input_y <= 0 || kernel_x <= 0 || kernel_y <= 0 || output_batches < 0 ||
        output_x < 0 || output_y < 0 || stride_x <= 0 || stride_y <= 0 || dilation_x <= 0 || dilation_y <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (conv_params->weight_format != ARM_NN_WEIGHT_FORMAT_STANDARD)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    if (output_x == 0 || output_y == 0 || input_batches == 0 || output_batches == 0)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }

    const int64_t rhs_cols_64 = (int64_t)kernel_y * kernel_x;
    if (rhs_cols_64 > INT32_MAX)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t rhs_cols = (int32_t)rhs_cols_64;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const bool output_fits = (int64_t)(output_x - 1) * stride_x + (int64_t)(kernel_x - 1) * dilation_x < input_x &&
        (int64_t)(output_y - 1) * stride_y + (int64_t)(kernel_y - 1) * dilation_y < input_y;

    if (rhs_cols <= 16 && pad_y == 0 && pad_x == 0 && output_fits)
    {
        const int32_t rhs_cols_0 = (rhs_cols < 8) ? rhs_cols : 8;
        const int32_t rhs_cols_1 = rhs_cols - rhs_cols_0;
        uint16x8_t offset_src_0 = vdupq_n_u16(0);
        uint16x8_t offset_src_1 = vdupq_n_u16(0);
        bool offsets_fit = true;

        for (int32_t ky = 0; ky < kernel_y && offsets_fit; ++ky)
        {
            for (int32_t kx = 0; kx < kernel_x; ++kx)
            {
                const int32_t idx = ky * kernel_x + kx;
                uint16_t offset;
                if (!arm_convolve_group_ch_mult_1_gather_offset(
                        ky, kx, dilation_y, dilation_x, input_x, input_ch, &offset))
                {
                    offsets_fit = false;
                    break;
                }
                if (idx < 8)
                {
                    offset_src_0[idx] = offset;
                }
                else
                {
                    offset_src_1[idx - 8] = offset;
                }
            }
        }

        if (!offsets_fit)
        {
            goto scalar_fallback;
        }

        size_t column_step;
        size_t row_step;
        if (!arm_convolve_group_ch_mult_1_pointer_steps(
                input_x, input_ch, output_x, output_y, stride_x, stride_y, &column_step, &row_step))
        {
            goto scalar_fallback;
        }
        const mve_pred16_t p0 = vctp16q((uint32_t)rhs_cols_0);
        const mve_pred16_t p1 = vctp16q((uint32_t)rhs_cols_1);

        for (int32_t b = 0; b < input_batches; ++b)
        {
            const float16_t *filter_ptr = filter_data;
            for (int32_t c = 0; c < output_ch; ++c)
            {
                const float16_t *input_ptr = input_data + c;
                float16_t *out_c = output_data + c;
                const float16x8_t weight_0 = vld1q_z(filter_ptr, p0);
                float16x8_t weight_1 = vdupq_n_f16((float16_t)0.0f);
                if (rhs_cols_1 > 0)
                {
                    weight_1 = vld1q_z(filter_ptr + 8, p1);
                }
                filter_ptr += rhs_cols;
                const float32_t bias = bias_data ? (float32_t)bias_data[c] : 0.0f;

                for (int32_t out_y = 0; out_y < output_y; ++out_y)
                {
                    for (int32_t out_x = 0; out_x < output_x - 1; ++out_x)
                    {
                        const float16x8_t input_0 = vldrhq_gather_shifted_offset_z(input_ptr, offset_src_0, p0);
                        float32_t acc32 = arm_convolve_group_ch_mult_1_widened_dot(weight_0, input_0);
                        if (rhs_cols_1 > 0)
                        {
                            const float16x8_t input_1 = vldrhq_gather_shifted_offset_z(input_ptr, offset_src_1, p1);
                            acc32 += arm_convolve_group_ch_mult_1_widened_dot(weight_1, input_1);
                        }
                        _Float16 acc = (_Float16)(acc32 + bias);
                        *out_c = (float16_t)arm_convolve_group_ch_mult_1_clamp_mve_compatible(
                            acc, activation_min, activation_max);
                        out_c += output_ch;
                        input_ptr += column_step;
                    }

                    const float16x8_t input_0 = vldrhq_gather_shifted_offset_z(input_ptr, offset_src_0, p0);
                    float32_t acc32 = arm_convolve_group_ch_mult_1_widened_dot(weight_0, input_0);
                    if (rhs_cols_1 > 0)
                    {
                        const float16x8_t input_1 = vldrhq_gather_shifted_offset_z(input_ptr, offset_src_1, p1);
                        acc32 += arm_convolve_group_ch_mult_1_widened_dot(weight_1, input_1);
                    }
                    _Float16 acc = (_Float16)(acc32 + bias);
                    *out_c = (float16_t)arm_convolve_group_ch_mult_1_clamp_mve_compatible(
                        acc, activation_min, activation_max);
                    if (out_y + 1 < output_y)
                    {
                        out_c += output_ch;
                        input_ptr += row_step;
                    }
                }
            }
            input_data += (size_t)input_y * input_x * input_ch;
            output_data += (size_t)output_y * output_x * output_ch;
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
scalar_fallback:
    #endif

    for (int32_t b = 0; b < input_batches; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_y * input_x * input_ch;
        float16_t *output_b = output_data + (size_t)b * output_y * output_x * output_ch;

        for (int32_t out_y = 0; out_y < output_y; ++out_y)
        {
            const int64_t base_y = (int64_t)out_y * stride_y - pad_y;
            for (int32_t out_x = 0; out_x < output_x; ++out_x)
            {
                const int64_t base_x = (int64_t)out_x * stride_x - pad_x;
                int64_t kernel_y_start;
                int64_t kernel_y_end;
                int64_t kernel_x_start;
                int64_t kernel_x_end;

                if (dilation_y > 1)
                {
                    kernel_y_start = (-base_y + dilation_y - 1) / dilation_y;
                    kernel_y_end = (input_y - base_y + dilation_y - 1) / dilation_y;
                }
                else
                {
                    kernel_y_start = -base_y;
                    kernel_y_end = input_y - base_y;
                }
                if (dilation_x > 1)
                {
                    kernel_x_start = (-base_x + dilation_x - 1) / dilation_x;
                    kernel_x_end = (input_x - base_x + dilation_x - 1) / dilation_x;
                }
                else
                {
                    kernel_x_start = -base_x;
                    kernel_x_end = input_x - base_x;
                }
                if (kernel_y_start < 0)
                {
                    kernel_y_start = 0;
                }
                if (kernel_y_end > kernel_y)
                {
                    kernel_y_end = kernel_y;
                }
                if (kernel_x_start < 0)
                {
                    kernel_x_start = 0;
                }
                if (kernel_x_end > kernel_x)
                {
                    kernel_x_end = kernel_x;
                }

                for (int32_t c = 0; c < output_ch; ++c)
                {
                    float32_t acc32 = bias_data ? (float32_t)bias_data[c] : 0.0f;
                    const float16_t *filter_c = filter_data + (size_t)c * rhs_cols;

                    for (int64_t ky = kernel_y_start; ky < kernel_y_end; ++ky)
                    {
                        const int64_t in_y = base_y + ky * dilation_y;
                        for (int64_t kx = kernel_x_start; kx < kernel_x_end; ++kx)
                        {
                            const int64_t in_x = base_x + kx * dilation_x;
                            const size_t input_index = ((size_t)in_y * input_x + in_x) * input_ch + c;
                            acc32 += (float32_t)input_b[input_index] * (float32_t)filter_c[ky * kernel_x + kx];
                        }
                    }

                    const size_t output_index = ((size_t)out_y * output_x + out_x) * output_ch + c;
                    const _Float16 acc = (_Float16)acc32;
                    output_b[output_index] = (float16_t)arm_nn_clamp_f16h(acc, activation_max, activation_min);
                }
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */

#endif /* ARM_NN_ENABLE_F16 */
