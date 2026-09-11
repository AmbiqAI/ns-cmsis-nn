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
 * Title:        arm_convolve_f16_fast_small_kernel.c
 * Description:  float16 convolution for kernels whose flattened receptive
 *               field (kernel_h * kernel_w * kernel_ch) fits in a single MVE
 *               float16 vector. Mirrors arm_convolve_s16_fast_small_kernel.
 *
 * $Date:        10 August 2026
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
__STATIC_FORCEINLINE float32_t arm_convolve_f16_widened_dot(float16x8_t lhs, float16x8_t rhs)
{
    const float32x4_t product_lo = vmulq(arm_nn_vcvtbq_f32_f16(lhs), arm_nn_vcvtbq_f32_f16(rhs));
    const float32x4_t product_hi = vmulq(arm_nn_vcvttq_f32_f16(lhs), arm_nn_vcvttq_f32_f16(rhs));
    return arm_nn_vec_reduce_add_f32(vaddq(product_lo, product_hi));
}

/* Match the established MVE convolution clamp: maxNum with the lower bound first makes NaN resolve to min. */
__STATIC_FORCEINLINE _Float16
arm_convolve_f16_clamp_mve_compatible(_Float16 value, _Float16 activation_min, _Float16 activation_max)
{
    return arm_nn_min_f16h(arm_nn_max_f16h(value, activation_min), activation_max);
}

__STATIC_FORCEINLINE bool arm_convolve_f16_gather_offset(int32_t kernel_y,
                                                         int32_t kernel_x,
                                                         int32_t dilation_y,
                                                         int32_t dilation_x,
                                                         int32_t input_x,
                                                         int32_t input_ch,
                                                         int32_t input_channel,
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
    value *= (uint32_t)input_ch;
    if ((uint32_t)input_channel > limit - value)
    {
        return false;
    }

    *offset = (uint16_t)(value + (uint32_t)input_channel);
    return true;
}

__STATIC_FORCEINLINE bool arm_convolve_f16_pointer_steps(int32_t input_x,
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

/*
 * Fast float16 convolution for small kernels. Optimal when
 * rhs_cols = kernel_h * kernel_w * kernel_ch <= 8 and padding is zero.
 * Supports grouped convolution (input_ch = groups * kernel_ch).
 */
arm_cmsis_nn_status arm_convolve_f16_fast_small_kernel(const cmsis_nn_context *ctx,
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

    const int32_t input_ch = input_dims->c;
    const int32_t kernel_ch = filter_dims->c;
    const int32_t output_ch = output_dims->c;

    if (kernel_ch <= 0 || input_ch <= 0 || input_ch % kernel_ch != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t groups = input_ch / kernel_ch;
    if (output_ch <= 0 || filter_dims->n != output_ch || output_ch % groups != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (input_dims->n < 0 || input_dims->w < 0 || input_dims->h < 0 || filter_dims->w < 0 || filter_dims->h < 0 ||
        output_dims->n < 0 || output_dims->w < 0 || output_dims->h < 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (conv_params->weight_format != ARM_NN_WEIGHT_FORMAT_STANDARD)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const int32_t input_batches = input_dims->n;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;

    const int32_t dilation_x = conv_params->dilation.w;
    const int32_t dilation_y = conv_params->dilation.h;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;

    const _Float16 act_min = (_Float16)conv_params->activation.min;
    const _Float16 act_max = (_Float16)conv_params->activation.max;

    const int32_t output_ch_per_group = output_ch / groups;

    /* Only handle the shapes this kernel is specialized for. */
    if (kernel_x <= 0 || kernel_y <= 0 || kernel_x > 8 || kernel_y > 8 || kernel_ch > 8 ||
        conv_params->padding.w != 0 || conv_params->padding.h != 0 || input_x <= 0 || input_y <= 0 ||
        conv_params->stride.w <= 0 || conv_params->stride.h <= 0 || conv_params->dilation.w <= 0 ||
        conv_params->dilation.h <= 0)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    const int32_t rhs_cols = kernel_ch * kernel_y * kernel_x;
    if (rhs_cols > 8)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    if (input_batches == 0 || output_dims->n == 0 || output_x == 0 || output_y == 0)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
    if ((int64_t)(output_x - 1) * stride_x + (int64_t)(kernel_x - 1) * dilation_x >= input_x ||
        (int64_t)(output_y - 1) * stride_y + (int64_t)(kernel_y - 1) * dilation_y >= input_y)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    /* Element-scaled gather uses 16-bit per-lane offsets; bail out if the window does not fit. */
    uint16x8_t offset_src = vdupq_n_u16(0);
    for (int32_t i = 0; i < kernel_y; i++)
    {
        for (int32_t j = 0; j < kernel_x; j++)
        {
            const int32_t idx = i * kernel_x + j;
            for (int32_t c = 0; c < kernel_ch; c++)
            {
                uint16_t offset;
                if (!arm_convolve_f16_gather_offset(i, j, dilation_y, dilation_x, input_x, input_ch, c, &offset))
                {
                    return ARM_CMSIS_NN_NO_IMPL_ERROR;
                }
                offset_src[idx * kernel_ch + c] = offset;
            }
        }
    }

    size_t column_step;
    size_t row_step;
    if (!arm_convolve_f16_pointer_steps(
            input_x, input_ch, output_x, output_y, stride_x, stride_y, &column_step, &row_step))
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    const mve_pred16_t p = vctp16q((uint32_t)rhs_cols);

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        const float16_t *filter_data_ptr = filter_data;

        for (int32_t i_group = 0; i_group < groups; i_group++)
        {
            for (int32_t c = 0; c < output_ch_per_group; c++)
            {
                const float16_t *input_data_pr = input_data + i_group * kernel_ch;
                float16_t *out_c = output_data + i_group * output_ch_per_group + c;
                const float16x8_t weight = vld1q_z(filter_data_ptr, p);
                filter_data_ptr += rhs_cols;
                const float32_t bias_val = bias_data ? (float32_t)bias_data[i_group * output_ch_per_group + c] : 0.0f;

                for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
                {
                    for (int32_t i_out_x = 0; i_out_x < output_x - 1; i_out_x++)
                    {
                        const float16x8_t in = vldrhq_gather_shifted_offset_z(input_data_pr, offset_src, p);
                        input_data_pr += column_step;
                        const float32_t acc32 = arm_convolve_f16_widened_dot(weight, in) + bias_val;
                        _Float16 acc = (_Float16)acc32;
                        acc = arm_convolve_f16_clamp_mve_compatible(acc, act_min, act_max);
                        *out_c = (float16_t)acc;
                        out_c += output_ch;
                    }

                    /* Last output column of the row advances to the next row. */
                    {
                        const float16x8_t in = vldrhq_gather_shifted_offset_z(input_data_pr, offset_src, p);
                        if (i_out_y + 1 < output_y)
                        {
                            input_data_pr += row_step;
                        }
                        const float32_t acc32 = arm_convolve_f16_widened_dot(weight, in) + bias_val;
                        _Float16 acc = (_Float16)acc32;
                        acc = arm_convolve_f16_clamp_mve_compatible(acc, act_min, act_max);
                        *out_c = (float16_t)acc;
                        if (i_out_y + 1 < output_y)
                        {
                            out_c += output_ch;
                        }
                    }
                }
            }
        }

        input_data += (size_t)input_x * input_y * input_ch;
        output_data += (size_t)output_x * output_y * output_ch;
    }

    return ARM_CMSIS_NN_SUCCESS;
    #else
    (void)bias_data;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
    #endif
}

/**
 * @} end of NNConv group
 */

#endif /* ARM_NN_ENABLE_F16 */
