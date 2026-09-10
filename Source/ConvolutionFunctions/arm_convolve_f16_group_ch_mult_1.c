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

    if (conv_params->weight_format != ARM_NN_WEIGHT_FORMAT_STANDARD)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }
    if (filter_dims->c != 1 || input_ch <= 0 || input_ch != output_ch || input_batches < 0 || input_x <= 0 ||
        input_y <= 0 || kernel_x <= 0 || kernel_y <= 0 || output_x < 0 || output_y < 0 || stride_x <= 0 ||
        stride_y <= 0 || dilation_x <= 0 || dilation_y <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (output_x == 0 || output_y == 0 || input_batches == 0)
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
    const int64_t max_offset_64 =
        ((int64_t)(kernel_y - 1) * dilation_y * input_x + (int64_t)(kernel_x - 1) * dilation_x) * input_ch;
    const bool output_fits = (int64_t)(output_x - 1) * stride_x + (int64_t)(kernel_x - 1) * dilation_x < input_x &&
        (int64_t)(output_y - 1) * stride_y + (int64_t)(kernel_y - 1) * dilation_y < input_y;

    if (rhs_cols <= 16 && pad_y == 0 && pad_x == 0 && output_fits && max_offset_64 >= 0 &&
        (uint64_t)max_offset_64 <= ARM_NN_MVE_F16_GATHER_OFFSET_MAX)
    {
        const int32_t rhs_cols_0 = (rhs_cols < 8) ? rhs_cols : 8;
        const int32_t rhs_cols_1 = rhs_cols - rhs_cols_0;
        uint16x8_t offset_src_0 = vdupq_n_u16(0);
        uint16x8_t offset_src_1 = vdupq_n_u16(0);

        for (int32_t ky = 0; ky < kernel_y; ++ky)
        {
            for (int32_t kx = 0; kx < kernel_x; ++kx)
            {
                const int32_t idx = ky * kernel_x + kx;
                const uint16_t offset =
                    (uint16_t)(((int64_t)ky * dilation_y * input_x + (int64_t)kx * dilation_x) * input_ch);
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

        const mve_pred16_t p0 = vctp16q((uint32_t)rhs_cols_0);
        const mve_pred16_t p1 = vctp16q((uint32_t)rhs_cols_1);
        const int32_t stride_edge = input_x - (output_x - 1) * stride_x + (stride_y - 1) * input_x;

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
                const _Float16 bias = bias_data ? (_Float16)bias_data[c] : (_Float16)0.0f;

                for (int32_t out_y = 0; out_y < output_y; ++out_y)
                {
                    for (int32_t out_x = 0; out_x < output_x - 1; ++out_x)
                    {
                        const float16x8_t input_0 = vldrhq_gather_shifted_offset_z(input_ptr, offset_src_0, p0);
                        _Float16 acc = (_Float16)arm_nn_vec_reduce_add_f16(vmulq(weight_0, input_0));
                        if (rhs_cols_1 > 0)
                        {
                            const float16x8_t input_1 = vldrhq_gather_shifted_offset_z(input_ptr, offset_src_1, p1);
                            acc += (_Float16)arm_nn_vec_reduce_add_f16(vmulq(weight_1, input_1));
                        }
                        acc += bias;
                        *out_c = (float16_t)arm_nn_clamp_f16h(acc, activation_max, activation_min);
                        out_c += output_ch;
                        input_ptr += input_ch * stride_x;
                    }

                    const float16x8_t input_0 = vldrhq_gather_shifted_offset_z(input_ptr, offset_src_0, p0);
                    _Float16 acc = (_Float16)arm_nn_vec_reduce_add_f16(vmulq(weight_0, input_0));
                    if (rhs_cols_1 > 0)
                    {
                        const float16x8_t input_1 = vldrhq_gather_shifted_offset_z(input_ptr, offset_src_1, p1);
                        acc += (_Float16)arm_nn_vec_reduce_add_f16(vmulq(weight_1, input_1));
                    }
                    acc += bias;
                    *out_c = (float16_t)arm_nn_clamp_f16h(acc, activation_max, activation_min);
                    out_c += output_ch;
                    input_ptr += input_ch * stride_edge;
                }
            }
            input_data += (size_t)input_y * input_x * input_ch;
            output_data += (size_t)output_y * output_x * output_ch;
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
    #endif

    for (int32_t b = 0; b < input_batches; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_y * input_x * input_ch;
        float16_t *output_b = output_data + (size_t)b * output_y * output_x * output_ch;

        for (int32_t out_y = 0; out_y < output_y; ++out_y)
        {
            const int32_t base_y = out_y * stride_y - pad_y;
            for (int32_t out_x = 0; out_x < output_x; ++out_x)
            {
                const int32_t base_x = out_x * stride_x - pad_x;
                int32_t kernel_y_start;
                int32_t kernel_y_end;
                int32_t kernel_x_start;
                int32_t kernel_x_end;

                if (dilation_y > 1)
                {
                    kernel_y_start = MAX(0, (-base_y + dilation_y - 1) / dilation_y);
                    kernel_y_end = MIN(kernel_y, (input_y - base_y + dilation_y - 1) / dilation_y);
                }
                else
                {
                    kernel_y_start = MAX(0, -base_y);
                    kernel_y_end = MIN(kernel_y, input_y - base_y);
                }
                if (dilation_x > 1)
                {
                    kernel_x_start = MAX(0, (-base_x + dilation_x - 1) / dilation_x);
                    kernel_x_end = MIN(kernel_x, (input_x - base_x + dilation_x - 1) / dilation_x);
                }
                else
                {
                    kernel_x_start = MAX(0, -base_x);
                    kernel_x_end = MIN(kernel_x, input_x - base_x);
                }

                for (int32_t c = 0; c < output_ch; ++c)
                {
                    _Float16 acc = bias_data ? (_Float16)bias_data[c] : (_Float16)0.0f;
                    const float16_t *filter_c = filter_data + (size_t)c * rhs_cols;

                    for (int32_t ky = kernel_y_start; ky < kernel_y_end; ++ky)
                    {
                        const int32_t in_y = base_y + ky * dilation_y;
                        for (int32_t kx = kernel_x_start; kx < kernel_x_end; ++kx)
                        {
                            const int32_t in_x = base_x + kx * dilation_x;
                            const size_t input_index = ((size_t)in_y * input_x + in_x) * input_ch + c;
                            acc += (_Float16)input_b[input_index] * (_Float16)filter_c[ky * kernel_x + kx];
                        }
                    }

                    const size_t output_index = ((size_t)out_y * output_x + out_x) * output_ch + c;
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
