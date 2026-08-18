/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
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
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    (void)ctx;
    (void)bias_dims;

    const int32_t input_batches = input_dims->n;
    const int32_t input_x = input_dims->w;
    const int32_t input_y = input_dims->h;
    const int32_t input_ch = input_dims->c;
    const int32_t kernel_x = filter_dims->w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_ch = filter_dims->c;
    const int32_t output_x = output_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_ch = output_dims->c;

    const int32_t dilation_x = conv_params->dilation.w;
    const int32_t dilation_y = conv_params->dilation.h;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t stride_y = conv_params->stride.h;

    const _Float16 act_min = (_Float16)conv_params->activation.min;
    const _Float16 act_max = (_Float16)conv_params->activation.max;

    if (kernel_ch <= 0 || input_ch % kernel_ch != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t groups = input_ch / kernel_ch;
    if (groups <= 0 || output_ch % groups != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t output_ch_per_group = output_ch / groups;
    const int32_t rhs_cols = kernel_ch * kernel_y * kernel_x;

    /* Only handle the shapes this kernel is specialized for. */
    if (rhs_cols <= 0 || rhs_cols > 8 || conv_params->padding.w != 0 || conv_params->padding.h != 0)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    /* Byte-scaled gather uses 16-bit per-lane offsets; bail out if the window does not fit. */
    uint16x8_t offset_src = vdupq_n_u16(0);
    size_t max_offset = 0;
    for (int32_t i = 0; i < kernel_y; i++)
    {
        const int32_t id = i * dilation_y;
        for (int32_t j = 0; j < kernel_x; j++)
        {
            const int32_t jd = j * dilation_x;
            const int32_t idx = i * kernel_x + j;
            for (int32_t c = 0; c < kernel_ch; c++)
            {
                const size_t off = (size_t)(id * input_x + jd) * (size_t)input_ch + (size_t)c;
                if (off > max_offset)
                {
                    max_offset = off;
                }
                offset_src[idx * kernel_ch + c] = (uint16_t)off;
            }
        }
    }
    if (max_offset > ARM_NN_MVE_F16_GATHER_OFFSET_MAX)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    /* Pointer step from the last output column of a row to the first of the next row. */
    const int32_t stride_edge = input_x - (output_x - 1) * stride_x + (stride_y - 1) * input_x;
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
                const _Float16 bias_val =
                    bias_data ? (_Float16)bias_data[i_group * output_ch_per_group + c] : (_Float16)0.0f;

                for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
                {
                    for (int32_t i_out_x = 0; i_out_x < output_x - 1; i_out_x++)
                    {
                        const float16x8_t in = vldrhq_gather_shifted_offset_z(input_data_pr, offset_src, p);
                        input_data_pr += input_ch * stride_x;
                        _Float16 acc = (_Float16)arm_nn_vec_reduce_add_f16(vmulq(weight, in)) + bias_val;
                        acc = CLAMP(acc, act_max, act_min);
                        *out_c = (float16_t)acc;
                        out_c += output_ch;
                    }

                    /* Last output column of the row advances to the next row. */
                    {
                        const float16x8_t in = vldrhq_gather_shifted_offset_z(input_data_pr, offset_src, p);
                        input_data_pr += input_ch * stride_edge;
                        _Float16 acc = (_Float16)arm_nn_vec_reduce_add_f16(vmulq(weight, in)) + bias_val;
                        acc = CLAMP(acc, act_max, act_min);
                        *out_c = (float16_t)acc;
                        out_c += output_ch;
                    }
                }
            }
        }

        input_data += input_x * input_y * input_ch;
        output_data += output_x * output_y * output_ch;
    }

    return ARM_CMSIS_NN_SUCCESS;
    #else
    (void)ctx;
    (void)conv_params;
    (void)input_dims;
    (void)input_data;
    (void)filter_dims;
    (void)filter_data;
    (void)bias_dims;
    (void)bias_data;
    (void)output_dims;
    (void)output_data;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
    #endif
}

/**
 * @} end of NNConv group
 */

#endif /* ARM_NN_ENABLE_F16 */
