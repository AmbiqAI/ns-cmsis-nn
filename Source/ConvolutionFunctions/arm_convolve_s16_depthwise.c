/*
 * SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_s16_depthwise.c
 * Description:  s16 depthwise convolution for the special case where the
 *               conv2d filter has filter_dims->c == 1 and input_ch == output_ch.
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @brief Depthwise convolution for the special case where the conv2d filter
 *        has filter_dims->c == 1 and input_ch == output_ch (i.e. each output
 *        channel depends on exactly one input channel).
 *
 * Filter layout expected: [C_OUT, HK, WK, 1] (standard conv2d layout with C_IN=1).
 * Supports arbitrary strides, dilation and padding.
 */
arm_cmsis_nn_status arm_convolve_s16_depthwise(const cmsis_nn_context *ctx,
                                               const cmsis_nn_conv_params *conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int16_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const cmsis_nn_bias_data *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int16_t *output_data)
{
    (void)ctx;
    (void)bias_dims;
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t input_ch = input_dims->c;
    const int32_t input_batches = input_dims->n;

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

    const int32_t activation_min = conv_params->activation.min;
    const int32_t activation_max = conv_params->activation.max;

    const int32_t *output_mult = quant_params->multiplier;
    const int32_t *output_shift = quant_params->shift;

    const bool has_bias = (bias_data != NULL) && (bias_data->data != NULL);
    const bool bias_is_s32 = has_bias && bias_data->is_int32_bias;
    const int32_t *bias_s32 = bias_is_s32 ? (const int32_t *)bias_data->data : NULL;
    const int64_t *bias_s64 = (!bias_is_s32 && has_bias) ? (const int64_t *)bias_data->data : NULL;

    /* Filter is [C_OUT, HK, WK, 1]; stride between output channels = kernel_y * kernel_x */
    const int32_t rhs_cols = kernel_y * kernel_x;

#if defined(ARM_MATH_MVEI)
    /* MVE fast path: kernel fits in 1 or 2 vectors (<= 16 elements), no padding.
       Gather-offset + vmladavq pattern from arm_convolve_s16_fast_small_kernel.
       For rhs_cols > 8 (e.g. 3x3=9), split across two vector pairs. */
    /* Guard: gather offsets are uint16, so max byte offset must fit in 16 bits.
       Worst-case tap is at (kernel_y-1)*dilation_y rows + (kernel_x-1)*dilation_x cols. */
    const int32_t max_byte_offset = (((kernel_y - 1) * dilation_y * input_x +
                                      (kernel_x - 1) * dilation_x) *
                                     input_ch * (int32_t)sizeof(int16_t));

    if (rhs_cols <= 16 && pad_y == 0 && pad_x == 0 && max_byte_offset <= 0xFFFF)
    {
        const int32_t rhs_cols_0 = (rhs_cols <= 8) ? rhs_cols : 8;
        const int32_t rhs_cols_1 = rhs_cols - rhs_cols_0;

        uint16x8_t offset_src_0;
        uint16x8_t offset_src_1;
        for (int32_t ky = 0; ky < kernel_y; ky++)
        {
            for (int32_t kx = 0; kx < kernel_x; kx++)
            {
                int32_t idx = ky * kernel_x + kx;
                uint16_t off = (uint16_t)((ky * dilation_y * input_x + kx * dilation_x) *
                                           input_ch * (int32_t)sizeof(int16_t));
                if (idx < 8)
                {
                    offset_src_0[idx] = off;
                }
                else
                {
                    offset_src_1[idx - 8] = off;
                }
            }
        }

        mve_pred16_t p0 = vctp16q(rhs_cols_0);
        mve_pred16_t p1 = vctp16q(rhs_cols_1);
        int32_t stride_edge = input_x - (output_x - 1) * stride_x + (stride_y - 1) * input_x;

        for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
        {
            const int8_t *filter_ptr = filter_data;
            const int32_t *mult_ptr = output_mult;
            const int32_t *shift_ptr = output_shift;
            const int32_t *bs32 = bias_s32;
            const int64_t *bs64 = bias_s64;

            for (int32_t c = 0; c < output_ch; c++)
            {
                int16_t *input_ptr = (int16_t *)input_data + c;
                int16_t *out_c = output_data + c;

                int16x8_t weight_0 = vldrbq_z_s16(filter_ptr, p0);
                int16x8_t weight_1;
                if (rhs_cols_1 > 0)
                {
                    weight_1 = vldrbq_z_s16(filter_ptr + 8, p1);
                }
                filter_ptr += rhs_cols;

                int32_t multiplier = *mult_ptr++;
                int32_t shift = *shift_ptr++;
                int32_t bias_s32_val = 0;
                int64_t bias_s64_val = 0;
                int32_t reduced_multiplier = 0;

                if (bias_is_s32)
                {
                    if (bs32)
                    {
                        bias_s32_val = *bs32++;
                    }
                }
                else
                {
                    if (bs64)
                    {
                        bias_s64_val = *bs64++;
                    }
                    reduced_multiplier = REDUCE_MULTIPLIER(multiplier);
                }

                for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
                {
                    for (int32_t i_out_x = 0; i_out_x < output_x - 1; i_out_x++)
                    {
                        int16x8_t in_0 = vldrhq_gather_offset_z_s16(input_ptr, offset_src_0, p0);
                        int32_t result = vmladavq_s16(weight_0, in_0);
                        if (rhs_cols_1 > 0)
                        {
                            int16x8_t in_1 = vldrhq_gather_offset_z_s16(input_ptr, offset_src_1, p1);
                            result += vmladavq_s16(weight_1, in_1);
                        }
                        input_ptr += input_ch * stride_x;

                        if (bias_is_s32)
                        {
                            result += bias_s32_val;
                            result = arm_nn_requantize(result, multiplier, shift);
                        }
                        else
                        {
                            int64_t acc_s64 = result;
                            acc_s64 += bias_s64_val;
                            result = arm_nn_requantize_s64(acc_s64, reduced_multiplier, shift);
                        }
                        result = MAX(result, activation_min);
                        result = MIN(result, activation_max);
                        *out_c = (int16_t)result;
                        out_c += output_ch;
                    }
                    /* Last column + advance to next row */
                    {
                        int16x8_t in_0 = vldrhq_gather_offset_z_s16(input_ptr, offset_src_0, p0);
                        int32_t result = vmladavq_s16(weight_0, in_0);
                        if (rhs_cols_1 > 0)
                        {
                            int16x8_t in_1 = vldrhq_gather_offset_z_s16(input_ptr, offset_src_1, p1);
                            result += vmladavq_s16(weight_1, in_1);
                        }
                        input_ptr += input_ch * stride_edge;

                        if (bias_is_s32)
                        {
                            result += bias_s32_val;
                            result = arm_nn_requantize(result, multiplier, shift);
                        }
                        else
                        {
                            int64_t acc_s64 = result;
                            acc_s64 += bias_s64_val;
                            result = arm_nn_requantize_s64(acc_s64, reduced_multiplier, shift);
                        }
                        result = MAX(result, activation_min);
                        result = MIN(result, activation_max);
                        *out_c = (int16_t)result;
                        out_c += output_ch;
                    }
                }
            }
            input_data += input_y * input_x * input_ch;
            output_data += output_y * output_x * output_ch;
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
#endif /* ARM_MATH_MVEI */

    /* Scalar fallback: supports arbitrary kernel size, stride, dilation, padding */
    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            const int32_t base_y = i_out_y * stride_y - pad_y;
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_x = i_out_x * stride_x - pad_x;

                int32_t ker_y_start, ker_y_end;
                if (dilation_y > 1)
                {
                    ker_y_start = MAX(0, (-base_y + dilation_y - 1) / dilation_y);
                    ker_y_end = MIN(kernel_y, (input_y - base_y + dilation_y - 1) / dilation_y);
                }
                else
                {
                    ker_y_start = MAX(0, -base_y);
                    ker_y_end = MIN(kernel_y, input_y - base_y);
                }

                int32_t ker_x_start, ker_x_end;
                if (dilation_x > 1)
                {
                    ker_x_start = MAX(0, (-base_x + dilation_x - 1) / dilation_x);
                    ker_x_end = MIN(kernel_x, (input_x - base_x + dilation_x - 1) / dilation_x);
                }
                else
                {
                    ker_x_start = MAX(0, -base_x);
                    ker_x_end = MIN(kernel_x, input_x - base_x);
                }

                for (int32_t c = 0; c < output_ch; c++)
                {
                    int64_t acc = 0;
                    if (bias_s64)
                    {
                        acc = bias_s64[c];
                    }
                    else if (bias_s32)
                    {
                        acc = (int64_t)bias_s32[c];
                    }

                    const int8_t *filter_ptr = filter_data + c * rhs_cols;

                    for (int32_t ky = ker_y_start; ky < ker_y_end; ky++)
                    {
                        const int32_t iy = base_y + ky * dilation_y;
                        for (int32_t kx = ker_x_start; kx < ker_x_end; kx++)
                        {
                            const int32_t ix = base_x + kx * dilation_x;
                            acc += (int64_t)input_data[(iy * input_x + ix) * input_ch + c] *
                                   (int64_t)filter_ptr[ky * kernel_x + kx];
                        }
                    }

                    int32_t result;
                    if (bias_is_s32)
                    {
                        result = (int32_t)acc;
                        result = arm_nn_requantize(result, output_mult[c], output_shift[c]);
                    }
                    else
                    {
                        const int32_t mult_val = output_mult[c];
                        int32_t reduced_mult = REDUCE_MULTIPLIER(mult_val);
                        result = arm_nn_requantize_s64(acc, reduced_mult, output_shift[c]);
                    }
                    result = MAX(result, activation_min);
                    result = MIN(result, activation_max);
                    *output_data++ = (int16_t)result;
                }
            }
        }
        input_data += input_y * input_x * input_ch;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
