/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
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
 * Title:        arm_convolve_1_x_n_s8.c
 * Description:  s8 version of 1xN convolution using symmetric quantization.
 *
 * $Date:        04 November 2024
 * $Revision:    V.3.6.1
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

#if defined(ARM_MATH_MVEI)
/* Copies input columns first_col to first_col + num_cols - 1 of one batch to dst, with pad_value in the columns that
   fall outside the input. */
static void arm_convolve_1_x_n_s8_stage(int8_t *dst,
                                        const int8_t *input,
                                        const int32_t input_x,
                                        const int32_t input_ch,
                                        const int64_t first_col,
                                        const int32_t num_cols,
                                        const int8_t pad_value)
{
    const int32_t lead = (int32_t)ARM_NN_MIN((int64_t)num_cols, ARM_NN_MAX(-first_col, 0));
    const int64_t real_start = ARM_NN_MAX(first_col, 0);
    const int32_t real =
        (int32_t)ARM_NN_MAX(ARM_NN_MIN(first_col + num_cols, (int64_t)input_x) - real_start, (int64_t)0);
    const int32_t trail = num_cols - lead - real;

    arm_memset_s8(dst, pad_value, (uint32_t)(lead * input_ch));
    dst += lead * input_ch;
    if (real > 0)
    {
        arm_memcpy_s8(dst, input + real_start * input_ch, (uint32_t)(real * input_ch));
        dst += real * input_ch;
    }
    arm_memset_s8(dst, pad_value, (uint32_t)(trail * input_ch));
}
#endif

/*
 * 1xN s8 convolution function.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_convolve_1_x_n_s8(const cmsis_nn_context *ctx,
                                          const cmsis_nn_context *weight_sum_ctx,
                                          const cmsis_nn_conv_params *conv_params,
                                          const cmsis_nn_per_channel_quant_params *quant_params,
                                          const cmsis_nn_dims *input_dims,
                                          const int8_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const int8_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const int32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          int8_t *output_data)
{
    arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;

    /* The wrapper API is the ultimate reference for argument check */
    if ((input_dims->h != 1) || (input_dims->w < 0) || (output_dims->w < 0) || conv_params->dilation.w != 1 ||
        ctx->buf == NULL || conv_params->stride.w <= 0 || (((int64_t)conv_params->stride.w * input_dims->c) % 4 != 0) ||
        !arm_nn_convolve_1_x_n_s8_padding_supported(conv_params, filter_dims, output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEI)
    (void)bias_dims;

    /* Only this MVE path reads the per-channel weight sums, through arm_nn_mat_mult_nt_t_s8(). Diagnose a
       missing buffer here rather than dereferencing NULL and silently returning garbage output. */
    if (weight_sum_ctx->buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t input_x = input_dims->w;
    const int32_t kernel_x = filter_dims->w;
    const int32_t output_x = output_dims->w;
    const int32_t input_ch = input_dims->c;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t stride_x = conv_params->stride.w;
    const int8_t pad_value = (int8_t)-conv_params->input_offset;

    /* The left-padded windows come first and the right-padded ones last. Each group is staged once, as a padded copy
       of the input columns it spans, and read from there at the layer's stride; the windows in between read the
       input in place. */
    int64_t left_num;
    int64_t right_num;
    arm_nn_convolve_1_x_n_padded_columns(conv_params, input_dims, filter_dims, output_dims, &left_num, &right_num);
    const int64_t no_pad_num = output_x - left_num - right_num;
    const int64_t left_cols = left_num > 0 ? (left_num - 1) * stride_x + kernel_x : 0;
    const int64_t right_cols = right_num > 0 ? (right_num - 1) * stride_x + kernel_x : 0;
    // arm_convolve_1_x_n_s8_get_buffer_size() reports this span, raised to at least one window.
    const int64_t staging_cols = ARM_NN_MAX(left_cols, right_cols);
    // Bounded before the multiply: a span above INT32_MAX times input_ch can wrap an int64_t.
    if (staging_cols > INT32_MAX)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int64_t staging_size = staging_cols * input_ch;

    /* A size of 0 means the caller does not report it. */
    if ((staging_size > INT32_MAX) || ((ctx->size != 0) && (ctx->size < staging_size)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rhs_cols = kernel_x * input_ch;
    const int32_t rhs_rows = output_dims->c;
    const int32_t lhs_offset = input_ch * stride_x;
    const int64_t right_first_col = (left_num + no_pad_num) * stride_x - pad_x;
    const int32_t lhs_rows[3] = {(int32_t)left_num, (int32_t)no_pad_num, (int32_t)right_num};

    for (int i_batch = 0; i_batch < input_dims->n; i_batch++)
    {
        for (int32_t section = 0; section < 3; section++)
        {
            if (lhs_rows[section] == 0)
            {
                continue;
            }

            const int8_t *lhs = (const int8_t *)ctx->buf;
            if (section == 0)
            {
                arm_convolve_1_x_n_s8_stage(
                    ctx->buf, input_data, input_x, input_ch, -(int64_t)pad_x, (int32_t)left_cols, pad_value);
            }
            else if (section == 1)
            {
                lhs = input_data + (left_num * stride_x - pad_x) * input_ch;
            }
            else
            {
                arm_convolve_1_x_n_s8_stage(
                    ctx->buf, input_data, input_x, input_ch, right_first_col, (int32_t)right_cols, pad_value);
            }

            arm_nn_mat_mult_nt_t_s8(weight_sum_ctx->buf,
                                    lhs,
                                    filter_data,
                                    bias_data,
                                    output_data,
                                    quant_params->multiplier,
                                    quant_params->shift,
                                    lhs_rows[section],
                                    rhs_rows,
                                    rhs_cols,
                                    conv_params->input_offset,
                                    conv_params->output_offset,
                                    conv_params->activation.min,
                                    conv_params->activation.max,
                                    rhs_rows,
                                    lhs_offset);

            output_data += lhs_rows[section] * rhs_rows;
        }

        /* Advance to the next batch */
        input_data += (input_x * input_ch);
    }
#else
    status = arm_convolve_s8(ctx,
                             weight_sum_ctx,
                             conv_params,
                             quant_params,
                             input_dims,
                             input_data,
                             filter_dims,
                             filter_data,
                             bias_dims,
                             bias_data,
                             NULL,
                             output_dims,
                             output_data);

#endif

    /* Return to application */
    return status;
}

/**
 * @} end of NNConv group
 */
