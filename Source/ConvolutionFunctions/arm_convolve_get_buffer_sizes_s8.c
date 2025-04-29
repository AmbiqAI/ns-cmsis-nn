/*
 * SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for the various s8 convolution layer functions.
 *
 * $Date:        31 October 2024
 * $Revision:    V.2.2.1
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_compiler.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "stdio.h"

/**
 *  @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */
__STATIC_INLINE int32_t arm_convolve_1x1_s8_fast_get_buffer_size_dsp(const cmsis_nn_dims *input_dims)
{
#if defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
    return (2 * input_dims->c) * (int32_t)sizeof(int16_t);
#else
    (void)input_dims;
    return 0;
#endif
}

__STATIC_INLINE int32_t arm_convolve_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                            const cmsis_nn_dims *filter_dims)
{
    int32_t col_length = input_dims->c * filter_dims->w * filter_dims->h;
    // Get number of complete lanes with int8 elements (multiple of 16) for given col_length. This is dependent on
    // implementation of arm_nn_mat_mult_nt_t_s8
    col_length = (col_length + 15) / 16;
    // 4 -> number of im2col buffers, 16 -> 16 elements per Q register
    return 4 * col_length * 16 * (int32_t)sizeof(int8_t);
}

__STATIC_INLINE int32_t arm_convolve_1_x_n_s8_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                                  const cmsis_nn_dims *input_dims,
                                                                  const cmsis_nn_dims *filter_dims,
                                                                  const cmsis_nn_dims *output_dims)
{
    const int32_t input_x = input_dims->w;
    const int32_t pad_x = conv_params->padding.w;
    const int32_t kernel_x = filter_dims->w;
    const int32_t output_x = output_dims->w;
    const int32_t stride_x = conv_params->stride.w;
    const int32_t total_pad = ((output_x - 1) * stride_x + kernel_x - input_x);
    const int32_t asym_pad = total_pad % 2;

    const int32_t right_pad_num = pad_x + asym_pad != 0 ? MAX(1, (pad_x + asym_pad + stride_x - 1) / stride_x) : 0;
    const int32_t left_pad_num = pad_x != 0 ? MAX(1, (pad_x + stride_x - 1) / stride_x) : 0;
    const int32_t no_pad_num = MAX(output_x - (right_pad_num + left_pad_num), 0);

    if (right_pad_num + no_pad_num + left_pad_num != output_x)
    {
        return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
    }

    const int32_t pad_size_left = pad_x * input_dims->c;
    const int32_t pad_size_right = asym_pad ? right_pad_num * input_dims->c : pad_size_left;
    const int32_t num_elem_left = kernel_x * input_dims->c;
    const int32_t num_elem_right = num_elem_left - input_dims->c;
    const int32_t size_1_x_n = MAX(num_elem_left + pad_size_left, num_elem_right + pad_size_right);

    return size_1_x_n;
}

int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
#else
    const int32_t rhs_cols = filter_dims->w * filter_dims->h * input_dims->c;
    const int32_t remainder = rhs_cols % 4;
    const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;
    return (2 * aligned_rhs_cols) * (int32_t)sizeof(int16_t);
#endif
}


int32_t arm_convolve_s8_get_weights_sum_size(const cmsis_nn_dims *output_dims)
{
#if !defined(ARM_MATH_MVEI)
    (void)output_dims;
    return 0;
#else
    const int32_t weights_sums_size = output_dims->c * (int32_t)sizeof(int32_t);
    return weights_sums_size;
#endif
}

arm_cmsis_nn_status arm_depthwise_convolve_weight_sum(
        int32_t* vector_sum_buf,
        int8_t* scratch_buf,
        const int8_t *rhs,
        const cmsis_nn_dw_conv_params *dw_conv_params,
        cmsis_nn_dims *input_dims,
        cmsis_nn_dims *filter_dims,
        cmsis_nn_dims *output_dims, 
        const int32_t lhs_offset,
        const int32_t *bias_data )
{

#if !defined(ARM_MATH_MVEI)
    (void)vector_sum_buf;
    (void)scratch_buf;
    (void)rhs;
    (void)dw_conv_params;
    (void)input_dims;
    (void)filter_dims;
    (void)output_dims;
    (void)lhs_offset;
    (void)bias_data;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#else //defined(ARM_MATH_MVEI)
    if (vector_sum_buf == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t output_channels = output_dims->c;
    (void)input_dims;
    (void)scratch_buf;
    (void)dw_conv_params;
    int32_t total_ch = output_channels;
    int32_t row_x_col = kernel_x * kernel_y;
    int32_t ch_left = total_ch;
    const int8_t* rhs_base = rhs;
    const int8_t* rhs_runner = rhs_base;
    
    if (bias_data)
    {
        memcpy(vector_sum_buf, bias_data, output_channels * sizeof(int32_t));
    }
    else
    {
        memset(vector_sum_buf, 0, output_channels * sizeof(int32_t));
    }
    if (lhs_offset ) {
        for (int i =0; i < (total_ch + 3) / 4; ++i) {
            mve_pred16_t p = vctp32q(ch_left);
            int32x4_t ker_sum = vdupq_n_s32(0);
            rhs_runner = rhs_base;
            for (int i_row_x_col = 0; i_row_x_col < row_x_col; i_row_x_col++) {
                //run through all of the kernel values for these 4 channels
                const int32x4_t ker_0 = vldrbq_z_s32(rhs_runner, p);
                ker_sum = vaddq_s32(ker_sum, ker_0);
                rhs_runner += total_ch;
            }
            ker_sum = vmulq_n_s32(ker_sum, lhs_offset);
            //vstrwq_p_s32(sum_buf_runner, ker_sum, p);
            for(int j = 0; j < MIN(4, ch_left); ++j) {
                vector_sum_buf[j] += ker_sum[j];
            }
            rhs_base += 4;
            vector_sum_buf += 4;
            ch_left -= 4;

        }
    }
#endif
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_convolve_weight_sum(
        int32_t* vector_sum_buf,
        const int8_t *rhs,
        cmsis_nn_dims *input_dims,
        cmsis_nn_dims *filter_dims,
        cmsis_nn_dims *output_dims, 
        const int32_t lhs_offset,
        const int32_t *bias_data )
{
#if !defined(ARM_MATH_MVEI)
    (void)vector_sum_buf;
    (void)rhs;
    (void)input_dims;
    (void)filter_dims;
    (void)output_dims;
    (void)lhs_offset;
    (void)bias_data;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#else //defined(ARM_MATH_MVEI)
    (void) input_dims;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t kernel_ch = filter_dims->c;
    const uint16_t output_channels = output_dims->c;
    const uint16_t rhs_cols = kernel_x * kernel_y * kernel_ch;
    arm_vector_sum_s8(vector_sum_buf, rhs_cols, output_channels, rhs, lhs_offset, 0, bias_data);
    return ARM_CMSIS_NN_SUCCESS;
#endif
}

int32_t arm_convolve_1_x_n_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *output_dims)
{
#if !defined(ARM_MATH_MVEI)
    (void)conv_params;
    (void)output_dims;

    return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
#else
    return arm_convolve_1_x_n_s8_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
#endif
}

int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    return arm_convolve_1x1_s8_fast_get_buffer_size_dsp(input_dims);
#else
    (void)input_dims;
#endif
    return 0;
}

/*
 * Get the required buffer size for arm_convolve_wrapper_s8. This is the recommended function convolve wrapper s8
 * function.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_wrapper_s8_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
#elif defined(ARM_MATH_DSP)
    return arm_convolve_wrapper_s8_get_buffer_size_dsp(conv_params, input_dims, filter_dims, output_dims);
#else
    (void)output_dims;
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (filter_dims->w == 1) &&
        (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            return arm_convolve_1x1_s8_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if ((input_dims->h == 1) && (conv_params->dilation.w == 1) && (filter_dims->h == 1) &&
             (conv_params->stride.w * input_dims->c % 4 == 0))
    {
        return arm_convolve_1_x_n_s8_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
    }
    else
    {
        return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
    }
#endif
}

int32_t arm_convolve_wrapper_s8_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims)
{
    (void)output_dims;
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (filter_dims->w == 1) &&
        (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            return arm_convolve_1x1_s8_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if ((input_dims->h == 1) && (conv_params->dilation.w == 1) && (filter_dims->h == 1) &&
             (conv_params->stride.w * input_dims->c % 4 == 0))
    {
        return arm_convolve_1_x_n_s8_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
    }
    else
    {
        return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
    }
}

int32_t arm_convolve_wrapper_s8_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims)
{
    (void)output_dims;
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (filter_dims->w == 1) &&
        (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            return arm_convolve_1x1_s8_fast_get_buffer_size_dsp(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if ((input_dims->h == 1) && (conv_params->dilation.w == 1) && (filter_dims->h == 1) &&
             (conv_params->stride.w * input_dims->c % 4 == 0))
    {
        return arm_convolve_1_x_n_s8_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
    }
    else
    {
        return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
    }
}

/**
 * @} end of GetBufferSizeNNConv group
 */
