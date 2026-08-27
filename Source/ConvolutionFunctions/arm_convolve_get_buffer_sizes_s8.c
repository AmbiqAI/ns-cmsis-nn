/*
 * SPDX-FileCopyrightText: Copyright 2023-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for the various s8 convolution layer functions.
 *
 * $Date:        6 Mar 2026
 * $Revision:    V.2.3.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_compiler.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

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
    const int64_t required_bytes = (2 * (int64_t)input_dims->c) * (int64_t)sizeof(int16_t);

    if ((input_dims->c < 0) || (required_bytes > INT32_MAX))
    {
        return -1;
    }

    return (int32_t)required_bytes;
#else
    (void)input_dims;
    return 0;
#endif
}

int32_t arm_convolve_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    // Folded one factor at a time so the accumulator stays bounded; see arm_nn_size_mul().
    int64_t col_elements = arm_nn_size_mul(1, input_dims->c);
    col_elements = arm_nn_size_mul(col_elements, filter_dims->w);
    col_elements = arm_nn_size_mul(col_elements, filter_dims->h);

    if (col_elements < 0)
    {
        return -1;
    }

    // Get number of complete lanes with int8 elements (multiple of 16) for given col_length. This is dependent on
    // implementation of arm_nn_mat_mult_nt_t_s8
    const int64_t col_length = (col_elements + 15) / 16;
    // 4 -> number of im2col buffers, 16 -> 16 elements per Q register
    int64_t required_bytes = arm_nn_size_mul(4, col_length);
    required_bytes = arm_nn_size_mul(required_bytes, 16);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int8_t));

    return (int32_t)required_bytes;
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

    if ((input_dims->c < 0) || (filter_dims->w < 0) || (filter_dims->h < 0) || (input_x < 0) || (output_x < 0) ||
        (pad_x < 0) || (stride_x <= 0))
    {
        return -1;
    }

    // total_pad and the pad-region counts are computed in 64 bits: (output_x - 1) * stride_x is signed-overflow UB
    // in int32_t, and a wrapped total_pad flips asym_pad, which selects a smaller size below.
    const int64_t total_pad = ((int64_t)output_x - 1) * (int64_t)stride_x + (int64_t)kernel_x - (int64_t)input_x;
    const int64_t asym_pad = total_pad % 2;

    const int64_t right_pad_num = pad_x + asym_pad != 0 ? MAX(1, (pad_x + asym_pad + stride_x - 1) / stride_x) : 0;
    const int64_t left_pad_num = pad_x != 0 ? MAX(1, ((int64_t)pad_x + stride_x - 1) / stride_x) : 0;
    const int64_t no_pad_num = MAX(output_x - (right_pad_num + left_pad_num), 0);

    if (right_pad_num + no_pad_num + left_pad_num != output_x)
    {
        return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
    }

    const int64_t pad_size_left = (int64_t)pad_x * (int64_t)input_dims->c;
    const int64_t pad_size_right = asym_pad ? right_pad_num * (int64_t)input_dims->c : pad_size_left;
    const int64_t num_elem_left = (int64_t)kernel_x * (int64_t)input_dims->c;
    const int64_t num_elem_right = num_elem_left - (int64_t)input_dims->c;
    // Both factors are bounded by INT32_MAX, so these products and sums cannot wrap an int64_t.
    const int64_t size_1_x_n = MAX(num_elem_left + pad_size_left, num_elem_right + pad_size_right);

    if (size_1_x_n > INT32_MAX)
    {
        return -1;
    }

    return (int32_t)size_1_x_n;
}

int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    // Dim sanity is validated once here so an invalid dim returns -1 on every build target. The byte count is
    // checked per leg, since the two legs use different formulas.
    if ((input_dims->c < 0) || (filter_dims->w < 0) || (filter_dims->h < 0))
    {
        return -1;
    }

#if defined(ARM_MATH_MVEI)
    return arm_convolve_s8_get_buffer_size_mve(input_dims, filter_dims);
#else
    int64_t rhs_cols = arm_nn_size_mul(1, filter_dims->w);
    rhs_cols = arm_nn_size_mul(rhs_cols, filter_dims->h);
    rhs_cols = arm_nn_size_mul(rhs_cols, input_dims->c);

    if (rhs_cols < 0)
    {
        return -1;
    }

    const int64_t remainder = rhs_cols % 4;
    const int64_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;
    int64_t required_bytes = arm_nn_size_mul(2, aligned_rhs_cols);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int16_t));

    return (int32_t)required_bytes;
#endif
}

int32_t arm_convolve_s8_get_weights_sum_size(const cmsis_nn_dims *output_dims)
{
#if !defined(ARM_MATH_MVEI)
    (void)output_dims;
    return 0;
#else
    const int64_t weights_sums_size = (int64_t)output_dims->c * (int64_t)sizeof(int32_t);

    if ((output_dims->c < 0) || (weights_sums_size > INT32_MAX))
    {
        return -1;
    }

    return (int32_t)weights_sums_size;
#endif
}

int32_t arm_convolve_1_x_n_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *output_dims)
{
    // Validated here rather than only in the MVE leg so that an out-of-range dim or a non-positive stride yields -1
    // on every build target, giving callers one portable contract to test against.
    if ((input_dims->c < 0) || (input_dims->w < 0) || (filter_dims->w < 0) || (filter_dims->h < 0) ||
        (output_dims->w < 0) || (conv_params->padding.w < 0) || (conv_params->stride.w <= 0))
    {
        return -1;
    }

#if !defined(ARM_MATH_MVEI)
    (void)output_dims;

    return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
#else
    return arm_convolve_1_x_n_s8_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
#endif
}

int32_t arm_convolve_1x1_out_s8_get_buffer_size(const cmsis_nn_dims *filter_dims)
{
    // Folded one factor at a time so the accumulator stays bounded; see arm_nn_size_mul(). Validated on every
    // build target rather than only on the MVE leg, so an invalid dim returns -1 uniformly instead of wrapping
    // to a small positive byte count that a caller would accept and then under-allocate.
    int64_t rhs_cols = arm_nn_size_mul(1, filter_dims->w);
    rhs_cols = arm_nn_size_mul(rhs_cols, filter_dims->h);
    rhs_cols = arm_nn_size_mul(rhs_cols, filter_dims->c);

    if (rhs_cols < 0)
    {
        return -1;
    }

    const int64_t remainder = rhs_cols % 4;
    const int64_t aligned_rhs_cols = arm_nn_size_add(rhs_cols, remainder != 0 ? 4 - remainder : 0);

    if (aligned_rhs_cols < 0)
    {
        return -1;
    }

#if !defined(ARM_MATH_MVEI)
    return 0;
#else
    return (int32_t)aligned_rhs_cols;
#endif
}

int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims)
{
    // Dim sanity is validated here so a negative channel count returns -1 on every build target, even though only
    // some targets actually need this buffer.
    if (input_dims->c < 0)
    {
        return -1;
    }

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    return arm_convolve_1x1_s8_fast_get_buffer_size_dsp(input_dims);
#else
    return 0;
#endif
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
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))
    {
        if (arm_nn_is_convolve_1x1_fast(conv_params))
        {
            return arm_convolve_1x1_s8_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if (arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims))
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
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))
    {
        if (arm_nn_is_convolve_1x1_fast(conv_params))
        {
            return arm_convolve_1x1_s8_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if (arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims))
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
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))
    {
        if (arm_nn_is_convolve_1x1_fast(conv_params))
        {
            return arm_convolve_1x1_s8_fast_get_buffer_size_dsp(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if (arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims))
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
