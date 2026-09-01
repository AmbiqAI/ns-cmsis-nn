/*
 * SPDX-FileCopyrightText: Copyright 2023-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_get_buffer_sizes_s4.c
 * Description:  Collection of get buffer size functions for the various s4 convolution layer functions.
 *
 * $Date:        27 Feb 2026
 * $Revision:    V.1.1.1
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

__STATIC_INLINE int32_t arm_convolve_s4_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                            const cmsis_nn_dims *filter_dims)
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
    // implementation of arm_nn_mat_mult_nt_t_s4
    const int64_t col_length = (col_elements + 15) / 16;
    // 4 -> number of im2col buffers, 16 -> 16 elements per Q register
    int64_t required_bytes = arm_nn_size_mul(4, col_length);
    required_bytes = arm_nn_size_mul(required_bytes, 16);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int8_t));

    return (int32_t)required_bytes;
}

__STATIC_INLINE int32_t arm_convolve_1_x_n_s4_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
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
    // in int32_t, and a wrapped total_pad flips asym_pad, which decides below whether the im2col buffer is needed.
    const int64_t total_pad = ((int64_t)output_x - 1) * (int64_t)stride_x + (int64_t)kernel_x - (int64_t)input_x;
    const int64_t asym_pad = total_pad % 2;

    const int64_t right_pad_num = pad_x + asym_pad != 0 ? MAX(1, (pad_x + asym_pad + stride_x - 1) / stride_x) : 0;
    const int64_t left_pad_num = pad_x != 0 ? MAX(1, ((int64_t)pad_x + stride_x - 1) / stride_x) : 0;
    const int64_t no_pad_num = MAX(output_x - (right_pad_num + left_pad_num), 0);

    if (right_pad_num + no_pad_num + left_pad_num != output_x)
    {
        return arm_convolve_s4_get_buffer_size_mve(input_dims, filter_dims);
    }

    return 0;
}

int32_t arm_convolve_s4_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    // Dim sanity is validated once here so an invalid dim returns -1 on every build target.
    if ((input_dims->c < 0) || (filter_dims->w < 0) || (filter_dims->h < 0))
    {
        return -1;
    }

    // Folded one factor at a time so the accumulator stays bounded; see arm_nn_size_mul().
    int64_t rhs_cols = arm_nn_size_mul(1, filter_dims->w);
    rhs_cols = arm_nn_size_mul(rhs_cols, filter_dims->h);
    rhs_cols = arm_nn_size_mul(rhs_cols, input_dims->c);

    if (rhs_cols < 0)
    {
        return -1;
    }

    int64_t required_bytes = arm_nn_size_mul(2, rhs_cols);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int16_t));

    return (int32_t)required_bytes;
}

/*
 * Get the required buffer size for arm_convolve_even_s4. A forward, not a copy: the even_s4 kernel stages up to
 * four im2col rows of filter_dims->w * filter_dims->h * input_dims->c int8 elements, which is byte-for-byte the
 * 2 * rhs_cols * sizeof(int16_t) that arm_convolve_s4_get_buffer_size() computes -- an exact fit with zero slack,
 * so the two must move together. The equality is pinned by buffer_size_even_arm_convolve_s4() in the Unity suite.
 *
 * Refer to header file for details.
 */
int32_t arm_convolve_even_s4_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    return arm_convolve_s4_get_buffer_size(input_dims, filter_dims);
}

int32_t arm_convolve_1_x_n_s4_get_buffer_size(const cmsis_nn_conv_params *conv_params,
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
    return arm_convolve_s4_get_buffer_size(input_dims, filter_dims);
#else
    return arm_convolve_1_x_n_s4_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
#endif
}

int32_t arm_convolve_1x1_s4_fast_get_buffer_size(const cmsis_nn_dims *input_dims)
{
    // Dim sanity is validated here so a negative channel count returns -1 on every build target, even though no
    // target actually needs this buffer.
    if (input_dims->c < 0)
    {
        return -1;
    }

    return 0;
}

/*
 * Get the required buffer size for arm_convolve_wrapper_s4. This is the
 * recommended convolve wrapper s4 function.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_convolve_wrapper_s4_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_wrapper_s4_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
#else
    (void)output_dims;
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))
    {
        if (arm_nn_is_convolve_1x1_fast(conv_params))
        {
            return arm_convolve_1x1_s4_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return arm_convolve_s4_get_buffer_size(input_dims, filter_dims);
    }
#endif
}

int32_t arm_convolve_wrapper_s4_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims)

{
    (void)output_dims;
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))
    {
        if (arm_nn_is_convolve_1x1_fast(conv_params))
        {
            return arm_convolve_1x1_s4_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if (arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims))
    {
        return arm_convolve_1_x_n_s4_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
    }
    else
    {
        return arm_convolve_s4_get_buffer_size_mve(input_dims, filter_dims);
    }
}

int32_t arm_convolve_wrapper_s4_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims)
{
    return arm_convolve_wrapper_s4_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
}
/**
 * @} end of GetBufferSizeNNConv group
 */
