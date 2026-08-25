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
 * Title:        arm_transpose_conv_get_buffer_sizes_s8.c
 * Description:  Collection of get buffer size functions for the transpose convolution layer functions.
 *
 * $Date:        9 Mars 2026
 * $Revision:    V.2.1.0
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

/*
 * Scratch size indexed by arm_transpose_conv_s8() itself: the rolling row buffer,
 * MAX(filter_y, stride_y) rows of ((input_x - 1) * stride_x + MAX(filter_x, stride_x)) * out_ch
 * int32 accumulators.
 */
static int32_t transpose_conv_s8_rolling_buffer_size(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *out_dims)
{
    if ((input_dims->w < 0) || (out_dims->c < 0) || (filter_dims->w < 0) || (filter_dims->h < 0) ||
        (transpose_conv_params->stride.w <= 0) || (transpose_conv_params->stride.h <= 0))
    {
        return -1;
    }

    // (w - 1) * stride_w + MAX(filter_w, stride_w) is bounded by ~2^62 and so cannot wrap an int64_t, but it can be
    // negative when w == 0, which upstream also treated as a valid (zero-row) geometry. Compute it directly and only
    // then start the bounded fold, which needs a non-negative accumulator. See arm_nn_size_mul().
    const int64_t row_span = ((int64_t)input_dims->w - 1) * (int64_t)transpose_conv_params->stride.w +
        MAX(filter_dims->w, transpose_conv_params->stride.w);

    if (row_span < 0)
    {
        return -1;
    }

    int64_t buf_x = arm_nn_size_mul(row_span, out_dims->c);

    const int32_t buf_y = MAX(filter_dims->h, transpose_conv_params->stride.h);

    int64_t required_bytes = arm_nn_size_mul(buf_x, buf_y);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int32_t));

    return (int32_t)required_bytes;
}

/*
 * Build the reverse-convolution input dims. The strided geometry is computed in 64 bits because
 * input_dims->{h,w} * stride is signed-overflow UB when left in int32_t. Returns -1 if the result
 * does not fit, 0 otherwise.
 */
static int32_t transpose_conv_s8_reverse_conv_input_dims(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                         const cmsis_nn_dims *input_dims,
                                                         cmsis_nn_dims *reverse_conv_input_dims)
{
    const int64_t strided_h = (int64_t)input_dims->h * (int64_t)transpose_conv_params->stride.h;
    const int64_t strided_w = (int64_t)input_dims->w * (int64_t)transpose_conv_params->stride.w;

    if ((input_dims->h < 0) || (input_dims->w < 0) || (strided_h > INT32_MAX) || (strided_w > INT32_MAX))
    {
        return -1;
    }

    reverse_conv_input_dims->n = input_dims->n;
    reverse_conv_input_dims->h = (int32_t)strided_h;
    reverse_conv_input_dims->w = (int32_t)strided_w;
    reverse_conv_input_dims->c = input_dims->c;

    return 0;
}

/*
 * Get the required buffer size for arm_transpose_conv_s8. This is the recommended transpose conv s8 get buffer size
 * function.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_transpose_conv_s8_get_buffer_size(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *out_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_transpose_conv_s8_get_buffer_size_mve(transpose_conv_params, input_dims, filter_dims, out_dims);
#else
    const bool reverse_conv_possible =
        ((transpose_conv_params->stride.w <= 2) && (transpose_conv_params->stride.h <= 2));
    const bool reverse_conv_efficient = (input_dims->c > REVERSE_TCOL_EFFICIENT_THRESHOLD);

    const int32_t rolling_size =
        transpose_conv_s8_rolling_buffer_size(transpose_conv_params, input_dims, filter_dims, out_dims);

    if (rolling_size < 0)
    {
        return -1;
    }

    if (reverse_conv_possible && reverse_conv_efficient)
    {
        cmsis_nn_dims reverse_conv_input_dims;

        if (transpose_conv_s8_reverse_conv_input_dims(transpose_conv_params, input_dims, &reverse_conv_input_dims) < 0)
        {
            return -1;
        }

        // This is the size arm_transpose_conv_wrapper_s8() needs when it routes to the reverse
        // convolution, but the documented contract of this function is the ctx size for
        // arm_transpose_conv_s8(), which callers are free to invoke directly. Return the larger of
        // the two so neither caller under-allocates (issue #261 defect 3): for in_ch > 16 with
        // both strides <= 2 the reverse-conv size can be smaller than what arm_transpose_conv_s8()
        // indexes in ctx->buf.
        const int32_t reverse_conv_size = arm_convolve_s8_get_buffer_size(&reverse_conv_input_dims, filter_dims);

        // Propagate the out-of-range sentinel before the MAX(): MAX(-1, rolling_size) would otherwise collapse to a
        // plausible positive size and the caller would under-allocate.
        if (reverse_conv_size < 0)
        {
            return -1;
        }

        return MAX(reverse_conv_size, rolling_size);
    }
    else
    {
        return rolling_size;
    }
#endif
}
int32_t arm_transpose_conv_s8_get_buffer_size_mve(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const cmsis_nn_dims *out_dims)
{

    const bool reverse_conv_possible =
        ((transpose_conv_params->stride.w <= 2) && (transpose_conv_params->stride.h <= 2));
    const bool reverse_conv_efficient = (input_dims->c > REVERSE_TCOL_EFFICIENT_THRESHOLD);

    const int32_t rolling_size =
        transpose_conv_s8_rolling_buffer_size(transpose_conv_params, input_dims, filter_dims, out_dims);

    if (rolling_size < 0)
    {
        return -1;
    }

    if (reverse_conv_possible && reverse_conv_efficient)
    {
        cmsis_nn_dims reverse_conv_input_dims;

        if (transpose_conv_s8_reverse_conv_input_dims(transpose_conv_params, input_dims, &reverse_conv_input_dims) < 0)
        {
            return -1;
        }

        // See arm_transpose_conv_s8_get_buffer_size(): cover the direct arm_transpose_conv_s8()
        // caller as well as the wrapper's reverse-conv route (issue #261 defect 3).
        const int32_t reverse_conv_size = arm_convolve_s8_get_buffer_size_mve(&reverse_conv_input_dims, filter_dims);

        // Propagate the out-of-range sentinel before the MAX(), or it collapses into a plausible positive size.
        if (reverse_conv_size < 0)
        {
            return -1;
        }

        return MAX(reverse_conv_size, rolling_size);
    }
    else
    {
        return rolling_size;
    }
}

int32_t arm_transpose_conv_s8_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims)
{
    const bool reverse_conv_possible =
        ((transpose_conv_params->stride.w <= 2) && (transpose_conv_params->stride.h <= 2));
    const bool reverse_conv_efficient = (input_dims->c > REVERSE_TCOL_EFFICIENT_THRESHOLD);

    if (reverse_conv_possible && reverse_conv_efficient)
    {
        int64_t required_elements = arm_nn_size_mul(1, input_dims->c);
        required_elements = arm_nn_size_mul(required_elements, filter_dims->w);
        required_elements = arm_nn_size_mul(required_elements, filter_dims->h);
        required_elements = arm_nn_size_mul(required_elements, filter_dims->n);

        return (int32_t)required_elements;
    }
    else
    {
        return 0;
    }
}

/**
 * @} end of GetBufferSizeNNConv group
 */
