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
    const int32_t buf_x =
        ((input_dims->w - 1) * transpose_conv_params->stride.w + MAX(filter_dims->w, transpose_conv_params->stride.w)) *
        out_dims->c;
    const int32_t buf_y = MAX(filter_dims->h, transpose_conv_params->stride.h);
    return buf_x * buf_y * (int32_t)sizeof(int32_t);
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

    if (reverse_conv_possible && reverse_conv_efficient)
    {
        const cmsis_nn_dims reverse_conv_input_dims = {input_dims->n,
                                                       input_dims->h * transpose_conv_params->stride.h,
                                                       input_dims->w * transpose_conv_params->stride.w,
                                                       input_dims->c};
        // This is the size arm_transpose_conv_wrapper_s8() needs when it routes to the reverse
        // convolution, but the documented contract of this function is the ctx size for
        // arm_transpose_conv_s8(), which callers are free to invoke directly. Return the larger of
        // the two so neither caller under-allocates (issue #261 defect 3): for in_ch > 16 with
        // both strides <= 2 the reverse-conv size can be smaller than what arm_transpose_conv_s8()
        // indexes in ctx->buf.
        return MAX(arm_convolve_s8_get_buffer_size(&reverse_conv_input_dims, filter_dims), rolling_size);
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

    if (reverse_conv_possible && reverse_conv_efficient)
    {
        const cmsis_nn_dims reverse_conv_input_dims = {input_dims->n,
                                                       input_dims->h * transpose_conv_params->stride.h,
                                                       input_dims->w * transpose_conv_params->stride.w,
                                                       input_dims->c};

        // See arm_transpose_conv_s8_get_buffer_size(): cover the direct arm_transpose_conv_s8()
        // caller as well as the wrapper's reverse-conv route (issue #261 defect 3).
        return MAX(arm_convolve_s8_get_buffer_size_mve(&reverse_conv_input_dims, filter_dims), rolling_size);
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
        return input_dims->c * filter_dims->w * filter_dims->h * filter_dims->n;
    }
    else
    {
        return 0;
    }
}

/**
 * @} end of GetBufferSizeNNConv group
 */
