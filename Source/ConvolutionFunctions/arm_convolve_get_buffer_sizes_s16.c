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
 * Title:        arm_convolve_get_buffer_sizes_s16.c
 * Description:  Collection of get buffer size functions for the various s16 convolution layer functions.
 *
 * $Date:        20 March 2024
 * $Revision:    V.2.0.0
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

__STATIC_INLINE int32_t arm_convolve_s16_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
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

    // Get number of complete lanes with int16 elements (multiple of 8) for given col_length. This is dependent on
    // implementation of arm_nn_mat_mult_nt_t_s16
    const int64_t col_length = (col_elements + 7) / 8;
    // 4 -> number of im2col buffers, 8 -> 8 elements per Q register
    int64_t required_bytes = arm_nn_size_mul(4, col_length);
    required_bytes = arm_nn_size_mul(required_bytes, 8);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int16_t));

    return (int32_t)required_bytes;
}

int32_t arm_convolve_s16_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    // Dim sanity is validated once here so an invalid dim returns -1 on every build target. The byte count is
    // checked per leg, since the two legs use different formulas.
    if ((input_dims->c < 0) || (filter_dims->w < 0) || (filter_dims->h < 0))
    {
        return -1;
    }

#if defined(ARM_MATH_MVEI)
    return arm_convolve_s16_get_buffer_size_mve(input_dims, filter_dims);
#else
    int64_t required_bytes = arm_nn_size_mul(2, input_dims->c);
    required_bytes = arm_nn_size_mul(required_bytes, filter_dims->w);
    required_bytes = arm_nn_size_mul(required_bytes, filter_dims->h);
    required_bytes = arm_nn_size_mul(required_bytes, (int32_t)sizeof(int16_t));

    return (int32_t)required_bytes;
#endif
}

/*
 * Get the required buffer size for arm_convolve_wrapper_s16. This is the recommended function convolve wrapper s16
 * function.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_convolve_wrapper_s16_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims)
{
    (void)conv_params;
    (void)output_dims;

    return arm_convolve_s16_get_buffer_size(input_dims, filter_dims);
}

int32_t arm_convolve_wrapper_s16_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims)
{
    return arm_convolve_wrapper_s16_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
}

int32_t arm_convolve_wrapper_s16_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims)
{
    (void)conv_params;
    (void)output_dims;

    return arm_convolve_s16_get_buffer_size_mve(input_dims, filter_dims);
}

/**
 * @} end of GetBufferSizeNNConv group
 */
