/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_transpose_conv_get_buffer_sizes_s16.c
 * Description:  Buffer-size query functions for arm_transpose_conv_s16.
 *
 * $Date:        07 April 2026
 * $Revision:    V.1.0.0
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
 * Get the required scratch buffer size for arm_transpose_conv_s16.
 *
 * The rolling int64 buffer has dimensions:
 *   buf_y         = MAX(filter_h, stride_h)
 *   buf_x_elems   = (input_w - 1) * stride_w + MAX(filter_w, stride_w)
 *   buf_x         = buf_x_elems * out_ch   (int64 elements per row)
 *   total bytes   = buf_y * buf_x * sizeof(int64_t)
 *
 * Refer to header file for parameter details.
 */
int32_t arm_transpose_conv_s16_get_buffer_size(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *out_dims)
{
    const int32_t buf_x =
        ((input_dims->w - 1) * transpose_conv_params->stride.w +
         MAX(filter_dims->w, transpose_conv_params->stride.w)) *
        out_dims->c;

    const int32_t buf_y = MAX(filter_dims->h, transpose_conv_params->stride.h);

    return buf_x * buf_y * (int32_t)sizeof(int64_t);
}

/**
 * @} end of GetBufferSizeNNConv group
 */
