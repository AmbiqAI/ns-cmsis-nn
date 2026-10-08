/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_transpose_conv_get_buffer_sizes_s16.c
 * Description:  Get buffer size functions for the s16 transpose convolution.
 *
 * $Date:        8 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

/**
 *  @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */

/*
 * Get the required buffer size for arm_transpose_conv_s16.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_transpose_conv_s16_get_buffer_size(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *out_dims)
{
    return arm_transpose_conv_s16_get_buffer_size_mve(transpose_conv_params, input_dims, filter_dims, out_dims);
}

int32_t arm_transpose_conv_s16_get_buffer_size_mve(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const cmsis_nn_dims *out_dims)
{
    if ((transpose_conv_params->stride.w <= 0) || (transpose_conv_params->stride.h <= 0) || (input_dims->n < 0) ||
        (input_dims->h < 0) || (input_dims->w < 0) || (input_dims->c < 0) || (filter_dims->h < 0) ||
        (filter_dims->w < 0) || (out_dims->h < 0) || (out_dims->w < 0) || (out_dims->c < 0))
    {
        return -1;
    }

    /* The golden kernel needs no scratch. */
    return 0;
}

/**
 * @} end of GetBufferSizeNNConv group
 */
