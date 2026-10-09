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
 * $Revision:    V.1.1.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */

/* Any negative dim or bad stride. */
static int32_t tconv_s16_bad_args(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                  const cmsis_nn_dims *input_dims,
                                  const cmsis_nn_dims *filter_dims,
                                  const cmsis_nn_dims *out_dims)
{
    /* An OR is negative if any operand is. */
    return (transpose_conv_params->stride.w <= 0) || (transpose_conv_params->stride.h <= 0) ||
        (input_dims->n | input_dims->h | input_dims->w | input_dims->c | filter_dims->n | filter_dims->h |
         filter_dims->w | filter_dims->c | out_dims->n | out_dims->h | out_dims->w | out_dims->c) < 0;
}

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
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    return arm_transpose_conv_s16_get_buffer_size_mve(transpose_conv_params, input_dims, filter_dims, out_dims);
#else
    return tconv_s16_bad_args(transpose_conv_params, input_dims, filter_dims, out_dims) ? -1 : 0;
#endif
}

int32_t arm_transpose_conv_s16_get_buffer_size_mve(const cmsis_nn_transpose_conv_params *transpose_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const cmsis_nn_dims *out_dims)
{
    if (tconv_s16_bad_args(transpose_conv_params, input_dims, filter_dims, out_dims))
    {
        return -1;
    }

    /* Zero when the packed path cannot run. */
    const int32_t col_max = arm_nn_transpose_conv_s16_col(transpose_conv_params, input_dims, filter_dims, out_dims);
    if (col_max == 0)
    {
        return 0;
    }

    /* Slack covers alignment and pointer steps. */
    const int32_t cap = 2048;
    const int32_t fixed = 32 + col_max * 2;
    const int32_t per_ch = 8 + col_max;
    int32_t group = ((cap - fixed) / per_ch) & ~3;
    if (out_dims->c < group)
    {
        group = (out_dims->c + 3) & ~3;
    }
    return group >= 4 ? fixed + group * per_ch : 0;
}

/**
 * @} end of GetBufferSizeNNConv group
 */
