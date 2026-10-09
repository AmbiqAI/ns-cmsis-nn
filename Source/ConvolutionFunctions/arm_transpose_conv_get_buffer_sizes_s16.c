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
    return (transpose_conv_params->stride.w <= 0) || (transpose_conv_params->stride.h <= 0) || (input_dims->n < 0) ||
        (input_dims->h < 0) || (input_dims->w < 0) || (input_dims->c < 0) || (filter_dims->n < 0) ||
        (filter_dims->h < 0) || (filter_dims->w < 0) || (filter_dims->c < 0) || (out_dims->n < 0) ||
        (out_dims->h < 0) || (out_dims->w < 0) || (out_dims->c < 0);
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

    /* Ceil division without overflow. */
    const int32_t stride_y = transpose_conv_params->stride.h;
    const int32_t stride_x = transpose_conv_params->stride.w;
    const int32_t taps_y = filter_dims->h / stride_y + (filter_dims->h % stride_y != 0);
    const int32_t taps_x = filter_dims->w / stride_x + (filter_dims->w % stride_x != 0);

    /* Only depth 1 to 255 uses scratch. */
    if ((uint64_t)((int64_t)taps_y * taps_x * input_dims->c - 1) >= 255)
    {
        return 0;
    }

    /* Slack covers alignment and pointer steps. */
    const int32_t cap = 2048;
    const int32_t col_max = taps_y * ((taps_x * input_dims->c + 7) & ~7);
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
