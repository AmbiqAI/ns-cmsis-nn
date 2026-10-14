/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_RESIZE_BILINEAR_H
#define ARM_NN_RESIZE_BILINEAR_H

#include "arm_nnfunctions.h"

static inline int arm_nn_resize_shape_valid(const cmsis_nn_dims *dims)
{
    if (!dims || dims->n <= 0 || dims->h <= 0 || dims->w <= 0 || dims->c <= 0)
    {
        return 0;
    }
    int64_t size = (int64_t)dims->n * dims->h;
    if (size > INT32_MAX)
    {
        return 0;
    }
    size *= dims->w;
    return size <= INT32_MAX && size * dims->c <= INT32_MAX;
}

static inline int arm_nn_resize_bilinear_valid(const cmsis_nn_resize_params *params,
                                               const cmsis_nn_dims *input_dims,
                                               const int8_t *input,
                                               const cmsis_nn_dims *output_dims,
                                               int8_t *output)
{
    return params && input && output && arm_nn_resize_shape_valid(input_dims) &&
        arm_nn_resize_shape_valid(output_dims) && input_dims->n == output_dims->n && input_dims->c == output_dims->c &&
        !(params->align_corners && params->half_pixel_centers);
}

#endif
