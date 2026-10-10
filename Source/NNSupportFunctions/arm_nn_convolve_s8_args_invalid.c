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
 * Title:        arm_nn_convolve_s8_args_invalid.c
 * Description:  Argument check shared by arm_convolve_s8() and its direct entries.
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

int32_t arm_nn_convolve_s8_args_invalid(const cmsis_nn_conv_params *conv_params,
                                        const cmsis_nn_dims *input_dims,
                                        const cmsis_nn_dims *filter_dims,
                                        const cmsis_nn_dims *output_dims)
{
    const uint32_t all = (uint32_t)(input_dims->c | output_dims->c | input_dims->w | input_dims->h | filter_dims->w |
                                    filter_dims->h | output_dims->w | output_dims->h | conv_params->padding.w |
                                    conv_params->padding.h | conv_params->stride.w | conv_params->stride.h);
    /* Formed wide enough not to overflow for any input; used only once every value above fits 16 bits */
    const uint64_t patch = (uint64_t)((uint32_t)filter_dims->w * (uint32_t)filter_dims->h) * (uint32_t)filter_dims->c;
    const int64_t reach_w = ((int64_t)filter_dims->w - 1) * conv_params->dilation.w;
    const int64_t reach_h = ((int64_t)filter_dims->h - 1) * conv_params->dilation.h;
    return all > UINT16_MAX || arm_nn_convolve_groups_invalid(input_dims, filter_dims, output_dims) ||
        patch > INT32_MAX / 4 || patch * (uint32_t)output_dims->c > INT32_MAX ||
        (uint64_t)((uint32_t)input_dims->w * (uint32_t)input_dims->h) * (uint32_t)input_dims->c > INT32_MAX ||
        (uint64_t)((uint32_t)output_dims->w * (uint32_t)output_dims->h) * (uint32_t)output_dims->c > INT32_MAX ||
        (uint32_t)output_dims->w * (uint32_t)conv_params->stride.w > INT32_MAX / 2 ||
        (uint32_t)output_dims->h * (uint32_t)conv_params->stride.h > INT32_MAX / 2 || reach_w > INT32_MAX / 2 ||
        reach_w < -(INT32_MAX / 2) || reach_h > INT32_MAX / 2 || reach_h < -(INT32_MAX / 2);
}

/**
 * @} end of supportConvolution group
 */
