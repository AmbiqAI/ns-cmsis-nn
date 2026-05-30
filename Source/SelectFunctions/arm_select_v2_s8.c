/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_select_v2_s8.c
 * Description:  SELECT_V2 operator for s8 tensors (with broadcast).
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Select
 * @{
 */

arm_cmsis_nn_status arm_select_v2_s8(const int8_t *condition,
                                     const int8_t *x,
                                     const int8_t *y,
                                     const cmsis_nn_select_v2_params *params,
                                     int8_t *output)
{
    if (!condition || !x || !y || !params || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = params->rank;
    const int32_t *output_shape = params->output_shape;
    const int32_t *cond_strides = params->cond_strides;
    const int32_t *x_strides = params->x_strides;
    const int32_t *y_strides = params->y_strides;

    if (rank < 1 || rank > 8)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Compute output strides */
    int32_t output_strides[8];
    output_strides[rank - 1] = 1;
    for (int32_t d = rank - 2; d >= 0; d--)
    {
        output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
    }

    /* Compute output size */
    int32_t output_size = 1;
    for (int32_t d = 0; d < rank; d++)
    {
        output_size *= output_shape[d];
    }

    for (int32_t out_idx = 0; out_idx < output_size; out_idx++)
    {
        int32_t remainder = out_idx;
        int32_t cond_idx = 0;
        int32_t x_idx = 0;
        int32_t y_idx = 0;

        for (int32_t d = 0; d < rank; d++)
        {
            int32_t coord = remainder / output_strides[d];
            remainder %= output_strides[d];
            cond_idx += coord * cond_strides[d];
            x_idx += coord * x_strides[d];
            y_idx += coord * y_strides[d];
        }

        output[out_idx] = condition[cond_idx] ? x[x_idx] : y[y_idx];
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Select group
 */
