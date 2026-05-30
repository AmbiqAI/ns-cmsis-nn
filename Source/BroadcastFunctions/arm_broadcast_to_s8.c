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
 * Title:        arm_broadcast_to_s8.c
 * Description:  Broadcast-to operator for s8 tensors.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#include <string.h>

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Broadcast
 * @{
 */

arm_cmsis_nn_status arm_broadcast_to_s8(const int8_t *input,
                                        const cmsis_nn_broadcast_to_params *params,
                                        int8_t *output)
{
    if (!input || !params || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = params->rank;
    const int32_t *input_shape = params->input_shape;
    const int32_t *output_shape = params->output_shape;

    if (rank < 1 || rank > 8)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Compute output strides and input strides (0 for broadcast dims) */
    int32_t output_strides[8];
    int32_t input_strides[8];

    output_strides[rank - 1] = 1;
    input_strides[rank - 1] = (input_shape[rank - 1] > 1) ? 1 : 0;

    for (int32_t d = rank - 2; d >= 0; d--)
    {
        output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
        input_strides[d] = (input_shape[d] > 1)
                               ? (input_strides[d + 1] * input_shape[d + 1])
                               : 0;
    }

    /* Find innermost contiguous (non-broadcast) dims for memcpy optimization */
    int32_t inner_size = 1;
    int32_t inner_dims = 0;
    for (int32_t d = rank - 1; d >= 0; d--)
    {
        if (input_shape[d] == output_shape[d])
        {
            inner_size *= output_shape[d];
            inner_dims++;
        }
        else
        {
            break;
        }
    }

    /* Compute output size */
    int32_t output_size = 1;
    for (int32_t d = 0; d < rank; d++)
    {
        output_size *= output_shape[d];
    }

    if (inner_size > 1)
    {
        const int32_t num_outer = output_size / inner_size;
        const int32_t outer_rank = rank - inner_dims;

        for (int32_t outer = 0; outer < num_outer; outer++)
        {
            int32_t remainder = outer;
            int32_t in_idx = 0;
            for (int32_t d = 0; d < outer_rank; d++)
            {
                int32_t outer_stride = 1;
                for (int32_t dd = d + 1; dd < outer_rank; dd++)
                {
                    outer_stride *= output_shape[dd];
                }
                int32_t coord = remainder / outer_stride;
                remainder %= outer_stride;
                in_idx += (input_shape[d] > 1 ? coord : 0) * input_strides[d];
            }
            memcpy(output + outer * inner_size, input + in_idx,
                   (size_t)inner_size * sizeof(int8_t));
        }
    }
    else
    {
        /* Element-wise fallback */
        for (int32_t out_idx = 0; out_idx < output_size; out_idx++)
        {
            int32_t remainder = out_idx;
            int32_t in_idx = 0;
            for (int32_t d = 0; d < rank; d++)
            {
                int32_t coord = remainder / output_strides[d];
                remainder %= output_strides[d];
                if (input_shape[d] > 1)
                {
                    in_idx += coord * input_strides[d];
                }
            }
            output[out_idx] = input[in_idx];
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Broadcast group
 */
