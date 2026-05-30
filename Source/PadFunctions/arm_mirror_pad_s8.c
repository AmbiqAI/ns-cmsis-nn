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
 * Title:        arm_mirror_pad_s8.c
 * Description:  Mirror pad operator for s8 tensors.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Pad
 * @{
 */

arm_cmsis_nn_status arm_mirror_pad_s8(const int8_t *input,
                                      const cmsis_nn_mirror_pad_params *params,
                                      int8_t *output)
{
    if (!input || !params || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = params->rank;
    const int32_t *input_shape = params->input_shape;
    const int32_t *output_shape = params->output_shape;
    const int32_t *pad_before = params->pad_before;
    const int32_t mode = params->mode; /* 0=REFLECT, 1=SYMMETRIC */

    if (rank < 1 || rank > 8)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Compute strides */
    int32_t out_strides[8];
    int32_t in_strides[8];

    out_strides[rank - 1] = 1;
    in_strides[rank - 1] = 1;
    for (int32_t d = rank - 2; d >= 0; d--)
    {
        out_strides[d] = out_strides[d + 1] * output_shape[d + 1];
        in_strides[d] = in_strides[d + 1] * input_shape[d + 1];
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
        int32_t in_idx = 0;

        for (int32_t d = 0; d < rank; d++)
        {
            int32_t out_coord = remainder / out_strides[d];
            remainder %= out_strides[d];

            /* Map output coordinate to input coordinate with mirroring */
            int32_t in_coord = out_coord - pad_before[d];
            const int32_t in_dim = input_shape[d];

            if (in_coord < 0)
            {
                if (mode == 0)
                {
                    /* REFLECT: skip boundary */
                    in_coord = -in_coord;
                }
                else
                {
                    /* SYMMETRIC: include boundary */
                    in_coord = -in_coord - 1;
                }
            }
            else if (in_coord >= in_dim)
            {
                if (mode == 0)
                {
                    in_coord = 2 * (in_dim - 1) - in_coord;
                }
                else
                {
                    in_coord = 2 * in_dim - 1 - in_coord;
                }
            }

            in_idx += in_coord * in_strides[d];
        }

        output[out_idx] = input[in_idx];
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Pad group
 */
