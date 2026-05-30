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
 * Title:        arm_reverse_sequence_s16.c
 * Description:  ReverseSequence operator for s16 tensors.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup ReverseSequence
 * @{
 */

arm_cmsis_nn_status arm_reverse_sequence_s16(const int16_t *input,
                                             const int32_t *seq_lengths,
                                             const cmsis_nn_reverse_sequence_params *params,
                                             int16_t *output)
{
    if (!input || !seq_lengths || !params || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = params->rank;
    const int32_t *shape = params->shape;
    const int32_t seq_dim = params->seq_dim;
    const int32_t batch_dim = params->batch_dim;

    if (rank < 1 || rank > 8)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t strides[8];
    strides[rank - 1] = 1;
    for (int32_t d = rank - 2; d >= 0; d--)
    {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    int32_t total_size = 1;
    for (int32_t d = 0; d < rank; d++)
    {
        total_size *= shape[d];
    }

    for (int32_t dst_idx = 0; dst_idx < total_size; dst_idx++)
    {
        int32_t remainder = dst_idx;
        int32_t coords[8];

        for (int32_t d = 0; d < rank; d++)
        {
            coords[d] = remainder / strides[d];
            remainder %= strides[d];
        }

        int32_t batch_idx = coords[batch_dim];
        int32_t seq_len = seq_lengths[batch_idx];
        int32_t seq_coord = coords[seq_dim];

        if (seq_coord < seq_len)
        {
            coords[seq_dim] = seq_len - 1 - seq_coord;
        }

        int32_t src_idx = 0;
        for (int32_t d = 0; d < rank; d++)
        {
            src_idx += coords[d] * strides[d];
        }

        output[dst_idx] = input[src_idx];
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of ReverseSequence group
 */
