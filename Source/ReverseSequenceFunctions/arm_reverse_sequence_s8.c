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
 * Title:        arm_reverse_sequence_s8.c
 * Description:  ReverseSequence operator for s8 tensors.
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

arm_cmsis_nn_status arm_reverse_sequence_s8(const int8_t *input,
                                            const int32_t *seq_lengths,
                                            const cmsis_nn_reverse_sequence_params *params,
                                            int8_t *output)
{
    if (!input || !seq_lengths || !params || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = params->rank;
    const int32_t *shape = params->shape;
    const int32_t seq_dim = params->seq_dim;
    const int32_t batch_dim = params->batch_dim;

    if (!shape || rank < 1 || rank > 8 || seq_dim < 0 || seq_dim >= rank || batch_dim < 0 || batch_dim >= rank)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    bool empty = false;
    for (int32_t d = 0; d < rank; d++)
    {
        if (shape[d] < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        empty = empty || shape[d] == 0;
    }

    int32_t strides[8];
    int32_t total_size = empty ? 0 : 1;
    for (int32_t d = rank - 1; d >= 0; d--)
    {
        strides[d] = total_size;
        const int64_t product = (int64_t)total_size * shape[d];
        if (product > INT32_MAX)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        total_size = (int32_t)product;
    }

    for (int32_t b = 0; b < shape[batch_dim]; b++)
    {
        if (seq_lengths[b] < 0 || seq_lengths[b] > shape[seq_dim])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
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

        /* Get batch index and seq_length for this batch */
        int32_t batch_idx = coords[batch_dim];
        int32_t seq_len = seq_lengths[batch_idx];
        int32_t seq_coord = coords[seq_dim];

        /* Reverse if within seq_length */
        if (seq_coord < seq_len)
        {
            coords[seq_dim] = seq_len - 1 - seq_coord;
        }

        /* Compute source flat index */
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
