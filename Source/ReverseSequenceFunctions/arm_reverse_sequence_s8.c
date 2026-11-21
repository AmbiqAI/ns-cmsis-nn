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

    if (!shape || rank < 1 || rank > 8 || seq_dim < 0 || seq_dim >= rank || batch_dim < 0 || batch_dim >= rank ||
        seq_dim == batch_dim)
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

    const int32_t batches = shape[batch_dim];
    const int32_t sequence_size = shape[seq_dim];
    for (int32_t b = 0; b < batches; b++)
    {
        if (seq_lengths[b] < 0 || seq_lengths[b] > sequence_size)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    /* Coordinates after both indexed axes form contiguous, unchanged slices. */
    const int32_t seq_stride = strides[seq_dim];
    const int32_t batch_stride = strides[batch_dim];
    const int32_t slice = strides[seq_dim > batch_dim ? seq_dim : batch_dim];
    for (int32_t dst = 0; dst < total_size; dst += slice)
    {
        const int32_t batch = (dst / batch_stride) % batches;
        const int32_t pos = (dst / seq_stride) % sequence_size;
        const int32_t length = seq_lengths[batch];
        /* Output may share storage with lengths already checked above. */
        if ((uint32_t)length > (uint32_t)sequence_size)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        const int32_t src = pos < length ? dst + (length - 1 - pos - pos) * seq_stride : dst;
        if (slice == 1)
        {
            output[dst] = input[src];
        }
        else
        {
            memmove(output + dst, input + src, (size_t)slice);
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of ReverseSequence group
 */
