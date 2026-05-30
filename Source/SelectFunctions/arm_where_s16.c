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
 * Title:        arm_where_s16.c
 * Description:  WHERE operator for s16 condition tensors.
 *               Returns indices of non-zero elements.
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

arm_cmsis_nn_status
arm_where_s16(const int16_t *condition, const cmsis_nn_where_params *params, int32_t *output, int32_t *num_true)
{
    if (!condition || !params || !output || !num_true)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = params->rank;
    const int32_t *shape = params->shape;

    if (rank < 1 || rank > 8)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Compute strides */
    int32_t strides[8];
    strides[rank - 1] = 1;
    for (int32_t d = rank - 2; d >= 0; d--)
    {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    /* Compute total size */
    int32_t total_size = 1;
    for (int32_t d = 0; d < rank; d++)
    {
        total_size *= shape[d];
    }

    int32_t count = 0;
    for (int32_t flat = 0; flat < total_size; flat++)
    {
        if (condition[flat])
        {
            int32_t remainder = flat;
            for (int32_t d = 0; d < rank; d++)
            {
                int32_t coord = remainder / strides[d];
                remainder %= strides[d];
                output[count * rank + d] = coord;
            }
            count++;
        }
    }

    *num_true = count;
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Select group
 */
