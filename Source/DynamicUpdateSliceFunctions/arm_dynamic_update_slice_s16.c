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
 * Title:        arm_dynamic_update_slice_s16.c
 * Description:  DynamicUpdateSlice operator for s16 tensors.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup DynamicUpdateSlice
 * @{
 */

arm_cmsis_nn_status arm_dynamic_update_slice_s16(const int16_t *operand,
                                                 const int16_t *update,
                                                 const int32_t *start_indices,
                                                 const cmsis_nn_dynamic_update_slice_params *params,
                                                 int16_t *output)
{
    if (!operand || !update || !start_indices || !params || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t rank = params->rank;
    const int32_t *operand_shape = params->operand_shape;
    const int32_t *update_shape = params->update_shape;
    const int32_t operand_size = params->operand_size;
    const int32_t update_size = params->update_size;
    const int32_t *operand_strides = params->operand_strides;

    if (rank < 1 || rank > 8)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    arm_memcpy_s16(output, operand, (uint32_t)operand_size);

    if (update_size == 0)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }

    int32_t clamped_starts[8];
    for (int32_t d = 0; d < rank; d++)
    {
        int32_t max_start = operand_shape[d] - update_shape[d];
        int32_t s = start_indices[d];
        if (s < 0)
        {
            s = 0;
        }
        if (s > max_start)
        {
            s = max_start;
        }
        clamped_starts[d] = s;
    }

    const int32_t inner_dim = update_shape[rank - 1];
    const int32_t num_rows = update_size / inner_dim;

    for (int32_t row = 0; row < num_rows; row++)
    {
        int32_t remainder = row;
        int32_t out_offset = 0;
        for (int32_t d = 0; d < rank - 1; d++)
        {
            int32_t row_stride = 1;
            for (int32_t dd = d + 1; dd < rank - 1; dd++)
            {
                row_stride *= update_shape[dd];
            }
            int32_t upd_coord = remainder / row_stride;
            remainder %= row_stride;
            out_offset += (clamped_starts[d] + upd_coord) * operand_strides[d];
        }
        out_offset += clamped_starts[rank - 1];
        arm_memcpy_s16(output + out_offset, update + row * inner_dim, (uint32_t)inner_dim);
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of DynamicUpdateSlice group
 */
