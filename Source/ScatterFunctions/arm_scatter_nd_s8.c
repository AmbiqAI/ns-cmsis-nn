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
 * Title:        arm_scatter_nd_s8.c
 * Description:  ScatterND operator for s8 tensors.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup ScatterND
 * @{
 */

arm_cmsis_nn_status arm_scatter_nd_s8(const int32_t *indices,
                                      const int8_t *updates,
                                      const cmsis_nn_scatter_nd_params *params,
                                      int8_t *output)
{
    if (!indices || !updates || !params || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t num_updates = params->num_updates;
    const int32_t index_depth = params->index_depth;
    const int32_t slice_size = params->slice_size;
    const int32_t output_size = params->output_size;
    const int32_t *output_strides = params->output_strides;

    if (num_updates < 0 || index_depth < 0 || slice_size < 0 || output_size < 0 ||
        (slice_size == 0 && output_size != 0) || (num_updates && index_depth && !output_strides))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Zero-initialize output */
    arm_memset_s8(output, 0, (uint32_t)output_size);

    if (num_updates == 0)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }

    /* Cache dense strides and extents for the usual tensor ranks. Check each
     * coordinate at the point of use, including when an output aliases indices. */
    enum
    {
        cached_axes = 8
    };
    int32_t strides[cached_axes];
    uint32_t extents[cached_axes];
    if (index_depth <= cached_axes)
    {
        int32_t span = output_size;
        for (int32_t d = 0; d < index_depth; ++d)
        {
            const int32_t stride = output_strides[d];
            if (stride < 0 || ((stride == 0) != (slice_size == 0)))
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            strides[d] = stride;
            extents[d] = stride ? (uint32_t)(span / stride) : (uint32_t)INT32_MAX + 1u;
            span = stride;
        }
        if (slice_size > span)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    for (int32_t i = 0; i < num_updates; ++i)
    {
        int32_t flat_index = 0;
        if (index_depth <= cached_axes)
        {
            for (int32_t d = 0; d < index_depth; ++d)
            {
                const int32_t coord = *indices++;
                if ((uint32_t)coord >= extents[d])
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                flat_index += coord * strides[d];
            }
        }
        else
        {
            /* Deeper tuples consume no additional stack. */
            int32_t remaining_span = output_size;
            for (int32_t d = 0; d < index_depth; ++d)
            {
                const int32_t stride = output_strides[d];
                const int32_t coord = *indices++;
                if (coord < 0 || stride < 0 || ((stride == 0) != (slice_size == 0)))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                const int64_t offset = (int64_t)coord * stride;
                if (offset + stride > remaining_span)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                flat_index += (int32_t)offset;
                remaining_span = stride;
            }
            if (slice_size > remaining_span)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }
        for (int32_t j = 0; j < slice_size; j++)
        {
            output[flat_index + j] += *updates++;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of ScatterND group
 */
