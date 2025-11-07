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
 * Title:        arm_gather_s16.c
 * Description:  Gather operator implementations for s16 tensors.
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Gather
 * @{
 */

#define CMSIS_NN_MAX_RANK (4)

static inline int32_t cmsis_nn_dim_at(const cmsis_nn_dims *dims, int32_t index)
{
    switch (index)
    {
        case 0:
            return dims->n;
        case 1:
            return dims->h;
        case 2:
            return dims->w;
        case 3:
            return dims->c;
        default:
            return 1;
    }
}

static inline size_t cmsis_nn_shape_product(const int32_t *shape, int32_t length)
{
    size_t product = 1;
    for (int32_t i = 0; i < length; ++i)
    {
        product *= (size_t)shape[i];
    }
    return product;
}


arm_cmsis_nn_status arm_gather_s16(const int16_t *input_data,
                                   const cmsis_nn_dims *input_dims,
                                   const int32_t *indices_data,
                                   const cmsis_nn_dims *indices_dims,
                                   const cmsis_nn_gather_params *params,
                                   int16_t *output_data,
                                   const cmsis_nn_dims *output_dims)
{
    if (!input_data || !indices_data || !output_data || !input_dims || !indices_dims || !params || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if ((params->input_rank < 1) || (params->input_rank > CMSIS_NN_MAX_RANK) || (params->coords_rank < 1) ||
        (params->coords_rank > CMSIS_NN_MAX_RANK))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t axis = params->axis;
    if (axis < 0)
    {
        axis += params->input_rank;
    }
    if ((axis < 0) || (axis >= params->input_rank))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t batch_dims = params->batch_dims;
    if (batch_dims < 0)
    {
        batch_dims += params->coords_rank;
    }
    if ((batch_dims < 0) || (batch_dims > params->coords_rank) || (batch_dims > axis))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t input_shape[CMSIS_NN_MAX_RANK];
    int32_t coords_shape[CMSIS_NN_MAX_RANK];

    for (int32_t i = 0; i < params->input_rank; ++i)
    {
        const int32_t dim = cmsis_nn_dim_at(input_dims, i);
        if (dim < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        input_shape[i] = dim;
    }

    for (int32_t i = 0; i < params->coords_rank; ++i)
    {
        const int32_t dim = cmsis_nn_dim_at(indices_dims, i);
        if (dim < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        coords_shape[i] = dim;
    }

    for (int32_t i = 0; i < batch_dims; ++i)
    {
        if (input_shape[i] != coords_shape[i])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    const int32_t axis_size = input_shape[axis];
    if (axis_size < 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    size_t batch_size = 1;
    for (int32_t i = 0; i < batch_dims; ++i)
    {
        batch_size *= (size_t)input_shape[i];
    }

    size_t outer_size = 1;
    for (int32_t i = batch_dims; i < axis; ++i)
    {
        outer_size *= (size_t)input_shape[i];
    }

    size_t inner_size = 1;
    for (int32_t i = axis + 1; i < params->input_rank; ++i)
    {
        inner_size *= (size_t)input_shape[i];
    }

    size_t coord_size = 1;
    for (int32_t i = batch_dims; i < params->coords_rank; ++i)
    {
        coord_size *= (size_t)coords_shape[i];
    }

    size_t indices_total = 1;
    for (int32_t i = 0; i < params->coords_rank; ++i)
    {
        indices_total *= (size_t)coords_shape[i];
    }

    const size_t expected_indices_total = batch_size * coord_size;
    if (indices_total != expected_indices_total)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    size_t input_total = cmsis_nn_shape_product(input_shape, params->input_rank);

    int32_t expected_rank = 0;
    int32_t expected_shape[CMSIS_NN_MAX_RANK] = {1, 1, 1, 1};

    for (int32_t i = 0; i < axis; ++i)
    {
        expected_shape[expected_rank++] = input_shape[i];
    }
    for (int32_t i = batch_dims; i < params->coords_rank; ++i)
    {
        expected_shape[expected_rank++] = coords_shape[i];
    }
    for (int32_t i = axis + 1; i < params->input_rank; ++i)
    {
        expected_shape[expected_rank++] = input_shape[i];
    }

    if (expected_rank > CMSIS_NN_MAX_RANK)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int32_t i = 0; i < expected_rank; ++i)
    {
        if (cmsis_nn_dim_at(output_dims, i) != expected_shape[i])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    size_t output_total = batch_size * outer_size * coord_size * inner_size;
    size_t output_dims_total = cmsis_nn_shape_product(expected_shape, expected_rank);
    if (output_total != output_dims_total)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (inner_size > UINT32_MAX)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        const size_t coord_batch_offset = batch * coord_size;
        for (size_t outer = 0; outer < outer_size; ++outer)
        {
            const size_t output_base = ((batch * outer_size) + outer) * coord_size * inner_size;
            const size_t input_base = ((batch * outer_size) + outer) * (size_t)axis_size * inner_size;

            for (size_t coord = 0; coord < coord_size; ++coord)
            {
                const size_t coord_index = coord_batch_offset + coord;
                const int32_t idx = indices_data[coord_index];
                if ((idx < 0) || (idx >= axis_size))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }

                const size_t src_offset = input_base + ((size_t)idx * inner_size);
                const size_t dst_offset = output_base + (coord * inner_size);

                if ((src_offset + inner_size) > input_total)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                if ((dst_offset + inner_size) > output_total)
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }

                arm_memcpy_s16(output_data + dst_offset, input_data + src_offset, (uint32_t)inner_size);
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
