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
 * Title:        arm_gather_nd_s16.c
 * Description:  Gather_nd operator implementations for s16 tensors.
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


arm_cmsis_nn_status arm_gather_nd_s16(const int16_t *params_data,
                                      const cmsis_nn_dims *params_dims,
                                      const int32_t *indices_data,
                                      const cmsis_nn_dims *indices_dims,
                                      const cmsis_nn_gather_nd_params *params,
                                      int16_t *output_data,
                                      const cmsis_nn_dims *output_dims)
{
    if (!params_data || !indices_data || !output_data || !params_dims || !indices_dims || !params || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if ((params->params_rank < 1) || (params->params_rank > CMSIS_NN_MAX_RANK) || (params->indices_rank < 1) ||
        (params->indices_rank > CMSIS_NN_MAX_RANK))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t batch_dims = params->batch_dims;
    if ((batch_dims < 0) || (batch_dims > params->params_rank) || (batch_dims > (params->indices_rank - 1)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t params_shape[CMSIS_NN_MAX_RANK];
    int32_t indices_shape[CMSIS_NN_MAX_RANK];

    for (int32_t i = 0; i < params->params_rank; ++i)
    {
        const int32_t dim = arm_cmsis_nn_dim_at(params_dims, i);
        if (dim < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        params_shape[i] = dim;
    }

    for (int32_t i = 0; i < params->indices_rank; ++i)
    {
        const int32_t dim = arm_cmsis_nn_dim_at(indices_dims, i);
        if (dim < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        indices_shape[i] = dim;
    }

    const int32_t indices_nd = indices_shape[params->indices_rank - 1];
    if ((indices_nd <= 0) || (indices_nd > (params->params_rank - batch_dims)))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    size_t batch_count = 1;
    for (int32_t i = 0; i < batch_dims; ++i)
    {
        if (params_shape[i] != indices_shape[i])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        batch_count *= (size_t)params_shape[i];
    }

    size_t n_slices_per_batch = 1;
    for (int32_t i = batch_dims; i < (params->indices_rank - 1); ++i)
    {
        n_slices_per_batch *= (size_t)indices_shape[i];
    }

    size_t total_slices = batch_count * n_slices_per_batch;

    size_t slice_size = 1;
    for (int32_t i = batch_dims + indices_nd; i < params->params_rank; ++i)
    {
        slice_size *= (size_t)params_shape[i];
    }

    size_t params_total = 1;
    size_t params_strides[CMSIS_NN_MAX_RANK] = {0};
    for (int32_t i = params->params_rank - 1; i >= 0; --i)
    {
        params_strides[i] = params_total;
        params_total *= (size_t)params_shape[i];
    }

    size_t indices_total = arm_cmsis_nn_shape_product(indices_shape, params->indices_rank);
    if (indices_total != (total_slices * (size_t)indices_nd))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t expected_shape[CMSIS_NN_MAX_RANK] = {1, 1, 1, 1};
    int32_t out_index = 0;
    for (int32_t i = 0; i < batch_dims; ++i)
    {
        expected_shape[out_index++] = indices_shape[i];
    }
    for (int32_t i = batch_dims; i < (params->indices_rank - 1); ++i)
    {
        expected_shape[out_index++] = indices_shape[i];
    }
    for (int32_t i = batch_dims + indices_nd; i < params->params_rank; ++i)
    {
        expected_shape[out_index++] = params_shape[i];
    }

    for (int32_t i = 0; i < out_index; ++i)
    {
        if (arm_cmsis_nn_dim_at(output_dims, i) != expected_shape[i])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    size_t output_total = total_slices * slice_size;
    size_t output_dims_total = arm_cmsis_nn_shape_product(expected_shape, out_index);
    if (output_total != output_dims_total)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (slice_size > UINT32_MAX)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (size_t batch = 0; batch < batch_count; ++batch)
    {
        size_t base_batch_offset = 0;
        if (batch_dims > 0)
        {
            size_t tmp = batch;
            for (int32_t dim = batch_dims - 1; dim >= 0; --dim)
            {
                const size_t coord = tmp % (size_t)params_shape[dim];
                tmp /= (size_t)params_shape[dim];
                base_batch_offset += coord * params_strides[dim];
            }
        }

        for (size_t slice = 0; slice < n_slices_per_batch; ++slice)
        {
            const size_t slice_linear = (batch * n_slices_per_batch + slice) * (size_t)indices_nd;
            size_t from_pos = base_batch_offset;

            for (int32_t j = 0; j < indices_nd; ++j)
            {
                const int32_t index_value = indices_data[slice_linear + (size_t)j];
                if ((index_value < 0) || (index_value >= params_shape[batch_dims + j]))
                {
                    return ARM_CMSIS_NN_ARG_ERROR;
                }
                from_pos += (size_t)index_value * params_strides[batch_dims + j];
            }

            if ((from_pos + slice_size) > params_total)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }

            arm_memcpy_s16(output_data + (batch * n_slices_per_batch + slice) * slice_size,
                           params_data + from_pos,
                           (uint32_t)slice_size);
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
