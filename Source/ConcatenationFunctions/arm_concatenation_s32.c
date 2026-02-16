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
 * Title:        arm_concatenation_s32.c
 * Description:  Concatenate s32 vectors
 *
 * $Date:        19 February 2025
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Concatenation
 * @{
 */

#define MAX_INPUTS 10

arm_cmsis_nn_status arm_concatenation_s32(const int32_t *const *input_data,
                                          const int32_t inputs_count,
                                          const int32_t *input_concat_dims,
                                          const int32_t axis,
                                          int32_t *output_data,
                                          const int32_t output_dims,
                                          const int32_t *output_shape)
{
    if ((input_data == NULL) || (output_data == NULL) || (output_shape == NULL) || (input_concat_dims == NULL))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if ((inputs_count <= 0) || (output_dims <= 0) || (axis < 0) || (axis >= output_dims))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int64_t concat_dim_sum = 0;
    for (int i = 0; i < inputs_count; i++)
    {
        if (input_concat_dims[i] <= 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        concat_dim_sum += input_concat_dims[i];
    }

    if (concat_dim_sum != output_shape[axis])
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Compute the "outer" size: the product of dimensions before the concat axis.
    int64_t outer_size = 1;
    for (int i = 0; i < axis; i++)
    {
        outer_size *= output_shape[i];
    }
    // Compute the "base inner" size: product of dimensions after the concat axis.
    int64_t base_inner_size = 1;
    for (int i = axis + 1; i < output_dims; i++)
    {
        base_inner_size *= output_shape[i];
    }

    int64_t copy_size[MAX_INPUTS] = {0};
    for (int i = 0; i < inputs_count; i++)
    {
        copy_size[i] = (int64_t)input_concat_dims[i] * base_inner_size;
        if ((copy_size[i] <= 0) || (copy_size[i] > (int64_t)UINT32_MAX))
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }

    // For each outer index...
    for (int64_t k = 0; k < outer_size; k++)
    {
        // For each input tensor...
        for (int i = 0; i < inputs_count; i++)
        {
            const int32_t *in_ptr = input_data[i] + k * copy_size[i];
            arm_memcpy_s32(output_data, in_ptr, (uint32_t)copy_size[i]);
            output_data += copy_size[i];
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Concatenation group
 */
