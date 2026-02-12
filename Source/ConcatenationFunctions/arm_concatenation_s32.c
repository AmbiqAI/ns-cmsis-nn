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

arm_cmsis_nn_status arm_concatenation_s32(const int32_t *const *input_data,
                                          const int32_t inputs_count,
                                          const int32_t *input_concat_dims,
                                          const int32_t axis,
                                          int32_t *output_data,
                                          const int32_t output_dims,
                                          const int32_t *output_shape)
{

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

    // For each outer index...
    for (int k = 0; k < outer_size; k++)
    {
        // For each input tensor...
        for (int i = 0; i < inputs_count; i++)
        {
            // The number of elements to copy from this input.
            const int copy_size = input_concat_dims[i] * base_inner_size;
            const int32_t *in_ptr = input_data[i] + k * copy_size;
            arm_memcpy_s32(output_data, in_ptr, copy_size);
            output_data += copy_size;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Concatenation group
 */
