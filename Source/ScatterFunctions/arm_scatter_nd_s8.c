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

#include <string.h>

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

    /* Zero-initialize output */
    memset(output, 0, (size_t)output_size * sizeof(int8_t));

    /* Scatter updates */
    for (int32_t i = 0; i < num_updates; i++)
    {
        int32_t flat_index = 0;
        for (int32_t d = 0; d < index_depth; d++)
        {
            flat_index += indices[i * index_depth + d] * output_strides[d];
        }
        memcpy(output + flat_index, updates + i * slice_size,
               (size_t)slice_size * sizeof(int8_t));
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of ScatterND group
 */
