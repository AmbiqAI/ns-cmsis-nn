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
 * Title:        arm_batch_to_space_nd_s16.c
 * Description:  Convert batch to space for s16 tensors
 *
 * $Date:        06 October 2025
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Reshape
 * @{
 */

/*
 * Convert batch to space for s16 tensors.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_batch_to_space_nd_s16(const int16_t *input_data,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_tile *block_shape,
                                              const cmsis_nn_dims *crop, // n->top, h->left, w->bottom, c->right
                                              int16_t *output_data,
                                              const cmsis_nn_dims *output_dims)
{
    if (!input_data || !output_data || !input_dims || !block_shape || !crop || !output_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    cmsis_nn_dims in8 = *input_dims, out8 = *output_dims;
    in8.c *= 2;
    out8.c *= 2;

    return arm_batch_to_space_nd_s8((const int8_t *)input_data,
                                    &in8,
                                    block_shape,
                                    crop, // cropping in spatial elements; unchanged
                                    (int8_t *)output_data,
                                    &out8);
}

/**
 * @} end of Reshape group
 */
