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
 * Title:        arm_depth_to_space_s16.c
 * Description:  Convert depth to space for s16 tensors
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
 * Convert depth to space for s16 tensors.
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status
arm_depth_to_space_s16(
    const int16_t *input_data,
    const cmsis_nn_dims *input_dims,
    const int32_t block_size,
    int16_t *output_data,
    const cmsis_nn_dims *output_dims
) {
    if (!input_data || !output_data || !input_dims || !output_dims) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Reuse the s8 kernel by operating in BYTES: C' = C * 2
    cmsis_nn_dims in8  = *input_dims;
    cmsis_nn_dims out8 = *output_dims;
    in8.c  = (int32_t)in8.c  * 2;   // bytes/channel for int16
    out8.c = (int32_t)out8.c * 2;

    return arm_depth_to_space_s8(
        (const int8_t *)input_data,
        &in8,
        block_size, // NOT scaled
        (int8_t *)output_data,
        &out8
    );
}

/**
 * @} end of Reshape group
 */
