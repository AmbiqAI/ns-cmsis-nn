/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_space_to_depth_s16.c
 * Description:  Convert space to depth for s16 tensors
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
 * Convert space to depth for s16 tensors.
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status
arm_space_to_depth_s16(
    const int16_t *input_data,
    const cmsis_nn_dims *input_dims,
    const int32_t block_size,
    int16_t *output_data,
    const cmsis_nn_dims *output_dims
) {
    if (!input_data || !output_data || !input_dims || !output_dims) {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    cmsis_nn_dims in8 = *input_dims, out8 = *output_dims;
    in8.c  *= 2;
    out8.c *= 2;

    return arm_space_to_depth_s8(
        (const int8_t *)input_data,
        &in8,
        block_size,
        (int8_t *)output_data,
        &out8
    );
}

/**
 * @} end of Reshape group
 */
