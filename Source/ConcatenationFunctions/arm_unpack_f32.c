/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_unpack_f32.c
 * Description:  Unpack a float32_t tensor into unit slices along one axis (any rank)
 *
 * $Date:        6 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_axis_copy_common.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Concatenation
 * @{
 */

arm_cmsis_nn_status arm_unpack_f32(const float32_t *input_data,
                                   const int32_t input_dims,
                                   const int32_t *input_shape,
                                   const int32_t axis,
                                   float32_t *const *output_data)
{
    int32_t outer;
    int32_t inner;
    if (input_dims < 1 || axis < 0 || axis >= input_dims || input_shape == NULL || input_shape[axis] < 1 ||
        arm_nn_axis_copy_plan(
            input_shape, input_dims, axis, axis + 1, input_shape[axis], input_shape[axis], NULL, &outer, &inner) !=
            ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t total = outer * input_shape[axis] * inner;
    if ((input_data == NULL && total != 0) ||
        !arm_nn_axis_copy_ptrs_ok((const void *const *)output_data, input_shape[axis], total))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_nn_axis_scatter_f32(input_data, outer, input_shape[axis], NULL, inner, output_data);
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Concatenation group
 */

#endif /* ARM_NN_ENABLE_F32 */
