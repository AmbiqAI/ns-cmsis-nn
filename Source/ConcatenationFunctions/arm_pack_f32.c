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
 * Title:        arm_pack_f32.c
 * Description:  Stack float32_t tensors along a new axis (any rank)
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

arm_cmsis_nn_status arm_pack_f32(const float32_t *const *input_data,
                                 const int32_t num_inputs,
                                 const int32_t input_dims,
                                 const int32_t *input_shape,
                                 const int32_t axis,
                                 float32_t *output_data)
{
    int32_t outer;
    int32_t inner;
    /* The new axis sits between shape[0..axis) and shape[axis..dims): inner_begin == axis. */
    if (input_dims < 0 || axis < 0 || axis > input_dims ||
        arm_nn_axis_copy_plan(input_shape, input_dims, axis, axis, num_inputs, num_inputs, NULL, &outer, &inner) !=
            ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t total = outer * num_inputs * inner;
    if ((output_data == NULL && total != 0) ||
        !arm_nn_axis_copy_ptrs_ok((const void *const *)input_data, num_inputs, total))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    arm_nn_axis_gather_f32(input_data, outer, num_inputs, NULL, inner, output_data);
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Concatenation group
 */

#endif /* ARM_NN_ENABLE_F32 */
