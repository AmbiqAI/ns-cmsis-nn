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
 * Title:        arm_resize_nearest_neighbor_f32.c
 * Description:  Resize nearest neighbor for float32_t tensors
 *
 * $Date:        6 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

    #include "Internal/arm_resize_nearest_neighbor_common.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup Reshape
 * @{
 */

/* Refer header file for details. */
int32_t arm_resize_nearest_neighbor_f32_get_buffer_size(const cmsis_nn_dims *output_dims)
{
    return arm_nn_resize_nearest_neighbor_scratch_bytes(output_dims);
}

/* Refer header file for details. */
ARM_RESIZE_NEAREST_NEIGHBOR_DEFINE(arm_resize_nearest_neighbor_f32, float32_t, arm_memcpy_f32)

/**
 * @} end of Reshape group
 */

#endif /* ARM_NN_ENABLE_F32 */
