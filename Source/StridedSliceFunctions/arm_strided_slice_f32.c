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
 * Title:        arm_strided_slice_f32.c
 * Description:  StridedSlice function for float32 data compatible with TF Lite.
 *
 * $Date:        12 August 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_strided_slice_common.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 *  @ingroup Public
 */

/**
 * @addtogroup StridedSlice
 * @{
 */

/*
 * float32 strided slice for TensorFlow Lite. StridedSlice is a pure
 * data-movement op, so the float32 payload is copied as-is.
 *
 * Refer header file for details.
 *
 */
ARM_STRIDED_SLICE_DEFINE(arm_strided_slice_f32, float32_t, arm_memcpy_f32)

/**
 * @} end of StridedSlice group
 */

#endif /* ARM_NN_ENABLE_F32 */
