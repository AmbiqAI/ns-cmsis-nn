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
 * Title:        arm_strided_slice_s32.c
 * Description:  StridedSlice function for s32 data compatible with TF Lite.
 *
 * $Date:        12 August 2026
 * $Revision:    V.2.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_strided_slice_common.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup StridedSlice
 * @{
 */

/*
 * S32 strided slice function for TensorFlow Lite
 *
 * Refer header file for details.
 *
 */
ARM_STRIDED_SLICE_DEFINE(arm_strided_slice_s32, int32_t, arm_memcpy_s32)

/**
 * @} end of StridedSlice group
 */
