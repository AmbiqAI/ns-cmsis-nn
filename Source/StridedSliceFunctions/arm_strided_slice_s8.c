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
 * Title:        arm_strided_slice_s8.c
 * Description:  StridedSlice function for s8 data compatible with TF Lite.
 *
 * $Date:        12 August 2026
 * $Revision:    V.1.1.0
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
 * S8 strided slice function for TensorFlow Lite
 *
 * Refer header file for details.
 *
 */
/* arm_memcpy_s8's byte-count contract coincides with the element count for 1-byte elements. */
ARM_STRIDED_SLICE_DEFINE(arm_strided_slice_s8, int8_t, arm_memcpy_s8)

/**
 * @} end of StridedSlice group
 */
