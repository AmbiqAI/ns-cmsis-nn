/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_concatenation_f32 (#411): rank-agnostic concatenation, cases in the shared axis-copy template,
 * plus a cross-check against the 4-D per-axis arm_concatenation_f32_{x,y,z,w} kernels. */

#define AC_OP AC_OP_CONCAT
#define AC_F16 0
#define AC_PREFIX concatenation_f32
#define AC_KERNEL arm_concatenation_f32

#include "../Utils/axis_copy_flt_cases.h"
