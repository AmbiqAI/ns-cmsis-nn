/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_concatenation_f16 (#411): rank-agnostic concatenation, cases in the shared axis-copy template,
 * plus a cross-check against the 4-D per-axis arm_concatenation_f16_{x,y,z,w} kernels. */

#define AC_OP AC_OP_CONCAT
#define AC_F16 1
#define AC_PREFIX concatenation_f16
#define AC_KERNEL arm_concatenation_f16

#include "../Utils/axis_copy_flt_cases.h"
