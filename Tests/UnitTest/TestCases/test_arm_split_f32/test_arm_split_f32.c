/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_split_f32 (#411): rank-agnostic split, cases in the shared axis-copy template. */

#define AC_OP AC_OP_SPLIT
#define AC_F16 0
#define AC_PREFIX split_f32
#define AC_KERNEL arm_split_f32

#include "../Utils/axis_copy_flt_cases.h"
