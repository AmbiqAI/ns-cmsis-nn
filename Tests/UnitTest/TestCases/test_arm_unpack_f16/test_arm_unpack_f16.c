/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_unpack_f16 (#411): rank-agnostic unpack, cases in the shared axis-copy template. */

#define AC_OP AC_OP_UNPACK
#define AC_F16 1
#define AC_PREFIX unpack_f16
#define AC_KERNEL arm_unpack_f16

#include "../Utils/axis_copy_flt_cases.h"
