/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_elementwise_sub_broadcast_f32 (#415): the cases live in the shared template, which
 * checks every output against a double reference computed from a NumPy-style index walk
 * and, in the fuzz case, against arm_elementwise_sub_f32 on the materialised operands. */

#define EW_BCAST_OP EW_BCAST_OP_SUB
#define EW_BCAST_F16 0
#define EW_BCAST_PREFIX sub_broadcast_f32
#define EW_BCAST_KERNEL arm_elementwise_sub_broadcast_f32
#define EW_BCAST_FLAT arm_elementwise_sub_f32

#include "../Utils/elementwise_broadcast_flt_cases.h"
