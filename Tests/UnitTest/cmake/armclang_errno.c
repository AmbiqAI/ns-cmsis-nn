/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* Arm errno codes differ from GNU's; keep their storage domains separate. */
static volatile int arm_errno;
volatile int *__aeabi_errno_addr(void)
{
    return &arm_errno;
}
