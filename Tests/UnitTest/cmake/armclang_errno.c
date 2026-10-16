/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <errno.h>

/* Bind Arm's errno accessor to the single-thread GNU test harness. */
volatile int *__aeabi_errno_addr(void)
{
    return &errno;
}
