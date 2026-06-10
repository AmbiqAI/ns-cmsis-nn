/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_select_v2_s8.c"
#include "unity.h"

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

void setUp(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
}

void tearDown(void) {}

void test_select_v2_broadcast_s8_arm_select_v2_s8(void) { select_v2_broadcast_s8_arm_select_v2_s8(); }
void test_select_v2_scalar_condition_true_s8_arm_select_v2_s8(void) { select_v2_scalar_condition_true_s8_arm_select_v2_s8(); }
void test_select_v2_row_mask_s8_arm_select_v2_s8(void) { select_v2_row_mask_s8_arm_select_v2_s8(); }