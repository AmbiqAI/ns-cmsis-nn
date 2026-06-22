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

#include "../test_arm_where_s16.c"
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

void test_where_2d_s16_arm_where_s16(void) { where_2d_s16_arm_where_s16(); }
void test_where_all_true_2d_s16_arm_where_s16(void) { where_all_true_2d_s16_arm_where_s16(); }
void test_where_3d_sparse_s16_arm_where_s16(void) { where_3d_sparse_s16_arm_where_s16(); }