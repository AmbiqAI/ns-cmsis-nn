/*
 * SPDX-FileCopyrightText: 2025 Ambiq
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

#include "../test_arm_gather_s8.c"
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

void test_gather_axis_c_basic_s8_arm_gather_s8(void) { gather_axis_c_basic_s8_arm_gather_s8(); }
void test_gather_axis_h_multi_s8_arm_gather_s8(void) { gather_axis_h_multi_s8_arm_gather_s8(); }
void test_gather_neg_axis_wide_s8_arm_gather_s8(void) { gather_neg_axis_wide_s8_arm_gather_s8(); }
void test_gather_batch_dims_s8_arm_gather_s8(void) { gather_batch_dims_s8_arm_gather_s8(); }
void test_gather_axis_n_reorder_s8_arm_gather_s8(void) { gather_axis_n_reorder_s8_arm_gather_s8(); }
