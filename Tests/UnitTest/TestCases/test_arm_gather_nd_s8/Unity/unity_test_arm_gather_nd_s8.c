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

#include "../test_arm_gather_nd_s8.c"
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

void test_gather_nd_point_s8_arm_gather_nd_s8(void) { gather_nd_point_s8_arm_gather_nd_s8(); }
void test_gather_nd_slice_hw_s8_arm_gather_nd_s8(void) { gather_nd_slice_hw_s8_arm_gather_nd_s8(); }
void test_gather_nd_batch_dims_s8_arm_gather_nd_s8(void) { gather_nd_batch_dims_s8_arm_gather_nd_s8(); }
void test_gather_nd_depth_s8_arm_gather_nd_s8(void) { gather_nd_depth_s8_arm_gather_nd_s8(); }
void test_gather_nd_reduce_rank_s8_arm_gather_nd_s8(void) { gather_nd_reduce_rank_s8_arm_gather_nd_s8(); }
