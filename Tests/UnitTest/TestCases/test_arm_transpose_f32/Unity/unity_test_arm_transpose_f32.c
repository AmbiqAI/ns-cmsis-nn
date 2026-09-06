/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_transpose_f32.c"
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
void test_transpose_dims_mismatch_arm_transpose_f32(void) { transpose_dims_mismatch_arm_transpose_f32(); }
void test_transpose_perm_duplicate_axis_arm_transpose_f32(void) { transpose_perm_duplicate_axis_arm_transpose_f32(); }
void test_transpose_zero_extent_arm_transpose_f32(void) { transpose_zero_extent_arm_transpose_f32(); }
