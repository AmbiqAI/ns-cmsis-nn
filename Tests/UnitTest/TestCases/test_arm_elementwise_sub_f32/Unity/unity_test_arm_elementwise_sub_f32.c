/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
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

#include "../test_arm_elementwise_sub_f32.c"
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

void test_sub_f32_arm_elementwise_sub_f32(void) { sub_f32_arm_elementwise_sub_f32(); }

void test_sub_f32_nan_inf_arm_elementwise_sub_f32(void) { sub_f32_nan_inf_arm_elementwise_sub_f32(); }

void test_sub_f32_arg_error_arm_elementwise_sub_f32(void) { sub_f32_arg_error_arm_elementwise_sub_f32(); }
