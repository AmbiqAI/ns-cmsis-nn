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

#include "../test_arm_reduce_sum_f16.c"
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

void test_rsum_f16_axis_c_arm_reduce_sum_f16(void) { rsum_f16_axis_c_arm_reduce_sum_f16(); }

void test_rsum_f16_axis_hwc_arm_reduce_sum_f16(void) { rsum_f16_axis_hwc_arm_reduce_sum_f16(); }

void test_rsum_f16_axis_all_arm_reduce_sum_f16(void) { rsum_f16_axis_all_arm_reduce_sum_f16(); }

void test_rsum_f16_axis_hw_arm_reduce_sum_f16(void) { rsum_f16_axis_hw_arm_reduce_sum_f16(); }

void test_rsum_f16_axis_h_arm_reduce_sum_f16(void) { rsum_f16_axis_h_arm_reduce_sum_f16(); }

void test_rsum_f16_arg_error_arm_reduce_sum_f16(void) { rsum_f16_arg_error_arm_reduce_sum_f16(); }
