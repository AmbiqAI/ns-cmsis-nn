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

#include "../test_arm_hard_swish_f32.c"
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

void test_hard_swish_f32_curved_arm_hard_swish_f32(void) { hard_swish_f32_curved_arm_hard_swish_f32(); }

void test_hard_swish_f32_identity_region_arm_hard_swish_f32(void) { hard_swish_f32_identity_region_arm_hard_swish_f32(); }

void test_hard_swish_f32_zero_region_arm_hard_swish_f32(void) { hard_swish_f32_zero_region_arm_hard_swish_f32(); }

void test_hard_swish_f32_knots_arm_hard_swish_f32(void) { hard_swish_f32_knots_arm_hard_swish_f32(); }

void test_hard_swish_f32_nan_inf_arm_hard_swish_f32(void) { hard_swish_f32_nan_inf_arm_hard_swish_f32(); }

void test_hard_swish_f32_denormal_arm_hard_swish_f32(void) { hard_swish_f32_denormal_arm_hard_swish_f32(); }

void test_hard_swish_f32_tail_sizes_arm_hard_swish_f32(void) { hard_swish_f32_tail_sizes_arm_hard_swish_f32(); }

void test_hard_swish_f32_fma_witness_arm_hard_swish_f32(void) { hard_swish_f32_fma_witness_arm_hard_swish_f32(); }

void test_hard_swish_f32_arg_error_arm_hard_swish_f32(void) { hard_swish_f32_arg_error_arm_hard_swish_f32(); }
