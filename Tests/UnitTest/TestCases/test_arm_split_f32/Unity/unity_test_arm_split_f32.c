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

#include "../test_arm_split_f32.c"
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

void test_split_f32_rank_axis_sweep(void) { split_f32_rank_axis_sweep(); }

void test_split_f32_tail_lengths(void) { split_f32_tail_lengths(); }

void test_split_f32_payload_preservation(void) { split_f32_payload_preservation(); }

void test_split_f32_zero_extent(void) { split_f32_zero_extent(); }

void test_split_f32_arg_error(void) { split_f32_arg_error(); }

void test_split_f32_fuzz(void) { split_f32_fuzz(); }

void test_split_f32_unequal_sizes(void) { split_f32_unequal_sizes(); }
