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

#include "../test_arm_concatenation_f16.c"
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

void test_concatenation_f16_rank_axis_sweep(void) { concatenation_f16_rank_axis_sweep(); }

void test_concatenation_f16_tail_lengths(void) { concatenation_f16_tail_lengths(); }

void test_concatenation_f16_payload_preservation(void) { concatenation_f16_payload_preservation(); }

void test_concatenation_f16_zero_extent(void) { concatenation_f16_zero_extent(); }

void test_concatenation_f16_arg_error(void) { concatenation_f16_arg_error(); }

void test_concatenation_f16_fuzz(void) { concatenation_f16_fuzz(); }

void test_concatenation_f16_unequal_sizes(void) { concatenation_f16_unequal_sizes(); }

void test_concatenation_f16_matches_4d_axis_kernels(void) { concatenation_f16_matches_4d_axis_kernels(); }
