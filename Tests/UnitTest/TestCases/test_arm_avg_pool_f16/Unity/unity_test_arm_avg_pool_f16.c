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

#include "../test_arm_avg_pool_f16.c"
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

void test_avg_pool_f16_empty_window(void) { avg_pool_f16_empty_window(); }

void test_avg_pool_f16_window_bound_limits(void) { avg_pool_f16_window_bound_limits(); }

void test_avg_pool_f16_empty_output(void) { avg_pool_f16_empty_output(); }

void test_avg_pool_f16_asymmetric_axes(void) { avg_pool_f16_asymmetric_axes(); }

void test_avg_pool_f16_negative_stride(void) { avg_pool_f16_negative_stride(); }

void test_avg_pool_f16_negative_padding(void) { avg_pool_f16_negative_padding(); }

void test_avg_pool_f16_batches(void) { avg_pool_f16_batches(); }

void test_avg_pool_f16_window_check_edges(void) { avg_pool_f16_window_check_edges(); }
