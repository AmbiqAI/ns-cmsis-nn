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

#include "../test_arm_max_pool_f32.c"
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

void test_max_pool_f32_empty_window(void) { max_pool_f32_empty_window(); }

void test_max_pool_f32_window_bound_limits(void) { max_pool_f32_window_bound_limits(); }

void test_max_pool_f32_empty_output(void) { max_pool_f32_empty_output(); }

void test_max_pool_f32_asymmetric_axes(void) { max_pool_f32_asymmetric_axes(); }

void test_max_pool_f32_negative_stride(void) { max_pool_f32_negative_stride(); }

void test_max_pool_f32_negative_padding(void) { max_pool_f32_negative_padding(); }

void test_max_pool_f32_batches(void) { max_pool_f32_batches(); }

void test_max_pool_f32_window_check_edges(void) { max_pool_f32_window_check_edges(); }
