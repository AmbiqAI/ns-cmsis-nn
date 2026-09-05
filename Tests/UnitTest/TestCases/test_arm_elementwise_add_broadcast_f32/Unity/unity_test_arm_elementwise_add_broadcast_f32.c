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

#include "../test_arm_elementwise_add_broadcast_f32.c"
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

void test_add_broadcast_f32_identical_shapes(void) { add_broadcast_f32_identical_shapes(); }

void test_add_broadcast_f32_per_channel_right(void) { add_broadcast_f32_per_channel_right(); }

void test_add_broadcast_f32_per_channel_left(void) { add_broadcast_f32_per_channel_left(); }

void test_add_broadcast_f32_scalar_right(void) { add_broadcast_f32_scalar_right(); }

void test_add_broadcast_f32_scalar_left(void) { add_broadcast_f32_scalar_left(); }

void test_add_broadcast_f32_both_broadcast(void) { add_broadcast_f32_both_broadcast(); }

void test_add_broadcast_f32_batch(void) { add_broadcast_f32_batch(); }

void test_add_broadcast_f32_tail_lengths(void) { add_broadcast_f32_tail_lengths(); }

void test_add_broadcast_f32_tight_clamp(void) { add_broadcast_f32_tight_clamp(); }

void test_add_broadcast_f32_nan_inf(void) { add_broadcast_f32_nan_inf(); }

void test_add_broadcast_f32_signed_zero(void) { add_broadcast_f32_signed_zero(); }

void test_add_broadcast_f32_arg_error(void) { add_broadcast_f32_arg_error(); }

void test_add_broadcast_f32_fuzz(void) { add_broadcast_f32_fuzz(); }
