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

#include "../test_arm_elementwise_sub_broadcast_f32.c"
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

void test_sub_broadcast_f32_identical_shapes(void) { sub_broadcast_f32_identical_shapes(); }

void test_sub_broadcast_f32_per_channel_right(void) { sub_broadcast_f32_per_channel_right(); }

void test_sub_broadcast_f32_per_channel_left(void) { sub_broadcast_f32_per_channel_left(); }

void test_sub_broadcast_f32_scalar_right(void) { sub_broadcast_f32_scalar_right(); }

void test_sub_broadcast_f32_scalar_left(void) { sub_broadcast_f32_scalar_left(); }

void test_sub_broadcast_f32_both_broadcast(void) { sub_broadcast_f32_both_broadcast(); }

void test_sub_broadcast_f32_batch(void) { sub_broadcast_f32_batch(); }

void test_sub_broadcast_f32_tail_lengths(void) { sub_broadcast_f32_tail_lengths(); }

void test_sub_broadcast_f32_tight_clamp(void) { sub_broadcast_f32_tight_clamp(); }

void test_sub_broadcast_f32_nan_inf(void) { sub_broadcast_f32_nan_inf(); }

void test_sub_broadcast_f32_signed_zero(void) { sub_broadcast_f32_signed_zero(); }

void test_sub_broadcast_f32_arg_error(void) { sub_broadcast_f32_arg_error(); }

void test_sub_broadcast_f32_fuzz(void) { sub_broadcast_f32_fuzz(); }
