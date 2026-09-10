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

#include "../test_arm_elementwise_sub_broadcast_f16.c"
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

void test_sub_broadcast_f16_identical_shapes(void) { sub_broadcast_f16_identical_shapes(); }

void test_sub_broadcast_f16_per_channel_right(void) { sub_broadcast_f16_per_channel_right(); }

void test_sub_broadcast_f16_per_channel_left(void) { sub_broadcast_f16_per_channel_left(); }

void test_sub_broadcast_f16_scalar_right(void) { sub_broadcast_f16_scalar_right(); }

void test_sub_broadcast_f16_scalar_left(void) { sub_broadcast_f16_scalar_left(); }

void test_sub_broadcast_f16_both_broadcast(void) { sub_broadcast_f16_both_broadcast(); }

void test_sub_broadcast_f16_batch(void) { sub_broadcast_f16_batch(); }

void test_sub_broadcast_f16_tail_lengths(void) { sub_broadcast_f16_tail_lengths(); }

void test_sub_broadcast_f16_tight_clamp(void) { sub_broadcast_f16_tight_clamp(); }

void test_sub_broadcast_f16_nan_inf(void) { sub_broadcast_f16_nan_inf(); }

void test_sub_broadcast_f16_signed_zero(void) { sub_broadcast_f16_signed_zero(); }

void test_sub_broadcast_f16_arg_error(void) { sub_broadcast_f16_arg_error(); }

void test_sub_broadcast_f16_fuzz(void) { sub_broadcast_f16_fuzz(); }
