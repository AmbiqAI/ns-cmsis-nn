/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_arm_prelu_f16.c"
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

void test_prelu_f16_per_channel_broadcast(void) { prelu_f16_per_channel_broadcast(); }

void test_prelu_f16_scalar_alpha(void) { prelu_f16_scalar_alpha(); }

void test_prelu_f16_rejects_invalid_broadcast(void) { prelu_f16_rejects_invalid_broadcast(); }
