/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <stdlib.h>

#include "../test_arm_gather_f32.c"
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

void test_gather_f32_rank_batch_sweep(void) { gather_f32_rank_batch_sweep(); }

void test_gather_f32_tails_and_payloads(void) { gather_f32_tails_and_payloads(); }

void test_gather_f32_empty(void) { gather_f32_empty(); }

void test_gather_f32_invalid_indices(void) { gather_f32_invalid_indices(); }

void test_gather_f32_invalid_metadata(void) { gather_f32_invalid_metadata(); }

void test_gather_f32_capacity_and_rank(void) { gather_f32_capacity_and_rank(); }

void test_gather_f32_oracle(void) { gather_f32_oracle(); }
