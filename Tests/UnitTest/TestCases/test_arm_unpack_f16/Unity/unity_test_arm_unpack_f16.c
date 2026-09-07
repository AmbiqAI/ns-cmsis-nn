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

#include "../test_arm_unpack_f16.c"
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

void test_unpack_f16_rank_axis_sweep(void) { unpack_f16_rank_axis_sweep(); }

void test_unpack_f16_tail_lengths(void) { unpack_f16_tail_lengths(); }

void test_unpack_f16_payload_preservation(void) { unpack_f16_payload_preservation(); }

void test_unpack_f16_zero_extent(void) { unpack_f16_zero_extent(); }

void test_unpack_f16_arg_error(void) { unpack_f16_arg_error(); }

void test_unpack_f16_fuzz(void) { unpack_f16_fuzz(); }

void test_unpack_f16_n_up_to_8(void) { unpack_f16_n_up_to_8(); }
