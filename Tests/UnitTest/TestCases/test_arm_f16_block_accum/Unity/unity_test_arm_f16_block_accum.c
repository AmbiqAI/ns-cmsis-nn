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

#include "../test_arm_f16_block_accum.c"
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

void test_fc_std_long_k(void) { ba_fc_std_long_k(); }

void test_fc_packed_long_k(void) { ba_fc_packed_long_k(); }

void test_fc_block_order(void) { ba_fc_block_order(); }

void test_fc_short_k(void) { ba_fc_short_k(); }

void test_conv1x1_long_k(void) { ba_conv1x1_long_k(); }

void test_conv_1xn_long_k(void) { ba_conv_1xn_long_k(); }

void test_conv1d_dilated_long_k(void) { ba_conv1d_dilated_long_k(); }

void test_conv1d_spec_long_k(void) { ba_conv1d_spec_long_k(); }

void test_conv_direct_long_k(void) { ba_conv_direct_long_k(); }

void test_depthwise_long_k(void) { ba_depthwise_long_k(); }
