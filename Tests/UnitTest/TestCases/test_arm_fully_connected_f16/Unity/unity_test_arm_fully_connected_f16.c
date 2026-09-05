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

#include "../test_arm_fully_connected_f16.c"
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

void test_fully_connected_1024_to_12_f16(void) { fully_connected_1024_to_12_f16(); }

void test_fully_connected_1024_to_13_batch2_f16(void) { fully_connected_1024_to_13_batch2_f16(); }

void test_fully_connected_k33_k39_n5_f16(void) { fully_connected_k33_k39_n5_f16(); }

void test_fully_connected_k39_n5_clamped_f16(void) { fully_connected_k39_n5_clamped_f16(); }

void test_fully_connected_k31_n9_f16(void) { fully_connected_k31_n9_f16(); }
