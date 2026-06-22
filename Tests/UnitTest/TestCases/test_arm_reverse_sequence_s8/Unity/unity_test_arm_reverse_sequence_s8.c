/*
 * SPDX-FileCopyrightText: 2026 Ambiq
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

#include "../test_arm_reverse_sequence_s8.c"
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

void test_reverse_sequence_2d_s8_arm_reverse_sequence_s8(void)
{
    reverse_sequence_2d_s8_arm_reverse_sequence_s8();
}

void test_reverse_sequence_full_and_single_s8_arm_reverse_sequence_s8(void)
{
    reverse_sequence_full_and_single_s8_arm_reverse_sequence_s8();
}

void test_reverse_sequence_axis0_s8_arm_reverse_sequence_s8(void)
{
    reverse_sequence_axis0_s8_arm_reverse_sequence_s8();
}