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

#include "../test_arm_prelu_s16.c"
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

void test_prelu_scalar_alpha_s16(void) { prelu_scalar_alpha_s16_arm_prelu_s16(); }
void test_prelu_per_channel_s16(void) { prelu_per_channel_s16_arm_prelu_s16(); }
void test_prelu_alpha_width_broadcast_s16(void) { prelu_alpha_width_broadcast_s16_arm_prelu_s16(); }
void test_prelu_output_shape_mismatch_s16(void) { prelu_output_shape_mismatch_s16_arm_prelu_s16(); }