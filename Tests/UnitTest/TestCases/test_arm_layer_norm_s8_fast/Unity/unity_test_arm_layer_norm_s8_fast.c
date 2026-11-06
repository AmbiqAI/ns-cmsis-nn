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

#include "../test_arm_layer_norm_s8_fast.c"
#include "unity.h"

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

void setUp(void)
{ /* This is run before EACH TEST */
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
}

void tearDown(void) {}

void test_layer_norm_s8_fast_cases(void) { layer_norm_s8_fast_reference_cases(); }
void test_layer_norm_s8_fast_in_place(void) { layer_norm_s8_fast_in_place(); }
void test_layer_norm_s8_fast_activation(void) { layer_norm_s8_fast_activation(); }
void test_layer_norm_s8_fast_invalid(void) { layer_norm_s8_fast_invalid(); }
