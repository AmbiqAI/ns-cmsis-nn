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

#include "../test_arm_resize_bilinear_f16.c"
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

void test_resize_bilinear_f16_reference(void) { resize_bilinear_reference(); }
void test_resize_bilinear_f16_copies(void) { resize_bilinear_copies(); }
void test_resize_bilinear_f16_rejects(void) { resize_bilinear_rejects(); }
void test_resize_bilinear_f16_finite_extrema(void) { resize_bilinear_finite_extrema(); }
