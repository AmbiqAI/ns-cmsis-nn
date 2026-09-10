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

#include "../test_arm_maximum_minimum_f16.c"
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

void test_maximum_broadcast_row_scalar_f16(void) { maximum_broadcast_row_scalar_f16(); }

void test_minimum_broadcast_row_scalar_f16(void) { minimum_broadcast_row_scalar_f16(); }

void test_maximum_broadcast_general_f16(void) { maximum_broadcast_general_f16(); }

void test_minmax_arg_error_f16(void) { minmax_arg_error_f16(); }
