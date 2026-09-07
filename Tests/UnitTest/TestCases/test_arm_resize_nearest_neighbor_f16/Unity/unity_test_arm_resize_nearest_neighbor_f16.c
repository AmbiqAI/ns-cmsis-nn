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

#include "../test_arm_resize_nearest_neighbor_f16.c"
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

void test_resize_nearest_neighbor_f16_tflite_cases(void) { resize_nearest_neighbor_f16_tflite_cases(); }

void test_resize_nearest_neighbor_f16_special_values(void) { resize_nearest_neighbor_f16_special_values(); }

void test_resize_nearest_neighbor_f16_rejects_bad_args(void) { resize_nearest_neighbor_f16_rejects_bad_args(); }

void test_resize_nearest_neighbor_f16_buffer_size(void) { resize_nearest_neighbor_f16_buffer_size(); }
