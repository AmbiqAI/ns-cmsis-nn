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

#include "../test_arm_dequantize_f16_f32.c"
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

void test_dequantize_f16_f32_block_sizes(void) { dequantize_f16_f32_block_sizes(); }

void test_dequantize_f16_f32_exhaustive(void) { dequantize_f16_f32_exhaustive(); }

void test_dequantize_f16_f32_nan_lanes_mixed(void) { dequantize_f16_f32_nan_lanes_mixed(); }

void test_dequantize_f16_f32_subnormals_and_zeros(void) { dequantize_f16_f32_subnormals_and_zeros(); }

void test_dequantize_f16_f32_arg_error(void) { dequantize_f16_f32_arg_error(); }
