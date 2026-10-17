/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_softmax_f16.c"
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
void test_softmax_f16_arm_softmax_f16(void) { softmax_f16_arm_softmax_f16(); }

void test_softmax_f16_scalar_exp_lower_range(void) { softmax_f16_scalar_exp_lower_range(); }

void test_softmax_f16_large_gap(void) { softmax_f16_large_gap(); }

void test_softmax_f16_wide_large_gap(void) { softmax_f16_wide_large_gap(); }

void test_softmax_f16_lut_lane_boundaries(void) { softmax_f16_lut_lane_boundaries(); }
