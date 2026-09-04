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

#include "../test_arm_softmax_f16_shapes.c"
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

void test_softmax_f16_shapes_arm_softmax_f16(void) { softmax_f16_shapes_arm_softmax_f16(); }

void test_softmax_f16_shapes_rows_sum_to_one_arm_softmax_f16(void)
{
    softmax_f16_shapes_rows_sum_to_one_arm_softmax_f16();
}

void test_softmax_f16_shapes_full_vector_arm_softmax_f16(void) { softmax_f16_shapes_full_vector_arm_softmax_f16(); }

void test_softmax_f16_shapes_wide_arm_softmax_f16(void) { softmax_f16_shapes_wide_arm_softmax_f16(); }

void test_softmax_f16_shapes_full_vector_rows_sum_to_one_arm_softmax_f16(void)
{
    softmax_f16_shapes_full_vector_rows_sum_to_one_arm_softmax_f16();
}

void test_softmax_f16_shapes_arg_error_arm_softmax_f16(void) { softmax_f16_shapes_arg_error_arm_softmax_f16(); }
