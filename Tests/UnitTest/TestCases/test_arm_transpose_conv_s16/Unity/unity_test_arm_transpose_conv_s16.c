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

#include "../test_arm_transpose_conv_s16.c"
#include "unity.h"

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

/* Unity runs this before each test. */
void setUp(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
}

/* Unity runs this after each test. */
void tearDown(void) {}
void test_transpose_conv_s16_1_arm_transpose_conv_s16(void) { transpose_conv_s16_1_arm_transpose_conv_s16(); }
void test_transpose_conv_s16_2_arm_transpose_conv_s16(void) { transpose_conv_s16_2_arm_transpose_conv_s16(); }
void test_transpose_conv_s16_3_arm_transpose_conv_s16(void) { transpose_conv_s16_3_arm_transpose_conv_s16(); }
void test_transpose_conv_s16_4_arm_transpose_conv_s16(void) { transpose_conv_s16_4_arm_transpose_conv_s16(); }
void test_transpose_conv_s16_5_arm_transpose_conv_s16(void) { transpose_conv_s16_5_arm_transpose_conv_s16(); }
void test_transpose_conv_s16_6_arm_transpose_conv_s16(void) { transpose_conv_s16_6_arm_transpose_conv_s16(); }
void test_transpose_conv_s16_7_arm_transpose_conv_s16(void) { transpose_conv_s16_7_arm_transpose_conv_s16(); }
void test_transpose_conv_s16_invalid_params_arm_transpose_conv_s16(void)
{
    transpose_conv_s16_invalid_params_arm_transpose_conv_s16();
}
void test_transpose_conv_s16_negative_dims_arm_transpose_conv_s16(void)
{
    transpose_conv_s16_negative_dims_arm_transpose_conv_s16();
}
void test_transpose_conv_s16_extreme_shapes_arm_transpose_conv_s16(void)
{
    transpose_conv_s16_extreme_shapes_arm_transpose_conv_s16();
}
