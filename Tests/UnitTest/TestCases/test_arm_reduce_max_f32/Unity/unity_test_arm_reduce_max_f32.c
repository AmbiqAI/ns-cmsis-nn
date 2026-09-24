/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "../test_arm_reduce_max_f32.c"
#include <stdlib.h>
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
void test_arm_reduce_max_f32_golden(void) { re_golden(); }
void test_arm_reduce_max_f32_masks(void) { re_masks(); }
void test_arm_reduce_max_f32_special_positions(void) { re_special_positions(); }
void test_arm_reduce_max_f32_ordered_pairs(void) { re_ordered_pairs(); }
void test_arm_reduce_max_f32_empty(void) { re_empty(); }
void test_arm_reduce_max_f32_invalid(void) { re_invalid(); }
void test_arm_reduce_max_f32_fp_controls(void) { re_fp_controls(); }

void test_arm_reduce_max_f32_random_pairs(void) { re_random_pairs(); }
