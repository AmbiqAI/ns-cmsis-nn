/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "../test_arm_reduce_min_f16.c"
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
void test_arm_reduce_min_f16_golden(void) { re_golden(); }
void test_arm_reduce_min_f16_masks(void) { re_masks(); }
void test_arm_reduce_min_f16_special_positions(void) { re_special_positions(); }
void test_arm_reduce_min_f16_ordered_pairs(void) { re_ordered_pairs(); }
void test_arm_reduce_min_f16_empty(void) { re_empty(); }
void test_arm_reduce_min_f16_invalid(void) { re_invalid(); }
void test_arm_reduce_min_f16_fp_controls(void) { re_fp_controls(); }
void test_arm_reduce_min_f16_exhaust_half(void) { re_exhaust_half(); }

void test_arm_reduce_min_f16_random_pairs(void) { re_random_pairs(); }
