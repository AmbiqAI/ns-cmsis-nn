/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_argmin_f32.c"
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
void test_arm_argmin_f32_golden(void) { ae_golden(); }
void test_arm_argmin_f32_axes(void) { ae_axes(); }
void test_arm_argmin_f32_special(void) { ae_special(); }
void test_arm_argmin_f32_patterns(void) { ae_patterns(); }
void test_arm_argmin_f32_empty(void) { ae_empty(); }
void test_arm_argmin_f32_invalid(void) { ae_invalid(); }
void test_arm_argmin_f32_fp_controls(void) { ae_fp_controls(); }
