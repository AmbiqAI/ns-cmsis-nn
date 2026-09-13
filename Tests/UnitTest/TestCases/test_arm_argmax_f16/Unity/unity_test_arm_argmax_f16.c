/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_argmax_f16.c"
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
void test_arm_argmax_f16_golden(void) { ae_golden(); }
void test_arm_argmax_f16_axes(void) { ae_axes(); }
void test_arm_argmax_f16_special(void) { ae_special(); }
void test_arm_argmax_f16_patterns(void) { ae_patterns(); }
void test_arm_argmax_f16_empty(void) { ae_empty(); }
void test_arm_argmax_f16_invalid(void) { ae_invalid(); }
void test_arm_argmax_f16_fp_controls(void) { ae_fp_controls(); }
