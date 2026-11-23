/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_topk_v2_f16.c"
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
void test_arm_topk_v2_f16_golden(void) { tk_golden(); }
void test_arm_topk_v2_f16_every_k(void) { tk_every_k(); }
void test_arm_topk_v2_f16_wide(void) { tk_wide(); }
void test_arm_topk_v2_f16_ties(void) { tk_ties(); }
void test_arm_topk_v2_f16_special(void) { tk_special(); }
void test_arm_topk_v2_f16_patterns(void) { tk_patterns(); }
void test_arm_topk_v2_f16_argmax(void) { tk_argmax(); }
void test_arm_topk_v2_f16_empty(void) { tk_empty(); }
void test_arm_topk_v2_f16_invalid(void) { tk_invalid(); }
void test_arm_topk_v2_f16_fp_controls(void) { tk_fp_controls(); }
