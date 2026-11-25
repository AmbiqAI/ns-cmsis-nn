/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_topk_v2_f32.c"
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
void test_arm_topk_v2_f32_golden(void) { tk_golden(); }
void test_arm_topk_v2_f32_every_k(void) { tk_every_k(); }
void test_arm_topk_v2_f32_wide(void) { tk_wide(); }
void test_arm_topk_v2_f32_ties(void) { tk_ties(); }
void test_arm_topk_v2_f32_special(void) { tk_special(); }
void test_arm_topk_v2_f32_patterns(void) { tk_patterns(); }
void test_arm_topk_v2_f32_argmax(void) { tk_argmax(); }
void test_arm_topk_v2_f32_empty(void) { tk_empty(); }
void test_arm_topk_v2_f32_invalid(void) { tk_invalid(); }
void test_arm_topk_v2_f32_fp_controls(void) { tk_fp_controls(); }
void test_arm_topk_v2_f32_differential_normals(void) { td_normals(); }
void test_arm_topk_v2_f32_differential_equal(void) { td_equal(); }
void test_arm_topk_v2_f32_differential_ties(void) { td_ties(); }
void test_arm_topk_v2_f32_differential_zeros(void) { td_zeros(); }
void test_arm_topk_v2_f32_differential_nan_lanes(void) { td_nan_lanes(); }
void test_arm_topk_v2_f32_differential_all_nan(void) { td_all_nan(); }
void test_arm_topk_v2_f32_differential_infinities(void) { td_infinities(); }
void test_arm_topk_v2_f32_differential_ascending(void) { td_ascending(); }
void test_arm_topk_v2_f32_differential_descending(void) { td_descending(); }
void test_arm_topk_v2_f32_differential_subnormals(void) { td_subnormals(); }
void test_arm_topk_v2_f32_differential_bits(void) { td_bits(); }
void test_arm_topk_v2_f32_differential_refusals(void) { td_refusals(); }
