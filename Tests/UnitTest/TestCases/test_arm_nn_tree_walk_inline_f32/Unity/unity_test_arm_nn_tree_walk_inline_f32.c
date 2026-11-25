/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_nn_tree_walk_inline_f32.c"
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
void test_arm_nn_tree_walk_inline_f32_capability(void) { ti_capability(); }
void test_arm_nn_tree_walk_inline_f32_golden(void) { ti_golden(); }
void test_arm_nn_tree_walk_inline_f32_nan_routes(void) { ti_nan_routes(); }
void test_arm_nn_tree_walk_inline_f32_ties(void) { ti_ties(); }
void test_arm_nn_tree_walk_inline_f32_reciprocal(void) { ti_reciprocal(); }
void test_arm_nn_tree_walk_inline_f32_malformed(void) { ti_malformed(); }
void test_arm_nn_tree_walk_inline_f32_count_checks(void) { ti_count_checks(); }
