/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_nn_tree_walk_values_f32.c"
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
void test_arm_nn_tree_walk_values_f32_capability(void) { tv_capability(); }
void test_arm_nn_tree_walk_values_f32_words(void) { tv_words_pass_through(); }
void test_arm_nn_tree_walk_values_f32_precedence(void) { tv_precedence(); }
void test_arm_nn_tree_walk_values_f32_sweep(void) { tv_sweep(); }
void test_arm_nn_tree_walk_values_f32_refusals(void) { tv_refusals(); }
void test_arm_nn_tree_walk_values_f32_limits(void) { tv_limits(); }
void test_arm_nn_tree_walk_values_f32_golden(void) { tv_golden(); }
