/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_nn_tree_walk_mean_argmax_differential_f32.c"
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
void test_arm_nn_tree_walk_mean_argmax_differential_f32_capability(void) { tmd_suite_capability(); }
void test_arm_nn_tree_walk_mean_argmax_differential_f32_class_rules(void) { tmd_suite_class_rules(); }
void test_arm_nn_tree_walk_mean_argmax_differential_f32_precedence(void) { tmd_suite_precedence(); }
void test_arm_nn_tree_walk_mean_argmax_differential_f32_sweep(void) { tmd_suite_sweep(); }
void test_arm_nn_tree_walk_mean_argmax_differential_f32_refusals(void) { tmd_suite_refusals(); }
void test_arm_nn_tree_walk_mean_argmax_differential_f32_limits(void) { tmd_suite_limits(); }
void test_arm_nn_tree_walk_mean_argmax_differential_f32_golden(void) { tmd_suite_golden(); }
void test_arm_nn_tree_walk_mean_argmax_differential_f32_random(void) { tmd_random(); }
