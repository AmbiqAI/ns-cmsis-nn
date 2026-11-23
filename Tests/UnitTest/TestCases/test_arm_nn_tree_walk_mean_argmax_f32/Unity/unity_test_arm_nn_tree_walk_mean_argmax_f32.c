/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_nn_tree_walk_mean_argmax_f32.c"
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
void test_arm_nn_tree_walk_mean_argmax_f32_capability(void) { tm_capability(); }
void test_arm_nn_tree_walk_mean_argmax_f32_class_rules(void) { tm_class_rules(); }
void test_arm_nn_tree_walk_mean_argmax_f32_precedence(void) { tm_precedence(); }
void test_arm_nn_tree_walk_mean_argmax_f32_sweep(void) { tm_sweep(); }
void test_arm_nn_tree_walk_mean_argmax_f32_refusals(void) { tm_refusals(); }
void test_arm_nn_tree_walk_mean_argmax_f32_limits(void) { tm_limits(); }
void test_arm_nn_tree_walk_mean_argmax_f32_golden(void) { tm_golden(); }
