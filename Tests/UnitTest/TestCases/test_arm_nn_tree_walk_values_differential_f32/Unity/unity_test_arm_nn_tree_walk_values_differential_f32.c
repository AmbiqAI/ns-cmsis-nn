/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_nn_tree_walk_values_differential_f32.c"
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
void test_arm_nn_tree_walk_values_differential_f32_capability(void) { tvd_suite_capability(); }
void test_arm_nn_tree_walk_values_differential_f32_words(void) { tvd_suite_words(); }
void test_arm_nn_tree_walk_values_differential_f32_precedence(void) { tvd_suite_precedence(); }
void test_arm_nn_tree_walk_values_differential_f32_sweep(void) { tvd_suite_sweep(); }
void test_arm_nn_tree_walk_values_differential_f32_refusals(void) { tvd_suite_refusals(); }
void test_arm_nn_tree_walk_values_differential_f32_limits(void) { tvd_suite_limits(); }
void test_arm_nn_tree_walk_values_differential_f32_golden(void) { tvd_suite_golden(); }
void test_arm_nn_tree_walk_values_differential_f32_random(void) { tvd_random(); }
