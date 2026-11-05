/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_nn_tree_walk_f32.c"
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
void test_arm_nn_tree_walk_f32_record_layout(void) { tw_record_layout(); }
void test_arm_nn_tree_walk_f32_random_tables(void) { tw_random_tables(); }
void test_arm_nn_tree_walk_f32_nan_routes(void) { tw_nan_routes(); }
void test_arm_nn_tree_walk_f32_infinities(void) { tw_infinities(); }
void test_arm_nn_tree_walk_f32_boundary_rows(void) { tw_boundary_rows(); }
void test_arm_nn_tree_walk_f32_signed_zeros(void) { tw_signed_zeros(); }
void test_arm_nn_tree_walk_f32_most_nodes_leaves_features(void) { tw_most_nodes_leaves_features(); }
void test_arm_nn_tree_walk_f32_most_trees(void) { tw_most_trees(); }
void test_arm_nn_tree_walk_f32_cycle_guard(void) { tw_cycle_guard(); }
void test_arm_nn_tree_walk_f32_refused_before_the_walk(void) { tw_refused_before_the_walk(); }
void test_arm_nn_tree_walk_f32_refused_in_the_first_tree(void) { tw_refused_in_the_first_tree(); }
void test_arm_nn_tree_walk_f32_later_tree_keeps_earlier_leaves(void) { tw_later_tree_keeps_earlier_leaves(); }
void test_arm_nn_tree_walk_f32_unreached_defect(void) { tw_unreached_defect(); }
void test_arm_nn_tree_walk_f32_tree_order(void) { tw_tree_order(); }
