/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_nn_rbf_kernel_f32.c"
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
void test_arm_nn_rbf_kernel_f32_capability(void) { rk_capability(); }
void test_arm_nn_rbf_kernel_f32_sum(void) { rk_sum_normal(); }
void test_arm_nn_rbf_kernel_f32_stage(void) { rk_stage_normal(); }
void test_arm_nn_rbf_kernel_f32_literal(void) { rk_literal(); }
void test_arm_nn_rbf_kernel_f32_entry(void) { rk_entry_refusals(); }
void test_arm_nn_rbf_kernel_f32_special(void) { rk_special_values(); }
