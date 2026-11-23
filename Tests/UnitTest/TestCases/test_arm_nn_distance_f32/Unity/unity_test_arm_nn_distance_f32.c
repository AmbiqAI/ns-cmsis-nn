/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "../test_arm_nn_distance_f32.c"
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
void test_arm_nn_distance_f32_random_shapes(void) { distances_random(); }
void test_arm_nn_distance_f32_special_values(void) { distances_special(); }
void test_arm_nn_distance_f32_invalid_arguments(void) { distances_invalid(); }
void test_arm_nn_distance_f32_overlapping_output(void) { distances_overlap(); }

void test_arm_nn_distance_f32_finite_overflow(void) { distances_finite_overflow(); }
