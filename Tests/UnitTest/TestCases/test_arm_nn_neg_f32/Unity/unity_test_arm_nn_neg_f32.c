/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "../test_arm_nn_neg_f32.c"

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
void test_neg_f32_bits(void) { neg_bits_f32(); }
void test_neg_f32_args(void) { neg_args_f32(); }
