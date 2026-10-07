/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include <stdlib.h>
#include "../test_arm_nn_gelu_f32.c"
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
void test_gelu_f32_finite(void) { gelu_f32_finite(); }
void test_gelu_f32_counts(void) { gelu_f32_counts(); }
void test_gelu_f32_in_place(void) { gelu_f32_in_place(); }
void test_gelu_f32_special(void) { gelu_f32_special(); }
void test_gelu_f32_arguments(void) { gelu_f32_arguments(); }
void test_gelu_f32_bounds(void) { gelu_f32_bounds(); }
