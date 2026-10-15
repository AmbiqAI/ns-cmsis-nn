/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "../test_arm_nn_gelu_f16.c"
#include <stdlib.h>
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
void test_gelu_f16_exhaustive(void) { gelu_f16_exhaustive(); }
void test_gelu_f16_accuracy(void) { gelu_f16_accuracy(); }
void test_gelu_f16_special(void) { gelu_f16_special(); }
void test_gelu_f16_counts(void) { gelu_f16_counts(); }
void test_gelu_f16_in_place(void) { gelu_f16_in_place(); }
void test_gelu_f16_arguments(void) { gelu_f16_arguments(); }
void test_gelu_f16_flush_to_zero(void) { gelu_f16_flush_to_zero(); }
void test_gelu_f16_bounds(void) { gelu_f16_bounds(); }
