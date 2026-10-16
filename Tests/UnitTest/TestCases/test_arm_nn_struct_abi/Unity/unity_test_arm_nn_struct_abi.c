/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "../test_arm_nn_struct_abi.c"
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
void test_struct_abi_weight_format_f32(void) { struct_abi_weight_format_f32(); }
void test_struct_abi_weight_format_f16(void) { struct_abi_weight_format_f16(); }
void test_struct_abi_layout(void) { struct_abi_layout(); }
