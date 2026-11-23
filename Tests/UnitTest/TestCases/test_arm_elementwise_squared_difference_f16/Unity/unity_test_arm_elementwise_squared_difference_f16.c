/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <Internal/arm_nn_fp_control.h>
#include <stdio.h>
#include <stdlib.h>

#include "../test_arm_elementwise_squared_difference_f16.c"
#include "unity.h"

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

void test_squared_difference_f16(void) { squared_difference_f16(); }
void test_squared_difference_f16_complex(void) { squared_difference_f16_complex(); }
void test_squared_difference_f16_nan_inf(void) { squared_difference_f16_nan_inf(); }
void test_squared_difference_f16_arg_error(void) { squared_difference_f16_arg_error(); }

void test_broadcast_f16_shapes(void) { squared_difference_broadcast_shapes(); }
void test_broadcast_f16_errors(void) { squared_difference_broadcast_errors(); }

void test_broadcast_f16_fp_modes(void)
{
#if defined(ARM_NN_FP_CONTROL_AVAILABLE)
    const arm_nn_fp_control_t saved = arm_nn_get_fp_control();
    arm_nn_set_fp_control(saved & ~((1u << 24) | (1u << 19)));
    broadcast_f16_compare_flat();
    arm_nn_set_fp_control(saved | (1u << 24) | (1u << 19));
    broadcast_f16_compare_flat();
    arm_nn_set_fp_control(saved);
#else
    broadcast_f16_compare_flat();
#endif
}
