/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_nn_mean_f32.c"
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

void test_mean_f32_axis_c_arm_nn_mean_f32(void) { mean_f32_axis_c_arm_nn_mean_f32(); }
void test_mean_f32_axis_h_arm_nn_mean_f32(void) { mean_f32_axis_h_arm_nn_mean_f32(); }
void test_mean_f32_axis_hc_arm_nn_mean_f32(void) { mean_f32_axis_hc_arm_nn_mean_f32(); }
void test_mean_f32_axis_all_arm_nn_mean_f32(void) { mean_f32_axis_all_arm_nn_mean_f32(); }
void test_mean_f32_identity_arm_nn_mean_f32(void) { mean_f32_identity_arm_nn_mean_f32(); }
void test_mean_f32_nan_inf_arm_nn_mean_f32(void) { mean_f32_nan_inf_arm_nn_mean_f32(); }
void test_mean_f32_axis_n_arm_nn_mean_f32(void) { mean_f32_axis_n_arm_nn_mean_f32(); }
void test_mean_f32_axis_w_arm_nn_mean_f32(void) { mean_f32_axis_w_arm_nn_mean_f32(); }
void test_mean_f32_axis_nw_arm_nn_mean_f32(void) { mean_f32_axis_nw_arm_nn_mean_f32(); }
void test_mean_f32_axis_hw_arm_nn_mean_f32(void) { mean_f32_axis_hw_arm_nn_mean_f32(); }
void test_mean_f32_axis_wc_arm_nn_mean_f32(void) { mean_f32_axis_wc_arm_nn_mean_f32(); }
void test_mean_f32_dim1_axis_arm_nn_mean_f32(void) { mean_f32_dim1_axis_arm_nn_mean_f32(); }
void test_mean_f32_flatten_long_accumulation_arm_nn_mean_f32(void) { mean_f32_flatten_long_accumulation_arm_nn_mean_f32(); }
void test_mean_f32_generic_long_accumulation_arm_nn_mean_f32(void) { mean_f32_generic_long_accumulation_arm_nn_mean_f32(); }
void test_mean_f32_generic_nan_inf_arm_nn_mean_f32(void) { mean_f32_generic_nan_inf_arm_nn_mean_f32(); }
void test_mean_f32_arg_error_arm_nn_mean_f32(void) { mean_f32_arg_error_arm_nn_mean_f32(); }
