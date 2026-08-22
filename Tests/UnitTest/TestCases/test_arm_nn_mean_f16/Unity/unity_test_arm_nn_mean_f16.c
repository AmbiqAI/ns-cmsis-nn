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

#include "../test_arm_nn_mean_f16.c"
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

void test_mean_f16_axis_c_arm_nn_mean_f16(void) { mean_f16_axis_c_arm_nn_mean_f16(); }
void test_mean_f16_axis_h_arm_nn_mean_f16(void) { mean_f16_axis_h_arm_nn_mean_f16(); }
void test_mean_f16_axis_hc_arm_nn_mean_f16(void) { mean_f16_axis_hc_arm_nn_mean_f16(); }
void test_mean_f16_axis_all_arm_nn_mean_f16(void) { mean_f16_axis_all_arm_nn_mean_f16(); }
void test_mean_f16_identity_arm_nn_mean_f16(void) { mean_f16_identity_arm_nn_mean_f16(); }
void test_mean_f16_large_sum_arm_nn_mean_f16(void) { mean_f16_large_sum_arm_nn_mean_f16(); }
void test_mean_f16_nan_inf_arm_nn_mean_f16(void) { mean_f16_nan_inf_arm_nn_mean_f16(); }
void test_mean_f16_arg_error_arm_nn_mean_f16(void) { mean_f16_arg_error_arm_nn_mean_f16(); }
