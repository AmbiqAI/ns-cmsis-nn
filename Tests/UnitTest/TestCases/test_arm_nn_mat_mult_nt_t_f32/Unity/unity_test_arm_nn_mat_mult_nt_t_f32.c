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

#include "../test_arm_nn_mat_mult_nt_t_f32.c"
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

void test_arm_nn_mat_mult_nt_t_f32_differential_small(void) { mm_differential_small(); }

void test_arm_nn_mat_mult_nt_t_f32_differential_2048(void) { mm_differential_2048(); }

void test_arm_nn_mat_mult_nt_t_f32_differential_refusals(void) { mm_differential_refusals(); }

void test_arm_nn_mat_mult_nt_t_f32_literal_library(void) { mm_literal_library(); }

void test_arm_nn_mat_mult_nt_t_f32_literal_c11(void) { mm_literal_c11(); }

void test_arm_nn_mat_mult_nt_t_f32_literal_refusals(void) { mm_literal_refusals(); }
