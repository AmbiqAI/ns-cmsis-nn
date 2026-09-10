/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_nn_activation_helpers_f16.c"
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
void test_nn_activation_helpers_f16_tanh_exhaustive_accuracy(void)
{
    nn_activation_helpers_f16_tanh_exhaustive_accuracy();
}
void test_nn_activation_helpers_f16_tanh_helia_rt_points(void) { nn_activation_helpers_f16_tanh_helia_rt_points(); }
void test_nn_activation_helpers_f16_tanh_non_finite_and_boundary(void)
{
    nn_activation_helpers_f16_tanh_non_finite_and_boundary();
}
void test_nn_activation_helpers_f16_tanh_scalar_vs_mve_agreement(void)
{
    nn_activation_helpers_f16_tanh_scalar_vs_mve_agreement();
}
