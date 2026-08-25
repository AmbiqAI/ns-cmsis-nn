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

#include "../test_arm_nn_activation_helpers_f32.c"
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
void test_nn_activation_helpers_f32_tanh_old_domain_bit_identical(void)
{
    nn_activation_helpers_f32_tanh_old_domain_bit_identical();
}
void test_nn_activation_helpers_f32_tanh_extended_band_accuracy(void)
{
    nn_activation_helpers_f32_tanh_extended_band_accuracy();
}
void test_nn_activation_helpers_f32_tanh_tail_clamp_step(void) { nn_activation_helpers_f32_tanh_tail_clamp_step(); }
void test_nn_activation_helpers_f32_tanh_non_finite_contract(void)
{
    nn_activation_helpers_f32_tanh_non_finite_contract();
}
void test_nn_activation_helpers_f32_sigmoid_contract_and_accuracy(void)
{
    nn_activation_helpers_f32_sigmoid_contract_and_accuracy();
}
