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

#include "../test_arm_depthwise_conv_f32.c"
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

void test_depthwise_conv_kws_layer_f32(void) { depthwise_conv_kws_layer_f32(); }

void test_depthwise_conv_layout_gate_f32(void) { depthwise_conv_layout_gate_f32(); }

void test_depthwise_conv_channel_sweep_f32(void) { depthwise_conv_channel_sweep_f32(); }

void test_depthwise_conv_kernel_stride_sweep_f32(void) { depthwise_conv_kernel_stride_sweep_f32(); }

void test_depthwise_conv_valid_and_asymmetric_f32(void) { depthwise_conv_valid_and_asymmetric_f32(); }

void test_depthwise_conv_pad_wider_than_kernel_f32(void) { depthwise_conv_pad_wider_than_kernel_f32(); }

void test_depthwise_conv_dilation_f32(void) { depthwise_conv_dilation_f32(); }

void test_depthwise_conv_batch2_f32(void) { depthwise_conv_batch2_f32(); }

void test_depthwise_conv_ch_mult2_f32(void) { depthwise_conv_ch_mult2_f32(); }

void test_depthwise_conv_nonfinite_inputs_f32(void) { depthwise_conv_nonfinite_inputs_f32(); }
