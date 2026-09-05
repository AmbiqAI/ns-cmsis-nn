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

#include "../test_arm_convolve_packed_f16.c"
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

void test_convolve_packed_3x3_f16(void) { convolve_packed_3x3_f16(); }

void test_convolve_packed_1xn_f16(void) { convolve_packed_1xn_f16(); }

void test_convolve_1xn_pad_wider_than_kernel_f16(void) { convolve_1xn_pad_wider_than_kernel_f16(); }

void test_convolve_packed_matmul_nan_f16(void) { convolve_packed_matmul_nan_f16(); }

void test_convolve_small_k_3x3_s2_f16(void) { convolve_small_k_3x3_s2_f16(); }

void test_convolve_small_k_few_filters_f16(void) { convolve_small_k_few_filters_f16(); }

void test_convolve_5x5_single_channel_f16(void) { convolve_5x5_single_channel_f16(); }

void test_convolve_small_c_dilated_f16(void) { convolve_small_c_dilated_f16(); }

void test_convolve_small_c_batch2_f16(void) { convolve_small_c_batch2_f16(); }

void test_convolve_full_c_partial_block_f16(void) { convolve_full_c_partial_block_f16(); }

void test_convolve_small_c_no_overread_f16(void) { convolve_small_c_no_overread_f16(); }
