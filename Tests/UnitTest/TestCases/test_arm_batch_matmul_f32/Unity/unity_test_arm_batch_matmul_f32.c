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

#include "../test_arm_batch_matmul_f32.c"
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

void test_batch_matmul_f32_packed_rhs_batch2_n5_arm_batch_matmul_f32(void)
{
    batch_matmul_f32_packed_rhs_batch2_n5_arm_batch_matmul_f32();
}

void test_batch_matmul_f32_standard_rhs_batch2_n5_arm_batch_matmul_f32(void)
{
    batch_matmul_f32_standard_rhs_batch2_n5_arm_batch_matmul_f32();
}

void test_batch_matmul_f32_packed_broadcast_shapes_arm_batch_matmul_f32(void)
{
    batch_matmul_f32_packed_broadcast_shapes_arm_batch_matmul_f32();
}

void test_batch_matmul_f32_packed_block_width_batch2_arm_batch_matmul_f32(void)
{
    batch_matmul_f32_packed_block_width_batch2_arm_batch_matmul_f32();
}
