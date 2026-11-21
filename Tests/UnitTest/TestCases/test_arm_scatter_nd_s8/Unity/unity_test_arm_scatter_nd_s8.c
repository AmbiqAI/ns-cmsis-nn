/*
 * SPDX-FileCopyrightText: 2026 Ambiq
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

#include "../test_arm_scatter_nd_s8.c"
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

void test_scatter_nd_1d_s8_arm_scatter_nd_s8(void) { scatter_nd_1d_s8_arm_scatter_nd_s8(); }
void test_scatter_nd_rows_s8_arm_scatter_nd_s8(void) { scatter_nd_rows_s8_arm_scatter_nd_s8(); }
void test_scatter_nd_points_2d_s8_arm_scatter_nd_s8(void) { scatter_nd_points_2d_s8_arm_scatter_nd_s8(); }

void test_scatter_nd_coordinate_bounds(void)
{
    scatter_nd_coordinate_bounds();
}

void test_scatter_nd_slice_bounds(void)
{
    scatter_nd_slice_bounds();
}

void test_scatter_nd_empty_slices(void)
{
    scatter_nd_empty_slices();
}

void test_scatter_nd_deep_indices(void)
{
    const int32_t strides[] = {4, 4, 4, 4, 4, 4, 4, 2, 1};
    const int32_t indices[] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    const cmsis_nn_scatter_nd_params params = {
        .num_updates = 2, .index_depth = 9, .slice_size = 1, .output_size = 4, .output_strides = strides};
    const int8_t updates[] = {5, 7};
    const int8_t expected[] = {0, 7, 5, 0};
    int8_t output[4];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_scatter_nd_s8(indices, updates, &params, output));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, 4);
}
