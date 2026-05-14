/*
 * SPDX-FileCopyrightText: 2025 Ambiq
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

#include "../test_arm_resize_nearest_neighbor_s8.c"
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

void test_resize_nearest_neighbor_s8_basic(void)
{
    resize_nn_basic_s8_arm_resize_nearest_neighbor_s8();
}

void test_resize_nearest_neighbor_s8_half_pixel(void)
{
    resize_nn_half_pixel_s8_arm_resize_nearest_neighbor_s8();
}

void test_resize_nearest_neighbor_s8_align_corners(void)
{
    resize_nn_align_corners_s8_arm_resize_nearest_neighbor_s8();
}

void test_resize_nearest_neighbor_s8_downsample(void)
{
    resize_nn_downsample_s8_arm_resize_nearest_neighbor_s8();
}

void test_resize_nearest_neighbor_s8_batch_channels(void)
{
    resize_nn_batch_channels_s8_arm_resize_nearest_neighbor_s8();
}
