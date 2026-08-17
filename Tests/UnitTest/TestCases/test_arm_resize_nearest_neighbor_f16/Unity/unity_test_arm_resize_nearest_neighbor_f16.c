/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_arm_resize_nearest_neighbor_f16.c"
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

void test_resize_nearest_neighbor_f16_width_double(void) { resize_nearest_neighbor_f16_width_double(); }

void test_resize_nearest_neighbor_f16_half_pixel_downsample(void)
{
    resize_nearest_neighbor_f16_half_pixel_downsample();
}

void test_resize_nearest_neighbor_f16_rejects_small_buffer(void) { resize_nearest_neighbor_f16_rejects_small_buffer(); }
