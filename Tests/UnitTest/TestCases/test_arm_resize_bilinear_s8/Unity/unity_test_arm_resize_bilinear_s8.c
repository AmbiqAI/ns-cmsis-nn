/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <stdlib.h>

#include "../test_arm_resize_bilinear_s8.c"
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
void test_resize_bilinear_shapes(void) { resize_bilinear_shapes(); }
void test_resize_bilinear_scale_channel_pairs(void) { resize_bilinear_scale_channel_pairs(); }
void test_resize_bilinear_signed_ties(void) { resize_bilinear_signed_ties(); }
void test_resize_bilinear_invalid(void) { resize_bilinear_invalid(); }
