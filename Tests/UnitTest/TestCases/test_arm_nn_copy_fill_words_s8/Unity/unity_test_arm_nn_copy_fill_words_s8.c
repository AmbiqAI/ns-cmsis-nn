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

#include "../test_arm_nn_copy_fill_words_s8.c"
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

void test_copy_words_arm_nn_copy_fill_words_s8(void) { copy_words_arm_nn_copy_fill_words_s8(); }

void test_fill_words_arm_nn_copy_fill_words_s8(void) { fill_words_arm_nn_copy_fill_words_s8(); }

void test_fill_s16_arm_nn_copy_fill_words_s8(void) { fill_s16_arm_nn_copy_fill_words_s8(); }
