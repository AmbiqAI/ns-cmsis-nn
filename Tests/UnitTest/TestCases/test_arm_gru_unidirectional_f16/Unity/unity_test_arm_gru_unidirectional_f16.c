/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_gru_unidirectional_f16.c"
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

void test_gru_small_f16_arm_gru_unidirectional_f16(void) { gru_small_f16_arm_gru_unidirectional_f16(); }
void test_gru_stream_f16_arm_gru_unidirectional_f16(void) { gru_stream_f16_arm_gru_unidirectional_f16(); }
