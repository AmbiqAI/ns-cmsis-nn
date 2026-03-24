/*
 * Copyright (C) 2026 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_sqrt_s8.c"
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

void test_sqrt_small_tensor_s8_arm_sqrt_s8(void) { sqrt_small_tensor_s8_arm_sqrt_s8(); }
void test_sqrt_long_row_s8_arm_sqrt_s8(void) { sqrt_long_row_s8_arm_sqrt_s8(); }
void test_sqrt_multi_batch_s8_arm_sqrt_s8(void) { sqrt_multi_batch_s8_arm_sqrt_s8(); }
