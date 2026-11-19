/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq Micro, Inc.
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include <stdlib.h>

#include "../test_arm_nn_accumulator_wrap_s8.c"
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
void test_vec_mat_accumulator_wrap_s8(void) { vec_mat_accumulator_wrap_s8(); }
void test_per_ch_accumulator_wrap_s8(void) { per_ch_accumulator_wrap_s8(); }
void test_svdf_accumulator_wrap_s8(void) { svdf_accumulator_wrap_s8(); }
void test_mat_mult_accumulator_wrap_s32(void) { mat_mult_accumulator_wrap_s32(); }
void test_lstm_accumulator_wrap_s8_s16(void) { lstm_accumulator_wrap_s8_s16(); }
