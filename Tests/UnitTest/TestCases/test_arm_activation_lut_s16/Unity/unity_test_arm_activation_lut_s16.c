/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <Internal/arm_nn_activation_lut_s16.h>
#include "arm_nnfunctions.h"
#include "unity.h"
#include <stdlib.h>

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

void test_activation_mixed_lanes_and_tail(void)
{
    const int16_t input[] = {0, 1, -1, 127, -128, 16384, -32768, 32767, 3};
    /* Scalar fixed-point interpolation and saturation at 2.5 table steps per unit. */
    const int16_t tanh_expected[] = {0, 1705, -1705, 32767, -32767, 32767, -32767, 32767, 5079};
    const int16_t logistic_expected[] = {16384, 17237, 15531, 32767, 1, 32767, 1, 32767, 18923};
    int16_t output[9];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_tanh_s16(input, output, 9, 640, 0));
    TEST_ASSERT_EQUAL_INT16_ARRAY(tanh_expected, output, 9);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_logistic_s16(input, output, 9, 1280, 0));
    TEST_ASSERT_EQUAL_INT16_ARRAY(logistic_expected, output, 9);
}

void test_activation_minimum_scaled_value(void)
{
    const int16_t input[] = {INT16_MIN, INT16_MAX};
    int16_t output[2];
    const int16_t tanh_expected[] = {-32767, 32767};
    const int16_t logistic_expected[] = {1, 32767};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_tanh_s16(input, output, 2, 65536, 0));
    TEST_ASSERT_EQUAL_INT16_ARRAY(tanh_expected, output, 2);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_logistic_s16(input, output, 2, 65536, 0));
    TEST_ASSERT_EQUAL_INT16_ARRAY(logistic_expected, output, 2);
}

void test_activation_last_interpolation_interval(void)
{
    const int16_t input[] = {32575, 32576, 32577, 32640, -32575, -32576, -32577, -32640};
    const int16_t tanh_expected[] = {32766, 32767, 32767, 32767, -32766, -32767, -32767, -32767};
    const int16_t logistic_expected[] = {32767, 32767, 32767, 32767, 1, 1, 1, 1};
    int16_t output[8];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_tanh_s16(input, output, 8, 2, 0));
    TEST_ASSERT_EQUAL_INT16_ARRAY(tanh_expected, output, 8);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_logistic_s16(input, output, 8, 4, 0));
    TEST_ASSERT_EQUAL_INT16_ARRAY(logistic_expected, output, 8);
}

void test_activation_lookup_lanes(void)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const uint32_t first[] = {254, 255, 32767, 32768};
    const uint32_t second[] = {65535, 0, 65536, 17};
    TEST_ASSERT_EQUAL_HEX16(0x000F, arm_nn_activation_lut_predicate_s16(vld1q_u32(first), vctp32q(4)));
    TEST_ASSERT_EQUAL_HEX16(0xF0F0, arm_nn_activation_lut_predicate_s16(vld1q_u32(second), vctp32q(4)));
    TEST_ASSERT_EQUAL_HEX16(0x00F0, arm_nn_activation_lut_predicate_s16(vld1q_u32(second), vctp32q(3)));
#else
    TEST_IGNORE_MESSAGE("Requires MVE integer intrinsics");
#endif
}
