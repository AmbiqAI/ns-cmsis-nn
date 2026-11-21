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

void test_activation_lookup_lanes(void)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const uint32_t even[] = {254, 32767, 65535, 65536};
    const uint32_t odd[] = {255, 32768, 0, 17};
    const uint16_t expected[] = {254, 255, 255, 255, 255, 0, 255, 17};
    uint16x8_t indices;
    const mve_pred16_t lookup = arm_nn_activation_lut_indices_s16(vld1q_u32(even), vld1q_u32(odd), &indices);
    uint16_t lanes[8];
    vst1q_u16(lanes, indices);
    TEST_ASSERT_EQUAL_UINT16_ARRAY(expected, lanes, 8);
    /* A 16-bit lane occupies two predicate bits. Only lanes 0, 5 and 7 may gather. */
    TEST_ASSERT_EQUAL_HEX16(0xCC03, lookup);
#else
    TEST_IGNORE_MESSAGE("Requires MVE integer intrinsics");
#endif
}
