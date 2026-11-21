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

#include "../test_arm_reverse_sequence_s8.c"
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

void test_reverse_sequence_2d_s8_arm_reverse_sequence_s8(void)
{
    reverse_sequence_2d_s8_arm_reverse_sequence_s8();
}

void test_reverse_sequence_full_and_single_s8_arm_reverse_sequence_s8(void)
{
    reverse_sequence_full_and_single_s8_arm_reverse_sequence_s8();
}

void test_reverse_sequence_axis0_s8_arm_reverse_sequence_s8(void)
{
    reverse_sequence_axis0_s8_arm_reverse_sequence_s8();
}

/* Each (sequence, batch) pair owns a contiguous two-byte slice. */
void test_reverse_sequence_batched_slices(void)
{
    const int32_t shape[] = {3, 2, 2};
    const int32_t lengths[] = {3, 2};
    const cmsis_nn_reverse_sequence_params params = {.rank = 3, .shape = shape, .seq_dim = 0, .batch_dim = 1};
    const int8_t input[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const int8_t expected[] = {8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 10, 11};
    int8_t output[sizeof(input)];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reverse_sequence_s8(input, lengths, &params, output));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, sizeof(output));
}

void test_reverse_sequence_length_bounds(void)
{
    reverse_sequence_length_bounds();
}

void test_reverse_sequence_shape_bounds(void)
{
    reverse_sequence_shape_bounds();
}

void test_reverse_sequence_output_reuses_shape_storage(void)
{
    int32_t shape_and_output[] = {2, 2, sizeof(int32_t), 0};
    const int32_t lengths[] = {1, 2};
    const cmsis_nn_reverse_sequence_params params = {.rank = 3, .shape = shape_and_output, .seq_dim = 1, .batch_dim = 0};
    const int32_t input[] = {1, 2, 3, 4};
    const int32_t expected[] = {1, 2, 4, 3};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_reverse_sequence_s8((const int8_t *)input, lengths, &params, (int8_t *)shape_and_output));
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, shape_and_output, 4);
}
