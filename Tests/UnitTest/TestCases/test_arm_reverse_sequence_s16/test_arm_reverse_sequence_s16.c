/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "arm_nn_types.h"
#include "unity.h"

#include "../TestData/reverse_sequence_2d_s16/test_data.h"

void reverse_sequence_2d_arm_reverse_sequence_s16(void)
{
    int16_t output[REVERSE_SEQUENCE_2D_S16_SIZE] = {0};

    const cmsis_nn_reverse_sequence_params params = {
        .rank = REVERSE_SEQUENCE_2D_S16_RANK,
        .shape = reverse_sequence_2d_s16_shape,
        .seq_dim = REVERSE_SEQUENCE_2D_S16_SEQ_DIM,
        .batch_dim = REVERSE_SEQUENCE_2D_S16_BATCH_DIM,
    };

    const arm_cmsis_nn_status result = arm_reverse_sequence_s16(
        reverse_sequence_2d_s16_input, reverse_sequence_2d_s16_seq_lengths, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(reverse_sequence_2d_s16_output, output, REVERSE_SEQUENCE_2D_S16_SIZE);
}
