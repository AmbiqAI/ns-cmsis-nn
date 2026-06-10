/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/reverse_sequence_2d_s16/test_data.h"
#include "../TestData/reverse_sequence_axis0_s16/test_data.h"
#include "../TestData/reverse_sequence_full_and_single_s16/test_data.h"
#include "../Utils/validate.h"

void reverse_sequence_2d_s16_arm_reverse_sequence_s16(void)
{
    const int32_t shape[] = REVERSE_SEQUENCE_2D_S16_SHAPE;
    const cmsis_nn_reverse_sequence_params params = {
        .rank = REVERSE_SEQUENCE_2D_S16_RANK,
        .shape = shape,
        .seq_dim = REVERSE_SEQUENCE_2D_S16_SEQ_DIM,
        .batch_dim = REVERSE_SEQUENCE_2D_S16_BATCH_DIM,
    };
    int16_t output[REVERSE_SEQUENCE_2D_S16_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_reverse_sequence_s16(reverse_sequence_2d_s16_input_tensor,
                                                                reverse_sequence_2d_s16_seq_lengths,
                                                                &params,
                                                                output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, reverse_sequence_2d_s16_output, REVERSE_SEQUENCE_2D_S16_SIZE));
}

void reverse_sequence_full_and_single_s16_arm_reverse_sequence_s16(void)
{
    const int32_t shape[] = REVERSE_SEQUENCE_FULL_AND_SINGLE_S16_SHAPE;
    const cmsis_nn_reverse_sequence_params params = {
        .rank = REVERSE_SEQUENCE_FULL_AND_SINGLE_S16_RANK,
        .shape = shape,
        .seq_dim = REVERSE_SEQUENCE_FULL_AND_SINGLE_S16_SEQ_DIM,
        .batch_dim = REVERSE_SEQUENCE_FULL_AND_SINGLE_S16_BATCH_DIM,
    };
    int16_t output[REVERSE_SEQUENCE_FULL_AND_SINGLE_S16_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_reverse_sequence_s16(reverse_sequence_full_and_single_s16_input_tensor,
                                                                reverse_sequence_full_and_single_s16_seq_lengths,
                                                                &params,
                                                                output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                                  reverse_sequence_full_and_single_s16_output,
                                  REVERSE_SEQUENCE_FULL_AND_SINGLE_S16_SIZE));
}

void reverse_sequence_axis0_s16_arm_reverse_sequence_s16(void)
{
    const int32_t shape[] = REVERSE_SEQUENCE_AXIS0_S16_SHAPE;
    const cmsis_nn_reverse_sequence_params params = {
        .rank = REVERSE_SEQUENCE_AXIS0_S16_RANK,
        .shape = shape,
        .seq_dim = REVERSE_SEQUENCE_AXIS0_S16_SEQ_DIM,
        .batch_dim = REVERSE_SEQUENCE_AXIS0_S16_BATCH_DIM,
    };
    int16_t output[REVERSE_SEQUENCE_AXIS0_S16_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_reverse_sequence_s16(reverse_sequence_axis0_s16_input_tensor,
                                                                reverse_sequence_axis0_s16_seq_lengths,
                                                                &params,
                                                                output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, reverse_sequence_axis0_s16_output, REVERSE_SEQUENCE_AXIS0_S16_SIZE));
}