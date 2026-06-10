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

#include "../TestData/reverse_sequence_2d_s8/test_data.h"
#include "../TestData/reverse_sequence_axis0_s8/test_data.h"
#include "../TestData/reverse_sequence_full_and_single_s8/test_data.h"
#include "../Utils/validate.h"

void reverse_sequence_2d_s8_arm_reverse_sequence_s8(void)
{
    const int32_t shape[] = REVERSE_SEQUENCE_2D_S8_SHAPE;
    const cmsis_nn_reverse_sequence_params params = {
        .rank = REVERSE_SEQUENCE_2D_S8_RANK,
        .shape = shape,
        .seq_dim = REVERSE_SEQUENCE_2D_S8_SEQ_DIM,
        .batch_dim = REVERSE_SEQUENCE_2D_S8_BATCH_DIM,
    };
    int8_t output[REVERSE_SEQUENCE_2D_S8_SIZE] = {0};

    const arm_cmsis_nn_status result =
        arm_reverse_sequence_s8(reverse_sequence_2d_s8_input_tensor, reverse_sequence_2d_s8_seq_lengths, &params, output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, reverse_sequence_2d_s8_output, REVERSE_SEQUENCE_2D_S8_SIZE));
}

void reverse_sequence_full_and_single_s8_arm_reverse_sequence_s8(void)
{
    const int32_t shape[] = REVERSE_SEQUENCE_FULL_AND_SINGLE_S8_SHAPE;
    const cmsis_nn_reverse_sequence_params params = {
        .rank = REVERSE_SEQUENCE_FULL_AND_SINGLE_S8_RANK,
        .shape = shape,
        .seq_dim = REVERSE_SEQUENCE_FULL_AND_SINGLE_S8_SEQ_DIM,
        .batch_dim = REVERSE_SEQUENCE_FULL_AND_SINGLE_S8_BATCH_DIM,
    };
    int8_t output[REVERSE_SEQUENCE_FULL_AND_SINGLE_S8_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_reverse_sequence_s8(reverse_sequence_full_and_single_s8_input_tensor,
                                                               reverse_sequence_full_and_single_s8_seq_lengths,
                                                               &params,
                                                               output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output,
                              reverse_sequence_full_and_single_s8_output,
                              REVERSE_SEQUENCE_FULL_AND_SINGLE_S8_SIZE));
}

void reverse_sequence_axis0_s8_arm_reverse_sequence_s8(void)
{
    const int32_t shape[] = REVERSE_SEQUENCE_AXIS0_S8_SHAPE;
    const cmsis_nn_reverse_sequence_params params = {
        .rank = REVERSE_SEQUENCE_AXIS0_S8_RANK,
        .shape = shape,
        .seq_dim = REVERSE_SEQUENCE_AXIS0_S8_SEQ_DIM,
        .batch_dim = REVERSE_SEQUENCE_AXIS0_S8_BATCH_DIM,
    };
    int8_t output[REVERSE_SEQUENCE_AXIS0_S8_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_reverse_sequence_s8(reverse_sequence_axis0_s8_input_tensor,
                                                               reverse_sequence_axis0_s8_seq_lengths,
                                                               &params,
                                                               output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, reverse_sequence_axis0_s8_output, REVERSE_SEQUENCE_AXIS0_S8_SIZE));
}