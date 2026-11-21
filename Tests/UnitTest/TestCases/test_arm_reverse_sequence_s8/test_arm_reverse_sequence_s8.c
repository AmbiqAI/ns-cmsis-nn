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

void reverse_sequence_length_bounds(void)
{
    const int32_t shape[] = {2, 2};
    const cmsis_nn_reverse_sequence_params params = {.rank = 2, .shape = shape, .seq_dim = 1, .batch_dim = 0};
    const int8_t input[] = {1, 2, 3, 4};
    int8_t output[4];
    int32_t lengths[] = {0, 2};
    const int8_t expected[] = {1, 2, 4, 3};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reverse_sequence_s8(input, lengths, &params, output));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, 4);

    const int32_t invalid[] = {-1, 3, INT32_MAX};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
    {
        lengths[1] = invalid[i];
        output[0] = 42;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_reverse_sequence_s8(input, lengths, &params, output));
        TEST_ASSERT_EQUAL_INT8(42, output[0]);
    }
}

void reverse_sequence_shape_bounds(void)
{
    int32_t shape[] = {1, 0};
    cmsis_nn_reverse_sequence_params params = {.rank = 2, .shape = shape, .seq_dim = 1, .batch_dim = 0};
    const int8_t input = 0;
    int8_t output = 42;
    const int32_t length = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reverse_sequence_s8(&input, &length, &params, &output));
    TEST_ASSERT_EQUAL_INT8(42, output);
    shape[0] = 2;
    shape[1] = INT32_MAX;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_reverse_sequence_s8(&input, &length, &params, &output));
    shape[0] = 1;
    shape[1] = 1;
    params.seq_dim = params.batch_dim;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_reverse_sequence_s8(&input, &length, &params, &output));
    TEST_ASSERT_EQUAL_INT8(42, output);
    params.seq_dim = 2;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_reverse_sequence_s8(&input, &length, &params, &output));
}
