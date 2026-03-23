/*
 * Copyright (C) 2026 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/rsqrt_s16/test_data.h"
#include "../Utils/validate.h"

void rsqrt_s16_universal_matches_reference(void)
{
    int16_t output[RSQRT_S16_BLOCK_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_rsqrt_s16_universal(rsqrt_s16_input,
                                                               RSQRT_S16_INPUT_OFFSET,
                                                               output,
                                                               RSQRT_S16_OUT_OFFSET,
                                                               RSQRT_S16_OUT_MULT,
                                                               RSQRT_S16_OUT_SHIFT,
                                                               RSQRT_S16_NEEDS_RESCALE,
                                                               RSQRT_S16_OUT_ACTIVATION_MIN,
                                                               RSQRT_S16_OUT_ACTIVATION_MAX,
                                                               RSQRT_S16_BLOCK_SIZE,
                                                               rsqrt_s16_universal_lut);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(
        validate_tol_s16(output, rsqrt_s16_universal_output, RSQRT_S16_BLOCK_SIZE, RSQRT_S16_UNIVERSAL_TOLERANCE));
}

void rsqrt_s16_per_op_matches_reference(void)
{
    int16_t output[RSQRT_S16_BLOCK_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_rsqrt_s16_per_op(rsqrt_s16_input,
                                                            RSQRT_S16_INPUT_OFFSET,
                                                            output,
                                                            RSQRT_S16_OUT_OFFSET,
                                                            RSQRT_S16_OUT_ACTIVATION_MIN,
                                                            RSQRT_S16_OUT_ACTIVATION_MAX,
                                                            RSQRT_S16_BLOCK_SIZE,
                                                            rsqrt_s16_per_op_lut);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(
        validate_tol_s16(output, rsqrt_s16_per_op_output, RSQRT_S16_BLOCK_SIZE, RSQRT_S16_PER_OP_TOLERANCE));
}

void rsqrt_s16_per_op_applies_input_offset_before_lookup(void)
{
    int16_t output[RSQRT_S16_OFFSET_BLOCK_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_rsqrt_s16_per_op(rsqrt_s16_offset_input,
                                                            RSQRT_S16_OFFSET_INPUT_OFFSET,
                                                            output,
                                                            RSQRT_S16_OFFSET_OUT_OFFSET,
                                                            RSQRT_S16_OFFSET_OUT_ACTIVATION_MIN,
                                                            RSQRT_S16_OFFSET_OUT_ACTIVATION_MAX,
                                                            RSQRT_S16_OFFSET_BLOCK_SIZE,
                                                            rsqrt_s16_per_op_lut);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_tol_s16(output,
                                      rsqrt_s16_offset_per_op_output,
                                      RSQRT_S16_OFFSET_BLOCK_SIZE,
                                      RSQRT_S16_OFFSET_PER_OP_TOLERANCE));
}

void rsqrt_s16_universal_rescales_and_applies_input_offset(void)
{
    int16_t output[RSQRT_S16_RESCALE_BLOCK_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_rsqrt_s16_universal(rsqrt_s16_rescale_input,
                                                               RSQRT_S16_RESCALE_INPUT_OFFSET,
                                                               output,
                                                               RSQRT_S16_RESCALE_OUT_OFFSET,
                                                               RSQRT_S16_RESCALE_OUT_MULT,
                                                               RSQRT_S16_RESCALE_OUT_SHIFT,
                                                               RSQRT_S16_RESCALE_NEEDS_RESCALE,
                                                               RSQRT_S16_RESCALE_OUT_ACTIVATION_MIN,
                                                               RSQRT_S16_RESCALE_OUT_ACTIVATION_MAX,
                                                               RSQRT_S16_RESCALE_BLOCK_SIZE,
                                                               rsqrt_s16_universal_lut);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_tol_s16(output,
                                      rsqrt_s16_rescaled_universal_output,
                                      RSQRT_S16_RESCALE_BLOCK_SIZE,
                                      RSQRT_S16_RESCALED_UNIVERSAL_TOLERANCE));
}

void rsqrt_s16_per_op_applies_input_offset_before_lookup(void)
{
    int16_t output[RSQRT_S16_OFFSET_BLOCK_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_rsqrt_s16_per_op(rsqrt_s16_offset_input,
                                                            RSQRT_S16_OFFSET_INPUT_OFFSET,
                                                            output,
                                                            RSQRT_S16_OFFSET_OUT_OFFSET,
                                                            RSQRT_S16_OFFSET_OUT_ACTIVATION_MIN,
                                                            RSQRT_S16_OFFSET_OUT_ACTIVATION_MAX,
                                                            RSQRT_S16_OFFSET_BLOCK_SIZE,
                                                            rsqrt_s16_per_op_lut);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, rsqrt_s16_offset_per_op_output, RSQRT_S16_OFFSET_BLOCK_SIZE));
}

void rsqrt_s16_universal_rescales_and_applies_input_offset(void)
{
    int16_t output[RSQRT_S16_RESCALE_BLOCK_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_rsqrt_s16_universal(rsqrt_s16_rescale_input,
                                                               RSQRT_S16_RESCALE_INPUT_OFFSET,
                                                               output,
                                                               RSQRT_S16_RESCALE_OUT_OFFSET,
                                                               RSQRT_S16_RESCALE_OUT_MULT,
                                                               RSQRT_S16_RESCALE_OUT_SHIFT,
                                                               RSQRT_S16_RESCALE_NEEDS_RESCALE,
                                                               RSQRT_S16_RESCALE_OUT_ACTIVATION_MIN,
                                                               RSQRT_S16_RESCALE_OUT_ACTIVATION_MAX,
                                                               RSQRT_S16_RESCALE_BLOCK_SIZE,
                                                               rsqrt_s16_universal_lut);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, rsqrt_s16_rescaled_universal_output, RSQRT_S16_RESCALE_BLOCK_SIZE));
}

void rsqrt_s16_rejects_negative_input(void)
{
    int16_t output[RSQRT_S16_NEGATIVE_BLOCK_SIZE] = {0};

    const arm_cmsis_nn_status result = arm_rsqrt_s16_per_op(rsqrt_s16_negative_input,
                                                            RSQRT_S16_INPUT_OFFSET,
                                                            output,
                                                            RSQRT_S16_OUT_OFFSET,
                                                            RSQRT_S16_OUT_ACTIVATION_MIN,
                                                            RSQRT_S16_OUT_ACTIVATION_MAX,
                                                            RSQRT_S16_NEGATIVE_BLOCK_SIZE,
                                                            rsqrt_s16_per_op_lut);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
}
