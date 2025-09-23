/*
 * SPDX-FileCopyrightText: Copyright 2010-2022
 * Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "unity.h"
#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>

#include "../Utils/validate.h"

/* Per-test generated data */
#include "../TestData/hard_swish_basic_s16/test_data.h"
#include "../TestData/hard_swish_sweep_edges_s16/test_data.h"
#include "../TestData/hard_swish_plateau_edges_s16/test_data.h"
#include "../TestData/hard_swish_dense_mixture_s16/test_data.h"
#include "../TestData/hard_swish_adversarial_step_s16/test_data.h"
#include "../TestData/hard_swish_batched_wide_s16/test_data.h"

#define REPEAT_NUM (1)

/* Helper to generate a Unity test for a given dataset.
 * NAME_UP:  UPPER_SNAKE prefix for the #defines
 * name_lc:  lower_snake prefix for the input/output symbols
 *
 * Expects the following macros from the generated headers:
 *   NAME_UP_INPUT_OFFSET
 *   NAME_UP_OUTPUT_OFFSET
 *   NAME_UP_RELU_MULTIPLIER_FP
 *   NAME_UP_RELU_MULTIPLIER_EXP
 *   NAME_UP_OUTPUT_MULTIPLIER_FP
 *   NAME_UP_OUTPUT_MULTIPLIER_EXP
 *   NAME_UP_OUTPUT_MULTIPLIER
 *   NAME_UP_OUTPUT_SHIFT
 *   NAME_UP_RELU_Q3
 *   NAME_UP_RELU_Q6
 *   NAME_UP_OUTPUT_LEN
 * and arrays:
 *   name_lc_input_tensor
 *   name_lc_output
 */
#define GEN_HS_TEST(NAME_UP, name_lc)                                                          \
    void name_lc##_arm_hard_swish_s16(void)                                                    \
    {                                                                                          \
        const int32_t output_size = NAME_UP##_OUTPUT_LEN;                                      \
                                                                                               \
        const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;                             \
        const int16_t *input_data = name_lc##_input_tensor;                                    \
        int16_t output[NAME_UP##_OUTPUT_LEN] = {0};                                            \
                                                                                               \
                                                                                               \
        for (int r = 0; r < REPEAT_NUM; ++r)                                                   \
        {                                                                                      \
            const arm_cmsis_nn_status result =                                                 \
                arm_hard_swish_precise_s16(                                                    \
                    input_data,                                                                \
                    NAME_UP##_INPUT_OFFSET,                                                    \
                    NAME_UP##_OUTPUT_OFFSET,                                                   \
                    NAME_UP##_OUTPUT_MULTIPLIER,                                               \
                    NAME_UP##_OUTPUT_SHIFT,                                                    \
                    NAME_UP##_RELU_Q3,                                                         \
                    NAME_UP##_RELU_Q6,                                                         \
                    0, /* prescale */                                                          \
                    output,                                                                    \
                    output_size                                                                \
                );                                                                             \
            TEST_ASSERT_EQUAL(expected, result);                                               \
        }                                                                                      \
                                                                                               \
        /* Allow small off-by-one since TFLM uses different rounding */                        \
        TEST_ASSERT_TRUE(validate_tol_s16(output, name_lc##_output, output_size, 1));          \
    }

/* Instantiate tests */
GEN_HS_TEST(HARD_SWISH_BASIC_S16,            hard_swish_basic_s16)
GEN_HS_TEST(HARD_SWISH_SWEEP_EDGES_S16,      hard_swish_sweep_edges_s16)
GEN_HS_TEST(HARD_SWISH_PLATEAU_EDGES_S16,    hard_swish_plateau_edges_s16)
GEN_HS_TEST(HARD_SWISH_DENSE_MIXTURE_S16,    hard_swish_dense_mixture_s16)
GEN_HS_TEST(HARD_SWISH_ADVERSARIAL_STEP_S16, hard_swish_adversarial_step_s16)
GEN_HS_TEST(HARD_SWISH_BATCHED_WIDE_S16,     hard_swish_batched_wide_s16)
