/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/lstm_large_f16/test_data.h"
#include "../TestData/lstm_match_1_f16/test_data.h"
#include "../TestData/lstm_match_2_f16/test_data.h"
#include "../TestData/lstm_match_one_time_step_f16/test_data.h"
#include "../TestData/lstm_medium_f16/test_data.h"
#include "../TestData/lstm_small_f16/test_data.h"

#define RUN_LSTM_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                           \
    void case_name##_arm_lstm_unidirectional_f16(void)                                                                 \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        float16_t temp1[CASE_PREFIX##_CELL_STATE_SIZE] = {0};                                                          \
        float16_t temp2[CASE_PREFIX##_CELL_STATE_SIZE] = {0};                                                          \
        float16_t cell_state[CASE_PREFIX##_CELL_STATE_SIZE] = {0};                                                     \
        const cmsis_nn_lstm_params_f16 params = {.time_major = CASE_PREFIX##_TIME_MAJOR,                               \
                                                 .batch_size = CASE_PREFIX##_BATCH_SIZE,                               \
                                                 .time_steps = CASE_PREFIX##_TIME_STEPS,                               \
                                                 .input_size = CASE_PREFIX##_INPUT_SIZE,                               \
                                                 .hidden_size = CASE_PREFIX##_HIDDEN_SIZE,                             \
                                                 .cell_clip = (float16_t)CASE_PREFIX##_CELL_CLIP,                      \
                                                 .forget_gate = {.input_weights = case_name##_forget_input_weights,    \
                                                                 .hidden_weights = case_name##_forget_hidden_weights,  \
                                                                 .bias = case_name##_forget_bias,                      \
                                                                 .activation_type = ARM_NN_FLT_ACT_SIGMOID},           \
                                                 .input_gate = {.input_weights = case_name##_input_input_weights,      \
                                                                .hidden_weights = case_name##_input_hidden_weights,    \
                                                                .bias = case_name##_input_bias,                        \
                                                                .activation_type = ARM_NN_FLT_ACT_SIGMOID},            \
                                                 .cell_gate = {.input_weights = case_name##_cell_input_weights,        \
                                                               .hidden_weights = case_name##_cell_hidden_weights,      \
                                                               .bias = case_name##_cell_bias,                          \
                                                               .activation_type = ARM_NN_FLT_ACT_TANH},                \
                                                 .output_gate = {.input_weights = case_name##_output_input_weights,    \
                                                                 .hidden_weights = case_name##_output_hidden_weights,  \
                                                                 .bias = case_name##_output_bias,                      \
                                                                 .activation_type = ARM_NN_FLT_ACT_SIGMOID}};          \
        cmsis_nn_lstm_context_f16 buffers = {.temp1 = temp1, .temp2 = temp2, .cell_state = cell_state};                \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_lstm_unidirectional_f16(case_name##_input, output, &params, &buffers));                  \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_LSTM_F16_CASE(LSTM_SMALL_F16, lstm_small_f16, 8.0e-2f)
RUN_LSTM_F16_CASE(LSTM_MEDIUM_F16, lstm_medium_f16, 1.0e-1f)
RUN_LSTM_F16_CASE(LSTM_LARGE_F16, lstm_large_f16, 1.2e-1f)
RUN_LSTM_F16_CASE(LSTM_MATCH_1_F16, lstm_match_1_f16, 1.2e-1f)
RUN_LSTM_F16_CASE(LSTM_MATCH_2_F16, lstm_match_2_f16, 1.2e-1f)
RUN_LSTM_F16_CASE(LSTM_MATCH_ONE_TIME_STEP_F16, lstm_match_one_time_step_f16, 1.2e-1f)

/*
 * Stateful (streaming) check: run lstm_small_f16 as one full pass (stateless)
 * and again as two time-chunked calls that carry cell_state + hidden_state via
 * the context. Both paths run identical arithmetic, so the chunked result must
 * match the full run bit-for-bit.
 */
#define LSTM_SMALL_F16_STATE_PARAMS(ts)                                                                                \
    {                                                                                                                  \
        .time_major = LSTM_SMALL_F16_TIME_MAJOR, .batch_size = LSTM_SMALL_F16_BATCH_SIZE, .time_steps = (ts),          \
        .input_size = LSTM_SMALL_F16_INPUT_SIZE, .hidden_size = LSTM_SMALL_F16_HIDDEN_SIZE,                            \
        .cell_clip = (float16_t)LSTM_SMALL_F16_CELL_CLIP,                                                              \
        .forget_gate = {.input_weights = lstm_small_f16_forget_input_weights,                                          \
                        .hidden_weights = lstm_small_f16_forget_hidden_weights,                                        \
                        .bias = lstm_small_f16_forget_bias,                                                            \
                        .activation_type = ARM_NN_FLT_ACT_SIGMOID},                                                    \
        .input_gate = {.input_weights = lstm_small_f16_input_input_weights,                                            \
                       .hidden_weights = lstm_small_f16_input_hidden_weights,                                          \
                       .bias = lstm_small_f16_input_bias,                                                              \
                       .activation_type = ARM_NN_FLT_ACT_SIGMOID},                                                     \
        .cell_gate = {.input_weights = lstm_small_f16_cell_input_weights,                                              \
                      .hidden_weights = lstm_small_f16_cell_hidden_weights,                                            \
                      .bias = lstm_small_f16_cell_bias,                                                                \
                      .activation_type = ARM_NN_FLT_ACT_TANH},                                                         \
        .output_gate = {                                                                                               \
            .input_weights = lstm_small_f16_output_input_weights,                                                      \
            .hidden_weights = lstm_small_f16_output_hidden_weights,                                                    \
            .bias = lstm_small_f16_output_bias,                                                                        \
            .activation_type = ARM_NN_FLT_ACT_SIGMOID                                                                  \
        }                                                                                                              \
    }

void lstm_small_f16_stateful_arm_lstm_unidirectional_f16(void)
{
    const int b = LSTM_SMALL_F16_BATCH_SIZE;
    const int hh = LSTM_SMALL_F16_HIDDEN_SIZE;
    const int in = LSTM_SMALL_F16_INPUT_SIZE;
    const int ts = LSTM_SMALL_F16_TIME_STEPS;
    const int h1 = ts / 2; /* first chunk length (time-major) */

    float16_t t1[LSTM_SMALL_F16_CELL_STATE_SIZE], t2[LSTM_SMALL_F16_CELL_STATE_SIZE];
    float16_t cs[LSTM_SMALL_F16_CELL_STATE_SIZE], hs[LSTM_SMALL_F16_CELL_STATE_SIZE];
    float16_t out_full[LSTM_SMALL_F16_DST_SIZE] = {0};
    float16_t out_split[LSTM_SMALL_F16_DST_SIZE] = {0};

    /* Full, stateless run (hidden_state = NULL). */
    cmsis_nn_lstm_context_f16 bf = {.temp1 = t1, .temp2 = t2, .cell_state = cs, .hidden_state = NULL};
    cmsis_nn_lstm_params_f16 pf = LSTM_SMALL_F16_STATE_PARAMS(ts);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_lstm_unidirectional_f16(lstm_small_f16_input, out_full, &pf, &bf));

    /* Chunked streaming run carrying cell + hidden state across two calls. */
    memset(cs, 0, sizeof(cs));
    memset(hs, 0, sizeof(hs));
    cmsis_nn_lstm_context_f16 bs = {.temp1 = t1, .temp2 = t2, .cell_state = cs, .hidden_state = hs};
    cmsis_nn_lstm_params_f16 p1 = LSTM_SMALL_F16_STATE_PARAMS(h1);
    cmsis_nn_lstm_params_f16 p2 = LSTM_SMALL_F16_STATE_PARAMS(ts - h1);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_lstm_unidirectional_f16(lstm_small_f16_input, out_split, &p1, &bs));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_lstm_unidirectional_f16(
                          lstm_small_f16_input + (size_t)h1 * b * in, out_split + (size_t)h1 * b * hh, &p2, &bs));

    for (int i = 0; i < LSTM_SMALL_F16_DST_SIZE; ++i)
    {
        uint16_t fa, sa;
        memcpy(&fa, &out_full[i], sizeof(fa));
        memcpy(&sa, &out_split[i], sizeof(sa));
        TEST_ASSERT_EQUAL_HEX16(fa, sa);
    }
}
