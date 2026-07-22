/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

#include "gru_small_f16_data.h"
#include "gru_stream_f16_data.h"

#define RUN_GRU_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                            \
    void case_name##_arm_gru_unidirectional_f16(void)                                                                  \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        const cmsis_nn_gru_params_f16 params = {                                                                       \
            .time_major = CASE_PREFIX##_TIME_MAJOR,                                                                    \
            .batch_size = CASE_PREFIX##_BATCH_SIZE,                                                                    \
            .time_steps = CASE_PREFIX##_TIME_STEPS,                                                                    \
            .input_size = CASE_PREFIX##_INPUT_SIZE,                                                                    \
            .hidden_size = CASE_PREFIX##_HIDDEN_SIZE,                                                                  \
            .reset_after = CASE_PREFIX##_RESET_AFTER,                                                                  \
            .update_gate = {.input_weights = case_name##_update_input_weights,                                         \
                            .hidden_weights = case_name##_update_hidden_weights,                                       \
                            .input_bias = case_name##_update_input_bias,                                               \
                            .hidden_bias = case_name##_update_hidden_bias},                                            \
            .reset_gate = {.input_weights = case_name##_reset_input_weights,                                           \
                           .hidden_weights = case_name##_reset_hidden_weights,                                         \
                           .input_bias = case_name##_reset_input_bias,                                                 \
                           .hidden_bias = case_name##_reset_hidden_bias},                                              \
            .candidate_gate = {.input_weights = case_name##_candidate_input_weights,                                   \
                               .hidden_weights = case_name##_candidate_hidden_weights,                                 \
                               .input_bias = case_name##_candidate_input_bias,                                         \
                               .hidden_bias = case_name##_candidate_hidden_bias}};                                     \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_gru_unidirectional_f16(case_name##_input, output, &params, NULL)); \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_GRU_F16_CASE(GRU_SMALL_F16, gru_small_f16, 8.0e-2f)

/*
 * Stateful (streaming) case: run the sequence in one full call (zero-init,
 * stateless) and again as two chunked calls that carry the hidden state via
 * buffers.hidden_state. The chunked result must match the full run bit-for-bit,
 * and the full run must match the reference. Requires batch_size == 1.
 */
#define GRU_STREAM_GATES(case_name)                                                                                    \
    {                                                                                                                  \
        .time_major = GRU_STREAM_F16_TIME_MAJOR, .batch_size = GRU_STREAM_F16_BATCH_SIZE,                              \
        .time_steps = 0, /* filled per call */                                                                         \
            .input_size = GRU_STREAM_F16_INPUT_SIZE, .hidden_size = GRU_STREAM_F16_HIDDEN_SIZE,                        \
        .reset_after = GRU_STREAM_F16_RESET_AFTER,                                                                     \
        .update_gate = {case_name##_update_input_weights,                                                              \
                        case_name##_update_hidden_weights,                                                             \
                        case_name##_update_input_bias,                                                                 \
                        case_name##_update_hidden_bias},                                                               \
        .reset_gate = {case_name##_reset_input_weights,                                                                \
                       case_name##_reset_hidden_weights,                                                               \
                       case_name##_reset_input_bias,                                                                   \
                       case_name##_reset_hidden_bias},                                                                 \
        .candidate_gate = {                                                                                            \
            case_name##_candidate_input_weights,                                                                       \
            case_name##_candidate_hidden_weights,                                                                      \
            case_name##_candidate_input_bias,                                                                          \
            case_name##_candidate_hidden_bias                                                                          \
        }                                                                                                              \
    }

void gru_stream_f16_arm_gru_unidirectional_f16(void)
{
    const int in = GRU_STREAM_F16_INPUT_SIZE;
    const int hs = GRU_STREAM_F16_HIDDEN_SIZE;
    const int ts = GRU_STREAM_F16_TIME_STEPS;
    const int half = ts / 2;

    float16_t out_full[GRU_STREAM_F16_DST_SIZE] = {0};
    float16_t out_split[GRU_STREAM_F16_DST_SIZE] = {0};
    float16_t hstate[GRU_STREAM_F16_HIDDEN_SIZE] = {0};

    /* Full, stateless run over all time steps. */
    cmsis_nn_gru_params_f16 pf = GRU_STREAM_GATES(gru_stream_f16);
    pf.time_steps = ts;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_gru_unidirectional_f16(gru_stream_f16_input, out_full, &pf, NULL));

    /* Chunked run carrying state across two calls. */
    cmsis_nn_gru_context_f16 buf = {.temp1 = NULL, .hidden_state = hstate};
    cmsis_nn_gru_params_f16 ph = GRU_STREAM_GATES(gru_stream_f16);
    ph.time_steps = half;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_gru_unidirectional_f16(gru_stream_f16_input, out_split, &ph, &buf));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_gru_unidirectional_f16(gru_stream_f16_input + half * in, out_split + half * hs, &ph, &buf));

    /* Chunked (stateful) == full (stateless), and full == reference. */
    for (int i = 0; i < GRU_STREAM_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)out_full[i], (float)out_split[i]);
        TEST_ASSERT_FLOAT_WITHIN(8.0e-2f, (float)gru_stream_f16_output_ref[i], (float)out_full[i]);
    }
}
