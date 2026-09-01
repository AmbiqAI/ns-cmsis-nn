/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <stdlib.h>
#include <unity.h>

#include "gru_prereset_f16_data.h"
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
 * buffers.hidden_state. Because both paths execute the identical per-step
 * arithmetic in the same order, the chunked result must match the full run
 * bit-for-bit; the full run must match the reference within tolerance.
 * Requires batch_size == 1.
 */
#define GRU_STREAM_GATES(case_name)                                                                                    \
    {                                                                                                                  \
        .time_major = GRU_STREAM_F16_TIME_MAJOR, .batch_size = GRU_STREAM_F16_BATCH_SIZE,                              \
        .time_steps = 0, /* filled per call */                                                                         \
            .input_size = GRU_STREAM_F16_INPUT_SIZE, .hidden_size = GRU_STREAM_F16_HIDDEN_SIZE,                        \
        .reset_after = GRU_STREAM_F16_RESET_AFTER,                                                                     \
        .update_gate = {.input_weights = case_name##_update_input_weights,                                             \
                        .hidden_weights = case_name##_update_hidden_weights,                                           \
                        .input_bias = case_name##_update_input_bias,                                                   \
                        .hidden_bias = case_name##_update_hidden_bias},                                                \
        .reset_gate = {.input_weights = case_name##_reset_input_weights,                                               \
                       .hidden_weights = case_name##_reset_hidden_weights,                                             \
                       .input_bias = case_name##_reset_input_bias,                                                     \
                       .hidden_bias = case_name##_reset_hidden_bias},                                                  \
        .candidate_gate = {                                                                                            \
            .input_weights = case_name##_candidate_input_weights,                                                      \
            .hidden_weights = case_name##_candidate_hidden_weights,                                                    \
            .input_bias = case_name##_candidate_input_bias,                                                            \
            .hidden_bias = case_name##_candidate_hidden_bias                                                           \
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

    /* Chunked run carrying state across two calls. reset_after != 0 needs no scratch: the published query
       must agree that temp1 may stay NULL. */
    cmsis_nn_gru_context_f16 buf = {.temp1 = NULL, .hidden_state = hstate};
    cmsis_nn_gru_params_f16 ph = GRU_STREAM_GATES(gru_stream_f16);
    TEST_ASSERT_EQUAL(0, arm_gru_unidirectional_f16_temp1_get_buffer_size(&ph));
    ph.time_steps = half;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_gru_unidirectional_f16(gru_stream_f16_input, out_split, &ph, &buf));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_gru_unidirectional_f16(gru_stream_f16_input + half * in, out_split + half * hs, &ph, &buf));

    /* Chunked (stateful) matches full (stateless) bit-for-bit; full matches reference. */
    for (int i = 0; i < GRU_STREAM_F16_DST_SIZE; ++i)
    {
        uint16_t full_bits, split_bits;
        memcpy(&full_bits, &out_full[i], sizeof(full_bits));
        memcpy(&split_bits, &out_split[i], sizeof(split_bits));
        TEST_ASSERT_EQUAL_HEX16(full_bits, split_bits);
        TEST_ASSERT_FLOAT_WITHIN(8.0e-2f, (float)gru_stream_f16_output_ref[i], (float)out_full[i]);
    }
}

/*
 * Pre-reset case (reset_after == 0): the reset gate is applied before the
 * recurrent matmul, which requires the buffers->temp1 scratch. Validates both
 * the argument-error path (missing temp1) and numerical correctness against a
 * Keras GRU(reset_after=False) reference.
 */
void gru_prereset_f16_arm_gru_unidirectional_f16(void)
{
    float16_t output[GRU_PRERESET_F16_DST_SIZE] = {0};

    const cmsis_nn_gru_params_f16 params = {
        .time_major = GRU_PRERESET_F16_TIME_MAJOR,
        .batch_size = GRU_PRERESET_F16_BATCH_SIZE,
        .time_steps = GRU_PRERESET_F16_TIME_STEPS,
        .input_size = GRU_PRERESET_F16_INPUT_SIZE,
        .hidden_size = GRU_PRERESET_F16_HIDDEN_SIZE,
        .reset_after = GRU_PRERESET_F16_RESET_AFTER,
        .update_gate = {.input_weights = gru_prereset_f16_update_input_weights,
                        .hidden_weights = gru_prereset_f16_update_hidden_weights,
                        .input_bias = gru_prereset_f16_update_input_bias,
                        .hidden_bias = gru_prereset_f16_update_hidden_bias},
        .reset_gate = {.input_weights = gru_prereset_f16_reset_input_weights,
                       .hidden_weights = gru_prereset_f16_reset_hidden_weights,
                       .input_bias = gru_prereset_f16_reset_input_bias,
                       .hidden_bias = gru_prereset_f16_reset_hidden_bias},
        .candidate_gate = {.input_weights = gru_prereset_f16_candidate_input_weights,
                           .hidden_weights = gru_prereset_f16_candidate_hidden_weights,
                           .input_bias = gru_prereset_f16_candidate_input_bias,
                           .hidden_bias = gru_prereset_f16_candidate_hidden_bias}};

    /* reset_after == 0 needs temp1: missing scratch must be rejected. */
    cmsis_nn_gru_context_f16 no_scratch = {.temp1 = NULL, .hidden_state = NULL};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_gru_unidirectional_f16(gru_prereset_f16_input, output, &params, &no_scratch));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_gru_unidirectional_f16(gru_prereset_f16_input, output, &params, NULL));

    /* Non-positive dimensions must be rejected, not silently produce output.
     * Mirrors the f32 suite: each case mutates exactly one field so only the
     * dimension check can reject it. */
    /* The published query must agree with the size this test derives by hand: the pre-reset path stages one
       reset-gate vector of hidden_size elements, reused across batches and time steps. */
    const int32_t temp1_size = arm_gru_unidirectional_f16_temp1_get_buffer_size(&params);
    TEST_ASSERT_EQUAL(GRU_PRERESET_F16_HIDDEN_SIZE * (int32_t)sizeof(float16_t), temp1_size);
    float16_t *temp1 = malloc((size_t)temp1_size);
    cmsis_nn_gru_context_f16 scratch_ok = {.temp1 = temp1, .hidden_state = NULL};
    cmsis_nn_gru_params_f16 bad = params;
    bad.input_size = -5;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_gru_unidirectional_f16(gru_prereset_f16_input, output, &bad, &scratch_ok));
    bad = params;
    bad.hidden_size = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_gru_unidirectional_f16(gru_prereset_f16_input, output, &bad, &scratch_ok));
    bad = params;
    bad.batch_size = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_gru_unidirectional_f16(gru_prereset_f16_input, output, &bad, &scratch_ok));
    bad = params;
    bad.time_steps = -1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_gru_unidirectional_f16(gru_prereset_f16_input, output, &bad, &scratch_ok));

    /* With scratch: succeeds and matches the reference. */
    cmsis_nn_gru_context_f16 buffers = {.temp1 = temp1, .hidden_state = NULL};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_gru_unidirectional_f16(gru_prereset_f16_input, output, &params, &buffers));
    free(temp1);
    for (int i = 0; i < GRU_PRERESET_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(8.0e-2f, (float)gru_prereset_f16_output_ref[i], (float)output[i]);
    }
}
