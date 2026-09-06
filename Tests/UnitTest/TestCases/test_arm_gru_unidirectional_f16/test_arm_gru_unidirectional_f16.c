/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
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
       r . h_prev vector of hidden_size elements, reused across batches and time steps. */
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

/*
 * Shape sweep against an in-test float64 reference (#251): hidden sizes that cover one lane, a partial
 * block, a full block plus tail and several blocks, each with reset_after 0 and 1, with and without biases,
 * batch-major and time-major. Data is a fixed LCG stream so the golden is reproducible without the kernel;
 * h_prev is NULL on the first time step of every case. Tolerance as for the generated cases above.
 */
#define GRU_SWEEP_F16_MAX_HIDDEN 100
#define GRU_SWEEP_F16_INPUT 3
#define GRU_SWEEP_F16_BATCH 2
#define GRU_SWEEP_F16_STEPS 3
#define GRU_SWEEP_F16_IO (GRU_SWEEP_F16_BATCH * GRU_SWEEP_F16_STEPS)

static float16_t gru_sweep_f16_w_in[3][GRU_SWEEP_F16_MAX_HIDDEN * GRU_SWEEP_F16_INPUT];
static float16_t gru_sweep_f16_w_hid[3][GRU_SWEEP_F16_MAX_HIDDEN * GRU_SWEEP_F16_MAX_HIDDEN];
static float16_t gru_sweep_f16_b_in[3][GRU_SWEEP_F16_MAX_HIDDEN];
static float16_t gru_sweep_f16_b_hid[3][GRU_SWEEP_F16_MAX_HIDDEN];
static float16_t gru_sweep_f16_x[GRU_SWEEP_F16_IO * GRU_SWEEP_F16_INPUT];
static float16_t gru_sweep_f16_out[GRU_SWEEP_F16_IO * GRU_SWEEP_F16_MAX_HIDDEN];
static double gru_sweep_f16_ref[GRU_SWEEP_F16_IO * GRU_SWEEP_F16_MAX_HIDDEN];
static float16_t gru_sweep_f16_temp1[GRU_SWEEP_F16_MAX_HIDDEN];

static uint32_t gru_sweep_f16_lcg;

static float16_t gru_sweep_f16_rand(float scale)
{
    gru_sweep_f16_lcg = gru_sweep_f16_lcg * 1664525u + 1013904223u;
    return (float16_t)(scale * ((float)(gru_sweep_f16_lcg >> 8) * (2.0f / 16777216.0f) - 1.0f));
}

static void gru_sweep_f16_fill(float16_t *dst, int32_t n, float scale)
{
    for (int32_t i = 0; i < n; i++)
    {
        dst[i] = gru_sweep_f16_rand(scale);
    }
}

static double gru_sweep_f16_dot(const float16_t *w, const double *v, int32_t n)
{
    double acc = 0.0;
    for (int32_t k = 0; k < n; k++)
    {
        acc += (double)(float)w[k] * v[k];
    }
    return acc;
}

/* Float64 GRU with exact activations; NULL bias pointers contribute zero, as in the kernel. */
static void gru_sweep_f16_reference(const cmsis_nn_gru_params_f16 *p, const float16_t *x, double *out)
{
    const int32_t B = p->batch_size, T = p->time_steps, I = p->input_size, H = p->hidden_size;
    const cmsis_nn_gru_gate_f16 *g[3] = {&p->update_gate, &p->reset_gate, &p->candidate_gate};
    double h[GRU_SWEEP_F16_MAX_HIDDEN], h_new[GRU_SWEEP_F16_MAX_HIDDEN];
    double xd[GRU_SWEEP_F16_INPUT], r[GRU_SWEEP_F16_MAX_HIDDEN], rh[GRU_SWEEP_F16_MAX_HIDDEN];

    for (int32_t b = 0; b < B; b++)
    {
        for (int32_t j = 0; j < H; j++)
        {
            h[j] = 0.0;
        }
        for (int32_t t = 0; t < T; t++)
        {
            const int32_t row = p->time_major ? (t * B + b) : (b * T + t);
            for (int32_t i = 0; i < I; i++)
            {
                xd[i] = (double)(float)x[row * I + i];
            }
            for (int32_t j = 0; j < H; j++)
            {
                const double r_pre = gru_sweep_f16_dot(g[1]->input_weights + j * I, xd, I) +
                    (g[1]->input_bias ? (double)(float)g[1]->input_bias[j] : 0.0) +
                    gru_sweep_f16_dot(g[1]->hidden_weights + j * H, h, H) +
                    (g[1]->hidden_bias ? (double)(float)g[1]->hidden_bias[j] : 0.0);
                r[j] = 1.0 / (1.0 + exp(-r_pre));
                rh[j] = r[j] * h[j];
            }
            for (int32_t j = 0; j < H; j++)
            {
                const double z_pre = gru_sweep_f16_dot(g[0]->input_weights + j * I, xd, I) +
                    (g[0]->input_bias ? (double)(float)g[0]->input_bias[j] : 0.0) +
                    gru_sweep_f16_dot(g[0]->hidden_weights + j * H, h, H) +
                    (g[0]->hidden_bias ? (double)(float)g[0]->hidden_bias[j] : 0.0);
                const double z = 1.0 / (1.0 + exp(-z_pre));
                const double xn = gru_sweep_f16_dot(g[2]->input_weights + j * I, xd, I) +
                    (g[2]->input_bias ? (double)(float)g[2]->input_bias[j] : 0.0);
                const double bhn = g[2]->hidden_bias ? (double)(float)g[2]->hidden_bias[j] : 0.0;
                double n_pre;
                if (p->reset_after)
                {
                    n_pre = xn + r[j] * (gru_sweep_f16_dot(g[2]->hidden_weights + j * H, h, H) + bhn);
                }
                else
                {
                    n_pre = xn + gru_sweep_f16_dot(g[2]->hidden_weights + j * H, rh, H) + bhn;
                }
                h_new[j] = z * h[j] + (1.0 - z) * tanh(n_pre);
            }
            for (int32_t j = 0; j < H; j++)
            {
                h[j] = h_new[j];
                out[row * H + j] = h[j];
            }
        }
    }
}

static void gru_sweep_f16_case(int32_t hidden, int32_t reset_after, int32_t use_bias, int32_t time_major)
{
    const int32_t input = GRU_SWEEP_F16_INPUT;
    gru_sweep_f16_lcg = 0x9E3779B9u ^ (uint32_t)(hidden * 8 + reset_after * 4 + use_bias * 2 + time_major);
    cmsis_nn_gru_params_f16 p;
    memset(&p, 0, sizeof(p));
    p.time_major = time_major;
    p.batch_size = GRU_SWEEP_F16_BATCH;
    p.time_steps = GRU_SWEEP_F16_STEPS;
    p.input_size = input;
    p.hidden_size = hidden;
    p.reset_after = reset_after;
    cmsis_nn_gru_gate_f16 *g[3] = {&p.update_gate, &p.reset_gate, &p.candidate_gate};
    for (int32_t i = 0; i < 3; i++)
    {
        gru_sweep_f16_fill(gru_sweep_f16_w_in[i], hidden * input, 0.5f / sqrtf((float)input));
        gru_sweep_f16_fill(gru_sweep_f16_w_hid[i], hidden * hidden, 0.5f / sqrtf((float)hidden));
        gru_sweep_f16_fill(gru_sweep_f16_b_in[i], hidden, 0.5f);
        gru_sweep_f16_fill(gru_sweep_f16_b_hid[i], hidden, 0.5f);
        g[i]->input_weights = gru_sweep_f16_w_in[i];
        g[i]->hidden_weights = gru_sweep_f16_w_hid[i];
        g[i]->input_bias = use_bias ? gru_sweep_f16_b_in[i] : NULL;
        g[i]->hidden_bias = use_bias ? gru_sweep_f16_b_hid[i] : NULL;
    }
    gru_sweep_f16_fill(gru_sweep_f16_x, GRU_SWEEP_F16_IO * input, 1.0f);

    cmsis_nn_gru_context_f16 buffers = {.temp1 = gru_sweep_f16_temp1, .hidden_state = NULL};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_gru_unidirectional_f16(gru_sweep_f16_x, gru_sweep_f16_out, &p, &buffers));
    gru_sweep_f16_reference(&p, gru_sweep_f16_x, gru_sweep_f16_ref);
    for (int32_t i = 0; i < GRU_SWEEP_F16_IO * hidden; i++)
    {
        TEST_ASSERT_FLOAT_WITHIN(8.0e-2f, (float)gru_sweep_f16_ref[i], (float)gru_sweep_f16_out[i]);
    }
}

void gru_sweep_f16_arm_gru_unidirectional_f16(void)
{
    static const int32_t hidden_sizes[] = {1, 5, 13, 31, 100};
    for (size_t i = 0; i < sizeof(hidden_sizes) / sizeof(hidden_sizes[0]); i++)
    {
        for (int32_t reset_after = 0; reset_after < 2; reset_after++)
        {
            for (int32_t use_bias = 0; use_bias < 2; use_bias++)
            {
                for (int32_t time_major = 0; time_major < 2; time_major++)
                {
                    gru_sweep_f16_case(hidden_sizes[i], reset_after, use_bias, time_major);
                }
            }
        }
    }
}
