/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <math.h>
#include <string.h>
#include <unity.h>

// Every hidden unit of one LSTM step must take the same tanh implementation. With hidden_size = 10
// an MVE build used to run units 0..7 through the vector LUT and units 8..9 through the scalar
// rational approximation, so identical units in one tensor came out different (#315). The gates
// are driven by their biases alone, so every unit sees the same cell state and must produce the
// same output bit pattern, on every build.
#define LSTM_STEP_F32_HIDDEN 10

static void lstm_step_f32_fill_gate(cmsis_nn_lstm_gate_f32 *gate,
                                    const float32_t *weights,
                                    const float32_t *bias,
                                    arm_nn_activation_type_flt activation)
{
    gate->input_weights = weights;
    gate->hidden_weights = weights;
    gate->bias = bias;
    gate->activation_type = activation;
}

void lstm_step_uniform_units_f32(void)
{
    static const float32_t zero_weights[LSTM_STEP_F32_HIDDEN * LSTM_STEP_F32_HIDDEN] = {0};
    static const float32_t bias_open[LSTM_STEP_F32_HIDDEN] = {(float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f};
    static const float32_t bias_closed[LSTM_STEP_F32_HIDDEN] = {(float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f};
    const float32_t input[1] = {(float32_t)0.0f};
    // Both state arrays carry a guard region past hidden_size so an over-wide predicated store is caught.
    float32_t cell_state[LSTM_STEP_F32_HIDDEN + 8];
    float32_t hidden_out[LSTM_STEP_F32_HIDDEN + 8];

    cmsis_nn_lstm_params_f32 params;
    memset(&params, 0, sizeof(params));
    params.batch_size = 1;
    params.time_steps = 1;
    params.input_size = 1;
    params.hidden_size = LSTM_STEP_F32_HIDDEN;
    params.cell_clip = (float32_t)0.0f;
    // forget ~1 and output ~1 (sigmoid(8)), input ~0 (sigmoid(-8)): c stays at 1.5, h = o * tanh(1.5)
    lstm_step_f32_fill_gate(&params.forget_gate, zero_weights, bias_open, ARM_NN_FLT_ACT_SIGMOID);
    lstm_step_f32_fill_gate(&params.input_gate, zero_weights, bias_closed, ARM_NN_FLT_ACT_SIGMOID);
    lstm_step_f32_fill_gate(&params.cell_gate, zero_weights, bias_open, ARM_NN_FLT_ACT_TANH);
    lstm_step_f32_fill_gate(&params.output_gate, zero_weights, bias_open, ARM_NN_FLT_ACT_SIGMOID);

    for (int32_t h = 0; h < LSTM_STEP_F32_HIDDEN; h++)
    {
        cell_state[h] = (float32_t)1.5f;
        hidden_out[h] = (float32_t)0.0f;
    }
    for (int32_t h = LSTM_STEP_F32_HIDDEN; h < LSTM_STEP_F32_HIDDEN + 8; h++)
    {
        cell_state[h] = (float32_t)-7.0f;
        hidden_out[h] = (float32_t)-7.0f;
    }
    cmsis_nn_lstm_context_f32 buffers = {.cell_state = cell_state};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_lstm_step_f32(input, NULL, hidden_out, &params, &buffers, 1));

    // Every unit must carry the same bit pattern as unit 0, and unit 0 must be tanh(1.5) within the
    // tolerance of the least accurate leg the library ships for this precision.
    uint32_t unit_0_bits;
    memcpy(&unit_0_bits, &hidden_out[0], sizeof(unit_0_bits));
    for (int32_t h = 1; h < LSTM_STEP_F32_HIDDEN; h++)
    {
        uint32_t unit_bits;
        memcpy(&unit_bits, &hidden_out[h], sizeof(unit_bits));
        TEST_ASSERT_EQUAL_HEX32(unit_0_bits, unit_bits);
    }
    TEST_ASSERT_FLOAT_WITHIN(3.0e-2f, tanhf(1.5f), (float32_t)hidden_out[0]);
    for (int32_t h = 0; h < LSTM_STEP_F32_HIDDEN; h++)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-3f, 1.5f, (float32_t)cell_state[h]);
    }
    for (int32_t h = LSTM_STEP_F32_HIDDEN; h < LSTM_STEP_F32_HIDDEN + 8; h++)
    {
        TEST_ASSERT_EQUAL_FLOAT(-7.0f, (float32_t)cell_state[h]);
        TEST_ASSERT_EQUAL_FLOAT(-7.0f, (float32_t)hidden_out[h]);
    }
}

// Two batches with batch_offset 2 and a cell clip: the partial predicated store of batch 0 must not
// touch batch 1's state, and both batches must agree unit for unit. The clip of 2.0 exercises the
// clamp without pinning the state: a clip at or below 1.5 would snap every unit to the same exact
// value on any build and hide the vector-body / scalar-tail split this suite exists to catch.
void lstm_step_two_batches_clipped_f32(void)
{
    static const float32_t zero_weights[LSTM_STEP_F32_HIDDEN * LSTM_STEP_F32_HIDDEN] = {0};
    static const float32_t bias_open[LSTM_STEP_F32_HIDDEN] = {(float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f,
                                                              (float32_t)8.0f};
    static const float32_t bias_closed[LSTM_STEP_F32_HIDDEN] = {(float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f,
                                                                (float32_t)-8.0f};
    const float32_t input[4] = {(float32_t)0.0f, (float32_t)0.0f, (float32_t)0.0f, (float32_t)0.0f};
    float32_t cell_state[2 * LSTM_STEP_F32_HIDDEN];
    float32_t hidden_out[4 * LSTM_STEP_F32_HIDDEN];

    cmsis_nn_lstm_params_f32 params;
    memset(&params, 0, sizeof(params));
    params.batch_size = 2;
    params.time_steps = 2;
    params.input_size = 1;
    params.hidden_size = LSTM_STEP_F32_HIDDEN;
    params.cell_clip = (float32_t)2.0f;
    lstm_step_f32_fill_gate(&params.forget_gate, zero_weights, bias_open, ARM_NN_FLT_ACT_SIGMOID);
    lstm_step_f32_fill_gate(&params.input_gate, zero_weights, bias_closed, ARM_NN_FLT_ACT_SIGMOID);
    lstm_step_f32_fill_gate(&params.cell_gate, zero_weights, bias_open, ARM_NN_FLT_ACT_TANH);
    lstm_step_f32_fill_gate(&params.output_gate, zero_weights, bias_open, ARM_NN_FLT_ACT_SIGMOID);

    for (int32_t i = 0; i < 2 * LSTM_STEP_F32_HIDDEN; i++)
    {
        cell_state[i] = (float32_t)1.5f;
    }
    for (int32_t i = 0; i < 4 * LSTM_STEP_F32_HIDDEN; i++)
    {
        hidden_out[i] = (float32_t)-7.0f;
    }
    cmsis_nn_lstm_context_f32 buffers = {.cell_state = cell_state};

    // batch_offset 2: batch b's output row lives at hidden_out + b * 2 * hidden_size (batch-major, 2 steps)
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_lstm_step_f32(input, NULL, hidden_out, &params, &buffers, 2));

    for (int32_t b = 0; b < 2; b++)
    {
        const float32_t *row = hidden_out + b * 2 * LSTM_STEP_F32_HIDDEN;
        uint32_t unit_0_bits;
        memcpy(&unit_0_bits, &row[0], sizeof(unit_0_bits));
        for (int32_t h = 1; h < LSTM_STEP_F32_HIDDEN; h++)
        {
            uint32_t unit_bits;
            memcpy(&unit_bits, &row[h], sizeof(unit_bits));
            TEST_ASSERT_EQUAL_HEX32(unit_0_bits, unit_bits);
        }
        TEST_ASSERT_FLOAT_WITHIN(3.0e-2f, tanhf(1.5f), (float32_t)row[0]);
        for (int32_t h = 0; h < LSTM_STEP_F32_HIDDEN; h++)
        {
            TEST_ASSERT_FLOAT_WITHIN(2.0e-3f, 1.5f, (float32_t)cell_state[b * LSTM_STEP_F32_HIDDEN + h]);
            // the second time-step slot of each batch row is untouched by a single step
            TEST_ASSERT_EQUAL_FLOAT(-7.0f, (float32_t)row[LSTM_STEP_F32_HIDDEN + h]);
        }
    }
}

void lstm_step_arg_error_f32(void)
{
    float32_t data[LSTM_STEP_F32_HIDDEN] = {0};
    cmsis_nn_lstm_params_f32 params;
    memset(&params, 0, sizeof(params));
    params.batch_size = 1;
    params.input_size = 1;
    params.hidden_size = LSTM_STEP_F32_HIDDEN;
    cmsis_nn_lstm_context_f32 buffers = {.cell_state = data};
    cmsis_nn_lstm_context_f32 no_cell = {.cell_state = NULL};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_lstm_step_f32(NULL, NULL, data, &params, &buffers, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_lstm_step_f32(data, NULL, data, &params, &no_cell, 1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_lstm_step_f32(data, NULL, data, &params, &buffers, 0));
}
