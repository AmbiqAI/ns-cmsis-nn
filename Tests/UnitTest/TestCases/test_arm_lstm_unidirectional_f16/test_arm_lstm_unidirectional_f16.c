/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <string.h>
#include <unity.h>

typedef struct
{
    int32_t batch_size;
    int32_t time_steps;
    int32_t input_size;
    int32_t hidden_size;
} lstm_dims_f16;

// Run a stateless (hidden_state NULL) layer over a 4-element input/output/cell-state set with the given
// dimensions. Every call builds its params from scratch, so a rejected case can never leak into the next one.
static arm_cmsis_nn_status lstm_unidirectional_run_f16(lstm_dims_f16 dims)
{
    static const float16_t weights[16] = {0};
    static const float16_t bias[4] = {0};
    float16_t input[4] = {0};
    float16_t output[4] = {0};
    float16_t cell_state[4] = {0};
    cmsis_nn_lstm_params_f16 params;
    cmsis_nn_lstm_context_f16 buffers = {.cell_state = cell_state};

    memset(&params, 0, sizeof(params));
    params.batch_size = dims.batch_size;
    params.time_steps = dims.time_steps;
    params.input_size = dims.input_size;
    params.hidden_size = dims.hidden_size;
    cmsis_nn_lstm_gate_f16 *const gates[4] = {
        &params.forget_gate, &params.input_gate, &params.cell_gate, &params.output_gate};
    for (int32_t g = 0; g < 4; g++)
    {
        gates[g]->input_weights = weights;
        gates[g]->hidden_weights = weights;
        gates[g]->bias = bias;
        gates[g]->activation_type = (gates[g] == &params.cell_gate) ? ARM_NN_FLT_ACT_TANH : ARM_NN_FLT_ACT_SIGMOID;
    }

    /* The published temp queries must agree that the float implementation needs no temp scratch: the
       context above deliberately leaves temp1/temp2 NULL. */
    TEST_ASSERT_EQUAL(0, arm_lstm_unidirectional_f16_temp1_get_buffer_size(&params));
    TEST_ASSERT_EQUAL(0, arm_lstm_unidirectional_f16_temp2_get_buffer_size(&params));

    return arm_lstm_unidirectional_f16(input, output, &params, &buffers);
}

// The two values that were memory-unsafe before the guard: with hidden_state NULL the kernel zeroes
// batch_size * hidden_size cell-state elements, so -1 * 4 wrapped to a ~4G element count and wrote far past
// the 4-element buffer (ASan: stack-buffer-overflow in arm_memset_f16). Each gets its own test, and they run
// first, so that each one is locked against the old kernel on its own.
void lstm_unidirectional_negative_hidden_size_f16(void)
{
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, lstm_unidirectional_run_f16((lstm_dims_f16){1, 1, 4, -1}));
}

void lstm_unidirectional_negative_batch_size_f16(void)
{
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, lstm_unidirectional_run_f16((lstm_dims_f16){-1, 1, 4, 4}));
}

// The remaining rejected values were memory-safe before the guard but silently ran a degenerate layer:
// a zero batch or hidden size, a non-positive input size (no input term) or a negative time step count
// (no steps). They now return ARG_ERROR, the contract arm_gru_unidirectional_f16 already has, while a valid
// one-step layer and the zero-step no-op still succeed.
void lstm_unidirectional_arg_error_f16(void)
{
    static const lstm_dims_f16 rejected[] = {
        {0, 1, 4, 4},  // batch_size 0
        {1, 1, 4, 0},  // hidden_size 0
        {1, 1, 0, 4},  // input_size 0
        {1, 1, -1, 4}, // input_size -1
        {1, -1, 4, 4}, // time_steps -1
    };

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, lstm_unidirectional_run_f16((lstm_dims_f16){1, 1, 4, 4}));

    for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, lstm_unidirectional_run_f16(rejected[i]));
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, lstm_unidirectional_run_f16((lstm_dims_f16){1, 0, 4, 4}));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, lstm_unidirectional_run_f16((lstm_dims_f16){1, 1, 4, 4}));
}
