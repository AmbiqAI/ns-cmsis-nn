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

// arm_lstm_unidirectional_f32 must reject a non-positive batch, input or hidden size and a negative time
// step count before it zeroes the cell state, the same contract arm_gru_unidirectional_f32 has. With
// batch_size -1 and hidden_size 4 the previous code computed a wrapped element count for that memset.
void lstm_unidirectional_arg_error_f32(void)
{
    static const float32_t weights[16] = {0};
    static const float32_t bias[4] = {0};
    float32_t input[4] = {0};
    float32_t output[4] = {0};
    float32_t cell_state[4] = {0};
    cmsis_nn_lstm_params_f32 params;
    cmsis_nn_lstm_context_f32 buffers = {.cell_state = cell_state};

    memset(&params, 0, sizeof(params));
    params.batch_size = 1;
    params.time_steps = 1;
    params.input_size = 4;
    params.hidden_size = 4;
    cmsis_nn_lstm_gate_f32 *const gates[4] = {
        &params.forget_gate, &params.input_gate, &params.cell_gate, &params.output_gate};
    for (int32_t g = 0; g < 4; g++)
    {
        gates[g]->input_weights = weights;
        gates[g]->hidden_weights = weights;
        gates[g]->bias = bias;
        gates[g]->activation_type = (gates[g] == &params.cell_gate) ? ARM_NN_FLT_ACT_TANH : ARM_NN_FLT_ACT_SIGMOID;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_lstm_unidirectional_f32(input, output, &params, &buffers));

    params.batch_size = -1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_lstm_unidirectional_f32(input, output, &params, &buffers));
    params.batch_size = 1;
    params.hidden_size = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_lstm_unidirectional_f32(input, output, &params, &buffers));
    params.hidden_size = 4;
    params.input_size = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_lstm_unidirectional_f32(input, output, &params, &buffers));
    params.input_size = 4;
    params.time_steps = -1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_lstm_unidirectional_f32(input, output, &params, &buffers));
    params.time_steps = 0;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_lstm_unidirectional_f32(input, output, &params, &buffers));
}
