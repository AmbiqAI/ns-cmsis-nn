/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../TestData/lstm_1_s16/test_data.h"
#include "../TestData/lstm_2_s16/test_data.h"
#include "../TestData/lstm_one_time_step_s16/test_data.h"
#include "../TestData/lstm_stateful_batch_major_multibatch_s16/test_data.h"
#include "../Utils/validate.h"
#include <arm_nnfunctions.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unity.h>

// The temp1/temp2 scratch is sized per test from the published queries rather than a shared
// hidden * batch * time_steps array, whose formula never matched the kernel's actual requirement.

void lstm_1_s16(void)
{
    int16_t output[LSTM_1_S16_BATCH_SIZE * LSTM_1_S16_TIME_STEPS * LSTM_1_S16_HIDDEN_SIZE] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *output_ref = &lstm_1_s16_output[0];
    const int32_t output_ref_size = LSTM_1_S16_BATCH_SIZE * LSTM_1_S16_TIME_STEPS * LSTM_1_S16_HIDDEN_SIZE;

    int64_t input_data_kernel_sum[LSTM_1_S16_HIDDEN_SIZE];
    int64_t forget_data_kernel_sum[LSTM_1_S16_HIDDEN_SIZE];
    int64_t cell_data_kernel_sum[LSTM_1_S16_HIDDEN_SIZE];
    int64_t output_data_kernel_sum[LSTM_1_S16_HIDDEN_SIZE];

    int64_t input_hidden_kernel_sum[LSTM_1_S16_HIDDEN_SIZE];
    int64_t forget_hidden_kernel_sum[LSTM_1_S16_HIDDEN_SIZE];
    int64_t cell_hidden_kernel_sum[LSTM_1_S16_HIDDEN_SIZE];
    int64_t output_hidden_kernel_sum[LSTM_1_S16_HIDDEN_SIZE];

    arm_vector_sum_s8_s64(&input_data_kernel_sum[0],
                          LSTM_1_S16_INPUT_SIZE,
                          LSTM_1_S16_HIDDEN_SIZE,
                          &lstm_1_s16_input_gate_input_weights[0],
                          LSTM_1_S16_INPUT_ZERO_POINT,
                          &lstm_1_s16_input_gate_bias[0]);
    arm_vector_sum_s8_s64(&forget_data_kernel_sum[0],
                          LSTM_1_S16_INPUT_SIZE,
                          LSTM_1_S16_HIDDEN_SIZE,
                          &lstm_1_s16_forget_gate_input_weights[0],
                          LSTM_1_S16_INPUT_ZERO_POINT,
                          &lstm_1_s16_forget_gate_bias[0]);
    arm_vector_sum_s8_s64(&cell_data_kernel_sum[0],
                          LSTM_1_S16_INPUT_SIZE,
                          LSTM_1_S16_HIDDEN_SIZE,
                          &lstm_1_s16_cell_gate_input_weights[0],
                          LSTM_1_S16_INPUT_ZERO_POINT,
                          &lstm_1_s16_cell_gate_bias[0]);
    arm_vector_sum_s8_s64(&output_data_kernel_sum[0],
                          LSTM_1_S16_INPUT_SIZE,
                          LSTM_1_S16_HIDDEN_SIZE,
                          &lstm_1_s16_output_gate_input_weights[0],
                          LSTM_1_S16_INPUT_ZERO_POINT,
                          &lstm_1_s16_output_gate_bias[0]);

    arm_vector_sum_s8_s64(&input_hidden_kernel_sum[0],
                          LSTM_1_S16_HIDDEN_SIZE,
                          LSTM_1_S16_HIDDEN_SIZE,
                          &lstm_1_s16_input_gate_hidden_weights[0],
                          -LSTM_1_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&forget_hidden_kernel_sum[0],
                          LSTM_1_S16_HIDDEN_SIZE,
                          LSTM_1_S16_HIDDEN_SIZE,
                          &lstm_1_s16_forget_gate_hidden_weights[0],
                          -LSTM_1_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&cell_hidden_kernel_sum[0],
                          LSTM_1_S16_HIDDEN_SIZE,
                          LSTM_1_S16_HIDDEN_SIZE,
                          &lstm_1_s16_cell_gate_hidden_weights[0],
                          -LSTM_1_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&output_hidden_kernel_sum[0],
                          LSTM_1_S16_HIDDEN_SIZE,
                          LSTM_1_S16_HIDDEN_SIZE,
                          &lstm_1_s16_output_gate_hidden_weights[0],
                          -LSTM_1_S16_OUTPUT_ZERO_POINT,
                          NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_1_S16_INPUT_GATE_INPUT_MULTIPLIER,
                                           LSTM_1_S16_INPUT_GATE_INPUT_SHIFT,
                                           &lstm_1_s16_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_1_S16_INPUT_GATE_HIDDEN_MULTIPLIER,
                                           LSTM_1_S16_INPUT_GATE_HIDDEN_SHIFT,
                                           &lstm_1_s16_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_1_s16_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_1_S16_FORGET_GATE_INPUT_MULTIPLIER,
                                            LSTM_1_S16_FORGET_GATE_INPUT_SHIFT,
                                            &lstm_1_s16_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_1_S16_FORGET_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_1_S16_FORGET_GATE_HIDDEN_SHIFT,
                                            &lstm_1_s16_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_1_s16_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_1_S16_CELL_GATE_INPUT_MULTIPLIER,
                                          LSTM_1_S16_CELL_GATE_INPUT_SHIFT,
                                          &lstm_1_s16_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_1_S16_CELL_GATE_HIDDEN_MULTIPLIER,
                                          LSTM_1_S16_CELL_GATE_HIDDEN_SHIFT,
                                          &lstm_1_s16_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_1_s16_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_1_S16_OUTPUT_GATE_INPUT_MULTIPLIER,
                                            LSTM_1_S16_OUTPUT_GATE_INPUT_SHIFT,
                                            &lstm_1_s16_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_1_S16_OUTPUT_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_1_S16_OUTPUT_GATE_HIDDEN_SHIFT,
                                            &lstm_1_s16_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_1_s16_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_1_S16_TIME_MAJOR,
                                         LSTM_1_S16_BATCH_SIZE,
                                         LSTM_1_S16_TIME_STEPS,
                                         LSTM_1_S16_INPUT_SIZE,
                                         LSTM_1_S16_HIDDEN_SIZE,
                                         LSTM_1_S16_INPUT_ZERO_POINT,
                                         LSTM_1_S16_FORGET_TO_CELL_MULTIPLIER,
                                         LSTM_1_S16_FORGET_TO_CELL_SHIFT,
                                         LSTM_1_S16_INPUT_TO_CELL_MULTIPLIER,
                                         LSTM_1_S16_INPUT_TO_CELL_SHIFT,
                                         LSTM_1_S16_CELL_CLIP,
                                         LSTM_1_S16_CELL_SCALE_POWER,
                                         LSTM_1_S16_OUTPUT_MULTIPLIER,
                                         LSTM_1_S16_OUTPUT_SHIFT,
                                         LSTM_1_S16_OUTPUT_ZERO_POINT,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    const int32_t temp1_size = arm_lstm_unidirectional_s16_temp1_get_buffer_size(&params);
    const int32_t temp2_size = arm_lstm_unidirectional_s16_temp2_get_buffer_size(&params);
    /* The published queries must agree with the size this test derives by hand from the kernel's writes: the
       temp buffers hold int16_t gate vectors of the step kernel's batch extent (the layer batch only for a
       time-major layer). */
    TEST_ASSERT_EQUAL((LSTM_1_S16_TIME_MAJOR ? LSTM_1_S16_BATCH_SIZE : 1) * LSTM_1_S16_HIDDEN_SIZE * (int32_t)sizeof(int16_t),
                      temp1_size);
    TEST_ASSERT_EQUAL(temp1_size, temp2_size);

    int16_t cell_state[LSTM_1_S16_BATCH_SIZE * LSTM_1_S16_HIDDEN_SIZE];

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = malloc((size_t)temp1_size);
    buffers.temp2 = malloc((size_t)temp2_size);
    buffers.cell_state = cell_state;
    buffers.hidden_state = NULL;

    arm_cmsis_nn_status result = arm_lstm_unidirectional_s16(lstm_1_s16_input_tensor, output, &params, &buffers);

    free(buffers.temp1);
    free(buffers.temp2);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}
void lstm_2_s16(void)
{
    int16_t output[LSTM_2_S16_BATCH_SIZE * LSTM_2_S16_TIME_STEPS * LSTM_2_S16_HIDDEN_SIZE] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *output_ref = &lstm_2_s16_output[0];
    const int32_t output_ref_size = LSTM_2_S16_BATCH_SIZE * LSTM_2_S16_TIME_STEPS * LSTM_2_S16_HIDDEN_SIZE;

    int64_t input_data_kernel_sum[LSTM_2_S16_HIDDEN_SIZE];
    int64_t forget_data_kernel_sum[LSTM_2_S16_HIDDEN_SIZE];
    int64_t cell_data_kernel_sum[LSTM_2_S16_HIDDEN_SIZE];
    int64_t output_data_kernel_sum[LSTM_2_S16_HIDDEN_SIZE];

    int64_t input_hidden_kernel_sum[LSTM_2_S16_HIDDEN_SIZE];
    int64_t forget_hidden_kernel_sum[LSTM_2_S16_HIDDEN_SIZE];
    int64_t cell_hidden_kernel_sum[LSTM_2_S16_HIDDEN_SIZE];
    int64_t output_hidden_kernel_sum[LSTM_2_S16_HIDDEN_SIZE];

    arm_vector_sum_s8_s64(&input_data_kernel_sum[0],
                          LSTM_2_S16_INPUT_SIZE,
                          LSTM_2_S16_HIDDEN_SIZE,
                          &lstm_2_s16_input_gate_input_weights[0],
                          LSTM_2_S16_INPUT_ZERO_POINT,
                          &lstm_2_s16_input_gate_bias[0]);
    arm_vector_sum_s8_s64(&forget_data_kernel_sum[0],
                          LSTM_2_S16_INPUT_SIZE,
                          LSTM_2_S16_HIDDEN_SIZE,
                          &lstm_2_s16_forget_gate_input_weights[0],
                          LSTM_2_S16_INPUT_ZERO_POINT,
                          &lstm_2_s16_forget_gate_bias[0]);
    arm_vector_sum_s8_s64(&cell_data_kernel_sum[0],
                          LSTM_2_S16_INPUT_SIZE,
                          LSTM_2_S16_HIDDEN_SIZE,
                          &lstm_2_s16_cell_gate_input_weights[0],
                          LSTM_2_S16_INPUT_ZERO_POINT,
                          &lstm_2_s16_cell_gate_bias[0]);
    arm_vector_sum_s8_s64(&output_data_kernel_sum[0],
                          LSTM_2_S16_INPUT_SIZE,
                          LSTM_2_S16_HIDDEN_SIZE,
                          &lstm_2_s16_output_gate_input_weights[0],
                          LSTM_2_S16_INPUT_ZERO_POINT,
                          &lstm_2_s16_output_gate_bias[0]);

    arm_vector_sum_s8_s64(&input_hidden_kernel_sum[0],
                          LSTM_2_S16_HIDDEN_SIZE,
                          LSTM_2_S16_HIDDEN_SIZE,
                          &lstm_2_s16_input_gate_hidden_weights[0],
                          -LSTM_2_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&forget_hidden_kernel_sum[0],
                          LSTM_2_S16_HIDDEN_SIZE,
                          LSTM_2_S16_HIDDEN_SIZE,
                          &lstm_2_s16_forget_gate_hidden_weights[0],
                          -LSTM_2_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&cell_hidden_kernel_sum[0],
                          LSTM_2_S16_HIDDEN_SIZE,
                          LSTM_2_S16_HIDDEN_SIZE,
                          &lstm_2_s16_cell_gate_hidden_weights[0],
                          -LSTM_2_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&output_hidden_kernel_sum[0],
                          LSTM_2_S16_HIDDEN_SIZE,
                          LSTM_2_S16_HIDDEN_SIZE,
                          &lstm_2_s16_output_gate_hidden_weights[0],
                          -LSTM_2_S16_OUTPUT_ZERO_POINT,
                          NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_2_S16_INPUT_GATE_INPUT_MULTIPLIER,
                                           LSTM_2_S16_INPUT_GATE_INPUT_SHIFT,
                                           &lstm_2_s16_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_2_S16_INPUT_GATE_HIDDEN_MULTIPLIER,
                                           LSTM_2_S16_INPUT_GATE_HIDDEN_SHIFT,
                                           &lstm_2_s16_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_2_s16_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_2_S16_FORGET_GATE_INPUT_MULTIPLIER,
                                            LSTM_2_S16_FORGET_GATE_INPUT_SHIFT,
                                            &lstm_2_s16_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_2_S16_FORGET_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_2_S16_FORGET_GATE_HIDDEN_SHIFT,
                                            &lstm_2_s16_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_2_s16_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_2_S16_CELL_GATE_INPUT_MULTIPLIER,
                                          LSTM_2_S16_CELL_GATE_INPUT_SHIFT,
                                          &lstm_2_s16_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_2_S16_CELL_GATE_HIDDEN_MULTIPLIER,
                                          LSTM_2_S16_CELL_GATE_HIDDEN_SHIFT,
                                          &lstm_2_s16_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_2_s16_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_2_S16_OUTPUT_GATE_INPUT_MULTIPLIER,
                                            LSTM_2_S16_OUTPUT_GATE_INPUT_SHIFT,
                                            &lstm_2_s16_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_2_S16_OUTPUT_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_2_S16_OUTPUT_GATE_HIDDEN_SHIFT,
                                            &lstm_2_s16_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_2_s16_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_2_S16_TIME_MAJOR,
                                         LSTM_2_S16_BATCH_SIZE,
                                         LSTM_2_S16_TIME_STEPS,
                                         LSTM_2_S16_INPUT_SIZE,
                                         LSTM_2_S16_HIDDEN_SIZE,
                                         LSTM_2_S16_INPUT_ZERO_POINT,
                                         LSTM_2_S16_FORGET_TO_CELL_MULTIPLIER,
                                         LSTM_2_S16_FORGET_TO_CELL_SHIFT,
                                         LSTM_2_S16_INPUT_TO_CELL_MULTIPLIER,
                                         LSTM_2_S16_INPUT_TO_CELL_SHIFT,
                                         LSTM_2_S16_CELL_CLIP,
                                         LSTM_2_S16_CELL_SCALE_POWER,
                                         LSTM_2_S16_OUTPUT_MULTIPLIER,
                                         LSTM_2_S16_OUTPUT_SHIFT,
                                         LSTM_2_S16_OUTPUT_ZERO_POINT,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    const int32_t temp1_size = arm_lstm_unidirectional_s16_temp1_get_buffer_size(&params);
    const int32_t temp2_size = arm_lstm_unidirectional_s16_temp2_get_buffer_size(&params);
    /* The published queries must agree with the size this test derives by hand from the kernel's writes: the
       temp buffers hold int16_t gate vectors of the step kernel's batch extent (the layer batch only for a
       time-major layer). */
    TEST_ASSERT_EQUAL((LSTM_2_S16_TIME_MAJOR ? LSTM_2_S16_BATCH_SIZE : 1) * LSTM_2_S16_HIDDEN_SIZE * (int32_t)sizeof(int16_t),
                      temp1_size);
    TEST_ASSERT_EQUAL(temp1_size, temp2_size);

    int16_t cell_state[LSTM_2_S16_BATCH_SIZE * LSTM_2_S16_HIDDEN_SIZE];

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = malloc((size_t)temp1_size);
    buffers.temp2 = malloc((size_t)temp2_size);
    buffers.cell_state = cell_state;
    buffers.hidden_state = NULL;

    arm_cmsis_nn_status result = arm_lstm_unidirectional_s16(lstm_2_s16_input_tensor, output, &params, &buffers);

    free(buffers.temp1);
    free(buffers.temp2);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}
void lstm_one_time_step_s16(void)
{
    int16_t output[LSTM_ONE_TIME_STEP_S16_BATCH_SIZE * LSTM_ONE_TIME_STEP_S16_TIME_STEPS *
                   LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *output_ref = &lstm_one_time_step_s16_output[0];
    const int32_t output_ref_size =
        LSTM_ONE_TIME_STEP_S16_BATCH_SIZE * LSTM_ONE_TIME_STEP_S16_TIME_STEPS * LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE;

    int64_t input_data_kernel_sum[LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];
    int64_t forget_data_kernel_sum[LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];
    int64_t cell_data_kernel_sum[LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];
    int64_t output_data_kernel_sum[LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];

    int64_t input_hidden_kernel_sum[LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];
    int64_t forget_hidden_kernel_sum[LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];
    int64_t cell_hidden_kernel_sum[LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];
    int64_t output_hidden_kernel_sum[LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];

    arm_vector_sum_s8_s64(&input_data_kernel_sum[0],
                          LSTM_ONE_TIME_STEP_S16_INPUT_SIZE,
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          &lstm_one_time_step_s16_input_gate_input_weights[0],
                          LSTM_ONE_TIME_STEP_S16_INPUT_ZERO_POINT,
                          &lstm_one_time_step_s16_input_gate_bias[0]);
    arm_vector_sum_s8_s64(&forget_data_kernel_sum[0],
                          LSTM_ONE_TIME_STEP_S16_INPUT_SIZE,
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          &lstm_one_time_step_s16_forget_gate_input_weights[0],
                          LSTM_ONE_TIME_STEP_S16_INPUT_ZERO_POINT,
                          &lstm_one_time_step_s16_forget_gate_bias[0]);
    arm_vector_sum_s8_s64(&cell_data_kernel_sum[0],
                          LSTM_ONE_TIME_STEP_S16_INPUT_SIZE,
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          &lstm_one_time_step_s16_cell_gate_input_weights[0],
                          LSTM_ONE_TIME_STEP_S16_INPUT_ZERO_POINT,
                          &lstm_one_time_step_s16_cell_gate_bias[0]);
    arm_vector_sum_s8_s64(&output_data_kernel_sum[0],
                          LSTM_ONE_TIME_STEP_S16_INPUT_SIZE,
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          &lstm_one_time_step_s16_output_gate_input_weights[0],
                          LSTM_ONE_TIME_STEP_S16_INPUT_ZERO_POINT,
                          &lstm_one_time_step_s16_output_gate_bias[0]);

    arm_vector_sum_s8_s64(&input_hidden_kernel_sum[0],
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          &lstm_one_time_step_s16_input_gate_hidden_weights[0],
                          -LSTM_ONE_TIME_STEP_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&forget_hidden_kernel_sum[0],
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          &lstm_one_time_step_s16_forget_gate_hidden_weights[0],
                          -LSTM_ONE_TIME_STEP_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&cell_hidden_kernel_sum[0],
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          &lstm_one_time_step_s16_cell_gate_hidden_weights[0],
                          -LSTM_ONE_TIME_STEP_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&output_hidden_kernel_sum[0],
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                          &lstm_one_time_step_s16_output_gate_hidden_weights[0],
                          -LSTM_ONE_TIME_STEP_S16_OUTPUT_ZERO_POINT,
                          NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_ONE_TIME_STEP_S16_INPUT_GATE_INPUT_MULTIPLIER,
                                           LSTM_ONE_TIME_STEP_S16_INPUT_GATE_INPUT_SHIFT,
                                           &lstm_one_time_step_s16_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_ONE_TIME_STEP_S16_INPUT_GATE_HIDDEN_MULTIPLIER,
                                           LSTM_ONE_TIME_STEP_S16_INPUT_GATE_HIDDEN_SHIFT,
                                           &lstm_one_time_step_s16_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_one_time_step_s16_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_ONE_TIME_STEP_S16_FORGET_GATE_INPUT_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_S16_FORGET_GATE_INPUT_SHIFT,
                                            &lstm_one_time_step_s16_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_ONE_TIME_STEP_S16_FORGET_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_S16_FORGET_GATE_HIDDEN_SHIFT,
                                            &lstm_one_time_step_s16_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_one_time_step_s16_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_ONE_TIME_STEP_S16_CELL_GATE_INPUT_MULTIPLIER,
                                          LSTM_ONE_TIME_STEP_S16_CELL_GATE_INPUT_SHIFT,
                                          &lstm_one_time_step_s16_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_ONE_TIME_STEP_S16_CELL_GATE_HIDDEN_MULTIPLIER,
                                          LSTM_ONE_TIME_STEP_S16_CELL_GATE_HIDDEN_SHIFT,
                                          &lstm_one_time_step_s16_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_one_time_step_s16_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_ONE_TIME_STEP_S16_OUTPUT_GATE_INPUT_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_S16_OUTPUT_GATE_INPUT_SHIFT,
                                            &lstm_one_time_step_s16_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_ONE_TIME_STEP_S16_OUTPUT_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_S16_OUTPUT_GATE_HIDDEN_SHIFT,
                                            &lstm_one_time_step_s16_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_one_time_step_s16_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_ONE_TIME_STEP_S16_TIME_MAJOR,
                                         LSTM_ONE_TIME_STEP_S16_BATCH_SIZE,
                                         LSTM_ONE_TIME_STEP_S16_TIME_STEPS,
                                         LSTM_ONE_TIME_STEP_S16_INPUT_SIZE,
                                         LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE,
                                         LSTM_ONE_TIME_STEP_S16_INPUT_ZERO_POINT,
                                         LSTM_ONE_TIME_STEP_S16_FORGET_TO_CELL_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_S16_FORGET_TO_CELL_SHIFT,
                                         LSTM_ONE_TIME_STEP_S16_INPUT_TO_CELL_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_S16_INPUT_TO_CELL_SHIFT,
                                         LSTM_ONE_TIME_STEP_S16_CELL_CLIP,
                                         LSTM_ONE_TIME_STEP_S16_CELL_SCALE_POWER,
                                         LSTM_ONE_TIME_STEP_S16_OUTPUT_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_S16_OUTPUT_SHIFT,
                                         LSTM_ONE_TIME_STEP_S16_OUTPUT_ZERO_POINT,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    const int32_t temp1_size = arm_lstm_unidirectional_s16_temp1_get_buffer_size(&params);
    const int32_t temp2_size = arm_lstm_unidirectional_s16_temp2_get_buffer_size(&params);
    /* Batch-major layer: the step kernel runs one batch at a time, so the queries report a single batch of
       int16_t gate vectors even though the layer batch is LSTM_ONE_TIME_STEP_S16_BATCH_SIZE. */
    TEST_ASSERT_EQUAL((LSTM_ONE_TIME_STEP_S16_TIME_MAJOR ? LSTM_ONE_TIME_STEP_S16_BATCH_SIZE : 1) * LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE * (int32_t)sizeof(int16_t),
                      temp1_size);
    TEST_ASSERT_EQUAL(temp1_size, temp2_size);

    int16_t cell_state[LSTM_ONE_TIME_STEP_S16_BATCH_SIZE * LSTM_ONE_TIME_STEP_S16_HIDDEN_SIZE];

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = malloc((size_t)temp1_size);
    buffers.temp2 = malloc((size_t)temp2_size);
    buffers.cell_state = cell_state;
    buffers.hidden_state = NULL;

    arm_cmsis_nn_status result =
        arm_lstm_unidirectional_s16(lstm_one_time_step_s16_input_tensor, output, &params, &buffers);

    free(buffers.temp1);
    free(buffers.temp2);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void lstm_stateful_batch_major_multibatch_s16(void)
{
    int16_t output[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_BATCH_SIZE *
                   LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_TIME_STEPS *
                   LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE] = {0};
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int16_t *output_ref = &lstm_stateful_batch_major_multibatch_s16_output[0];
    const int32_t output_ref_size = LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_BATCH_SIZE *
        LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_TIME_STEPS * LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE;

    int64_t input_data_kernel_sum[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];
    int64_t forget_data_kernel_sum[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];
    int64_t cell_data_kernel_sum[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];
    int64_t output_data_kernel_sum[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];

    int64_t input_hidden_kernel_sum[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];
    int64_t forget_hidden_kernel_sum[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];
    int64_t cell_hidden_kernel_sum[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];
    int64_t output_hidden_kernel_sum[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];

    arm_vector_sum_s8_s64(&input_data_kernel_sum[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_SIZE,
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          &lstm_stateful_batch_major_multibatch_s16_input_gate_input_weights[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_ZERO_POINT,
                          &lstm_stateful_batch_major_multibatch_s16_input_gate_bias[0]);
    arm_vector_sum_s8_s64(&forget_data_kernel_sum[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_SIZE,
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          &lstm_stateful_batch_major_multibatch_s16_forget_gate_input_weights[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_ZERO_POINT,
                          &lstm_stateful_batch_major_multibatch_s16_forget_gate_bias[0]);
    arm_vector_sum_s8_s64(&cell_data_kernel_sum[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_SIZE,
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          &lstm_stateful_batch_major_multibatch_s16_cell_gate_input_weights[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_ZERO_POINT,
                          &lstm_stateful_batch_major_multibatch_s16_cell_gate_bias[0]);
    arm_vector_sum_s8_s64(&output_data_kernel_sum[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_SIZE,
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          &lstm_stateful_batch_major_multibatch_s16_output_gate_input_weights[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_ZERO_POINT,
                          &lstm_stateful_batch_major_multibatch_s16_output_gate_bias[0]);

    arm_vector_sum_s8_s64(&input_hidden_kernel_sum[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          &lstm_stateful_batch_major_multibatch_s16_input_gate_hidden_weights[0],
                          -LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&forget_hidden_kernel_sum[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          &lstm_stateful_batch_major_multibatch_s16_forget_gate_hidden_weights[0],
                          -LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&cell_hidden_kernel_sum[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          &lstm_stateful_batch_major_multibatch_s16_cell_gate_hidden_weights[0],
                          -LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_ZERO_POINT,
                          NULL);
    arm_vector_sum_s8_s64(&output_hidden_kernel_sum[0],
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                          &lstm_stateful_batch_major_multibatch_s16_output_gate_hidden_weights[0],
                          -LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_ZERO_POINT,
                          NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_GATE_INPUT_MULTIPLIER,
                                           LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_GATE_INPUT_SHIFT,
                                           &lstm_stateful_batch_major_multibatch_s16_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_GATE_HIDDEN_MULTIPLIER,
                                           LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_GATE_HIDDEN_SHIFT,
                                           &lstm_stateful_batch_major_multibatch_s16_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_stateful_batch_major_multibatch_s16_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_FORGET_GATE_INPUT_MULTIPLIER,
                                            LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_FORGET_GATE_INPUT_SHIFT,
                                            &lstm_stateful_batch_major_multibatch_s16_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_FORGET_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_FORGET_GATE_HIDDEN_SHIFT,
                                            &lstm_stateful_batch_major_multibatch_s16_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_stateful_batch_major_multibatch_s16_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_CELL_GATE_INPUT_MULTIPLIER,
                                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_CELL_GATE_INPUT_SHIFT,
                                          &lstm_stateful_batch_major_multibatch_s16_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_CELL_GATE_HIDDEN_MULTIPLIER,
                                          LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_CELL_GATE_HIDDEN_SHIFT,
                                          &lstm_stateful_batch_major_multibatch_s16_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_stateful_batch_major_multibatch_s16_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_GATE_INPUT_MULTIPLIER,
                                            LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_GATE_INPUT_SHIFT,
                                            &lstm_stateful_batch_major_multibatch_s16_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_GATE_HIDDEN_SHIFT,
                                            &lstm_stateful_batch_major_multibatch_s16_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_stateful_batch_major_multibatch_s16_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_TIME_MAJOR,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_BATCH_SIZE,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_TIME_STEPS,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_SIZE,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_ZERO_POINT,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_FORGET_TO_CELL_MULTIPLIER,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_FORGET_TO_CELL_SHIFT,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_TO_CELL_MULTIPLIER,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_INPUT_TO_CELL_SHIFT,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_CELL_CLIP,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_CELL_SCALE_POWER,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_MULTIPLIER,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_SHIFT,
                                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_OUTPUT_ZERO_POINT,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    // Allocate zero-initialized hidden state buffer to test the non-null path!
    int16_t hidden_state[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_BATCH_SIZE *
                         LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE] = {0};

    const int32_t temp1_size = arm_lstm_unidirectional_s16_temp1_get_buffer_size(&params);
    const int32_t temp2_size = arm_lstm_unidirectional_s16_temp2_get_buffer_size(&params);
    /* Batch-major layer: the step kernel runs one batch at a time, so the queries report a single batch of
       int16_t gate vectors even though the layer batch is LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_BATCH_SIZE. */
    TEST_ASSERT_EQUAL((LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_TIME_MAJOR ? LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_BATCH_SIZE : 1) * LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE * (int32_t)sizeof(int16_t),
                      temp1_size);
    TEST_ASSERT_EQUAL(temp1_size, temp2_size);

    /* Streaming call: the caller owns the cell state, seeded to zero for a fresh sequence. */
    int16_t cell_state[LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_BATCH_SIZE * LSTM_STATEFUL_BATCH_MAJOR_MULTIBATCH_S16_HIDDEN_SIZE];
    memset(cell_state, 0, sizeof(cell_state));

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = malloc((size_t)temp1_size);
    buffers.temp2 = malloc((size_t)temp2_size);
    buffers.cell_state = cell_state;
    buffers.hidden_state = hidden_state;

    arm_cmsis_nn_status result =
        arm_lstm_unidirectional_s16(lstm_stateful_batch_major_multibatch_s16_input_tensor, output, &params, &buffers);

    free(buffers.temp1);
    free(buffers.temp2);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}
