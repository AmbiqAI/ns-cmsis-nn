/*
 * SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "unity.h"
#include <arm_nnfunctions.h>

#include "../TestData/svdf_int8/test_data.h"
#include "../TestData/svdf_int8_2/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (1)

void svdf_int8_arm_svdf_s8(void)
{
    const int32_t output_ref_size = SVDF_INT8_DST_SIZE;
    const int8_t *output_ref = svdf_int8_output_ref;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_context input_ctx;
    cmsis_nn_context output_ctx;
    cmsis_nn_svdf_params svdf_int8_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims weights_feature_dims;
    cmsis_nn_dims weights_time_dims;
    cmsis_nn_dims state_dims;
    cmsis_nn_dims output_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_per_tensor_quant_params input_quant_params;
    cmsis_nn_per_tensor_quant_params output_quant_params;
    int8_t output_data[SVDF_INT8_DST_SIZE] = {1};
    const int8_t *weights_feature_data = svdf_int8_weights_feature;
    const int8_t *weights_time_data = svdf_int8_weights_time;

    input_dims.n = SVDF_INT8_INPUT_BATCHES;
    input_dims.h = SVDF_INT8_INPUT_SIZE;
    weights_feature_dims.n = SVDF_INT8_FEATURE_BATCHES;
    weights_time_dims.h = SVDF_INT8_TIME_BATCHES;

    input_quant_params.multiplier = SVDF_INT8_MULTIPLIER_IN;
    input_quant_params.shift = SVDF_INT8_SHIFT_1;
    output_quant_params.multiplier = SVDF_INT8_MULTIPLIER_OUT;
    output_quant_params.shift = SVDF_INT8_SHIFT_2;

    svdf_int8_params.input_activation.min = SVDF_INT8_IN_ACTIVATION_MIN;
    svdf_int8_params.input_activation.max = SVDF_INT8_IN_ACTIVATION_MAX;
    svdf_int8_params.output_activation.min = SVDF_INT8_OUT_ACTIVATION_MIN;
    svdf_int8_params.output_activation.max = SVDF_INT8_OUT_ACTIVATION_MAX;
    svdf_int8_params.input_offset = SVDF_INT8_INPUT_OFFSET;
    svdf_int8_params.output_offset = SVDF_INT8_OUTPUT_OFFSET;
    svdf_int8_params.rank = SVDF_INT8_RANK;

    const int input_round_size = SVDF_INT8_INPUT_BATCHES * SVDF_INT8_INPUT_SIZE;
    const int number_inputs = sizeof(svdf_int8_input_sequence) / input_round_size;
    const int32_t number_units = SVDF_INT8_FEATURE_BATCHES / SVDF_INT8_RANK;
    const int scratch_size = SVDF_INT8_INPUT_BATCHES * SVDF_INT8_FEATURE_BATCHES * sizeof(int32_t);
    const int scratch_size_out = SVDF_INT8_INPUT_BATCHES * number_units * sizeof(int32_t);

    cmsis_nn_context ctx;
    const int32_t buf_size = arm_svdf_s8_get_buffer_size(&weights_feature_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *kernel_sum_buf = ctx.buf;
    arm_vector_sum_s8(
        kernel_sum_buf, input_dims.h, weights_feature_dims.n, weights_feature_data, -SVDF_INT8_INPUT_OFFSET, 0, NULL);
#endif

    // + SVDF_INT8_TIME_BATCHES additional bytes to make sure it is not overwritten
    const int state_data_size = sizeof(svdf_int8_state) + SVDF_INT8_TIME_BATCHES;
    const int8_t initial_data = 66;

    input_ctx.buf = malloc(scratch_size);
    output_ctx.buf = malloc(scratch_size_out);

    int8_t *input_data = malloc(input_round_size);
    int8_t *state_data = malloc(state_data_size);

    memset(state_data, initial_data, state_data_size);

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        memcpy(state_data, svdf_int8_state, sizeof(svdf_int8_state));
        for (int j = 0; j < number_inputs; j++)
        {
            memcpy(input_data, svdf_int8_input_sequence + j * input_round_size, input_round_size);
            arm_cmsis_nn_status result = arm_svdf_s8(&ctx,
                                                     &input_ctx,
                                                     &output_ctx,
                                                     &svdf_int8_params,
                                                     &input_quant_params,
                                                     &output_quant_params,
                                                     &input_dims,
                                                     input_data,
                                                     &state_dims,
                                                     state_data,
                                                     &weights_feature_dims,
                                                     weights_feature_data,
                                                     &weights_time_dims,
                                                     weights_time_data,
                                                     &bias_dims,
                                                     svdf_int8_biases,
                                                     &output_dims,
                                                     output_data);
            TEST_ASSERT_EQUAL(expected, result);
        }

        TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    // Make sure state data is not written outside boundary
    for (int i = sizeof(svdf_int8_state); i < state_data_size; i++)
    {
        TEST_ASSERT_EQUAL(state_data[i], initial_data);
    }

    free(state_data);
    free(input_data);
    free(input_ctx.buf);
    free(output_ctx.buf);
}

void svdf_int8_2_arm_svdf_s8(void)
{
    const int32_t output_ref_size = SVDF_INT8_2_DST_SIZE;
    const int8_t *output_ref = svdf_int8_2_output_ref;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    cmsis_nn_context input_ctx;
    cmsis_nn_context output_ctx;
    cmsis_nn_svdf_params svdf_int8_2_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims weights_feature_dims;
    cmsis_nn_dims weights_time_dims;
    cmsis_nn_dims state_dims;
    cmsis_nn_dims output_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_per_tensor_quant_params input_quant_params;
    cmsis_nn_per_tensor_quant_params output_quant_params;
    int8_t output_data[SVDF_INT8_2_DST_SIZE] = {1};
    const int8_t *weights_feature_data = svdf_int8_2_weights_feature;
    const int8_t *weights_time_data = svdf_int8_2_weights_time;

    input_dims.n = SVDF_INT8_2_INPUT_BATCHES;
    input_dims.h = SVDF_INT8_2_INPUT_SIZE;
    weights_feature_dims.n = SVDF_INT8_2_FEATURE_BATCHES;
    weights_time_dims.h = SVDF_INT8_2_TIME_BATCHES;

    input_quant_params.multiplier = SVDF_INT8_2_MULTIPLIER_IN;
    input_quant_params.shift = SVDF_INT8_2_SHIFT_1;
    output_quant_params.multiplier = SVDF_INT8_2_MULTIPLIER_OUT;
    output_quant_params.shift = SVDF_INT8_2_SHIFT_2;

    svdf_int8_2_params.input_activation.min = SVDF_INT8_2_IN_ACTIVATION_MIN;
    svdf_int8_2_params.input_activation.max = SVDF_INT8_2_IN_ACTIVATION_MAX;
    svdf_int8_2_params.output_activation.min = SVDF_INT8_2_OUT_ACTIVATION_MIN;
    svdf_int8_2_params.output_activation.max = SVDF_INT8_2_OUT_ACTIVATION_MAX;
    svdf_int8_2_params.input_offset = SVDF_INT8_2_INPUT_OFFSET;
    svdf_int8_2_params.output_offset = SVDF_INT8_2_OUTPUT_OFFSET;
    svdf_int8_2_params.rank = SVDF_INT8_2_RANK;

    const int input_round_size = SVDF_INT8_2_INPUT_BATCHES * SVDF_INT8_2_INPUT_SIZE;
    const int number_inputs = sizeof(svdf_int8_2_input_sequence) / input_round_size;
    const int32_t number_units = SVDF_INT8_2_FEATURE_BATCHES / SVDF_INT8_2_RANK;
    const int scratch_size = SVDF_INT8_2_INPUT_BATCHES * SVDF_INT8_2_FEATURE_BATCHES * sizeof(int32_t);
    const int scratch_size_out = SVDF_INT8_2_INPUT_BATCHES * number_units * sizeof(int32_t);

    cmsis_nn_context ctx;
    const int32_t buf_size = arm_svdf_s8_get_buffer_size(&weights_feature_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *kernel_sum_buf = ctx.buf;
    arm_vector_sum_s8(
        kernel_sum_buf, input_dims.h, weights_feature_dims.n, weights_feature_data, -SVDF_INT8_2_INPUT_OFFSET, 0, NULL);
#endif

    const int state_data_size = sizeof(svdf_int8_2_state);

    input_ctx.buf = malloc(scratch_size);
    output_ctx.buf = malloc(scratch_size_out);

    int8_t *input_data = malloc(input_round_size);
    int8_t *state_data = malloc(state_data_size);

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        memcpy(state_data, svdf_int8_2_state, sizeof(svdf_int8_2_state));
        for (int j = 0; j < number_inputs; j++)
        {
            memcpy(input_data, svdf_int8_2_input_sequence + j * input_round_size, input_round_size);
            arm_cmsis_nn_status result = arm_svdf_s8(&ctx,
                                                     &input_ctx,
                                                     &output_ctx,
                                                     &svdf_int8_2_params,
                                                     &input_quant_params,
                                                     &output_quant_params,
                                                     &input_dims,
                                                     input_data,
                                                     &state_dims,
                                                     state_data,
                                                     &weights_feature_dims,
                                                     weights_feature_data,
                                                     &weights_time_dims,
                                                     weights_time_data,
                                                     &bias_dims,
                                                     svdf_int8_2_biases,
                                                     &output_dims,
                                                     output_data);
            TEST_ASSERT_EQUAL(expected, result);
        }

        TEST_ASSERT_TRUE(validate(output_data, output_ref, output_ref_size));
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    free(state_data);
    free(input_data);
    free(input_ctx.buf);
    free(output_ctx.buf);
}

/*
 * Regression test for the ctx sizing contract, issue #269.
 *
 * arm_svdf_s8 reads one int32_t kernel sum per feature batch, i.e. weights_feature_dims->n of them, which is what
 * arm_svdf_s8_get_buffer_size() returns. The header used to name arm_fully_connected_s8_get_buffer_size() instead,
 * which sizes from ->c; the two differ whenever n != c, and the cases above never set weights_feature_dims.c at all.
 * This case sets both fields, sizes ctx through the documented helper for a shape with n > c, and runs the kernel
 * twice with different bytes living past the end of that allocation. Any read past the end shows up as a difference
 * between the two runs; any write shows up in the guard.
 */
#define SVDF_CTX_INPUT_BATCHES 1
#define SVDF_CTX_INPUT_SIZE 4
#define SVDF_CTX_FEATURE_BATCHES 16
#define SVDF_CTX_RANK 1
#define SVDF_CTX_TIME_BATCHES 4
#define SVDF_CTX_GUARD_WORDS 16

static arm_cmsis_nn_status svdf_ctx_sizing_run(int32_t canary, int8_t *output_data, int32_t *guard_clobbered)
{
    cmsis_nn_context input_ctx;
    cmsis_nn_context output_ctx;
    cmsis_nn_context ctx;
    cmsis_nn_svdf_params params;
    cmsis_nn_per_tensor_quant_params input_quant_params = {1073741824, 1};
    cmsis_nn_per_tensor_quant_params output_quant_params = {1073741824, -3};
    cmsis_nn_dims input_dims = {SVDF_CTX_INPUT_BATCHES, SVDF_CTX_INPUT_SIZE, 1, 1};
    cmsis_nn_dims weights_feature_dims = {SVDF_CTX_FEATURE_BATCHES, 1, 1, SVDF_CTX_INPUT_SIZE};
    cmsis_nn_dims weights_time_dims = {1, SVDF_CTX_TIME_BATCHES, 1, 1};
    cmsis_nn_dims state_dims = {1, 1, 1, SVDF_CTX_INPUT_BATCHES * SVDF_CTX_FEATURE_BATCHES * SVDF_CTX_TIME_BATCHES};
    cmsis_nn_dims bias_dims = {1, 1, 1, SVDF_CTX_FEATURE_BATCHES / SVDF_CTX_RANK};
    cmsis_nn_dims output_dims = {SVDF_CTX_INPUT_BATCHES, 1, 1, SVDF_CTX_FEATURE_BATCHES / SVDF_CTX_RANK};

    int8_t weights_feature_data[SVDF_CTX_FEATURE_BATCHES * SVDF_CTX_INPUT_SIZE];
    int8_t weights_time_data[SVDF_CTX_FEATURE_BATCHES * SVDF_CTX_TIME_BATCHES];
    int8_t input_data[SVDF_CTX_INPUT_BATCHES * SVDF_CTX_INPUT_SIZE];
    int8_t state_data[SVDF_CTX_INPUT_BATCHES * SVDF_CTX_FEATURE_BATCHES * SVDF_CTX_TIME_BATCHES] = {0};

    for (int i = 0; i < SVDF_CTX_FEATURE_BATCHES * SVDF_CTX_INPUT_SIZE; i++)
    {
        weights_feature_data[i] = (int8_t)((i % 9) - 4);
    }
    for (int i = 0; i < SVDF_CTX_FEATURE_BATCHES * SVDF_CTX_TIME_BATCHES; i++)
    {
        weights_time_data[i] = (int8_t)((i % 7) - 3);
    }
    for (int i = 0; i < SVDF_CTX_INPUT_BATCHES * SVDF_CTX_INPUT_SIZE; i++)
    {
        input_data[i] = (int8_t)(i + 1);
    }

    params.input_activation.min = -128;
    params.input_activation.max = 127;
    params.output_activation.min = -128;
    params.output_activation.max = 127;
    params.input_offset = 5;
    params.output_offset = -3;
    params.rank = SVDF_CTX_RANK;

    const int32_t buf_size = arm_svdf_s8_get_buffer_size(&weights_feature_dims);
#if defined(ARM_MATH_MVEI)
    // The kernel indexes one sum per feature batch, not one per input element.
    TEST_ASSERT_TRUE(weights_feature_dims.n > weights_feature_dims.c);
    TEST_ASSERT_TRUE(buf_size >= (int32_t)(weights_feature_dims.n * sizeof(int32_t)));
#endif

    const int32_t buf_words = buf_size / (int32_t)sizeof(int32_t);
    int32_t *base = malloc((size_t)(buf_size + SVDF_CTX_GUARD_WORDS * (int32_t)sizeof(int32_t)));
    TEST_ASSERT_NOT_NULL(base);
    ctx.buf = base;
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    arm_vector_sum_s8(base, input_dims.h, weights_feature_dims.n, weights_feature_data, -params.input_offset, 0, NULL);
#endif
    for (int32_t i = buf_words; i < buf_words + SVDF_CTX_GUARD_WORDS; i++)
    {
        base[i] = canary;
    }

    const int scratch_size = SVDF_CTX_INPUT_BATCHES * SVDF_CTX_FEATURE_BATCHES * (int)sizeof(int32_t);
    const int scratch_size_out =
        SVDF_CTX_INPUT_BATCHES * (SVDF_CTX_FEATURE_BATCHES / SVDF_CTX_RANK) * (int)sizeof(int32_t);
    input_ctx.buf = malloc(scratch_size);
    output_ctx.buf = malloc(scratch_size_out);
    TEST_ASSERT_NOT_NULL(input_ctx.buf);
    TEST_ASSERT_NOT_NULL(output_ctx.buf);

    arm_cmsis_nn_status result = arm_svdf_s8(&ctx,
                                             &input_ctx,
                                             &output_ctx,
                                             &params,
                                             &input_quant_params,
                                             &output_quant_params,
                                             &input_dims,
                                             input_data,
                                             &state_dims,
                                             state_data,
                                             &weights_feature_dims,
                                             weights_feature_data,
                                             &weights_time_dims,
                                             weights_time_data,
                                             &bias_dims,
                                             NULL,
                                             &output_dims,
                                             output_data);

    *guard_clobbered = 0;
    for (int32_t i = buf_words; i < buf_words + SVDF_CTX_GUARD_WORDS; i++)
    {
        if (base[i] != canary)
        {
            (*guard_clobbered)++;
        }
    }

    // The caller is responsible to clear the scratch buffers for security reasons if applicable.
    memset(base, 0, (size_t)(buf_size + SVDF_CTX_GUARD_WORDS * (int32_t)sizeof(int32_t)));
    free(base);
    free(input_ctx.buf);
    free(output_ctx.buf);
    return result;
}

void svdf_int8_ctx_sizing_arm_svdf_s8(void)
{
    int8_t output_a[SVDF_CTX_INPUT_BATCHES * (SVDF_CTX_FEATURE_BATCHES / SVDF_CTX_RANK)] = {0};
    int8_t output_b[SVDF_CTX_INPUT_BATCHES * (SVDF_CTX_FEATURE_BATCHES / SVDF_CTX_RANK)] = {0};
    int32_t guard_a = 0;
    int32_t guard_b = 0;

    // Small canaries: an out-of-range value would saturate the activation clamp and hide the difference.
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, svdf_ctx_sizing_run(100, output_a, &guard_a));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, svdf_ctx_sizing_run(-100, output_b, &guard_b));

    TEST_ASSERT_EQUAL_INT32(0, guard_a);
    TEST_ASSERT_EQUAL_INT32(0, guard_b);
    TEST_ASSERT_EQUAL_INT8_ARRAY(output_a, output_b, sizeof(output_a));
}
