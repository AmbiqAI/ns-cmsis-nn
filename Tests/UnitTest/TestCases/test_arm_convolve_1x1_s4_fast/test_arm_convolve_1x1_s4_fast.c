/*
 * SPDX-FileCopyrightText: Copyright 2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <stdlib.h>
#include <unity.h>

#include "../TestData/kernel1x1_int4/test_data.h"
#include "../TestData/kernel1x1_int4_2/test_data.h"
#include "../TestData/kernel1x1_int4_3/test_data.h"
#include "../TestData/kernel1x1_stride_x_int4/test_data.h"
#include "../TestData/kernel1x1_stride_x_y_1_int4/test_data.h"
#include "../TestData/kernel1x1_stride_x_y_2_int4/test_data.h"
#include "../TestData/kernel1x1_stride_x_y_int4/test_data.h"
#include "../Utils/validate.h"

void kernel1x1_arm_convolve_1x1_s4_fast(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_int4_biases;
    const int8_t *input_data = kernel1x1_int4_input;

    input_dims.n = KERNEL1X1_INT4_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_INT4_INPUT_H;
    input_dims.w = KERNEL1X1_INT4_INPUT_W;
    input_dims.c = KERNEL1X1_INT4_IN_CH;
    filter_dims.n = KERNEL1X1_INT4_OUT_CH;
    filter_dims.h = KERNEL1X1_INT4_FILTER_Y;
    filter_dims.w = KERNEL1X1_INT4_FILTER_X;
    filter_dims.c = KERNEL1X1_INT4_IN_CH;
    output_dims.n = KERNEL1X1_INT4_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_INT4_OUTPUT_H;
    output_dims.w = KERNEL1X1_INT4_OUTPUT_W;
    output_dims.c = KERNEL1X1_INT4_OUT_CH;

    conv_params.padding.h = KERNEL1X1_INT4_PAD_Y;
    conv_params.padding.w = KERNEL1X1_INT4_PAD_X;
    conv_params.stride.h = KERNEL1X1_INT4_STRIDE_Y;
    conv_params.stride.w = KERNEL1X1_INT4_STRIDE_X;
    conv_params.dilation.w = KERNEL1X1_INT4_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_INT4_DILATION_Y;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.input_offset = KERNEL1X1_INT4_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_int4_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_int4_output_shift;

    const int32_t buf_size = arm_convolve_1x1_s4_fast_get_buffer_size(&input_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1x1_s4_fast(&ctx,
                                                          &conv_params,
                                                          &quant_params,
                                                          &input_dims,
                                                          input_data,
                                                          &filter_dims,
                                                          kernel1x1_int4_weights,
                                                          &bias_dims,
                                                          bias_data,
                                                          &output_dims,
                                                          output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, kernel1x1_int4_output_ref, KERNEL1X1_INT4_DST_SIZE));
}

void kernel1x1_2_arm_convolve_1x1_s4_fast(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_INT4_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_int4_2_biases;
    const int8_t *input_data = kernel1x1_int4_2_input;

    input_dims.n = KERNEL1X1_INT4_2_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_INT4_2_INPUT_H;
    input_dims.w = KERNEL1X1_INT4_2_INPUT_W;
    input_dims.c = KERNEL1X1_INT4_2_IN_CH;
    filter_dims.n = KERNEL1X1_INT4_2_OUT_CH;
    filter_dims.h = KERNEL1X1_INT4_2_FILTER_Y;
    filter_dims.w = KERNEL1X1_INT4_2_FILTER_X;
    filter_dims.c = KERNEL1X1_INT4_2_IN_CH;
    output_dims.n = KERNEL1X1_INT4_2_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_INT4_2_OUTPUT_H;
    output_dims.w = KERNEL1X1_INT4_2_OUTPUT_W;
    output_dims.c = KERNEL1X1_INT4_2_OUT_CH;

    conv_params.padding.h = KERNEL1X1_INT4_2_PAD_Y;
    conv_params.padding.w = KERNEL1X1_INT4_2_PAD_X;
    conv_params.stride.h = KERNEL1X1_INT4_2_STRIDE_Y;
    conv_params.stride.w = KERNEL1X1_INT4_2_STRIDE_X;
    conv_params.dilation.w = KERNEL1X1_INT4_2_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_INT4_2_DILATION_Y;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.input_offset = KERNEL1X1_INT4_2_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_INT4_2_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_INT4_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_INT4_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_int4_2_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_int4_2_output_shift;

    const int32_t buf_size = arm_convolve_1x1_s4_fast_get_buffer_size(&input_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1x1_s4_fast(&ctx,
                                                          &conv_params,
                                                          &quant_params,
                                                          &input_dims,
                                                          input_data,
                                                          &filter_dims,
                                                          kernel1x1_int4_2_weights,
                                                          &bias_dims,
                                                          bias_data,
                                                          &output_dims,
                                                          output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, kernel1x1_int4_2_output_ref, KERNEL1X1_INT4_2_DST_SIZE));
}
void kernel1x1_3_arm_convolve_1x1_s4_fast(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_INT4_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_int4_3_biases;
    const int8_t *input_data = kernel1x1_int4_3_input;

    input_dims.n = KERNEL1X1_INT4_3_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_INT4_3_INPUT_H;
    input_dims.w = KERNEL1X1_INT4_3_INPUT_W;
    input_dims.c = KERNEL1X1_INT4_3_IN_CH;
    filter_dims.n = KERNEL1X1_INT4_3_OUT_CH;
    filter_dims.h = KERNEL1X1_INT4_3_FILTER_Y;
    filter_dims.w = KERNEL1X1_INT4_3_FILTER_X;
    filter_dims.c = KERNEL1X1_INT4_3_IN_CH;
    output_dims.n = KERNEL1X1_INT4_3_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_INT4_3_OUTPUT_H;
    output_dims.w = KERNEL1X1_INT4_3_OUTPUT_W;
    output_dims.c = KERNEL1X1_INT4_3_OUT_CH;

    conv_params.padding.h = KERNEL1X1_INT4_3_PAD_Y;
    conv_params.padding.w = KERNEL1X1_INT4_3_PAD_X;
    conv_params.stride.h = KERNEL1X1_INT4_3_STRIDE_Y;
    conv_params.stride.w = KERNEL1X1_INT4_3_STRIDE_X;
    conv_params.dilation.w = KERNEL1X1_INT4_3_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_INT4_3_DILATION_Y;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.input_offset = KERNEL1X1_INT4_3_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_INT4_3_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_INT4_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_INT4_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_int4_3_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_int4_3_output_shift;

    const int32_t buf_size = arm_convolve_1x1_s4_fast_get_buffer_size(&input_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1x1_s4_fast(&ctx,
                                                          &conv_params,
                                                          &quant_params,
                                                          &input_dims,
                                                          input_data,
                                                          &filter_dims,
                                                          kernel1x1_int4_3_weights,
                                                          &bias_dims,
                                                          bias_data,
                                                          &output_dims,
                                                          output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, kernel1x1_int4_3_output_ref, KERNEL1X1_INT4_3_DST_SIZE));
}

void kernel1x1_stride_x_arm_convolve_1x1_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_STRIDE_X_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_stride_x_int4_biases;
    const int8_t *input_data = kernel1x1_stride_x_int4_input;

    input_dims.n = KERNEL1X1_STRIDE_X_INT4_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_STRIDE_X_INT4_INPUT_H;
    input_dims.w = KERNEL1X1_STRIDE_X_INT4_INPUT_W;
    input_dims.c = KERNEL1X1_STRIDE_X_INT4_IN_CH;

    filter_dims.n = KERNEL1X1_STRIDE_X_INT4_OUT_CH;
    filter_dims.h = KERNEL1X1_STRIDE_X_INT4_FILTER_Y;
    filter_dims.w = KERNEL1X1_STRIDE_X_INT4_FILTER_X;
    filter_dims.c = KERNEL1X1_STRIDE_X_INT4_IN_CH;

    output_dims.n = KERNEL1X1_STRIDE_X_INT4_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_STRIDE_X_INT4_OUTPUT_H;
    output_dims.w = KERNEL1X1_STRIDE_X_INT4_OUTPUT_W;
    output_dims.c = KERNEL1X1_STRIDE_X_INT4_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.padding.w = KERNEL1X1_STRIDE_X_INT4_PAD_X;
    conv_params.padding.h = KERNEL1X1_STRIDE_X_INT4_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X_INT4_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_X_INT4_STRIDE_Y;
    conv_params.dilation.w = KERNEL1X1_STRIDE_X_INT4_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_STRIDE_X_INT4_DILATION_Y;

    conv_params.input_offset = KERNEL1X1_STRIDE_X_INT4_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_STRIDE_X_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_STRIDE_X_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_STRIDE_X_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_stride_x_int4_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_stride_x_int4_output_shift;

    const int32_t buf_size = arm_convolve_1x1_s4_fast_get_buffer_size(&input_dims);
    ctx.buf = NULL;
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1x1_s4_fast(&ctx,
                                                          &conv_params,
                                                          &quant_params,
                                                          &input_dims,
                                                          input_data,
                                                          &filter_dims,
                                                          kernel1x1_stride_x_int4_weights,
                                                          &bias_dims,
                                                          bias_data,
                                                          &output_dims,
                                                          output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

    result = arm_convolve_1x1_s4(&ctx,
                                 &conv_params,
                                 &quant_params,
                                 &input_dims,
                                 input_data,
                                 &filter_dims,
                                 kernel1x1_stride_x_int4_weights,
                                 &bias_dims,
                                 bias_data,
                                 &output_dims,
                                 output);
    TEST_ASSERT_EQUAL(expected, result);

    TEST_ASSERT_TRUE(validate(output, kernel1x1_stride_x_int4_output_ref, KERNEL1X1_STRIDE_X_INT4_DST_SIZE));
}

void kernel1x1_stride_x_y_arm_convolve_1x1_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_STRIDE_X_Y_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_stride_x_y_int4_biases;
    const int8_t *input_data = kernel1x1_stride_x_y_int4_input;

    input_dims.n = KERNEL1X1_STRIDE_X_Y_INT4_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_STRIDE_X_Y_INT4_INPUT_H;
    input_dims.w = KERNEL1X1_STRIDE_X_Y_INT4_INPUT_W;
    input_dims.c = KERNEL1X1_STRIDE_X_Y_INT4_IN_CH;

    filter_dims.n = KERNEL1X1_STRIDE_X_Y_INT4_OUT_CH;
    filter_dims.h = KERNEL1X1_STRIDE_X_Y_INT4_FILTER_Y;
    filter_dims.w = KERNEL1X1_STRIDE_X_Y_INT4_FILTER_X;
    filter_dims.c = KERNEL1X1_STRIDE_X_Y_INT4_IN_CH;

    output_dims.n = KERNEL1X1_STRIDE_X_Y_INT4_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_STRIDE_X_Y_INT4_OUTPUT_H;
    output_dims.w = KERNEL1X1_STRIDE_X_Y_INT4_OUTPUT_W;
    output_dims.c = KERNEL1X1_STRIDE_X_Y_INT4_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.padding.w = KERNEL1X1_STRIDE_X_Y_INT4_PAD_X;
    conv_params.padding.h = KERNEL1X1_STRIDE_X_Y_INT4_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X_Y_INT4_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_X_Y_INT4_STRIDE_Y;
    conv_params.dilation.w = KERNEL1X1_STRIDE_X_Y_INT4_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_STRIDE_X_Y_INT4_DILATION_Y;

    conv_params.input_offset = KERNEL1X1_STRIDE_X_Y_INT4_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_STRIDE_X_Y_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_STRIDE_X_Y_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_STRIDE_X_Y_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_stride_x_y_int4_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_stride_x_y_int4_output_shift;

    const int32_t buf_size =
        arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_cmsis_nn_status result = arm_convolve_wrapper_s4(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel1x1_stride_x_y_int4_weights,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);
    TEST_ASSERT_TRUE(validate(output, kernel1x1_stride_x_y_int4_output_ref, KERNEL1X1_STRIDE_X_Y_INT4_DST_SIZE));

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
        ctx.size = 0;
    }
    memset(output, 0, sizeof(output));

    result = arm_convolve_1x1_s4(&ctx,
                                 &conv_params,
                                 &quant_params,
                                 &input_dims,
                                 input_data,
                                 &filter_dims,
                                 kernel1x1_stride_x_y_int4_weights,
                                 &bias_dims,
                                 bias_data,
                                 &output_dims,
                                 output);
    TEST_ASSERT_EQUAL(expected, result);

    TEST_ASSERT_TRUE(validate(output, kernel1x1_stride_x_y_int4_output_ref, KERNEL1X1_STRIDE_X_Y_INT4_DST_SIZE));
}

void kernel1x1_stride_x_y_1_arm_convolve_1x1_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_STRIDE_X_Y_1_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_stride_x_y_1_int4_biases;
    const int8_t *input_data = kernel1x1_stride_x_y_1_int4_input;

    input_dims.n = KERNEL1X1_STRIDE_X_Y_1_INT4_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_STRIDE_X_Y_1_INT4_INPUT_H;
    input_dims.w = KERNEL1X1_STRIDE_X_Y_1_INT4_INPUT_W;
    input_dims.c = KERNEL1X1_STRIDE_X_Y_1_INT4_IN_CH;

    filter_dims.n = KERNEL1X1_STRIDE_X_Y_1_INT4_OUT_CH;
    filter_dims.h = KERNEL1X1_STRIDE_X_Y_1_INT4_FILTER_Y;
    filter_dims.w = KERNEL1X1_STRIDE_X_Y_1_INT4_FILTER_X;
    filter_dims.c = KERNEL1X1_STRIDE_X_Y_1_INT4_IN_CH;

    output_dims.n = KERNEL1X1_STRIDE_X_Y_1_INT4_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_STRIDE_X_Y_1_INT4_OUTPUT_H;
    output_dims.w = KERNEL1X1_STRIDE_X_Y_1_INT4_OUTPUT_W;
    output_dims.c = KERNEL1X1_STRIDE_X_Y_1_INT4_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.padding.w = KERNEL1X1_STRIDE_X_Y_1_INT4_PAD_X;
    conv_params.padding.h = KERNEL1X1_STRIDE_X_Y_1_INT4_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X_Y_1_INT4_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_X_Y_1_INT4_STRIDE_Y;
    conv_params.dilation.w = KERNEL1X1_STRIDE_X_Y_1_INT4_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_STRIDE_X_Y_1_INT4_DILATION_Y;

    conv_params.input_offset = KERNEL1X1_STRIDE_X_Y_1_INT4_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_STRIDE_X_Y_1_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_STRIDE_X_Y_1_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_STRIDE_X_Y_1_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_stride_x_y_1_int4_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_stride_x_y_1_int4_output_shift;

    ctx.size = 0;
    ctx.buf = NULL;

    arm_cmsis_nn_status result = arm_convolve_1x1_s4(&ctx,
                                                     &conv_params,
                                                     &quant_params,
                                                     &input_dims,
                                                     input_data,
                                                     &filter_dims,
                                                     kernel1x1_stride_x_y_1_int4_weights,
                                                     &bias_dims,
                                                     bias_data,
                                                     &output_dims,
                                                     output);
    TEST_ASSERT_EQUAL(expected, result);

    TEST_ASSERT_TRUE(validate(output, kernel1x1_stride_x_y_1_int4_output_ref, KERNEL1X1_STRIDE_X_Y_1_INT4_DST_SIZE));
}

void kernel1x1_stride_x_y_2_arm_convolve_1x1_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[KERNEL1X1_STRIDE_X_Y_2_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = kernel1x1_stride_x_y_2_int4_biases;
    const int8_t *input_data = kernel1x1_stride_x_y_2_int4_input;

    input_dims.n = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_BATCHES;
    input_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_H;
    input_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_W;
    input_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_IN_CH;

    filter_dims.n = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_CH;
    filter_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_FILTER_Y;
    filter_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_FILTER_X;
    filter_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_IN_CH;

    output_dims.n = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_BATCHES;
    output_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_H;
    output_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_W;
    output_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.padding.w = KERNEL1X1_STRIDE_X_Y_2_INT4_PAD_X;
    conv_params.padding.h = KERNEL1X1_STRIDE_X_Y_2_INT4_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X_Y_2_INT4_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_X_Y_2_INT4_STRIDE_Y;
    conv_params.dilation.w = KERNEL1X1_STRIDE_X_Y_2_INT4_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_STRIDE_X_Y_2_INT4_DILATION_Y;

    conv_params.input_offset = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_stride_x_y_2_int4_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_stride_x_y_2_int4_output_shift;

    ctx.size = 0;
    ctx.buf = NULL;

    arm_cmsis_nn_status result = arm_convolve_1x1_s4(&ctx,
                                                     &conv_params,
                                                     &quant_params,
                                                     &input_dims,
                                                     input_data,
                                                     &filter_dims,
                                                     kernel1x1_stride_x_y_2_int4_weights,
                                                     &bias_dims,
                                                     bias_data,
                                                     &output_dims,
                                                     output);
    TEST_ASSERT_EQUAL(expected, result);

    TEST_ASSERT_TRUE(validate(output, kernel1x1_stride_x_y_2_int4_output_ref, KERNEL1X1_STRIDE_X_Y_2_INT4_DST_SIZE));
}

void buffer_size_arm_convolve_1x1_s4_fast(void)
{
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_BATCHES;
    input_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_W;
    input_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_H;
    input_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_IN_CH;
    filter_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_FILTER_X;
    filter_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_FILTER_Y;
    filter_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_IN_CH;
    output_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_W;
    output_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_H;
    output_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_CH;

    conv_params.padding.w = KERNEL1X1_STRIDE_X_Y_2_INT4_PAD_X;
    conv_params.padding.h = KERNEL1X1_STRIDE_X_Y_2_INT4_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X_Y_2_INT4_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_X_Y_2_INT4_STRIDE_Y;
    conv_params.dilation.w = KERNEL1X1_STRIDE_X_Y_2_INT4_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_STRIDE_X_Y_2_INT4_DILATION_Y;

    conv_params.input_offset = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_ACTIVATION_MAX;

    const int32_t buf_size = arm_convolve_1x1_s4_fast_get_buffer_size(&input_dims);
    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, buf_size);
}

void buffer_size_mve_arm_convolve_1x1_s4_fast(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_BATCHES;
    input_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_W;
    input_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_H;
    input_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_IN_CH;
    filter_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_FILTER_X;
    filter_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_FILTER_Y;
    filter_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_IN_CH;
    output_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_W;
    output_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_H;
    output_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_CH;

    conv_params.padding.w = KERNEL1X1_STRIDE_X_Y_2_INT4_PAD_X;
    conv_params.padding.h = KERNEL1X1_STRIDE_X_Y_2_INT4_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X_Y_2_INT4_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_X_Y_2_INT4_STRIDE_Y;
    conv_params.dilation.w = KERNEL1X1_STRIDE_X_Y_2_INT4_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_STRIDE_X_Y_2_INT4_DILATION_Y;

    conv_params.input_offset = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_convolve_1x1_s4_fast(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_BATCHES;
    input_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_W;
    input_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_H;
    input_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_IN_CH;
    filter_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_FILTER_X;
    filter_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_FILTER_Y;
    filter_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_IN_CH;
    output_dims.w = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_W;
    output_dims.h = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_H;
    output_dims.c = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_CH;

    conv_params.padding.w = KERNEL1X1_STRIDE_X_Y_2_INT4_PAD_X;
    conv_params.padding.h = KERNEL1X1_STRIDE_X_Y_2_INT4_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X_Y_2_INT4_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_X_Y_2_INT4_STRIDE_Y;
    conv_params.dilation.w = KERNEL1X1_STRIDE_X_Y_2_INT4_DILATION_X;
    conv_params.dilation.h = KERNEL1X1_STRIDE_X_Y_2_INT4_DILATION_Y;

    conv_params.input_offset = KERNEL1X1_STRIDE_X_Y_2_INT4_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_STRIDE_X_Y_2_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_STRIDE_X_Y_2_INT4_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}

/* Compare active predicated lanes with the same arithmetic used for full rows. */
void requantize_s4_predicated_rounding(void)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    uint32_t seed = 671;
    for (int shift = -8; shift <= 5; ++shift)
    {
        for (int sample = 0; sample < 128; ++sample)
        {
            int32_t input[4], actual[4];
            for (int lane = 0; lane < 4; ++lane)
            {
                seed = seed * 1664525u + 1013904223u;
                input[lane] = (int32_t)(seed & 0x1fffffu) - 0x100000;
            }
            seed = seed * 1664525u + 1013904223u;
            const int32_t multiplier = (int32_t)((seed & 0x3fffffffu) + 0x40000000u);
            for (int lanes = 1; lanes <= 4; ++lanes)
            {
                const int32x4_t result = arm_requantize_mve_pred(vldrwq_s32(input), multiplier, shift, vctp32q(lanes));
                vstrwq_s32(actual, result);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    TEST_ASSERT_EQUAL_INT32(arm_nn_requantize(input[lane], multiplier, shift), actual[lane]);
                }
                for (int lane = lanes; lane < 4; ++lane)
                {
                    TEST_ASSERT_EQUAL_INT32(0, actual[lane]);
                }
            }
        }
    }
#endif
}

/* Identical pixels must not depend on whether they occupy a full block or tail. */
void matrix_s4_rounding_row_groups(void)
{
    int8_t input[4 * 16], packed_weights[8 * 16 / 2], output[4 * 8];
    int32_t bias[8], multiplier[8], shift[8];
    uint32_t seed = 681;
    for (int sample = 0; sample < 32; ++sample)
    {
        for (unsigned i = 0; i < sizeof(input); ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            input[i] = (int8_t)((int32_t)(seed >> 24) - 128);
        }
        for (unsigned i = 0; i < sizeof(packed_weights); ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            packed_weights[i] = (int8_t)((int32_t)(seed >> 24) - 128);
        }
        for (int ch = 0; ch < 8; ++ch)
        {
            bias[ch] = ch * 17 - 63;
            multiplier[ch] = 1073741824 + ch * 7654321;
            shift[ch] = -5 - ch % 3;
        }
        for (int rows = 1; rows <= 4; ++rows)
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                              arm_nn_mat_mult_nt_t_s4(input, packed_weights, bias, output, multiplier, shift,
                                                      rows, 8, 16, 3, 0, -128, 127, 16));
            for (int row = 0; row < rows; ++row)
            {
                for (int ch = 0; ch < 8; ++ch)
                {
                    int64_t acc = bias[ch];
                    for (int k = 0; k < 16; ++k)
                    {
                        const int index = ch * 16 + k;
                        const int nibble = ((uint8_t)packed_weights[index / 2] >> (4 * (index % 2))) & 15;
                        acc += (input[row * 16 + k] + 3) * (nibble < 8 ? nibble : nibble - 16);
                    }
#if defined(CMSIS_NN_USE_SINGLE_ROUNDING)
                    const int right = 31 - shift[ch];
                    int32_t expected = (int32_t)((acc * multiplier[ch] + ((int64_t)1 << (right - 1))) >> right);
#else
                    int32_t expected = arm_nn_requantize((int32_t)acc, multiplier[ch], shift[ch]);
#endif
                    expected = ARM_NN_MAX(-128, ARM_NN_MIN(127, expected));
                    TEST_ASSERT_EQUAL_INT8(expected, output[row * 8 + ch]);
                }
            }
        }
    }
}
