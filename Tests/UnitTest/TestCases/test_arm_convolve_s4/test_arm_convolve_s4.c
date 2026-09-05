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

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <unity.h>

#include "../TestData/basic_2_int4/test_data.h"
#include "../TestData/basic_int4/test_data.h"
#include "../TestData/conv_1_x_n_1_int4/test_data.h"
#include "../TestData/conv_1_x_n_2_int4/test_data.h"
#include "../TestData/conv_1_x_n_3_int4/test_data.h"
#include "../TestData/conv_1_x_n_4_int4/test_data.h"
#include "../TestData/conv_1_x_n_5_int4/test_data.h"
#include "../TestData/conv_2_int4/test_data.h"
#include "../TestData/conv_2x2_dilation_5x5_input_int4/test_data.h"
#include "../TestData/conv_2x2_dilation_int4/test_data.h"
#include "../TestData/conv_2x3_dilation_int4/test_data.h"
#include "../TestData/conv_3_int4/test_data.h"
#include "../TestData/conv_3x2_dilation_int4/test_data.h"
#include "../TestData/conv_3x3_dilation_5x5_input_int4/test_data.h"
#include "../TestData/conv_4_int4/test_data.h"
#include "../TestData/conv_5_int4/test_data.h"
#include "../TestData/conv_dilation_golden_int4/test_data.h"
#include "../TestData/conv_out_activation_int4/test_data.h"
#include "../TestData/stride2pad1_int4/test_data.h"
#include "../Utils/validate.h"

void basic_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[BASIC_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_int4_biases;
    const int8_t *kernel_data = basic_int4_weights;
    const int8_t *input_data = basic_int4_input;
    const int8_t *output_ref = basic_int4_output_ref;
    const int32_t output_ref_size = BASIC_INT4_DST_SIZE;

    input_dims.n = BASIC_INT4_INPUT_BATCHES;
    input_dims.w = BASIC_INT4_INPUT_W;
    input_dims.h = BASIC_INT4_INPUT_H;
    input_dims.c = BASIC_INT4_IN_CH;
    filter_dims.w = BASIC_INT4_FILTER_X;
    filter_dims.h = BASIC_INT4_FILTER_Y;
    output_dims.w = BASIC_INT4_OUTPUT_W;
    output_dims.h = BASIC_INT4_OUTPUT_H;
    output_dims.c = BASIC_INT4_OUT_CH;

    conv_params.padding.w = BASIC_INT4_PAD_X;
    conv_params.padding.h = BASIC_INT4_PAD_Y;
    conv_params.stride.w = BASIC_INT4_STRIDE_X;
    conv_params.stride.h = BASIC_INT4_STRIDE_Y;
    conv_params.dilation.w = BASIC_INT4_DILATION_X;
    conv_params.dilation.h = BASIC_INT4_DILATION_Y;

    conv_params.input_offset = BASIC_INT4_INPUT_OFFSET;
    conv_params.output_offset = BASIC_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = BASIC_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = BASIC_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_int4_output_mult;
    quant_params.shift = (int32_t *)basic_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void basic_2_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[BASIC_2_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_2_int4_biases;
    const int8_t *kernel_data = basic_2_int4_weights;
    const int8_t *input_data = basic_2_int4_input;
    const int8_t *output_ref = basic_2_int4_output_ref;
    const int32_t output_ref_size = BASIC_2_INT4_DST_SIZE;

    input_dims.n = BASIC_2_INT4_INPUT_BATCHES;
    input_dims.w = BASIC_2_INT4_INPUT_W;
    input_dims.h = BASIC_2_INT4_INPUT_H;
    input_dims.c = BASIC_2_INT4_IN_CH;
    filter_dims.w = BASIC_2_INT4_FILTER_X;
    filter_dims.h = BASIC_2_INT4_FILTER_Y;
    output_dims.w = BASIC_2_INT4_OUTPUT_W;
    output_dims.h = BASIC_2_INT4_OUTPUT_H;
    output_dims.c = BASIC_2_INT4_OUT_CH;

    conv_params.padding.w = BASIC_2_INT4_PAD_X;
    conv_params.padding.h = BASIC_2_INT4_PAD_Y;
    conv_params.stride.w = BASIC_2_INT4_STRIDE_X;
    conv_params.stride.h = BASIC_2_INT4_STRIDE_Y;
    conv_params.dilation.w = BASIC_2_INT4_DILATION_X;
    conv_params.dilation.h = BASIC_2_INT4_DILATION_Y;

    conv_params.input_offset = BASIC_2_INT4_INPUT_OFFSET;
    conv_params.output_offset = BASIC_2_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = BASIC_2_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = BASIC_2_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_2_int4_output_mult;
    quant_params.shift = (int32_t *)basic_2_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void stride2pad1_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[STRIDE2PAD1_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = stride2pad1_int4_biases;
    const int8_t *kernel_data = stride2pad1_int4_weights;
    const int8_t *input_data = stride2pad1_int4_input;
    const int8_t *output_ref = stride2pad1_int4_output_ref;
    const int32_t output_ref_size = STRIDE2PAD1_INT4_DST_SIZE;

    input_dims.n = STRIDE2PAD1_INT4_INPUT_BATCHES;
    input_dims.w = STRIDE2PAD1_INT4_INPUT_W;
    input_dims.h = STRIDE2PAD1_INT4_INPUT_H;
    input_dims.c = STRIDE2PAD1_INT4_IN_CH;
    filter_dims.w = STRIDE2PAD1_INT4_FILTER_X;
    filter_dims.h = STRIDE2PAD1_INT4_FILTER_Y;
    output_dims.w = STRIDE2PAD1_INT4_OUTPUT_W;
    output_dims.h = STRIDE2PAD1_INT4_OUTPUT_H;
    output_dims.c = STRIDE2PAD1_INT4_OUT_CH;

    conv_params.padding.w = STRIDE2PAD1_INT4_PAD_X;
    conv_params.padding.h = STRIDE2PAD1_INT4_PAD_Y;
    conv_params.stride.w = STRIDE2PAD1_INT4_STRIDE_X;
    conv_params.stride.h = STRIDE2PAD1_INT4_STRIDE_Y;
    conv_params.dilation.w = STRIDE2PAD1_INT4_DILATION_X;
    conv_params.dilation.h = STRIDE2PAD1_INT4_DILATION_Y;

    conv_params.input_offset = STRIDE2PAD1_INT4_INPUT_OFFSET;
    conv_params.output_offset = STRIDE2PAD1_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = STRIDE2PAD1_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = STRIDE2PAD1_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)stride2pad1_int4_output_mult;
    quant_params.shift = (int32_t *)stride2pad1_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_2_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2_int4_biases;
    const int8_t *kernel_data = conv_2_int4_weights;
    const int8_t *input_data = conv_2_int4_input;
    const int8_t *output_ref = conv_2_int4_output_ref;
    const int32_t output_ref_size = CONV_2_INT4_DST_SIZE;

    input_dims.n = CONV_2_INT4_INPUT_BATCHES;
    input_dims.w = CONV_2_INT4_INPUT_W;
    input_dims.h = CONV_2_INT4_INPUT_H;
    input_dims.c = CONV_2_INT4_IN_CH;
    filter_dims.w = CONV_2_INT4_FILTER_X;
    filter_dims.h = CONV_2_INT4_FILTER_Y;
    output_dims.w = CONV_2_INT4_OUTPUT_W;
    output_dims.h = CONV_2_INT4_OUTPUT_H;
    output_dims.c = CONV_2_INT4_OUT_CH;

    conv_params.padding.w = CONV_2_INT4_PAD_X;
    conv_params.padding.h = CONV_2_INT4_PAD_Y;
    conv_params.stride.w = CONV_2_INT4_STRIDE_X;
    conv_params.stride.h = CONV_2_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_2_INT4_DILATION_X;
    conv_params.dilation.h = CONV_2_INT4_DILATION_Y;

    conv_params.input_offset = CONV_2_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_2_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2_int4_output_mult;
    quant_params.shift = (int32_t *)conv_2_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_2_int4_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_3_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_3_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3_int4_biases;
    const int8_t *kernel_data = conv_3_int4_weights;
    const int8_t *input_data = conv_3_int4_input;
    const int8_t *output_ref = conv_3_int4_output_ref;
    const int32_t output_ref_size = CONV_3_INT4_DST_SIZE;

    input_dims.n = CONV_3_INT4_INPUT_BATCHES;
    input_dims.w = CONV_3_INT4_INPUT_W;
    input_dims.h = CONV_3_INT4_INPUT_H;
    input_dims.c = CONV_3_INT4_IN_CH;
    filter_dims.w = CONV_3_INT4_FILTER_X;
    filter_dims.h = CONV_3_INT4_FILTER_Y;
    output_dims.w = CONV_3_INT4_OUTPUT_W;
    output_dims.h = CONV_3_INT4_OUTPUT_H;
    output_dims.c = CONV_3_INT4_OUT_CH;

    conv_params.padding.w = CONV_3_INT4_PAD_X;
    conv_params.padding.h = CONV_3_INT4_PAD_Y;
    conv_params.stride.w = CONV_3_INT4_STRIDE_X;
    conv_params.stride.h = CONV_3_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_3_INT4_DILATION_X;
    conv_params.dilation.h = CONV_3_INT4_DILATION_Y;

    conv_params.input_offset = CONV_3_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_3_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3_int4_output_mult;
    quant_params.shift = (int32_t *)conv_3_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_3_int4_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_4_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_4_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_4_int4_biases;
    const int8_t *kernel_data = conv_4_int4_weights;
    const int8_t *input_data = conv_4_int4_input;
    const int8_t *output_ref = conv_4_int4_output_ref;
    const int32_t output_ref_size = CONV_4_INT4_DST_SIZE;

    input_dims.n = CONV_4_INT4_INPUT_BATCHES;
    input_dims.w = CONV_4_INT4_INPUT_W;
    input_dims.h = CONV_4_INT4_INPUT_H;
    input_dims.c = CONV_4_INT4_IN_CH;
    filter_dims.w = CONV_4_INT4_FILTER_X;
    filter_dims.h = CONV_4_INT4_FILTER_Y;
    output_dims.w = CONV_4_INT4_OUTPUT_W;
    output_dims.h = CONV_4_INT4_OUTPUT_H;
    output_dims.c = CONV_4_INT4_OUT_CH;

    conv_params.padding.w = CONV_4_INT4_PAD_X;
    conv_params.padding.h = CONV_4_INT4_PAD_Y;
    conv_params.stride.w = CONV_4_INT4_STRIDE_X;
    conv_params.stride.h = CONV_4_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_4_INT4_DILATION_X;
    conv_params.dilation.h = CONV_4_INT4_DILATION_Y;

    conv_params.input_offset = CONV_4_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_4_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_4_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_4_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_4_int4_output_mult;
    quant_params.shift = (int32_t *)conv_4_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_4_int4_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_out_activation_arm_convolve_s4(void)
{
    int8_t output[CONV_OUT_ACTIVATION_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_out_activation_int4_biases;
    const int8_t *kernel_data = conv_out_activation_int4_weights;
    const int8_t *input_data = conv_out_activation_int4_input;
    const int8_t *output_ref = conv_out_activation_int4_output_ref;
    const int32_t output_ref_size = CONV_OUT_ACTIVATION_INT4_DST_SIZE;

    input_dims.n = CONV_OUT_ACTIVATION_INT4_INPUT_BATCHES;
    input_dims.w = CONV_OUT_ACTIVATION_INT4_INPUT_W;
    input_dims.h = CONV_OUT_ACTIVATION_INT4_INPUT_H;
    input_dims.c = CONV_OUT_ACTIVATION_INT4_IN_CH;
    filter_dims.w = CONV_OUT_ACTIVATION_INT4_FILTER_X;
    filter_dims.h = CONV_OUT_ACTIVATION_INT4_FILTER_Y;
    output_dims.w = CONV_OUT_ACTIVATION_INT4_OUTPUT_W;
    output_dims.h = CONV_OUT_ACTIVATION_INT4_OUTPUT_H;
    output_dims.c = CONV_OUT_ACTIVATION_INT4_OUT_CH;

    conv_params.padding.w = CONV_OUT_ACTIVATION_INT4_PAD_X;
    conv_params.padding.h = CONV_OUT_ACTIVATION_INT4_PAD_Y;
    conv_params.stride.w = CONV_OUT_ACTIVATION_INT4_STRIDE_X;
    conv_params.stride.h = CONV_OUT_ACTIVATION_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_OUT_ACTIVATION_INT4_DILATION_X;
    conv_params.dilation.h = CONV_OUT_ACTIVATION_INT4_DILATION_Y;

    conv_params.input_offset = CONV_OUT_ACTIVATION_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_OUT_ACTIVATION_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_OUT_ACTIVATION_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_OUT_ACTIVATION_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_out_activation_int4_output_mult;
    quant_params.shift = (int32_t *)conv_out_activation_int4_output_shift;

    int32_t buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_wrapper_s4(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel_data,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2x2_dilation_arm_convolve_s4(void)
{
    int8_t output[CONV_2X2_DILATION_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int32_t *bias_data = conv_2x2_dilation_int4_biases;
    const int8_t *kernel_data = conv_2x2_dilation_int4_weights;
    const int8_t *input_data = conv_2x2_dilation_int4_input;
    const int8_t *output_ref = conv_2x2_dilation_int4_output_ref;
    const int32_t output_ref_size = CONV_2X2_DILATION_INT4_DST_SIZE;

    input_dims.n = CONV_2X2_DILATION_INT4_INPUT_BATCHES;
    input_dims.w = CONV_2X2_DILATION_INT4_INPUT_W;
    input_dims.h = CONV_2X2_DILATION_INT4_INPUT_H;
    input_dims.c = CONV_2X2_DILATION_INT4_IN_CH;
    filter_dims.w = CONV_2X2_DILATION_INT4_FILTER_X;
    filter_dims.h = CONV_2X2_DILATION_INT4_FILTER_Y;
    output_dims.w = CONV_2X2_DILATION_INT4_OUTPUT_W;
    output_dims.h = CONV_2X2_DILATION_INT4_OUTPUT_H;
    output_dims.c = CONV_2X2_DILATION_INT4_OUT_CH;

    conv_params.padding.w = CONV_2X2_DILATION_INT4_PAD_X;
    conv_params.padding.h = CONV_2X2_DILATION_INT4_PAD_Y;
    conv_params.stride.w = CONV_2X2_DILATION_INT4_STRIDE_X;
    conv_params.stride.h = CONV_2X2_DILATION_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_2X2_DILATION_INT4_DILATION_X;
    conv_params.dilation.h = CONV_2X2_DILATION_INT4_DILATION_Y;

    conv_params.input_offset = CONV_2X2_DILATION_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X2_DILATION_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X2_DILATION_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X2_DILATION_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x2_dilation_int4_output_mult;
    quant_params.shift = (int32_t *)conv_2x2_dilation_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2x2_dilation_5x5_input_arm_convolve_s4(void)
{
    int8_t output[CONV_2X2_DILATION_5X5_INPUT_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2x2_dilation_5x5_input_int4_biases;
    const int8_t *kernel_data = conv_2x2_dilation_5x5_input_int4_weights;
    const int8_t *input_data = conv_2x2_dilation_5x5_input_int4_input;
    const int8_t *output_ref = conv_2x2_dilation_5x5_input_int4_output_ref;
    const int32_t output_ref_size = CONV_2X2_DILATION_5X5_INPUT_INT4_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_2X2_DILATION_5X5_INPUT_INT4_INPUT_BATCHES;
    input_dims.w = CONV_2X2_DILATION_5X5_INPUT_INT4_INPUT_W;
    input_dims.h = CONV_2X2_DILATION_5X5_INPUT_INT4_INPUT_H;
    input_dims.c = CONV_2X2_DILATION_5X5_INPUT_INT4_IN_CH;
    filter_dims.w = CONV_2X2_DILATION_5X5_INPUT_INT4_FILTER_X;
    filter_dims.h = CONV_2X2_DILATION_5X5_INPUT_INT4_FILTER_Y;
    output_dims.w = CONV_2X2_DILATION_5X5_INPUT_INT4_OUTPUT_W;
    output_dims.h = CONV_2X2_DILATION_5X5_INPUT_INT4_OUTPUT_H;
    output_dims.c = CONV_2X2_DILATION_5X5_INPUT_INT4_OUT_CH;

    conv_params.padding.w = CONV_2X2_DILATION_5X5_INPUT_INT4_PAD_X;
    conv_params.padding.h = CONV_2X2_DILATION_5X5_INPUT_INT4_PAD_Y;
    conv_params.stride.w = CONV_2X2_DILATION_5X5_INPUT_INT4_STRIDE_X;
    conv_params.stride.h = CONV_2X2_DILATION_5X5_INPUT_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_2X2_DILATION_5X5_INPUT_INT4_DILATION_X;
    conv_params.dilation.h = CONV_2X2_DILATION_5X5_INPUT_INT4_DILATION_Y;

    conv_params.input_offset = CONV_2X2_DILATION_5X5_INPUT_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X2_DILATION_5X5_INPUT_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X2_DILATION_5X5_INPUT_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X2_DILATION_5X5_INPUT_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x2_dilation_5x5_input_int4_output_mult;
    quant_params.shift = (int32_t *)conv_2x2_dilation_5x5_input_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_3x3_dilation_5x5_input_arm_convolve_s4(void)
{
    int8_t output[CONV_3X3_DILATION_5X5_INPUT_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3x3_dilation_5x5_input_int4_biases;
    const int8_t *kernel_data = conv_3x3_dilation_5x5_input_int4_weights;
    const int8_t *input_data = conv_3x3_dilation_5x5_input_int4_input;
    const int8_t *output_ref = conv_3x3_dilation_5x5_input_int4_output_ref;
    const int32_t output_ref_size = CONV_3X3_DILATION_5X5_INPUT_INT4_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_3X3_DILATION_5X5_INPUT_INT4_INPUT_BATCHES;
    input_dims.w = CONV_3X3_DILATION_5X5_INPUT_INT4_INPUT_W;
    input_dims.h = CONV_3X3_DILATION_5X5_INPUT_INT4_INPUT_H;
    input_dims.c = CONV_3X3_DILATION_5X5_INPUT_INT4_IN_CH;
    filter_dims.w = CONV_3X3_DILATION_5X5_INPUT_INT4_FILTER_X;
    filter_dims.h = CONV_3X3_DILATION_5X5_INPUT_INT4_FILTER_Y;
    output_dims.w = CONV_3X3_DILATION_5X5_INPUT_INT4_OUTPUT_W;
    output_dims.h = CONV_3X3_DILATION_5X5_INPUT_INT4_OUTPUT_H;
    output_dims.c = CONV_3X3_DILATION_5X5_INPUT_INT4_OUT_CH;

    conv_params.padding.w = CONV_3X3_DILATION_5X5_INPUT_INT4_PAD_X;
    conv_params.padding.h = CONV_3X3_DILATION_5X5_INPUT_INT4_PAD_Y;
    conv_params.stride.w = CONV_3X3_DILATION_5X5_INPUT_INT4_STRIDE_X;
    conv_params.stride.h = CONV_3X3_DILATION_5X5_INPUT_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_3X3_DILATION_5X5_INPUT_INT4_DILATION_X;
    conv_params.dilation.h = CONV_3X3_DILATION_5X5_INPUT_INT4_DILATION_Y;

    conv_params.input_offset = CONV_3X3_DILATION_5X5_INPUT_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_3X3_DILATION_5X5_INPUT_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3X3_DILATION_5X5_INPUT_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3X3_DILATION_5X5_INPUT_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3x3_dilation_5x5_input_int4_output_mult;
    quant_params.shift = (int32_t *)conv_3x3_dilation_5x5_input_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2x3_dilation_arm_convolve_s4(void)
{
    int8_t output[CONV_2X3_DILATION_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2x3_dilation_int4_biases;
    const int8_t *kernel_data = conv_2x3_dilation_int4_weights;
    const int8_t *input_data = conv_2x3_dilation_int4_input;
    const int8_t *output_ref = conv_2x3_dilation_int4_output_ref;
    const int32_t output_ref_size = CONV_2X3_DILATION_INT4_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_2X3_DILATION_INT4_INPUT_BATCHES;
    input_dims.w = CONV_2X3_DILATION_INT4_INPUT_W;
    input_dims.h = CONV_2X3_DILATION_INT4_INPUT_H;
    input_dims.c = CONV_2X3_DILATION_INT4_IN_CH;
    filter_dims.w = CONV_2X3_DILATION_INT4_FILTER_X;
    filter_dims.h = CONV_2X3_DILATION_INT4_FILTER_Y;
    output_dims.w = CONV_2X3_DILATION_INT4_OUTPUT_W;
    output_dims.h = CONV_2X3_DILATION_INT4_OUTPUT_H;
    output_dims.c = CONV_2X3_DILATION_INT4_OUT_CH;

    conv_params.padding.w = CONV_2X3_DILATION_INT4_PAD_X;
    conv_params.padding.h = CONV_2X3_DILATION_INT4_PAD_Y;
    conv_params.stride.w = CONV_2X3_DILATION_INT4_STRIDE_X;
    conv_params.stride.h = CONV_2X3_DILATION_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_2X3_DILATION_INT4_DILATION_X;
    conv_params.dilation.h = CONV_2X3_DILATION_INT4_DILATION_Y;

    conv_params.input_offset = CONV_2X3_DILATION_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X3_DILATION_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X3_DILATION_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X3_DILATION_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x3_dilation_int4_output_mult;
    quant_params.shift = (int32_t *)conv_2x3_dilation_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_3x2_dilation_arm_convolve_s4(void)
{
    int8_t output[CONV_3X2_DILATION_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3x2_dilation_int4_biases;
    const int8_t *kernel_data = conv_3x2_dilation_int4_weights;
    const int8_t *input_data = conv_3x2_dilation_int4_input;
    const int8_t *output_ref = conv_3x2_dilation_int4_output_ref;
    const int32_t output_ref_size = CONV_3X2_DILATION_INT4_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_3X2_DILATION_INT4_INPUT_BATCHES;
    input_dims.w = CONV_3X2_DILATION_INT4_INPUT_W;
    input_dims.h = CONV_3X2_DILATION_INT4_INPUT_H;
    input_dims.c = CONV_3X2_DILATION_INT4_IN_CH;
    filter_dims.w = CONV_3X2_DILATION_INT4_FILTER_X;
    filter_dims.h = CONV_3X2_DILATION_INT4_FILTER_Y;
    output_dims.w = CONV_3X2_DILATION_INT4_OUTPUT_W;
    output_dims.h = CONV_3X2_DILATION_INT4_OUTPUT_H;
    output_dims.c = CONV_3X2_DILATION_INT4_OUT_CH;

    conv_params.padding.w = CONV_3X2_DILATION_INT4_PAD_X;
    conv_params.padding.h = CONV_3X2_DILATION_INT4_PAD_Y;
    conv_params.stride.w = CONV_3X2_DILATION_INT4_STRIDE_X;
    conv_params.stride.h = CONV_3X2_DILATION_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_3X2_DILATION_INT4_DILATION_X;
    conv_params.dilation.h = CONV_3X2_DILATION_INT4_DILATION_Y;

    conv_params.input_offset = CONV_3X2_DILATION_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_3X2_DILATION_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3X2_DILATION_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3X2_DILATION_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3x2_dilation_int4_output_mult;
    quant_params.shift = (int32_t *)conv_3x2_dilation_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_dilation_golden_arm_convolve_s4(void)
{
    int8_t output[CONV_DILATION_GOLDEN_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_dilation_golden_int4_biases;
    const int8_t *kernel_data = conv_dilation_golden_int4_weights;
    const int8_t *input_data = conv_dilation_golden_int4_input;
    const int8_t *output_ref = conv_dilation_golden_int4_output_ref;
    const int32_t output_ref_size = CONV_DILATION_GOLDEN_INT4_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_DILATION_GOLDEN_INT4_INPUT_BATCHES;
    input_dims.w = CONV_DILATION_GOLDEN_INT4_INPUT_W;
    input_dims.h = CONV_DILATION_GOLDEN_INT4_INPUT_H;
    input_dims.c = CONV_DILATION_GOLDEN_INT4_IN_CH;
    filter_dims.w = CONV_DILATION_GOLDEN_INT4_FILTER_X;
    filter_dims.h = CONV_DILATION_GOLDEN_INT4_FILTER_Y;
    output_dims.w = CONV_DILATION_GOLDEN_INT4_OUTPUT_W;
    output_dims.h = CONV_DILATION_GOLDEN_INT4_OUTPUT_H;
    output_dims.c = CONV_DILATION_GOLDEN_INT4_OUT_CH;

    conv_params.padding.w = CONV_DILATION_GOLDEN_INT4_PAD_X;
    conv_params.padding.h = CONV_DILATION_GOLDEN_INT4_PAD_Y;
    conv_params.stride.w = CONV_DILATION_GOLDEN_INT4_STRIDE_X;
    conv_params.stride.h = CONV_DILATION_GOLDEN_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_DILATION_GOLDEN_INT4_DILATION_X;
    conv_params.dilation.h = CONV_DILATION_GOLDEN_INT4_DILATION_Y;

    conv_params.input_offset = CONV_DILATION_GOLDEN_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_DILATION_GOLDEN_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_DILATION_GOLDEN_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_DILATION_GOLDEN_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_dilation_golden_int4_output_mult;
    quant_params.shift = (int32_t *)conv_dilation_golden_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_5_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_5_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_5_int4_biases;
    const int8_t *kernel_data = conv_5_int4_weights;
    const int8_t *input_data = conv_5_int4_input;
    const int8_t *output_ref = conv_5_int4_output_ref;
    const int32_t output_ref_size = CONV_5_INT4_DST_SIZE;

    input_dims.n = CONV_5_INT4_INPUT_BATCHES;
    input_dims.w = CONV_5_INT4_INPUT_W;
    input_dims.h = CONV_5_INT4_INPUT_H;
    input_dims.c = CONV_5_INT4_IN_CH;
    filter_dims.w = CONV_5_INT4_FILTER_X;
    filter_dims.h = CONV_5_INT4_FILTER_Y;
    output_dims.w = CONV_5_INT4_OUTPUT_W;
    output_dims.h = CONV_5_INT4_OUTPUT_H;
    output_dims.c = CONV_5_INT4_OUT_CH;

    conv_params.padding.w = CONV_5_INT4_PAD_X;
    conv_params.padding.h = CONV_5_INT4_PAD_Y;
    conv_params.stride.w = CONV_5_INT4_STRIDE_X;
    conv_params.stride.h = CONV_5_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_5_INT4_DILATION_X;
    conv_params.dilation.h = CONV_5_INT4_DILATION_Y;

    conv_params.input_offset = CONV_5_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_5_int4_output_mult;
    quant_params.shift = (int32_t *)conv_5_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_s4(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_5_int4_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &conv_params,
                                     &quant_params,
                                     &input_dims,
                                     input_data,
                                     &filter_dims,
                                     kernel_data,
                                     &bias_dims,
                                     bias_data,
                                     &output_dims,
                                     output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void buffer_size_arm_convolve_s4(void)
{
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INT4_INPUT_BATCHES;
    input_dims.w = CONV_5_INT4_INPUT_W;
    input_dims.h = CONV_5_INT4_INPUT_H;
    input_dims.c = CONV_5_INT4_IN_CH;
    filter_dims.w = CONV_5_INT4_FILTER_X;
    filter_dims.h = CONV_5_INT4_FILTER_Y;
    output_dims.w = CONV_5_INT4_OUTPUT_W;
    output_dims.h = CONV_5_INT4_OUTPUT_H;
    output_dims.c = CONV_5_INT4_OUT_CH;

    conv_params.padding.w = CONV_5_INT4_PAD_X;
    conv_params.padding.h = CONV_5_INT4_PAD_Y;
    conv_params.stride.w = CONV_5_INT4_STRIDE_X;
    conv_params.stride.h = CONV_5_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_5_INT4_DILATION_X;
    conv_params.dilation.h = CONV_5_INT4_DILATION_Y;

    conv_params.input_offset = CONV_5_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_INT4_OUT_ACTIVATION_MAX;

    const int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, buf_size);
}

void buffer_size_mve_arm_convolve_s4(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INT4_INPUT_BATCHES;
    input_dims.w = CONV_5_INT4_INPUT_W;
    input_dims.h = CONV_5_INT4_INPUT_H;
    input_dims.c = CONV_5_INT4_IN_CH;
    filter_dims.w = CONV_5_INT4_FILTER_X;
    filter_dims.h = CONV_5_INT4_FILTER_Y;
    output_dims.w = CONV_5_INT4_OUTPUT_W;
    output_dims.h = CONV_5_INT4_OUTPUT_H;
    output_dims.c = CONV_5_INT4_OUT_CH;

    conv_params.padding.w = CONV_5_INT4_PAD_X;
    conv_params.padding.h = CONV_5_INT4_PAD_Y;
    conv_params.stride.w = CONV_5_INT4_STRIDE_X;
    conv_params.stride.h = CONV_5_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_5_INT4_DILATION_X;
    conv_params.dilation.h = CONV_5_INT4_DILATION_Y;

    conv_params.input_offset = CONV_5_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_INT4_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_convolve_s4(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INT4_INPUT_BATCHES;
    input_dims.w = CONV_5_INT4_INPUT_W;
    input_dims.h = CONV_5_INT4_INPUT_H;
    input_dims.c = CONV_5_INT4_IN_CH;
    filter_dims.w = CONV_5_INT4_FILTER_X;
    filter_dims.h = CONV_5_INT4_FILTER_Y;
    output_dims.w = CONV_5_INT4_OUTPUT_W;
    output_dims.h = CONV_5_INT4_OUTPUT_H;
    output_dims.c = CONV_5_INT4_OUT_CH;

    conv_params.padding.w = CONV_5_INT4_PAD_X;
    conv_params.padding.h = CONV_5_INT4_PAD_Y;
    conv_params.stride.w = CONV_5_INT4_STRIDE_X;
    conv_params.stride.h = CONV_5_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_5_INT4_DILATION_X;
    conv_params.dilation.h = CONV_5_INT4_DILATION_Y;

    conv_params.input_offset = CONV_5_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_INT4_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_convolve_wrapper_s4_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}

void buffer_size_out_of_range_arm_convolve_s4(void)
{
    /* Dimension validation is deliberately target-independent, so these hold on every build. Before the sizers
     * were hardened the products below wrapped in int32_t: the issue #317 shape, 1 * 1 * (2^30 + 1) columns, made
     * arm_convolve_s4_get_buffer_size() report 4 bytes for a buffer the kernel walks for about 4 GiB, and
     * 2^29 + 1 columns produced a wrapped negative value that is not the -1 sentinel. */
    const cmsis_nn_conv_params conv_params = {
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
    };
    const cmsis_nn_dims output_dims = {1, 1, 1, 8};

    /* A small valid shape still yields the published figure, so the range checks do not disturb the formula:
     * 2 * (1 * 1 * 3) * sizeof(int16_t). */
    const cmsis_nn_dims valid_input_dims = {1, 1, 1, 3};
    const cmsis_nn_dims valid_filter_dims = {8, 1, 1, 3};
    TEST_ASSERT_EQUAL(12, arm_convolve_s4_get_buffer_size(&valid_input_dims, &valid_filter_dims));

    /* c = 2^30 + 1 with a 1x1 filter: 2 * rhs_cols * sizeof(int16_t) exceeds INT32_MAX. */
    const cmsis_nn_dims c_2_pow_30_input_dims = {1, 1, 1, 1073741825};
    const cmsis_nn_dims c_2_pow_30_filter_dims = {8, 1, 1, 1073741825};
    TEST_ASSERT_EQUAL(-1, arm_convolve_s4_get_buffer_size(&c_2_pow_30_input_dims, &c_2_pow_30_filter_dims));

    /* The 1xN sizer must report the same shape. pad_x = 1 with a single output column keeps the Helium leg on the
     * im2col route, the one that computes a byte count; its pad-aligned route needs no buffer and returns 0. */
    const cmsis_nn_conv_params padded_conv_params = {
        .stride = {1, 1},
        .padding = {1, 0},
        .dilation = {1, 1},
    };
    TEST_ASSERT_EQUAL(-1,
                      arm_convolve_1_x_n_s4_get_buffer_size(
                          &padded_conv_params, &c_2_pow_30_input_dims, &c_2_pow_30_filter_dims, &output_dims));

    /* The wrapper sizer must report it too. A 2x2 filter keeps the wrapper off the 1x1 routes, which need no buffer
     * on any build; c = 2^28 + 1 then wraps the unguarded product to 16. */
    const cmsis_nn_dims c_2_pow_28_input_dims = {1, 2, 2, 268435457};
    const cmsis_nn_dims c_2_pow_28_filter_dims = {8, 2, 2, 268435457};
    TEST_ASSERT_EQUAL(-1,
                      arm_convolve_wrapper_s4_get_buffer_size(
                          &conv_params, &c_2_pow_28_input_dims, &c_2_pow_28_filter_dims, &output_dims));

    /* c = 2^29 + 1: rhs_cols itself fits, but the byte count wraps to a negative value that is not -1. */
    const cmsis_nn_dims c_2_pow_29_input_dims = {1, 1, 1, 536870913};
    const cmsis_nn_dims c_2_pow_29_filter_dims = {8, 1, 1, 536870913};
    TEST_ASSERT_EQUAL(-1, arm_convolve_s4_get_buffer_size(&c_2_pow_29_input_dims, &c_2_pow_29_filter_dims));

    /* A negative dimension is rejected by every sizer, on every route that reads it. */
    const cmsis_nn_dims negative_c_input_dims = {1, 2, 2, -1};
    const cmsis_nn_dims negative_c_filter_dims = {8, 2, 2, -1};
    TEST_ASSERT_EQUAL(-1, arm_convolve_s4_get_buffer_size(&negative_c_input_dims, &negative_c_filter_dims));
    TEST_ASSERT_EQUAL(-1, arm_convolve_1x1_s4_fast_get_buffer_size(&negative_c_input_dims));
    TEST_ASSERT_EQUAL(-1,
                      arm_convolve_1_x_n_s4_get_buffer_size(
                          &conv_params, &negative_c_input_dims, &negative_c_filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(-1,
                      arm_convolve_wrapper_s4_get_buffer_size(
                          &conv_params, &negative_c_input_dims, &negative_c_filter_dims, &output_dims));

    /* A non-positive stride is rejected by the 1xN sizer before it can divide by it. */
    const cmsis_nn_conv_params zero_stride_conv_params = {
        .stride = {0, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
    };
    TEST_ASSERT_EQUAL(-1,
                      arm_convolve_1_x_n_s4_get_buffer_size(
                          &zero_stride_conv_params, &valid_input_dims, &valid_filter_dims, &output_dims));
}

// Issue #378: the arm_convolve_even_s4 ctx documentation used to cite arm_convolve_s4_get_buffer_size() in prose --
// an exact fit with zero slack (up to four im2col rows of rhs_cols int8 vs 2 * rhs_cols * sizeof(int16_t)) that
// nothing enforced, so a drift on either side would have shipped an out-of-bounds write with no test to catch it.
// arm_convolve_even_s4_get_buffer_size() now makes the citation a checkable name, and this sweep pins the forward:
// the two sizers must answer identically for every shape, boundary and out-of-range shapes included.
void buffer_size_even_arm_convolve_s4(void)
{
    const struct
    {
        int32_t filter_w;
        int32_t filter_h;
        int32_t in_ch;
    } shapes[] = {
        {1, 1, 2},          // smallest legal even_s4 shape: rhs_cols = 2
        {1, 1, 1},          // rhs_cols odd: even_s4 rejects it at run time, but the sizers must still agree
        {2, 2, 16},         // rhs_cols = 64: whole 32-byte interleave blocks, no matmul tail
        {3, 3, 5},          // rhs_cols = 45, exercising both the 16-element spill and the predicated remainder
        {1, 1, 16},         // rhs_cols = 16: the 16-element spill alone
        {5, 5, 2},          // rhs_cols = 50: spill plus a 2-byte remainder
        {7, 7, 6},          // rhs_cols = 294
        {1, 1, 536870911},  // rhs_cols = 2^29 - 1: the largest byte count that still fits an int32_t
        {1, 1, 536870912},  // rhs_cols = 2^29: 4 * rhs_cols is 2^31, one past INT32_MAX, both must answer -1
        {1, 1, 1073741825}, // the issue #317 shape: both must answer -1, not a wrapped 4
        {2, 2, 268435457},  // the byte count (4 * rhs_cols) overflows int32: both must answer -1
        {-1, 1, 2},         // negative dims must agree too: both -1, on every field either sizer reads
        {1, -1, 2},
        {1, 1, -1},
    };

    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
    {
        const cmsis_nn_dims input_dims = {1, 8, 8, shapes[i].in_ch};
        const cmsis_nn_dims filter_dims = {8, shapes[i].filter_h, shapes[i].filter_w, shapes[i].in_ch};

        TEST_ASSERT_EQUAL(arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims),
                          arm_convolve_even_s4_get_buffer_size(&input_dims, &filter_dims));
    }

    /* The forward is only safe while the s4 figure covers the even_s4 kernel's writes: four im2col rows of
     * rhs_cols int8 elements. Pin the published formula on one shape so a resize of either side is caught here
     * and not by a heap overflow: 2 * (2 * 2 * 16) * sizeof(int16_t) = 4 * 64. */
    const cmsis_nn_dims pinned_input_dims = {1, 8, 8, 16};
    const cmsis_nn_dims pinned_filter_dims = {8, 2, 2, 16};
    TEST_ASSERT_EQUAL(256, arm_convolve_even_s4_get_buffer_size(&pinned_input_dims, &pinned_filter_dims));

    /* Negative dims answer with the -1 sentinel itself, not merely any two equal values. */
    const cmsis_nn_dims negative_input_dims = {1, 8, 8, -1};
    const cmsis_nn_dims negative_filter_dims = {8, 1, 1, -1};
    TEST_ASSERT_EQUAL(-1, arm_convolve_even_s4_get_buffer_size(&negative_input_dims, &negative_filter_dims));
}

/* One generated s4 fixture reduced to the arguments arm_convolve_even_s4 and arm_convolve_wrapper_s4 both take,
 * so the same vectors can be driven through the kernel directly and through the wrapper. */
typedef struct
{
    const int8_t *input;
    const int8_t *weights;
    const int32_t *biases;
    const int8_t *output_ref;
    const int32_t *output_mult;
    const int32_t *output_shift;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;
    cmsis_nn_conv_params conv_params;
    int32_t output_size;
} conv_s4_fixture;

#define CONV_S4_FIXTURE(name, NAME)                                                                                    \
    {                                                                                                                  \
        .input = name##_input, .weights = name##_weights, .biases = name##_biases,                                     \
        .output_ref = name##_output_ref, .output_mult = name##_output_mult, .output_shift = name##_output_shift,       \
        .input_dims = {NAME##_INPUT_BATCHES, NAME##_INPUT_H, NAME##_INPUT_W, NAME##_IN_CH},                            \
        .filter_dims = {NAME##_OUT_CH, NAME##_FILTER_Y, NAME##_FILTER_X, NAME##_IN_CH},                                \
        .output_dims = {NAME##_INPUT_BATCHES, NAME##_OUTPUT_H, NAME##_OUTPUT_W, NAME##_OUT_CH},                        \
        .conv_params = {.input_offset = NAME##_INPUT_OFFSET,                                                           \
                        .output_offset = NAME##_OUTPUT_OFFSET,                                                         \
                        .stride = {NAME##_STRIDE_X, NAME##_STRIDE_Y},                                                  \
                        .padding = {NAME##_PAD_X, NAME##_PAD_Y},                                                       \
                        .dilation = {NAME##_DILATION_X, NAME##_DILATION_Y},                                            \
                        .activation = {NAME##_OUT_ACTIVATION_MIN, NAME##_OUT_ACTIVATION_MAX}},                         \
        .output_size = NAME##_DST_SIZE                                                                                 \
    }

/* Runs one fixture through arm_convolve_even_s4 and then, on the same vectors, through arm_convolve_wrapper_s4.
 * expected_even is what the kernel must answer for the fixture's rhs_cols parity. see AmbiqAI/ns-cmsis-nn#378 */
static void run_even_s4_fixture(const conv_s4_fixture *fixture, const arm_cmsis_nn_status expected_even)
{
    cmsis_nn_context ctx;
    cmsis_nn_dims bias_dims = {0, 0, 0, fixture->output_dims.c};
    cmsis_nn_per_channel_quant_params quant_params = {(int32_t *)fixture->output_mult,
                                                      (int32_t *)fixture->output_shift};

    int8_t *output = malloc(fixture->output_size);
    TEST_ASSERT_NOT_NULL(output);
    memset(output, 0, fixture->output_size);

    int32_t buf_size = arm_convolve_even_s4_get_buffer_size(&fixture->input_dims, &fixture->filter_dims);
    ctx.buf = malloc(buf_size);
    TEST_ASSERT_NOT_NULL(ctx.buf);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_even_s4(&ctx,
                                                      &fixture->conv_params,
                                                      &quant_params,
                                                      &fixture->input_dims,
                                                      fixture->input,
                                                      &fixture->filter_dims,
                                                      fixture->weights,
                                                      &bias_dims,
                                                      fixture->biases,
                                                      &fixture->output_dims,
                                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

#if defined(ARM_MATH_MVEI)
    TEST_ASSERT_EQUAL(expected_even, result);
    if (expected_even == ARM_CMSIS_NN_SUCCESS)
    {
        TEST_ASSERT_TRUE(validate(output, fixture->output_ref, fixture->output_size));
    }
#else
    /* Off MVE the kernel is a stub, so the wrapper half below is what carries this fixture on m0/m4 and the host. */
    (void)expected_even;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, result);
#endif

    memset(output, 0, fixture->output_size);
    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&fixture->conv_params,
                                                       &fixture->input_dims,
                                                       &fixture->filter_dims,
                                                       &fixture->output_dims);
    ctx.buf = malloc(buf_size);
    TEST_ASSERT_NOT_NULL(ctx.buf);
    ctx.size = 0;

    result = arm_convolve_wrapper_s4(&ctx,
                                     &fixture->conv_params,
                                     &quant_params,
                                     &fixture->input_dims,
                                     fixture->input,
                                     &fixture->filter_dims,
                                     fixture->weights,
                                     &bias_dims,
                                     fixture->biases,
                                     &fixture->output_dims,
                                     output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, fixture->output_ref, fixture->output_size));

    free(output);
}

// Issue #378: arm_convolve_even_s4 is public and MVE-only, and no test called it directly -- it was reached only as a
// side effect of arm_convolve_wrapper_s4 picking its branch, so nothing pinned the public entry point or the branch
// predicate that leads to it. These fixtures assert the wrapper's own guards (not 1x1, not 1xN, even rhs_cols) with
// the same helpers the wrapper uses, then require the kernel and the wrapper to reproduce the generated reference
// tensor element for element. see AmbiqAI/ns-cmsis-nn#378
void even_arm_convolve_s4(void)
{
    static const conv_s4_fixture fixtures[] = {
        CONV_S4_FIXTURE(basic_int4, BASIC_INT4),   // rhs_cols = 4 * 2 * 1 = 8, no padding
        CONV_S4_FIXTURE(conv_2_int4, CONV_2_INT4), // rhs_cols = 3 * 3 * 2 = 18, padded, clamped activation
        // rhs_cols = 3 * 3 * 128 = 1152, stride 4. The kernel's blk_cnt is rhs_cols >> 5, so the two above skip
        // the 32-byte vld2q/vstrbq interleave entirely; this one runs 36 blocks, but its trailing block is
        // constant padding for every output pixel, so on its own it cannot see a block-count drift.
        CONV_S4_FIXTURE(conv_5_int4, CONV_5_INT4),
        // rhs_cols = 10 * 4 * 3 = 120, so three interleave blocks, and they carry live input rather than padding:
        // dropping one block changes 228 of 250 outputs. see AmbiqAI/ns-cmsis-nn#378
        CONV_S4_FIXTURE(conv_3_int4, CONV_3_INT4),
    };

    for (size_t i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); i++)
    {
        const conv_s4_fixture *fixture = &fixtures[i];
        const int32_t rhs_cols = fixture->filter_dims.h * fixture->filter_dims.w * fixture->input_dims.c;

        /* Pins the fixture's shape class against the same helpers the wrapper's predicate uses. Which kernel the
         * wrapper picks is not observable from the output: the even branch is a performance path that returns the
         * same numbers as arm_convolve_s4. */
        TEST_ASSERT_FALSE(arm_nn_is_convolve_1x1(&fixture->conv_params, &fixture->input_dims, &fixture->filter_dims));
        TEST_ASSERT_FALSE(arm_nn_is_convolve_1_x_n(&fixture->conv_params, &fixture->input_dims, &fixture->filter_dims));
        TEST_ASSERT_EQUAL(0, rhs_cols & 0x1);

        run_even_s4_fixture(fixture, ARM_CMSIS_NN_SUCCESS);
    }
}

// Issue #378 companion: the odd side of the same predicate. arm_convolve_even_s4 must refuse an odd rhs_cols rather
// than read past the interleaved im2col rows, and arm_convolve_wrapper_s4 must still answer correctly for that shape
// by declining its even branch, so the branch boundary itself is covered and not just the kernel.
void even_odd_shape_arm_convolve_s4(void)
{
    static const conv_s4_fixture fixture = CONV_S4_FIXTURE(basic_2_int4, BASIC_2_INT4);
    const int32_t rhs_cols = fixture.filter_dims.h * fixture.filter_dims.w * fixture.input_dims.c;

    TEST_ASSERT_FALSE(arm_nn_is_convolve_1x1(&fixture.conv_params, &fixture.input_dims, &fixture.filter_dims));
    TEST_ASSERT_FALSE(arm_nn_is_convolve_1_x_n(&fixture.conv_params, &fixture.input_dims, &fixture.filter_dims));
    TEST_ASSERT_EQUAL(1, rhs_cols & 0x1); // 5 * 5 * 5 = 125

    run_even_s4_fixture(&fixture, ARM_CMSIS_NN_ARG_ERROR);
}

void conv_1_x_n_1_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_1_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_1_x_n_1_int4_biases;
    const int8_t *kernel_data = conv_1_x_n_1_int4_weights;
    const int8_t *input_data = conv_1_x_n_1_int4_input;
    const int8_t *output_ref = conv_1_x_n_1_int4_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_1_INT4_DST_SIZE;

    input_dims.n = CONV_1_X_N_1_INT4_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_1_INT4_INPUT_W;
    input_dims.h = CONV_1_X_N_1_INT4_INPUT_H;
    input_dims.c = CONV_1_X_N_1_INT4_IN_CH;
    filter_dims.w = CONV_1_X_N_1_INT4_FILTER_X;
    filter_dims.h = CONV_1_X_N_1_INT4_FILTER_Y;
    output_dims.w = CONV_1_X_N_1_INT4_OUTPUT_W;
    output_dims.h = CONV_1_X_N_1_INT4_OUTPUT_H;
    output_dims.c = CONV_1_X_N_1_INT4_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_1_INT4_PAD_X;
    conv_params.padding.h = CONV_1_X_N_1_INT4_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_1_INT4_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_1_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_1_INT4_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_1_INT4_DILATION_Y;

    conv_params.input_offset = CONV_1_X_N_1_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_1_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_1_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_1_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_1_int4_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_1_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_wrapper_s4(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel_data,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_1_x_n_s4(&ctx,
                                   &conv_params,
                                   &quant_params,
                                   &input_dims,
                                   input_data,
                                   &filter_dims,
                                   kernel_data,
                                   &bias_dims,
                                   bias_data,
                                   &output_dims,
                                   output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_2_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_2_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_1_x_n_2_int4_biases;
    const int8_t *kernel_data = conv_1_x_n_2_int4_weights;
    const int8_t *input_data = conv_1_x_n_2_int4_input;
    const int8_t *output_ref = conv_1_x_n_2_int4_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_2_INT4_DST_SIZE;

    input_dims.n = CONV_1_X_N_2_INT4_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_2_INT4_INPUT_W;
    input_dims.h = CONV_1_X_N_2_INT4_INPUT_H;
    input_dims.c = CONV_1_X_N_2_INT4_IN_CH;
    filter_dims.w = CONV_1_X_N_2_INT4_FILTER_X;
    filter_dims.h = CONV_1_X_N_2_INT4_FILTER_Y;
    output_dims.w = CONV_1_X_N_2_INT4_OUTPUT_W;
    output_dims.h = CONV_1_X_N_2_INT4_OUTPUT_H;
    output_dims.c = CONV_1_X_N_2_INT4_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_2_INT4_PAD_X;
    conv_params.padding.h = CONV_1_X_N_2_INT4_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_2_INT4_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_2_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_2_INT4_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_2_INT4_DILATION_Y;

    conv_params.input_offset = CONV_1_X_N_2_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_2_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_2_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_2_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_2_int4_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_2_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_wrapper_s4(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel_data,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    ctx.buf = malloc(buf_size);

    result = arm_convolve_1_x_n_s4(&ctx,
                                   &conv_params,
                                   &quant_params,
                                   &input_dims,
                                   input_data,
                                   &filter_dims,
                                   kernel_data,
                                   &bias_dims,
                                   bias_data,
                                   &output_dims,
                                   output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_3_arm_convolve_s4(void)
{
    int8_t output[CONV_1_X_N_3_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_1_x_n_3_int4_biases;
    const int8_t *kernel_data = conv_1_x_n_3_int4_weights;
    const int8_t *input_data = conv_1_x_n_3_int4_input;
    const int8_t *output_ref = conv_1_x_n_3_int4_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_3_INT4_DST_SIZE;

    input_dims.n = CONV_1_X_N_3_INT4_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_3_INT4_INPUT_W;
    input_dims.h = CONV_1_X_N_3_INT4_INPUT_H;
    input_dims.c = CONV_1_X_N_3_INT4_IN_CH;
    filter_dims.w = CONV_1_X_N_3_INT4_FILTER_X;
    filter_dims.h = CONV_1_X_N_3_INT4_FILTER_Y;
    output_dims.w = CONV_1_X_N_3_INT4_OUTPUT_W;
    output_dims.h = CONV_1_X_N_3_INT4_OUTPUT_H;
    output_dims.c = CONV_1_X_N_3_INT4_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_3_INT4_PAD_X;
    conv_params.padding.h = CONV_1_X_N_3_INT4_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_3_INT4_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_3_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_3_INT4_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_3_INT4_DILATION_Y;

    conv_params.input_offset = CONV_1_X_N_3_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_3_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_3_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_3_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_3_int4_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_3_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_wrapper_s4(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel_data,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_1_x_n_s4(&ctx,
                                   &conv_params,
                                   &quant_params,
                                   &input_dims,
                                   input_data,
                                   &filter_dims,
                                   kernel_data,
                                   &bias_dims,
                                   bias_data,
                                   &output_dims,
                                   output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_4_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_4_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_1_x_n_4_int4_biases;
    const int8_t *kernel_data = conv_1_x_n_4_int4_weights;
    const int8_t *input_data = conv_1_x_n_4_int4_input;
    const int8_t *output_ref = conv_1_x_n_4_int4_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_4_INT4_DST_SIZE;

    input_dims.n = CONV_1_X_N_4_INT4_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_4_INT4_INPUT_W;
    input_dims.h = CONV_1_X_N_4_INT4_INPUT_H;
    input_dims.c = CONV_1_X_N_4_INT4_IN_CH;
    filter_dims.w = CONV_1_X_N_4_INT4_FILTER_X;
    filter_dims.h = CONV_1_X_N_4_INT4_FILTER_Y;
    output_dims.w = CONV_1_X_N_4_INT4_OUTPUT_W;
    output_dims.h = CONV_1_X_N_4_INT4_OUTPUT_H;
    output_dims.c = CONV_1_X_N_4_INT4_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_4_INT4_PAD_X;
    conv_params.padding.h = CONV_1_X_N_4_INT4_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_4_INT4_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_4_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_4_INT4_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_4_INT4_DILATION_Y;

    conv_params.input_offset = CONV_1_X_N_4_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_4_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_4_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_4_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_4_int4_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_4_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_wrapper_s4(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel_data,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    ctx.buf = malloc(buf_size);

    result = arm_convolve_1_x_n_s4(&ctx,
                                   &conv_params,
                                   &quant_params,
                                   &input_dims,
                                   input_data,
                                   &filter_dims,
                                   kernel_data,
                                   &bias_dims,
                                   bias_data,
                                   &output_dims,
                                   output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_5_arm_convolve_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_5_INT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_1_x_n_5_int4_biases;
    const int8_t *kernel_data = conv_1_x_n_5_int4_weights;
    const int8_t *input_data = conv_1_x_n_5_int4_input;
    const int8_t *output_ref = conv_1_x_n_5_int4_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_5_INT4_DST_SIZE;

    input_dims.n = CONV_1_X_N_5_INT4_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_5_INT4_INPUT_W;
    input_dims.h = CONV_1_X_N_5_INT4_INPUT_H;
    input_dims.c = CONV_1_X_N_5_INT4_IN_CH;
    filter_dims.w = CONV_1_X_N_5_INT4_FILTER_X;
    filter_dims.h = CONV_1_X_N_5_INT4_FILTER_Y;
    output_dims.w = CONV_1_X_N_5_INT4_OUTPUT_W;
    output_dims.h = CONV_1_X_N_5_INT4_OUTPUT_H;
    output_dims.c = CONV_1_X_N_5_INT4_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_5_INT4_PAD_X;
    conv_params.padding.h = CONV_1_X_N_5_INT4_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_5_INT4_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_5_INT4_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_5_INT4_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_5_INT4_DILATION_Y;

    conv_params.input_offset = CONV_1_X_N_5_INT4_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_5_INT4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_5_INT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_5_INT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_5_int4_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_5_int4_output_shift;

    int32_t buf_size = arm_convolve_s4_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_wrapper_s4(&ctx,
                                                         &conv_params,
                                                         &quant_params,
                                                         &input_dims,
                                                         input_data,
                                                         &filter_dims,
                                                         kernel_data,
                                                         &bias_dims,
                                                         bias_data,
                                                         &output_dims,
                                                         output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s4_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_1_x_n_s4(&ctx,
                                   &conv_params,
                                   &quant_params,
                                   &input_dims,
                                   input_data,
                                   &filter_dims,
                                   kernel_data,
                                   &bias_dims,
                                   bias_data,
                                   &output_dims,
                                   output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}
