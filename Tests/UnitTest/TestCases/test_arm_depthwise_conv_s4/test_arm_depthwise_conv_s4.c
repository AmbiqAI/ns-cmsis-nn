/*
 * SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unity.h>

#include "../TestData/depthwise_int4_generic/test_data.h"
#include "../TestData/depthwise_int4_generic_2/test_data.h"
#include "../TestData/depthwise_int4_generic_3/test_data.h"
#include "../TestData/depthwise_int4_generic_4/test_data.h"
#include "../TestData/depthwise_int4_generic_5/test_data.h"
#include "../TestData/depthwise_int4_generic_6/test_data.h"
#include "../Utils/utils.h"
#include "../Utils/validate.h"

void depthwise_int4_generic_arm_depthwise_conv_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_GENERIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_generic_biases;
    const int8_t *kernel_data = depthwise_int4_generic_weights;
    const int8_t *input_data = depthwise_int4_generic_input;

    input_dims.n = DEPTHWISE_INT4_GENERIC_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_GENERIC_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_GENERIC_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_GENERIC_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_GENERIC_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_GENERIC_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_GENERIC_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_GENERIC_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_GENERIC_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_GENERIC_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_GENERIC_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_GENERIC_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_GENERIC_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_GENERIC_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_GENERIC_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_GENERIC_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_GENERIC_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_GENERIC_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_GENERIC_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_GENERIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_generic_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_generic_output_shift;

    ctx.size = arm_depthwise_conv_wrapper_s4_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(ctx.size == 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4(&ctx,
                                                       &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_output_ref, DEPTHWISE_INT4_GENERIC_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_DST_SIZE);

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
                                           &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_output_ref, DEPTHWISE_INT4_GENERIC_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_DST_SIZE);
}

void depthwise_int4_generic_2_arm_depthwise_conv_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_GENERIC_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_generic_2_biases;
    const int8_t *kernel_data = depthwise_int4_generic_2_weights;
    const int8_t *input_data = depthwise_int4_generic_2_input;

    input_dims.n = DEPTHWISE_INT4_GENERIC_2_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_GENERIC_2_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_GENERIC_2_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_GENERIC_2_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_GENERIC_2_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_GENERIC_2_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_GENERIC_2_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_GENERIC_2_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_GENERIC_2_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_GENERIC_2_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_GENERIC_2_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_GENERIC_2_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_GENERIC_2_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_GENERIC_2_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_GENERIC_2_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_GENERIC_2_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_GENERIC_2_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_GENERIC_2_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_GENERIC_2_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_GENERIC_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_generic_2_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_generic_2_output_shift;

    ctx.size = arm_depthwise_conv_wrapper_s4_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(ctx.size == 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4(&ctx,
                                                       &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_2_output_ref, DEPTHWISE_INT4_GENERIC_2_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_2_DST_SIZE);

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
                                           &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_2_output_ref, DEPTHWISE_INT4_GENERIC_2_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_2_DST_SIZE);
}

void depthwise_int4_generic_3_arm_depthwise_conv_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_GENERIC_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_generic_3_biases;
    const int8_t *kernel_data = depthwise_int4_generic_3_weights;
    const int8_t *input_data = depthwise_int4_generic_3_input;

    input_dims.n = DEPTHWISE_INT4_GENERIC_3_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_GENERIC_3_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_GENERIC_3_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_GENERIC_3_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_GENERIC_3_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_GENERIC_3_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_GENERIC_3_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_GENERIC_3_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_GENERIC_3_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_GENERIC_3_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_GENERIC_3_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_GENERIC_3_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_GENERIC_3_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_GENERIC_3_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_GENERIC_3_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_GENERIC_3_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_GENERIC_3_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_GENERIC_3_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_GENERIC_3_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_GENERIC_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_generic_3_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_generic_3_output_shift;

    ctx.size = arm_depthwise_conv_wrapper_s4_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(ctx.size == 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4(&ctx,
                                                       &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_3_output_ref, DEPTHWISE_INT4_GENERIC_3_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_3_DST_SIZE);

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
                                           &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_3_output_ref, DEPTHWISE_INT4_GENERIC_3_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_3_DST_SIZE);
}

void depthwise_int4_generic_4_arm_depthwise_conv_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_GENERIC_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_generic_4_biases;
    const int8_t *kernel_data = depthwise_int4_generic_4_weights;
    const int8_t *input_data = depthwise_int4_generic_4_input;

    input_dims.n = DEPTHWISE_INT4_GENERIC_4_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_GENERIC_4_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_GENERIC_4_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_GENERIC_4_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_GENERIC_4_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_GENERIC_4_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_GENERIC_4_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_GENERIC_4_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_GENERIC_4_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_GENERIC_4_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_GENERIC_4_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_GENERIC_4_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_GENERIC_4_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_GENERIC_4_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_GENERIC_4_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_GENERIC_4_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_GENERIC_4_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_GENERIC_4_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_GENERIC_4_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_GENERIC_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_generic_4_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_generic_4_output_shift;

    ctx.size = arm_depthwise_conv_wrapper_s4_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(ctx.size == 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4(&ctx,
                                                       &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_4_output_ref, DEPTHWISE_INT4_GENERIC_4_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_4_DST_SIZE);

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
                                           &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_4_output_ref, DEPTHWISE_INT4_GENERIC_4_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_4_DST_SIZE);
}

void depthwise_int4_generic_5_arm_depthwise_conv_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_GENERIC_5_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_generic_5_biases;
    const int8_t *kernel_data = depthwise_int4_generic_5_weights;
    const int8_t *input_data = depthwise_int4_generic_5_input;

    input_dims.n = DEPTHWISE_INT4_GENERIC_5_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_GENERIC_5_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_GENERIC_5_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_GENERIC_5_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_GENERIC_5_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_GENERIC_5_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_GENERIC_5_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_GENERIC_5_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_GENERIC_5_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_GENERIC_5_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_GENERIC_5_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_GENERIC_5_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_GENERIC_5_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_GENERIC_5_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_GENERIC_5_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_GENERIC_5_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_GENERIC_5_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_GENERIC_5_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_GENERIC_5_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_GENERIC_5_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_generic_5_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_generic_5_output_shift;

    ctx.size = arm_depthwise_conv_wrapper_s4_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(ctx.size == 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4(&ctx,
                                                       &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_5_output_ref, DEPTHWISE_INT4_GENERIC_5_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_5_DST_SIZE);

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
                                           &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_5_output_ref, DEPTHWISE_INT4_GENERIC_5_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_5_DST_SIZE);
}

void depthwise_int4_generic_6_arm_depthwise_conv_s4(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_GENERIC_6_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_generic_6_biases;
    const int8_t *kernel_data = depthwise_int4_generic_6_weights;
    const int8_t *input_data = depthwise_int4_generic_6_input;

    input_dims.n = DEPTHWISE_INT4_GENERIC_6_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_GENERIC_6_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_GENERIC_6_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_GENERIC_6_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_GENERIC_6_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_GENERIC_6_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_GENERIC_6_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_GENERIC_6_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_GENERIC_6_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_GENERIC_6_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_GENERIC_6_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_GENERIC_6_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_GENERIC_6_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_GENERIC_6_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_GENERIC_6_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_GENERIC_6_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_GENERIC_6_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_GENERIC_6_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_GENERIC_6_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_GENERIC_6_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_generic_6_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_generic_6_output_shift;

    ctx.size = arm_depthwise_conv_wrapper_s4_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(ctx.size == 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4(&ctx,
                                                       &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_6_output_ref, DEPTHWISE_INT4_GENERIC_6_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_6_DST_SIZE);

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
                                           &dw_conv_params,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_generic_6_output_ref, DEPTHWISE_INT4_GENERIC_6_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_GENERIC_6_DST_SIZE);
}

/* Weight t * C_OUT + c of a packed int4 filter, low nibble first. */
static int32_t s4_weight(const int8_t *filter, int32_t index)
{
    const uint8_t byte = (uint8_t)filter[index >> 1];
    const int32_t nibble = (index & 1) ? (byte >> 4) : (byte & 0x0f);
    return (nibble ^ 8) - 8;
}

/* Odd and even channel counts, channel multipliers, 1-D, square and non-square kernels, with and without padding,
   dilation and stride 2, against a scalar reference. With an odd channel count a filter tap can start in the middle of
   a byte. */
void channel_parity_arm_depthwise_conv_s4(void)
{
    enum
    {
        max_in = 4 * 6 * 7,
        max_out = 5 * 6 * 21,
        max_w = 9 * 21
    };
    static int8_t input[max_in], filter[(max_w + 1) / 2], output[max_out], reference[max_out];
    static int32_t bias[21], mult[21], shift[21];
    const int32_t channels[] = {1, 2, 3, 4, 5, 7};
    const int32_t multipliers[] = {1, 2, 3};
    const int32_t shapes[][4] = {{1, 6, 1, 3}, {4, 6, 3, 3}, {4, 6, 2, 3}}; /* input h, w, kernel h, w */
    const cmsis_nn_context ctx = {NULL, 0};
    for (size_t i_s = 0; i_s < sizeof(shapes) / sizeof(shapes[0]); i_s++)
    {
        for (size_t i_c = 0; i_c < sizeof(channels) / sizeof(channels[0]); i_c++)
        {
            for (size_t i_m = 0; i_m < sizeof(multipliers) / sizeof(multipliers[0]); i_m++)
            {
                for (int32_t pad = 0; pad < 2; pad++)
                {
                    for (int32_t dil = 1; dil < 3; dil++)
                    {
                        for (int32_t stride = 1; stride < 3; stride++)
                        {
                            const int32_t ih = shapes[i_s][0], iw = shapes[i_s][1], kh = shapes[i_s][2],
                                          kw = shapes[i_s][3];
                            const int32_t ic = channels[i_c], cm = multipliers[i_m], oc = ic * cm;
                            const int32_t pad_h = kh > 1 ? pad : 0, pad_w = pad * dil;
                            const int32_t oh = (ih + 2 * pad_h - (kh - 1) * dil - 1) / stride + 1,
                                          ow = (iw + 2 * pad_w - (kw - 1) * dil - 1) / stride + 1;
                            if (ih + 2 * pad_h - (kh - 1) * dil < 1 || iw + 2 * pad_w - (kw - 1) * dil < 1)
                            {
                                continue;
                            }
                            TEST_ASSERT_TRUE(oh * ow * oc <= max_out);
                            uint32_t seed = (uint32_t)(ic * 131 + cm * 17 + pad * 7 + dil * 3 + (int32_t)i_s);
                            for (int32_t i = 0; i < ih * iw * ic; i++)
                            {
                                seed = seed * 1664525u + 1013904223u;
                                input[i] = (int8_t)(seed >> 24);
                            }
                            for (int32_t i = 0; i < (kh * kw * oc + 1) / 2; i++)
                            {
                                seed = seed * 1664525u + 1013904223u;
                                filter[i] = (int8_t)(seed >> 24);
                            }
                            for (int32_t i = 0; i < oc; i++)
                            {
                                seed = seed * 1664525u + 1013904223u;
                                bias[i] = (int32_t)(seed >> 20) - 2048;
                                mult[i] = 0x40000000 + (i % 7) * 0x4000000;
                                shift[i] = -5 - (i % 3);
                            }
                            const cmsis_nn_dw_conv_params params = {.input_offset = 3,
                                                                    .output_offset = -2,
                                                                    .ch_mult = cm,
                                                                    .stride = {stride, stride},
                                                                    .padding = {pad_w, pad_h},
                                                                    .dilation = {dil, dil},
                                                                    .activation = {-128, 127}};
                            for (int32_t y = 0; y < oh; y++)
                            {
                                for (int32_t x = 0; x < ow; x++)
                                {
                                    for (int32_t c = 0; c < oc; c++)
                                    {
                                        int32_t acc = bias[c];
                                        for (int32_t ty = 0; ty < kh; ty++)
                                        {
                                            for (int32_t tx = 0; tx < kw; tx++)
                                            {
                                                const int32_t iy = y * stride - pad_h + ty * dil,
                                                              ix = x * stride - pad_w + tx * dil;
                                                if (iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                                                {
                                                    acc += (input[(iy * iw + ix) * ic + c / cm] + params.input_offset) *
                                                        s4_weight(filter, (ty * kw + tx) * oc + c);
                                                }
                                            }
                                        }
                                        int32_t r = arm_nn_requantize(acc, mult[c], shift[c]) + params.output_offset;
                                        r = r < -128 ? -128 : (r > 127 ? 127 : r);
                                        reference[(y * ow + x) * oc + c] = (int8_t)r;
                                    }
                                }
                            }
                            const cmsis_nn_per_channel_quant_params quant = {mult, shift};
                            const cmsis_nn_dims input_dims = {1, ih, iw, ic}, filter_dims = {1, kh, kw, oc},
                                                bias_dims = {1, 1, 1, oc}, output_dims = {1, oh, ow, oc};
                            memset(output, 0x5A, sizeof(output));
                            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                                              arm_depthwise_conv_s4(&ctx,
                                                                    &params,
                                                                    &quant,
                                                                    &input_dims,
                                                                    input,
                                                                    &filter_dims,
                                                                    filter,
                                                                    &bias_dims,
                                                                    bias,
                                                                    &output_dims,
                                                                    output));
                            char message[128];
                            snprintf(message,
                                     sizeof(message),
                                     "ic %ld cm %ld k %ldx%ld pad %ld dil %ld stride %ld",
                                     (long)ic,
                                     (long)cm,
                                     (long)kh,
                                     (long)kw,
                                     (long)pad,
                                     (long)dil,
                                     (long)stride);
                            TEST_ASSERT_EQUAL_INT8_ARRAY_MESSAGE(reference, output, oh * ow * oc, message);
                        }
                    }
                }
            }
        }
    }
}
