/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include <unity.h>

#include "../TestData/depthwise_2/test_data.h"
#include "../TestData/depthwise_dilation/test_data.h"
#include "../TestData/depthwise_mult_batches/test_data.h"
#include "../TestData/depthwise_null_bias_1/test_data.h"
#include "../TestData/depthwise_single_in_many_out_ch/test_data.h"
#include "../TestData/in_ch_one_out_ch_larger_one/test_data.h"
#include "../Utils/utils.h"
#include "../Utils/validate.h"

void depthwise_2_arm_depthwise_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_2_DST_SIZE;
    const int32_t *bias_data = get_bias_address(depthwise_2_biases, DEPTHWISE_2_OUT_CH);
    const int8_t *kernel_data = depthwise_2_weights;
    const int8_t *input_data = depthwise_2_input;

    input_dims.n = DEPTHWISE_2_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_2_INPUT_W;
    input_dims.h = DEPTHWISE_2_INPUT_H;
    input_dims.c = DEPTHWISE_2_IN_CH;
    filter_dims.w = DEPTHWISE_2_FILTER_X;
    filter_dims.h = DEPTHWISE_2_FILTER_Y;
    output_dims.w = DEPTHWISE_2_OUTPUT_W;
    output_dims.h = DEPTHWISE_2_OUTPUT_H;
    output_dims.c = DEPTHWISE_2_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_2_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_2_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_2_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_2_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_2_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_2_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_2_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_2_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_2_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_2_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_2_output_mult;
    quant_params.shift = (int32_t *)depthwise_2_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8(&ctx,
                                                       &weights_sum_ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_2_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_EQUAL(buf_size, 0);

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;

    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);
    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
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

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_2_output_ref, output_ref_size));
}

void depthwise_mult_batches_arm_depthwise_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_MULT_BATCHES_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_MULT_BATCHES_DST_SIZE;
    const int32_t *bias_data = get_bias_address(depthwise_mult_batches_biases, DEPTHWISE_MULT_BATCHES_OUT_CH);
    const int8_t *kernel_data = depthwise_mult_batches_weights;
    const int8_t *input_data = depthwise_mult_batches_input;

    input_dims.n = DEPTHWISE_MULT_BATCHES_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_MULT_BATCHES_INPUT_W;
    input_dims.h = DEPTHWISE_MULT_BATCHES_INPUT_H;
    input_dims.c = DEPTHWISE_MULT_BATCHES_IN_CH;
    filter_dims.w = DEPTHWISE_MULT_BATCHES_FILTER_X;
    filter_dims.h = DEPTHWISE_MULT_BATCHES_FILTER_Y;
    output_dims.w = DEPTHWISE_MULT_BATCHES_OUTPUT_W;
    output_dims.h = DEPTHWISE_MULT_BATCHES_OUTPUT_H;
    output_dims.c = DEPTHWISE_MULT_BATCHES_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_MULT_BATCHES_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_MULT_BATCHES_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_MULT_BATCHES_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_MULT_BATCHES_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_MULT_BATCHES_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_MULT_BATCHES_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_MULT_BATCHES_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_MULT_BATCHES_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_MULT_BATCHES_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_MULT_BATCHES_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_MULT_BATCHES_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_mult_batches_output_mult;
    quant_params.shift = (int32_t *)depthwise_mult_batches_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8(&ctx,
                                                       &weights_sum_ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_mult_batches_output_ref, output_ref_size));

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(buf_size, 0);

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);
    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
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

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_mult_batches_output_ref, output_ref_size));
}

void depthwise_null_bias_1_arm_depthwise_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_NULL_BIAS_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(depthwise_null_bias_1_biases, DEPTHWISE_NULL_BIAS_1_OUT_CH);
    const int8_t *kernel_data = depthwise_null_bias_1_weights;
    const int8_t *input_data = depthwise_null_bias_1_input;

    input_dims.n = DEPTHWISE_NULL_BIAS_1_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_NULL_BIAS_1_INPUT_W;
    input_dims.h = DEPTHWISE_NULL_BIAS_1_INPUT_H;
    input_dims.c = DEPTHWISE_NULL_BIAS_1_IN_CH;
    filter_dims.w = DEPTHWISE_NULL_BIAS_1_FILTER_X;
    filter_dims.h = DEPTHWISE_NULL_BIAS_1_FILTER_Y;
    output_dims.w = DEPTHWISE_NULL_BIAS_1_OUTPUT_W;
    output_dims.h = DEPTHWISE_NULL_BIAS_1_OUTPUT_H;
    output_dims.c = DEPTHWISE_NULL_BIAS_1_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_NULL_BIAS_1_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_NULL_BIAS_1_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_NULL_BIAS_1_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_NULL_BIAS_1_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_NULL_BIAS_1_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_NULL_BIAS_1_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_NULL_BIAS_1_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_NULL_BIAS_1_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_NULL_BIAS_1_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_NULL_BIAS_1_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_NULL_BIAS_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_null_bias_1_output_mult;
    quant_params.shift = (int32_t *)depthwise_null_bias_1_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8(&ctx,
                                                       &weights_sum_ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_null_bias_1_output_ref, DEPTHWISE_NULL_BIAS_1_DST_SIZE));

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(buf_size, 0);

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);
    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
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

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_null_bias_1_output_ref, DEPTHWISE_NULL_BIAS_1_DST_SIZE));
}

void depthwise_dilation_arm_depthwise_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_DILATION_DST_SIZE;
    const int32_t *bias_data = get_bias_address(depthwise_dilation_biases, DEPTHWISE_DILATION_OUT_CH);
    const int8_t *kernel_data = depthwise_dilation_weights;
    const int8_t *input_data = depthwise_dilation_input;

    input_dims.n = DEPTHWISE_DILATION_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_DILATION_INPUT_W;
    input_dims.h = DEPTHWISE_DILATION_INPUT_H;
    input_dims.c = DEPTHWISE_DILATION_IN_CH;
    filter_dims.w = DEPTHWISE_DILATION_FILTER_X;
    filter_dims.h = DEPTHWISE_DILATION_FILTER_Y;
    output_dims.w = DEPTHWISE_DILATION_OUTPUT_W;
    output_dims.h = DEPTHWISE_DILATION_OUTPUT_H;
    output_dims.c = DEPTHWISE_DILATION_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_DILATION_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_DILATION_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_DILATION_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_DILATION_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_DILATION_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_DILATION_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_DILATION_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_DILATION_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_DILATION_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_DILATION_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_dilation_output_mult;
    quant_params.shift = (int32_t *)depthwise_dilation_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;
    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8(&ctx,
                                                       &weights_sum_ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_dilation_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_EQUAL(buf_size, 0);

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;

    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);
    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
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

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_dilation_output_ref, output_ref_size));
}

void buffer_size_mve_arm_depthwise_conv_s8(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DEPTHWISE_DILATION_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_DILATION_INPUT_W;
    input_dims.h = DEPTHWISE_DILATION_INPUT_H;
    input_dims.c = DEPTHWISE_DILATION_IN_CH;
    filter_dims.w = DEPTHWISE_DILATION_FILTER_X;
    filter_dims.h = DEPTHWISE_DILATION_FILTER_Y;
    output_dims.w = DEPTHWISE_DILATION_OUTPUT_W;
    output_dims.h = DEPTHWISE_DILATION_OUTPUT_H;
    output_dims.c = DEPTHWISE_DILATION_OUT_CH;

    conv_params.padding.w = DEPTHWISE_DILATION_PAD_X;
    conv_params.padding.h = DEPTHWISE_DILATION_PAD_Y;
    conv_params.stride.w = DEPTHWISE_DILATION_STRIDE_X;
    conv_params.stride.h = DEPTHWISE_DILATION_STRIDE_Y;
    conv_params.dilation.w = DEPTHWISE_DILATION_DILATION_X;
    conv_params.dilation.h = DEPTHWISE_DILATION_DILATION_Y;
    conv_params.ch_mult = DEPTHWISE_DILATION_CH_MULT;
    conv_params.input_offset = DEPTHWISE_DILATION_INPUT_OFFSET;
    conv_params.output_offset = DEPTHWISE_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = DEPTHWISE_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DEPTHWISE_DILATION_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_depthwise_conv_s8(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DEPTHWISE_DILATION_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_DILATION_INPUT_W;
    input_dims.h = DEPTHWISE_DILATION_INPUT_H;
    input_dims.c = DEPTHWISE_DILATION_IN_CH;
    filter_dims.w = DEPTHWISE_DILATION_FILTER_X;
    filter_dims.h = DEPTHWISE_DILATION_FILTER_Y;
    output_dims.w = DEPTHWISE_DILATION_OUTPUT_W;
    output_dims.h = DEPTHWISE_DILATION_OUTPUT_H;
    output_dims.c = DEPTHWISE_DILATION_OUT_CH;

    conv_params.padding.w = DEPTHWISE_DILATION_PAD_X;
    conv_params.padding.h = DEPTHWISE_DILATION_PAD_Y;
    conv_params.stride.w = DEPTHWISE_DILATION_STRIDE_X;
    conv_params.stride.h = DEPTHWISE_DILATION_STRIDE_Y;
    conv_params.dilation.w = DEPTHWISE_DILATION_DILATION_X;
    conv_params.dilation.h = DEPTHWISE_DILATION_DILATION_Y;

    conv_params.ch_mult = DEPTHWISE_DILATION_CH_MULT;

    conv_params.input_offset = DEPTHWISE_DILATION_INPUT_OFFSET;
    conv_params.output_offset = DEPTHWISE_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = DEPTHWISE_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DEPTHWISE_DILATION_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}

void in_ch_one_out_ch_larger_one_arm_depthwise_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[IN_CH_ONE_OUT_CH_LARGER_ONE_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;
    const int32_t output_ref_size = IN_CH_ONE_OUT_CH_LARGER_ONE_DST_SIZE;
    const int32_t *bias_data = get_bias_address(in_ch_one_out_ch_larger_one_biases, IN_CH_ONE_OUT_CH_LARGER_ONE_OUT_CH);
    const int8_t *kernel_data = in_ch_one_out_ch_larger_one_weights;
    const int8_t *input_data = in_ch_one_out_ch_larger_one_input;
    input_dims.n = IN_CH_ONE_OUT_CH_LARGER_ONE_INPUT_BATCHES;
    input_dims.w = IN_CH_ONE_OUT_CH_LARGER_ONE_INPUT_W;
    input_dims.h = IN_CH_ONE_OUT_CH_LARGER_ONE_INPUT_H;
    input_dims.c = IN_CH_ONE_OUT_CH_LARGER_ONE_IN_CH;
    filter_dims.n = IN_CH_ONE_OUT_CH_LARGER_ONE_IN_CH;
    filter_dims.w = IN_CH_ONE_OUT_CH_LARGER_ONE_FILTER_X;
    filter_dims.h = IN_CH_ONE_OUT_CH_LARGER_ONE_FILTER_Y;
    filter_dims.c = IN_CH_ONE_OUT_CH_LARGER_ONE_OUT_CH;
    output_dims.w = IN_CH_ONE_OUT_CH_LARGER_ONE_OUTPUT_W;
    output_dims.h = IN_CH_ONE_OUT_CH_LARGER_ONE_OUTPUT_H;
    output_dims.c = IN_CH_ONE_OUT_CH_LARGER_ONE_OUT_CH;
    dw_conv_params.padding.w = IN_CH_ONE_OUT_CH_LARGER_ONE_PAD_X;
    dw_conv_params.padding.h = IN_CH_ONE_OUT_CH_LARGER_ONE_PAD_Y;
    dw_conv_params.stride.w = IN_CH_ONE_OUT_CH_LARGER_ONE_STRIDE_X;
    dw_conv_params.stride.h = IN_CH_ONE_OUT_CH_LARGER_ONE_STRIDE_Y;
    dw_conv_params.dilation.w = IN_CH_ONE_OUT_CH_LARGER_ONE_DILATION_X;
    dw_conv_params.dilation.h = IN_CH_ONE_OUT_CH_LARGER_ONE_DILATION_Y;
    dw_conv_params.ch_mult = IN_CH_ONE_OUT_CH_LARGER_ONE_CH_MULT;
    dw_conv_params.input_offset = IN_CH_ONE_OUT_CH_LARGER_ONE_INPUT_OFFSET;
    dw_conv_params.output_offset = IN_CH_ONE_OUT_CH_LARGER_ONE_OUTPUT_OFFSET;
    dw_conv_params.activation.min = IN_CH_ONE_OUT_CH_LARGER_ONE_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = IN_CH_ONE_OUT_CH_LARGER_ONE_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)in_ch_one_out_ch_larger_one_output_mult;
    quant_params.shift = (int32_t *)in_ch_one_out_ch_larger_one_output_shift;
    ctx.buf = NULL;
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8(&ctx,
                                                       &weights_sum_ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, in_ch_one_out_ch_larger_one_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);
    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
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

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, in_ch_one_out_ch_larger_one_output_ref, output_ref_size));
}

/* ------------------------------------------------------------------------ */
/*  Weight–presum sanity-check (no convolution)                             */
/* ------------------------------------------------------------------------ */
void weight_presum(void)
{
    /* 2-out-ch × 3-in-ch, 1 × 1 kernel
       kernel = {1,2,3 | 4,5,6}
       bias    = {10,20}
       lhs_off = 5  → expected sums = {10+5*(1+2+3)=40 , 20+5*(4+5+6)=95} */
    const int8_t  kernel[6]   = {1, 2, 3, 4, 5, 6};
    const int32_t bias[3]     = {10, 20, 11};
    const uint32_t lhs_offset = 1;
    static const cmsis_nn_dw_conv_params dw_conv_params = {
        .padding        = { .w = 0, .h = 0 },
        .stride         = { .w = 1, .h = 1 },
        .dilation       = { .w = 1, .h = 1 },
        .ch_mult        = 1,                 /* depth-multiplier             */
        .input_offset   = 0,
        .output_offset  = 0,
        .activation     = { .min = -128, .max = 127 },
    };

    cmsis_nn_dims in_dims    = { .n = 1, .w = 1, .h = 2, .c = 3 };
    cmsis_nn_dims filt_dims  = { .n = 1, .w = 1, .h = 2, .c = 3 };
    cmsis_nn_dims out_dims   = { .n = 1, .w = 1, .h = 2, .c = 3 };

    const int32_t buf_sz = arm_depthwise_conv_wrapper_s8_get_buffer_size(
          &dw_conv_params, &in_dims, &filt_dims, &out_dims);
    const int32_t scratch_size = arm_convolve_s8_get_weights_sum_size(&out_dims);
    int32_t *sum_buf     = (int32_t *)malloc(buf_sz);
    int8_t *scratch_buf = (int8_t *)malloc(scratch_size);



    arm_cmsis_nn_status result = arm_depthwise_convolve_weight_sum(sum_buf,
                            scratch_buf,
                            kernel,
                            &dw_conv_params,
                            &in_dims,
                            &filt_dims,
                            &out_dims,
                            lhs_offset,
                            bias);
    free(scratch_buf);


    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    const int32_t expected[3] = {15,27,20};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, sum_buf, 3);
    memset(sum_buf, 0, buf_sz);
    free(sum_buf);

}

/* ------------------------------------------------------------------------ */
/*  Tiny depth-wise conv, 2 × 1 kernel, no bias                             */
/* ------------------------------------------------------------------------ */
void simple_dconv_no_bias(void)
{
    const int8_t input[4]   = {1,1,2,2};               
    const int8_t kernel[4]  = {1,3,2,4};              
    int8_t       output[2]  = {0};
    const int8_t ref[2]     = {8, 18};     

    cmsis_nn_dims in_dims  = { .n = 1, .w = 2, .h = 1, .c = 2 };
    cmsis_nn_dims filt_dims= { .n = 1, .w = 2, .h = 1, .c = 2 }; /* ch_mult=1 */
    cmsis_nn_dims bias_dims= {};
    cmsis_nn_dims out_dims = { .n = 1, .w = 1, .h = 1, .c = 2 };

    cmsis_nn_dw_conv_params dw_conv_params = {
        .padding  = { .w = 0, .h = 0 },
        .stride   = { .w = 1, .h = 1 },
        .dilation = { .w = 1, .h = 1 },
        .ch_mult  = 1,
        .input_offset  = 1,
        .output_offset = 0,
        .activation = { .min = -128, .max = 127 },
    };

    int32_t mult[2] = {~(1 << 31), ~(1<<31)};   /* Q31 ≈ 1.0 */
    int32_t shift[2]= {0,0};
    cmsis_nn_per_channel_quant_params q = {
        .multiplier = mult,
        .shift      = shift
    };

    int32_t* bias = 0;

    const int32_t buf_sz = arm_depthwise_conv_wrapper_s8_get_buffer_size(
          &dw_conv_params, &in_dims, &filt_dims, &out_dims);
    const int32_t scratch_size = arm_convolve_s8_get_weights_sum_size(&out_dims);
    int32_t *sum_buf     = (int32_t *)malloc(scratch_size);
    int8_t *scratch_buf = (int8_t *)malloc(buf_sz);

    uint32_t lhs_offset = dw_conv_params.input_offset;

    cmsis_nn_context weights_sum_ctx, ctx;
    weights_sum_ctx.buf = sum_buf;
    weights_sum_ctx.size = buf_sz;

    ctx.buf = scratch_buf;
    ctx.size = 0;

    arm_depthwise_convolve_weight_sum(sum_buf,
                            scratch_buf,
                            kernel,
                            &dw_conv_params,
                            &in_dims,
                            &filt_dims,
                            &out_dims,
                            lhs_offset,
                            bias);

    arm_cmsis_nn_status res = arm_depthwise_conv_wrapper_s8(&ctx,
                                       &weights_sum_ctx,
                                       &dw_conv_params,
                                       &q,
                                       &in_dims,
                                       input,
                                       &filt_dims,
                                       kernel,
                                       &bias_dims,
                                       bias,
                                       &out_dims,
                                       output);
    free(scratch_buf);

    free(sum_buf);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, res);
    TEST_ASSERT_EQUAL_INT8_ARRAY(ref, output, 2);
}

void single_in_many_out_ch(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_SINGLE_IN_MANY_OUT_CH_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;
    const int32_t output_ref_size = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_DST_SIZE;
    const int32_t *bias_data = get_bias_address(depthwise_single_in_many_out_ch_biases, DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUT_CH);
    
    const int8_t *input_data = depthwise_single_in_many_out_ch_input_tensor;

    //need to transpose weights as a hack because generated from conv kernel
    const int8_t *conv_kernel_data = depthwise_single_in_many_out_ch_weights;
    int8_t* kernel_data = malloc(DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUT_CH * DEPTHWISE_SINGLE_IN_MANY_OUT_CH_FILTER_X * DEPTHWISE_SINGLE_IN_MANY_OUT_CH_FILTER_Y);
    //transpose from conv to dconv format

    input_dims.n = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_INPUT_W;
    input_dims.h = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_INPUT_H;
    input_dims.c = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_IN_CH;

    filter_dims.n = 1;
    filter_dims.w = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_FILTER_X;
    filter_dims.h = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_FILTER_Y;
    filter_dims.c = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUT_CH;

    output_dims.n = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_INPUT_BATCHES;
    output_dims.w = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUTPUT_W;
    output_dims.h = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUTPUT_H;
    output_dims.c = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUT_CH;
    dw_conv_params.padding.w = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_DILATION_Y;
    //because we generated a convolution for this test, ch_mult is not generated
    dw_conv_params.ch_mult = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUT_CH;

    dw_conv_params.input_offset = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_single_in_many_out_ch_output_mult;
    quant_params.shift = (int32_t *)depthwise_single_in_many_out_ch_output_shift;

    cmsis_nn_dims filter_input_dims;
    filter_input_dims.n = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_OUT_CH;
    filter_input_dims.w = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_FILTER_X;
    filter_input_dims.h = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_FILTER_Y;
    filter_input_dims.c = DEPTHWISE_SINGLE_IN_MANY_OUT_CH_IN_CH;

    const uint32_t perm[4] = {3, 1, 2, 0};
    const cmsis_nn_transpose_params transpose_params = {4, perm};
    arm_cmsis_nn_status status = arm_transpose_s8(conv_kernel_data, kernel_data, &filter_input_dims, &filter_dims, &transpose_params);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);

    ctx.buf = NULL;
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.size = weights_sum_buf_size;

    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    uint32_t lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8(&ctx,
                                                       &weights_sum_ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_single_in_many_out_ch_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    result = arm_depthwise_convolve_weight_sum(weights_sum_ctx.buf,
                            ctx.buf,
                            kernel_data,
                            &dw_conv_params,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            lhs_offset,
                            bias_data);
    result = arm_depthwise_conv_wrapper_s8(&ctx,
                                           &weights_sum_ctx,
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

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_single_in_many_out_ch_output_ref, output_ref_size));
}
