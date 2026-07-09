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

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <unity.h>

#include "../TestData/basic/test_data.h"
#include "../TestData/conv_2/test_data.h"
#include "../TestData/conv_2x2_dilation/test_data.h"
#include "../TestData/conv_2x2_dilation_5x5_input/test_data.h"
#include "../TestData/conv_2x3_dilation/test_data.h"
#include "../TestData/conv_3/test_data.h"
#include "../TestData/conv_3x2_dilation/test_data.h"
#include "../TestData/conv_3x3_dilation_5x5_input/test_data.h"
#include "../TestData/conv_4/test_data.h"
#include "../TestData/conv_5/test_data.h"
#include "../TestData/conv_dilation_golden/test_data.h"
#include "../TestData/conv_out_activation/test_data.h"
#include "../TestData/stride2pad1/test_data.h"
#include "../Utils/validate.h"
#include "../TestData/fc_conv_int8_dilated/test_data.h"
#include "../TestData/fc_conv_int8_diff_channels/test_data.h"
#include "../TestData/fc_conv_int8_non_4_multiple/test_data.h"
#include "../TestData/fc_conv_int8_1x1_kernel/test_data.h"
//#include "../TestData/fc_conv_int8_dilated/input_weights.h"


static arm_cmsis_nn_status conv_1x1_out_wrapper(cmsis_nn_context *ctx,
        cmsis_nn_conv_params *conv_params,
        cmsis_nn_per_channel_quant_params *quant_params,
        cmsis_nn_dims *input_dims,
        cmsis_nn_dims *filter_dims,
        cmsis_nn_dims *bias_dims,
        cmsis_nn_dims *output_dims,
        const int32_t *bias_data,
        const int8_t *kernel_data,
        const int8_t *input_data,
        int8_t *output
    )
{
    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params->input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,input_dims, filter_dims, output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result;
    result = arm_convolve_wrapper_s8(ctx,
                                     &weights_sum_ctx,
                                     conv_params,
                                     quant_params,
                                     input_dims,
                                     input_data,
                                     filter_dims,
                                     kernel_data,
                                     bias_dims,
                                     bias_data,
                                     output_dims,
                                     output);

    if (weights_sum_ctx.buf)
    {
        memset(weights_sum_ctx.buf, 0, weights_sum_ctx.size);
        free(weights_sum_ctx.buf);
    }
    return result;
}





void basic_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[BASIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_biases;
    const int8_t *kernel_data = basic_weights;
    const int8_t *input_data = basic_input;
    const int8_t *output_ref = basic_output_ref;
    const int32_t output_ref_size = BASIC_DST_SIZE;

    input_dims.n = BASIC_INPUT_BATCHES;
    input_dims.w = BASIC_INPUT_W;
    input_dims.h = BASIC_INPUT_H;
    input_dims.c = BASIC_IN_CH;

    filter_dims.w = BASIC_FILTER_X;
    filter_dims.h = BASIC_FILTER_Y;
    filter_dims.c = BASIC_IN_CH;

    output_dims.w = BASIC_OUTPUT_W;
    output_dims.h = BASIC_OUTPUT_H;
    output_dims.c = BASIC_OUT_CH;

    conv_params.padding.w = BASIC_PAD_X;
    conv_params.padding.h = BASIC_PAD_Y;
    conv_params.stride.w = BASIC_STRIDE_X;
    conv_params.stride.h = BASIC_STRIDE_Y;
    conv_params.dilation.w = BASIC_DILATION_X;
    conv_params.dilation.h = BASIC_DILATION_Y;

    conv_params.input_offset = BASIC_INPUT_OFFSET;
    conv_params.output_offset = BASIC_OUTPUT_OFFSET;
    conv_params.activation.min = BASIC_OUT_ACTIVATION_MIN;
    conv_params.activation.max = BASIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_output_mult;
    quant_params.shift = (int32_t *)basic_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);
    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void stride2pad1_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[STRIDE2PAD1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = stride2pad1_biases;
    const int8_t *kernel_data = stride2pad1_weights;
    const int8_t *input_data = stride2pad1_input;
    const int8_t *output_ref = stride2pad1_output_ref;
    const int32_t output_ref_size = STRIDE2PAD1_DST_SIZE;

    input_dims.n = STRIDE2PAD1_INPUT_BATCHES;
    input_dims.w = STRIDE2PAD1_INPUT_W;
    input_dims.h = STRIDE2PAD1_INPUT_H;
    input_dims.c = STRIDE2PAD1_IN_CH;
    filter_dims.w = STRIDE2PAD1_FILTER_X;
    filter_dims.h = STRIDE2PAD1_FILTER_Y;
    filter_dims.c = STRIDE2PAD1_IN_CH;
    output_dims.w = STRIDE2PAD1_OUTPUT_W;
    output_dims.h = STRIDE2PAD1_OUTPUT_H;
    output_dims.c = STRIDE2PAD1_OUT_CH;

    conv_params.padding.w = STRIDE2PAD1_PAD_X;
    conv_params.padding.h = STRIDE2PAD1_PAD_Y;
    conv_params.stride.w = STRIDE2PAD1_STRIDE_X;
    conv_params.stride.h = STRIDE2PAD1_STRIDE_Y;
    conv_params.dilation.w = STRIDE2PAD1_DILATION_X;
    conv_params.dilation.h = STRIDE2PAD1_DILATION_Y;

    conv_params.input_offset = STRIDE2PAD1_INPUT_OFFSET;
    conv_params.output_offset = STRIDE2PAD1_OUTPUT_OFFSET;
    conv_params.activation.min = STRIDE2PAD1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = STRIDE2PAD1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)stride2pad1_output_mult;
    quant_params.shift = (int32_t *)stride2pad1_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2_biases;
    const int8_t *kernel_data = conv_2_weights;
    const int8_t *input_data = conv_2_input;
    const int8_t *output_ref = conv_2_output_ref;
    const int32_t output_ref_size = CONV_2_DST_SIZE;

    input_dims.n = CONV_2_INPUT_BATCHES;
    input_dims.w = CONV_2_INPUT_W;
    input_dims.h = CONV_2_INPUT_H;
    input_dims.c = CONV_2_IN_CH;
    filter_dims.w = CONV_2_FILTER_X;
    filter_dims.h = CONV_2_FILTER_Y;
    filter_dims.c = CONV_2_IN_CH;
    output_dims.w = CONV_2_OUTPUT_W;
    output_dims.h = CONV_2_OUTPUT_H;
    output_dims.c = CONV_2_OUT_CH;

    conv_params.padding.w = CONV_2_PAD_X;
    conv_params.padding.h = CONV_2_PAD_Y;
    conv_params.stride.w = CONV_2_STRIDE_X;
    conv_params.stride.h = CONV_2_STRIDE_Y;
    conv_params.dilation.w = CONV_2_DILATION_X;
    conv_params.dilation.h = CONV_2_DILATION_Y;

    conv_params.input_offset = CONV_2_INPUT_OFFSET;
    conv_params.output_offset = CONV_2_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2_output_mult;
    quant_params.shift = (int32_t *)conv_2_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_2_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;


    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_3_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3_biases;
    const int8_t *kernel_data = conv_3_weights;
    const int8_t *input_data = conv_3_input;
    const int8_t *output_ref = conv_3_output_ref;
    const int32_t output_ref_size = CONV_3_DST_SIZE;

    input_dims.n = CONV_3_INPUT_BATCHES;
    input_dims.w = CONV_3_INPUT_W;
    input_dims.h = CONV_3_INPUT_H;
    input_dims.c = CONV_3_IN_CH;
    filter_dims.w = CONV_3_FILTER_X;
    filter_dims.h = CONV_3_FILTER_Y;
    filter_dims.c = CONV_3_IN_CH;
    output_dims.w = CONV_3_OUTPUT_W;
    output_dims.h = CONV_3_OUTPUT_H;
    output_dims.c = CONV_3_OUT_CH;

    conv_params.padding.w = CONV_3_PAD_X;
    conv_params.padding.h = CONV_3_PAD_Y;
    conv_params.stride.w = CONV_3_STRIDE_X;
    conv_params.stride.h = CONV_3_STRIDE_Y;
    conv_params.dilation.w = CONV_3_DILATION_X;
    conv_params.dilation.h = CONV_3_DILATION_Y;

    conv_params.input_offset = CONV_3_INPUT_OFFSET;
    conv_params.output_offset = CONV_3_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3_output_mult;
    quant_params.shift = (int32_t *)conv_3_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_3_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_4_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_4_biases;
    const int8_t *kernel_data = conv_4_weights;
    const int8_t *input_data = conv_4_input;
    const int8_t *output_ref = conv_4_output_ref;
    const int32_t output_ref_size = CONV_4_DST_SIZE;

    input_dims.n = CONV_4_INPUT_BATCHES;
    input_dims.w = CONV_4_INPUT_W;
    input_dims.h = CONV_4_INPUT_H;
    input_dims.c = CONV_4_IN_CH;
    filter_dims.w = CONV_4_FILTER_X;
    filter_dims.h = CONV_4_FILTER_Y;
    filter_dims.c = CONV_4_IN_CH;
    output_dims.w = CONV_4_OUTPUT_W;
    output_dims.h = CONV_4_OUTPUT_H;
    output_dims.c = CONV_4_OUT_CH;

    conv_params.padding.w = CONV_4_PAD_X;
    conv_params.padding.h = CONV_4_PAD_Y;
    conv_params.stride.w = CONV_4_STRIDE_X;
    conv_params.stride.h = CONV_4_STRIDE_Y;
    conv_params.dilation.w = CONV_4_DILATION_X;
    conv_params.dilation.h = CONV_4_DILATION_Y;

    conv_params.input_offset = CONV_4_INPUT_OFFSET;
    conv_params.output_offset = CONV_4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_4_output_mult;
    quant_params.shift = (int32_t *)conv_4_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_4_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_out_activation_arm_convolve_s8(void)
{
    int8_t output[CONV_OUT_ACTIVATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_out_activation_biases;
    const int8_t *kernel_data = conv_out_activation_weights;
    const int8_t *input_data = conv_out_activation_input;
    const int8_t *output_ref = conv_out_activation_output_ref;
    const int32_t output_ref_size = CONV_OUT_ACTIVATION_DST_SIZE;

    input_dims.n = CONV_OUT_ACTIVATION_INPUT_BATCHES;
    input_dims.w = CONV_OUT_ACTIVATION_INPUT_W;
    input_dims.h = CONV_OUT_ACTIVATION_INPUT_H;
    input_dims.c = CONV_OUT_ACTIVATION_IN_CH;
    filter_dims.n = CONV_OUT_ACTIVATION_OUT_CH;
    filter_dims.w = CONV_OUT_ACTIVATION_FILTER_X;
    filter_dims.h = CONV_OUT_ACTIVATION_FILTER_Y;
    filter_dims.c = CONV_OUT_ACTIVATION_IN_CH;
    output_dims.w = CONV_OUT_ACTIVATION_OUTPUT_W;
    output_dims.h = CONV_OUT_ACTIVATION_OUTPUT_H;
    output_dims.c = CONV_OUT_ACTIVATION_OUT_CH;

    conv_params.padding.w = CONV_OUT_ACTIVATION_PAD_X;
    conv_params.padding.h = CONV_OUT_ACTIVATION_PAD_Y;
    conv_params.stride.w = CONV_OUT_ACTIVATION_STRIDE_X;
    conv_params.stride.h = CONV_OUT_ACTIVATION_STRIDE_Y;
    conv_params.dilation.w = CONV_OUT_ACTIVATION_DILATION_X;
    conv_params.dilation.h = CONV_OUT_ACTIVATION_DILATION_Y;

    conv_params.input_offset = CONV_OUT_ACTIVATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_OUT_ACTIVATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_OUT_ACTIVATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_OUT_ACTIVATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_out_activation_output_mult;
    quant_params.shift = (int32_t *)conv_out_activation_output_shift;

    int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2x2_dilation_arm_convolve_s8(void)
{
    int8_t output[CONV_2X2_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int32_t *bias_data = conv_2x2_dilation_biases;
    const int8_t *kernel_data = conv_2x2_dilation_weights;
    const int8_t *input_data = conv_2x2_dilation_input;
    const int8_t *output_ref = conv_2x2_dilation_output_ref;
    const int32_t output_ref_size = CONV_2X2_DILATION_DST_SIZE;

    input_dims.n = CONV_2X2_DILATION_INPUT_BATCHES;
    input_dims.w = CONV_2X2_DILATION_INPUT_W;
    input_dims.h = CONV_2X2_DILATION_INPUT_H;
    input_dims.c = CONV_2X2_DILATION_IN_CH;
    filter_dims.w = CONV_2X2_DILATION_FILTER_X;
    filter_dims.h = CONV_2X2_DILATION_FILTER_Y;
    filter_dims.c = CONV_2X2_DILATION_IN_CH;
    output_dims.w = CONV_2X2_DILATION_OUTPUT_W;
    output_dims.h = CONV_2X2_DILATION_OUTPUT_H;
    output_dims.c = CONV_2X2_DILATION_OUT_CH;

    conv_params.padding.w = CONV_2X2_DILATION_PAD_X;
    conv_params.padding.h = CONV_2X2_DILATION_PAD_Y;
    conv_params.stride.w = CONV_2X2_DILATION_STRIDE_X;
    conv_params.stride.h = CONV_2X2_DILATION_STRIDE_Y;
    conv_params.dilation.w = CONV_2X2_DILATION_DILATION_X;
    conv_params.dilation.h = CONV_2X2_DILATION_DILATION_Y;

    conv_params.input_offset = CONV_2X2_DILATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X2_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X2_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X2_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x2_dilation_output_mult;
    quant_params.shift = (int32_t *)conv_2x2_dilation_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2x2_dilation_5x5_input_arm_convolve_s8(void)
{
    int8_t output[CONV_2X2_DILATION_5X5_INPUT_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2x2_dilation_5x5_input_biases;
    const int8_t *kernel_data = conv_2x2_dilation_5x5_input_weights;
    const int8_t *input_data = conv_2x2_dilation_5x5_input_input;
    const int8_t *output_ref = conv_2x2_dilation_5x5_input_output_ref;
    const int32_t output_ref_size = CONV_2X2_DILATION_5X5_INPUT_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_2X2_DILATION_5X5_INPUT_INPUT_BATCHES;
    input_dims.w = CONV_2X2_DILATION_5X5_INPUT_INPUT_W;
    input_dims.h = CONV_2X2_DILATION_5X5_INPUT_INPUT_H;
    input_dims.c = CONV_2X2_DILATION_5X5_INPUT_IN_CH;
    filter_dims.w = CONV_2X2_DILATION_5X5_INPUT_FILTER_X;
    filter_dims.h = CONV_2X2_DILATION_5X5_INPUT_FILTER_Y;
    filter_dims.c = CONV_2X2_DILATION_5X5_INPUT_IN_CH;
    output_dims.w = CONV_2X2_DILATION_5X5_INPUT_OUTPUT_W;
    output_dims.h = CONV_2X2_DILATION_5X5_INPUT_OUTPUT_H;
    output_dims.c = CONV_2X2_DILATION_5X5_INPUT_OUT_CH;

    conv_params.padding.w = CONV_2X2_DILATION_5X5_INPUT_PAD_X;
    conv_params.padding.h = CONV_2X2_DILATION_5X5_INPUT_PAD_Y;
    conv_params.stride.w = CONV_2X2_DILATION_5X5_INPUT_STRIDE_X;
    conv_params.stride.h = CONV_2X2_DILATION_5X5_INPUT_STRIDE_Y;
    conv_params.dilation.w = CONV_2X2_DILATION_5X5_INPUT_DILATION_X;
    conv_params.dilation.h = CONV_2X2_DILATION_5X5_INPUT_DILATION_Y;

    conv_params.input_offset = CONV_2X2_DILATION_5X5_INPUT_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X2_DILATION_5X5_INPUT_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X2_DILATION_5X5_INPUT_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X2_DILATION_5X5_INPUT_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x2_dilation_5x5_input_output_mult;
    quant_params.shift = (int32_t *)conv_2x2_dilation_5x5_input_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);
    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_3x3_dilation_5x5_input_arm_convolve_s8(void)
{
    int8_t output[CONV_3X3_DILATION_5X5_INPUT_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3x3_dilation_5x5_input_biases;
    const int8_t *kernel_data = conv_3x3_dilation_5x5_input_weights;
    const int8_t *input_data = conv_3x3_dilation_5x5_input_input;
    const int8_t *output_ref = conv_3x3_dilation_5x5_input_output_ref;
    const int32_t output_ref_size = CONV_3X3_DILATION_5X5_INPUT_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_3X3_DILATION_5X5_INPUT_INPUT_BATCHES;
    input_dims.w = CONV_3X3_DILATION_5X5_INPUT_INPUT_W;
    input_dims.h = CONV_3X3_DILATION_5X5_INPUT_INPUT_H;
    input_dims.c = CONV_3X3_DILATION_5X5_INPUT_IN_CH;
    filter_dims.w = CONV_3X3_DILATION_5X5_INPUT_FILTER_X;
    filter_dims.h = CONV_3X3_DILATION_5X5_INPUT_FILTER_Y;
    filter_dims.c = CONV_3X3_DILATION_5X5_INPUT_IN_CH;
    output_dims.w = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_W;
    output_dims.h = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_H;
    output_dims.c = CONV_3X3_DILATION_5X5_INPUT_OUT_CH;

    conv_params.padding.w = CONV_3X3_DILATION_5X5_INPUT_PAD_X;
    conv_params.padding.h = CONV_3X3_DILATION_5X5_INPUT_PAD_Y;
    conv_params.stride.w = CONV_3X3_DILATION_5X5_INPUT_STRIDE_X;
    conv_params.stride.h = CONV_3X3_DILATION_5X5_INPUT_STRIDE_Y;
    conv_params.dilation.w = CONV_3X3_DILATION_5X5_INPUT_DILATION_X;
    conv_params.dilation.h = CONV_3X3_DILATION_5X5_INPUT_DILATION_Y;

    conv_params.input_offset = CONV_3X3_DILATION_5X5_INPUT_INPUT_OFFSET;
    conv_params.output_offset = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3X3_DILATION_5X5_INPUT_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3X3_DILATION_5X5_INPUT_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3x3_dilation_5x5_input_output_mult;
    quant_params.shift = (int32_t *)conv_3x3_dilation_5x5_input_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);


    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_2x3_dilation_arm_convolve_s8(void)
{
    int8_t output[CONV_2X3_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2x3_dilation_biases;
    const int8_t *kernel_data = conv_2x3_dilation_weights;
    const int8_t *input_data = conv_2x3_dilation_input;
    const int8_t *output_ref = conv_2x3_dilation_output_ref;
    const int32_t output_ref_size = CONV_2X3_DILATION_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_2X3_DILATION_INPUT_BATCHES;
    input_dims.w = CONV_2X3_DILATION_INPUT_W;
    input_dims.h = CONV_2X3_DILATION_INPUT_H;
    input_dims.c = CONV_2X3_DILATION_IN_CH;
    filter_dims.w = CONV_2X3_DILATION_FILTER_X;
    filter_dims.h = CONV_2X3_DILATION_FILTER_Y;
    filter_dims.c = CONV_2X3_DILATION_IN_CH;
    output_dims.w = CONV_2X3_DILATION_OUTPUT_W;
    output_dims.h = CONV_2X3_DILATION_OUTPUT_H;
    output_dims.c = CONV_2X3_DILATION_OUT_CH;

    conv_params.padding.w = CONV_2X3_DILATION_PAD_X;
    conv_params.padding.h = CONV_2X3_DILATION_PAD_Y;
    conv_params.stride.w = CONV_2X3_DILATION_STRIDE_X;
    conv_params.stride.h = CONV_2X3_DILATION_STRIDE_Y;
    conv_params.dilation.w = CONV_2X3_DILATION_DILATION_X;
    conv_params.dilation.h = CONV_2X3_DILATION_DILATION_Y;

    conv_params.input_offset = CONV_2X3_DILATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X3_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X3_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X3_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_2x3_dilation_output_mult;
    quant_params.shift = (int32_t *)conv_2x3_dilation_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_3x2_dilation_arm_convolve_s8(void)
{
    int8_t output[CONV_3X2_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3x2_dilation_biases;
    const int8_t *kernel_data = conv_3x2_dilation_weights;
    const int8_t *input_data = conv_3x2_dilation_input;
    const int8_t *output_ref = conv_3x2_dilation_output_ref;
    const int32_t output_ref_size = CONV_3X2_DILATION_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_3X2_DILATION_INPUT_BATCHES;
    input_dims.w = CONV_3X2_DILATION_INPUT_W;
    input_dims.h = CONV_3X2_DILATION_INPUT_H;
    input_dims.c = CONV_3X2_DILATION_IN_CH;
    filter_dims.w = CONV_3X2_DILATION_FILTER_X;
    filter_dims.h = CONV_3X2_DILATION_FILTER_Y;
    filter_dims.c = CONV_3X2_DILATION_IN_CH;
    output_dims.w = CONV_3X2_DILATION_OUTPUT_W;
    output_dims.h = CONV_3X2_DILATION_OUTPUT_H;
    output_dims.c = CONV_3X2_DILATION_OUT_CH;

    conv_params.padding.w = CONV_3X2_DILATION_PAD_X;
    conv_params.padding.h = CONV_3X2_DILATION_PAD_Y;
    conv_params.stride.w = CONV_3X2_DILATION_STRIDE_X;
    conv_params.stride.h = CONV_3X2_DILATION_STRIDE_Y;
    conv_params.dilation.w = CONV_3X2_DILATION_DILATION_X;
    conv_params.dilation.h = CONV_3X2_DILATION_DILATION_Y;

    conv_params.input_offset = CONV_3X2_DILATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_3X2_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3X2_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3X2_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_3x2_dilation_output_mult;
    quant_params.shift = (int32_t *)conv_3x2_dilation_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_dilation_golden_arm_convolve_s8(void)
{
    int8_t output[CONV_DILATION_GOLDEN_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_dilation_golden_biases;
    const int8_t *kernel_data = conv_dilation_golden_weights;
    const int8_t *input_data = conv_dilation_golden_input;
    const int8_t *output_ref = conv_dilation_golden_output_ref;
    const int32_t output_ref_size = CONV_DILATION_GOLDEN_DST_SIZE;
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

    input_dims.n = CONV_DILATION_GOLDEN_INPUT_BATCHES;
    input_dims.w = CONV_DILATION_GOLDEN_INPUT_W;
    input_dims.h = CONV_DILATION_GOLDEN_INPUT_H;
    input_dims.c = CONV_DILATION_GOLDEN_IN_CH;
    filter_dims.w = CONV_DILATION_GOLDEN_FILTER_X;
    filter_dims.h = CONV_DILATION_GOLDEN_FILTER_Y;
    filter_dims.c = CONV_DILATION_GOLDEN_IN_CH;
    output_dims.w = CONV_DILATION_GOLDEN_OUTPUT_W;
    output_dims.h = CONV_DILATION_GOLDEN_OUTPUT_H;
    output_dims.c = CONV_DILATION_GOLDEN_OUT_CH;

    conv_params.padding.w = CONV_DILATION_GOLDEN_PAD_X;
    conv_params.padding.h = CONV_DILATION_GOLDEN_PAD_Y;
    conv_params.stride.w = CONV_DILATION_GOLDEN_STRIDE_X;
    conv_params.stride.h = CONV_DILATION_GOLDEN_STRIDE_Y;
    conv_params.dilation.w = CONV_DILATION_GOLDEN_DILATION_X;
    conv_params.dilation.h = CONV_DILATION_GOLDEN_DILATION_Y;

    conv_params.input_offset = CONV_DILATION_GOLDEN_INPUT_OFFSET;
    conv_params.output_offset = CONV_DILATION_GOLDEN_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_DILATION_GOLDEN_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_DILATION_GOLDEN_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_dilation_golden_output_mult;
    quant_params.shift = (int32_t *)conv_dilation_golden_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel_data,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_5_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_5_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_5_biases;
    const int8_t *kernel_data = conv_5_weights;
    const int8_t *input_data = conv_5_input;
    const int8_t *output_ref = conv_5_output_ref;
    const int32_t output_ref_size = CONV_5_DST_SIZE;

    input_dims.n = CONV_5_INPUT_BATCHES;
    input_dims.w = CONV_5_INPUT_W;
    input_dims.h = CONV_5_INPUT_H;
    input_dims.c = CONV_5_IN_CH;
    filter_dims.w = CONV_5_FILTER_X;
    filter_dims.h = CONV_5_FILTER_Y;
    filter_dims.c = CONV_5_IN_CH;
    output_dims.w = CONV_5_OUTPUT_W;
    output_dims.h = CONV_5_OUTPUT_H;
    output_dims.c = CONV_5_OUT_CH;

    conv_params.padding.w = CONV_5_PAD_X;
    conv_params.padding.h = CONV_5_PAD_Y;
    conv_params.stride.w = CONV_5_STRIDE_X;
    conv_params.stride.h = CONV_5_STRIDE_Y;
    conv_params.dilation.w = CONV_5_DILATION_X;
    conv_params.dilation.h = CONV_5_DILATION_Y;

    conv_params.input_offset = CONV_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_5_output_mult;
    quant_params.shift = (int32_t *)conv_5_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);

    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
                                                 &weights_sum_ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 conv_5_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 NULL,
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

    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = arm_convolve_wrapper_s8(&ctx,
                                     &weights_sum_ctx,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_refactored_fc_conv_dilated(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FC_CONV_INT8_DILATED_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fc_conv_int8_dilated_biases;
    const int8_t *kernel_data = fc_conv_int8_dilated_weights;
    const int8_t *input_data = fc_conv_int8_dilated_input_tensor;
    const int8_t *output_ref = fc_conv_int8_dilated_output_ref;
    const int32_t output_ref_size = FC_CONV_INT8_DILATED_DST_SIZE;

    input_dims.n = FC_CONV_INT8_DILATED_INPUT_BATCHES;
    input_dims.w = FC_CONV_INT8_DILATED_INPUT_W;
    input_dims.h = FC_CONV_INT8_DILATED_INPUT_H;
    input_dims.c = FC_CONV_INT8_DILATED_IN_CH;
    filter_dims.w = FC_CONV_INT8_DILATED_FILTER_X;
    filter_dims.h = FC_CONV_INT8_DILATED_FILTER_Y;
    filter_dims.c = FC_CONV_INT8_DILATED_IN_CH;
    output_dims.w = FC_CONV_INT8_DILATED_OUTPUT_W;
    output_dims.h = FC_CONV_INT8_DILATED_OUTPUT_H;
    output_dims.c = FC_CONV_INT8_DILATED_OUT_CH;

    conv_params.padding.w = FC_CONV_INT8_DILATED_PAD_X;
    conv_params.padding.h = FC_CONV_INT8_DILATED_PAD_Y;
    conv_params.stride.w = FC_CONV_INT8_DILATED_STRIDE_X;
    conv_params.stride.h = FC_CONV_INT8_DILATED_STRIDE_Y;
    conv_params.dilation.w = FC_CONV_INT8_DILATED_DILATION_X;
    conv_params.dilation.h = FC_CONV_INT8_DILATED_DILATION_Y;

    conv_params.input_offset = FC_CONV_INT8_DILATED_INPUT_OFFSET;
    conv_params.output_offset = FC_CONV_INT8_DILATED_OUTPUT_OFFSET;
    conv_params.activation.min = FC_CONV_INT8_DILATED_OUT_ACTIVATION_MIN;
    conv_params.activation.max = FC_CONV_INT8_DILATED_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)fc_conv_int8_dilated_output_mult;
    quant_params.shift = (int32_t *)fc_conv_int8_dilated_output_shift;
    arm_cmsis_nn_status result;
    int32_t buf_size;
    memset(output, 0, sizeof(output));


    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    
    result = conv_1x1_out_wrapper(&ctx,
        &conv_params,
        &quant_params,
        &input_dims,
        &filter_dims,
        &bias_dims,
        &output_dims,
        bias_data,
        kernel_data,
        input_data,
        output
    );
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}


void conv_refactored_fc_conv_int8_diff_channels(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FC_CONV_INT8_DIFF_CHANNELS_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fc_conv_int8_diff_channels_biases;
    const int8_t *kernel_data = fc_conv_int8_diff_channels_weights;
    const int8_t *input_data = fc_conv_int8_diff_channels_input_tensor;
    const int8_t *output_ref = fc_conv_int8_diff_channels_output_ref;
    const int32_t output_ref_size = FC_CONV_INT8_DIFF_CHANNELS_DST_SIZE;

    input_dims.n = FC_CONV_INT8_DIFF_CHANNELS_INPUT_BATCHES;
    input_dims.w = FC_CONV_INT8_DIFF_CHANNELS_INPUT_W;
    input_dims.h = FC_CONV_INT8_DIFF_CHANNELS_INPUT_H;
    input_dims.c = FC_CONV_INT8_DIFF_CHANNELS_IN_CH;
    filter_dims.w = FC_CONV_INT8_DIFF_CHANNELS_FILTER_X;
    filter_dims.h = FC_CONV_INT8_DIFF_CHANNELS_FILTER_Y;
    filter_dims.c = FC_CONV_INT8_DIFF_CHANNELS_IN_CH;
    output_dims.w = FC_CONV_INT8_DIFF_CHANNELS_OUTPUT_W;
    output_dims.h = FC_CONV_INT8_DIFF_CHANNELS_OUTPUT_H;
    output_dims.c = FC_CONV_INT8_DIFF_CHANNELS_OUT_CH;

    conv_params.padding.w = FC_CONV_INT8_DIFF_CHANNELS_PAD_X;
    conv_params.padding.h = FC_CONV_INT8_DIFF_CHANNELS_PAD_Y;
    conv_params.stride.w = FC_CONV_INT8_DIFF_CHANNELS_STRIDE_X;
    conv_params.stride.h = FC_CONV_INT8_DIFF_CHANNELS_STRIDE_Y;
    conv_params.dilation.w = FC_CONV_INT8_DIFF_CHANNELS_DILATION_X;
    conv_params.dilation.h = FC_CONV_INT8_DIFF_CHANNELS_DILATION_Y;

    conv_params.input_offset = FC_CONV_INT8_DIFF_CHANNELS_INPUT_OFFSET;
    conv_params.output_offset = FC_CONV_INT8_DIFF_CHANNELS_OUTPUT_OFFSET;
    conv_params.activation.min = FC_CONV_INT8_DIFF_CHANNELS_OUT_ACTIVATION_MIN;
    conv_params.activation.max = FC_CONV_INT8_DIFF_CHANNELS_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)fc_conv_int8_diff_channels_output_mult;
    quant_params.shift = (int32_t *)fc_conv_int8_diff_channels_output_shift;
    arm_cmsis_nn_status result;
    int32_t buf_size;
    memset(output, 0, sizeof(output));


    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = conv_1x1_out_wrapper(&ctx,
        &conv_params,
        &quant_params,
        &input_dims,
        &filter_dims,
        &bias_dims,
        &output_dims,
        bias_data,
        kernel_data,
        input_data,
        output
    );

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_refactored_fc_conv_int8_non_4_multiple(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FC_CONV_INT8_NON_4_MULTIPLE_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fc_conv_int8_non_4_multiple_biases;
    const int8_t *kernel_data = fc_conv_int8_non_4_multiple_weights;
    const int8_t *input_data = fc_conv_int8_non_4_multiple_input_tensor;
    const int8_t *output_ref = fc_conv_int8_non_4_multiple_output_ref;
    const int32_t output_ref_size = FC_CONV_INT8_NON_4_MULTIPLE_DST_SIZE;

    input_dims.n = FC_CONV_INT8_NON_4_MULTIPLE_INPUT_BATCHES;
    input_dims.w = FC_CONV_INT8_NON_4_MULTIPLE_INPUT_W;
    input_dims.h = FC_CONV_INT8_NON_4_MULTIPLE_INPUT_H;
    input_dims.c = FC_CONV_INT8_NON_4_MULTIPLE_IN_CH;
    filter_dims.w = FC_CONV_INT8_NON_4_MULTIPLE_FILTER_X;
    filter_dims.h = FC_CONV_INT8_NON_4_MULTIPLE_FILTER_Y;
    filter_dims.c = FC_CONV_INT8_NON_4_MULTIPLE_IN_CH;
    output_dims.w = FC_CONV_INT8_NON_4_MULTIPLE_OUTPUT_W;
    output_dims.h = FC_CONV_INT8_NON_4_MULTIPLE_OUTPUT_H;
    output_dims.c = FC_CONV_INT8_NON_4_MULTIPLE_OUT_CH;

    conv_params.padding.w = FC_CONV_INT8_NON_4_MULTIPLE_PAD_X;
    conv_params.padding.h = FC_CONV_INT8_NON_4_MULTIPLE_PAD_Y;
    conv_params.stride.w = FC_CONV_INT8_NON_4_MULTIPLE_STRIDE_X;
    conv_params.stride.h = FC_CONV_INT8_NON_4_MULTIPLE_STRIDE_Y;
    conv_params.dilation.w = FC_CONV_INT8_NON_4_MULTIPLE_DILATION_X;
    conv_params.dilation.h = FC_CONV_INT8_NON_4_MULTIPLE_DILATION_Y;

    conv_params.input_offset = FC_CONV_INT8_NON_4_MULTIPLE_INPUT_OFFSET;
    conv_params.output_offset = FC_CONV_INT8_NON_4_MULTIPLE_OUTPUT_OFFSET;
    conv_params.activation.min = FC_CONV_INT8_NON_4_MULTIPLE_OUT_ACTIVATION_MIN;
    conv_params.activation.max = FC_CONV_INT8_NON_4_MULTIPLE_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)fc_conv_int8_non_4_multiple_output_mult;
    quant_params.shift = (int32_t *)fc_conv_int8_non_4_multiple_output_shift;
    arm_cmsis_nn_status result;
    int32_t buf_size;
    memset(output, 0, sizeof(output));


    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = conv_1x1_out_wrapper(&ctx,
        &conv_params,
        &quant_params,
        &input_dims,
        &filter_dims,
        &bias_dims,
        &output_dims,
        bias_data,
        kernel_data,
        input_data,
        output
    );
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_refactored_fc_conv_int8_1x1_kernel(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FC_CONV_INT8_1X1_KERNEL_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fc_conv_int8_1x1_kernel_biases;
    const int8_t *kernel_data = fc_conv_int8_1x1_kernel_weights;
    const int8_t *input_data = fc_conv_int8_1x1_kernel_input_tensor;
    const int8_t *output_ref = fc_conv_int8_1x1_kernel_output_ref;
    const int32_t output_ref_size = FC_CONV_INT8_1X1_KERNEL_DST_SIZE;

    input_dims.n = FC_CONV_INT8_1X1_KERNEL_INPUT_BATCHES;
    input_dims.w = FC_CONV_INT8_1X1_KERNEL_INPUT_W;
    input_dims.h = FC_CONV_INT8_1X1_KERNEL_INPUT_H;
    input_dims.c = FC_CONV_INT8_1X1_KERNEL_IN_CH;
    filter_dims.n = FC_CONV_INT8_1X1_KERNEL_OUT_CH;
    filter_dims.w = FC_CONV_INT8_1X1_KERNEL_FILTER_X;
    filter_dims.h = FC_CONV_INT8_1X1_KERNEL_FILTER_Y;
    filter_dims.c = FC_CONV_INT8_1X1_KERNEL_IN_CH;
    output_dims.w = FC_CONV_INT8_1X1_KERNEL_OUTPUT_W;
    output_dims.h = FC_CONV_INT8_1X1_KERNEL_OUTPUT_H;
    output_dims.c = FC_CONV_INT8_1X1_KERNEL_OUT_CH;

    conv_params.padding.w = FC_CONV_INT8_1X1_KERNEL_PAD_X;
    conv_params.padding.h = FC_CONV_INT8_1X1_KERNEL_PAD_Y;
    conv_params.stride.w = FC_CONV_INT8_1X1_KERNEL_STRIDE_X;
    conv_params.stride.h = FC_CONV_INT8_1X1_KERNEL_STRIDE_Y;
    conv_params.dilation.w = FC_CONV_INT8_1X1_KERNEL_DILATION_X;
    conv_params.dilation.h = FC_CONV_INT8_1X1_KERNEL_DILATION_Y;

    conv_params.input_offset = FC_CONV_INT8_1X1_KERNEL_INPUT_OFFSET;
    conv_params.output_offset = FC_CONV_INT8_1X1_KERNEL_OUTPUT_OFFSET;
    conv_params.activation.min = FC_CONV_INT8_1X1_KERNEL_OUT_ACTIVATION_MIN;
    conv_params.activation.max = FC_CONV_INT8_1X1_KERNEL_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)fc_conv_int8_1x1_kernel_output_mult;
    quant_params.shift = (int32_t *)fc_conv_int8_1x1_kernel_output_shift;
    arm_cmsis_nn_status result;
    int32_t buf_size;
    memset(output, 0, sizeof(output));
    //TODO need to skip the wrapper here because this case is not handled by the 1x1 image, because it has 1x1 kernel
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    result = conv_1x1_out_wrapper(&ctx,
        &conv_params,
        &quant_params,
        &input_dims,
        &filter_dims,
        &bias_dims,
        &output_dims,
        bias_data,
        kernel_data,
        input_data,
        output
    );

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1x1_out_tail_arm_convolve_s8(void)
{
#if defined(ARM_MATH_MVEI)
    enum
    {
        max_output_channels = 9,
        input_channels = 4,
        kernel_elements = 4,
        input_size = input_channels * kernel_elements,
        output_elements = 1
    };
    const int32_t output_channels_to_test[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int8_t guard_value = 0x5A;
    const int32_t input_offset = 5;
    const int32_t output_offset = -3;
    const int32_t activation_min = -11;
    const int32_t activation_max = 9;
    int8_t input[input_size];
    int8_t kernel[max_output_channels * input_size];
    int32_t bias[max_output_channels];
    int32_t multiplier[max_output_channels];
    int32_t shift[max_output_channels];
    int32_t weight_sum[max_output_channels + 4];
    int8_t output_storage[max_output_channels + 4];

    for (int i = 0; i < input_size; i++)
    {
        input[i] = (int8_t)((i * 7) % 23 - 11);
    }
    for (int i = 0; i < max_output_channels; i++)
    {
        bias[i] = i * 37 - 100;
        multiplier[i] = (i % 3 == 0) ? (1 << 30) : ((i % 3 == 1) ? (1 << 29) : (3 << 29));
        shift[i] = (i % 4) - 2;
        for (int j = 0; j < input_size; j++)
        {
            kernel[i * input_size + j] = (int8_t)((i * 5 + j * 3) % 17 - 8);
        }
    }

    cmsis_nn_dims input_dims = {1, 2, 2, input_channels};
    cmsis_nn_dims filter_dims = {max_output_channels, 2, 2, input_channels};
    cmsis_nn_dims bias_dims = {1, 1, 1, max_output_channels};
    cmsis_nn_dims output_dims = {1, 1, 1, max_output_channels};
    cmsis_nn_conv_params conv_params = {
        .input_offset = input_offset,
        .output_offset = output_offset,
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
        .activation = {activation_min, activation_max},
    };
    cmsis_nn_context ctx;
    cmsis_nn_context weight_sum_ctx;
    cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multiplier,
        .shift = shift,
    };

    const int32_t buffer_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(buffer_size > 0);
    ctx.buf = malloc(buffer_size);
    ctx.size = buffer_size;
    weight_sum_ctx.buf = weight_sum;
    weight_sum_ctx.size = max_output_channels * (int32_t)sizeof(int32_t);

    for (size_t test_index = 0; test_index < sizeof(output_channels_to_test) / sizeof(output_channels_to_test[0]);
         test_index++)
    {
        const int32_t output_channels = output_channels_to_test[test_index];
        output_dims.c = output_channels;
        filter_dims.n = output_channels;
        bias_dims.c = output_channels;
        memset(output_storage, guard_value, sizeof(output_storage));
        memset(weight_sum + output_channels, 0xA5, (max_output_channels + 4 - output_channels) * sizeof(int32_t));

        TEST_ASSERT_EQUAL(
            ARM_CMSIS_NN_SUCCESS,
            arm_convolve_weight_sum(weight_sum, kernel, &input_dims, &filter_dims, &output_dims, input_offset, bias));

        const arm_cmsis_nn_status result = arm_convolve_wrapper_s8(&ctx,
                                                                   &weight_sum_ctx,
                                                                   &conv_params,
                                                                   &quant_params,
                                                                   &input_dims,
                                                                   input,
                                                                   &filter_dims,
                                                                   kernel,
                                                                   &bias_dims,
                                                                   bias,
                                                                   &output_dims,
                                                                   output_storage + 2);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        TEST_ASSERT_EQUAL_INT8(guard_value, output_storage[0]);
        TEST_ASSERT_EQUAL_INT8(guard_value, output_storage[1]);
        TEST_ASSERT_EQUAL_INT8(guard_value, output_storage[output_channels + 2]);
        TEST_ASSERT_EQUAL_INT8(guard_value, output_storage[output_channels + 3]);

        int saw_clip = 0;
        for (int i = 0; i < output_channels; i++)
        {
            int32_t acc = bias[i];
            for (int j = 0; j < input_size; j++)
            {
                acc += (input[j] + input_offset) * kernel[i * input_size + j];
            }
            int32_t expected = arm_nn_requantize(acc, multiplier[i], shift[i]) + output_offset;
            expected = MAX(expected, activation_min);
            expected = MIN(expected, activation_max);
            saw_clip |= expected == activation_min || expected == activation_max;
            TEST_ASSERT_EQUAL_INT8((int8_t)expected, output_storage[i + 2]);
        }
        TEST_ASSERT_TRUE(saw_clip);
        for (int i = output_channels; i < max_output_channels; i++)
        {
            TEST_ASSERT_EQUAL_INT32(0xA5A5A5A5, weight_sum[i]);
        }
    }

    memset(ctx.buf, 0, buffer_size);
    free(ctx.buf);
#endif
}

void buffer_size_arm_convolve_s8(void)
{
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INPUT_BATCHES;
    input_dims.w = CONV_5_INPUT_W;
    input_dims.h = CONV_5_INPUT_H;
    input_dims.c = CONV_5_IN_CH;
    filter_dims.w = CONV_5_FILTER_X;
    filter_dims.h = CONV_5_FILTER_Y;
    filter_dims.c = CONV_5_IN_CH;
    output_dims.w = CONV_5_OUTPUT_W;
    output_dims.h = CONV_5_OUTPUT_H;
    output_dims.c = CONV_5_OUT_CH;

    conv_params.padding.w = CONV_5_PAD_X;
    conv_params.padding.h = CONV_5_PAD_Y;
    conv_params.stride.w = CONV_5_STRIDE_X;
    conv_params.stride.h = CONV_5_STRIDE_Y;
    conv_params.dilation.w = CONV_5_DILATION_X;
    conv_params.dilation.h = CONV_5_DILATION_Y;

    conv_params.input_offset = CONV_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_OUT_ACTIVATION_MAX;

    const int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, buf_size);
}

void buffer_size_mve_arm_convolve_s8(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INPUT_BATCHES;
    input_dims.w = CONV_5_INPUT_W;
    input_dims.h = CONV_5_INPUT_H;
    input_dims.c = CONV_5_IN_CH;
    filter_dims.w = CONV_5_FILTER_X;
    filter_dims.h = CONV_5_FILTER_Y;
    filter_dims.c = CONV_5_IN_CH;
    output_dims.w = CONV_5_OUTPUT_W;
    output_dims.h = CONV_5_OUTPUT_H;
    output_dims.c = CONV_5_OUT_CH;

    conv_params.padding.w = CONV_5_PAD_X;
    conv_params.padding.h = CONV_5_PAD_Y;
    conv_params.stride.w = CONV_5_STRIDE_X;
    conv_params.stride.h = CONV_5_STRIDE_Y;
    conv_params.dilation.w = CONV_5_DILATION_X;
    conv_params.dilation.h = CONV_5_DILATION_Y;

    conv_params.input_offset = CONV_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_convolve_s8(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = CONV_5_INPUT_BATCHES;
    input_dims.w = CONV_5_INPUT_W;
    input_dims.h = CONV_5_INPUT_H;
    input_dims.c = CONV_5_IN_CH;
    filter_dims.w = CONV_5_FILTER_X;
    filter_dims.h = CONV_5_FILTER_Y;
    filter_dims.c = CONV_5_IN_CH;
    output_dims.w = CONV_5_OUTPUT_W;
    output_dims.h = CONV_5_OUTPUT_H;
    output_dims.c = CONV_5_OUT_CH;

    conv_params.padding.w = CONV_5_PAD_X;
    conv_params.padding.h = CONV_5_PAD_Y;
    conv_params.stride.w = CONV_5_STRIDE_X;
    conv_params.stride.h = CONV_5_STRIDE_Y;
    conv_params.dilation.w = CONV_5_DILATION_X;
    conv_params.dilation.h = CONV_5_DILATION_Y;

    conv_params.input_offset = CONV_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_5_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_convolve_wrapper_s8_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}
