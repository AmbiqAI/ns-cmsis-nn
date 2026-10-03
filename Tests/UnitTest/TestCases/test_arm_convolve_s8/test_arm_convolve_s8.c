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

#include "../Utils/mpu_guard.h"

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
    TEST_ASSERT_NOT_NULL(ctx.buf);
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
            expected = ARM_NN_MAX(expected, activation_min);
            expected = ARM_NN_MIN(expected, activation_max);
            saw_clip |= expected == activation_min || expected == activation_max;
            TEST_ASSERT_EQUAL_INT8((int8_t)expected, output_storage[i + 2]);
        }
        TEST_ASSERT_TRUE(saw_clip);
        for (int i = output_channels; i < max_output_channels + 4; i++)
        {
            TEST_ASSERT_EQUAL_INT32(0xA5A5A5A5, weight_sum[i]);
        }
    }

    memset(ctx.buf, 0, buffer_size);
    free(ctx.buf);
#endif
}

/* Quantization classes for the right-shift-only requantization: 0 every shift in [-30, -1]; 1 as 0 with dead
 * channels (multiplier 0, shift 0); 2 as 0 with one shift of +1; 3 as 0 with one shift of 0 and a non-zero multiplier;
 * 4 shifts in [0, 2]. Classes 0 and 1 take the right-shift path, the others the general one. Shifts stop at -30:
 * the scalar arm_divide_by_power_of_two() overflows forming its remainder mask for an exponent of 31. */
static void conv_1x1_quant_class(int32_t cls, int32_t channels, int32_t *multiplier, int32_t *shift, uint32_t *seed)
{
    for (int32_t i = 0; i < channels; i++)
    {
        *seed = *seed * 1664525u + 1013904223u;
        multiplier[i] = (int32_t)(0x40000000u + ((*seed >> 2) & 0x3FFFFFFFu));
        *seed = *seed * 1664525u + 1013904223u;
        shift[i] = cls == 4 ? (int32_t)((*seed >> 8) % 3u) : -1 - (int32_t)((*seed >> 8) % 30u);
        if (cls == 1 && i % 3 == 1)
        {
            multiplier[i] = 0;
            shift[i] = 0;
        }
    }
    if (cls == 2)
    {
        shift[channels - 1] = 1;
    }
    if (cls == 3)
    {
        shift[channels / 2] = 0;
    }
    /* Keep the channels the general path requantizes inside the int8 range, so that taking the wrong path changes
     * outputs instead of saturating them. */
    for (int32_t i = 0; i < channels; i++)
    {
        if (shift[i] >= 0 && multiplier[i] != 0)
        {
            multiplier[i] = (0x30000 + i * 0x1235) >> shift[i];
        }
    }
}

/* Every quantization class through the 1x1 path against arm_nn_requantize(). 30 pixels, 20 input and 7 output channels
 * reach arm_nn_mat_mult_nt_t_s8()'s four-row loop and its one-row loop (2 remaining pixels), the channel blocks of four
 * and the channel tail. */
void conv_1x1_requant_classes_arm_convolve_s8(void)
{
    enum
    {
        in_h = 6,
        in_w = 5,
        in_c = 20,
        out_c = 7,
        pixels = in_h * in_w
    };
    const int32_t input_offset = 128;
    const int32_t output_offset = -3;
    const int32_t activation_min = -128;
    const int32_t activation_max = 127;
    int8_t input[pixels * in_c];
    int8_t kernel[out_c * in_c];
    int32_t bias[out_c];
    int32_t multiplier[out_c];
    int32_t shift[out_c];
    int32_t weight_sum[out_c];
    int8_t output[pixels * out_c + 4];
    uint32_t seed = 41u;

    for (int32_t i = 0; i < pixels * in_c; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        input[i] = (int8_t)(seed >> 24);
    }
    for (int32_t i = 0; i < out_c * in_c; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        kernel[i] = (int8_t)(seed >> 24);
    }
    for (int32_t i = 0; i < out_c; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        bias[i] = (int32_t)(seed >> 14) - (1 << 17);
    }

    const cmsis_nn_dims input_dims = {1, in_h, in_w, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 1, 1, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, in_h, in_w, out_c};
    const cmsis_nn_conv_params conv_params = {
        .input_offset = input_offset,
        .output_offset = output_offset,
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
        .activation = {activation_min, activation_max},
    };
    const cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multiplier,
        .shift = shift,
    };
    cmsis_nn_context ctx = {0};
    const cmsis_nn_context weight_sum_ctx = {weight_sum, (int32_t)sizeof(weight_sum)};
    ctx.size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    if (ctx.size > 0)
    {
        ctx.buf = malloc((size_t)ctx.size);
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }
#if defined(ARM_MATH_MVEI)
    /* Only the MVE path reads the weight sums. */
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_convolve_weight_sum(weight_sum, kernel, &input_dims, &filter_dims, &output_dims, input_offset, bias));
#endif

    static const char *const cls_name[5] = {"rshift", "rshift+dead", "shift+1", "shift0", "lshift"};
    for (int32_t cls = 0; cls < 5; cls++)
    {
        conv_1x1_quant_class(cls, out_c, multiplier, shift, &seed);
        memset(output, 0x5A, sizeof(output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_convolve_wrapper_s8(&ctx,
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
                                                  output + 2));
        TEST_ASSERT_EQUAL_INT8(0x5A, output[0]);
        TEST_ASSERT_EQUAL_INT8(0x5A, output[1]);
        TEST_ASSERT_EQUAL_INT8(0x5A, output[pixels * out_c + 2]);
        TEST_ASSERT_EQUAL_INT8(0x5A, output[pixels * out_c + 3]);
        for (int32_t px = 0; px < pixels; px++)
        {
            for (int32_t oc = 0; oc < out_c; oc++)
            {
                int32_t acc = bias[oc];
                for (int32_t ic = 0; ic < in_c; ic++)
                {
                    acc += (input[px * in_c + ic] + input_offset) * kernel[oc * in_c + ic];
                }
                int32_t expected = arm_nn_requantize(acc, multiplier[oc], shift[oc]) + output_offset;
                expected = ARM_NN_MAX(expected, activation_min);
                expected = ARM_NN_MIN(expected, activation_max);
                TEST_ASSERT_EQUAL_INT8_MESSAGE((int8_t)expected, output[2 + px * out_c + oc], cls_name[cls]);
            }
        }
    }
    free(ctx.buf);
}

/* arm_requantize_mve_rshift() and arm_requantize_mve_32x4_rshift() against arm_requantize_mve() and
 * arm_requantize_mve_32x4() on every input arm_nn_requantize_rshift_only() admits: shifts in [-40, -1], including
 * those below -31 that the scalar reference does not define, extreme values and multipliers, and multiplier 0. */
void requantize_rshift_helpers_arm_convolve_s8(void)
{
#if defined(ARM_MATH_MVEI) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    const int32_t edges[] = {INT32_MIN, INT32_MIN + 1, -65537, -2, -1, 0, 1, 2, 65535, INT32_MAX - 1, INT32_MAX};
    uint32_t seed = 47u;
    for (int32_t shift = -40; shift <= 0; shift++)
    {
        for (int32_t round = 0; round < 64; round++)
        {
            int32_t vals[4];
            int32_t mults[4];
            int32_t shifts[4];
            for (int32_t l = 0; l < 4; l++)
            {
                seed = seed * 1664525u + 1013904223u;
                vals[l] = round < 3 ? edges[(round * 4 + l) % 11] : (int32_t)seed;
                seed = seed * 1664525u + 1013904223u;
                mults[l] = shift == 0 ? 0 : (round == 0 ? INT32_MAX : (int32_t)(0x40000000u + (seed >> 2)));
                if (round == 1 && shift != 0)
                {
                    /* A negative multiplier, which the check also admits. */
                    mults[l] = -mults[l];
                }
                seed = seed * 1664525u + 1013904223u;
                shifts[l] = shift == 0 ? 0 : -1 - (int32_t)((seed >> 8) % 40u);
            }
            TEST_ASSERT_TRUE(arm_nn_requantize_rshift_only(mults, shifts, 4));
            const int32x4_t v = vldrwq_s32(vals);
            int32_t want[4];
            int32_t got[4];
            vstrwq_s32(want, arm_requantize_mve(v, mults[0], shift));
            vstrwq_s32(got, arm_requantize_mve_rshift(v, mults[0], shift));
            TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 4);
            vstrwq_s32(want, arm_requantize_mve_32x4(v, vldrwq_s32(mults), vldrwq_s32(shifts)));
            vstrwq_s32(got, arm_requantize_mve_32x4_rshift(v, vldrwq_s32(mults), vldrwq_s32(shifts)));
            TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 4);
        }
    }
    const int32_t mult_lshift[2] = {1 << 30, 1 << 30};
    const int32_t shift_lshift[2] = {-3, 0};
    TEST_ASSERT_FALSE(arm_nn_requantize_rshift_only(mult_lshift, shift_lshift, 2));
    const int32_t mult_dead[2] = {1 << 30, 0};
    const int32_t shift_dead[2] = {-3, 5};
    TEST_ASSERT_TRUE(arm_nn_requantize_rshift_only(mult_dead, shift_dead, 2));
    /* Channel counts that end in a partial vector, with the one channel that fails the check last. */
    for (int32_t num_ch = 5; num_ch <= 7; num_ch++)
    {
        int32_t mult_tail[8];
        int32_t shift_tail[8];
        for (int32_t i = 0; i < 8; i++)
        {
            mult_tail[i] = 1 << 30;
            shift_tail[i] = -4;
        }
        TEST_ASSERT_TRUE(arm_nn_requantize_rshift_only(mult_tail, shift_tail, num_ch));
        shift_tail[num_ch - 1] = 0;
        TEST_ASSERT_FALSE(arm_nn_requantize_rshift_only(mult_tail, shift_tail, num_ch));
        shift_tail[num_ch - 1] = -4;
        shift_tail[num_ch] = 0; /* past the last channel: must not count */
        TEST_ASSERT_TRUE(arm_nn_requantize_rshift_only(mult_tail, shift_tail, num_ch));
    }
    /* A zero multiplier with a positive shift, which the check also admits. */
    for (int32_t i = 0; i < 11; i += 4)
    {
        const int32x4_t v = vldrwq_s32(&edges[i < 8 ? i : 7]);
        int32_t want[4];
        int32_t got[4];
        vstrwq_s32(want, arm_requantize_mve(v, 0, 5));
        vstrwq_s32(got, arm_requantize_mve_rshift(v, 0, 5));
        TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 4);
    }
#endif
}

/* arm_convolve_1x1_s8_short_k() against arm_nn_requantize(), at every input depth from 1 to 16 and at 17 (outside the
 * gate), with every pixel tail, the paired-pixel depth of 8, one and two batches, the full and a narrower activation
 * range, and every quantization class, including the two that mix right-shift-only and general channels in one layer.
 * Outside the gate, and on builds without the kernel, it returns ARM_CMSIS_NN_NO_IMPL_ERROR and writes nothing. Where
 * the MPU guard is available, input, filter and output each end at the guard gap in turn. */
void conv_1x1_short_k_arm_convolve_1x1_s8_short_k(void)
{
    enum
    {
        max_px = 2 * 13,
        max_cin = 17,
        max_cout = 8
    };
    const int32_t cins[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    const int32_t pixel_counts[] = {7, 8, 9, 10, 11, 13};
    const int32_t couts[] = {1, 5, 8};
    const int32_t classes[] = {0, 1, 2, 3, 4};
    const int32_t batches[] = {1, 2};
    const int32_t input_offset = 128;
    const int32_t output_offset = 5;
    int8_t input[max_px * max_cin];
    int8_t kernel[max_cout * max_cin];
    int32_t bias[max_cout];
    int32_t multiplier[max_cout];
    int32_t shift[max_cout];
    int32_t weight_sum[max_cout];
    int8_t output[max_px * max_cout + 4];
    uint32_t seed = 53u;

    for (size_t a = 0; a < sizeof(cins) / sizeof(cins[0]) * 2; a++)
    {
        /* Each depth runs as one batch, then as two batches of that many pixels each. */
        const int32_t batch = batches[a & 1];
        for (size_t b = 0; b < sizeof(pixel_counts) / sizeof(pixel_counts[0]); b++)
        {
            for (size_t c = 0; c < sizeof(couts) / sizeof(couts[0]); c++)
            {
                const int32_t in_c = cins[a >> 1];
                const int32_t pixels = batch * pixel_counts[b];
                const int32_t out_c = couts[c];
                for (int32_t i = 0; i < pixels * in_c; i++)
                {
                    seed = seed * 1664525u + 1013904223u;
                    input[i] = (int8_t)(seed >> 24);
                }
                for (int32_t i = 0; i < out_c * in_c; i++)
                {
                    seed = seed * 1664525u + 1013904223u;
                    kernel[i] = (int8_t)(seed >> 24);
                }
                for (int32_t i = 0; i < out_c; i++)
                {
                    seed = seed * 1664525u + 1013904223u;
                    bias[i] = (int32_t)(seed >> 16) - (1 << 15);
                }
                const cmsis_nn_dims input_dims = {batch, 1, pixels / batch, in_c};
                const cmsis_nn_dims filter_dims = {out_c, 1, 1, in_c};
                const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
                const cmsis_nn_dims output_dims = {batch, 1, pixels / batch, out_c};
                cmsis_nn_conv_params conv_params = {
                    .input_offset = input_offset,
                    .output_offset = output_offset,
                    .stride = {1, 1},
                    .padding = {0, 0},
                    .dilation = {1, 1},
                    .activation = {-128, 127},
                };
                const cmsis_nn_per_channel_quant_params quant_params = {
                    .multiplier = multiplier,
                    .shift = shift,
                };
                const cmsis_nn_context weight_sum_ctx = {weight_sum, out_c * (int32_t)sizeof(int32_t)};
                const cmsis_nn_context ctx = {0};
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
                const int32_t runs = in_c <= 16;
#else
                const int32_t runs = 0;
#endif
#if defined(ARM_MATH_MVEI)
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                                  arm_convolve_weight_sum(
                                      weight_sum, kernel, &input_dims, &filter_dims, &output_dims, input_offset, bias));
#endif
                for (size_t k = 0; k < sizeof(classes) / sizeof(classes[0]); k++)
                {
                    conv_1x1_quant_class(classes[k], out_c, multiplier, shift, &seed);
                    /* Alternate the full int8 range with a narrower clamp. */
                    conv_params.activation.min = (k & 1) ? -100 : -128;
                    conv_params.activation.max = (k & 1) ? 90 : 127;
                    const int32_t activation_min = conv_params.activation.min;
                    const int32_t activation_max = conv_params.activation.max;
                    memset(output, 0x5A, sizeof(output));
                    TEST_ASSERT_EQUAL(runs ? ARM_CMSIS_NN_SUCCESS : ARM_CMSIS_NN_NO_IMPL_ERROR,
                                      arm_convolve_1x1_s8_short_k(&ctx,
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
                                                                  output + 2));
                    TEST_ASSERT_EQUAL_INT8(0x5A, output[0]);
                    TEST_ASSERT_EQUAL_INT8(0x5A, output[1]);
                    TEST_ASSERT_EQUAL_INT8(0x5A, output[pixels * out_c + 2]);
                    TEST_ASSERT_EQUAL_INT8(0x5A, output[pixels * out_c + 3]);
                    if (!runs)
                    {
                        for (int32_t i = 0; i < pixels * out_c; i++)
                        {
                            TEST_ASSERT_EQUAL_INT8(0x5A, output[2 + i]);
                        }
                        continue;
                    }
                    for (int32_t px = 0; px < pixels; px++)
                    {
                        for (int32_t oc = 0; oc < out_c; oc++)
                        {
                            int32_t acc = bias[oc];
                            for (int32_t ic = 0; ic < in_c; ic++)
                            {
                                acc += (input[px * in_c + ic] + input_offset) * kernel[oc * in_c + ic];
                            }
                            int32_t expected = arm_nn_requantize(acc, multiplier[oc], shift[oc]) + output_offset;
                            expected = ARM_NN_MAX(expected, activation_min);
                            expected = ARM_NN_MIN(expected, activation_max);
                            TEST_ASSERT_EQUAL_INT8((int8_t)expected, output[2 + px * out_c + oc]);
                        }
                    }
#if defined(MPU_GUARD_AVAILABLE)
                    if (k == 0)
                    {
                        /* Each operand in turn ends at the MPU gap, so a read or write past it faults. */
                        for (int32_t at_gap = 0; at_gap < 3; at_gap++)
                        {
                            const int8_t *in = at_gap == 0 ? guard_place(input, (size_t)(pixels * in_c)) : input;
                            const int8_t *w = at_gap == 1 ? guard_place(kernel, (size_t)(out_c * in_c)) : kernel;
                            int8_t *out = at_gap == 2 ? guard_end((size_t)(pixels * out_c)) : output + 2;
                            guard_gap_enable();
                            const arm_cmsis_nn_status st = arm_convolve_1x1_s8_short_k(&ctx,
                                                                                    &weight_sum_ctx,
                                                                                    &conv_params,
                                                                                    &quant_params,
                                                                                    &input_dims,
                                                                                    in,
                                                                                    &filter_dims,
                                                                                    w,
                                                                                    &bias_dims,
                                                                                    bias,
                                                                                    &output_dims,
                                                                                    out);
                            guard_gap_disable();
                            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, st);
                            if (at_gap == 2)
                            {
                                TEST_ASSERT_EQUAL_INT8_ARRAY(output + 2, out, pixels * out_c);
                            }
                        }
                    }
#endif
                }
            }
        }
    }
}

/* arm_convolve_1x1_s8_short_k() declines layers outside its gate (a stride, padding, a filter depth unlike the input
 * depth, a dilation) with ARM_CMSIS_NN_NO_IMPL_ERROR and writes nothing; with NULL weight sums it reports
 * ARM_CMSIS_NN_ARG_ERROR on builds that have the kernel. */
void conv_1x1_short_k_declines_arm_convolve_1x1_s8_short_k(void)
{
    int8_t input[8 * 8 * 8] = {0};
    int8_t kernel[4 * 8] = {0};
    int32_t bias[4] = {0}, multiplier[4] = {1 << 30, 1 << 30, 1 << 30, 1 << 30}, shift[4] = {-1, -1, -1, -1};
    int32_t weight_sum[4] = {0};
    int8_t output[8 * 8 * 4];
    const cmsis_nn_per_channel_quant_params quant_params = {multiplier, shift};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_context weight_sum_ctx = {weight_sum, (int32_t)sizeof(weight_sum)};
    const cmsis_nn_dims bias_dims = {1, 1, 1, 4};
    for (int32_t k = 0; k < 4; k++)
    {
        cmsis_nn_conv_params conv_params = {.stride = {1, 1}, .dilation = {1, 1}, .activation = {-128, 127}};
        cmsis_nn_dims input_dims = {1, 4, 4, 8}, filter_dims = {4, 1, 1, 8}, output_dims = {1, 4, 4, 4};
        if (k == 0)
        {
            conv_params.stride.w = 2;
            output_dims.w = 2;
        }
        else if (k == 1)
        {
            conv_params.padding.h = 1;
        }
        else if (k == 2)
        {
            filter_dims.c = 4;
        }
        else
        {
            conv_params.dilation.h = 2;
        }
        memset(output, 0x5A, sizeof(output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                          arm_convolve_1x1_s8_short_k(&ctx,
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
                                                      output));
        for (int32_t i = 0; i < (int32_t)sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_INT8(0x5A, output[i]);
        }
    }
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const cmsis_nn_conv_params conv_params = {.stride = {1, 1}, .dilation = {1, 1}, .activation = {-128, 127}};
    const cmsis_nn_dims input_dims = {1, 4, 4, 8}, filter_dims = {4, 1, 1, 8}, output_dims = {1, 4, 4, 4};
    const cmsis_nn_context no_sums = {NULL, 0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_convolve_1x1_s8_short_k(&ctx,
                                                  &no_sums,
                                                  &conv_params,
                                                  &quant_params,
                                                  &input_dims,
                                                  input,
                                                  &filter_dims,
                                                  kernel,
                                                  &bias_dims,
                                                  bias,
                                                  &output_dims,
                                                  output));
#endif
}

void conv_1x1_out_null_weight_sum_arm_convolve_1x1_out_s8(void)
{
    /* arm_convolve_1x1_out_s8() is only compiled on builds with the MVE extension, and always reads
     * weight_sum_ctx->buf. A NULL buf must be diagnosed rather than silently producing garbage output. */
#if defined(ARM_MATH_MVEI)
    enum
    {
        output_channels = 9,
        input_channels = 4,
        kernel_elements = 4,
        input_size = input_channels * kernel_elements
    };
    const int32_t input_offset = 5;
    int8_t input[input_size];
    int8_t kernel[output_channels * input_size];
    int32_t bias[output_channels];
    int32_t multiplier[output_channels];
    int32_t shift[output_channels];
    int8_t output[output_channels];

    for (int i = 0; i < input_size; i++)
    {
        input[i] = (int8_t)((i * 7) % 23 - 11);
    }
    for (int i = 0; i < output_channels; i++)
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
    cmsis_nn_dims filter_dims = {output_channels, 2, 2, input_channels};
    cmsis_nn_dims bias_dims = {1, 1, 1, output_channels};
    cmsis_nn_dims output_dims = {1, 1, 1, output_channels};
    cmsis_nn_conv_params conv_params = {
        .input_offset = input_offset,
        .output_offset = -3,
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
        .activation = {-11, 9},
    };
    cmsis_nn_context ctx;
    /* weight_sum_ctx is left zeroed (buf == NULL) on purpose: this is the precondition the NULL guard exists to
     * diagnose, so every other argument must be entirely valid. */
    cmsis_nn_context weight_sum_ctx = {0};
    cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multiplier,
        .shift = shift,
    };

    /* ctx->buf must be valid: arm_convolve_1x1_out_s8() rejects a NULL one up front, which would mask the
     * weight-sum guard under test. */
    const int32_t buffer_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(buffer_size > 0);
    ctx.buf = malloc(buffer_size);
    TEST_ASSERT_NOT_NULL(ctx.buf);
    ctx.size = buffer_size;

    const arm_cmsis_nn_status result = arm_convolve_1x1_out_s8(&ctx,
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
                                                               output);

    memset(ctx.buf, 0, buffer_size);
    free(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
#endif
}

void conv_1x1_out_buffer_size_arm_convolve_1x1_out_s8(void)
{
    /* Dimension validation is deliberately target-independent, so these hold on every build - including the
     * non-MVE ones, where the sizer returns 0 for valid dims because the kernel does not exist there. */
    const cmsis_nn_dims negative_c_dims = {9, 3, 1, -1};
    const cmsis_nn_dims overflowing_dims = {9, 3, 3, 1 << 28};
    TEST_ASSERT_EQUAL(-1, arm_convolve_1x1_out_s8_get_buffer_size(&negative_c_dims));
    TEST_ASSERT_EQUAL(-1, arm_convolve_1x1_out_s8_get_buffer_size(&overflowing_dims));

    /* arm_convolve_1x1_out_s8() has no sizer of its own until now, so direct callers hand-copied its im2col
     * requirement. Pin the published figure and the opt-in ctx->size check that goes with it. */
#if defined(ARM_MATH_MVEI)
    enum
    {
        output_channels = 9,
        input_channels = 4,
        kernel_elements = 4,
        input_size = input_channels * kernel_elements,
        /* round_up_4(KH * KW * C_IN) = round_up_4(2 * 2 * 4) */
        expected_buffer_size = 16
    };
    const int32_t input_offset = 5;
    int8_t input[input_size];
    int8_t kernel[output_channels * input_size];
    int32_t bias[output_channels];
    int32_t multiplier[output_channels];
    int32_t shift[output_channels];
    int32_t weight_sums[output_channels];
    int8_t output[output_channels];

    for (int i = 0; i < input_size; i++)
    {
        input[i] = (int8_t)((i * 7) % 23 - 11);
    }
    for (int i = 0; i < output_channels; i++)
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
    cmsis_nn_dims filter_dims = {output_channels, 2, 2, input_channels};
    cmsis_nn_dims bias_dims = {1, 1, 1, output_channels};
    cmsis_nn_dims output_dims = {1, 1, 1, output_channels};
    cmsis_nn_conv_params conv_params = {
        .input_offset = input_offset,
        .output_offset = -3,
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
        .activation = {-11, 9},
    };
    cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multiplier,
        .shift = shift,
    };

    const int32_t buffer_size = arm_convolve_1x1_out_s8_get_buffer_size(&filter_dims);
    TEST_ASSERT_EQUAL(expected_buffer_size, buffer_size);

    /* The wrapper sizer must remain a safe upper bound for this kernel, since callers reaching it through
     * arm_convolve_wrapper_s8() size their scratch with that function and never with this one. */
    TEST_ASSERT_TRUE(arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims) >=
                     buffer_size);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_weight_sum(
                          weight_sums, kernel, &input_dims, &filter_dims, &output_dims, input_offset, bias));

    cmsis_nn_context weight_sum_ctx;
    weight_sum_ctx.buf = weight_sums;
    weight_sum_ctx.size = arm_convolve_s8_get_weights_sum_size(&output_dims);

    /* The buffer is always allocated at the full required size. Only the declared ctx->size varies, so a rejection
     * can only come from the size check and never from an actual overrun. */
    cmsis_nn_context ctx;
    ctx.buf = malloc(buffer_size);
    TEST_ASSERT_NOT_NULL(ctx.buf);

    #define CONV_1X1_OUT_RUN()                                                                                         \
        arm_convolve_1x1_out_s8(&ctx,                                                                                  \
                                &weight_sum_ctx,                                                                       \
                                &conv_params,                                                                          \
                                &quant_params,                                                                         \
                                &input_dims,                                                                           \
                                input,                                                                                 \
                                &filter_dims,                                                                          \
                                kernel,                                                                                \
                                &bias_dims,                                                                            \
                                bias,                                                                                  \
                                &output_dims,                                                                          \
                                output)

    /* A declared buffer one byte short of the requirement is rejected. */
    ctx.size = buffer_size - 1;
    const arm_cmsis_nn_status undersized = CONV_1X1_OUT_RUN();

    /* An exactly sized buffer is accepted, so the check is not off by one. */
    ctx.size = buffer_size;
    const arm_cmsis_nn_status exact = CONV_1X1_OUT_RUN();

    /* size == 0 opts out of the check entirely. TFLite Micro and derivatives leave the field unset, so this must
     * keep working. */
    ctx.size = 0;
    const arm_cmsis_nn_status unset = CONV_1X1_OUT_RUN();

    #undef CONV_1X1_OUT_RUN

    memset(ctx.buf, 0, buffer_size);
    free(ctx.buf);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, undersized);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, exact);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, unset);
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

/*
 * The direct entries arm_convolve_s8_small_cin() and arm_convolve_s8_3x3_c16_s1() on their shapes and on out-of-gate
 * neighbours, checked against a scalar reference convolution together with arm_convolve_s8(). Inputs and weights span
 * the full int8 range, the scratch buffer is exactly arm_convolve_s8_get_buffer_size() bytes followed by guard bytes,
 * and the output is surrounded by guard bytes. An entry that declines a layer must write nothing.
 */
typedef struct
{
    int32_t n, in_h, in_w, in_c, k_h, k_w, out_c, stride_y, stride_x, pad_y, pad_x, dil_y, dil_x, out_h, out_w;
    int32_t input_offset, act_min, act_max;
} low_depth_case_t;

#define LOW_DEPTH_GUARD 32
#define LOW_DEPTH_GUARD_VALUE ((int8_t)0x5A)

static uint32_t low_depth_seed;

static int32_t low_depth_rand(void)
{
    low_depth_seed = low_depth_seed * 1664525u + 1013904223u;
    return (int32_t)(low_depth_seed >> 8);
}

static void low_depth_reference(const low_depth_case_t *tc,
                                const int8_t *input,
                                const int8_t *weights,
                                const int32_t *bias,
                                const int32_t *multiplier,
                                const int32_t *shift,
                                int32_t output_offset,
                                int8_t *output)
{
    for (int32_t b = 0; b < tc->n; b++)
    {
        for (int32_t oy = 0; oy < tc->out_h; oy++)
        {
            for (int32_t ox = 0; ox < tc->out_w; ox++)
            {
                for (int32_t oc = 0; oc < tc->out_c; oc++)
                {
                    int32_t acc = bias[oc];
                    for (int32_t ky = 0; ky < tc->k_h; ky++)
                    {
                        const int32_t iy = oy * tc->stride_y - tc->pad_y + ky * tc->dil_y;
                        for (int32_t kx = 0; kx < tc->k_w; kx++)
                        {
                            const int32_t ix = ox * tc->stride_x - tc->pad_x + kx * tc->dil_x;
                            if (iy < 0 || iy >= tc->in_h || ix < 0 || ix >= tc->in_w)
                            {
                                continue;
                            }
                            for (int32_t ic = 0; ic < tc->in_c; ic++)
                            {
                                const int32_t in_val =
                                    input[((b * tc->in_h + iy) * tc->in_w + ix) * tc->in_c + ic] + tc->input_offset;
                                const int32_t w_val = weights[((oc * tc->k_h + ky) * tc->k_w + kx) * tc->in_c + ic];
                                acc += in_val * w_val;
                            }
                        }
                    }
                    int32_t res = arm_nn_requantize(acc, multiplier[oc], shift[oc]) + output_offset;
                    res = ARM_NN_MAX(res, tc->act_min);
                    res = ARM_NN_MIN(res, tc->act_max);
                    output[((b * tc->out_h + oy) * tc->out_w + ox) * tc->out_c + oc] = (int8_t)res;
                }
            }
        }
    }
}

/* The entry expected to take a layer: LOW_DEPTH_GENERAL when neither direct entry does. */
typedef enum
{
    LOW_DEPTH_GENERAL = 0,
    LOW_DEPTH_SMALL_CIN = 1,
    LOW_DEPTH_3X3_C16_S1 = 2
} low_depth_entry_t;

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    #define LOW_DEPTH_DIRECT_AVAILABLE 1
#endif

#define LOW_DEPTH_SCRATCH_FILL ((int8_t)0x33)

typedef arm_cmsis_nn_status (*low_depth_fn_t)(const cmsis_nn_context *,
                                              const cmsis_nn_context *,
                                              const cmsis_nn_conv_params *,
                                              const cmsis_nn_per_channel_quant_params *,
                                              const cmsis_nn_dims *,
                                              const int8_t *,
                                              const cmsis_nn_dims *,
                                              const int8_t *,
                                              const cmsis_nn_dims *,
                                              const int32_t *,
                                              const cmsis_nn_dims *,
                                              const cmsis_nn_dims *,
                                              int8_t *);

static low_depth_fn_t low_depth_fn(low_depth_entry_t entry)
{
    switch (entry)
    {
    case LOW_DEPTH_SMALL_CIN:
        return arm_convolve_s8_small_cin;
    case LOW_DEPTH_3X3_C16_S1:
        return arm_convolve_s8_3x3_c16_s1;
    default:
        return arm_convolve_s8;
    }
}

/* The output and its guard bytes, the whole scratch and its guard bytes all still hold their fill. */
static void
low_depth_expect_untouched(const int8_t *output, int32_t output_size, const int8_t *scratch, int32_t buf_size)
{
    for (int32_t i = 0; i < output_size + 2 * LOW_DEPTH_GUARD; i++)
    {
        TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_GUARD_VALUE, output[i]);
    }
    for (int32_t i = 0; i < buf_size; i++)
    {
        TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_SCRATCH_FILL, scratch[i]);
    }
    for (int32_t i = 0; i < LOW_DEPTH_GUARD; i++)
    {
        TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_GUARD_VALUE, scratch[buf_size + i]);
    }
}

/* Flags of low_depth_check_at(). */
#define LOW_DEPTH_AT_GAP 1
#define LOW_DEPTH_DISTINCT_SCRATCH 2
#define LOW_DEPTH_GENERAL_AT_GAP 4

/* Whether arm_convolve_wrapper_s8() takes its arm_convolve_s8() branch for a layer, where it may run a direct entry. */
static int low_depth_wrapper_calls_conv(const cmsis_nn_conv_params *conv_params,
                                        const cmsis_nn_dims *input_dims,
                                        const cmsis_nn_dims *filter_dims,
                                        const cmsis_nn_dims *output_dims)
{
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims) ||
        arm_nn_is_convolve_1_x_n(conv_params, input_dims, filter_dims))
    {
        return 0;
    }
#if defined(ARM_MATH_MVEI)
    if ((output_dims->h == 1) && (output_dims->w == 1) && (((int64_t)conv_params->stride.w * input_dims->c) % 4 == 0) &&
        (input_dims->c == filter_dims->c))
    {
        return 0;
    }
#else
    (void)output_dims;
#endif
    return 1;
}

/* Runs both direct entries, arm_convolve_s8() and arm_convolve_wrapper_s8() on one layer. arm_convolve_s8(), the
   wrapper and the entry named by entry (on builds that have it) must match the reference; the other entry must return
   ARM_CMSIS_NN_NO_IMPL_ERROR and write nothing. When the wrapper takes its arm_convolve_s8() branch, its scratch size
   must equal arm_convolve_s8_get_buffer_size() and it must leave the scratch as the entry that takes the layer does,
   or as arm_convolve_s8() does for a layer outside both gates. With LOW_DEPTH_DISTINCT_SCRATCH, the scratch the entry
   leaves must also differ from what arm_convolve_s8() leaves, so that this identifies the route. With
   LOW_DEPTH_AT_GAP, where the MPU guard is available, the entry that takes the layer reads the weights from a copy
   that ends at an unmapped MPU gap; with LOW_DEPTH_GENERAL_AT_GAP, arm_convolve_s8() does. */
static void low_depth_check_at(const low_depth_case_t *tc, uint32_t seed, low_depth_entry_t entry, int flags)
{
    const int32_t input_size = tc->n * tc->in_h * tc->in_w * tc->in_c;
    const int32_t rhs_cols = tc->k_h * tc->k_w * tc->in_c;
    const int32_t weights_size = tc->out_c * rhs_cols;
    const int32_t output_size = tc->n * tc->out_h * tc->out_w * tc->out_c;

    int8_t *input = malloc(input_size);
    int8_t *weights = malloc(weights_size);
    int32_t *bias = malloc(tc->out_c * sizeof(int32_t));
    int32_t *multiplier = malloc(tc->out_c * sizeof(int32_t));
    int32_t *shift = malloc(tc->out_c * sizeof(int32_t));
    int32_t *weight_sum = malloc(tc->out_c * sizeof(int32_t));
    int8_t *expected = malloc(output_size);
    int8_t *output = malloc(output_size + 2 * LOW_DEPTH_GUARD);
    TEST_ASSERT_NOT_NULL(input);
    TEST_ASSERT_NOT_NULL(weights);
    TEST_ASSERT_NOT_NULL(bias);
    TEST_ASSERT_NOT_NULL(multiplier);
    TEST_ASSERT_NOT_NULL(shift);
    TEST_ASSERT_NOT_NULL(weight_sum);
    TEST_ASSERT_NOT_NULL(expected);
    TEST_ASSERT_NOT_NULL(output);

    low_depth_seed = seed;
    for (int32_t i = 0; i < input_size; i++)
    {
        input[i] = (int8_t)(low_depth_rand() & 0xFF);
    }
    for (int32_t i = 0; i < weights_size; i++)
    {
        weights[i] = (int8_t)(low_depth_rand() & 0xFF);
    }
    /* Scale so that the typical sum lands inside the int8 range; one channel gets a left shift. */
    int32_t base_shift = -6;
    for (int32_t k = rhs_cols; k > 1; k >>= 2)
    {
        base_shift--;
    }
    for (int32_t i = 0; i < tc->out_c; i++)
    {
        bias[i] = low_depth_rand() % 20001 - 10000;
        multiplier[i] = 1073741824 + low_depth_rand() % 1073741823;
        shift[i] = i == 1 ? 1 : base_shift - low_depth_rand() % 3;
    }
    const int32_t output_offset = low_depth_rand() % 256 - 128;

    const cmsis_nn_dims input_dims = {tc->n, tc->in_h, tc->in_w, tc->in_c};
    const cmsis_nn_dims filter_dims = {tc->out_c, tc->k_h, tc->k_w, tc->in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, tc->out_c};
    const cmsis_nn_dims output_dims = {tc->n, tc->out_h, tc->out_w, tc->out_c};
    const cmsis_nn_conv_params conv_params = {.input_offset = tc->input_offset,
                                              .output_offset = output_offset,
                                              .stride = {tc->stride_x, tc->stride_y},
                                              .padding = {tc->pad_x, tc->pad_y},
                                              .dilation = {tc->dil_x, tc->dil_y},
                                              .activation = {tc->act_min, tc->act_max}};
    const cmsis_nn_per_channel_quant_params quant_params = {multiplier, shift};

    const int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    TEST_ASSERT_TRUE(buf_size >= 0);
    int8_t *scratch = malloc(buf_size + LOW_DEPTH_GUARD);
    int8_t *images = malloc(3 * buf_size + 1);
    TEST_ASSERT_NOT_NULL(scratch);
    TEST_ASSERT_NOT_NULL(images);
    const cmsis_nn_context ctx = {scratch, buf_size};
    const cmsis_nn_context weight_sum_ctx = {weight_sum, tc->out_c * (int32_t)sizeof(int32_t)};
    const arm_cmsis_nn_status sum_status =
        arm_convolve_weight_sum(weight_sum, weights, &input_dims, &filter_dims, &output_dims, tc->input_offset, bias);
#if defined(ARM_MATH_MVEI)
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, sum_status);
#else
    (void)sum_status;
#endif

    low_depth_reference(tc, input, weights, bias, multiplier, shift, output_offset, expected);

    for (int32_t e = LOW_DEPTH_GENERAL; e <= LOW_DEPTH_3X3_C16_S1; e++)
    {
#if defined(LOW_DEPTH_DIRECT_AVAILABLE)
        const int takes = e == LOW_DEPTH_GENERAL || e == (int32_t)entry;
#else
        (void)entry;
        const int takes = e == LOW_DEPTH_GENERAL;
#endif
        memset(scratch, LOW_DEPTH_SCRATCH_FILL, buf_size);
        memset(scratch + buf_size, LOW_DEPTH_GUARD_VALUE, LOW_DEPTH_GUARD);
        memset(output, LOW_DEPTH_GUARD_VALUE, output_size + 2 * LOW_DEPTH_GUARD);
        const int8_t *kernel_weights = weights;
#if defined(MPU_GUARD_AVAILABLE)
        const int at_gap = ((flags & LOW_DEPTH_AT_GAP) && e != LOW_DEPTH_GENERAL && e == (int32_t)entry) ||
            ((flags & LOW_DEPTH_GENERAL_AT_GAP) && e == LOW_DEPTH_GENERAL);
        if (at_gap)
        {
            TEST_ASSERT_TRUE(weights_size <= GUARD_OFFSET);
            kernel_weights = guard_place(weights, (size_t)weights_size);
            guard_gap_enable();
        }
#endif
        const arm_cmsis_nn_status status = low_depth_fn((low_depth_entry_t)e)(&ctx,
                                                                              &weight_sum_ctx,
                                                                              &conv_params,
                                                                              &quant_params,
                                                                              &input_dims,
                                                                              input,
                                                                              &filter_dims,
                                                                              kernel_weights,
                                                                              &bias_dims,
                                                                              bias,
                                                                              NULL,
                                                                              &output_dims,
                                                                              output + LOW_DEPTH_GUARD);
#if defined(MPU_GUARD_AVAILABLE)
        if (at_gap)
        {
            guard_gap_disable();
        }
#endif
        if (takes)
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
            TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output + LOW_DEPTH_GUARD, output_size);
            for (int32_t i = 0; i < LOW_DEPTH_GUARD; i++)
            {
                TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_GUARD_VALUE, output[i]);
                TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_GUARD_VALUE, output[LOW_DEPTH_GUARD + output_size + i]);
                TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_GUARD_VALUE, scratch[buf_size + i]);
            }
        }
        else
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, status);
            low_depth_expect_untouched(output, output_size, scratch, buf_size);
        }
        memcpy(images + e * buf_size, scratch, buf_size);
    }

#if defined(LOW_DEPTH_DIRECT_AVAILABLE)
    const int32_t taker = (int32_t)entry;
#else
    const int32_t taker = LOW_DEPTH_GENERAL;
#endif
    const int32_t wrap_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(wrap_size >= 0);
    int8_t *wrap_scratch = malloc(wrap_size + LOW_DEPTH_GUARD);
    TEST_ASSERT_NOT_NULL(wrap_scratch);
    memset(wrap_scratch, LOW_DEPTH_SCRATCH_FILL, wrap_size);
    memset(wrap_scratch + wrap_size, LOW_DEPTH_GUARD_VALUE, LOW_DEPTH_GUARD);
    memset(output, LOW_DEPTH_GUARD_VALUE, output_size + 2 * LOW_DEPTH_GUARD);
    const cmsis_nn_context wrap_ctx = {wrap_scratch, wrap_size};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_wrapper_s8(&wrap_ctx,
                                              &weight_sum_ctx,
                                              &conv_params,
                                              &quant_params,
                                              &input_dims,
                                              input,
                                              &filter_dims,
                                              weights,
                                              &bias_dims,
                                              bias,
                                              &output_dims,
                                              output + LOW_DEPTH_GUARD));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output + LOW_DEPTH_GUARD, output_size);
    for (int32_t i = 0; i < LOW_DEPTH_GUARD; i++)
    {
        TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_GUARD_VALUE, output[i]);
        TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_GUARD_VALUE, output[LOW_DEPTH_GUARD + output_size + i]);
        TEST_ASSERT_EQUAL_INT8(LOW_DEPTH_GUARD_VALUE, wrap_scratch[wrap_size + i]);
    }
    if (low_depth_wrapper_calls_conv(&conv_params, &input_dims, &filter_dims, &output_dims))
    {
        TEST_ASSERT_EQUAL(buf_size, wrap_size);
        if (buf_size > 0)
        {
            TEST_ASSERT_EQUAL_INT8_ARRAY(images + taker * buf_size, wrap_scratch, buf_size);
        }
        if ((flags & LOW_DEPTH_DISTINCT_SCRATCH) && taker != LOW_DEPTH_GENERAL)
        {
            TEST_ASSERT_TRUE(memcmp(images + LOW_DEPTH_GENERAL * buf_size, images + taker * buf_size, buf_size) != 0);
        }
    }
    else
    {
        TEST_ASSERT_FALSE(flags & LOW_DEPTH_DISTINCT_SCRATCH);
    }

    free(wrap_scratch);
    free(images);
    free(scratch);
    free(output);
    free(expected);
    free(weight_sum);
    free(shift);
    free(multiplier);
    free(bias);
    free(weights);
    free(input);
}

/* n, in_h, in_w, in_c, k_h, k_w, out_c, stride_y, stride_x, pad_y, pad_x, dil_y, dil_x, out_h, out_w, in_off, min, max
 */
static const low_depth_case_t small_cin_cases[] = {
    /* Input depth 1, 2 and 3 with SAME padding. */
    {1, 9, 11, 1, 3, 3, 8, 1, 1, 1, 1, 1, 1, 9, 11, 3, -128, 127},
    {1, 7, 9, 2, 3, 3, 8, 1, 1, 1, 1, 1, 1, 7, 9, -7, -128, 127},
    {1, 8, 7, 3, 3, 3, 8, 1, 1, 1, 1, 1, 1, 8, 7, 128, -128, 127},
    /* K of 16, 32 and 48: the largest K for one, two and three 16-byte chunks. */
    {1, 6, 7, 1, 4, 4, 4, 1, 1, 1, 2, 1, 1, 5, 8, 17, -128, 127},
    {1, 6, 7, 2, 4, 4, 8, 1, 1, 2, 1, 1, 1, 7, 6, -127, -128, 127},
    {1, 7, 6, 3, 4, 4, 12, 1, 1, 1, 1, 1, 1, 6, 5, 127, -100, 90},
    /* K of 17 and 33 (one value past a chunk boundary) with tall kernels, mostly over padding rows. */
    {1, 5, 6, 1, 17, 1, 4, 1, 1, 8, 0, 1, 1, 5, 6, 5, -128, 127},
    {1, 6, 6, 3, 11, 1, 8, 1, 1, 5, 0, 1, 1, 6, 6, 5, -128, 127},
    /* Kernel rows wholly in the padding at the top and bottom (pad_y larger than the kernel). */
    {1, 4, 6, 1, 3, 3, 8, 2, 1, 4, 1, 1, 1, 6, 6, 9, -128, 127},
    {1, 3, 8, 3, 1, 5, 4, 1, 1, 3, 2, 1, 1, 9, 8, -9, -128, 127},
    /* Kernel wider than the input: both edges cut the same kernel row. */
    {1, 5, 4, 2, 3, 8, 4, 1, 1, 1, 4, 1, 1, 5, 5, 128, -128, 127},
    /* Batch of 2, stride 2, output channels 12. */
    {2, 11, 9, 3, 3, 3, 12, 2, 2, 1, 1, 1, 1, 6, 5, 128, -128, 127},
    /* 3x3 input depth 3 stride 2, SAME with the extra padding on the right and bottom edge. */
    {1, 10, 10, 3, 3, 3, 8, 2, 2, 0, 0, 1, 1, 5, 5, 128, -128, 127},
    /* 1x9 input depth 1 stride 2 pad 3 (right edge padded by 4). */
    {1, 1, 40, 1, 1, 9, 16, 1, 2, 0, 3, 1, 1, 1, 20, -24, -128, 127},
    /* Non-3-row kernels in the interior; activation clamps inside the int8 range. */
    {1, 9, 10, 3, 5, 3, 8, 2, 1, 2, 1, 1, 1, 5, 10, 17, -60, 70},
    {1, 20, 19, 1, 5, 5, 4, 1, 1, 2, 2, 1, 1, 20, 19, 3, -128, 127},
    {1, 3, 40, 1, 3, 16, 4, 1, 3, 1, 7, 1, 1, 3, 14, 0, -128, 127},
    /* 1x1 depth 1: output columns fewer than four in the last group (3 pixels). */
    {1, 1, 7, 1, 1, 1, 4, 1, 1, 0, 0, 1, 1, 1, 7, 5, -128, 127},
    /* Kernel rows wholly left of the input (pad_x of 17 bytes) and wholly right of it. */
    {1, 4, 5, 1, 3, 3, 4, 1, 1, 2, 17, 1, 1, 6, 40, 5, -128, 127},
    /* Output larger than the input with no leading padding: kernels wholly below and right of the input. */
    {3, 4, 4, 2, 2, 2, 4, 1, 1, 0, 0, 1, 1, 7, 6, 9, -128, 127},
};

static const low_depth_case_t mlperf_first_layer_cases[] = {
    /* KWS L0: 49x10x1, 10x4 kernel, stride 2, 64 output channels. */
    {1, 49, 10, 1, 10, 4, 64, 2, 2, 4, 1, 1, 1, 25, 5, -83, -128, 127},
    /* VWW L0: 96x96x3, 3x3, stride 2, VALID, 8 output channels. */
    {1, 96, 96, 3, 3, 3, 8, 2, 2, 0, 0, 1, 1, 47, 47, 128, -128, 127},
    /* IC L0: 32x32x3, 3x3, stride 1, SAME, 16 output channels. */
    {1, 32, 32, 3, 3, 3, 16, 1, 1, 1, 1, 1, 1, 32, 32, 128, -128, 127},
    /* heart-arr L1: 1x512x1, 1x9, stride (1, 2), pad 3, 16 output channels. */
    {1, 1, 512, 1, 1, 9, 16, 1, 2, 0, 3, 1, 1, 1, 256, -24, -128, 127},
};

static const low_depth_case_t c16_3x3_cases[] = {
    /* IC L1/L2: 32x32x16, 3x3, stride 1, SAME, 16 output channels. */
    {1, 32, 32, 16, 3, 3, 16, 1, 1, 1, 1, 1, 1, 32, 32, 128, -128, 127},
    /* SAME with 35 output pixels (a tail of 3) and an odd channel count. */
    {1, 5, 7, 16, 3, 3, 5, 1, 1, 1, 1, 1, 1, 5, 7, 128, -50, 100},
    /* VALID, 99 input pixels, 63 output pixels. */
    {1, 9, 11, 16, 3, 3, 12, 1, 1, 0, 0, 1, 1, 7, 9, -3, -128, 127},
    /* Batch of 2, one output channel, 1x2 output (a tail in each batch). */
    {2, 3, 4, 16, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 2, 11, -128, 127},
    /* Input smaller than the kernel: every patch crosses the border. */
    {1, 2, 2, 16, 3, 3, 8, 1, 1, 1, 1, 1, 1, 2, 2, -127, -128, 127},
};

static const low_depth_case_t low_depth_neighbour_cases[] = {
    /* 3x3 over 16 channels with stride 2 (IC L4 style) and with odd sizes. */
    {1, 16, 16, 16, 3, 3, 32, 2, 2, 1, 1, 1, 1, 8, 8, 128, -128, 127},
    {1, 9, 7, 16, 3, 3, 6, 2, 2, 1, 1, 1, 1, 5, 4, 128, -128, 127},
    /* 3x3 over 16 channels with a stride of 2 on one axis only. */
    {1, 7, 9, 16, 3, 3, 8, 1, 2, 1, 1, 1, 1, 7, 5, 128, -128, 127},
    {1, 9, 7, 16, 3, 3, 8, 2, 1, 1, 1, 1, 1, 5, 7, 128, -128, 127},
    /* Input depth 16 with other kernels or dilation. */
    {1, 8, 8, 16, 5, 5, 8, 1, 1, 2, 2, 1, 1, 8, 8, 5, -128, 127},
    {1, 4, 9, 16, 1, 3, 4, 1, 1, 0, 1, 1, 1, 4, 9, 128, -128, 127},
    {1, 9, 9, 16, 3, 3, 8, 1, 1, 2, 1, 2, 1, 9, 9, 128, -128, 127},
    /* Input depth 17 and 4. */
    {1, 8, 8, 17, 3, 3, 16, 1, 1, 1, 1, 1, 1, 8, 8, 128, -128, 127},
    {1, 9, 9, 4, 3, 3, 8, 1, 1, 1, 1, 1, 1, 9, 9, 128, -128, 127},
    /* Input depth 3 with 6 output channels, with dilation 2 and with a 21-byte kernel row. */
    {1, 9, 9, 3, 3, 3, 6, 1, 1, 1, 1, 1, 1, 9, 9, 128, -128, 127},
    {1, 9, 9, 3, 3, 3, 8, 1, 1, 2, 2, 2, 2, 9, 9, 128, -128, 127},
    {1, 12, 12, 3, 7, 7, 8, 2, 2, 3, 3, 1, 1, 6, 6, 128, -128, 127},
    /* Input depth 1 with K = 49. */
    {1, 10, 10, 1, 7, 7, 8, 1, 1, 3, 3, 1, 1, 10, 10, 128, -128, 127},
};

static void low_depth_check(const low_depth_case_t *tc, uint32_t seed, low_depth_entry_t entry)
{
    low_depth_check_at(tc, seed, entry, 0);
}

static void low_depth_check_all(const low_depth_case_t *cases, int32_t count, uint32_t seed, low_depth_entry_t entry)
{
    for (int32_t i = 0; i < count; i++)
    {
        low_depth_check(&cases[i], seed + 97u * (uint32_t)i, entry);
    }
}

void small_cin_arm_convolve_s8(void)
{
    low_depth_check_all(
        small_cin_cases, sizeof(small_cin_cases) / sizeof(small_cin_cases[0]), 11u, LOW_DEPTH_SMALL_CIN);
}

void small_cin_out_ch_arm_convolve_s8(void)
{
    /* Output channel counts 4, 8 and 12 for each input depth, with padding on every side. */
    for (int32_t in_c = 1; in_c <= 3; in_c++)
    {
        for (int32_t out_c = 4; out_c <= 12; out_c += 4)
        {
            const low_depth_case_t tc = {1, 6, 5, in_c, 3, 3, out_c, 1, 1, 1, 1, 1, 1, 6, 5, 7 - in_c, -128, 127};
            low_depth_check(&tc, 23u * (uint32_t)(in_c * 16 + out_c), LOW_DEPTH_SMALL_CIN);
        }
    }
}

void mlperf_first_layers_arm_convolve_s8(void)
{
    low_depth_check_all(mlperf_first_layer_cases,
                        sizeof(mlperf_first_layer_cases) / sizeof(mlperf_first_layer_cases[0]),
                        31u,
                        LOW_DEPTH_SMALL_CIN);
}

#if defined(MPU_GUARD_AVAILABLE)
/* Weights that end at an unmapped MPU gap, on arm_convolve_s8_small_cin(): 1 to 48 filter values per output channel,
   across the 8 values where the GEMM starts loading its first three filters' last chunks whole, and every K-chunk
   count. A load past the last filter faults. general_weights_at_gap_arm_convolve_s8() runs the same layers through
   arm_convolve_s8() with its weights at the gap. */
static const low_depth_case_t small_cin_weights_at_gap_cases[] = {
    {1, 3, 5, 1, 1, 1, 4, 1, 1, 0, 0, 1, 1, 3, 5, 3, -128, 127},
    {1, 3, 5, 2, 1, 1, 4, 1, 1, 0, 0, 1, 1, 3, 5, -9, -128, 127},
    {1, 3, 5, 3, 1, 1, 8, 1, 1, 0, 0, 1, 1, 3, 5, 128, -128, 127},
    {1, 4, 5, 1, 2, 2, 4, 1, 1, 0, 0, 1, 1, 3, 4, 7, -128, 127},
    {1, 3, 6, 3, 1, 2, 4, 1, 1, 0, 0, 1, 1, 3, 5, 128, -128, 127},
    {1, 4, 5, 1, 1, 5, 4, 1, 1, 0, 2, 1, 1, 4, 5, 11, -128, 127},
    {1, 3, 9, 1, 1, 7, 4, 1, 1, 0, 3, 1, 1, 3, 9, -5, -128, 127},
    {1, 4, 10, 1, 1, 8, 4, 1, 1, 0, 0, 1, 1, 4, 3, 5, -128, 127},
    {1, 6, 6, 1, 3, 3, 4, 1, 1, 1, 1, 1, 1, 6, 6, 17, -128, 127},
    {1, 4, 8, 3, 1, 5, 8, 1, 1, 0, 2, 1, 1, 4, 8, 128, -128, 127},
    {1, 6, 6, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 5, 5, -3, -128, 127},
    {1, 5, 6, 1, 17, 1, 4, 1, 1, 8, 0, 1, 1, 5, 6, 5, -128, 127},
    {1, 5, 5, 2, 3, 3, 4, 1, 1, 1, 1, 1, 1, 5, 5, 9, -128, 127},
    {1, 9, 9, 3, 3, 3, 8, 2, 2, 1, 1, 1, 1, 5, 5, 128, -128, 127},
    {1, 6, 6, 2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 5, 5, -7, -128, 127},
    {1, 12, 4, 3, 11, 1, 4, 1, 1, 5, 0, 1, 1, 12, 4, 128, -128, 127},
    {1, 20, 10, 1, 10, 4, 8, 2, 2, 4, 1, 1, 1, 10, 5, -83, -128, 127},
    {1, 6, 6, 3, 4, 4, 4, 1, 1, 1, 1, 1, 1, 5, 5, 128, -128, 127},
};
#endif

void small_cin_weights_at_gap_arm_convolve_s8(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    for (size_t i = 0; i < sizeof(small_cin_weights_at_gap_cases) / sizeof(small_cin_weights_at_gap_cases[0]); i++)
    {
        low_depth_check_at(
            &small_cin_weights_at_gap_cases[i], 61u + 97u * (uint32_t)i, LOW_DEPTH_SMALL_CIN, LOW_DEPTH_AT_GAP);
    }
#endif
}

/* The same layers through arm_convolve_s8() with the weights ending at the gap: its GEMM must not read past the last
   filter either (#587). */
void general_weights_at_gap_arm_convolve_s8(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    for (size_t i = 0; i < sizeof(small_cin_weights_at_gap_cases) / sizeof(small_cin_weights_at_gap_cases[0]); i++)
    {
        low_depth_check_at(
            &small_cin_weights_at_gap_cases[i], 61u + 97u * (uint32_t)i, LOW_DEPTH_SMALL_CIN, LOW_DEPTH_GENERAL_AT_GAP);
    }
#endif
}

void c16_3x3_arm_convolve_s8(void)
{
    low_depth_check_all(c16_3x3_cases, sizeof(c16_3x3_cases) / sizeof(c16_3x3_cases[0]), 41u, LOW_DEPTH_3X3_C16_S1);
}

void low_depth_neighbours_arm_convolve_s8(void)
{
    low_depth_check_all(low_depth_neighbour_cases,
                        sizeof(low_depth_neighbour_cases) / sizeof(low_depth_neighbour_cases[0]),
                        53u,
                        LOW_DEPTH_GENERAL);
}

/* One layer given to a direct entry with zeroed tensors: the status must be expect and nothing may be written.
   filter_c is CK, upscale passes a 2x2 upscale_dims, and null_buf and null_sums pass a NULL ctx->buf or
   weight_sum_ctx->buf. */
static void low_depth_expect_status(low_depth_entry_t entry,
                                    const low_depth_case_t *tc,
                                    int32_t filter_c,
                                    int upscale,
                                    int null_buf,
                                    int null_sums,
                                    arm_cmsis_nn_status expect)
{
    static int8_t input[2048];
    static int8_t weights[2048];
    static int8_t output[1024 + 2 * LOW_DEPTH_GUARD];
    static int8_t scratch[1536 + LOW_DEPTH_GUARD];
    static int32_t zeros[64];
    const cmsis_nn_dims input_dims = {tc->n, tc->in_h, tc->in_w, tc->in_c};
    const cmsis_nn_dims filter_dims = {tc->out_c, tc->k_h, tc->k_w, filter_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, tc->out_c};
    const cmsis_nn_dims output_dims = {tc->n, tc->out_h, tc->out_w, tc->out_c};
    const cmsis_nn_dims upscale_dims = {1, 2, 2, 1};
    const cmsis_nn_conv_params conv_params = {.input_offset = tc->input_offset,
                                              .output_offset = 0,
                                              .stride = {tc->stride_x, tc->stride_y},
                                              .padding = {tc->pad_x, tc->pad_y},
                                              .dilation = {tc->dil_x, tc->dil_y},
                                              .activation = {tc->act_min, tc->act_max}};
    const cmsis_nn_per_channel_quant_params quant_params = {zeros, zeros};
    const int32_t output_size = tc->n * tc->out_h * tc->out_w * tc->out_c;
    const int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    TEST_ASSERT_TRUE(tc->n * tc->in_h * tc->in_w * tc->in_c <= (int32_t)sizeof(input));
    TEST_ASSERT_TRUE(tc->out_c * tc->k_h * tc->k_w * filter_c <= (int32_t)sizeof(weights));
    TEST_ASSERT_TRUE(output_size <= 1024);
    TEST_ASSERT_TRUE(tc->out_c <= 64);
    TEST_ASSERT_TRUE(buf_size >= 0 && buf_size <= 1536);

    memset(scratch, LOW_DEPTH_SCRATCH_FILL, buf_size);
    memset(scratch + buf_size, LOW_DEPTH_GUARD_VALUE, LOW_DEPTH_GUARD);
    memset(output, LOW_DEPTH_GUARD_VALUE, output_size + 2 * LOW_DEPTH_GUARD);
    const cmsis_nn_context ctx = {null_buf ? NULL : scratch, buf_size};
    const cmsis_nn_context weight_sum_ctx = {null_sums ? NULL : zeros, (int32_t)sizeof(zeros)};
    const arm_cmsis_nn_status status = low_depth_fn(entry)(&ctx,
                                                           &weight_sum_ctx,
                                                           &conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input,
                                                           &filter_dims,
                                                           weights,
                                                           &bias_dims,
                                                           zeros,
                                                           upscale ? &upscale_dims : NULL,
                                                           &output_dims,
                                                           output + LOW_DEPTH_GUARD);
    TEST_ASSERT_EQUAL(expect, status);
    low_depth_expect_untouched(output, output_size, scratch, buf_size);
}

void small_cin_gate_declines_arm_convolve_s8(void)
{
    /* One gate condition broken at a time, from an in-gate layer. */
    const low_depth_case_t in_gate = {1, 6, 6, 3, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127};
    const struct
    {
        low_depth_case_t tc;
        int32_t filter_c;
        int upscale;
    } declined[] = {
        /* upscale_dims given */
        {{1, 6, 6, 3, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127}, 3, 1},
        /* input depth 4 and 0 */
        {{1, 6, 6, 4, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127}, 4, 0},
        {{1, 6, 6, 0, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127}, 0, 0},
        /* two groups: CK 1 against input depth 2 */
        {{1, 6, 6, 2, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127}, 1, 0},
        /* dilation 2 in x, then in y */
        {{1, 6, 6, 3, 3, 3, 8, 1, 1, 2, 2, 1, 2, 6, 6, 128, -128, 127}, 3, 0},
        {{1, 6, 6, 3, 3, 3, 8, 1, 1, 2, 2, 2, 1, 6, 6, 128, -128, 127}, 3, 0},
        /* kernel width 0, then height 0 */
        {{1, 6, 6, 3, 3, 0, 8, 1, 1, 0, 0, 1, 1, 4, 6, 128, -128, 127}, 3, 0},
        {{1, 6, 6, 3, 0, 3, 8, 1, 1, 0, 0, 1, 1, 6, 4, 128, -128, 127}, 3, 0},
        /* a kernel row of 17 and 18 values with at most 48 in the filter */
        {{1, 2, 20, 1, 1, 17, 4, 1, 1, 0, 0, 1, 1, 2, 4, 5, -128, 127}, 1, 0},
        {{1, 2, 8, 3, 1, 6, 4, 1, 1, 0, 0, 1, 1, 2, 3, 5, -128, 127}, 3, 0},
        /* 49 filter values with a row of 7 */
        {{1, 8, 8, 1, 7, 7, 8, 1, 1, 3, 3, 1, 1, 8, 8, 5, -128, 127}, 1, 0},
        /* 0, 2 and 6 output channels */
        {{1, 6, 6, 3, 3, 3, 0, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127}, 3, 0},
        {{1, 6, 6, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127}, 3, 0},
        {{1, 6, 6, 3, 3, 3, 6, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127}, 3, 0},
    };
    for (size_t i = 0; i < sizeof(declined) / sizeof(declined[0]); i++)
    {
        low_depth_expect_status(LOW_DEPTH_SMALL_CIN,
                                &declined[i].tc,
                                declined[i].filter_c,
                                declined[i].upscale,
                                0,
                                0,
                                ARM_CMSIS_NN_NO_IMPL_ERROR);
    }
    /* Kernel dimensions whose width x depth or width x height x depth products leave int32_t are declined. */
    {
        const cmsis_nn_conv_params conv_params = {.dilation = {1, 1}};
        const cmsis_nn_dims input_dims = {1, 6, 6, 3};
        const cmsis_nn_dims output_dims = {1, 6, 6, 8};
        const cmsis_nn_dims wide_filter = {8, 1, INT32_MAX, 3};
        const cmsis_nn_dims tall_filter = {8, INT32_MAX, 5, 3};
        TEST_ASSERT_EQUAL(0,
                          arm_nn_is_convolve_s8_small_cin(&conv_params, &input_dims, &wide_filter, &output_dims, NULL));
        TEST_ASSERT_EQUAL(0,
                          arm_nn_is_convolve_s8_small_cin(&conv_params, &input_dims, &tall_filter, &output_dims, NULL));
    }
    /* The in-gate layer itself is taken, where the entry exists. */
    low_depth_check(&in_gate, 71u, LOW_DEPTH_SMALL_CIN);
}

void c16_3x3_gate_declines_arm_convolve_s8(void)
{
    const low_depth_case_t in_gate = {1, 5, 5, 16, 3, 3, 8, 1, 1, 1, 1, 1, 1, 5, 5, 128, -128, 127};
    const struct
    {
        low_depth_case_t tc;
        int32_t filter_c;
        int upscale;
    } declined[] = {
        /* upscale_dims given */
        {{1, 5, 5, 16, 3, 3, 8, 1, 1, 1, 1, 1, 1, 5, 5, 128, -128, 127}, 16, 1},
        /* two groups: input depth 32 with CK 16, and input depth 16 with CK 8 */
        {{1, 5, 5, 32, 3, 3, 8, 1, 1, 1, 1, 1, 1, 5, 5, 128, -128, 127}, 16, 0},
        {{1, 5, 5, 16, 3, 3, 8, 1, 1, 1, 1, 1, 1, 5, 5, 128, -128, 127}, 8, 0},
        /* depth 15 and 17 */
        {{1, 5, 5, 15, 3, 3, 8, 1, 1, 1, 1, 1, 1, 5, 5, 128, -128, 127}, 15, 0},
        {{1, 5, 5, 17, 3, 3, 8, 1, 1, 1, 1, 1, 1, 5, 5, 128, -128, 127}, 17, 0},
        /* kernel width 2 and 4, height 2 and 4 */
        {{1, 5, 5, 16, 3, 2, 8, 1, 1, 1, 0, 1, 1, 5, 4, 128, -128, 127}, 16, 0},
        {{1, 5, 5, 16, 3, 4, 8, 1, 1, 1, 1, 1, 1, 5, 4, 128, -128, 127}, 16, 0},
        {{1, 5, 5, 16, 2, 3, 8, 1, 1, 0, 1, 1, 1, 4, 5, 128, -128, 127}, 16, 0},
        {{1, 5, 5, 16, 4, 3, 8, 1, 1, 1, 1, 1, 1, 4, 5, 128, -128, 127}, 16, 0},
        /* stride 2 in x, then in y */
        {{1, 5, 5, 16, 3, 3, 8, 1, 2, 1, 1, 1, 1, 5, 3, 128, -128, 127}, 16, 0},
        {{1, 5, 5, 16, 3, 3, 8, 2, 1, 1, 1, 1, 1, 3, 5, 128, -128, 127}, 16, 0},
        /* dilation 2 in x, then in y */
        {{1, 5, 5, 16, 3, 3, 8, 1, 1, 1, 2, 1, 2, 5, 5, 128, -128, 127}, 16, 0},
        {{1, 5, 5, 16, 3, 3, 8, 1, 1, 2, 1, 2, 1, 5, 5, 128, -128, 127}, 16, 0},
    };
    for (size_t i = 0; i < sizeof(declined) / sizeof(declined[0]); i++)
    {
        low_depth_expect_status(LOW_DEPTH_3X3_C16_S1,
                                &declined[i].tc,
                                declined[i].filter_c,
                                declined[i].upscale,
                                0,
                                0,
                                ARM_CMSIS_NN_NO_IMPL_ERROR);
    }
    low_depth_check(&in_gate, 73u, LOW_DEPTH_3X3_C16_S1);
}

void low_depth_arg_errors_arm_convolve_s8(void)
{
    /* The argument errors of arm_convolve_s8() come first, for layers in and out of the gate. */
    const low_depth_case_t small = {1, 6, 6, 3, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127};
    const low_depth_case_t c16 = {1, 6, 6, 16, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127};
    const low_depth_case_t outside = {1, 6, 6, 4, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127};
    /* groups = 5 / 2 = 2 does not divide input depth 5; groups = 4 / 2 = 2 does not divide 3 output channels */
    const low_depth_case_t bad_in_groups = {1, 6, 6, 5, 3, 3, 8, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127};
    const low_depth_case_t bad_out_groups = {1, 6, 6, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 6, 6, 128, -128, 127};
#if defined(ARM_MATH_MVEI)
    const arm_cmsis_nn_status null_sums_status = ARM_CMSIS_NN_ARG_ERROR;
#else
    const arm_cmsis_nn_status null_sums_status = ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
    for (int32_t e = LOW_DEPTH_SMALL_CIN; e <= LOW_DEPTH_3X3_C16_S1; e++)
    {
        const low_depth_entry_t entry = (low_depth_entry_t)e;
        const low_depth_case_t *own = entry == LOW_DEPTH_SMALL_CIN ? &small : &c16;
        low_depth_expect_status(entry, own, own->in_c, 0, 1, 0, ARM_CMSIS_NN_ARG_ERROR);
        low_depth_expect_status(entry, &outside, 4, 0, 1, 0, ARM_CMSIS_NN_ARG_ERROR);
        low_depth_expect_status(entry, own, own->in_c, 0, 0, 1, null_sums_status);
        low_depth_expect_status(entry, &outside, 4, 0, 0, 1, null_sums_status);
        low_depth_expect_status(entry, &bad_in_groups, 2, 0, 0, 0, ARM_CMSIS_NN_ARG_ERROR);
        low_depth_expect_status(entry, &bad_out_groups, 2, 0, 0, 0, ARM_CMSIS_NN_ARG_ERROR);
    }
}

void wrapper_route_arm_convolve_s8(void)
{
    /* Layers on the arm_convolve_s8() branch of arm_convolve_wrapper_s8() whose entry leaves the scratch unlike
       arm_convolve_s8() does, so the scratch shows which one the wrapper ran: 27, 40 and 9 filter values (the column
       tails the small input-depth path zeroes lie past the general path's columns), and a 16-channel 3x3 layer with
       every patch inside the input (the path reads them in place and writes no scratch). */
    const struct
    {
        low_depth_case_t tc;
        low_depth_entry_t entry;
    } routed[] = {
        {{1, 8, 7, 3, 3, 3, 8, 1, 1, 1, 1, 1, 1, 8, 7, 128, -128, 127}, LOW_DEPTH_SMALL_CIN},
        {{1, 20, 10, 1, 10, 4, 8, 2, 2, 4, 1, 1, 1, 10, 5, -83, -128, 127}, LOW_DEPTH_SMALL_CIN},
        {{1, 1, 40, 1, 1, 9, 16, 1, 2, 0, 3, 1, 1, 1, 20, -24, -128, 127}, LOW_DEPTH_SMALL_CIN},
        {{1, 9, 11, 16, 3, 3, 12, 1, 1, 0, 0, 1, 1, 7, 9, -3, -128, 127}, LOW_DEPTH_3X3_C16_S1},
    };
    for (size_t i = 0; i < sizeof(routed) / sizeof(routed[0]); i++)
    {
        low_depth_check_at(&routed[i].tc, 83u + 97u * (uint32_t)i, routed[i].entry, LOW_DEPTH_DISTINCT_SCRATCH);
    }
}

/* arm_convolve_1x1_out_s8() with its input, then its weights, then its scratch (which holds the matrix kernel's left
   operand) ending at an unmapped MPU gap: neither the im2col copy nor the MVE matrix kernel may load past them (#605).
   The output must match arm_convolve_s8() on the same layer with everything in place. */
void conv_1x1_out_operands_at_gap_arm_convolve_1x1_out_s8(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    enum
    {
        out_ch = 9,
        in_ch = 16,
        taps = 4,
        input_size = in_ch * taps
    };
    int8_t input[input_size];
    int8_t kernel[out_ch * input_size];
    int32_t bias[out_ch];
    int32_t multiplier[out_ch];
    int32_t shift[out_ch];
    int32_t weight_sum[out_ch + 4];
    int8_t expected[out_ch];
    int8_t output[out_ch];
    for (int i = 0; i < input_size; i++)
    {
        input[i] = (int8_t)((i * 7) % 23 - 11);
    }
    for (int i = 0; i < out_ch; i++)
    {
        bias[i] = i * 37 - 100;
        multiplier[i] = (i % 2 == 0) ? (1 << 30) : (3 << 28);
        shift[i] = (i % 3) - 3;
        for (int j = 0; j < input_size; j++)
        {
            kernel[i * input_size + j] = (int8_t)((i * 5 + j * 3) % 17 - 8);
        }
    }
    cmsis_nn_dims input_dims = {1, 2, 2, in_ch};
    cmsis_nn_dims filter_dims = {out_ch, 2, 2, in_ch};
    cmsis_nn_dims bias_dims = {1, 1, 1, out_ch};
    cmsis_nn_dims output_dims = {1, 1, 1, out_ch};
    cmsis_nn_conv_params conv_params = {
        .input_offset = 5,
        .output_offset = -3,
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
        .activation = {-128, 127},
    };
    cmsis_nn_per_channel_quant_params quant_params = {.multiplier = multiplier, .shift = shift};
    const int32_t buffer_size =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims) +
        arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    cmsis_nn_context ctx = {malloc(buffer_size), buffer_size};
    cmsis_nn_context weight_sum_ctx = {weight_sum, (int32_t)sizeof(weight_sum)};
    TEST_ASSERT_NOT_NULL(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_weight_sum(
                          weight_sum, kernel, &input_dims, &filter_dims, &output_dims, conv_params.input_offset, bias));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_s8(&ctx,
                                      &weight_sum_ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input,
                                      &filter_dims,
                                      kernel,
                                      &bias_dims,
                                      bias,
                                      NULL,
                                      &output_dims,
                                      expected));
    const int32_t scratch_size = arm_convolve_1x1_out_s8_get_buffer_size(&filter_dims);
    TEST_ASSERT_TRUE(scratch_size > 0 && scratch_size <= GUARD_OFFSET);
    for (int at_gap = 0; at_gap < 3; at_gap++)
    {
        const int8_t *in = at_gap == 0 ? guard_place(input, sizeof(input)) : input;
        const int8_t *w = at_gap == 1 ? guard_place(kernel, sizeof(kernel)) : kernel;
        const cmsis_nn_context scratch_ctx = {guard_end((size_t)scratch_size), scratch_size};
        memset(output, 0x55, sizeof(output));
        guard_gap_enable();
        const arm_cmsis_nn_status result = arm_convolve_1x1_out_s8(at_gap == 2 ? &scratch_ctx : &ctx,
                                                                   &weight_sum_ctx,
                                                                   &conv_params,
                                                                   &quant_params,
                                                                   &input_dims,
                                                                   in,
                                                                   &filter_dims,
                                                                   w,
                                                                   &bias_dims,
                                                                   bias,
                                                                   &output_dims,
                                                                   output);
        guard_gap_disable();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, out_ch);
    }
    memset(ctx.buf, 0, buffer_size);
    free(ctx.buf);
#endif
}
