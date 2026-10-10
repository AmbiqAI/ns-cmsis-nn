/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com> All rights
 * reserved.
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
#include <unity.h>

#include "../Utils/mpu_guard.h"
#include "../TestData/int16xint8_kernel_less_than_9/test_data.h"
#include "../TestData/int16xint8_1x1_ns_np_nd/test_data.h"
#include "../TestData/int16xint8_group_batch2_dilated/test_data.h"
#include "../TestData/int16xint8_group_depthwise/test_data.h"
#include "../TestData/int16xint8_group_depthwise_3x3/test_data.h"
#include "../TestData/int16xint8_group_depthwise_3x3_pad/test_data.h"
#include "../TestData/int16xint8_group_depthwise_3x3_stride_dilation/test_data.h"
#include "../TestData/int16xint8_group2/test_data.h"
#include "../TestData/int16xint8_group_same/test_data.h"
#include "../TestData/int16xint8/test_data.h"
#include "../TestData/int16xint8_dilation_1/test_data.h"
#include "../TestData/int16xint8_dilation_2/test_data.h"
#include "../TestData/int16xint8_dilation_3/test_data.h"
#include "../TestData/int16xint8_spill/test_data.h"
#include "../TestData/int16xint8_spill2/test_data.h"
#include "../TestData/int16xint8xint32_1/test_data.h"
#include "../TestData/int16xint8xint32_2/test_data.h"
#include "../TestData/int16xint8xint32_3/test_data.h"
#include "../TestData/int16xint8xint32_4/test_data.h"
#include "../TestData/int16xint8xint32_5/test_data.h"
#include "../TestData/int16xint8xint32_6/test_data.h"
#include "../TestData/requantize_s64/test_data.h"
#include "../Utils/validate.h"

void int16xint8_1x1_ns_np_nd_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_1X1_NS_NP_ND_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_1x1_ns_np_nd_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_1x1_ns_np_nd_weights;
    const int16_t *input_data = int16xint8_1x1_ns_np_nd_input;
    const int16_t *output_ref = int16xint8_1x1_ns_np_nd_output_ref;
    const int32_t output_ref_size = INT16XINT8_1X1_NS_NP_ND_DST_SIZE;

    input_dims.n = INT16XINT8_1X1_NS_NP_ND_INPUT_BATCHES;
    input_dims.w = INT16XINT8_1X1_NS_NP_ND_INPUT_W;
    input_dims.h = INT16XINT8_1X1_NS_NP_ND_INPUT_H;
    input_dims.c = INT16XINT8_1X1_NS_NP_ND_IN_CH;
    filter_dims.w = INT16XINT8_1X1_NS_NP_ND_FILTER_X;
    filter_dims.h = INT16XINT8_1X1_NS_NP_ND_FILTER_Y;
    filter_dims.c = INT16XINT8_1X1_NS_NP_ND_IN_CH;
    output_dims.w = INT16XINT8_1X1_NS_NP_ND_OUTPUT_W;
    output_dims.h = INT16XINT8_1X1_NS_NP_ND_OUTPUT_H;
    output_dims.c = INT16XINT8_1X1_NS_NP_ND_OUT_CH;

    conv_params.padding.w = INT16XINT8_1X1_NS_NP_ND_PAD_X;
    conv_params.padding.h = INT16XINT8_1X1_NS_NP_ND_PAD_Y;
    conv_params.stride.w = INT16XINT8_1X1_NS_NP_ND_STRIDE_X;
    conv_params.stride.h = INT16XINT8_1X1_NS_NP_ND_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_1X1_NS_NP_ND_DILATION_X;
    conv_params.dilation.h = INT16XINT8_1X1_NS_NP_ND_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_1X1_NS_NP_ND_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_1X1_NS_NP_ND_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_1x1_ns_np_nd_output_mult;
    quant_params.shift = (int32_t *)int16xint8_1x1_ns_np_nd_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE_MESSAGE(validate_s16(output, output_ref, output_ref_size),
                             "int16xint8_1x1_ns_np_nd_arm_convolve_s16 direct mismatch");
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_kernel_less_than_9_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_KERNEL_LESS_THAN_9_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_kernel_less_than_9_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_kernel_less_than_9_weights;
    const int16_t *input_data = int16xint8_kernel_less_than_9_input;
    const int16_t *output_ref = int16xint8_kernel_less_than_9_output_ref;
    const int32_t output_ref_size = INT16XINT8_KERNEL_LESS_THAN_9_DST_SIZE;

    input_dims.n = INT16XINT8_KERNEL_LESS_THAN_9_INPUT_BATCHES;
    input_dims.w = INT16XINT8_KERNEL_LESS_THAN_9_INPUT_W;
    input_dims.h = INT16XINT8_KERNEL_LESS_THAN_9_INPUT_H;
    input_dims.c = INT16XINT8_KERNEL_LESS_THAN_9_IN_CH;
    filter_dims.w = INT16XINT8_KERNEL_LESS_THAN_9_FILTER_X;
    filter_dims.h = INT16XINT8_KERNEL_LESS_THAN_9_FILTER_Y;
    filter_dims.c = INT16XINT8_KERNEL_LESS_THAN_9_FILTER_CH;
    output_dims.w = INT16XINT8_KERNEL_LESS_THAN_9_OUTPUT_W;
    output_dims.h = INT16XINT8_KERNEL_LESS_THAN_9_OUTPUT_H;
    output_dims.c = INT16XINT8_KERNEL_LESS_THAN_9_OUT_CH;

    conv_params.padding.w = INT16XINT8_KERNEL_LESS_THAN_9_PAD_X;
    conv_params.padding.h = INT16XINT8_KERNEL_LESS_THAN_9_PAD_Y;
    conv_params.stride.w = INT16XINT8_KERNEL_LESS_THAN_9_STRIDE_X;
    conv_params.stride.h = INT16XINT8_KERNEL_LESS_THAN_9_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_KERNEL_LESS_THAN_9_DILATION_X;
    conv_params.dilation.h = INT16XINT8_KERNEL_LESS_THAN_9_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_KERNEL_LESS_THAN_9_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_KERNEL_LESS_THAN_9_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_kernel_less_than_9_output_mult;
    quant_params.shift = (int32_t *)int16xint8_kernel_less_than_9_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_group_depthwise_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_GROUP_DEPTHWISE_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_group_depthwise_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_group_depthwise_weights;
    const int16_t *input_data = int16xint8_group_depthwise_input;
    const int16_t *output_ref = int16xint8_group_depthwise_output_ref;
    const int32_t output_ref_size = INT16XINT8_GROUP_DEPTHWISE_DST_SIZE;

    input_dims.n = INT16XINT8_GROUP_DEPTHWISE_INPUT_BATCHES;
    input_dims.w = INT16XINT8_GROUP_DEPTHWISE_INPUT_W;
    input_dims.h = INT16XINT8_GROUP_DEPTHWISE_INPUT_H;
    input_dims.c = INT16XINT8_GROUP_DEPTHWISE_IN_CH;
    filter_dims.w = INT16XINT8_GROUP_DEPTHWISE_FILTER_X;
    filter_dims.h = INT16XINT8_GROUP_DEPTHWISE_FILTER_Y;
    filter_dims.c = INT16XINT8_GROUP_DEPTHWISE_FILTER_CH;
    output_dims.w = INT16XINT8_GROUP_DEPTHWISE_OUTPUT_W;
    output_dims.h = INT16XINT8_GROUP_DEPTHWISE_OUTPUT_H;
    output_dims.c = INT16XINT8_GROUP_DEPTHWISE_OUT_CH;

    conv_params.padding.w = INT16XINT8_GROUP_DEPTHWISE_PAD_X;
    conv_params.padding.h = INT16XINT8_GROUP_DEPTHWISE_PAD_Y;
    conv_params.stride.w = INT16XINT8_GROUP_DEPTHWISE_STRIDE_X;
    conv_params.stride.h = INT16XINT8_GROUP_DEPTHWISE_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_GROUP_DEPTHWISE_DILATION_X;
    conv_params.dilation.h = INT16XINT8_GROUP_DEPTHWISE_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_GROUP_DEPTHWISE_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_GROUP_DEPTHWISE_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_group_depthwise_output_mult;
    quant_params.shift = (int32_t *)int16xint8_group_depthwise_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_group_depthwise_3x3_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_GROUP_DEPTHWISE_3X3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_group_depthwise_3x3_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_group_depthwise_3x3_weights;
    const int16_t *input_data = int16xint8_group_depthwise_3x3_input;
    const int16_t *output_ref = int16xint8_group_depthwise_3x3_output_ref;
    const int32_t output_ref_size = INT16XINT8_GROUP_DEPTHWISE_3X3_DST_SIZE;

    input_dims.n = INT16XINT8_GROUP_DEPTHWISE_3X3_INPUT_BATCHES;
    input_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_INPUT_W;
    input_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_INPUT_H;
    input_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_IN_CH;
    filter_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_FILTER_X;
    filter_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_FILTER_Y;
    filter_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_FILTER_CH;
    output_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_OUTPUT_W;
    output_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_OUTPUT_H;
    output_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_OUT_CH;

    conv_params.padding.w = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_X;
    conv_params.padding.h = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_Y;
    conv_params.stride.w = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_X;
    conv_params.stride.h = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_GROUP_DEPTHWISE_3X3_DILATION_X;
    conv_params.dilation.h = INT16XINT8_GROUP_DEPTHWISE_3X3_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_GROUP_DEPTHWISE_3X3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_GROUP_DEPTHWISE_3X3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_group_depthwise_3x3_output_mult;
    quant_params.shift = (int32_t *)int16xint8_group_depthwise_3x3_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_group_depthwise_3x3_pad_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_group_depthwise_3x3_pad_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_group_depthwise_3x3_pad_weights;
    const int16_t *input_data = int16xint8_group_depthwise_3x3_pad_input;
    const int16_t *output_ref = int16xint8_group_depthwise_3x3_pad_output_ref;
    const int32_t output_ref_size = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_DST_SIZE;

    input_dims.n = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_INPUT_BATCHES;
    input_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_INPUT_W;
    input_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_INPUT_H;
    input_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_IN_CH;
    filter_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_FILTER_X;
    filter_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_FILTER_Y;
    filter_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_FILTER_CH;
    output_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_OUTPUT_W;
    output_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_OUTPUT_H;
    output_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_OUT_CH;

    conv_params.padding.w = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_PAD_X;
    conv_params.padding.h = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_PAD_Y;
    conv_params.stride.w = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_STRIDE_X;
    conv_params.stride.h = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_DILATION_X;
    conv_params.dilation.h = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_GROUP_DEPTHWISE_3X3_PAD_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_group_depthwise_3x3_pad_output_mult;
    quant_params.shift = (int32_t *)int16xint8_group_depthwise_3x3_pad_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_group_depthwise_3x3_stride_dilation_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_group_depthwise_3x3_stride_dilation_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_group_depthwise_3x3_stride_dilation_weights;
    const int16_t *input_data = int16xint8_group_depthwise_3x3_stride_dilation_input;
    const int16_t *output_ref = int16xint8_group_depthwise_3x3_stride_dilation_output_ref;
    const int32_t output_ref_size = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_DST_SIZE;

    input_dims.n = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_INPUT_BATCHES;
    input_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_INPUT_W;
    input_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_INPUT_H;
    input_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_IN_CH;
    filter_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_FILTER_X;
    filter_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_FILTER_Y;
    filter_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_FILTER_CH;
    output_dims.w = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_OUTPUT_W;
    output_dims.h = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_OUTPUT_H;
    output_dims.c = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_OUT_CH;

    conv_params.padding.w = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_PAD_X;
    conv_params.padding.h = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_PAD_Y;
    conv_params.stride.w = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_STRIDE_X;
    conv_params.stride.h = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_DILATION_X;
    conv_params.dilation.h = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_GROUP_DEPTHWISE_3X3_STRIDE_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_group_depthwise_3x3_stride_dilation_output_mult;
    quant_params.shift = (int32_t *)int16xint8_group_depthwise_3x3_stride_dilation_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_group2_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_GROUP2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_group2_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_group2_weights;
    const int16_t *input_data = int16xint8_group2_input;
    const int16_t *output_ref = int16xint8_group2_output_ref;
    const int32_t output_ref_size = INT16XINT8_GROUP2_DST_SIZE;

    input_dims.n = INT16XINT8_GROUP2_INPUT_BATCHES;
    input_dims.w = INT16XINT8_GROUP2_INPUT_W;
    input_dims.h = INT16XINT8_GROUP2_INPUT_H;
    input_dims.c = INT16XINT8_GROUP2_IN_CH;
    filter_dims.w = INT16XINT8_GROUP2_FILTER_X;
    filter_dims.h = INT16XINT8_GROUP2_FILTER_Y;
    filter_dims.c = INT16XINT8_GROUP2_FILTER_CH;
    output_dims.w = INT16XINT8_GROUP2_OUTPUT_W;
    output_dims.h = INT16XINT8_GROUP2_OUTPUT_H;
    output_dims.c = INT16XINT8_GROUP2_OUT_CH;

    conv_params.padding.w = INT16XINT8_GROUP2_PAD_X;
    conv_params.padding.h = INT16XINT8_GROUP2_PAD_Y;
    conv_params.stride.w = INT16XINT8_GROUP2_STRIDE_X;
    conv_params.stride.h = INT16XINT8_GROUP2_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_GROUP2_DILATION_X;
    conv_params.dilation.h = INT16XINT8_GROUP2_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_GROUP2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_GROUP2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_group2_output_mult;
    quant_params.shift = (int32_t *)int16xint8_group2_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_group_same_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_GROUP_SAME_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_group_same_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_group_same_weights;
    const int16_t *input_data = int16xint8_group_same_input_tensor;
    const int16_t *output_ref = int16xint8_group_same_output_ref;
    const int32_t output_ref_size = INT16XINT8_GROUP_SAME_DST_SIZE;

    input_dims.n = INT16XINT8_GROUP_SAME_INPUT_BATCHES;
    input_dims.w = INT16XINT8_GROUP_SAME_INPUT_W;
    input_dims.h = INT16XINT8_GROUP_SAME_INPUT_H;
    input_dims.c = INT16XINT8_GROUP_SAME_IN_CH;
    filter_dims.w = INT16XINT8_GROUP_SAME_FILTER_X;
    filter_dims.h = INT16XINT8_GROUP_SAME_FILTER_Y;
    filter_dims.c = INT16XINT8_GROUP_SAME_FILTER_CH;
    output_dims.w = INT16XINT8_GROUP_SAME_OUTPUT_W;
    output_dims.h = INT16XINT8_GROUP_SAME_OUTPUT_H;
    output_dims.c = INT16XINT8_GROUP_SAME_OUT_CH;

    conv_params.padding.w = INT16XINT8_GROUP_SAME_PAD_X;
    conv_params.padding.h = INT16XINT8_GROUP_SAME_PAD_Y;
    conv_params.stride.w = INT16XINT8_GROUP_SAME_STRIDE_X;
    conv_params.stride.h = INT16XINT8_GROUP_SAME_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_GROUP_SAME_DILATION_X;
    conv_params.dilation.h = INT16XINT8_GROUP_SAME_DILATION_Y;

    conv_params.input_offset = INT16XINT8_GROUP_SAME_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_GROUP_SAME_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_GROUP_SAME_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_GROUP_SAME_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_group_same_output_mult;
    quant_params.shift = (int32_t *)int16xint8_group_same_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);

    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE_MESSAGE(validate_s16(output, output_ref, output_ref_size),
                             "int16xint8_group_same_arm_convolve_s16 direct mismatch");
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_group_batch2_dilated_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_GROUP_BATCH2_DILATED_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *int32_bias_data = int16xint8_group_batch2_dilated_biases;
    const cmsis_nn_bias_data bias_data = {int32_bias_data, true};
    const int8_t *kernel_data = int16xint8_group_batch2_dilated_weights;
    const int16_t *input_data = int16xint8_group_batch2_dilated_input_tensor;
    const int16_t *output_ref = int16xint8_group_batch2_dilated_output_ref;
    const int32_t output_ref_size = INT16XINT8_GROUP_BATCH2_DILATED_DST_SIZE;

    input_dims.n = INT16XINT8_GROUP_BATCH2_DILATED_INPUT_BATCHES;
    input_dims.w = INT16XINT8_GROUP_BATCH2_DILATED_INPUT_W;
    input_dims.h = INT16XINT8_GROUP_BATCH2_DILATED_INPUT_H;
    input_dims.c = INT16XINT8_GROUP_BATCH2_DILATED_IN_CH;
    filter_dims.w = INT16XINT8_GROUP_BATCH2_DILATED_FILTER_X;
    filter_dims.h = INT16XINT8_GROUP_BATCH2_DILATED_FILTER_Y;
    filter_dims.c = INT16XINT8_GROUP_BATCH2_DILATED_FILTER_CH;
    output_dims.w = INT16XINT8_GROUP_BATCH2_DILATED_OUTPUT_W;
    output_dims.h = INT16XINT8_GROUP_BATCH2_DILATED_OUTPUT_H;
    output_dims.c = INT16XINT8_GROUP_BATCH2_DILATED_OUT_CH;

    conv_params.padding.w = INT16XINT8_GROUP_BATCH2_DILATED_PAD_X;
    conv_params.padding.h = INT16XINT8_GROUP_BATCH2_DILATED_PAD_Y;
    conv_params.stride.w = INT16XINT8_GROUP_BATCH2_DILATED_STRIDE_X;
    conv_params.stride.h = INT16XINT8_GROUP_BATCH2_DILATED_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_GROUP_BATCH2_DILATED_DILATION_X;
    conv_params.dilation.h = INT16XINT8_GROUP_BATCH2_DILATED_DILATION_Y;

    conv_params.input_offset = INT16XINT8_GROUP_BATCH2_DILATED_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_GROUP_BATCH2_DILATED_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_GROUP_BATCH2_DILATED_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_GROUP_BATCH2_DILATED_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_group_batch2_dilated_output_mult;
    quant_params.shift = (int32_t *)int16xint8_group_batch2_dilated_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_weights;
    const int16_t *input_data = int16xint8_input;
    const int16_t *output_ref = int16xint8_output_ref;
    const int32_t output_ref_size = INT16XINT8_DST_SIZE;

    input_dims.n = INT16XINT8_INPUT_BATCHES;
    input_dims.w = INT16XINT8_INPUT_W;
    input_dims.h = INT16XINT8_INPUT_H;
    input_dims.c = INT16XINT8_IN_CH;
    filter_dims.w = INT16XINT8_FILTER_X;
    filter_dims.h = INT16XINT8_FILTER_Y;
    filter_dims.c = INT16XINT8_IN_CH;
    output_dims.w = INT16XINT8_OUTPUT_W;
    output_dims.h = INT16XINT8_OUTPUT_H;
    output_dims.c = INT16XINT8_OUT_CH;

    conv_params.padding.w = INT16XINT8_PAD_X;
    conv_params.padding.h = INT16XINT8_PAD_Y;
    conv_params.stride.w = INT16XINT8_STRIDE_X;
    conv_params.stride.h = INT16XINT8_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_DILATION_X;
    conv_params.dilation.h = INT16XINT8_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_output_mult;
    quant_params.shift = (int32_t *)int16xint8_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void requantize_s64_arm_convolve_s16(void)
{
    int16_t output[REQUANTIZE_S64_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = requantize_s64_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = requantize_s64_weights;
    const int16_t *input_data = requantize_s64_input;
    const int16_t *output_ref = requantize_s64_output_ref;
    const int32_t output_ref_size = REQUANTIZE_S64_DST_SIZE;

    input_dims.n = REQUANTIZE_S64_INPUT_BATCHES;
    input_dims.w = REQUANTIZE_S64_INPUT_W;
    input_dims.h = REQUANTIZE_S64_INPUT_H;
    input_dims.c = REQUANTIZE_S64_IN_CH;
    filter_dims.w = REQUANTIZE_S64_FILTER_X;
    filter_dims.h = REQUANTIZE_S64_FILTER_Y;
    filter_dims.c = REQUANTIZE_S64_IN_CH;
    
    output_dims.w = REQUANTIZE_S64_OUTPUT_W;
    output_dims.h = REQUANTIZE_S64_OUTPUT_H;
    output_dims.c = REQUANTIZE_S64_OUT_CH;

    conv_params.padding.w = REQUANTIZE_S64_PAD_X;
    conv_params.padding.h = REQUANTIZE_S64_PAD_Y;
    conv_params.stride.w = REQUANTIZE_S64_STRIDE_X;
    conv_params.stride.h = REQUANTIZE_S64_STRIDE_Y;
    conv_params.dilation.w = REQUANTIZE_S64_DILATION_X;
    conv_params.dilation.h = REQUANTIZE_S64_DILATION_Y;

    conv_params.input_offset = REQUANTIZE_S64_INPUT_OFFSET;
    conv_params.output_offset = REQUANTIZE_S64_OUTPUT_OFFSET;
    conv_params.activation.min = REQUANTIZE_S64_OUT_ACTIVATION_MIN;
    conv_params.activation.max = REQUANTIZE_S64_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)requantize_s64_output_mult;
    quant_params.shift = (int32_t *)requantize_s64_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s16(&ctx,
                                                  &conv_params,
                                                  &quant_params,
                                                  &input_dims,
                                                  input_data,
                                                  &filter_dims,
                                                  kernel_data,
                                                  &bias_dims,
                                                  &bias_data,
                                                  &output_dims,
                                                  output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_dilation_1_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_DILATION_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_dilation_1_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_dilation_1_weights;
    const int16_t *input_data = int16xint8_dilation_1_input;
    const int16_t *output_ref = int16xint8_dilation_1_output_ref;
    const int32_t output_ref_size = INT16XINT8_DILATION_1_DST_SIZE;

    input_dims.n = INT16XINT8_DILATION_1_INPUT_BATCHES;
    input_dims.w = INT16XINT8_DILATION_1_INPUT_W;
    input_dims.h = INT16XINT8_DILATION_1_INPUT_H;
    input_dims.c = INT16XINT8_DILATION_1_IN_CH;
    filter_dims.w = INT16XINT8_DILATION_1_FILTER_X;
    filter_dims.h = INT16XINT8_DILATION_1_FILTER_Y;
    filter_dims.c = INT16XINT8_DILATION_1_IN_CH;
    
    output_dims.w = INT16XINT8_DILATION_1_OUTPUT_W;
    output_dims.h = INT16XINT8_DILATION_1_OUTPUT_H;
    output_dims.c = INT16XINT8_DILATION_1_OUT_CH;

    conv_params.padding.w = INT16XINT8_DILATION_1_PAD_X;
    conv_params.padding.h = INT16XINT8_DILATION_1_PAD_Y;
    conv_params.stride.w = INT16XINT8_DILATION_1_STRIDE_X;
    conv_params.stride.h = INT16XINT8_DILATION_1_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_DILATION_1_DILATION_X;
    conv_params.dilation.h = INT16XINT8_DILATION_1_DILATION_Y;

    conv_params.input_offset = INT16XINT8_DILATION_1_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_DILATION_1_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_DILATION_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_DILATION_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_dilation_1_output_mult;
    quant_params.shift = (int32_t *)int16xint8_dilation_1_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s16(&ctx,
                                                  &conv_params,
                                                  &quant_params,
                                                  &input_dims,
                                                  input_data,
                                                  &filter_dims,
                                                  kernel_data,
                                                  &bias_dims,
                                                  &bias_data,
                                                  &output_dims,
                                                  output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_dilation_2_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_DILATION_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_dilation_2_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_dilation_2_weights;
    const int16_t *input_data = int16xint8_dilation_2_input;
    const int16_t *output_ref = int16xint8_dilation_2_output_ref;
    const int32_t output_ref_size = INT16XINT8_DILATION_2_DST_SIZE;

    input_dims.n = INT16XINT8_DILATION_2_INPUT_BATCHES;
    input_dims.w = INT16XINT8_DILATION_2_INPUT_W;
    input_dims.h = INT16XINT8_DILATION_2_INPUT_H;
    input_dims.c = INT16XINT8_DILATION_2_IN_CH;
    filter_dims.w = INT16XINT8_DILATION_2_FILTER_X;
    filter_dims.h = INT16XINT8_DILATION_2_FILTER_Y;
    filter_dims.c = INT16XINT8_DILATION_2_IN_CH;
    
    output_dims.w = INT16XINT8_DILATION_2_OUTPUT_W;
    output_dims.h = INT16XINT8_DILATION_2_OUTPUT_H;
    output_dims.c = INT16XINT8_DILATION_2_OUT_CH;

    conv_params.padding.w = INT16XINT8_DILATION_2_PAD_X;
    conv_params.padding.h = INT16XINT8_DILATION_2_PAD_Y;
    conv_params.stride.w = INT16XINT8_DILATION_2_STRIDE_X;
    conv_params.stride.h = INT16XINT8_DILATION_2_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_DILATION_2_DILATION_X;
    conv_params.dilation.h = INT16XINT8_DILATION_2_DILATION_Y;

    conv_params.input_offset = INT16XINT8_DILATION_2_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_DILATION_2_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_DILATION_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_DILATION_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_dilation_2_output_mult;
    quant_params.shift = (int32_t *)int16xint8_dilation_2_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s16(&ctx,
                                                  &conv_params,
                                                  &quant_params,
                                                  &input_dims,
                                                  input_data,
                                                  &filter_dims,
                                                  kernel_data,
                                                  &bias_dims,
                                                  &bias_data,
                                                  &output_dims,
                                                  output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_dilation_3_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_DILATION_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_dilation_3_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_dilation_3_weights;
    const int16_t *input_data = int16xint8_dilation_3_input;
    const int16_t *output_ref = int16xint8_dilation_3_output_ref;
    const int32_t output_ref_size = INT16XINT8_DILATION_3_DST_SIZE;

    input_dims.n = INT16XINT8_DILATION_3_INPUT_BATCHES;
    input_dims.w = INT16XINT8_DILATION_3_INPUT_W;
    input_dims.h = INT16XINT8_DILATION_3_INPUT_H;
    input_dims.c = INT16XINT8_DILATION_3_IN_CH;
    filter_dims.w = INT16XINT8_DILATION_3_FILTER_X;
    filter_dims.h = INT16XINT8_DILATION_3_FILTER_Y;
    filter_dims.c = INT16XINT8_DILATION_3_IN_CH;
    output_dims.w = INT16XINT8_DILATION_3_OUTPUT_W;
    output_dims.h = INT16XINT8_DILATION_3_OUTPUT_H;
    output_dims.c = INT16XINT8_DILATION_3_OUT_CH;

    conv_params.padding.w = INT16XINT8_DILATION_3_PAD_X;
    conv_params.padding.h = INT16XINT8_DILATION_3_PAD_Y;
    conv_params.stride.w = INT16XINT8_DILATION_3_STRIDE_X;
    conv_params.stride.h = INT16XINT8_DILATION_3_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_DILATION_3_DILATION_X;
    conv_params.dilation.h = INT16XINT8_DILATION_3_DILATION_Y;

    conv_params.input_offset = INT16XINT8_DILATION_3_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_DILATION_3_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_DILATION_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_DILATION_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_dilation_3_output_mult;
    quant_params.shift = (int32_t *)int16xint8_dilation_3_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_convolve_s16(&ctx,
                                                  &conv_params,
                                                  &quant_params,
                                                  &input_dims,
                                                  input_data,
                                                  &filter_dims,
                                                  kernel_data,
                                                  &bias_dims,
                                                  &bias_data,
                                                  &output_dims,
                                                  output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void buffer_size_arm_convolve_s16(void)
{
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = INT16XINT8_DILATION_3_INPUT_BATCHES;
    input_dims.w = INT16XINT8_DILATION_3_INPUT_W;
    input_dims.h = INT16XINT8_DILATION_3_INPUT_H;
    input_dims.c = INT16XINT8_DILATION_3_IN_CH;
    filter_dims.w = INT16XINT8_DILATION_3_FILTER_X;
    filter_dims.h = INT16XINT8_DILATION_3_FILTER_Y;
    filter_dims.c = INT16XINT8_DILATION_3_IN_CH;
    output_dims.w = INT16XINT8_DILATION_3_OUTPUT_W;
    output_dims.h = INT16XINT8_DILATION_3_OUTPUT_H;
    output_dims.c = INT16XINT8_DILATION_3_OUT_CH;

    conv_params.padding.w = INT16XINT8_DILATION_3_PAD_X;
    conv_params.padding.h = INT16XINT8_DILATION_3_PAD_Y;
    conv_params.stride.w = INT16XINT8_DILATION_3_STRIDE_X;
    conv_params.stride.h = INT16XINT8_DILATION_3_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_DILATION_3_DILATION_X;
    conv_params.dilation.h = INT16XINT8_DILATION_3_DILATION_Y;

    conv_params.input_offset = INT16XINT8_DILATION_3_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_DILATION_3_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_DILATION_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_DILATION_3_OUT_ACTIVATION_MAX;

    const int32_t buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, buf_size);
}

void buffer_size_mve_arm_convolve_s16(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = INT16XINT8_DILATION_3_INPUT_BATCHES;
    input_dims.w = INT16XINT8_DILATION_3_INPUT_W;
    input_dims.h = INT16XINT8_DILATION_3_INPUT_H;
    input_dims.c = INT16XINT8_DILATION_3_IN_CH;
    filter_dims.w = INT16XINT8_DILATION_3_FILTER_X;
    filter_dims.h = INT16XINT8_DILATION_3_FILTER_Y;
    filter_dims.c = INT16XINT8_DILATION_3_IN_CH;
    output_dims.w = INT16XINT8_DILATION_3_OUTPUT_W;
    output_dims.h = INT16XINT8_DILATION_3_OUTPUT_H;
    output_dims.c = INT16XINT8_DILATION_3_OUT_CH;

    conv_params.padding.w = INT16XINT8_DILATION_3_PAD_X;
    conv_params.padding.h = INT16XINT8_DILATION_3_PAD_Y;
    conv_params.stride.w = INT16XINT8_DILATION_3_STRIDE_X;
    conv_params.stride.h = INT16XINT8_DILATION_3_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_DILATION_3_DILATION_X;
    conv_params.dilation.h = INT16XINT8_DILATION_3_DILATION_Y;

    conv_params.input_offset = INT16XINT8_DILATION_3_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_DILATION_3_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_DILATION_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_DILATION_3_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_convolve_wrapper_s16_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_convolve_s16(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = INT16XINT8_DILATION_3_INPUT_BATCHES;
    input_dims.w = INT16XINT8_DILATION_3_INPUT_W;
    input_dims.h = INT16XINT8_DILATION_3_INPUT_H;
    input_dims.c = INT16XINT8_DILATION_3_IN_CH;
    filter_dims.w = INT16XINT8_DILATION_3_FILTER_X;
    filter_dims.h = INT16XINT8_DILATION_3_FILTER_Y;
    filter_dims.c = INT16XINT8_DILATION_3_IN_CH;
    
    output_dims.w = INT16XINT8_DILATION_3_OUTPUT_W;
    output_dims.h = INT16XINT8_DILATION_3_OUTPUT_H;
    output_dims.c = INT16XINT8_DILATION_3_OUT_CH;

    conv_params.padding.w = INT16XINT8_DILATION_3_PAD_X;
    conv_params.padding.h = INT16XINT8_DILATION_3_PAD_Y;
    conv_params.stride.w = INT16XINT8_DILATION_3_STRIDE_X;
    conv_params.stride.h = INT16XINT8_DILATION_3_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_DILATION_3_DILATION_X;
    conv_params.dilation.h = INT16XINT8_DILATION_3_DILATION_Y;

    conv_params.input_offset = INT16XINT8_DILATION_3_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_DILATION_3_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_DILATION_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_DILATION_3_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_convolve_wrapper_s16_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}

void int16xint8_spill_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_SPILL_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_spill_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_spill_weights;
    const int16_t *input_data = int16xint8_spill_input;
    const int16_t *output_ref = int16xint8_spill_output_ref;
    const int32_t output_ref_size = INT16XINT8_SPILL_DST_SIZE;

    input_dims.n = INT16XINT8_SPILL_INPUT_BATCHES;
    input_dims.w = INT16XINT8_SPILL_INPUT_W;
    input_dims.h = INT16XINT8_SPILL_INPUT_H;
    input_dims.c = INT16XINT8_SPILL_IN_CH;
    filter_dims.w = INT16XINT8_SPILL_FILTER_X;
    filter_dims.h = INT16XINT8_SPILL_FILTER_Y;
    filter_dims.c = INT16XINT8_SPILL_IN_CH;
    
    output_dims.w = INT16XINT8_SPILL_OUTPUT_W;
    output_dims.h = INT16XINT8_SPILL_OUTPUT_H;
    output_dims.c = INT16XINT8_SPILL_OUT_CH;

    conv_params.padding.w = INT16XINT8_SPILL_PAD_X;
    conv_params.padding.h = INT16XINT8_SPILL_PAD_Y;
    conv_params.stride.w = INT16XINT8_SPILL_STRIDE_X;
    conv_params.stride.h = INT16XINT8_SPILL_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_SPILL_DILATION_X;
    conv_params.dilation.h = INT16XINT8_SPILL_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_SPILL_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_SPILL_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_spill_output_mult;
    quant_params.shift = (int32_t *)int16xint8_spill_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8_spill2_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_SPILL2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_spill2_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_spill2_weights;
    const int16_t *input_data = int16xint8_spill2_input;
    const int16_t *output_ref = int16xint8_spill2_output_ref;
    const int32_t output_ref_size = INT16XINT8_SPILL2_DST_SIZE;

    input_dims.n = INT16XINT8_SPILL2_INPUT_BATCHES;
    input_dims.w = INT16XINT8_SPILL2_INPUT_W;
    input_dims.h = INT16XINT8_SPILL2_INPUT_H;
    input_dims.c = INT16XINT8_SPILL2_IN_CH;
    filter_dims.w = INT16XINT8_SPILL2_FILTER_X;
    filter_dims.h = INT16XINT8_SPILL2_FILTER_Y;
    filter_dims.c = INT16XINT8_SPILL2_IN_CH;
    output_dims.w = INT16XINT8_SPILL2_OUTPUT_W;
    output_dims.h = INT16XINT8_SPILL2_OUTPUT_H;
    output_dims.c = INT16XINT8_SPILL2_OUT_CH;

    conv_params.padding.w = INT16XINT8_SPILL2_PAD_X;
    conv_params.padding.h = INT16XINT8_SPILL2_PAD_Y;
    conv_params.stride.w = INT16XINT8_SPILL2_STRIDE_X;
    conv_params.stride.h = INT16XINT8_SPILL2_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_SPILL2_DILATION_X;
    conv_params.dilation.h = INT16XINT8_SPILL2_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8_SPILL2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_SPILL2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_spill2_output_mult;
    quant_params.shift = (int32_t *)int16xint8_spill2_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8xint32_1_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8XINT32_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *int32_bias_data = int16xint8xint32_1_biases;
    const cmsis_nn_bias_data bias_data = {int32_bias_data, true};
    const int8_t *kernel_data = int16xint8xint32_1_weights;
    const int16_t *input_data = int16xint8xint32_1_input;
    const int16_t *output_ref = int16xint8xint32_1_output_ref;
    const int32_t output_ref_size = INT16XINT8XINT32_1_DST_SIZE;

    input_dims.n = INT16XINT8XINT32_1_INPUT_BATCHES;
    input_dims.w = INT16XINT8XINT32_1_INPUT_W;
    input_dims.h = INT16XINT8XINT32_1_INPUT_H;
    input_dims.c = INT16XINT8XINT32_1_IN_CH;
    filter_dims.w = INT16XINT8XINT32_1_FILTER_X;
    filter_dims.h = INT16XINT8XINT32_1_FILTER_Y;
    filter_dims.c = INT16XINT8XINT32_1_IN_CH;
    output_dims.w = INT16XINT8XINT32_1_OUTPUT_W;
    output_dims.h = INT16XINT8XINT32_1_OUTPUT_H;
    output_dims.c = INT16XINT8XINT32_1_OUT_CH;

    conv_params.padding.w = INT16XINT8XINT32_1_PAD_X;
    conv_params.padding.h = INT16XINT8XINT32_1_PAD_Y;
    conv_params.stride.w = INT16XINT8XINT32_1_STRIDE_X;
    conv_params.stride.h = INT16XINT8XINT32_1_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8XINT32_1_DILATION_X;
    conv_params.dilation.h = INT16XINT8XINT32_1_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8XINT32_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8XINT32_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8xint32_1_output_mult;
    quant_params.shift = (int32_t *)int16xint8xint32_1_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8xint32_2_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8XINT32_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *int32_bias_data = int16xint8xint32_2_biases;
    const cmsis_nn_bias_data bias_data = {int32_bias_data, true};
    const int8_t *kernel_data = int16xint8xint32_2_weights;
    const int16_t *input_data = int16xint8xint32_2_input;
    const int16_t *output_ref = int16xint8xint32_2_output_ref;
    const int32_t output_ref_size = INT16XINT8XINT32_2_DST_SIZE;

    input_dims.n = INT16XINT8XINT32_2_INPUT_BATCHES;
    input_dims.w = INT16XINT8XINT32_2_INPUT_W;
    input_dims.h = INT16XINT8XINT32_2_INPUT_H;
    input_dims.c = INT16XINT8XINT32_2_IN_CH;
    filter_dims.w = INT16XINT8XINT32_2_FILTER_X;
    filter_dims.h = INT16XINT8XINT32_2_FILTER_Y;
    filter_dims.c = INT16XINT8XINT32_2_IN_CH;
    output_dims.w = INT16XINT8XINT32_2_OUTPUT_W;
    output_dims.h = INT16XINT8XINT32_2_OUTPUT_H;
    output_dims.c = INT16XINT8XINT32_2_OUT_CH;

    conv_params.padding.w = INT16XINT8XINT32_2_PAD_X;
    conv_params.padding.h = INT16XINT8XINT32_2_PAD_Y;
    conv_params.stride.w = INT16XINT8XINT32_2_STRIDE_X;
    conv_params.stride.h = INT16XINT8XINT32_2_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8XINT32_2_DILATION_X;
    conv_params.dilation.h = INT16XINT8XINT32_2_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8XINT32_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8XINT32_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8xint32_2_output_mult;
    quant_params.shift = (int32_t *)int16xint8xint32_2_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8xint32_3_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8XINT32_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *int32_bias_data = int16xint8xint32_3_biases;
    const cmsis_nn_bias_data bias_data = {int32_bias_data, true};
    const int8_t *kernel_data = int16xint8xint32_3_weights;
    const int16_t *input_data = int16xint8xint32_3_input;
    const int16_t *output_ref = int16xint8xint32_3_output_ref;
    const int32_t output_ref_size = INT16XINT8XINT32_3_DST_SIZE;

    input_dims.n = INT16XINT8XINT32_3_INPUT_BATCHES;
    input_dims.w = INT16XINT8XINT32_3_INPUT_W;
    input_dims.h = INT16XINT8XINT32_3_INPUT_H;
    input_dims.c = INT16XINT8XINT32_3_IN_CH;
    filter_dims.w = INT16XINT8XINT32_3_FILTER_X;
    filter_dims.h = INT16XINT8XINT32_3_FILTER_Y;
    filter_dims.c = INT16XINT8XINT32_3_IN_CH;
    output_dims.w = INT16XINT8XINT32_3_OUTPUT_W;
    output_dims.h = INT16XINT8XINT32_3_OUTPUT_H;
    output_dims.c = INT16XINT8XINT32_3_OUT_CH;

    conv_params.padding.w = INT16XINT8XINT32_3_PAD_X;
    conv_params.padding.h = INT16XINT8XINT32_3_PAD_Y;
    conv_params.stride.w = INT16XINT8XINT32_3_STRIDE_X;
    conv_params.stride.h = INT16XINT8XINT32_3_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8XINT32_3_DILATION_X;
    conv_params.dilation.h = INT16XINT8XINT32_3_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8XINT32_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8XINT32_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8xint32_3_output_mult;
    quant_params.shift = (int32_t *)int16xint8xint32_3_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8xint32_4_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8XINT32_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *int32_bias_data = int16xint8xint32_4_biases;
    const cmsis_nn_bias_data bias_data = {int32_bias_data, true};
    const int8_t *kernel_data = int16xint8xint32_4_weights;
    const int16_t *input_data = int16xint8xint32_4_input;
    const int16_t *output_ref = int16xint8xint32_4_output_ref;
    const int32_t output_ref_size = INT16XINT8XINT32_4_DST_SIZE;

    input_dims.n = INT16XINT8XINT32_4_INPUT_BATCHES;
    input_dims.w = INT16XINT8XINT32_4_INPUT_W;
    input_dims.h = INT16XINT8XINT32_4_INPUT_H;
    input_dims.c = INT16XINT8XINT32_4_IN_CH;
    filter_dims.w = INT16XINT8XINT32_4_FILTER_X;
    filter_dims.h = INT16XINT8XINT32_4_FILTER_Y;
    filter_dims.c = INT16XINT8XINT32_4_IN_CH;
    output_dims.w = INT16XINT8XINT32_4_OUTPUT_W;
    output_dims.h = INT16XINT8XINT32_4_OUTPUT_H;
    output_dims.c = INT16XINT8XINT32_4_OUT_CH;

    conv_params.padding.w = INT16XINT8XINT32_4_PAD_X;
    conv_params.padding.h = INT16XINT8XINT32_4_PAD_Y;
    conv_params.stride.w = INT16XINT8XINT32_4_STRIDE_X;
    conv_params.stride.h = INT16XINT8XINT32_4_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8XINT32_4_DILATION_X;
    conv_params.dilation.h = INT16XINT8XINT32_4_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8XINT32_4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8XINT32_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8xint32_4_output_mult;
    quant_params.shift = (int32_t *)int16xint8xint32_4_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8xint32_5_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8XINT32_5_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *int32_bias_data = int16xint8xint32_5_biases;
    const cmsis_nn_bias_data bias_data = {int32_bias_data, true};
    const int8_t *kernel_data = int16xint8xint32_5_weights;
    const int16_t *input_data = int16xint8xint32_5_input;
    const int16_t *output_ref = int16xint8xint32_5_output_ref;
    const int32_t output_ref_size = INT16XINT8XINT32_5_DST_SIZE;

    input_dims.n = INT16XINT8XINT32_5_INPUT_BATCHES;
    input_dims.w = INT16XINT8XINT32_5_INPUT_W;
    input_dims.h = INT16XINT8XINT32_5_INPUT_H;
    input_dims.c = INT16XINT8XINT32_5_IN_CH;
    filter_dims.w = INT16XINT8XINT32_5_FILTER_X;
    filter_dims.h = INT16XINT8XINT32_5_FILTER_Y;
    filter_dims.c = INT16XINT8XINT32_5_IN_CH;
    output_dims.w = INT16XINT8XINT32_5_OUTPUT_W;
    output_dims.h = INT16XINT8XINT32_5_OUTPUT_H;
    output_dims.c = INT16XINT8XINT32_5_OUT_CH;

    conv_params.padding.w = INT16XINT8XINT32_5_PAD_X;
    conv_params.padding.h = INT16XINT8XINT32_5_PAD_Y;
    conv_params.stride.w = INT16XINT8XINT32_5_STRIDE_X;
    conv_params.stride.h = INT16XINT8XINT32_5_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8XINT32_5_DILATION_X;
    conv_params.dilation.h = INT16XINT8XINT32_5_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8XINT32_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8XINT32_5_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8xint32_5_output_mult;
    quant_params.shift = (int32_t *)int16xint8xint32_5_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void int16xint8xint32_6_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8XINT32_6_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *int32_bias_data = int16xint8xint32_6_biases;
    const cmsis_nn_bias_data bias_data = {int32_bias_data, true};
    const int8_t *kernel_data = int16xint8xint32_6_weights;
    const int16_t *input_data = int16xint8xint32_6_input;
    const int16_t *output_ref = int16xint8xint32_6_output_ref;
    const int32_t output_ref_size = INT16XINT8XINT32_6_DST_SIZE;

    input_dims.n = INT16XINT8XINT32_6_INPUT_BATCHES;
    input_dims.w = INT16XINT8XINT32_6_INPUT_W;
    input_dims.h = INT16XINT8XINT32_6_INPUT_H;
    input_dims.c = INT16XINT8XINT32_6_IN_CH;
    filter_dims.w = INT16XINT8XINT32_6_FILTER_X;
    filter_dims.h = INT16XINT8XINT32_6_FILTER_Y;
    filter_dims.c = INT16XINT8XINT32_6_IN_CH;
    output_dims.w = INT16XINT8XINT32_6_OUTPUT_W;
    output_dims.h = INT16XINT8XINT32_6_OUTPUT_H;
    output_dims.c = INT16XINT8XINT32_6_OUT_CH;

    conv_params.padding.w = INT16XINT8XINT32_6_PAD_X;
    conv_params.padding.h = INT16XINT8XINT32_6_PAD_Y;
    conv_params.stride.w = INT16XINT8XINT32_6_STRIDE_X;
    conv_params.stride.h = INT16XINT8XINT32_6_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8XINT32_6_DILATION_X;
    conv_params.dilation.h = INT16XINT8XINT32_6_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8XINT32_6_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8XINT32_6_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8xint32_6_output_mult;
    quant_params.shift = (int32_t *)int16xint8xint32_6_output_shift;

    int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    arm_cmsis_nn_status result;
    result = arm_convolve_s16(&ctx,
                              &conv_params,
                              &quant_params,
                              &input_dims,
                              input_data,
                              &filter_dims,
                              kernel_data,
                              &bias_dims,
                              &bias_data,
                              &output_dims,
                              output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_wrapper_s16(&ctx,
                                      &conv_params,
                                      &quant_params,
                                      &input_dims,
                                      input_data,
                                      &filter_dims,
                                      kernel_data,
                                      &bias_dims,
                                      &bias_data,
                                      &output_dims,
                                      output);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

/* arm_convolve_s16() with its weights ending at an unmapped MPU gap: the MVE matrix kernel must not load past the
   last filter (#605). Sixteen output pixels run its four-row block; the output must match the run with the weights in
   place. */
void weights_at_gap_arm_convolve_s16(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    enum
    {
        hw = 4,
        in_ch = 8,
        out_ch = 8,
        k = 3,
        input_size = hw * hw * in_ch,
        weights_size = out_ch * k * k * in_ch,
        output_size = hw * hw * out_ch
    };
    int16_t input[input_size];
    int8_t kernel[weights_size];
    int64_t bias[out_ch];
    int32_t multiplier[out_ch];
    int32_t shift[out_ch];
    int16_t expected[output_size];
    int16_t output[output_size];
    for (int i = 0; i < input_size; i++)
    {
        input[i] = (int16_t)((i * 193) % 4001 - 2000);
    }
    for (int i = 0; i < weights_size; i++)
    {
        kernel[i] = (int8_t)((i * 5) % 31 - 15);
    }
    for (int i = 0; i < out_ch; i++)
    {
        bias[i] = (int64_t)i * 1000 - 3000;
        multiplier[i] = 1 << 30;
        shift[i] = -4 - (i % 3);
    }
    const cmsis_nn_dims input_dims = {1, hw, hw, in_ch};
    const cmsis_nn_dims filter_dims = {out_ch, k, k, in_ch};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_ch};
    const cmsis_nn_dims output_dims = {1, hw, hw, out_ch};
    const cmsis_nn_conv_params conv_params = {
        .input_offset = 0,
        .output_offset = 0,
        .stride = {1, 1},
        .padding = {1, 1},
        .dilation = {1, 1},
        .activation = {-32768, 32767},
    };
    const cmsis_nn_per_channel_quant_params quant_params = {multiplier, shift};
    const cmsis_nn_bias_data bias_data = {bias, false};
    const int32_t buffer_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
    cmsis_nn_context ctx = {malloc(buffer_size), buffer_size};
    TEST_ASSERT_TRUE(buffer_size == 0 || ctx.buf != NULL);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_s16(&ctx,
                                       &conv_params,
                                       &quant_params,
                                       &input_dims,
                                       input,
                                       &filter_dims,
                                       kernel,
                                       &bias_dims,
                                       &bias_data,
                                       &output_dims,
                                       expected));
    const int8_t *w = guard_place(kernel, sizeof(kernel));
    memset(output, 0x55, sizeof(output));
    guard_gap_enable();
    const arm_cmsis_nn_status result = arm_convolve_s16(
        &ctx, &conv_params, &quant_params, &input_dims, input, &filter_dims, w, &bias_dims, &bias_data, &output_dims, output);
    guard_gap_disable();
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(expected, output, output_size);
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buffer_size);
        free(ctx.buf);
    }
#endif
}

/* Depths that do not make whole groups: each entry must return ARM_CMSIS_NN_ARG_ERROR before it divides by the
   filter depth or the group count. */
void group_arg_errors_arm_convolve_s16(void)
{
    /* {C_IN, filter C, C_OUT} */
    static const int32_t cases[][3] = {
        {6, 4, 4},   /* C_IN not a multiple of the filter depth */
        {6, 0, 4},   /* filter depth 0 */
        {4, 8, 4},   /* filter deeper than the input: zero groups */
        {0, 4, 4},   /* no input channels */
        {8, 4, 3},   /* 3 output channels for 2 groups */
        {8, 4, -2},  /* negative output depth */
        {-8, -4, 4}, /* negative depths that divide */
        {8, -4, 4},  /* negative filter depth */
        {-4, 4, 4}   /* negative input depth */
    };
    int16_t input[3 * 3 * 8] = {0};
    int8_t kernel[8 * 2 * 2 * 8] = {0};
    int64_t bias[8] = {0};
    int32_t multiplier[8];
    int32_t shift[8];
    int16_t output[2 * 2 * 8];
    int16_t scratch[2 * 2 * 8 * 4];
    for (int i = 0; i < 8; i++)
    {
        multiplier[i] = 1 << 30;
        shift[i] = -1;
    }
    const cmsis_nn_conv_params conv_params = {
        .input_offset = 0,
        .output_offset = 0,
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
        .activation = {-32768, 32767},
    };
    const cmsis_nn_per_channel_quant_params quant_params = {multiplier, shift};
    const cmsis_nn_bias_data bias_data = {bias, false};
    cmsis_nn_context ctx = {scratch, sizeof(scratch)};
    /* arm_convolve_s16_fast_small_kernel() has only an MVE implementation */
#if defined(ARM_MATH_MVEI)
    const arm_cmsis_nn_status fast_status = ARM_CMSIS_NN_ARG_ERROR;
#else
    const arm_cmsis_nn_status fast_status = ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
    for (uint32_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
    {
        const cmsis_nn_dims input_dims = {1, 3, 3, cases[c][0]};
        const cmsis_nn_dims filter_dims = {cases[c][2], 2, 2, cases[c][1]};
        const cmsis_nn_dims bias_dims = {1, 1, 1, cases[c][2]};
        const cmsis_nn_dims output_dims = {1, 2, 2, cases[c][2]};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_convolve_s16(&ctx,
                                           &conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input,
                                           &filter_dims,
                                           kernel,
                                           &bias_dims,
                                           &bias_data,
                                           &output_dims,
                                           output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_convolve_wrapper_s16(&ctx,
                                                   &conv_params,
                                                   &quant_params,
                                                   &input_dims,
                                                   input,
                                                   &filter_dims,
                                                   kernel,
                                                   &bias_dims,
                                                   &bias_data,
                                                   &output_dims,
                                                   output));
        TEST_ASSERT_EQUAL(fast_status,
                          arm_convolve_s16_fast_small_kernel(&ctx,
                                                             &conv_params,
                                                             &quant_params,
                                                             &input_dims,
                                                             input,
                                                             &filter_dims,
                                                             kernel,
                                                             &bias_dims,
                                                             &bias_data,
                                                             &output_dims,
                                                             output));
    }

    /* {C_IN, filter C, C_OUT, kernel size} through the wrapper's own routes: the 1x1 kernel (MVE), and the
       channel-multiplier-1 kernel (filter C 1, C_IN = C_OUT; on MVE only for 9 or more filter elements), neither of
       which checks the depths itself */
    static const int32_t routed[][4] = {
        {0, 0, 4, 1}, {-4, -4, 4, 1}, {4, 4, -2, 1}, {0, 1, 0, 2}, {-3, 1, -3, 2}, {0, 1, 0, 3}, {-3, 1, -3, 3}};
    for (uint32_t c = 0; c < sizeof(routed) / sizeof(routed[0]); c++)
    {
        const int32_t k = routed[c][3];
        const cmsis_nn_dims input_dims = {1, 3, 3, routed[c][0]};
        const cmsis_nn_dims filter_dims = {routed[c][2], k, k, routed[c][1]};
        const cmsis_nn_dims bias_dims = {1, 1, 1, routed[c][2]};
        const cmsis_nn_dims output_dims = {1, 4 - k, 4 - k, routed[c][2]};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_convolve_wrapper_s16(&ctx,
                                                   &conv_params,
                                                   &quant_params,
                                                   &input_dims,
                                                   input,
                                                   &filter_dims,
                                                   kernel,
                                                   &bias_dims,
                                                   &bias_data,
                                                   &output_dims,
                                                   output));
    }

    /* arm_convolve_1x1_s16_ns_np_nd() called directly over 3 pixels (#763): it must not write a negative output
       depth's worth of values, nor divide by a filter depth of 0. {C_IN, filter C, C_OUT} */
    static const int32_t direct_1x1[][3] = {{4, 4, -5}, {4, 4, -1}, {0, 0, 4}, {-4, -4, 4}, {6, 4, 4}};
    for (uint32_t c = 0; c < sizeof(direct_1x1) / sizeof(direct_1x1[0]); c++)
    {
        const cmsis_nn_dims input_dims = {1, 1, 3, direct_1x1[c][0]};
        const cmsis_nn_dims filter_dims = {direct_1x1[c][2], 1, 1, direct_1x1[c][1]};
        const cmsis_nn_dims bias_dims = {1, 1, 1, direct_1x1[c][2]};
        const cmsis_nn_dims output_dims = {1, 1, 3, direct_1x1[c][2]};
        for (uint32_t i = 0; i < sizeof(output) / sizeof(output[0]); i++)
        {
            output[i] = 0x5555;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_convolve_1x1_s16_ns_np_nd(&ctx,
                                                        &conv_params,
                                                        &quant_params,
                                                        &input_dims,
                                                        input,
                                                        &filter_dims,
                                                        kernel,
                                                        &bias_dims,
                                                        &bias_data,
                                                        &output_dims,
                                                        output));
        for (uint32_t i = 0; i < sizeof(output) / sizeof(output[0]); i++)
        {
            TEST_ASSERT_EQUAL_INT16(0x5555, output[i]);
        }
    }

    /* Controls: two whole groups, and no output channels, are accepted */
    static const int32_t valid[][3] = {{8, 4, 4}, {8, 4, 0}};
    for (uint32_t c = 0; c < sizeof(valid) / sizeof(valid[0]); c++)
    {
        const cmsis_nn_dims input_dims = {1, 3, 3, valid[c][0]};
        const cmsis_nn_dims filter_dims = {valid[c][2], 2, 2, valid[c][1]};
        const cmsis_nn_dims bias_dims = {1, 1, 1, valid[c][2]};
        const cmsis_nn_dims output_dims = {1, 2, 2, valid[c][2]};
        TEST_ASSERT_TRUE(arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims) <= (int32_t)sizeof(scratch));
        TEST_ASSERT_TRUE(arm_convolve_wrapper_s16_get_buffer_size(
                             &conv_params, &input_dims, &filter_dims, &output_dims) <= (int32_t)sizeof(scratch));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_convolve_s16(&ctx,
                                           &conv_params,
                                           &quant_params,
                                           &input_dims,
                                           input,
                                           &filter_dims,
                                           kernel,
                                           &bias_dims,
                                           &bias_data,
                                           &output_dims,
                                           output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_convolve_wrapper_s16(&ctx,
                                                   &conv_params,
                                                   &quant_params,
                                                   &input_dims,
                                                   input,
                                                   &filter_dims,
                                                   kernel,
                                                   &bias_dims,
                                                   &bias_data,
                                                   &output_dims,
                                                   output));
    }
}
