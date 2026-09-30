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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <unity.h>

#include "../TestData/conv_1_x_n_1/test_data.h"
#include "../TestData/conv_1_x_n_2/test_data.h"
#include "../TestData/conv_1_x_n_3/test_data.h"
#include "../TestData/conv_1_x_n_4/test_data.h"
#include "../TestData/conv_1_x_n_5/test_data.h"
#include "../TestData/conv_1_x_n_6_generic/test_data.h"
#include "../TestData/conv_1_x_n_7/test_data.h"
#include "../TestData/conv_1_x_n_8/test_data.h"
#include "../Utils/mpu_guard.h"

#include "../Utils/validate.h"

void conv_1_x_n_1_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx; // New weights-sum context.
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_1_biases;
    const int8_t *kernel_data = conv_1_x_n_1_weights;
    const int8_t *input_data = conv_1_x_n_1_input;
    const int8_t *output_ref = conv_1_x_n_1_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_1_DST_SIZE;

    input_dims.n = CONV_1_X_N_1_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_1_INPUT_W;
    input_dims.h = CONV_1_X_N_1_INPUT_H;
    input_dims.c = CONV_1_X_N_1_IN_CH;
    filter_dims.w = CONV_1_X_N_1_FILTER_X;
    filter_dims.h = CONV_1_X_N_1_FILTER_Y;
    filter_dims.c = CONV_1_X_N_1_IN_CH;
    output_dims.w = CONV_1_X_N_1_OUTPUT_W;
    output_dims.h = CONV_1_X_N_1_OUTPUT_H;
    output_dims.c = CONV_1_X_N_1_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_1_PAD_X;
    conv_params.padding.h = CONV_1_X_N_1_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_1_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_1_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_1_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_1_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_1_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_1_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_1_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_1_output_shift;

    // Prepare weights_sum_ctx for arm_convolve_1_x_n_s8.
    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size );
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;


    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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
        free(weights_sum_ctx.buf);
    }
    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    // Repeat for arm_convolve_s8 call.
    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    result = arm_convolve_s8(&ctx,
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

void conv_1_x_n_1_null_weight_sum_arm_convolve_1_x_n_s8(void)
{
    /* arm_convolve_1_x_n_s8() only reads weight_sum_ctx->buf on builds with the MVE extension - that is exactly
     * where the NULL guard lives, and exactly where a NULL buf must be diagnosed rather than silently producing
     * garbage output. On any other build the parameter is unread, NULL is accepted, and the call succeeds - so
     * the ARG_ERROR assertion below must not even compile there. */
#if defined(ARM_MATH_MVEI)
    int8_t output[CONV_1_X_N_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx = {0};
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_1_biases;
    const int8_t *kernel_data = conv_1_x_n_1_weights;
    const int8_t *input_data = conv_1_x_n_1_input;

    input_dims.n = CONV_1_X_N_1_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_1_INPUT_W;
    input_dims.h = CONV_1_X_N_1_INPUT_H;
    input_dims.c = CONV_1_X_N_1_IN_CH;
    filter_dims.w = CONV_1_X_N_1_FILTER_X;
    filter_dims.h = CONV_1_X_N_1_FILTER_Y;
    filter_dims.c = CONV_1_X_N_1_IN_CH;
    output_dims.w = CONV_1_X_N_1_OUTPUT_W;
    output_dims.h = CONV_1_X_N_1_OUTPUT_H;
    output_dims.c = CONV_1_X_N_1_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    conv_params.padding.w = CONV_1_X_N_1_PAD_X;
    conv_params.padding.h = CONV_1_X_N_1_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_1_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_1_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_1_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_1_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_1_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_1_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_1_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_1_output_shift;

    /* ctx->buf must be valid: arm_convolve_1_x_n_s8() rejects a NULL one up front, which would mask the
     * weight-sum guard under test. */
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;
    TEST_ASSERT_NOT_NULL(ctx.buf);

    /* weights_sum_ctx is left as {0} (buf == NULL) on purpose: this is the precondition the NULL guard exists to
     * diagnose, so every other argument must be entirely valid. */
    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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

    if (ctx.buf)
    {
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
#endif
}

void conv_1_x_n_2_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_2_biases;
    const int8_t *kernel_data = conv_1_x_n_2_weights;
    const int8_t *input_data = conv_1_x_n_2_input;
    const int8_t *output_ref = conv_1_x_n_2_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_2_DST_SIZE;

    input_dims.n = CONV_1_X_N_2_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_2_INPUT_W;
    input_dims.h = CONV_1_X_N_2_INPUT_H;
    input_dims.c = CONV_1_X_N_2_IN_CH;
    filter_dims.n = CONV_1_X_N_2_OUT_CH;
    filter_dims.w = CONV_1_X_N_2_FILTER_X;
    filter_dims.h = CONV_1_X_N_2_FILTER_Y;
    filter_dims.c = CONV_1_X_N_2_IN_CH;
    output_dims.w = CONV_1_X_N_2_OUTPUT_W;
    output_dims.h = CONV_1_X_N_2_OUTPUT_H;
    output_dims.c = CONV_1_X_N_2_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_2_PAD_X;
    conv_params.padding.h = CONV_1_X_N_2_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_2_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_2_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_2_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_2_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_2_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_2_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_2_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_2_output_shift;

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }

    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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
    memset(output, 0, sizeof(output));

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

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
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_3_arm_convolve_s8(void)
{
    int8_t output[CONV_1_X_N_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_3_biases;
    const int8_t *kernel_data = conv_1_x_n_3_weights;
    const int8_t *input_data = conv_1_x_n_3_input;
    const int8_t *output_ref = conv_1_x_n_3_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_3_DST_SIZE;

    input_dims.n = CONV_1_X_N_3_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_3_INPUT_W;
    input_dims.h = CONV_1_X_N_3_INPUT_H;
    input_dims.c = CONV_1_X_N_3_IN_CH;
    filter_dims.w = CONV_1_X_N_3_FILTER_X;
    filter_dims.h = CONV_1_X_N_3_FILTER_Y;
    filter_dims.c = CONV_1_X_N_3_IN_CH;
    output_dims.w = CONV_1_X_N_3_OUTPUT_W;
    output_dims.h = CONV_1_X_N_3_OUTPUT_H;
    output_dims.c = CONV_1_X_N_3_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_3_PAD_X;
    conv_params.padding.h = CONV_1_X_N_3_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_3_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_3_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_3_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_3_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_3_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_3_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_3_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_3_output_shift;

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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
    memset(output, 0, sizeof(output));

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

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
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_4_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_4_biases;
    const int8_t *kernel_data = conv_1_x_n_4_weights;
    const int8_t *input_data = conv_1_x_n_4_input;
    const int8_t *output_ref = conv_1_x_n_4_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_4_DST_SIZE;

    input_dims.n = CONV_1_X_N_4_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_4_INPUT_W;
    input_dims.h = CONV_1_X_N_4_INPUT_H;
    input_dims.c = CONV_1_X_N_4_IN_CH;
    filter_dims.w = CONV_1_X_N_4_FILTER_X;
    filter_dims.h = CONV_1_X_N_4_FILTER_Y;
    filter_dims.c = CONV_1_X_N_4_IN_CH;
    output_dims.w = CONV_1_X_N_4_OUTPUT_W;
    output_dims.h = CONV_1_X_N_4_OUTPUT_H;
    output_dims.c = CONV_1_X_N_4_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_4_PAD_X;
    conv_params.padding.h = CONV_1_X_N_4_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_4_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_4_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_4_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_4_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_4_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_4_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_4_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_4_output_shift;

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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
    memset(output, 0, sizeof(output));

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

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
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void conv_1_x_n_5_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_5_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_5_biases;
    const int8_t *kernel_data = conv_1_x_n_5_weights;
    const int8_t *input_data = conv_1_x_n_5_input;
    const int8_t *output_ref = conv_1_x_n_5_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_5_DST_SIZE;

    input_dims.n = CONV_1_X_N_5_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_5_INPUT_W;
    input_dims.h = CONV_1_X_N_5_INPUT_H;
    input_dims.c = CONV_1_X_N_5_IN_CH;
    filter_dims.w = CONV_1_X_N_5_FILTER_X;
    filter_dims.h = CONV_1_X_N_5_FILTER_Y;
    filter_dims.c = CONV_1_X_N_5_IN_CH;
    output_dims.w = CONV_1_X_N_5_OUTPUT_W;
    output_dims.h = CONV_1_X_N_5_OUTPUT_H;
    output_dims.c = CONV_1_X_N_5_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_5_PAD_X;
    conv_params.padding.h = CONV_1_X_N_5_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_5_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_5_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_5_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_5_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_5_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_5_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_5_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_5_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_5_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_5_output_shift;

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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
    memset(output, 0, sizeof(output));

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

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

void conv_1_x_n_6_arm_convolve_s8(void)
{
    int8_t output[CONV_1_X_N_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_3_biases;
    const int8_t *kernel_data = conv_1_x_n_3_weights;
    const int8_t *input_data = conv_1_x_n_3_input;

    input_dims.n = CONV_1_X_N_3_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_3_INPUT_W;
    input_dims.h = CONV_1_X_N_3_INPUT_H + 1;  // Intentional error.
    input_dims.c = CONV_1_X_N_3_IN_CH;
    filter_dims.w = CONV_1_X_N_3_FILTER_X;
    filter_dims.h = CONV_1_X_N_3_FILTER_Y;
    filter_dims.c = CONV_1_X_N_3_IN_CH;
    output_dims.w = CONV_1_X_N_3_OUTPUT_W;
    output_dims.h = CONV_1_X_N_3_OUTPUT_H;
    output_dims.c = CONV_1_X_N_3_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_3_PAD_X;
    conv_params.padding.h = CONV_1_X_N_3_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_3_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_3_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_3_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_3_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_3_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_3_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_3_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_3_output_shift;

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

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

    input_dims.h = CONV_1_X_N_3_INPUT_H;
    conv_params.dilation.w = CONV_1_X_N_3_DILATION_X + 1;

    buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    result = arm_convolve_1_x_n_s8(&ctx,
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
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

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

    conv_params.dilation.w = CONV_1_X_N_3_DILATION_X;
    input_dims.c = CONV_1_X_N_3_IN_CH + 1;

    buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;
    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    result = arm_convolve_1_x_n_s8(&ctx,
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
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);

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
}

void conv_1_x_n_7_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_7_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_7_biases;
    const int8_t *kernel_data = conv_1_x_n_7_weights;
    const int8_t *input_data = conv_1_x_n_7_input;
    const int8_t *output_ref = conv_1_x_n_7_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_7_DST_SIZE;

    input_dims.n = CONV_1_X_N_7_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_7_INPUT_W;
    input_dims.h = CONV_1_X_N_7_INPUT_H;
    input_dims.c = CONV_1_X_N_7_IN_CH;
    filter_dims.w = CONV_1_X_N_7_FILTER_X;
    filter_dims.h = CONV_1_X_N_7_FILTER_Y;
    filter_dims.c = CONV_1_X_N_7_IN_CH;
    output_dims.w = CONV_1_X_N_7_OUTPUT_W;
    output_dims.h = CONV_1_X_N_7_OUTPUT_H;
    output_dims.c = CONV_1_X_N_7_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_7_PAD_X;
    conv_params.padding.h = CONV_1_X_N_7_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_7_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_7_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_7_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_7_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_7_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_7_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_7_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_7_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_7_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_7_output_shift;

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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
    memset(output, 0, sizeof(output));
}

void conv_1_x_n_6_generic_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_6_GENERIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_6_generic_biases;
    const int8_t *kernel_data = conv_1_x_n_6_generic_weights;
    const int8_t *input_data = conv_1_x_n_6_generic_input;
    const int8_t *output_ref = conv_1_x_n_6_generic_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_6_GENERIC_DST_SIZE;

    input_dims.n = CONV_1_X_N_6_GENERIC_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_6_GENERIC_INPUT_W;
    input_dims.h = CONV_1_X_N_6_GENERIC_INPUT_H;
    input_dims.c = CONV_1_X_N_6_GENERIC_IN_CH;
    filter_dims.w = CONV_1_X_N_6_GENERIC_FILTER_X;
    filter_dims.h = CONV_1_X_N_6_GENERIC_FILTER_Y;
    filter_dims.c = CONV_1_X_N_6_GENERIC_IN_CH;
    output_dims.w = CONV_1_X_N_6_GENERIC_OUTPUT_W;
    output_dims.h = CONV_1_X_N_6_GENERIC_OUTPUT_H;
    output_dims.c = CONV_1_X_N_6_GENERIC_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_6_GENERIC_PAD_X;
    conv_params.padding.h = CONV_1_X_N_6_GENERIC_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_6_GENERIC_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_6_GENERIC_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_6_GENERIC_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_6_GENERIC_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_6_GENERIC_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_6_GENERIC_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_6_GENERIC_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_6_GENERIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_6_generic_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_6_generic_output_shift;

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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
    memset(output, 0, sizeof(output));

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

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

void conv_1_x_n_8_arm_convolve_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[CONV_1_X_N_8_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weights_sum_ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

    const int32_t *bias_data = conv_1_x_n_8_biases;
    const int8_t *kernel_data = conv_1_x_n_8_weights;
    const int8_t *input_data = conv_1_x_n_8_input;
    const int8_t *output_ref = conv_1_x_n_8_output_ref;
    const int32_t output_ref_size = CONV_1_X_N_8_DST_SIZE;

    input_dims.n = CONV_1_X_N_8_INPUT_BATCHES;
    input_dims.w = CONV_1_X_N_8_INPUT_W;
    input_dims.h = CONV_1_X_N_8_INPUT_H;
    input_dims.c = CONV_1_X_N_8_IN_CH;
    filter_dims.w = CONV_1_X_N_8_FILTER_X;
    filter_dims.h = CONV_1_X_N_8_FILTER_Y;
    filter_dims.c = CONV_1_X_N_8_IN_CH;
    output_dims.w = CONV_1_X_N_8_OUTPUT_W;
    output_dims.h = CONV_1_X_N_8_OUTPUT_H;
    output_dims.c = CONV_1_X_N_8_OUT_CH;

    conv_params.padding.w = CONV_1_X_N_8_PAD_X;
    conv_params.padding.h = CONV_1_X_N_8_PAD_Y;
    conv_params.stride.w = CONV_1_X_N_8_STRIDE_X;
    conv_params.stride.h = CONV_1_X_N_8_STRIDE_Y;
    conv_params.dilation.w = CONV_1_X_N_8_DILATION_X;
    conv_params.dilation.h = CONV_1_X_N_8_DILATION_Y;
    conv_params.input_offset = CONV_1_X_N_8_INPUT_OFFSET;
    conv_params.output_offset = CONV_1_X_N_8_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_1_X_N_8_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_1_X_N_8_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)conv_1_x_n_8_output_mult;
    quant_params.shift = (int32_t *)conv_1_x_n_8_output_shift;

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_convolve_1_x_n_s8(&ctx,
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
    memset(output, 0, sizeof(output));

    {
        int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
        weights_sum_ctx.buf = malloc(weights_sum_buf_size);
        weights_sum_ctx.size = weights_sum_buf_size;
        uint32_t lhs_offset = conv_params.input_offset;
        arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    }
    buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

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

// Issue #367: arm_nn_is_convolve_1_x_n multiplied stride.w by the channel count in 32 bits before any range
// guard, so a stride and channel count that are each in range on their own overflowed the routing predicate --
// signed-overflow UB reachable from the public wrapper sizer. The predicate now folds to 64 bits; both routing
// answers below are what a non-wrapping product selects, pinned against the routed-to sizer so a regression
// that flips the route changes the returned size.
void buffer_size_predicate_overflow_arm_convolve_1_x_n_s8(void)
{
    cmsis_nn_conv_params conv_params;
    // Input W = (output W - 1) * stride.w + filter W: no padding needed, so only the stride.w * c check decides.
    cmsis_nn_dims input_dims = {1, 1, 65538, 65536};
    cmsis_nn_dims filter_dims = {1, 1, 2, 65536};
    cmsis_nn_dims output_dims = {1, 1, 2, 1};

    conv_params.padding.w = 0;
    conv_params.padding.h = 0;
    conv_params.stride.w = 65536;
    conv_params.stride.h = 1;
    conv_params.dilation.w = 1;
    conv_params.dilation.h = 1;
    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = -128;
    conv_params.activation.max = 127;

    // stride.w * c = 65536 * 65536: the product is a multiple of 4, so this stays a 1xN route.
    const int32_t routed_1_x_n =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(routed_1_x_n >= 0);
    TEST_ASSERT_EQUAL(routed_1_x_n,
                      arm_convolve_1_x_n_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(routed_1_x_n,
                      arm_convolve_wrapper_s8_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims));

    // stride.w * c = 65538 * 65535: the product is 2 mod 4, so this routes to the generic sizer instead.
    conv_params.stride.w = 65538;
    input_dims.w = 65540;
    input_dims.c = 65535;
    filter_dims.c = 65535;
    const int32_t routed_generic =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(routed_generic >= 0);
    TEST_ASSERT_EQUAL(routed_generic, arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims));
}

/* Runs a 1xN layer (16 output channels, input offset 3) through arm_convolve_wrapper_s8() and arm_convolve_s8() and
   expects the same status and output, with the scratch each one's sizer gives. The layer's padding is one the 1xN
   kernels do not handle, so the wrapper must route it elsewhere. */
static void wrapper_matches_convolve_s8(const int32_t in_w,
                                        const int32_t in_c,
                                        const int32_t k_w,
                                        const int32_t stride,
                                        const int32_t pad,
                                        const int32_t out_w)
{
    enum
    {
        max_in_w = 40,
        max_in_c = 27,
        out_c = 16,
        max_k_w = 5,
        max_out_w = 18
    };
    static int8_t input[max_in_w * max_in_c];
    static int8_t kernel[out_c * max_k_w * max_in_c];
    static int32_t bias[out_c];
    static int32_t mult[out_c];
    static int32_t shift[out_c];
    static int32_t wsum[out_c + 4];
    static int8_t expected[max_out_w * out_c];
    static int8_t output[max_out_w * out_c];
    TEST_ASSERT_TRUE(in_w <= max_in_w && in_c <= max_in_c && k_w <= max_k_w && out_w <= max_out_w);
    for (int i = 0; i < (int)sizeof(input); i++)
    {
        input[i] = (int8_t)((i * 37) % 251 - 125);
    }
    for (int i = 0; i < (int)sizeof(kernel); i++)
    {
        kernel[i] = (int8_t)((i * 11) % 29 - 14);
    }
    for (int i = 0; i < out_c; i++)
    {
        bias[i] = i * 97 - 700;
        mult[i] = 1300000000 + i * 1000;
        shift[i] = -7;
    }
    memset(expected, 0, sizeof(expected));
    memset(output, 0x55, sizeof(output));
    const cmsis_nn_conv_params conv_params = {.input_offset = 3,
                                              .output_offset = -2,
                                              .stride = {stride, 1},
                                              .padding = {pad, 0},
                                              .dilation = {1, 1},
                                              .activation = {-128, 127}};
    const cmsis_nn_per_channel_quant_params quant = {mult, shift};
    const cmsis_nn_dims input_dims = {1, 1, in_w, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 1, k_w, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, 1, out_w, out_c};
    TEST_ASSERT_TRUE(arm_nn_is_convolve_1_x_n(&conv_params, &input_dims, &filter_dims));
    TEST_ASSERT_FALSE(arm_nn_convolve_1_x_n_padding_supported(&conv_params, &input_dims, &filter_dims, &output_dims));
    // The weight sums are an MVE-only input; other builds report that and ignore the buffer.
    const arm_cmsis_nn_status wsum_status =
        arm_convolve_weight_sum(wsum, kernel, &input_dims, &filter_dims, &output_dims, conv_params.input_offset, bias);
#if defined(ARM_MATH_MVEI)
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, wsum_status);
#else
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, wsum_status);
#endif
    const cmsis_nn_context wsum_ctx = {wsum, (int32_t)sizeof(wsum)};

    const int32_t ref_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    cmsis_nn_context ref_ctx = {ref_size > 0 ? malloc(ref_size) : NULL, ref_size};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_s8(&ref_ctx,
                                      &wsum_ctx,
                                      &conv_params,
                                      &quant,
                                      &input_dims,
                                      input,
                                      &filter_dims,
                                      kernel,
                                      &bias_dims,
                                      bias,
                                      NULL,
                                      &output_dims,
                                      expected));

    const int32_t size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(size >= 0);
    cmsis_nn_context ctx = {size > 0 ? malloc(size) : NULL, size};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_wrapper_s8(&ctx,
                                              &wsum_ctx,
                                              &conv_params,
                                              &quant,
                                              &input_dims,
                                              input,
                                              &filter_dims,
                                              kernel,
                                              &bias_dims,
                                              bias,
                                              &output_dims,
                                              output));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, out_w * out_c);
    free(ctx.buf);
    free(ref_ctx.buf);
}

/* 1xN layers whose horizontal padding arm_convolve_1_x_n_s8() does not handle, where total pad is
   (output W - 1) * stride + filter W - input W. The wrapper must match arm_convolve_s8(). */
void wrapper_unsupported_padding_arm_convolve_1_x_n_s8(void)
{
    // VALID, stride leaves trailing input unused: total pad -3 (the kernel fails), and -5 with one output column.
    wrapper_matches_convolve_s8(40, 27, 5, 4, 0, 9);
    wrapper_matches_convolve_s8(10, 27, 5, 4, 0, 1);
    // VALID, total pad -1: the kernel accepts it and replaces the last tap of the final window with padding.
    wrapper_matches_convolve_s8(10, 4, 3, 2, 0, 4);
    // SAME with more padded output columns than output columns.
    wrapper_matches_convolve_s8(3, 4, 5, 1, 2, 3);
    // Explicit padding wider than the filter: the outer windows read no input column.
    wrapper_matches_convolve_s8(10, 4, 3, 1, 5, 18);
    // A filter wider than the input.
    wrapper_matches_convolve_s8(4, 4, 5, 1, 1, 2);
}

/* Runs a 1xN layer (input offset 3, 16 output channels) through arm_convolve_wrapper_s8() and arm_convolve_s8() and
   returns whether the status and output match. */
static bool wrapper_agrees_with_convolve_s8(const int32_t in_w,
                                            const int32_t in_c,
                                            const int32_t k_w,
                                            const int32_t stride,
                                            const int32_t pad_w,
                                            const int32_t pad_h,
                                            const int32_t out_w,
                                            const int32_t out_h)
{
    enum
    {
        max_in_w = 24,
        max_in_c = 12,
        out_c = 16,
        max_k_w = 6,
        max_out = 24 * 3
    };
    static int8_t input[max_in_w * max_in_c];
    static int8_t kernel[out_c * max_k_w * max_in_c];
    static int32_t bias[out_c];
    static int32_t mult[out_c];
    static int32_t shift[out_c];
    static int32_t wsum[out_c + 4];
    static int8_t expected[max_out * out_c];
    static int8_t output[max_out * out_c];
    TEST_ASSERT_TRUE(in_w <= max_in_w && in_c <= max_in_c && k_w <= max_k_w && out_w * out_h <= max_out);
    for (int i = 0; i < (int)sizeof(input); i++)
    {
        input[i] = (int8_t)((i * 37) % 251 - 125);
    }
    for (int i = 0; i < (int)sizeof(kernel); i++)
    {
        kernel[i] = (int8_t)((i * 11) % 29 - 14);
    }
    for (int i = 0; i < out_c; i++)
    {
        bias[i] = i * 97 - 700;
        mult[i] = 1300000000 + i * 1000;
        shift[i] = -7;
    }
    memset(expected, 0, sizeof(expected));
    memset(output, 0x55, sizeof(output));
    const cmsis_nn_conv_params conv_params = {.input_offset = 3,
                                              .output_offset = -2,
                                              .stride = {stride, 1},
                                              .padding = {pad_w, pad_h},
                                              .dilation = {1, 1},
                                              .activation = {-128, 127}};
    const cmsis_nn_per_channel_quant_params quant = {mult, shift};
    const cmsis_nn_dims input_dims = {1, 1, in_w, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 1, k_w, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, out_h, out_w, out_c};
    // The weight sums are an MVE-only input; other builds ignore the buffer.
    (void)arm_convolve_weight_sum(
        wsum, kernel, &input_dims, &filter_dims, &output_dims, conv_params.input_offset, bias);
    const cmsis_nn_context wsum_ctx = {wsum, (int32_t)sizeof(wsum)};

    const int32_t ref_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    cmsis_nn_context ref_ctx = {ref_size > 0 ? malloc(ref_size) : NULL, ref_size};
    const arm_cmsis_nn_status ref_status = arm_convolve_s8(&ref_ctx,
                                                           &wsum_ctx,
                                                           &conv_params,
                                                           &quant,
                                                           &input_dims,
                                                           input,
                                                           &filter_dims,
                                                           kernel,
                                                           &bias_dims,
                                                           bias,
                                                           NULL,
                                                           &output_dims,
                                                           expected);
    const int32_t size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    cmsis_nn_context ctx = {size > 0 ? malloc(size) : NULL, size};
    const arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&ctx,
                                                               &wsum_ctx,
                                                               &conv_params,
                                                               &quant,
                                                               &input_dims,
                                                               input,
                                                               &filter_dims,
                                                               kernel,
                                                               &bias_dims,
                                                               bias,
                                                               &output_dims,
                                                               output);
    free(ctx.buf);
    free(ref_ctx.buf);
    return (size >= 0) && (ref_status == ARM_CMSIS_NN_SUCCESS) && (status == ARM_CMSIS_NN_SUCCESS) &&
        (memcmp(expected, output, (size_t)(out_w * out_h * out_c)) == 0);
}

/* Every TFLite SAME and VALID 1xN layer with input width 1..24, filter width 1..6 and stride 1..4 (channel counts
   chosen so stride * channels is a multiple of 4, as the 1xN route requires) must match arm_convolve_s8(). This
   covers right padding that arm_convolve_1_x_n_s8() stages incorrectly for stride > 1, e.g. SAME with input
   width 6, filter width 5, stride 2. */
void wrapper_same_valid_sweep_arm_convolve_1_x_n_s8(void)
{
    const int32_t channels_stride[][2] = {{4, 1}, {4, 2}, {4, 3}, {4, 4}, {12, 1}, {12, 2}, {12, 3}, {3, 4}, {12, 4}};
    int32_t mismatches = 0;
    char first[96] = "none";
    for (size_t i = 0; i < sizeof(channels_stride) / sizeof(channels_stride[0]); i++)
    {
        const int32_t in_c = channels_stride[i][0];
        const int32_t stride = channels_stride[i][1];
        for (int32_t in_w = 1; in_w <= 24; in_w++)
        {
            for (int32_t k_w = 1; k_w <= 6 && k_w <= in_w; k_w++)
            {
                // SAME: output width ceil(in_w / stride), the smaller half of the total padding on the left.
                const int32_t same_w = (in_w + stride - 1) / stride;
                const int32_t same_total = ARM_NN_MAX((same_w - 1) * stride + k_w - in_w, 0);
                // VALID: no padding.
                const int32_t valid_w = (in_w - k_w) / stride + 1;
                const int32_t layers[2][2] = {{same_total / 2, same_w}, {0, valid_w}};
                for (int32_t l = 0; l < 2; l++)
                {
                    if (!wrapper_agrees_with_convolve_s8(in_w, in_c, k_w, stride, layers[l][0], 0, layers[l][1], 1))
                    {
                        if (mismatches++ == 0)
                        {
                            snprintf(first,
                                     sizeof(first),
                                     "in_w %d c %d k_w %d stride %d pad %d out_w %d",
                                     (int)in_w,
                                     (int)in_c,
                                     (int)k_w,
                                     (int)stride,
                                     (int)layers[l][0],
                                     (int)layers[l][1]);
                        }
                    }
                }
            }
        }
    }
    TEST_ASSERT_EQUAL_MESSAGE(0, mismatches, first);
}

/* A 1xN input and filter with vertical padding (output height 3) is not a single-row 1xN convolution; the wrapper must
   match arm_convolve_s8(). */
void wrapper_vertical_padding_arm_convolve_1_x_n_s8(void)
{
    TEST_ASSERT_TRUE(wrapper_agrees_with_convolve_s8(6, 4, 3, 1, 1, 1, 6, 3));
}

/* A VALID 1xN layer whose last window ends on the last input column has no right-padded windows (input width 6,
   filter width 2, stride 2). arm_convolve_wrapper_s8() must not read past its input, which ends at an unmapped gap. */
void wrapper_valid_input_bounds_arm_convolve_1_x_n_s8(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    enum
    {
        in_w = 6,
        in_c = 4,
        out_c = 16,
        k_w = 2,
        out_w = 3
    };
    int8_t input[in_w * in_c];
    int8_t kernel[out_c * k_w * in_c];
    int32_t bias[out_c];
    int32_t mult[out_c];
    int32_t shift[out_c];
    int32_t wsum[out_c];
    int8_t expected[out_w * out_c];
    int8_t output[out_w * out_c];
    for (int i = 0; i < (int)sizeof(input); i++)
    {
        input[i] = (int8_t)((i * 37) % 251 - 125);
    }
    for (int i = 0; i < (int)sizeof(kernel); i++)
    {
        kernel[i] = (int8_t)((i * 11) % 29 - 14);
    }
    for (int i = 0; i < out_c; i++)
    {
        bias[i] = i * 97 - 700;
        mult[i] = 1300000000 + i * 1000;
        shift[i] = -7;
    }
    const cmsis_nn_conv_params conv_params = {.input_offset = 3,
                                              .output_offset = -2,
                                              .stride = {2, 1},
                                              .padding = {0, 0},
                                              .dilation = {1, 1},
                                              .activation = {-128, 127}};
    const cmsis_nn_per_channel_quant_params quant = {mult, shift};
    const cmsis_nn_dims input_dims = {1, 1, in_w, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 1, k_w, in_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, 1, out_w, out_c};
    TEST_ASSERT_TRUE(arm_nn_is_convolve_1_x_n(&conv_params, &input_dims, &filter_dims) &&
                     arm_nn_convolve_1_x_n_padding_supported(&conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_weight_sum(
                          wsum, kernel, &input_dims, &filter_dims, &output_dims, conv_params.input_offset, bias));
    const cmsis_nn_context wsum_ctx = {wsum, (int32_t)sizeof(wsum)};

    const int32_t ref_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    cmsis_nn_context ref_ctx = {ref_size > 0 ? malloc(ref_size) : NULL, ref_size};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_s8(&ref_ctx,
                                      &wsum_ctx,
                                      &conv_params,
                                      &quant,
                                      &input_dims,
                                      input,
                                      &filter_dims,
                                      kernel,
                                      &bias_dims,
                                      bias,
                                      NULL,
                                      &output_dims,
                                      expected));
    free(ref_ctx.buf);

    const int32_t size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(size >= 0);
    cmsis_nn_context ctx = {size > 0 ? malloc(size) : NULL, size};
    const int8_t *guarded_input = guard_place(input, sizeof(input));
    guard_gap_enable();
    const arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&ctx,
                                                               &wsum_ctx,
                                                               &conv_params,
                                                               &quant,
                                                               &input_dims,
                                                               guarded_input,
                                                               &filter_dims,
                                                               kernel,
                                                               &bias_dims,
                                                               bias,
                                                               &output_dims,
                                                               output);
    guard_gap_disable();
    free(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, out_w * out_c);
#endif
}

