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
#include <unity.h>

#include "../TestData/conv_1_x_n_1/test_data.h"
#include "../TestData/conv_1_x_n_2/test_data.h"
#include "../TestData/conv_1_x_n_3/test_data.h"
#include "../TestData/conv_1_x_n_4/test_data.h"
#include "../TestData/conv_1_x_n_5/test_data.h"
#include "../TestData/conv_1_x_n_6_generic/test_data.h"
#include "../TestData/conv_1_x_n_7/test_data.h"
#include "../TestData/conv_1_x_n_8/test_data.h"

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
    cmsis_nn_dims input_dims = {1, 1, 4, 65536};
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
    input_dims.c = 65535;
    filter_dims.c = 65535;
    const int32_t routed_generic =
        arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_TRUE(routed_generic >= 0);
    TEST_ASSERT_EQUAL(routed_generic, arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims));
}
