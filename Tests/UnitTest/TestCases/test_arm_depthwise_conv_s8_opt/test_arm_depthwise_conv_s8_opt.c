/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include <string.h>
#include <unity.h>

#include "../TestData/basic/test_data.h"
#include "../TestData/depthwise_eq_in_out_ch/test_data.h"
#include "../TestData/depthwise_null_bias_0/test_data.h"
#include "../TestData/depthwise_out_activation/test_data.h"
#include "../TestData/depthwise_sub_block/test_data.h"
#include "../TestData/depthwise_x_stride/test_data.h"
#include "../Utils/utils.h"
#include "../Utils/validate.h"

void basic_arm_depthwise_conv_s8_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[BASIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_biases;
    const int8_t *kernel_data = basic_weights;
    const int8_t *input_data = basic_input;

    input_dims.n = BASIC_INPUT_BATCHES;
    input_dims.w = BASIC_INPUT_W;
    input_dims.h = BASIC_INPUT_H;
    input_dims.c = BASIC_IN_CH;
    filter_dims.w = BASIC_FILTER_X;
    filter_dims.h = BASIC_FILTER_Y;
    output_dims.w = BASIC_OUTPUT_W;
    output_dims.h = BASIC_OUTPUT_H;
    output_dims.c = BASIC_OUT_CH;

    dw_conv_params.padding.w = BASIC_PAD_X;
    dw_conv_params.padding.h = BASIC_PAD_Y;
    dw_conv_params.stride.w = BASIC_STRIDE_X;
    dw_conv_params.stride.h = BASIC_STRIDE_Y;
    dw_conv_params.dilation.w = BASIC_DILATION_X;
    dw_conv_params.dilation.h = BASIC_DILATION_Y;

    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = BASIC_INPUT_OFFSET;
    dw_conv_params.output_offset = BASIC_OUTPUT_OFFSET;
    dw_conv_params.activation.min = BASIC_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = BASIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_output_mult;
    quant_params.shift = (int32_t *)basic_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
                                      ctx.buf,
                                      kernel_data,
                                      &dw_conv_params,
                                      &input_dims,
                                      &filter_dims,
                                      &output_dims,
                                      lhs_offset,
                                      bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8_opt(&ctx,
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
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, basic_output_ref, BASIC_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
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
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, basic_output_ref, BASIC_DST_SIZE));
}

void depthwise_null_weight_sum_arm_depthwise_conv_s8_opt(void)
{
    /* arm_depthwise_conv_s8_opt() only reads weight_sum_ctx->buf on builds where ARM_MATH_DSP and ARM_MATH_MVEI
     * are both defined - that is exactly where the NULL guard lives, and exactly where a NULL buf must be
     * diagnosed rather than silently producing garbage output. On any other build the parameter is unread, NULL
     * is accepted, and the call succeeds - so the ARG_ERROR assertion below must not even compile there. */
#if defined(ARM_MATH_DSP) && defined(ARM_MATH_MVEI)
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_ARG_ERROR;
    int8_t output[BASIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context weight_sum_ctx = {0};
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = basic_biases;
    const int8_t *kernel_data = basic_weights;
    const int8_t *input_data = basic_input;

    input_dims.n = BASIC_INPUT_BATCHES;
    input_dims.w = BASIC_INPUT_W;
    input_dims.h = BASIC_INPUT_H;
    input_dims.c = BASIC_IN_CH;
    filter_dims.w = BASIC_FILTER_X;
    filter_dims.h = BASIC_FILTER_Y;
    output_dims.w = BASIC_OUTPUT_W;
    output_dims.h = BASIC_OUTPUT_H;
    output_dims.c = BASIC_OUT_CH;

    dw_conv_params.padding.w = BASIC_PAD_X;
    dw_conv_params.padding.h = BASIC_PAD_Y;
    dw_conv_params.stride.w = BASIC_STRIDE_X;
    dw_conv_params.stride.h = BASIC_STRIDE_Y;
    dw_conv_params.dilation.w = BASIC_DILATION_X;
    dw_conv_params.dilation.h = BASIC_DILATION_Y;

    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = BASIC_INPUT_OFFSET;
    dw_conv_params.output_offset = BASIC_OUTPUT_OFFSET;
    dw_conv_params.activation.min = BASIC_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = BASIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_output_mult;
    quant_params.shift = (int32_t *)basic_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc((size_t)ctx.size);
    TEST_ASSERT_TRUE(ctx.size == 0 || ctx.buf != NULL);

    /* weight_sum_ctx is left as {0} (buf == NULL) on purpose: this is the precondition the NULL guard exists to
     * diagnose, so ctx must otherwise be entirely valid. */
    arm_cmsis_nn_status result = arm_depthwise_conv_s8_opt(&ctx,
                                                           &weight_sum_ctx,
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
        free(ctx.buf);
    }

    TEST_ASSERT_EQUAL(expected, result);
#endif
}

void depthwise_eq_in_out_ch_arm_depthwise_conv_s8_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_EQ_IN_OUT_CH_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(depthwise_eq_in_out_ch_biases, DEPTHWISE_EQ_IN_OUT_CH_IN_CH);
    const int8_t *kernel_data = depthwise_eq_in_out_ch_weights;
    const int8_t *input_data = depthwise_eq_in_out_ch_input;

    input_dims.n = DEPTHWISE_EQ_IN_OUT_CH_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_EQ_IN_OUT_CH_INPUT_W;
    input_dims.h = DEPTHWISE_EQ_IN_OUT_CH_INPUT_H;
    input_dims.c = DEPTHWISE_EQ_IN_OUT_CH_IN_CH;
    filter_dims.w = DEPTHWISE_EQ_IN_OUT_CH_FILTER_X;
    filter_dims.h = DEPTHWISE_EQ_IN_OUT_CH_FILTER_Y;
    output_dims.w = DEPTHWISE_EQ_IN_OUT_CH_OUTPUT_W;
    output_dims.h = DEPTHWISE_EQ_IN_OUT_CH_OUTPUT_H;
    output_dims.c = DEPTHWISE_EQ_IN_OUT_CH_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_EQ_IN_OUT_CH_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_EQ_IN_OUT_CH_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_EQ_IN_OUT_CH_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_EQ_IN_OUT_CH_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_EQ_IN_OUT_CH_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_EQ_IN_OUT_CH_DILATION_Y;

    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = DEPTHWISE_EQ_IN_OUT_CH_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_EQ_IN_OUT_CH_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_EQ_IN_OUT_CH_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_EQ_IN_OUT_CH_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_eq_in_out_ch_output_mult;
    quant_params.shift = (int32_t *)depthwise_eq_in_out_ch_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
                                      ctx.buf,
                                      kernel_data,
                                      &dw_conv_params,
                                      &input_dims,
                                      &filter_dims,
                                      &output_dims,
                                      lhs_offset,
                                      bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_eq_in_out_ch_output_ref, DEPTHWISE_EQ_IN_OUT_CH_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
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
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_eq_in_out_ch_output_ref, DEPTHWISE_EQ_IN_OUT_CH_DST_SIZE));
}

void depthwise_sub_block_arm_depthwise_conv_s8_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_SUB_BLOCK_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(depthwise_sub_block_biases, DEPTHWISE_SUB_BLOCK_IN_CH);
    const int8_t *kernel_data = depthwise_sub_block_weights;
    const int8_t *input_data = depthwise_sub_block_input;

    input_dims.n = DEPTHWISE_SUB_BLOCK_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_SUB_BLOCK_INPUT_W;
    input_dims.h = DEPTHWISE_SUB_BLOCK_INPUT_H;
    input_dims.c = DEPTHWISE_SUB_BLOCK_IN_CH;
    filter_dims.w = DEPTHWISE_SUB_BLOCK_FILTER_X;
    filter_dims.h = DEPTHWISE_SUB_BLOCK_FILTER_Y;
    output_dims.w = DEPTHWISE_SUB_BLOCK_OUTPUT_W;
    output_dims.h = DEPTHWISE_SUB_BLOCK_OUTPUT_H;
    output_dims.c = DEPTHWISE_SUB_BLOCK_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_SUB_BLOCK_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_SUB_BLOCK_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_SUB_BLOCK_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_SUB_BLOCK_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_SUB_BLOCK_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_SUB_BLOCK_DILATION_Y;

    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = DEPTHWISE_SUB_BLOCK_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_SUB_BLOCK_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_SUB_BLOCK_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_SUB_BLOCK_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_sub_block_output_mult;
    quant_params.shift = (int32_t *)depthwise_sub_block_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
                                      ctx.buf,
                                      kernel_data,
                                      &dw_conv_params,
                                      &input_dims,
                                      &filter_dims,
                                      &output_dims,
                                      lhs_offset,
                                      bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_sub_block_output_ref, DEPTHWISE_SUB_BLOCK_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;

    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
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
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_sub_block_output_ref, DEPTHWISE_SUB_BLOCK_DST_SIZE));
}

void depthwise_out_activation_arm_depthwise_conv_s8_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_OUT_ACTIVATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_OUT_ACTIVATION_DST_SIZE;
    const int32_t *bias_data = get_bias_address(depthwise_out_activation_biases, DEPTHWISE_OUT_ACTIVATION_OUT_CH);
    const int8_t *kernel_data = depthwise_out_activation_weights;
    const int8_t *input_data = depthwise_out_activation_input;

    input_dims.n = DEPTHWISE_OUT_ACTIVATION_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_OUT_ACTIVATION_INPUT_W;
    input_dims.h = DEPTHWISE_OUT_ACTIVATION_INPUT_H;
    input_dims.c = DEPTHWISE_OUT_ACTIVATION_IN_CH;
    filter_dims.w = DEPTHWISE_OUT_ACTIVATION_FILTER_X;
    filter_dims.h = DEPTHWISE_OUT_ACTIVATION_FILTER_Y;
    output_dims.w = DEPTHWISE_OUT_ACTIVATION_OUTPUT_W;
    output_dims.h = DEPTHWISE_OUT_ACTIVATION_OUTPUT_H;
    output_dims.c = DEPTHWISE_OUT_ACTIVATION_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_OUT_ACTIVATION_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_OUT_ACTIVATION_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_OUT_ACTIVATION_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_OUT_ACTIVATION_STRIDE_Y;
    dw_conv_params.ch_mult = DEPTHWISE_OUT_ACTIVATION_CH_MULT;
    dw_conv_params.dilation.w = DEPTHWISE_OUT_ACTIVATION_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_OUT_ACTIVATION_DILATION_Y;

    dw_conv_params.input_offset = DEPTHWISE_OUT_ACTIVATION_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_OUT_ACTIVATION_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_OUT_ACTIVATION_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_OUT_ACTIVATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_out_activation_output_mult;
    quant_params.shift = (int32_t *)depthwise_out_activation_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
                                      ctx.buf,
                                      kernel_data,
                                      &dw_conv_params,
                                      &input_dims,
                                      &filter_dims,
                                      &output_dims,
                                      lhs_offset,
                                      bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_out_activation_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(buf_size, ctx.size);

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
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
    TEST_ASSERT_TRUE(validate(output, depthwise_out_activation_output_ref, output_ref_size));
}

void depthwise_null_bias_0_arm_depthwise_conv_s8_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_NULL_BIAS_0_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(depthwise_null_bias_0_biases, DEPTHWISE_NULL_BIAS_0_OUT_CH);
    const int8_t *kernel_data = depthwise_null_bias_0_weights;
    const int8_t *input_data = depthwise_null_bias_0_input;

    input_dims.n = DEPTHWISE_NULL_BIAS_0_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_NULL_BIAS_0_INPUT_W;
    input_dims.h = DEPTHWISE_NULL_BIAS_0_INPUT_H;
    input_dims.c = DEPTHWISE_NULL_BIAS_0_IN_CH;
    filter_dims.w = DEPTHWISE_NULL_BIAS_0_FILTER_X;
    filter_dims.h = DEPTHWISE_NULL_BIAS_0_FILTER_Y;
    output_dims.w = DEPTHWISE_NULL_BIAS_0_OUTPUT_W;
    output_dims.h = DEPTHWISE_NULL_BIAS_0_OUTPUT_H;
    output_dims.c = DEPTHWISE_NULL_BIAS_0_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_NULL_BIAS_0_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_NULL_BIAS_0_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_NULL_BIAS_0_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_NULL_BIAS_0_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_NULL_BIAS_0_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_NULL_BIAS_0_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_NULL_BIAS_0_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_NULL_BIAS_0_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_NULL_BIAS_0_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_NULL_BIAS_0_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_NULL_BIAS_0_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_null_bias_0_output_mult;
    quant_params.shift = (int32_t *)depthwise_null_bias_0_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
                                      ctx.buf,
                                      kernel_data,
                                      &dw_conv_params,
                                      &input_dims,
                                      &filter_dims,
                                      &output_dims,
                                      lhs_offset,
                                      bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_null_bias_0_output_ref, DEPTHWISE_NULL_BIAS_0_DST_SIZE));

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(buf_size, ctx.size);

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
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
    TEST_ASSERT_TRUE(validate(output, depthwise_null_bias_0_output_ref, DEPTHWISE_NULL_BIAS_0_DST_SIZE));
}

void depthwise_x_stride_arm_depthwise_conv_s8_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_X_STRIDE_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = get_bias_address(depthwise_x_stride_biases, DEPTHWISE_X_STRIDE_IN_CH);
    const int8_t *kernel_data = depthwise_x_stride_weights;
    const int8_t *input_data = depthwise_x_stride_input;

    input_dims.n = DEPTHWISE_X_STRIDE_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_X_STRIDE_INPUT_W;
    input_dims.h = DEPTHWISE_X_STRIDE_INPUT_H;
    input_dims.c = DEPTHWISE_X_STRIDE_IN_CH;
    filter_dims.w = DEPTHWISE_X_STRIDE_FILTER_X;
    filter_dims.h = DEPTHWISE_X_STRIDE_FILTER_Y;
    output_dims.w = DEPTHWISE_X_STRIDE_OUTPUT_W;
    output_dims.h = DEPTHWISE_X_STRIDE_OUTPUT_H;
    output_dims.c = DEPTHWISE_X_STRIDE_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_X_STRIDE_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_X_STRIDE_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_X_STRIDE_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_X_STRIDE_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_X_STRIDE_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_X_STRIDE_DILATION_Y;

    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = DEPTHWISE_X_STRIDE_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_X_STRIDE_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_X_STRIDE_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_X_STRIDE_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_x_stride_output_mult;
    quant_params.shift = (int32_t *)depthwise_x_stride_output_shift;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(ctx.size > 0);
#else
    TEST_ASSERT_EQUAL(ctx.size, 0);
#endif

    ctx.buf = malloc(ctx.size);

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
                                      ctx.buf,
                                      kernel_data,
                                      &dw_conv_params,
                                      &input_dims,
                                      &filter_dims,
                                      &output_dims,
                                      lhs_offset,
                                      bias_data);

    arm_cmsis_nn_status result = arm_depthwise_conv_s8_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_x_stride_output_ref, DEPTHWISE_X_STRIDE_DST_SIZE));

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, ctx.size);

    ctx.buf = malloc(wrapper_buf_size);

    weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    lhs_offset = dw_conv_params.input_offset;
    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
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
        memset(ctx.buf, 0, wrapper_buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_x_stride_output_ref, DEPTHWISE_X_STRIDE_DST_SIZE));
}

void depthwise_nt_t_tail_arm_depthwise_conv_s8_opt(void)
{
#if defined(ARM_MATH_MVEI)
    enum
    {
        max_channels = 126,
        packed_patches = 5,
        channel_block = 124
    };
    const int32_t channels_to_test[] = {1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 123, 124, 125, 126};
    const int8_t input_guard = 0x35;
    const int8_t output_guard = 0x6B;
    const int32_t input_offset = 4;
    const int32_t output_offset = -3;
    const int32_t activation_min = -10;
    const int32_t activation_max = 11;
    int8_t input[packed_patches * max_channels];
    int8_t kernel[max_channels];
    int32_t bias[max_channels];
    int32_t multiplier[max_channels];
    int32_t shift[max_channels];
    int32_t weight_sum[max_channels + 4];
    int8_t output_storage[packed_patches * max_channels + 8];

    for (int i = 0; i < max_channels; i++)
    {
        kernel[i] = (int8_t)((i * 11) % 19 - 9);
        bias[i] = (i * 29) % 127 - 63;
        multiplier[i] = (i % 4 == 0) ? (1 << 30) : ((i % 4 == 1) ? (1 << 29) : ((i % 4 == 2) ? (3 << 29) : (1 << 28)));
        shift[i] = (i % 5) - 2;
    }

    cmsis_nn_dims input_dims = {1, packed_patches, 1, max_channels};
    cmsis_nn_dims filter_dims = {1, 1, 1, max_channels};
    cmsis_nn_dims bias_dims = {1, 1, 1, max_channels};
    cmsis_nn_dims output_dims = {1, packed_patches, 1, max_channels};
    cmsis_nn_dw_conv_params dw_conv_params = {
        .input_offset = input_offset,
        .output_offset = output_offset,
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
        .ch_mult = 1,
        .activation = {activation_min, activation_max},
    };
    cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multiplier,
        .shift = shift,
    };
    cmsis_nn_context ctx;
    cmsis_nn_context weight_sum_ctx;

    ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    TEST_ASSERT_EQUAL(4 * channel_block, ctx.size);
    ctx.buf = malloc(ctx.size);
    TEST_ASSERT_NOT_NULL(ctx.buf);
    weight_sum_ctx.buf = weight_sum;
    weight_sum_ctx.size = max_channels * (int32_t)sizeof(int32_t);

    for (size_t test_index = 0; test_index < sizeof(channels_to_test) / sizeof(channels_to_test[0]); test_index++)
    {
        const int32_t channels = channels_to_test[test_index];
        input_dims.c = channels;
        filter_dims.c = channels;
        bias_dims.c = channels;
        output_dims.c = channels;

        for (int i_patch = 0; i_patch < packed_patches; i_patch++)
        {
            for (int i_ch = 0; i_ch < channels; i_ch++)
            {
                input[i_patch * channels + i_ch] = (int8_t)((i_patch * 17 + i_ch * 7) % 31 - 15);
            }
        }
        memset(input + packed_patches * channels, input_guard, sizeof(input) - packed_patches * channels);
        memset(output_storage, output_guard, sizeof(output_storage));
        memset(weight_sum + channels, 0xA5, (max_channels + 4 - channels) * sizeof(int32_t));

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_depthwise_convolve_weight_sum(weight_sum,
                                                            ctx.buf,
                                                            kernel,
                                                            &dw_conv_params,
                                                            &input_dims,
                                                            &filter_dims,
                                                            &output_dims,
                                                            input_offset,
                                                            bias));

        const arm_cmsis_nn_status result = arm_depthwise_conv_s8_opt(&ctx,
                                                                     &weight_sum_ctx,
                                                                     &dw_conv_params,
                                                                     &quant_params,
                                                                     &input_dims,
                                                                     input,
                                                                     &filter_dims,
                                                                     kernel,
                                                                     &bias_dims,
                                                                     bias,
                                                                     &output_dims,
                                                                     output_storage + 4);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        for (int i = 0; i < 4; i++)
        {
            TEST_ASSERT_EQUAL_INT8(output_guard, output_storage[i]);
            TEST_ASSERT_EQUAL_INT8(output_guard, output_storage[packed_patches * channels + 4 + i]);
        }

        int saw_clip = 0;
        for (int i_patch = 0; i_patch < packed_patches; i_patch++)
        {
            for (int i_ch = 0; i_ch < channels; i_ch++)
            {
                int32_t acc = bias[i_ch] + (input[i_patch * channels + i_ch] + input_offset) * kernel[i_ch];
                int32_t expected = arm_nn_requantize(acc, multiplier[i_ch], shift[i_ch]) + output_offset;
                expected = ARM_NN_MAX(expected, activation_min);
                expected = ARM_NN_MIN(expected, activation_max);
                saw_clip |= expected == activation_min || expected == activation_max;
                TEST_ASSERT_EQUAL_INT8((int8_t)expected, output_storage[4 + i_patch * channels + i_ch]);
            }
        }
        TEST_ASSERT_TRUE(saw_clip);
        for (int i = channels; i < max_channels + 4; i++)
        {
            TEST_ASSERT_EQUAL_INT32((int32_t)0xA5A5A5A5, weight_sum[i]);
        }
    }

    memset(ctx.buf, 0, ctx.size);
    free(ctx.buf);
#endif
}

void depthwise_boundary_matrix_arm_depthwise_conv_s8_opt(void)
{
#if defined(ARM_MATH_MVEI)
    typedef struct
    {
        int32_t input_w;
        int32_t input_h;
        int32_t filter_w;
        int32_t filter_h;
        int32_t pad_w;
        int32_t pad_h;
        int32_t stride_w;
        int32_t stride_h;
        int32_t dilation_w;
        int32_t dilation_h;
        int32_t use_wrapper;
    } depthwise_test_case;

    const int32_t channels_to_test[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 123, 124, 125, 126};
    const depthwise_test_case test_cases[] = {
        {5, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0},
        {6, 1, 2, 1, 0, 0, 1, 1, 1, 1, 0},
        {7, 1, 3, 1, 0, 0, 1, 1, 1, 1, 0},
        {8, 1, 4, 1, 0, 0, 1, 1, 1, 1, 0},
        {9, 1, 5, 1, 0, 0, 1, 1, 1, 1, 0},
        {19, 1, 15, 1, 0, 0, 1, 1, 1, 1, 0},
        {20, 1, 16, 1, 0, 0, 1, 1, 1, 1, 0},
        {21, 1, 17, 1, 0, 0, 1, 1, 1, 1, 0},
        {7, 6, 3, 3, 1, 1, 2, 2, 1, 1, 0},
        {9, 7, 3, 2, 2, 1, 2, 1, 1, 1, 0},
        {9, 7, 3, 3, 2, 2, 1, 1, 2, 2, 1},
    };
    const int32_t input_offset = 3;
    const int32_t output_offset = -2;
    const int32_t activation_min = -101;
    const int32_t activation_max = 103;
    const int8_t output_guard = 0x5A;

    for (size_t case_index = 0; case_index < sizeof(test_cases) / sizeof(test_cases[0]); case_index++)
    {
        const depthwise_test_case *test_case = &test_cases[case_index];
        const int32_t effective_filter_w = (test_case->filter_w - 1) * test_case->dilation_w + 1;
        const int32_t effective_filter_h = (test_case->filter_h - 1) * test_case->dilation_h + 1;
        const int32_t output_w =
            (test_case->input_w + 2 * test_case->pad_w - effective_filter_w) / test_case->stride_w + 1;
        const int32_t output_h =
            (test_case->input_h + 2 * test_case->pad_h - effective_filter_h) / test_case->stride_h + 1;

        TEST_ASSERT_GREATER_THAN_INT32(0, output_w);
        TEST_ASSERT_GREATER_THAN_INT32(0, output_h);

        for (size_t channel_index = 0; channel_index < sizeof(channels_to_test) / sizeof(channels_to_test[0]);
             channel_index++)
        {
            const int32_t channels = channels_to_test[channel_index];
            const size_t input_size = (size_t)test_case->input_w * test_case->input_h * channels;
            const size_t kernel_size = (size_t)test_case->filter_w * test_case->filter_h * channels;
            const size_t output_size = (size_t)output_w * output_h * channels;
            int8_t *input = malloc(input_size);
            int8_t *kernel = malloc(kernel_size);
            int32_t *bias = malloc((size_t)channels * sizeof(int32_t));
            int32_t *multiplier = malloc((size_t)channels * sizeof(int32_t));
            int32_t *shift = malloc((size_t)channels * sizeof(int32_t));
            int8_t *output_storage = malloc(output_size + 8);
            int8_t *reference = malloc(output_size);

            TEST_ASSERT_NOT_NULL(input);
            TEST_ASSERT_NOT_NULL(kernel);
            TEST_ASSERT_NOT_NULL(bias);
            TEST_ASSERT_NOT_NULL(multiplier);
            TEST_ASSERT_NOT_NULL(shift);
            TEST_ASSERT_NOT_NULL(output_storage);
            TEST_ASSERT_NOT_NULL(reference);

            for (size_t i = 0; i < input_size; i++)
            {
                input[i] = (int8_t)((i * 13 + case_index * 7) % 31 - 15);
            }
            for (size_t i = 0; i < kernel_size; i++)
            {
                kernel[i] = (int8_t)((i * 5 + channel_index * 3) % 15 - 7);
            }
            for (int32_t i = 0; i < channels; i++)
            {
                bias[i] = (i * 17 + (int32_t)case_index * 11) % 97 - 48;
                multiplier[i] = (i & 1) ? (1 << 29) : (1 << 30);
                shift[i] = (i % 3) - 1;
            }
            memset(output_storage, output_guard, output_size + 8);

            const cmsis_nn_dims input_dims = {1, test_case->input_h, test_case->input_w, channels};
            const cmsis_nn_dims filter_dims = {1, test_case->filter_h, test_case->filter_w, channels};
            const cmsis_nn_dims bias_dims = {1, 1, 1, channels};
            const cmsis_nn_dims output_dims = {1, output_h, output_w, channels};
            const cmsis_nn_dw_conv_params dw_conv_params = {
                .input_offset = input_offset,
                .output_offset = output_offset,
                .stride = {test_case->stride_w, test_case->stride_h},
                .padding = {test_case->pad_w, test_case->pad_h},
                .dilation = {test_case->dilation_w, test_case->dilation_h},
                .ch_mult = 1,
                .activation = {activation_min, activation_max},
            };
            const cmsis_nn_per_channel_quant_params quant_params = {
                .multiplier = multiplier,
                .shift = shift,
            };
            cmsis_nn_context ctx = {0};
            cmsis_nn_context weight_sum_ctx = {0};

            if (test_case->use_wrapper)
            {
                ctx.size = arm_depthwise_conv_wrapper_s8_get_buffer_size(
                    &dw_conv_params, &input_dims, &filter_dims, &output_dims);
            }
            else
            {
                ctx.size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
            }
            if (ctx.size > 0)
            {
                ctx.buf = malloc((size_t)ctx.size);
                TEST_ASSERT_NOT_NULL(ctx.buf);
            }

            /* Fill weight_sum_ctx on every route, wrapper included: even though the sole use_wrapper test case
             * currently has dilation != 1 and so does not reach arm_depthwise_conv_s8_opt(), a future wrapper
             * case must not silently rely on an unfilled buffer. Note that dilation == 1 alone does not guarantee
             * the wrapper reaches arm_depthwise_conv_s8_opt() either: on MVE, input_dims->c == 1 with an output
             * channel count above CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD (8 on armclang, 1
             * otherwise) diverts to the conv-conversion route (arm_depthwise_conv_to_conv_s8()) instead, which
             * wants conv-style sums from arm_convolve_weight_sum() rather than these depthwise sums. */
            weight_sum_ctx.size = channels * (int32_t)sizeof(int32_t);
            weight_sum_ctx.buf = malloc((size_t)weight_sum_ctx.size);
            TEST_ASSERT_NOT_NULL(weight_sum_ctx.buf);
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                              arm_depthwise_convolve_weight_sum(weight_sum_ctx.buf,
                                                                ctx.buf,
                                                                kernel,
                                                                &dw_conv_params,
                                                                &input_dims,
                                                                &filter_dims,
                                                                &output_dims,
                                                                input_offset,
                                                                bias));

            arm_cmsis_nn_status result;
            if (test_case->use_wrapper)
            {
                result = arm_depthwise_conv_wrapper_s8(&ctx,
                                                       &weight_sum_ctx,
                                                       &dw_conv_params,
                                                       &quant_params,
                                                       &input_dims,
                                                       input,
                                                       &filter_dims,
                                                       kernel,
                                                       &bias_dims,
                                                       bias,
                                                       &output_dims,
                                                       output_storage + 4);
            }
            else
            {
                result = arm_depthwise_conv_s8_opt(&ctx,
                                                   &weight_sum_ctx,
                                                   &dw_conv_params,
                                                   &quant_params,
                                                   &input_dims,
                                                   input,
                                                   &filter_dims,
                                                   kernel,
                                                   &bias_dims,
                                                   bias,
                                                   &output_dims,
                                                   output_storage + 4);
            }
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);

            for (int32_t out_y = 0; out_y < output_h; out_y++)
            {
                for (int32_t out_x = 0; out_x < output_w; out_x++)
                {
                    const int32_t base_y = out_y * test_case->stride_h - test_case->pad_h;
                    const int32_t base_x = out_x * test_case->stride_w - test_case->pad_w;
                    for (int32_t ch = 0; ch < channels; ch++)
                    {
                        int32_t acc = bias[ch];
                        for (int32_t ker_y = 0; ker_y < test_case->filter_h; ker_y++)
                        {
                            const int32_t in_y = base_y + ker_y * test_case->dilation_h;
                            for (int32_t ker_x = 0; ker_x < test_case->filter_w; ker_x++)
                            {
                                const int32_t in_x = base_x + ker_x * test_case->dilation_w;
                                if (in_y >= 0 && in_y < test_case->input_h && in_x >= 0 && in_x < test_case->input_w)
                                {
                                    const int32_t input_index = (in_y * test_case->input_w + in_x) * channels + ch;
                                    const int32_t kernel_index = (ker_y * test_case->filter_w + ker_x) * channels + ch;
                                    acc += (input[input_index] + input_offset) * kernel[kernel_index];
                                }
                            }
                        }
                        int32_t value = arm_nn_requantize(acc, multiplier[ch], shift[ch]) + output_offset;
                        value = ARM_NN_MAX(value, activation_min);
                        value = ARM_NN_MIN(value, activation_max);
                        reference[(out_y * output_w + out_x) * channels + ch] = (int8_t)value;
                    }
                }
            }

            TEST_ASSERT_EQUAL_INT8_ARRAY(reference, output_storage + 4, output_size);
            for (int32_t i = 0; i < 4; i++)
            {
                TEST_ASSERT_EQUAL_INT8(output_guard, output_storage[i]);
                TEST_ASSERT_EQUAL_INT8(output_guard, output_storage[output_size + 4 + i]);
            }

            free(ctx.buf);
            free(weight_sum_ctx.buf);
            free(reference);
            free(output_storage);
            free(shift);
            free(multiplier);
            free(bias);
            free(kernel);
            free(input);
        }
    }
#endif
}

void buffer_size_arm_depthwise_conv_s8_opt(void)
{
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DEPTHWISE_X_STRIDE_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_X_STRIDE_INPUT_W;
    input_dims.h = DEPTHWISE_X_STRIDE_INPUT_H;
    input_dims.c = DEPTHWISE_X_STRIDE_IN_CH;
    filter_dims.w = DEPTHWISE_X_STRIDE_FILTER_X;
    filter_dims.h = DEPTHWISE_X_STRIDE_FILTER_Y;
    output_dims.w = DEPTHWISE_X_STRIDE_OUTPUT_W;
    output_dims.h = DEPTHWISE_X_STRIDE_OUTPUT_H;
    output_dims.c = DEPTHWISE_X_STRIDE_OUT_CH;

    conv_params.padding.w = DEPTHWISE_X_STRIDE_PAD_X;
    conv_params.padding.h = DEPTHWISE_X_STRIDE_PAD_Y;
    conv_params.stride.w = DEPTHWISE_X_STRIDE_STRIDE_X;
    conv_params.stride.h = DEPTHWISE_X_STRIDE_STRIDE_Y;
    conv_params.dilation.w = DEPTHWISE_X_STRIDE_DILATION_X;
    conv_params.dilation.h = DEPTHWISE_X_STRIDE_DILATION_Y;
    conv_params.ch_mult = DEPTHWISE_X_STRIDE_CH_MULT;
    conv_params.input_offset = DEPTHWISE_X_STRIDE_INPUT_OFFSET;
    conv_params.output_offset = DEPTHWISE_X_STRIDE_OUTPUT_OFFSET;
    conv_params.activation.min = DEPTHWISE_X_STRIDE_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DEPTHWISE_X_STRIDE_OUT_ACTIVATION_MAX;

    const int32_t buf_size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, buf_size);
}

void buffer_size_mve_arm_depthwise_conv_s8_opt(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DEPTHWISE_X_STRIDE_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_X_STRIDE_INPUT_W;
    input_dims.h = DEPTHWISE_X_STRIDE_INPUT_H;
    input_dims.c = DEPTHWISE_X_STRIDE_IN_CH;
    filter_dims.w = DEPTHWISE_X_STRIDE_FILTER_X;
    filter_dims.h = DEPTHWISE_X_STRIDE_FILTER_Y;
    output_dims.w = DEPTHWISE_X_STRIDE_OUTPUT_W;
    output_dims.h = DEPTHWISE_X_STRIDE_OUTPUT_H;
    output_dims.c = DEPTHWISE_X_STRIDE_OUT_CH;

    conv_params.padding.w = DEPTHWISE_X_STRIDE_PAD_X;
    conv_params.padding.h = DEPTHWISE_X_STRIDE_PAD_Y;
    conv_params.stride.w = DEPTHWISE_X_STRIDE_STRIDE_X;
    conv_params.stride.h = DEPTHWISE_X_STRIDE_STRIDE_Y;
    conv_params.dilation.w = DEPTHWISE_X_STRIDE_DILATION_X;
    conv_params.dilation.h = DEPTHWISE_X_STRIDE_DILATION_Y;
    conv_params.ch_mult = DEPTHWISE_X_STRIDE_CH_MULT;
    conv_params.input_offset = DEPTHWISE_X_STRIDE_INPUT_OFFSET;
    conv_params.output_offset = DEPTHWISE_X_STRIDE_OUTPUT_OFFSET;
    conv_params.activation.min = DEPTHWISE_X_STRIDE_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DEPTHWISE_X_STRIDE_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_depthwise_conv_s8_opt(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DEPTHWISE_X_STRIDE_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_X_STRIDE_INPUT_W;
    input_dims.h = DEPTHWISE_X_STRIDE_INPUT_H;
    input_dims.c = DEPTHWISE_X_STRIDE_IN_CH;
    filter_dims.w = DEPTHWISE_X_STRIDE_FILTER_X;
    filter_dims.h = DEPTHWISE_X_STRIDE_FILTER_Y;
    output_dims.w = DEPTHWISE_X_STRIDE_OUTPUT_W;
    output_dims.h = DEPTHWISE_X_STRIDE_OUTPUT_H;
    output_dims.c = DEPTHWISE_X_STRIDE_OUT_CH;

    conv_params.padding.w = DEPTHWISE_X_STRIDE_PAD_X;
    conv_params.padding.h = DEPTHWISE_X_STRIDE_PAD_Y;
    conv_params.stride.w = DEPTHWISE_X_STRIDE_STRIDE_X;
    conv_params.stride.h = DEPTHWISE_X_STRIDE_STRIDE_Y;
    conv_params.dilation.w = DEPTHWISE_X_STRIDE_DILATION_X;
    conv_params.dilation.h = DEPTHWISE_X_STRIDE_DILATION_Y;

    conv_params.ch_mult = DEPTHWISE_X_STRIDE_CH_MULT;

    conv_params.input_offset = DEPTHWISE_X_STRIDE_INPUT_OFFSET;
    conv_params.output_offset = DEPTHWISE_X_STRIDE_OUTPUT_OFFSET;
    conv_params.activation.min = DEPTHWISE_X_STRIDE_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DEPTHWISE_X_STRIDE_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}

// Issue #318: the Helium leg sizes its scratch buffer from a fixed channel block, so it never reads
// input_dims->c in its own arithmetic. Without the dispatcher's dimension gate it therefore answered a negative
// channel count with a plausible positive byte count where arm_depthwise_conv_s8_opt_get_buffer_size() returned
// -1, and arm_depthwise_conv_wrapper_s8_get_buffer_size_mve() inherited that answer. Deliberately not gated on
// ARM_MATH_MVEI: the leg variants are plain C and are compiled and callable on every build target.
void buffer_size_out_of_range_mve_arm_depthwise_conv_s8_opt(void)
{
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;
    cmsis_nn_dw_conv_params dw_conv_params;

    memset(&dw_conv_params, 0, sizeof(dw_conv_params));
    dw_conv_params.stride.w = 1;
    dw_conv_params.stride.h = 1;
    dw_conv_params.dilation.w = 1;
    dw_conv_params.dilation.h = 1;
    dw_conv_params.ch_mult = 1;

    // The shape from the issue. c = -1 lands in all three dims structs below, but the gate's other two
    // conditions (filter_dims->w, filter_dims->h) are positive here, so it is input_dims->c that fires it --
    // the one dimension the Helium leg never reads, which is why the leg used to answer without it.
    input_dims.n = 1;
    input_dims.h = 65536;
    input_dims.w = 2;
    input_dims.c = -1;
    filter_dims = input_dims;
    output_dims = input_dims;

    TEST_ASSERT_EQUAL(-1, arm_depthwise_conv_s8_opt_get_buffer_size_mve(&input_dims, &filter_dims));
    TEST_ASSERT_EQUAL(-1, arm_depthwise_conv_s8_opt_get_buffer_size_dsp(&input_dims, &filter_dims));
    TEST_ASSERT_EQUAL(-1, arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims));
    TEST_ASSERT_EQUAL(
        -1,
        arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        -1, arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    // Zero spatial dims with the same negative channel count: the byte count folds to 0, so the sentinel has to
    // come from the dimension gate rather than from the overflow check.
    input_dims.h = 0;
    input_dims.w = 0;
    filter_dims = input_dims;
    output_dims = input_dims;

    TEST_ASSERT_EQUAL(-1, arm_depthwise_conv_s8_opt_get_buffer_size_mve(&input_dims, &filter_dims));
    TEST_ASSERT_EQUAL(
        -1,
        arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    // A negative filter dimension was already rejected by the bounded fold; pin it so the added gate does not
    // become the only thing catching it.
    input_dims.c = 4;
    filter_dims.n = 1;
    filter_dims.h = -1;
    filter_dims.w = 3;
    filter_dims.c = 4;
    TEST_ASSERT_EQUAL(-1, arm_depthwise_conv_s8_opt_get_buffer_size_mve(&input_dims, &filter_dims));

    // An in-range shape is undisturbed. The Helium leg stages CH_IN_BLOCK_MVE channels of int32 accumulators for
    // each of the filter_dims->w * filter_dims->h taps, so this figure is fixed by the geometry, not by the data.
    input_dims.n = 1;
    input_dims.h = 8;
    input_dims.w = 8;
    input_dims.c = 4;
    filter_dims.h = 3;
    filter_dims.w = 3;
    TEST_ASSERT_EQUAL(4 * CH_IN_BLOCK_MVE * 3 * 3,
                      arm_depthwise_conv_s8_opt_get_buffer_size_mve(&input_dims, &filter_dims));
}
