/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include <stdlib.h>
#include <string.h>
#include <unity.h>

#include "../TestData/basic/test_data.h"
#include "../TestData/depthwise_eq_in_out_ch/test_data.h"
#include "../TestData/depthwise_null_bias_0/test_data.h"
#include "../TestData/depthwise_out_activation/test_data.h"
#include "../TestData/depthwise_sub_block/test_data.h"
#include "../TestData/depthwise_x_stride/test_data.h"
/* The largest operand placed against the gap is the 4 x 9 x CH_IN_BLOCK_MVE padded lhs. */
#define GUARD_OFFSET 4608
#include "../Utils/mpu_guard.h"
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
             * currently is dilated in both dimensions and so does not reach arm_depthwise_conv_s8_opt(), a future
             * wrapper case must not silently rely on an unfilled buffer. Note that dilation == 1 alone does not
             * guarantee the wrapper reaches arm_depthwise_conv_s8_opt() either: on MVE, input_dims->c == 1 with an
             * output channel count above CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD (8 on
             * armclang, 1 otherwise) diverts to the conv-conversion route (arm_depthwise_conv_to_conv_s8()) instead,
             * which wants conv-style sums from arm_convolve_weight_sum() rather than these depthwise sums. */
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

static void
test_dilated_1d_s8_case(int32_t input_len, int32_t filter_len, int32_t channels, int32_t dilation, int32_t pad)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int32_t output_len = (input_len + 2 * pad - (filter_len - 1) * dilation - 1) + 1;
    TEST_ASSERT_TRUE(output_len > 0);

    const int32_t input_size = input_len * channels;
    const int32_t filter_size = filter_len * channels;
    const int32_t output_size = output_len * channels;

    int8_t *input_data = (int8_t *)malloc((size_t)input_size * sizeof(int8_t));
    int8_t *filter_data = (int8_t *)malloc((size_t)filter_size * sizeof(int8_t));
    int32_t *bias_data = (int32_t *)malloc((size_t)channels * sizeof(int32_t));
    int32_t *output_mult = (int32_t *)malloc((size_t)channels * sizeof(int32_t));
    int32_t *output_shift = (int32_t *)malloc((size_t)channels * sizeof(int32_t));
    int8_t *output_ref = (int8_t *)malloc((size_t)output_size * sizeof(int8_t));
    int8_t *output_opt = (int8_t *)malloc((size_t)output_size * sizeof(int8_t));

    TEST_ASSERT_NOT_NULL(input_data);
    TEST_ASSERT_NOT_NULL(filter_data);
    TEST_ASSERT_NOT_NULL(bias_data);
    TEST_ASSERT_NOT_NULL(output_mult);
    TEST_ASSERT_NOT_NULL(output_shift);
    TEST_ASSERT_NOT_NULL(output_ref);
    TEST_ASSERT_NOT_NULL(output_opt);

    memset(output_ref, 0, (size_t)output_size * sizeof(int8_t));
    memset(output_opt, 0, (size_t)output_size * sizeof(int8_t));

    for (int32_t i = 0; i < input_size; i++)
    {
        input_data[i] = (int8_t)(((i * 17 + 5) % 251) - 128);
    }
    for (int32_t i = 0; i < filter_size; i++)
    {
        filter_data[i] = (int8_t)(((i * 31 + 13) % 251) - 128);
    }
    for (int32_t i = 0; i < channels; i++)
    {
        bias_data[i] = (int32_t)((i * 101 - 50) * 16);
        output_mult[i] = (int32_t)(0x40000000 + ((i % 64) * 0x1000000)); /* stays positive for any channel count */
        output_shift[i] = -7;
    }

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims = {1, 1, input_len, channels};
    cmsis_nn_dims filter_dims = {1, 1, filter_len, channels};
    cmsis_nn_dims bias_dims = {1, 1, 1, channels};
    cmsis_nn_dims output_dims = {1, 1, output_len, channels};

    dw_conv_params.padding.w = pad;
    dw_conv_params.padding.h = 0;
    dw_conv_params.stride.w = 1;
    dw_conv_params.stride.h = 1;
    dw_conv_params.dilation.w = dilation;
    dw_conv_params.dilation.h = 1;
    dw_conv_params.ch_mult = 1;
    dw_conv_params.input_offset = 128;
    dw_conv_params.output_offset = -10;
    dw_conv_params.activation.min = -128;
    dw_conv_params.activation.max = 127;
    quant_params.multiplier = output_mult;
    quant_params.shift = output_shift;

    arm_cmsis_nn_status ref_status = arm_depthwise_conv_s8(&ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           filter_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output_ref);
    TEST_ASSERT_EQUAL(expected, ref_status);

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_EQUAL(arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims), buf_size);
    ctx.size = buf_size;
    if (buf_size > 0)
    {
        ctx.buf = malloc((size_t)buf_size);
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }

    cmsis_nn_context weights_sum_ctx = {NULL, 0};
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    /* The size is 0 on builds that do not read the sums; allocate at least one entry so the buffer is valid. */
    weights_sum_ctx.buf = malloc((size_t)(weights_sum_buf_size > 0 ? weights_sum_buf_size : (int32_t)sizeof(int32_t)));
    weights_sum_ctx.size = weights_sum_buf_size;
    TEST_ASSERT_NOT_NULL(weights_sum_ctx.buf);

    arm_depthwise_convolve_weight_sum((int32_t *)weights_sum_ctx.buf,
                                      ctx.buf,
                                      filter_data,
                                      &dw_conv_params,
                                      &input_dims,
                                      &filter_dims,
                                      &output_dims,
                                      dw_conv_params.input_offset,
                                      bias_data);

    arm_cmsis_nn_status opt_status = arm_depthwise_conv_wrapper_s8(&ctx,
                                                                   &weights_sum_ctx,
                                                                   &dw_conv_params,
                                                                   &quant_params,
                                                                   &input_dims,
                                                                   input_data,
                                                                   &filter_dims,
                                                                   filter_data,
                                                                   &bias_dims,
                                                                   bias_data,
                                                                   &output_dims,
                                                                   output_opt);
    TEST_ASSERT_EQUAL(expected, opt_status);
    TEST_ASSERT_EQUAL_INT8_ARRAY(output_ref, output_opt, output_size);

    if (weights_sum_ctx.buf)
    {
        free(weights_sum_ctx.buf);
    }
    if (ctx.buf)
    {
        free(ctx.buf);
    }
    free(output_opt);
    free(output_ref);
    free(output_shift);
    free(output_mult);
    free(bias_data);
    free(filter_data);
    free(input_data);
}

void dilated_1d_arm_depthwise_conv_s8_opt(void)
{
    const int32_t dilations[] = {2, 4, 8};
    const int32_t filters[] = {7, 9};
    const int32_t channels[] = {16, 24, 32};
    const int32_t input_len = 256;

    for (size_t d = 0; d < sizeof(dilations) / sizeof(dilations[0]); d++)
    {
        for (size_t f = 0; f < sizeof(filters) / sizeof(filters[0]); f++)
        {
            for (size_t c = 0; c < sizeof(channels) / sizeof(channels[0]); c++)
            {
                const int32_t dilation = dilations[d];
                const int32_t filter_len = filters[f];
                const int32_t ch = channels[c];
                const int32_t pad = ((filter_len - 1) * dilation) / 2;

                // SAME padding (boundary + interior)
                test_dilated_1d_s8_case(input_len, filter_len, ch, dilation, pad);

                // VALID padding (pad = 0)
                test_dilated_1d_s8_case(input_len, filter_len, ch, dilation, 0);
            }
        }
    }

    /* Dilated 1D layers from sleepkit TCN models (width 240, kernel 5 and 7, dilation up to 16, up to 64
       channels), then channel-tail, multi-block, even-kernel and short-input shapes. SAME padding. */
    const int32_t extra[][4] = {
        /* input_len, filter_len, channels, dilation */
        {240, 5, 24, 2},
        {240, 5, 32, 4},
        {240, 5, 48, 8},
        {240, 7, 48, 16},
        {240, 7, 64, 16},
        {37, 3, 5, 3},
        {64, 2, 125, 16},
        {9, 7, 3, 2},
        {1, 3, 17, 4},
    };
    for (size_t i = 0; i < sizeof(extra) / sizeof(extra[0]); i++)
    {
        const int32_t pad = ((extra[i][1] - 1) * extra[i][3]) / 2;
        test_dilated_1d_s8_case(extra[i][0], extra[i][1], extra[i][2], extra[i][3], pad);
    }
}

void dilated_scope_gate_arm_depthwise_conv_s8_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int32_t channels = 16;
    const int32_t input_w = 32;
    const int32_t input_h = 4;
    const int32_t filter_w = 7;
    const int32_t filter_h = 1;
    const int32_t dilation_x = 2;
    const int32_t pad_x = 6;
    const int32_t output_w = (input_w + 2 * pad_x - (filter_w - 1) * dilation_x - 1) + 1;
    const int32_t output_h = 4;

    const int32_t input_size = input_w * input_h * channels;
    const int32_t filter_size = filter_w * filter_h * channels;
    const int32_t output_size = output_w * output_h * channels;

    int8_t *input_data = (int8_t *)malloc((size_t)input_size * sizeof(int8_t));
    int8_t *filter_data = (int8_t *)malloc((size_t)filter_size * sizeof(int8_t));
    int32_t *bias_data = (int32_t *)malloc((size_t)channels * sizeof(int32_t));
    int32_t *output_mult = (int32_t *)malloc((size_t)channels * sizeof(int32_t));
    int32_t *output_shift = (int32_t *)malloc((size_t)channels * sizeof(int32_t));
    int8_t *output_ref = (int8_t *)malloc((size_t)output_size * sizeof(int8_t));
    int8_t *output_wrapper = (int8_t *)malloc((size_t)output_size * sizeof(int8_t));

    TEST_ASSERT_NOT_NULL(input_data);
    TEST_ASSERT_NOT_NULL(filter_data);
    TEST_ASSERT_NOT_NULL(bias_data);
    TEST_ASSERT_NOT_NULL(output_mult);
    TEST_ASSERT_NOT_NULL(output_shift);
    TEST_ASSERT_NOT_NULL(output_ref);
    TEST_ASSERT_NOT_NULL(output_wrapper);

    memset(output_ref, 0, (size_t)output_size * sizeof(int8_t));
    memset(output_wrapper, 0, (size_t)output_size * sizeof(int8_t));

    for (int32_t i = 0; i < input_size; i++)
    {
        input_data[i] = (int8_t)(((i * 13 + 7) % 251) - 128);
    }
    for (int32_t i = 0; i < filter_size; i++)
    {
        filter_data[i] = (int8_t)(((i * 29 + 11) % 251) - 128);
    }
    for (int32_t i = 0; i < channels; i++)
    {
        bias_data[i] = (int32_t)((i * 50 - 25) * 16);
        output_mult[i] = (int32_t)(0x40000000 + ((i % 64) * 0x1000000)); /* stays positive for any channel count */
        output_shift[i] = -7;
    }

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims = {1, input_h, input_w, channels};
    cmsis_nn_dims filter_dims = {1, filter_h, filter_w, channels};
    cmsis_nn_dims bias_dims = {1, 1, 1, channels};
    cmsis_nn_dims output_dims = {1, output_h, output_w, channels};

    dw_conv_params.padding.w = pad_x;
    dw_conv_params.padding.h = 0;
    dw_conv_params.stride.w = 1;
    dw_conv_params.stride.h = 1;
    dw_conv_params.dilation.w = dilation_x;
    dw_conv_params.dilation.h = 1;
    dw_conv_params.ch_mult = 1;
    dw_conv_params.input_offset = 128;
    dw_conv_params.output_offset = -10;
    dw_conv_params.activation.min = -128;
    dw_conv_params.activation.max = 127;
    quant_params.multiplier = output_mult;
    quant_params.shift = output_shift;

    // Multi-row input (input_h > 1) with dilation.w > 1 must NOT enter the optimized path;
    // all wrapper sizers must return 0 (indicating generic reference fallback).
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    // Reference computation
    arm_cmsis_nn_status ref_status = arm_depthwise_conv_s8(&ctx,
                                                           &dw_conv_params,
                                                           &quant_params,
                                                           &input_dims,
                                                           input_data,
                                                           &filter_dims,
                                                           filter_data,
                                                           &bias_dims,
                                                           bias_data,
                                                           &output_dims,
                                                           output_ref);
    TEST_ASSERT_EQUAL(expected, ref_status);

    // Wrapper computation (routes to generic reference)
    int32_t scope_weight_sums[16] = {0};
    const cmsis_nn_context scope_weight_sum_ctx = {scope_weight_sums, (int32_t)sizeof(scope_weight_sums)};
    arm_cmsis_nn_status wrapper_status = arm_depthwise_conv_wrapper_s8(&ctx,
                                                                       &scope_weight_sum_ctx,
                                                                       &dw_conv_params,
                                                                       &quant_params,
                                                                       &input_dims,
                                                                       input_data,
                                                                       &filter_dims,
                                                                       filter_data,
                                                                       &bias_dims,
                                                                       bias_data,
                                                                       &output_dims,
                                                                       output_wrapper);
    TEST_ASSERT_EQUAL(expected, wrapper_status);
    TEST_ASSERT_EQUAL_INT8_ARRAY(output_ref, output_wrapper, output_size);

    // Multi-row case with non-zero vertical padding must also remain outside the optimized path
    dw_conv_params.padding.h = 1;
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    // 1D case (input_h = 1, output_h = 1, padding.h = 0) with dilation.w > 1 DOES enter optimized path
    input_dims.h = 1;
    output_dims.h = 1;
    dw_conv_params.padding.h = 0;
    const int32_t expected_opt_buf_size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    TEST_ASSERT_EQUAL(
        expected_opt_buf_size,
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        arm_depthwise_conv_s8_opt_get_buffer_size_dsp(&input_dims, &filter_dims),
        arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        arm_depthwise_conv_s8_opt_get_buffer_size_mve(&input_dims, &filter_dims),
        arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    // 1D case with non-zero vertical padding must NOT enter optimized path
    dw_conv_params.padding.h = 1;
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        0, arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    free(output_wrapper);
    free(output_ref);
    free(output_shift);
    free(output_mult);
    free(bias_data);
    free(filter_data);
    free(input_data);
}

/* The dilated 1D route runs arm_depthwise_conv_s8_opt(): with no scratch buffer it is rejected on builds that need
   one (DSP and MVE), whereas the reference route that a 2D-dilated layer takes needs none and succeeds. */
void dilated_1d_route_arm_depthwise_conv_s8_opt(void)
{
    enum
    {
        CH = 16,
        LEN = 64,
        K = 5,
        DIL = 4
    };
    static int8_t input[2 * LEN * CH];
    static int8_t filter[K * K * CH];
    static int8_t output[2 * LEN * CH];
    int32_t bias[CH], mult[CH], shift[CH], weight_sums[CH];
    for (int32_t i = 0; i < (int32_t)sizeof(input); i++)
    {
        input[i] = (int8_t)((i * 7 + 3) % 200 - 100);
    }
    for (int32_t i = 0; i < (int32_t)sizeof(filter); i++)
    {
        filter[i] = (int8_t)((i * 5 + 1) % 120 - 60);
    }
    for (int32_t i = 0; i < CH; i++)
    {
        bias[i] = i * 10;
        mult[i] = 0x40000000;
        shift[i] = -7;
    }
    const int32_t pad = ((K - 1) * DIL) / 2;
    cmsis_nn_dw_conv_params params = {.input_offset = 5,
                                      .output_offset = -2,
                                      .ch_mult = 1,
                                      .stride = {1, 1},
                                      .padding = {pad, 0},
                                      .dilation = {DIL, 1},
                                      .activation = {-128, 127}};
    const cmsis_nn_per_channel_quant_params quant = {mult, shift};
    const cmsis_nn_dims input_dims = {1, 1, LEN, CH}, filter_dims = {1, 1, K, CH}, bias_dims = {1, 1, 1, CH},
                        output_dims = {1, 1, LEN, CH};
    (void)arm_depthwise_convolve_weight_sum(
        weight_sums, NULL, filter, &params, &input_dims, &filter_dims, &output_dims, params.input_offset, bias);
    const cmsis_nn_context weight_sum_ctx = {weight_sums, (int32_t)sizeof(weight_sums)};
    const cmsis_nn_context no_scratch = {NULL, 0};

    const arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(&no_scratch,
                                                                     &weight_sum_ctx,
                                                                     &params,
                                                                     &quant,
                                                                     &input_dims,
                                                                     input,
                                                                     &filter_dims,
                                                                     filter,
                                                                     &bias_dims,
                                                                     bias,
                                                                     &output_dims,
                                                                     output);
#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(arm_depthwise_conv_wrapper_s8_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims) >
                     0);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, status);
#else
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
#endif

    /* Control: the same layer made 2D and dilated in both dimensions stays on the reference route. */
    params.padding.h = 0;
    params.dilation.h = 2;
    const cmsis_nn_dims input_2d = {1, 2, LEN, CH}, filter_2d = {1, 1, K, CH}, output_2d = {1, 2, LEN, CH};
    TEST_ASSERT_EQUAL(0, arm_depthwise_conv_wrapper_s8_get_buffer_size(&params, &input_2d, &filter_2d, &output_2d));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_s8(&no_scratch,
                                                    &weight_sum_ctx,
                                                    &params,
                                                    &quant,
                                                    &input_2d,
                                                    input,
                                                    &filter_2d,
                                                    filter,
                                                    &bias_dims,
                                                    bias,
                                                    &output_2d,
                                                    output));

    /* A 1D layer with vertical dilation is outside the route too: no scratch needed, reference result. */
    static int8_t reference[LEN * CH];
    const cmsis_nn_context none = {NULL, 0};
    params.dilation.h = 2;
    TEST_ASSERT_EQUAL(0,
                      arm_depthwise_conv_wrapper_s8_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_s8(&none,
                                            &params,
                                            &quant,
                                            &input_dims,
                                            input,
                                            &filter_dims,
                                            filter,
                                            &bias_dims,
                                            bias,
                                            &output_dims,
                                            reference));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_s8(&no_scratch,
                                                    &weight_sum_ctx,
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
    TEST_ASSERT_EQUAL_INT8_ARRAY(reference, output, LEN * CH);
}

/* arm_depthwise_conv_s8_opt() steps only the horizontal tap index by dilation, so it rejects vertical dilation and a
   non-positive horizontal dilation instead of computing a wrong result. */
void dilation_arg_check_arm_depthwise_conv_s8_opt(void)
{
    enum
    {
        CH = 4,
        H = 4,
        W = 8,
        K = 3
    };
    static int8_t input[H * W * CH];
    static int8_t filter[K * K * CH];
    static int8_t output[H * W * CH];
    static int8_t scratch[4 * 124 * K * K];
    int32_t bias[CH] = {0}, mult[CH], shift[CH], weight_sums[CH] = {0};
    for (int32_t i = 0; i < CH; i++)
    {
        mult[i] = 0x40000000;
        shift[i] = -7;
    }
    const cmsis_nn_per_channel_quant_params quant = {mult, shift};
    const cmsis_nn_dims input_dims = {1, H, W, CH}, filter_dims = {1, K, K, CH}, bias_dims = {1, 1, 1, CH},
                        output_dims = {1, H, W, CH};
    const cmsis_nn_context ctx = {scratch, (int32_t)sizeof(scratch)};
    const cmsis_nn_context weight_sum_ctx = {weight_sums, (int32_t)sizeof(weight_sums)};
    const cmsis_nn_tile bad_dilations[] = {{2, 2}, {1, 2}, {0, 1}, {-1, 1}};
    for (size_t i = 0; i < sizeof(bad_dilations) / sizeof(bad_dilations[0]); i++)
    {
        const cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                                .output_offset = 0,
                                                .ch_mult = 1,
                                                .stride = {1, 1},
                                                .padding = {1, 1},
                                                .dilation = bad_dilations[i],
                                                .activation = {-128, 127}};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_depthwise_conv_s8_opt(&ctx,
                                                    &weight_sum_ctx,
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
    }
}

/* Few-channel and 1xk layers, including the model shapes that take the pixel-vectorized path and neighbours on
   either side of its gate, must match arm_depthwise_conv_s8() byte for byte and stay inside the reported scratch. */
#define PLANAR_MAX_IO (48 * 48 * 8)
#define PLANAR_GUARD (32)
static int8_t planar_in[PLANAR_MAX_IO], planar_ker[16 * 16 * 40], planar_out[PLANAR_MAX_IO + PLANAR_GUARD],
    planar_ref[PLANAR_MAX_IO];
static int32_t planar_bias[40], planar_mult[40], planar_shift[40], planar_wsum[40];
static int8_t planar_scratch[16 * 1024 + PLANAR_GUARD];

static void planar_case(int32_t ih,
                        int32_t iw,
                        int32_t ch,
                        int32_t kh,
                        int32_t kw,
                        int32_t dil,
                        int32_t pad_h,
                        int32_t pad_w,
                        int32_t expect_planar,
                        int32_t input_offset,
                        int32_t act_min,
                        int32_t act_max)
{
    const int32_t oh = ih + 2 * pad_h - (kh - 1);
    const int32_t ow = iw + 2 * pad_w - (kw - 1) * dil;
    TEST_ASSERT_TRUE(ih * iw * ch <= PLANAR_MAX_IO && oh * ow * ch <= PLANAR_MAX_IO && kh * kw <= 256 && ch <= 40);
    uint32_t seed = (uint32_t)(ih * 977 + iw * 131 + ch * 7 + kw * 3 + dil);
    for (int32_t i = 0; i < ih * iw * ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        planar_in[i] = (int8_t)(seed >> 24);
    }
    for (int32_t i = 0; i < kh * kw * ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        planar_ker[i] = (int8_t)(seed >> 24);
    }
    for (int32_t i = 0; i < ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        planar_bias[i] = (int32_t)(seed >> 20) - 2048;
        planar_mult[i] = 0x40000000 + (i % 7) * 0x4000000;
        planar_shift[i] = -7 - (i % 3);
    }
    const cmsis_nn_dw_conv_params params = {.input_offset = input_offset,
                                            .output_offset = -3,
                                            .ch_mult = 1,
                                            .stride = {1, 1},
                                            .padding = {pad_w, pad_h},
                                            .dilation = {dil, 1},
                                            .activation = {act_min, act_max}};
    const cmsis_nn_per_channel_quant_params quant = {planar_mult, planar_shift};
    const cmsis_nn_dims input_dims = {1, ih, iw, ch}, filter_dims = {1, kh, kw, ch}, bias_dims = {1, 1, 1, ch},
                        output_dims = {1, oh, ow, ch};
    const cmsis_nn_context none = {NULL, 0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_s8(&none,
                                            &params,
                                            &quant,
                                            &input_dims,
                                            planar_in,
                                            &filter_dims,
                                            planar_ker,
                                            &bias_dims,
                                            planar_bias,
                                            &output_dims,
                                            planar_ref));

    const int32_t size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    TEST_ASSERT_TRUE(size >= 0 && size + PLANAR_GUARD <= (int32_t)sizeof(planar_scratch));
    const cmsis_nn_context ctx = {planar_scratch, size};
    /* Weight sums exist only on MVE; the other legs of arm_depthwise_conv_s8_opt() do not read them. */
    (void)arm_depthwise_convolve_weight_sum(planar_wsum,
                                            planar_scratch,
                                            planar_ker,
                                            &params,
                                            &input_dims,
                                            &filter_dims,
                                            &output_dims,
                                            input_offset,
                                            planar_bias);
    const cmsis_nn_context wsum = {planar_wsum, ch * (int32_t)sizeof(int32_t)};
    memset(planar_scratch, 0x3C, sizeof(planar_scratch));
    memset(planar_out, 0x5A, sizeof(planar_out));
#if defined(ARM_MATH_MVEI)
    /* The model shapes take the pixel-vectorized path; the gate's neighbours are declined without writing output. */
    const arm_cmsis_nn_status planar_status = arm_nn_depthwise_conv_s8_planar(
        &ctx, &wsum, &params, &quant, &input_dims, planar_in, &filter_dims, planar_ker, &output_dims, planar_out);
    TEST_ASSERT_EQUAL(expect_planar ? ARM_CMSIS_NN_SUCCESS : ARM_CMSIS_NN_NO_IMPL_ERROR, planar_status);
    if (!expect_planar)
    {
        for (int32_t i = 0; i < oh * ow * ch + PLANAR_GUARD; i++)
        {
            TEST_ASSERT_EQUAL_INT8(0x5A, planar_out[i]);
        }
    }
    memset(planar_out, 0x5A, sizeof(planar_out));
#else
    (void)expect_planar;
#endif
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_s8_opt(&ctx,
                                                &wsum,
                                                &params,
                                                &quant,
                                                &input_dims,
                                                planar_in,
                                                &filter_dims,
                                                planar_ker,
                                                &bias_dims,
                                                planar_bias,
                                                &output_dims,
                                                planar_out));
    TEST_ASSERT_EQUAL_INT8_ARRAY(planar_ref, planar_out, oh * ow * ch);
    for (int32_t i = 0; i < PLANAR_GUARD; i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, planar_out[oh * ow * ch + i]);
        TEST_ASSERT_EQUAL_INT8(0x3C, planar_scratch[size + i]);
    }
#if defined(ARM_MATH_DSP) && defined(ARM_MATH_MVEI)
    /* The planar path uses only the first plane bytes of the scratch, while the channel path writes im2col rows past
       them, so an untouched remainder shows which path arm_depthwise_conv_s8_opt() took. */
    if (expect_planar)
    {
        const int32_t plane =
            arm_nn_depthwise_conv_s8_planar_bytes(&params, &input_dims, &filter_dims, &output_dims);
        TEST_ASSERT_TRUE(plane > 0 && plane <= size);
        for (int32_t i = plane; i < size; i++)
        {
            TEST_ASSERT_EQUAL_INT8(0x3C, planar_scratch[i]);
        }
    }
#endif

    /* The predicate is plain C and gives the same answer on every build. */
    TEST_ASSERT_EQUAL(expect_planar,
                      arm_depthwise_conv_s8_opt_planar_supported(&params, &input_dims, &filter_dims, &output_dims));
    /* arm_depthwise_conv_s8_opt() skips the planar kernel for layers outside these cheap conditions. */
    if (expect_planar)
    {
        TEST_ASSERT_TRUE(arm_nn_depthwise_conv_s8_planar_candidate(&params, &input_dims));
    }

    /* The direct entries: the channel-vectorized one computes every layer; the planar one computes the layers the
       predicate accepts and writes nothing otherwise. */
    memset(planar_out, 0x5A, sizeof(planar_out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_s8_opt_channelwise(&ctx,
                                                            &wsum,
                                                            &params,
                                                            &quant,
                                                            &input_dims,
                                                            planar_in,
                                                            &filter_dims,
                                                            planar_ker,
                                                            &bias_dims,
                                                            planar_bias,
                                                            &output_dims,
                                                            planar_out));
    TEST_ASSERT_EQUAL_INT8_ARRAY(planar_ref, planar_out, oh * ow * ch);
    memset(planar_out, 0x5A, sizeof(planar_out));
    const arm_cmsis_nn_status direct = arm_depthwise_conv_s8_opt_planar(&ctx,
                                                                        &wsum,
                                                                        &params,
                                                                        &quant,
                                                                        &input_dims,
                                                                        planar_in,
                                                                        &filter_dims,
                                                                        planar_ker,
                                                                        &bias_dims,
                                                                        planar_bias,
                                                                        &output_dims,
                                                                        planar_out);
#if defined(ARM_MATH_DSP) && defined(ARM_MATH_MVEI)
    TEST_ASSERT_EQUAL(expect_planar ? ARM_CMSIS_NN_SUCCESS : ARM_CMSIS_NN_NO_IMPL_ERROR, direct);
#else
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, direct);
#endif
    if (direct == ARM_CMSIS_NN_SUCCESS)
    {
        TEST_ASSERT_EQUAL_INT8_ARRAY(planar_ref, planar_out, oh * ow * ch);
    }
    else
    {
        for (int32_t i = 0; i < oh * ow * ch; i++)
        {
            TEST_ASSERT_EQUAL_INT8(0x5A, planar_out[i]);
        }
    }
}

void planar_shapes_arm_depthwise_conv_s8_opt(void)
{
    /* ih, iw, ch, kh, kw, dilation, pad_h, pad_w, takes the pixel-vectorized path */
    const int32_t shapes[][9] = {
        {48, 48, 8, 3, 3, 1, 1, 1, 1},  /* VWW DW1 */
        {1, 240, 14, 1, 3, 1, 0, 1, 1}, /* tcn32 */
        {1, 240, 8, 1, 3, 2, 0, 2, 1},   {1, 240, 8, 1, 3, 4, 0, 4, 1},  {1, 240, 8, 1, 3, 8, 0, 8, 1},
        {1, 256, 16, 1, 9, 1, 0, 4, 1}, /* heart-arr */
        {1, 128, 24, 1, 9, 1, 0, 4, 1},  {1, 64, 32, 1, 9, 1, 0, 4, 1},  {1, 32, 40, 1, 9, 1, 0, 4, 0},
        {1, 256, 1, 1, 7, 1, 0, 3, 1}, /* heart-seg */
        {1, 256, 16, 1, 7, 2, 0, 6, 1},  {1, 256, 24, 1, 7, 4, 0, 12, 1}, {1, 256, 32, 1, 7, 8, 0, 24, 0},
        {7, 9, 5, 3, 3, 1, 1, 1, 1},     {9, 17, 16, 5, 5, 1, 2, 2, 1},  {6, 23, 3, 3, 3, 2, 1, 2, 1},
        {1, 11, 6, 1, 5, 1, 0, 0, 0},    {1, 100, 3, 1, 16, 5, 0, 37, 1}, {1, 9, 1, 1, 7, 1, 0, 3, 1},
        {1, 64, 33, 1, 7, 1, 0, 3, 0},   {1, 40, 17, 1, 5, 1, 0, 2, 1},  {3, 31, 2, 3, 3, 1, 0, 1, 1},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
    {
        const int32_t *s = shapes[i];
        planar_case(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], 128, -128, 127);
        planar_case(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], -5, -60, 70);
    }
}

/* The public predicate is the planar kernel's own rule: over a grid of shapes on both sides of every limit, it
   accepts exactly the layers the kernel computes with the arm_depthwise_conv_s8_opt() scratch. */
void planar_predicate_grid_arm_depthwise_conv_s8_opt(void)
{
#if defined(ARM_MATH_DSP) && defined(ARM_MATH_MVEI)
    const int32_t channels[] = {1, 8, 9, 16, 17, 24, 32, 33};
    const int32_t kernels[][2] = {{3, 3}, {5, 5}, {1, 3}, {1, 5}, {1, 9}, {1, 16}, {1, 17}};
    const int32_t dilations[] = {1, 2, 4, 8};
    const int32_t sizes[][2] = {{1, 8}, {1, 24}, {1, 64}, {8, 8}, {8, 16}, {12, 32}, {80, 80}};
    const int32_t strides[] = {1, 2};
    int32_t accepted = 0, declined = 0, fit_declined = 0;
    for (size_t i_c = 0; i_c < sizeof(channels) / sizeof(channels[0]); i_c++)
    {
        for (size_t i_k = 0; i_k < sizeof(kernels) / sizeof(kernels[0]); i_k++)
        {
            for (size_t i_d = 0; i_d < sizeof(dilations) / sizeof(dilations[0]); i_d++)
            {
                for (size_t i_s = 0; i_s < sizeof(sizes) / sizeof(sizes[0]); i_s++)
                {
                    for (size_t i_st = 0; i_st < sizeof(strides) / sizeof(strides[0]); i_st++)
                    {
                        const int32_t ch = channels[i_c], kh = kernels[i_k][0], kw = kernels[i_k][1];
                        const int32_t dil = dilations[i_d], ih = sizes[i_s][0], iw = sizes[i_s][1];
                        const int32_t stride = strides[i_st];
                        if (kh > ih)
                        {
                            continue;
                        }
                        const int32_t pad_w = ((kw - 1) * dil) / 2, pad_h = (kh - 1) / 2;
                        const int32_t ow = (iw + 2 * pad_w - (kw - 1) * dil - 1) / stride + 1;
                        const int32_t oh = (ih + 2 * pad_h - (kh - 1) - 1) / stride + 1;
                        if (ow < 1 || oh < 1 || ih * iw * ch > PLANAR_MAX_IO || oh * ow * ch > PLANAR_MAX_IO)
                        {
                            continue;
                        }
                        const cmsis_nn_dw_conv_params params = {.input_offset = 3,
                                                                .output_offset = -3,
                                                                .ch_mult = 1,
                                                                .stride = {stride, stride},
                                                                .padding = {pad_w, pad_h},
                                                                .dilation = {dil, 1},
                                                                .activation = {-128, 127}};
                        const cmsis_nn_per_channel_quant_params quant = {planar_mult, planar_shift};
                        const cmsis_nn_dims input_dims = {1, ih, iw, ch}, filter_dims = {1, kh, kw, ch},
                                            output_dims = {1, oh, ow, ch};
                        const int32_t size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
                        TEST_ASSERT_TRUE(size >= 0 && size <= (int32_t)sizeof(planar_scratch));
                        const cmsis_nn_context ctx = {planar_scratch, size};
                        const cmsis_nn_context wsum = {planar_wsum, ch * (int32_t)sizeof(int32_t)};
                        const int32_t supported = arm_depthwise_conv_s8_opt_planar_supported(
                            &params, &input_dims, &filter_dims, &output_dims);
                        const arm_cmsis_nn_status status = arm_nn_depthwise_conv_s8_planar(&ctx,
                                                                                           &wsum,
                                                                                           &params,
                                                                                           &quant,
                                                                                           &input_dims,
                                                                                           planar_in,
                                                                                           &filter_dims,
                                                                                           planar_ker,
                                                                                           &output_dims,
                                                                                           planar_out);
                        TEST_ASSERT_EQUAL(supported ? ARM_CMSIS_NN_SUCCESS : ARM_CMSIS_NN_NO_IMPL_ERROR, status);
                        if (supported)
                        {
                            TEST_ASSERT_TRUE(arm_nn_depthwise_conv_s8_planar_candidate(&params, &input_dims));
                        }
                        accepted += supported;
                        declined += !supported;
                        fit_declined += !supported &&
                            arm_nn_depthwise_conv_s8_planar_bytes(&params, &input_dims, &filter_dims, &output_dims) >=
                                0;
                    }
                }
            }
        }
    }
    /* The grid reaches both outcomes, and a plane that only the scratch size declines. */
    TEST_ASSERT_TRUE(accepted > 0 && declined > 0 && fit_declined > 0);
#endif
}

/* The direct entries reject the same arguments as arm_depthwise_conv_s8_opt(). When ctx->size cannot hold the plane,
   the planar entry declines without writing. */
void direct_entries_arm_depthwise_conv_s8_opt(void)
{
    const int32_t ih = 7, iw = 9, ch = 5, kh = 3, kw = 3;
    planar_case(ih, iw, ch, kh, kw, 1, 1, 1, 1, 3, -128, 127);
    const cmsis_nn_dw_conv_params params = {.input_offset = 3,
                                            .output_offset = -3,
                                            .ch_mult = 1,
                                            .stride = {1, 1},
                                            .padding = {1, 1},
                                            .dilation = {1, 1},
                                            .activation = {-128, 127}};
    const cmsis_nn_per_channel_quant_params quant = {planar_mult, planar_shift};
    const cmsis_nn_dims input_dims = {1, ih, iw, ch}, filter_dims = {1, kh, kw, ch}, bias_dims = {1, 1, 1, ch},
                        output_dims = {1, ih, iw, ch};
    const int32_t size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    const cmsis_nn_context ctx = {planar_scratch, size};
    const cmsis_nn_context wsum = {planar_wsum, ch * (int32_t)sizeof(int32_t)};

    typedef arm_cmsis_nn_status (*dw_fn)(const cmsis_nn_context *,
                                         const cmsis_nn_context *,
                                         const cmsis_nn_dw_conv_params *,
                                         const cmsis_nn_per_channel_quant_params *,
                                         const cmsis_nn_dims *,
                                         const int8_t *,
                                         const cmsis_nn_dims *,
                                         const int8_t *,
                                         const cmsis_nn_dims *,
                                         const int32_t *,
                                         const cmsis_nn_dims *,
                                         int8_t *);
    const dw_fn entries[] = {
        arm_depthwise_conv_s8_opt, arm_depthwise_conv_s8_opt_planar, arm_depthwise_conv_s8_opt_channelwise};
    const cmsis_nn_dims wrong_out = {1, ih, iw, ch + 1};
    cmsis_nn_dw_conv_params dil_h = params, dil_w = params;
    dil_h.dilation.h = 2;
    dil_w.dilation.w = 0;
    const cmsis_nn_context no_buf = {NULL, size};
    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); i++)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          entries[i](&ctx,
                                     &wsum,
                                     &params,
                                     &quant,
                                     &input_dims,
                                     planar_in,
                                     &filter_dims,
                                     planar_ker,
                                     &bias_dims,
                                     planar_bias,
                                     &wrong_out,
                                     planar_out));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          entries[i](&ctx,
                                     &wsum,
                                     &dil_h,
                                     &quant,
                                     &input_dims,
                                     planar_in,
                                     &filter_dims,
                                     planar_ker,
                                     &bias_dims,
                                     planar_bias,
                                     &output_dims,
                                     planar_out));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          entries[i](&ctx,
                                     &wsum,
                                     &dil_w,
                                     &quant,
                                     &input_dims,
                                     planar_in,
                                     &filter_dims,
                                     planar_ker,
                                     &bias_dims,
                                     planar_bias,
                                     &output_dims,
                                     planar_out));
        if (size > 0)
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                              entries[i](&no_buf,
                                         &wsum,
                                         &params,
                                         &quant,
                                         &input_dims,
                                         planar_in,
                                         &filter_dims,
                                         planar_ker,
                                         &bias_dims,
                                         planar_bias,
                                         &output_dims,
                                         planar_out));
        }
    }

#if defined(ARM_MATH_DSP) && defined(ARM_MATH_MVEI)
    /* Where the weight sums are read, a NULL buffer is an argument error on every entry, including the planar one for
       a layer it takes. */
    const cmsis_nn_context no_wsum = {NULL, 0};
    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); i++)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          entries[i](&ctx,
                                     &no_wsum,
                                     &params,
                                     &quant,
                                     &input_dims,
                                     planar_in,
                                     &filter_dims,
                                     planar_ker,
                                     &bias_dims,
                                     planar_bias,
                                     &output_dims,
                                     planar_out));
    }
#endif

    /* A context too small for the plane: the planar entry declines without writing. arm_depthwise_conv_s8_opt() is
       not called here, because its channel path needs the full arm_depthwise_conv_s8_opt_get_buffer_size(); see
       planar_no_fit_arm_depthwise_conv_s8_opt() for its fallback. */
    const cmsis_nn_context small = {planar_scratch, 16};
    memset(planar_out, 0x5A, sizeof(planar_out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                      arm_depthwise_conv_s8_opt_planar(&small,
                                                       &wsum,
                                                       &params,
                                                       &quant,
                                                       &input_dims,
                                                       planar_in,
                                                       &filter_dims,
                                                       planar_ker,
                                                       &bias_dims,
                                                       planar_bias,
                                                       &output_dims,
                                                       planar_out));
    for (int32_t i = 0; i < ih * iw * ch; i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, planar_out[i]);
    }
}

/* A layer that meets every planar condition except the scratch: its plane exceeds the
   arm_depthwise_conv_s8_opt_get_buffer_size() scratch, so the planar entry declines without writing and
   arm_depthwise_conv_s8_opt() computes the layer on the channel path without writing past that scratch. */
void planar_no_fit_arm_depthwise_conv_s8_opt(void)
{
    const int32_t ih = 64, iw = 80, ch = 2, kh = 3, kw = 3;
#if defined(ARM_MATH_DSP) && defined(ARM_MATH_MVEI)
    const cmsis_nn_dw_conv_params params = {
        .ch_mult = 1, .stride = {1, 1}, .padding = {1, 1}, .dilation = {1, 1}, .activation = {-128, 127}};
    const cmsis_nn_dims input_dims = {1, ih, iw, ch}, filter_dims = {1, kh, kw, ch}, output_dims = {1, ih, iw, ch};
    TEST_ASSERT_TRUE(arm_depthwise_conv_s8_opt_planar_supported(&params, &input_dims, &filter_dims, &output_dims) == 0);
    TEST_ASSERT_TRUE(arm_nn_depthwise_conv_s8_planar_bytes(&params, &input_dims, &filter_dims, &output_dims) >
                     arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims));
#endif
    planar_case(ih, iw, ch, kh, kw, 1, 1, 1, 0, 3, -128, 127);
}

/* A plane whose size does not fit in int32 (65536 x 65536 bytes) must be declined, not wrapped into a small size that
   passes the scratch check. The data buffers are never touched. */
void planar_size_overflow_arm_depthwise_conv_s8_opt(void)
{
#if defined(ARM_MATH_MVEI)
    const cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                            .output_offset = 0,
                                            .ch_mult = 1,
                                            .stride = {1, 1},
                                            .padding = {0, 0},
                                            .dilation = {1, 1},
                                            .activation = {-128, 127}};
    const cmsis_nn_per_channel_quant_params quant = {planar_mult, planar_shift};
    const cmsis_nn_dims input_dims = {1, 65534, 8, 1}, filter_dims = {1, 3, 65529, 1}, output_dims = {1, 65534, 8, 1};
    const cmsis_nn_context ctx = {planar_scratch, (int32_t)sizeof(planar_scratch)};
    const cmsis_nn_context wsum = {planar_wsum, (int32_t)sizeof(planar_wsum)};
    memset(planar_scratch, 0x3C, sizeof(planar_scratch));
    memset(planar_out, 0x5A, sizeof(planar_out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                      arm_nn_depthwise_conv_s8_planar(
                          &ctx, &wsum, &params, &quant, &input_dims, planar_in, &filter_dims, planar_ker, &output_dims, planar_out));
    for (size_t i = 0; i < sizeof(planar_out); i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, planar_out[i]);
    }
    for (size_t i = 0; i < sizeof(planar_scratch); i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x3C, planar_scratch[i]);
    }

    /* Extents near INT32_MAX on both sides must be declined before the 64-bit plane product can overflow. */
    const cmsis_nn_dw_conv_params wide_params = {.input_offset = 0,
                                                 .output_offset = 0,
                                                 .ch_mult = 1,
                                                 .stride = {1, 1},
                                                 .padding = {0, 0},
                                                 .dilation = {128, 1},
                                                 .activation = {-128, 127}};
    const cmsis_nn_dims wide_in = {1, INT32_MAX, INT32_MAX, 1}, wide_filter = {1, INT32_MAX, INT32_MAX, 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                      arm_nn_depthwise_conv_s8_planar(
                          &ctx, &wsum, &wide_params, &quant, &wide_in, planar_in, &wide_filter, planar_ker, &wide_in, planar_out));
    for (size_t i = 0; i < sizeof(planar_out); i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, planar_out[i]);
    }
#endif
}

/* arm_nn_depthwise_conv_s8_planar() declines a layer it would take, writing nothing, when the context is smaller than
   its plane, and takes it once the plane fits. This test calls the planar kernel only. */
void planar_small_context_arm_depthwise_conv_s8_opt(void)
{
#if defined(ARM_MATH_MVEI)
    const cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                            .output_offset = 0,
                                            .ch_mult = 1,
                                            .stride = {1, 1},
                                            .padding = {1, 1},
                                            .dilation = {1, 1},
                                            .activation = {-128, 127}};
    const cmsis_nn_per_channel_quant_params quant = {planar_mult, planar_shift};
    const cmsis_nn_dims input_dims = {1, 7, 9, 5}, filter_dims = {1, 3, 3, 5}, output_dims = {1, 7, 9, 5};
    /* The plane is (9 + 2) * (7 + 2) + 32 = 131 bytes. */
    const cmsis_nn_context ctx = {planar_scratch, 130};
    const cmsis_nn_context wsum = {planar_wsum, (int32_t)sizeof(planar_wsum)};
    memset(planar_scratch, 0x3C, sizeof(planar_scratch));
    memset(planar_out, 0x5A, sizeof(planar_out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                      arm_nn_depthwise_conv_s8_planar(
                          &ctx, &wsum, &params, &quant, &input_dims, planar_in, &filter_dims, planar_ker, &output_dims, planar_out));
    for (size_t i = 0; i < sizeof(planar_out); i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, planar_out[i]);
    }
    for (size_t i = 0; i < sizeof(planar_scratch); i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x3C, planar_scratch[i]);
    }
    const cmsis_nn_context fits = {planar_scratch, 131};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_depthwise_conv_s8_planar(
                          &fits, &wsum, &params, &quant, &input_dims, planar_in, &filter_dims, planar_ker, &output_dims, planar_out));
#endif
}

/* arm_nn_depthwise_conv_nt_t_padded_s8() with a channel count that is not a multiple of 4 must stay inside every
   operand. Each operand in turn is placed against an MPU gap and the result is compared with a scalar reference. The
   active channels are also run as the first part of a wider tensor, and without a bias. */
void padded_nt_t_bounds_arm_depthwise_conv_s8_opt(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    enum
    {
        op_none,
        op_lhs,
        op_lhs_rows,
        op_rhs,
        op_bias,
        op_mult,
        op_shift,
        op_out,
        op_end
    };
    enum
    {
        max_ch = 20
    };
    static int8_t lhs[4 * 9 * CH_IN_BLOCK_MVE], rhs[9 * max_ch], out[4 * max_ch], expected[4 * max_ch];
    static int32_t bias[max_ch], mult[max_ch], shift[max_ch];
    const int32_t channels[] = {1, 2, 3, 5, 7, 17};
    const int32_t taps[] = {1, 3, 9};
    const int32_t input_offset = 5, out_offset = -3;
    for (int32_t variant = 0; variant < 3; variant++)
    {
        const int32_t no_bias = variant == 2;
        for (size_t i_ch = 0; i_ch < sizeof(channels) / sizeof(channels[0]); i_ch++)
        {
            for (size_t i_tap = 0; i_tap < sizeof(taps) / sizeof(taps[0]); i_tap++)
            {
                const int32_t ch = channels[i_ch], k = taps[i_tap];
                const int32_t total = variant == 1 ? ch + 3 : ch;
                const int32_t lhs_bytes = 4 * k * CH_IN_BLOCK_MVE;
                /* The rows in use end with the last active channel of the last row. */
                const int32_t lhs_rows = (4 * k - 1) * CH_IN_BLOCK_MVE + ch;
                const int32_t rhs_bytes = (k - 1) * total + ch;
                const int32_t out_bytes = 3 * total + ch;
                uint32_t seed = (uint32_t)(k * 131 + ch * 7 + variant);
                for (int32_t i = 0; i < lhs_bytes; i++)
                {
                    seed = seed * 1664525u + 1013904223u;
                    lhs[i] = (int8_t)(seed >> 24);
                }
                for (int32_t i = 0; i < k * total; i++)
                {
                    seed = seed * 1664525u + 1013904223u;
                    rhs[i] = (int8_t)(seed >> 24);
                }
                for (int32_t i = 0; i < ch; i++)
                {
                    seed = seed * 1664525u + 1013904223u;
                    bias[i] = no_bias ? 0 : (int32_t)(seed >> 20) - 2048;
                    mult[i] = 0x40000000 + (i % 7) * 0x4000000;
                    shift[i] = -7 - (i % 3);
                }
                memset(expected, 0x5A, sizeof(expected));
                for (int32_t r = 0; r < 4; r++)
                {
                    for (int32_t c = 0; c < ch; c++)
                    {
                        int32_t acc = bias[c];
                        for (int32_t t = 0; t < k; t++)
                        {
                            acc += (lhs[(r * k + t) * CH_IN_BLOCK_MVE + c] + input_offset) * rhs[t * total + c];
                        }
                        int32_t v = arm_nn_requantize(acc, mult[c], shift[c]) + out_offset;
                        v = v < -128 ? -128 : (v > 127 ? 127 : v);
                        expected[r * total + c] = (int8_t)v;
                    }
                }
                for (int op = op_none; op < op_end; op++)
                {
                    if (op == op_bias && no_bias)
                    {
                        continue;
                    }
                    const int8_t *l = op == op_lhs ? guard_place(lhs, lhs_bytes)
                                                   : (op == op_lhs_rows ? guard_place(lhs, lhs_rows) : lhs);
                    const int8_t *w = op == op_rhs ? guard_place(rhs, rhs_bytes) : rhs;
                    const int32_t *b =
                        no_bias ? NULL : (op == op_bias ? guard_place(bias, ch * sizeof(int32_t)) : bias);
                    const int32_t *m = op == op_mult ? guard_place(mult, ch * sizeof(int32_t)) : mult;
                    const int32_t *sh = op == op_shift ? guard_place(shift, ch * sizeof(int32_t)) : shift;
                    int8_t *o = op == op_out ? guard_end(out_bytes) : out;
                    memset(o, 0x5A, out_bytes);

                    guard_gap_enable();
                    const arm_cmsis_nn_status status = arm_nn_depthwise_conv_nt_t_padded_s8(
                        l, w, input_offset, ch, total, sh, m, out_offset, -128, 127, (uint16_t)k, b, o);
                    guard_gap_disable();

                    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
                    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, o, out_bytes);
                }
            }
        }
    }
#endif
}

/* The direct 3x3 entries: bytewise equal to arm_depthwise_conv_s8() inside their gate, with guard bytes around the
   output and the scratch, and NO_IMPL with nothing written outside it. */
typedef arm_cmsis_nn_status (*dw3_entry_fn)(const cmsis_nn_context *,
                                            const cmsis_nn_context *,
                                            const cmsis_nn_dw_conv_params *,
                                            const cmsis_nn_per_channel_quant_params *,
                                            const cmsis_nn_dims *,
                                            const int8_t *,
                                            const cmsis_nn_dims *,
                                            const int8_t *,
                                            const cmsis_nn_dims *,
                                            const int32_t *,
                                            const cmsis_nn_dims *,
                                            int8_t *);

/* Static (.bss) buffers of about 61 KB in all, kept within a 64 KiB budget on the Corstone-300 build. The scratch is
   16-byte aligned so that the misalignment of each case below is the offset it adds, whatever the linker placement. */
#define DW3_MAX_IN (24 * 24 * 32)
#define DW3_MAX_OUT (16 * 32 * 32)
#define DW3_MAX_CH (192)
#define DW3_GUARD (32)
#define DW3_MAX_SCRATCH (6144)
static int8_t dw3_in[DW3_MAX_IN], dw3_ker[9 * DW3_MAX_CH], dw3_ref[DW3_MAX_OUT];
static int8_t dw3_out[DW3_GUARD + DW3_MAX_OUT + DW3_GUARD];
static int8_t dw3_scratch[DW3_GUARD + DW3_MAX_SCRATCH + DW3_GUARD] __attribute__((aligned(16)));
static int32_t dw3_bias[DW3_MAX_CH], dw3_mult[DW3_MAX_CH], dw3_shift[DW3_MAX_CH], dw3_wsum[DW3_MAX_CH];

/* Scratch bytes the direct entries need, from the public sizer; dw3_buffer_size checks it against the documented
   3008 + input W x C + 16. */
static int32_t dw3_scratch_need(const cmsis_nn_dims *input_dims)
{
    return arm_depthwise_conv_s8_opt_3x3_get_buffer_size(input_dims);
}

typedef struct
{
    int32_t ih, iw, ch, sy, sx, pad_y, pad_x;
    int32_t oh, ow; /* 0: (i + 2 * pad - 3) / s + 1 */
    int32_t exact;  /* 1: ctx->size is exactly the documented minimum */
} dw3_shape;

/* Seeded data. Shifts cycle through positive and negative values; a positive-shift channel gets a small multiplier
   so that its outputs are not all clamped. */
static void dw3_fill(const dw3_shape *s, uint32_t seed)
{
    static const int32_t shifts[] = {-9, -8, 1, -10, -7, 0, -11, -6};
    for (int32_t i = 0; i < s->ih * s->iw * s->ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        dw3_in[i] = (int8_t)(seed >> 24);
    }
    for (int32_t i = 0; i < 9 * s->ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        dw3_ker[i] = (int8_t)(seed >> 24);
    }
    for (int32_t i = 0; i < s->ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        dw3_bias[i] = (int32_t)(seed >> 16) - 32768;
        dw3_shift[i] = shifts[i % 8];
        dw3_mult[i] = dw3_shift[i] >= 0 ? 0x00100000 + i * 0x1000 : 0x40000000 + (int32_t)((seed >> 2) & 0x3FFFFFFF);
    }
}

static void dw3_dims(const dw3_shape *s,
                     cmsis_nn_dims *input_dims,
                     cmsis_nn_dims *filter_dims,
                     cmsis_nn_dims *bias_dims,
                     cmsis_nn_dims *output_dims)
{
    const int32_t oh = s->oh ? s->oh : (s->ih + 2 * s->pad_y - 3) / s->sy + 1;
    const int32_t ow = s->ow ? s->ow : (s->iw + 2 * s->pad_x - 3) / s->sx + 1;
    *input_dims = (cmsis_nn_dims){1, s->ih, s->iw, s->ch};
    *filter_dims = (cmsis_nn_dims){1, 3, 3, s->ch};
    *bias_dims = (cmsis_nn_dims){1, 1, 1, s->ch};
    *output_dims = (cmsis_nn_dims){1, oh, ow, s->ch};
}

static void dw3_check_untouched(const int8_t *buf, int32_t n, int8_t fill)
{
    for (int32_t i = 0; i < n; i++)
    {
        TEST_ASSERT_EQUAL_INT8(fill, buf[i]);
    }
}

static void dw3_case(const dw3_shape *s, int32_t input_offset, int32_t output_offset, int32_t act_min, int32_t act_max)
{
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
    dw3_dims(s, &input_dims, &filter_dims, &bias_dims, &output_dims);
    const int32_t out_bytes = output_dims.h * output_dims.w * s->ch;
    TEST_ASSERT_TRUE(s->ih * s->iw * s->ch <= DW3_MAX_IN && out_bytes <= DW3_MAX_OUT && s->ch <= DW3_MAX_CH);
    dw3_fill(s, (uint32_t)(s->ih * 977 + s->iw * 131 + s->ch * 7 + s->sy * 3 + s->sx + input_offset));

    const cmsis_nn_dw_conv_params params = {.input_offset = input_offset,
                                            .output_offset = output_offset,
                                            .ch_mult = 1,
                                            .stride = {s->sx, s->sy},
                                            .padding = {s->pad_x, s->pad_y},
                                            .dilation = {1, 1},
                                            .activation = {act_min, act_max}};
    const cmsis_nn_per_channel_quant_params quant = {dw3_mult, dw3_shift};
    const cmsis_nn_context none = {NULL, 0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_s8(&none,
                                            &params,
                                            &quant,
                                            &input_dims,
                                            dw3_in,
                                            &filter_dims,
                                            dw3_ker,
                                            &bias_dims,
                                            dw3_bias,
                                            &output_dims,
                                            dw3_ref));
    (void)arm_depthwise_convolve_weight_sum(dw3_wsum,
                                            dw3_scratch,
                                            dw3_ker,
                                            &params,
                                            &input_dims,
                                            &filter_dims,
                                            &output_dims,
                                            input_offset,
                                            dw3_bias);
    const cmsis_nn_context wsum = {dw3_wsum, s->ch * (int32_t)sizeof(int32_t)};

    /* The documented scratch where it suffices, else the documented minimum; the buffer start is misaligned by a
       shape-dependent amount so the kernel's own alignment stays inside ctx->size. */
    const int32_t need = dw3_scratch_need(&input_dims);
    const int32_t opt_size = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    const int32_t size = (s->exact || opt_size < need) ? need : opt_size;
    TEST_ASSERT_TRUE(size <= DW3_MAX_SCRATCH);
#if defined(ARM_MATH_MVEI)
    /* The 3x3 scratch of arm_depthwise_conv_s8_opt() covers every input W x C <= 1440. */
    TEST_ASSERT_TRUE(opt_size >= 3008 + 1440 + 16);
    TEST_ASSERT_EQUAL(s->exact || s->iw * s->ch > 1440, size == need);
#endif
    const int32_t misalign = (s->ih + s->iw + s->ch + s->sx) % 16;
    const cmsis_nn_context ctx = {dw3_scratch + DW3_GUARD + misalign, size};

    const dw3_entry_fn entries[] = {arm_depthwise_conv_s8_opt_3x3, arm_depthwise_conv_s8_opt_3x3_c64_s1};
    for (size_t e = 0; e < sizeof(entries) / sizeof(entries[0]); e++)
    {
        memset(dw3_out, 0x5A, sizeof(dw3_out));
        memset(dw3_scratch, 0x3C, sizeof(dw3_scratch));
        const arm_cmsis_nn_status status = entries[e](&ctx,
                                                      &wsum,
                                                      &params,
                                                      &quant,
                                                      &input_dims,
                                                      dw3_in,
                                                      &filter_dims,
                                                      dw3_ker,
                                                      &bias_dims,
                                                      dw3_bias,
                                                      &output_dims,
                                                      dw3_out + DW3_GUARD);
#if defined(ARM_MATH_MVEI)
        const int32_t takes = e == 0 || (s->ch == 64 && s->sy == 1);
#else
        const int32_t takes = 0;
#endif
        if (status != (takes ? ARM_CMSIS_NN_SUCCESS : ARM_CMSIS_NN_NO_IMPL_ERROR) ||
            (takes && memcmp(dw3_ref, dw3_out + DW3_GUARD, (size_t)out_bytes) != 0))
        {
            printf("dw3 entry %d: %dx%dx%d stride %d,%d pad %d,%d input_offset %d\n",
                   (int)e,
                   (int)s->ih,
                   (int)s->iw,
                   (int)s->ch,
                   (int)s->sy,
                   (int)s->sx,
                   (int)s->pad_y,
                   (int)s->pad_x,
                   (int)input_offset);
        }
        TEST_ASSERT_EQUAL(takes ? ARM_CMSIS_NN_SUCCESS : ARM_CMSIS_NN_NO_IMPL_ERROR, status);
        if (takes)
        {
            TEST_ASSERT_EQUAL_INT8_ARRAY(dw3_ref, dw3_out + DW3_GUARD, out_bytes);
            dw3_check_untouched(dw3_out, DW3_GUARD, 0x5A);
            dw3_check_untouched(dw3_out + DW3_GUARD + out_bytes, DW3_GUARD, 0x5A);
            dw3_check_untouched(dw3_scratch, DW3_GUARD + misalign, 0x3C);
            dw3_check_untouched(dw3_scratch + DW3_GUARD + misalign + size, DW3_GUARD, 0x3C);
        }
        else
        {
            dw3_check_untouched(dw3_out, (int32_t)sizeof(dw3_out), 0x5A);
            dw3_check_untouched(dw3_scratch, (int32_t)sizeof(dw3_scratch), 0x3C);
        }
    }
}

void dw3_shapes_arm_depthwise_conv_s8_opt(void)
{
    /* ih, iw, ch, stride_y, stride_x, pad_y, pad_x, oh, ow, exact scratch */
    const dw3_shape shapes[] = {
        {25, 5, 64, 1, 1, 1, 1, 0, 0, 0},    /* KWS DS-CNN, output_y % 3 == 1 */
        {12, 12, 64, 1, 1, 1, 1, 0, 0, 0},   /* VWW dw9, output_y % 3 == 0 */
        {12, 12, 64, 2, 2, 0, 0, 6, 6, 0},   /* VWW dw11, SAME: bottom and right padding only */
        {6, 6, 128, 1, 1, 1, 1, 0, 0, 0},    /* VWW dw13-21 */
        {24, 24, 32, 2, 2, 0, 0, 12, 12, 0}, /* VWW dw7 */
        {16, 32, 32, 1, 1, 1, 1, 0, 0, 0},   /* VWW dw5 24x24x32, reduced to 16 x 32 */
        {32, 32, 16, 2, 2, 0, 0, 16, 16, 0}, /* VWW dw3, reduced */
        {11, 7, 16, 1, 1, 1, 1, 0, 0, 0},    /* output_y % 3 == 2 */
        {10, 9, 16, 2, 2, 1, 1, 0, 0, 0},
        {4, 4, 16, 1, 1, 1, 1, 0, 0, 0}, /* 16 output pixels */
        {10, 13, 32, 1, 2, 1, 0, 0, 0, 0},
        {9, 6, 36, 2, 1, 0, 1, 0, 0, 0},
        {7, 11, 64, 1, 2, 1, 1, 0, 0, 0},
        {11, 5, 64, 2, 1, 1, 0, 0, 0, 0},
        {8, 6, 64, 1, 1, 0, 1, 0, 0, 0},
        {9, 4, 64, 1, 1, 1, 0, 0, 0, 0},
        {5, 8, 64, 1, 1, 1, 1, 0, 0, 1},
        {3, 6, 64, 1, 1, 1, 1, 0, 0, 0}, /* 3 output rows: one row block */
        {8, 5, 68, 1, 1, 1, 1, 0, 0, 0}, /* channel passes 64 + 4 */
        {9, 9, 68, 2, 2, 1, 1, 0, 0, 0},
        {4, 6, 128, 1, 1, 1, 1, 0, 0, 0},
        {9, 9, 128, 2, 2, 0, 0, 0, 0, 0},
        {4, 4, 192, 1, 1, 1, 1, 0, 0, 0},   /* three channel passes */
        {6, 24, 68, 1, 1, 1, 1, 0, 0, 0},   /* W x C > 1440: the documented minimum scratch */
        {5, 48, 32, 2, 1, 1, 1, 0, 0, 1},   /* W x C > 1440 */
        {8, 8, 16, 1, 1, 1, 1, 0, 0, 0},    /* last window centre on the last input column */
        {7, 9, 36, 1, 2, 1, 1, 0, 0, 1},    /* stride 2: last centre on the last column */
        {6, 3, 16, 1, 1, 1, 1, 0, 0, 0},    /* input W 3: both edges in one window */
        {3, 13, 16, 1, 2, 1, 0, 0, 0, 0},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
    {
        dw3_case(&shapes[i], 128, -3, -128, 127);
        dw3_case(&shapes[i], -127, 5, -60, 70);
        dw3_case(&shapes[i], 7, -128, -128, 0);
    }
}

/* Each layer here is a valid depthwise layer just outside one gate condition. Every entry declines it with
   NO_IMPL and writes neither the output nor the scratch. */
static void dw3_expect_decline(const dw3_entry_fn entry,
                               const cmsis_nn_context *ctx,
                               const cmsis_nn_context *wsum,
                               const cmsis_nn_dw_conv_params *params,
                               const cmsis_nn_dims *input_dims,
                               const cmsis_nn_dims *filter_dims,
                               const cmsis_nn_dims *output_dims)
{
    const cmsis_nn_per_channel_quant_params quant = {dw3_mult, dw3_shift};
    const cmsis_nn_dims bias_dims = {1, 1, 1, output_dims->c};
    memset(dw3_out, 0x5A, sizeof(dw3_out));
    memset(dw3_scratch, 0x3C, sizeof(dw3_scratch));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                      entry(ctx,
                            wsum,
                            params,
                            &quant,
                            input_dims,
                            dw3_in,
                            filter_dims,
                            dw3_ker,
                            &bias_dims,
                            dw3_bias,
                            output_dims,
                            dw3_out + DW3_GUARD));
    dw3_check_untouched(dw3_out, (int32_t)sizeof(dw3_out), 0x5A);
    dw3_check_untouched(dw3_scratch, (int32_t)sizeof(dw3_scratch), 0x3C);
}

static void dw3_declines(const dw3_entry_fn entry, const int32_t ch)
{
    const dw3_shape base_shape = {8, 8, ch, 1, 1, 1, 1, 0, 0, 0};
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
    dw3_dims(&base_shape, &input_dims, &filter_dims, &bias_dims, &output_dims);
    dw3_fill(&base_shape, 11u);
    const cmsis_nn_dw_conv_params params = {.input_offset = 3,
                                            .output_offset = -3,
                                            .ch_mult = 1,
                                            .stride = {1, 1},
                                            .padding = {1, 1},
                                            .dilation = {1, 1},
                                            .activation = {-128, 127}};
    const int32_t need = dw3_scratch_need(&input_dims);
    const cmsis_nn_context ctx = {dw3_scratch + DW3_GUARD, need};
    const cmsis_nn_context wsum = {dw3_wsum, ch * (int32_t)sizeof(int32_t)};

    /* The unmodified layer is taken, so each decline below comes from its one change. */
    const cmsis_nn_per_channel_quant_params quant = {dw3_mult, dw3_shift};
    (void)arm_depthwise_convolve_weight_sum(dw3_wsum,
                                            dw3_scratch,
                                            dw3_ker,
                                            &params,
                                            &input_dims,
                                            &filter_dims,
                                            &output_dims,
                                            params.input_offset,
                                            dw3_bias);
    const arm_cmsis_nn_status status = entry(&ctx,
                                             &wsum,
                                             &params,
                                             &quant,
                                             &input_dims,
                                             dw3_in,
                                             &filter_dims,
                                             dw3_ker,
                                             &bias_dims,
                                             dw3_bias,
                                             &output_dims,
                                             dw3_out + DW3_GUARD);
#if defined(ARM_MATH_MVEI)
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
#else
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, status);
#endif

    /* Channel count: below 16, not a multiple of 4. */
    const int32_t bad_ch[] = {12, 62, 18};
    for (size_t i = 0; i < sizeof(bad_ch) / sizeof(bad_ch[0]); i++)
    {
        const cmsis_nn_dims in_c = {1, 8, 8, bad_ch[i]}, f_c = {1, 3, 3, bad_ch[i]}, out_c = {1, 8, 8, bad_ch[i]};
        dw3_expect_decline(entry, &ctx, &wsum, &params, &in_c, &f_c, &out_c);
    }
    {
        /* The KWS shape with 62 channels */
        const cmsis_nn_dims in_c = {1, 25, 5, 62}, f_c = {1, 3, 3, 62}, out_c = {1, 25, 5, 62};
        const cmsis_nn_context big = {dw3_scratch + DW3_GUARD, 3008 + 5 * 62 + 16};
        dw3_expect_decline(entry, &big, &wsum, &params, &in_c, &f_c, &out_c);
    }
    /* Filter 5x5; 1x3, 3x1, 5x3 and 3x5 (H x W), each with the output of the 3x3 layer so that only the filter is
       outside the gate */
    {
        const cmsis_nn_dims f5 = {1, 5, 5, ch}, out5 = {1, 6, 6, ch};
        dw3_expect_decline(entry, &ctx, &wsum, &params, &input_dims, &f5, &out5);
        const cmsis_nn_dims f_hw[] = {{1, 1, 3, ch}, {1, 3, 1, ch}, {1, 5, 3, ch}, {1, 3, 5, ch}};
        for (size_t i = 0; i < sizeof(f_hw) / sizeof(f_hw[0]); i++)
        {
            dw3_expect_decline(entry, &ctx, &wsum, &params, &input_dims, &f_hw[i], &output_dims);
        }
    }
    /* Stride 0 in each dimension */
    {
        cmsis_nn_dw_conv_params p = params;
        p.stride.w = 0;
        dw3_expect_decline(entry, &ctx, &wsum, &p, &input_dims, &filter_dims, &output_dims);
        p = params;
        p.stride.h = 0;
        dw3_expect_decline(entry, &ctx, &wsum, &p, &input_dims, &filter_dims, &output_dims);
    }
    /* Input H 0 and output W 0 */
    {
        const cmsis_nn_dims in_h0 = {1, 0, 8, ch}, out_w0 = {1, 8, 0, ch};
        dw3_expect_decline(entry, &ctx, &wsum, &params, &in_h0, &filter_dims, &output_dims);
        dw3_expect_decline(entry, &ctx, &wsum, &params, &input_dims, &filter_dims, &out_w0);
    }
    /* One dimension of 4097, the rest inside the gate. The dimensions and ctx->size overstate the buffers, which the
       gate declines before touching. An output W over 4096 also puts the last window centre past an input W that is
       inside the gate, so its input is 4097 wide too. */
    {
        const cmsis_nn_dims in[] = {{1, 1, 4097, ch}, {1, 4097, 3, ch}, {1, 3, 4097, ch}, {1, 1, 3, ch}};
        const cmsis_nn_dims out[] = {{1, 3, 6, ch}, {1, 6, 3, ch}, {1, 3, 4097, ch}, {1, 4097, 1, ch}};
        for (size_t i = 0; i < sizeof(in) / sizeof(in[0]); i++)
        {
            const cmsis_nn_context big = {dw3_scratch + DW3_GUARD, dw3_scratch_need(&in[i])};
            dw3_expect_decline(entry, &big, &wsum, &params, &in[i], &filter_dims, &out[i]);
        }
    }
    /* Dilation 2, stride 3, padding 2 and -1, each in one dimension */
    for (int32_t dim = 0; dim < 2; dim++)
    {
        cmsis_nn_dw_conv_params p = params;
        cmsis_nn_dims out = output_dims;
        if (dim == 0)
        {
            p.dilation.w = 2;
            out.w = 6;
        }
        else
        {
            p.dilation.h = 2;
            out.h = 6;
        }
        dw3_expect_decline(entry, &ctx, &wsum, &p, &input_dims, &filter_dims, &out);

        p = params;
        out = output_dims;
        if (dim == 0)
        {
            p.stride.w = 3;
            out.w = 3;
        }
        else
        {
            p.stride.h = 3;
            out.h = 3;
        }
        dw3_expect_decline(entry, &ctx, &wsum, &p, &input_dims, &filter_dims, &out);

        p = params;
        out = output_dims;
        if (dim == 0)
        {
            p.padding.w = 2;
            out.w = 10;
        }
        else
        {
            p.padding.h = 2;
            out.h = 10;
        }
        dw3_expect_decline(entry, &ctx, &wsum, &p, &input_dims, &filter_dims, &out);

        p = params;
        out = output_dims;
        if (dim == 0)
        {
            p.padding.w = -1;
            out.w = 4;
        }
        else
        {
            p.padding.h = -1;
            out.h = 4;
        }
        dw3_expect_decline(entry, &ctx, &wsum, &p, &input_dims, &filter_dims, &out);
    }
    /* Batch 2 */
    {
        const cmsis_nn_dims in_n = {2, 4, 8, ch}, out_n = {2, 4, 8, ch};
        dw3_expect_decline(entry, &ctx, &wsum, &params, &in_n, &filter_dims, &out_n);
    }
    /* Channel multiplier 2: C_OUT = 2 x C_IN */
    {
        cmsis_nn_dw_conv_params p = params;
        p.ch_mult = 2;
        const cmsis_nn_dims in_m = {1, 8, 8, 16}, f_m = {1, 3, 3, 32}, out_m = {1, 8, 8, 32};
        dw3_expect_decline(entry, &ctx, &wsum, &p, &in_m, &f_m, &out_m);
    }
    /* Output of 2 rows; 15 output pixels (3 x 5); input width 2 */
    {
        const cmsis_nn_dims in_2 = {1, 2, 8, ch}, out_2 = {1, 2, 8, ch};
        dw3_expect_decline(entry, &ctx, &wsum, &params, &in_2, &filter_dims, &out_2);
        const cmsis_nn_dims in_15 = {1, 3, 5, ch}, out_15 = {1, 3, 5, ch};
        dw3_expect_decline(entry, &ctx, &wsum, &params, &in_15, &filter_dims, &out_15);
        const cmsis_nn_dims in_w2 = {1, 8, 2, ch}, out_w2 = {1, 8, 8, ch};
        dw3_expect_decline(entry, &ctx, &wsum, &params, &in_w2, &filter_dims, &out_w2);
    }
    /* A last output column whose window centre is past the input: 9 columns from 8 with padding 1, and 5 at
       stride 2. */
    {
        const cmsis_nn_dims out_w = {1, 8, 9, ch};
        dw3_expect_decline(entry, &ctx, &wsum, &params, &input_dims, &filter_dims, &out_w);
        cmsis_nn_dw_conv_params p = params;
        p.stride.w = 2;
        const cmsis_nn_dims out_s2 = {1, 8, 5, ch};
        dw3_expect_decline(entry, &ctx, &wsum, &p, &input_dims, &filter_dims, &out_s2);
    }
    /* Scratch one byte short of the documented minimum, NULL scratch, NULL weight sums */
    {
        const cmsis_nn_context short_ctx = {dw3_scratch + DW3_GUARD, need - 1};
        dw3_expect_decline(entry, &short_ctx, &wsum, &params, &input_dims, &filter_dims, &output_dims);
        const cmsis_nn_context no_buf = {NULL, need};
        dw3_expect_decline(entry, &no_buf, &wsum, &params, &input_dims, &filter_dims, &output_dims);
        dw3_expect_decline(entry, NULL, &wsum, &params, &input_dims, &filter_dims, &output_dims);
        const cmsis_nn_context no_wsum = {NULL, 0};
        dw3_expect_decline(entry, &ctx, &no_wsum, &params, &input_dims, &filter_dims, &output_dims);
        dw3_expect_decline(entry, &ctx, NULL, &params, &input_dims, &filter_dims, &output_dims);
    }
}

void dw3_declines_arm_depthwise_conv_s8_opt(void)
{
    dw3_declines(arm_depthwise_conv_s8_opt_3x3, 16);
    dw3_declines(arm_depthwise_conv_s8_opt_3x3, 64);
    dw3_declines(arm_depthwise_conv_s8_opt_3x3_c64_s1, 64);

    /* 2052 channels, and an input or an output of 2^31 elements with every dimension inside the gate. The dimensions
       and ctx->size overstate the buffers, which the gate declines before touching. With the MPU gap, the scratch is
       placed so that the pad row of a kernel that ran these layers would start at the gap and fault on its first
       byte, rather than run hundreds of KB past the static buffers. */
    {
        const cmsis_nn_dw_conv_params params = {.input_offset = 3,
                                                .output_offset = -3,
                                                .ch_mult = 1,
                                                .stride = {1, 1},
                                                .padding = {1, 1},
                                                .dilation = {1, 1},
                                                .activation = {-128, 127}};
        const cmsis_nn_dims in[] = {{1, 1, 3, 2052}, {1, 4096, 256, 2048}, {1, 1, 256, 2048}};
        const cmsis_nn_dims out[] = {{1, 6, 3, 2052}, {1, 3, 6, 2048}, {1, 4096, 256, 2048}};
        for (size_t i = 0; i < sizeof(in) / sizeof(in[0]); i++)
        {
            const cmsis_nn_dims f = {1, 3, 3, in[i].c};
            int8_t *buf = dw3_scratch + DW3_GUARD;
#if defined(MPU_GUARD_AVAILABLE)
            buf = guard_end(3008);
            guard_gap_enable();
#endif
            const cmsis_nn_context big = {buf, dw3_scratch_need(&in[i])};
            const cmsis_nn_context wsum = {dw3_wsum, in[i].c * (int32_t)sizeof(int32_t)};
            dw3_expect_decline(arm_depthwise_conv_s8_opt_3x3, &big, &wsum, &params, &in[i], &f, &out[i]);
            dw3_expect_decline(arm_depthwise_conv_s8_opt_3x3_c64_s1, &big, &wsum, &params, &in[i], &f, &out[i]);
#if defined(MPU_GUARD_AVAILABLE)
            guard_gap_disable();
#endif
        }
    }

    /* The C = 64, stride_h 1 entry also declines other channel counts and a vertical stride of 2 that the generic
       entry takes. */
    const int32_t chans[] = {32, 68, 64};
    for (size_t i = 0; i < sizeof(chans) / sizeof(chans[0]); i++)
    {
        const int32_t ch = chans[i];
        const int32_t sy = ch == 64 ? 2 : 1;
        const dw3_shape s = {8, 8, ch, sy, 1, 1, 1, 0, 0, 0};
        cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
        dw3_dims(&s, &input_dims, &filter_dims, &bias_dims, &output_dims);
        dw3_fill(&s, 5u);
        const cmsis_nn_dw_conv_params params = {.input_offset = 3,
                                                .output_offset = -3,
                                                .ch_mult = 1,
                                                .stride = {1, sy},
                                                .padding = {1, 1},
                                                .dilation = {1, 1},
                                                .activation = {-128, 127}};
        const cmsis_nn_context ctx = {dw3_scratch + DW3_GUARD, dw3_scratch_need(&input_dims)};
        const cmsis_nn_context wsum = {dw3_wsum, ch * (int32_t)sizeof(int32_t)};
        dw3_expect_decline(
            arm_depthwise_conv_s8_opt_3x3_c64_s1, &ctx, &wsum, &params, &input_dims, &filter_dims, &output_dims);
    }
}

/* The sizer returns 3008 + input W x C + 16 on every build, reads only W and C, and returns -1 for a negative W or C
   or a size past INT32_MAX. */
void dw3_buffer_size_arm_depthwise_conv_s8_opt(void)
{
    const int32_t wc[][2] = {{3, 16}, {5, 64}, {24, 68}, {4096, 2048}, {0, 0}, {1, INT32_MAX - 3024}};
    for (size_t i = 0; i < sizeof(wc) / sizeof(wc[0]); i++)
    {
        const cmsis_nn_dims in = {-7, -7, wc[i][0], wc[i][1]};
        TEST_ASSERT_EQUAL_INT32((int32_t)(3008 + (int64_t)wc[i][0] * wc[i][1] + 16),
                                arm_depthwise_conv_s8_opt_3x3_get_buffer_size(&in));
    }
    const int32_t bad[][2] = {{-1, 16}, {3, -4}, {1, INT32_MAX - 3023}, {INT32_MAX, 2}, {65536, 65536}};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
    {
        const cmsis_nn_dims in = {1, 8, bad[i][0], bad[i][1]};
        TEST_ASSERT_EQUAL_INT32(-1, arm_depthwise_conv_s8_opt_3x3_get_buffer_size(&in));
    }
}

/* Every operand ends at an unmapped gap in turn: no entry reads or writes past the input, filter, weight sums,
   multipliers, shifts, output or the documented minimum scratch. op_layout also ends the kernel's own 16-byte aligned
   scratch layout at the gap, so a write past the pad row faults even where the documented minimum leaves slack. */
void dw3_bounds_arm_depthwise_conv_s8_opt(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    enum
    {
        op_none,
        op_in,
        op_ker,
        op_wsum,
        op_mult,
        op_shift,
        op_out,
        op_scratch,
        op_layout,
        op_end
    };
    const dw3_shape shapes[] = {
        {6, 5, 16, 1, 1, 1, 1, 0, 0, 1},
        {7, 6, 64, 1, 1, 1, 1, 0, 0, 1},
        {5, 11, 36, 2, 2, 1, 1, 0, 0, 1},
        {8, 8, 16, 2, 2, 0, 0, 4, 4, 1},
        {4, 4, 68, 1, 1, 1, 1, 0, 0, 1},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
    {
        const dw3_shape *s = &shapes[i];
        cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
        dw3_dims(s, &input_dims, &filter_dims, &bias_dims, &output_dims);
        dw3_fill(s, (uint32_t)i + 1u);
        const int32_t in_bytes = s->ih * s->iw * s->ch;
        const int32_t out_bytes = output_dims.h * output_dims.w * s->ch;
        const int32_t need = dw3_scratch_need(&input_dims);
        TEST_ASSERT_TRUE(in_bytes <= GUARD_OFFSET && out_bytes <= GUARD_OFFSET && need <= GUARD_OFFSET);
        const cmsis_nn_dw_conv_params params = {.input_offset = 128,
                                                .output_offset = -3,
                                                .ch_mult = 1,
                                                .stride = {s->sx, s->sy},
                                                .padding = {s->pad_x, s->pad_y},
                                                .dilation = {1, 1},
                                                .activation = {-128, 127}};
        const cmsis_nn_context none = {NULL, 0};
        const cmsis_nn_per_channel_quant_params quant_ref = {dw3_mult, dw3_shift};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_depthwise_conv_s8(&none,
                                                &params,
                                                &quant_ref,
                                                &input_dims,
                                                dw3_in,
                                                &filter_dims,
                                                dw3_ker,
                                                &bias_dims,
                                                dw3_bias,
                                                &output_dims,
                                                dw3_ref));
        (void)arm_depthwise_convolve_weight_sum(dw3_wsum,
                                                dw3_scratch,
                                                dw3_ker,
                                                &params,
                                                &input_dims,
                                                &filter_dims,
                                                &output_dims,
                                                params.input_offset,
                                                dw3_bias);
        const dw3_entry_fn entries[] = {arm_depthwise_conv_s8_opt_3x3, arm_depthwise_conv_s8_opt_3x3_c64_s1};
        const size_t n_entries = (s->ch == 64 && s->sy == 1) ? 2 : 1;
        for (size_t e = 0; e < n_entries; e++)
        {
            for (int op = op_none; op < op_end; op++)
            {
                const size_t ch_bytes = (size_t)s->ch * sizeof(int32_t);
                const int8_t *in = op == op_in ? guard_place(dw3_in, (size_t)in_bytes) : dw3_in;
                const int8_t *ker = op == op_ker ? guard_place(dw3_ker, (size_t)(9 * s->ch)) : dw3_ker;
                const cmsis_nn_context wsum = {op == op_wsum ? guard_place(dw3_wsum, ch_bytes) : dw3_wsum,
                                               (int32_t)ch_bytes};
                const cmsis_nn_per_channel_quant_params quant = {
                    op == op_mult ? guard_place(dw3_mult, ch_bytes) : dw3_mult,
                    op == op_shift ? guard_place(dw3_shift, ch_bytes) : dw3_shift};
                int8_t *out = op == op_out ? guard_end((size_t)out_bytes) : dw3_out;
                int8_t *scratch = op == op_scratch ? guard_end((size_t)need) : dw3_scratch;
                if (op == op_layout)
                {
                    /* The layout (parameters, then the pad row) is need - 16 bytes from a 16-byte aligned start.
                       ctx->buf one byte past a 16-byte boundary spends 15 of the 16 slack bytes on alignment, so the
                       layout ends at the gap when W x C is a multiple of 16 (every shape here but 11 x 36), and the
                       last byte of ctx->size lies in the gap. */
                    const uintptr_t layout_start = ((uintptr_t)guard_end(0) - (uintptr_t)(need - 16)) & ~(uintptr_t)15;
                    scratch = (int8_t *)(layout_start - 15);
                }
                const cmsis_nn_context ctx = {scratch, need};
                memset(out, 0x5A, (size_t)out_bytes);

                guard_gap_enable();
                const arm_cmsis_nn_status status = entries[e](&ctx,
                                                              &wsum,
                                                              &params,
                                                              &quant,
                                                              &input_dims,
                                                              in,
                                                              &filter_dims,
                                                              ker,
                                                              &bias_dims,
                                                              dw3_bias,
                                                              &output_dims,
                                                              out);
                guard_gap_disable();

                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
                TEST_ASSERT_EQUAL_INT8_ARRAY(dw3_ref, out, out_bytes);
            }
        }
    }
#endif
}

/* The channel path of arm_depthwise_conv_s8_opt() and _channelwise() writes across the whole
   arm_depthwise_conv_s8_opt_get_buffer_size() scratch, so a non-zero ctx->size below it is rejected before any write
   (#582). A ctx->size of 0 opts out of the check. The stride-2 layer is outside the planar path. */
void undersized_context_arm_depthwise_conv_s8_opt(void)
{
    enum
    {
        ch = 16,
        in_hw = 5,
        out_hw = 3
    };
    static int8_t input[in_hw * in_hw * ch];
    static int8_t kernel[3 * 3 * ch];
    static int32_t bias[ch];
    static int32_t mult[ch];
    static int32_t shift[ch];
    static int32_t wsum[ch + 4];
    static int8_t output[out_hw * out_hw * ch];
    for (int i = 0; i < (int)sizeof(input); i++)
    {
        input[i] = (int8_t)((i * 7) % 19 - 9);
    }
    for (int i = 0; i < (int)sizeof(kernel); i++)
    {
        kernel[i] = (int8_t)((i * 5) % 13 - 6);
    }
    for (int i = 0; i < ch; i++)
    {
        mult[i] = 1 << 30;
        shift[i] = -3;
    }
    const cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                            .output_offset = 0,
                                            .ch_mult = 1,
                                            .stride = {2, 2},
                                            .padding = {1, 1},
                                            .dilation = {1, 1},
                                            .activation = {-128, 127}};
    const cmsis_nn_per_channel_quant_params quant = {mult, shift};
    const cmsis_nn_dims input_dims = {1, in_hw, in_hw, ch}, filter_dims = {1, 3, 3, ch},
                        bias_dims = {1, 1, 1, ch}, output_dims = {1, out_hw, out_hw, ch};
    const cmsis_nn_context wsum_ctx = {wsum, (int32_t)sizeof(wsum)};
    const int32_t need = arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &filter_dims);
    TEST_ASSERT_TRUE(need >= 0);
    if (need == 0)
    {
        return; /* no scratch on this build, so nothing to undersize */
    }
    int8_t *scratch = malloc((size_t)need);
    TEST_ASSERT_NOT_NULL(scratch);
    memset(wsum, 0, sizeof(wsum));

    for (int entry = 0; entry < 2; entry++)
    {
        arm_cmsis_nn_status (*const fn)(const cmsis_nn_context *,
                                        const cmsis_nn_context *,
                                        const cmsis_nn_dw_conv_params *,
                                        const cmsis_nn_per_channel_quant_params *,
                                        const cmsis_nn_dims *,
                                        const int8_t *,
                                        const cmsis_nn_dims *,
                                        const int8_t *,
                                        const cmsis_nn_dims *,
                                        const int32_t *,
                                        const cmsis_nn_dims *,
                                        int8_t *) =
            entry == 0 ? arm_depthwise_conv_s8_opt : arm_depthwise_conv_s8_opt_channelwise;

        const cmsis_nn_context small = {scratch, need - 1};
        memset(scratch, 0x3C, (size_t)need);
        memset(output, 0x5A, sizeof(output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          fn(&small,
                             &wsum_ctx,
                             &params,
                             &quant,
                             &input_dims,
                             input,
                             &filter_dims,
                             kernel,
                             &bias_dims,
                             bias,
                             &output_dims,
                             output));
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_INT8(0x5A, output[i]);
        }
        for (int32_t i = 0; i < need; i++)
        {
            TEST_ASSERT_EQUAL_INT8(0x3C, scratch[i]);
        }

        const cmsis_nn_context exact = {scratch, need};
        const cmsis_nn_context undeclared = {scratch, 0};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          fn(&exact,
                             &wsum_ctx,
                             &params,
                             &quant,
                             &input_dims,
                             input,
                             &filter_dims,
                             kernel,
                             &bias_dims,
                             bias,
                             &output_dims,
                             output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          fn(&undeclared,
                             &wsum_ctx,
                             &params,
                             &quant,
                             &input_dims,
                             input,
                             &filter_dims,
                             kernel,
                             &bias_dims,
                             bias,
                             &output_dims,
                             output));
    }

    /* Dimensions the sizer cannot size (-1) are rejected too, whatever ctx->size says. */
    const cmsis_nn_dims bad_filter_dims = {1, -3, 3, ch};
    const cmsis_nn_context declared = {scratch, need};
    memset(output, 0x5A, sizeof(output));
    TEST_ASSERT_EQUAL(-1, arm_depthwise_conv_s8_opt_get_buffer_size(&input_dims, &bad_filter_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_depthwise_conv_s8_opt_channelwise(&declared,
                                                            &wsum_ctx,
                                                            &params,
                                                            &quant,
                                                            &input_dims,
                                                            input,
                                                            &bad_filter_dims,
                                                            kernel,
                                                            &bias_dims,
                                                            bias,
                                                            &output_dims,
                                                            output));
    for (size_t i = 0; i < sizeof(output); i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, output[i]);
    }
    free(scratch);
}
