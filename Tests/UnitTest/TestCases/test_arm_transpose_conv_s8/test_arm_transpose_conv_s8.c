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

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <unity.h>

#include "../TestData/transpose_conv_1/test_data.h"
#include "../TestData/transpose_conv_2/test_data.h"
#include "../TestData/transpose_conv_3/test_data.h"
#include "../TestData/transpose_conv_4/test_data.h"
#include "../TestData/transpose_conv_5/test_data.h"
#include "../TestData/transpose_conv_6/test_data.h"
#include "../TestData/transpose_conv_7/test_data.h"
#include "../Utils/utils.h"
#include "../Utils/validate.h"

void transpose_conv_1_arm_transpose_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[TRANSPOSE_CONV_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_1_biases;
    const int8_t *kernel_data = transpose_conv_1_weights;
    const int8_t *input_data = transpose_conv_1_input;
    const int8_t *output_ref = transpose_conv_1_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_1_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_1_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_1_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_1_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_1_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_1_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_1_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_1_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_1_IN_CH;
    output_dims.n = TRANSPOSE_CONV_1_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_1_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_1_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_1_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_1_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_1_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_1_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_1_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_1_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_1_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_1_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_1_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_1_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_1_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_1_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_1_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_1_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = transpose_conv_params.input_offset;
    arm_convolve_weight_sum(
        weights_sum_ctx.buf, kernel_data, &input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    arm_cmsis_nn_status result = arm_transpose_conv_wrapper_s8(&ctx,
                                                               &weights_sum_ctx,
                                                               &reverse_conv_ctx,
                                                               &transpose_conv_params,
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
    if (reverse_conv_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void transpose_conv_2_arm_transpose_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[TRANSPOSE_CONV_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_2_biases;
    const int8_t *kernel_data = transpose_conv_2_weights;
    const int8_t *input_data = transpose_conv_2_input;
    const int8_t *output_ref = transpose_conv_2_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_2_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_2_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_2_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_2_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_2_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_2_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_2_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_2_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_2_IN_CH;
    output_dims.n = TRANSPOSE_CONV_2_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_2_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_2_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_2_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_2_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_2_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_2_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_2_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_2_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_2_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_2_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_2_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_2_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_2_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_2_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_2_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_2_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = transpose_conv_params.input_offset;
    arm_convolve_weight_sum(
        weights_sum_ctx.buf, kernel_data, &input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    arm_cmsis_nn_status result = arm_transpose_conv_wrapper_s8(&ctx,
                                                               &weights_sum_ctx,
                                                               &reverse_conv_ctx,
                                                               &transpose_conv_params,
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
    if (reverse_conv_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void transpose_conv_3_arm_transpose_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[TRANSPOSE_CONV_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_3_biases;
    const int8_t *kernel_data = transpose_conv_3_weights;
    const int8_t *input_data = transpose_conv_3_input;
    const int8_t *output_ref = transpose_conv_3_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_3_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_3_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_3_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_3_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_3_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_3_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_3_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_3_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_3_IN_CH;
    output_dims.n = TRANSPOSE_CONV_3_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_3_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_3_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_3_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_3_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_3_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_3_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_3_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_3_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_3_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_3_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_3_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_3_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_3_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_3_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_3_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_3_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = transpose_conv_params.input_offset;
    arm_convolve_weight_sum(
        weights_sum_ctx.buf, kernel_data, &input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    arm_cmsis_nn_status result = arm_transpose_conv_wrapper_s8(&ctx,
                                                               &weights_sum_ctx,
                                                               &reverse_conv_ctx,
                                                               &transpose_conv_params,
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
    if (reverse_conv_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void transpose_conv_4_arm_transpose_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[TRANSPOSE_CONV_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_4_biases;
    const int8_t *kernel_data = transpose_conv_4_weights;
    const int8_t *input_data = transpose_conv_4_input;
    const int8_t *output_ref = transpose_conv_4_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_4_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_4_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_4_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_4_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_4_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_4_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_4_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_4_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_4_IN_CH;
    output_dims.n = TRANSPOSE_CONV_4_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_4_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_4_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_4_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_4_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_4_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_4_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_4_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_4_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_4_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_4_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_4_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_4_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_4_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_4_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_4_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_4_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = transpose_conv_params.input_offset;
    arm_convolve_weight_sum(
        weights_sum_ctx.buf, kernel_data, &input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    arm_cmsis_nn_status result = arm_transpose_conv_wrapper_s8(&ctx,
                                                               &weights_sum_ctx,
                                                               &reverse_conv_ctx,
                                                               &transpose_conv_params,
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
    if (reverse_conv_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

/* Regression pin for ns-cmsis-nn#258 / upstream CMSIS-NN issue #230
 * (fixed upstream by commit 4a8f5b5d): row-misalignment in
 * arm_transpose_conv_s8's rolling-buffer flush schedule when
 * pad_h % stride_h != 0. transpose_conv_1..4 above all happen to use
 * pad % stride == 0 pairs and cannot catch this. TestData/transpose_conv_5
 * is hand-authored (see its config_data.h banner) with pad_h=1, stride_h=2
 * -- mirroring upstream issue #230's own repro shape -- against an
 * independent int32-accumulate + per-channel-requantize reference.
 */
void transpose_conv_5_arm_transpose_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[TRANSPOSE_CONV_5_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_5_biases;
    const int8_t *kernel_data = transpose_conv_5_weights;
    const int8_t *input_data = transpose_conv_5_input;
    const int8_t *output_ref = transpose_conv_5_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_5_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_5_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_5_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_5_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_5_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_5_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_5_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_5_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_5_IN_CH;
    output_dims.n = TRANSPOSE_CONV_5_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_5_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_5_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_5_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_5_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_5_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_5_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_5_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_5_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_5_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_5_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_5_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_5_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_5_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_5_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_5_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_5_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_5_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = transpose_conv_params.input_offset;
    arm_convolve_weight_sum(
        weights_sum_ctx.buf, kernel_data, &input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    arm_cmsis_nn_status result = arm_transpose_conv_wrapper_s8(&ctx,
                                                               &weights_sum_ctx,
                                                               &reverse_conv_ctx,
                                                               &transpose_conv_params,
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
    if (reverse_conv_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

/*
 * Regression pin for issue #261 defect 1: filter_h < stride_h under TFLite VALID padding.
 * Pre-fix the main-loop flush wrote input_h * stride_h rows per batch instead of output_h,
 * running past output_data and displacing every row of the second batch.
 */
void transpose_conv_6_arm_transpose_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[TRANSPOSE_CONV_6_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_6_biases;
    const int8_t *kernel_data = transpose_conv_6_weights;
    const int8_t *input_data = transpose_conv_6_input;
    const int8_t *output_ref = transpose_conv_6_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_6_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_6_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_6_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_6_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_6_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_6_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_6_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_6_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_6_IN_CH;
    output_dims.n = TRANSPOSE_CONV_6_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_6_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_6_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_6_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_6_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_6_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_6_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_6_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_6_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_6_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_6_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_6_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_6_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_6_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_6_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_6_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_6_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_6_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = transpose_conv_params.input_offset;
    arm_convolve_weight_sum(
        weights_sum_ctx.buf, kernel_data, &input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    arm_cmsis_nn_status result = arm_transpose_conv_wrapper_s8(&ctx,
                                                               &weights_sum_ctx,
                                                               &reverse_conv_ctx,
                                                               &transpose_conv_params,
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
    if (reverse_conv_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

/*
 * Regression pin for issue #261 defect 2: pad_h > input_h * stride_h. Pre-fix the leftover-row
 * loop emitted rows from the wrong rolling-buffer slot, so the output was shifted by
 * pad_h - input_h * stride_h rows.
 */
void transpose_conv_7_arm_transpose_conv_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[TRANSPOSE_CONV_7_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_7_biases;
    const int8_t *kernel_data = transpose_conv_7_weights;
    const int8_t *input_data = transpose_conv_7_input;
    const int8_t *output_ref = transpose_conv_7_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_7_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_7_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_7_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_7_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_7_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_7_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_7_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_7_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_7_IN_CH;
    output_dims.n = TRANSPOSE_CONV_7_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_7_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_7_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_7_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_7_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_7_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_7_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_7_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_7_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_7_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_7_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_7_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_7_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_7_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_7_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_7_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_7_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_7_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = transpose_conv_params.input_offset;
    arm_convolve_weight_sum(
        weights_sum_ctx.buf, kernel_data, &input_dims, &filter_dims, &output_dims, lhs_offset, bias_data);
    arm_cmsis_nn_status result = arm_transpose_conv_wrapper_s8(&ctx,
                                                               &weights_sum_ctx,
                                                               &reverse_conv_ctx,
                                                               &transpose_conv_params,
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
    if (reverse_conv_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

/*
 * Regression pin for issue #261 defect 4: dilation is not implemented by this kernel, so it must
 * be rejected rather than silently producing non-dilated output with ARM_CMSIS_NN_SUCCESS.
 */
void transpose_conv_dilation_rejected_arm_transpose_conv_s8(void)
{
    int8_t output[TRANSPOSE_CONV_7_DST_SIZE] = {0};
    int32_t scratch[512];

    cmsis_nn_context ctx = {scratch, (int32_t)sizeof(scratch)};
    cmsis_nn_context reverse_conv_ctx = {scratch, (int32_t)sizeof(scratch)};
    cmsis_nn_context weights_sum_ctx = {scratch, (int32_t)sizeof(scratch)};
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    input_dims.n = TRANSPOSE_CONV_7_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_7_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_7_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_7_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_7_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_7_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_7_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_7_IN_CH;
    output_dims.n = TRANSPOSE_CONV_7_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_7_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_7_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_7_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_7_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_7_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_7_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_7_PAD_Y_WITH_OFFSET;
    transpose_conv_params.stride.w = TRANSPOSE_CONV_7_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_7_STRIDE_Y;
    transpose_conv_params.input_offset = TRANSPOSE_CONV_7_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_7_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_7_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_7_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_7_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_7_output_shift;

    const cmsis_nn_tile bad_dilations[] = {{2, 1}, {1, 2}, {2, 2}};
    for (size_t i = 0; i < sizeof(bad_dilations) / sizeof(bad_dilations[0]); i++)
    {
        transpose_conv_params.dilation = bad_dilations[i];
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_transpose_conv_s8(&ctx,
                                                &reverse_conv_ctx,
                                                &transpose_conv_params,
                                                &quant_params,
                                                &input_dims,
                                                transpose_conv_7_input,
                                                &filter_dims,
                                                transpose_conv_7_weights,
                                                &bias_dims,
                                                transpose_conv_7_biases,
                                                &output_dims,
                                                output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_transpose_conv_wrapper_s8(&ctx,
                                                        &weights_sum_ctx,
                                                        &reverse_conv_ctx,
                                                        &transpose_conv_params,
                                                        &quant_params,
                                                        &input_dims,
                                                        transpose_conv_7_input,
                                                        &filter_dims,
                                                        transpose_conv_7_weights,
                                                        &bias_dims,
                                                        transpose_conv_7_biases,
                                                        &output_dims,
                                                        output));
    }

    transpose_conv_params.dilation.w = 1;
    transpose_conv_params.dilation.h = 1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_transpose_conv_s8(&ctx,
                                            &reverse_conv_ctx,
                                            &transpose_conv_params,
                                            &quant_params,
                                            &input_dims,
                                            transpose_conv_7_input,
                                            &filter_dims,
                                            transpose_conv_7_weights,
                                            &bias_dims,
                                            transpose_conv_7_biases,
                                            &output_dims,
                                            output));
}

/*
 * Regression pin for issue #261 defect 3: arm_transpose_conv_s8_get_buffer_size() is documented as
 * the ctx size for arm_transpose_conv_s8(), which callers may invoke directly, so it must never
 * return less than the rolling row buffer that kernel indexes -- not even for the in_ch > 16,
 * stride <= 2 shapes the wrapper would route to the reverse convolution instead.
 */
void transpose_conv_buffer_size_covers_direct_call_arm_transpose_conv_s8(void)
{
    /* The shape reported in issue #261: 80 bytes returned, 96 bytes indexed. */
    const int32_t in_ch = 17, out_ch = 1, input_w = 6, filter_w = 1, filter_h = 1, stride = 2;

    cmsis_nn_transpose_conv_params transpose_conv_params;
    memset(&transpose_conv_params, 0, sizeof(transpose_conv_params));
    transpose_conv_params.stride.w = stride;
    transpose_conv_params.stride.h = stride;
    transpose_conv_params.dilation.w = 1;
    transpose_conv_params.dilation.h = 1;

    cmsis_nn_dims input_dims = {1, 1, input_w, in_ch};
    cmsis_nn_dims filter_dims = {out_ch, filter_h, filter_w, in_ch};
    cmsis_nn_dims output_dims = {1, filter_h, (input_w - 1) * stride + filter_w, out_ch};

    const int32_t rolling_buffer_bytes =
        MAX(filter_h, stride) * ((input_w - 1) * stride + MAX(filter_w, stride)) * out_ch * (int32_t)sizeof(int32_t);

    TEST_ASSERT_TRUE(arm_transpose_conv_s8_get_buffer_size(
                         &transpose_conv_params, &input_dims, &filter_dims, &output_dims) >= rolling_buffer_bytes);
    TEST_ASSERT_TRUE(arm_transpose_conv_s8_get_buffer_size_mve(
                         &transpose_conv_params, &input_dims, &filter_dims, &output_dims) >= rolling_buffer_bytes);
}
