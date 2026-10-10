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
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unity.h>

#include "../TestData/dw_int16xint8_fast/test_data.h"
#include "../TestData/dw_int16xint8_fast_multiple_batches_uneven_buffers/test_data.h"
#include "../TestData/dw_int16xint8_fast_multiple_batches_uneven_buffers_null_bias/test_data.h"
#include "../TestData/dw_int16xint8_fast_null_bias/test_data.h"
#include "../TestData/dw_int16xint8_fast_spill/test_data.h"
#include "../TestData/dw_int16xint8_fast_spill_null_bias/test_data.h"
#include "../TestData/dw_int16xint8_fast_stride/test_data.h"
#include "../TestData/dw_int16xint8_fast_stride_null_bias/test_data.h"
#include "../TestData/dw_int16xint8_fast_test_bias/test_data.h"
#include "../Utils/mpu_guard.h"
#include "../Utils/utils.h"
#include "../Utils/validate.h"

void dw_int16xint8_fast_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = get_bias_s64_address(dw_int16xint8_fast_biases, DW_INT16XINT8_FAST_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_input;
    const int8_t *kernel_data = dw_int16xint8_fast_weights;
    const int16_t *output_ref = dw_int16xint8_fast_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_spill_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_SPILL_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = get_bias_s64_address(dw_int16xint8_fast_spill_biases, DW_INT16XINT8_FAST_SPILL_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_spill_input;
    const int8_t *kernel_data = dw_int16xint8_fast_spill_weights;
    const int16_t *output_ref = dw_int16xint8_fast_spill_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_SPILL_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_SPILL_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_SPILL_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_SPILL_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_SPILL_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_SPILL_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_SPILL_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_SPILL_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_SPILL_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_SPILL_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_SPILL_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_SPILL_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_SPILL_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_SPILL_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_SPILL_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_SPILL_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_SPILL_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_SPILL_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_SPILL_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_SPILL_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_SPILL_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_spill_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_spill_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_stride_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_STRIDE_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = get_bias_s64_address(dw_int16xint8_fast_stride_biases, DW_INT16XINT8_FAST_STRIDE_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_stride_input;
    const int8_t *kernel_data = dw_int16xint8_fast_stride_weights;
    const int16_t *output_ref = dw_int16xint8_fast_stride_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_STRIDE_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_STRIDE_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_STRIDE_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_STRIDE_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_STRIDE_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_STRIDE_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_STRIDE_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_STRIDE_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_STRIDE_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_STRIDE_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_STRIDE_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_STRIDE_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_STRIDE_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_STRIDE_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_STRIDE_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_STRIDE_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_STRIDE_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_STRIDE_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_STRIDE_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_STRIDE_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_STRIDE_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_stride_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_stride_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_null_bias_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_NULL_BIAS_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data =
        get_bias_s64_address(dw_int16xint8_fast_null_bias_biases, DW_INT16XINT8_FAST_NULL_BIAS_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_null_bias_input;
    const int8_t *kernel_data = dw_int16xint8_fast_null_bias_weights;
    const int16_t *output_ref = dw_int16xint8_fast_null_bias_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_NULL_BIAS_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_NULL_BIAS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_NULL_BIAS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_NULL_BIAS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_NULL_BIAS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_NULL_BIAS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_NULL_BIAS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_NULL_BIAS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_NULL_BIAS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_NULL_BIAS_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_NULL_BIAS_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_NULL_BIAS_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_NULL_BIAS_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_NULL_BIAS_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_NULL_BIAS_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_NULL_BIAS_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_NULL_BIAS_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_NULL_BIAS_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_NULL_BIAS_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_NULL_BIAS_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_NULL_BIAS_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_null_bias_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_null_bias_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_stride_null_bias_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data =
        get_bias_s64_address(dw_int16xint8_fast_stride_null_bias_biases, DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_stride_null_bias_input;
    const int8_t *kernel_data = dw_int16xint8_fast_stride_null_bias_weights;
    const int16_t *output_ref = dw_int16xint8_fast_stride_null_bias_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_STRIDE_NULL_BIAS_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_stride_null_bias_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_stride_null_bias_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_spill_null_bias_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_SPILL_NULL_BIAS_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data =
        get_bias_s64_address(dw_int16xint8_fast_spill_null_bias_biases, DW_INT16XINT8_FAST_SPILL_NULL_BIAS_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_spill_null_bias_input;
    const int8_t *kernel_data = dw_int16xint8_fast_spill_null_bias_weights;
    const int16_t *output_ref = dw_int16xint8_fast_spill_null_bias_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_SPILL_NULL_BIAS_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_spill_null_bias_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_spill_null_bias_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_test_bias_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_TEST_BIAS_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data =
        get_bias_s64_address(dw_int16xint8_fast_test_bias_biases, DW_INT16XINT8_FAST_TEST_BIAS_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_test_bias_input;
    const int8_t *kernel_data = dw_int16xint8_fast_test_bias_weights;
    const int16_t *output_ref = dw_int16xint8_fast_test_bias_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_TEST_BIAS_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_TEST_BIAS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_TEST_BIAS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_TEST_BIAS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_TEST_BIAS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_TEST_BIAS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_TEST_BIAS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_TEST_BIAS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_TEST_BIAS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_TEST_BIAS_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_TEST_BIAS_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_TEST_BIAS_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_TEST_BIAS_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_TEST_BIAS_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_TEST_BIAS_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_TEST_BIAS_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_TEST_BIAS_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_TEST_BIAS_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_TEST_BIAS_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_TEST_BIAS_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_TEST_BIAS_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_test_bias_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_test_bias_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_multiple_batches_uneven_buffers_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = get_bias_s64_address(dw_int16xint8_fast_multiple_batches_uneven_buffers_biases,
                                                    DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_multiple_batches_uneven_buffers_input;
    const int8_t *kernel_data = dw_int16xint8_fast_multiple_batches_uneven_buffers_weights;
    const int16_t *output_ref = dw_int16xint8_fast_multiple_batches_uneven_buffers_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_multiple_batches_uneven_buffers_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_multiple_batches_uneven_buffers_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_multiple_batches_uneven_buffers_null_bias_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data =
        get_bias_s64_address(dw_int16xint8_fast_multiple_batches_uneven_buffers_null_bias_biases,
                             DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_CH);
    const int16_t *input_data = dw_int16xint8_fast_multiple_batches_uneven_buffers_null_bias_input;
    const int8_t *kernel_data = dw_int16xint8_fast_multiple_batches_uneven_buffers_null_bias_weights;
    const int16_t *output_ref = dw_int16xint8_fast_multiple_batches_uneven_buffers_null_bias_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_multiple_batches_uneven_buffers_null_bias_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_multiple_batches_uneven_buffers_null_bias_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    TEST_ASSERT_EQUAL(
        buf_size,
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));

    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void buffer_size_arm_depthwise_conv_fast_s16(void)
{
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_CH;

    conv_params.padding.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_PAD_X;
    conv_params.padding.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_PAD_Y;
    conv_params.stride.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_STRIDE_X;
    conv_params.stride.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_STRIDE_Y;
    conv_params.dilation.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DILATION_X;
    conv_params.dilation.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DILATION_Y;
    conv_params.ch_mult = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_CH_MULT;
    conv_params.input_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_OFFSET;
    conv_params.output_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_OFFSET;
    conv_params.activation.min = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_ACTIVATION_MAX;

    const int32_t buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, buf_size);
}

void buffer_size_mve_arm_depthwise_conv_fast_s16(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_CH;

    conv_params.padding.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_PAD_X;
    conv_params.padding.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_PAD_Y;
    conv_params.stride.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_STRIDE_X;
    conv_params.stride.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_STRIDE_Y;
    conv_params.dilation.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DILATION_X;
    conv_params.dilation.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DILATION_Y;
    conv_params.ch_mult = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_CH_MULT;
    conv_params.input_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_OFFSET;
    conv_params.output_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_OFFSET;
    conv_params.activation.min = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_depthwise_conv_fast_s16(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_CH;

    conv_params.padding.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_PAD_X;
    conv_params.padding.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_PAD_Y;
    conv_params.stride.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_STRIDE_X;
    conv_params.stride.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_STRIDE_Y;
    conv_params.dilation.w = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DILATION_X;
    conv_params.dilation.h = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_DILATION_Y;

    conv_params.ch_mult = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_CH_MULT;

    conv_params.input_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_INPUT_OFFSET;
    conv_params.output_offset = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUTPUT_OFFSET;
    conv_params.activation.min = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DW_INT16XINT8_FAST_MULTIPLE_BATCHES_UNEVEN_BUFFERS_NULL_BIAS_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}

/* Dilated 1D 16x8 layers (heartkit and sleepkit TCN shapes plus tails) run through arm_depthwise_conv_wrapper_s16()
   and must match arm_depthwise_conv_s16(). */
#define DIL_MAX_LEN 256
#define DIL_MAX_CH 64
static int16_t dil_input[DIL_MAX_LEN * DIL_MAX_CH];
static int8_t dil_filter[9 * DIL_MAX_CH];
static int16_t dil_output[DIL_MAX_LEN * DIL_MAX_CH];
static int16_t dil_reference[DIL_MAX_LEN * DIL_MAX_CH];
static int64_t dil_bias[DIL_MAX_CH];
static int32_t dil_mult[DIL_MAX_CH];
static int32_t dil_shift[DIL_MAX_CH];
static int16_t dil_scratch[8192];

static void dilated_1d_s16_case(int32_t batches, int32_t len, int32_t k, int32_t ch, int32_t dil, int32_t pad)
{
    const int32_t out_len = len + 2 * pad - (k - 1) * dil;
    TEST_ASSERT_TRUE(out_len > 0 && batches * len * ch <= DIL_MAX_LEN * DIL_MAX_CH &&
                     batches * out_len * ch <= DIL_MAX_LEN * DIL_MAX_CH);
    uint32_t seed = (uint32_t)(len * 131 + k * 17 + ch * 7 + dil);
    for (int32_t i = 0; i < batches * len * ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        dil_input[i] = (int16_t)(seed >> 16);
    }
    for (int32_t i = 0; i < k * ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        dil_filter[i] = (int8_t)(seed >> 24);
    }
    for (int32_t i = 0; i < ch; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        /* Small enough that most outputs stay inside the int16 range, so the comparison checks the taps. */
        dil_bias[i] = (int64_t)(int32_t)seed / 4096;
        dil_mult[i] = 0x40000000 + (i % 64) * 0x800000;
        dil_shift[i] = -7 - (i % 3);
    }
    const cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                            .output_offset = 0,
                                            .ch_mult = 1,
                                            .stride = {1, 1},
                                            .padding = {pad, 0},
                                            .dilation = {dil, 1},
                                            .activation = {-32768, 32767}};
    const cmsis_nn_per_channel_quant_params quant = {dil_mult, dil_shift};
    const cmsis_nn_dims input_dims = {batches, 1, len, ch}, filter_dims = {1, 1, k, ch}, bias_dims = {1, 1, 1, ch},
                        output_dims = {batches, 1, out_len, ch};
    const cmsis_nn_context none = {NULL, 0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_s16(&none,
                                             &params,
                                             &quant,
                                             &input_dims,
                                             dil_input,
                                             &filter_dims,
                                             dil_filter,
                                             &bias_dims,
                                             dil_bias,
                                             &output_dims,
                                             dil_reference));
    int32_t unclamped = 0;
    for (int32_t i = 0; i < batches * out_len * ch; i++)
    {
        unclamped += (dil_reference[i] != INT16_MAX && dil_reference[i] != INT16_MIN);
    }
    TEST_ASSERT_TRUE(2 * unclamped >= batches * out_len * ch);
    const int32_t size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims);
    TEST_ASSERT_EQUAL(arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims), size);
    TEST_ASSERT_TRUE(size >= 0 && size <= (int32_t)sizeof(dil_scratch));
    const cmsis_nn_context ctx = {size > 0 ? dil_scratch : NULL, size};
    memset(dil_output, 0x5A, sizeof(dil_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_s16(&ctx,
                                                     &params,
                                                     &quant,
                                                     &input_dims,
                                                     dil_input,
                                                     &filter_dims,
                                                     dil_filter,
                                                     &bias_dims,
                                                     dil_bias,
                                                     &output_dims,
                                                     dil_output));
    TEST_ASSERT_EQUAL_INT16_ARRAY(dil_reference, dil_output, batches * out_len * ch);
}

void dilated_1d_arm_depthwise_conv_fast_s16(void)
{
    const int32_t cases[][4] = {
        /* input_len, filter_len, channels, dilation */
        {256, 7, 16, 2},
        {256, 7, 24, 4},
        {256, 7, 32, 8},
        {240, 5, 24, 2},
        {240, 5, 32, 4},
        {240, 5, 48, 8},
        {240, 7, 48, 16},
        {240, 7, 64, 16},
        {37, 3, 5, 3},
        {64, 2, 33, 16},
        {9, 7, 3, 2},
        {1, 3, 17, 4},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const int32_t eff = (cases[i][1] - 1) * cases[i][3];
        dilated_1d_s16_case(1, cases[i][0], cases[i][1], cases[i][2], cases[i][3], eff / 2);
        if (cases[i][0] > eff)
        {
            dilated_1d_s16_case(1, cases[i][0], cases[i][1], cases[i][2], cases[i][3], 0);
        }
    }
    /* The fast kernel loops over batches itself. */
    dilated_1d_s16_case(3, 37, 5, 19, 3, 6);
}

/* The dilated 1D route runs arm_depthwise_conv_fast_s16(): with no scratch it is rejected on builds that need one,
   whereas a 2D-dilated layer and a vertically dilated 1D layer stay on arm_depthwise_conv_s16(), which needs none. */
void dilated_1d_route_arm_depthwise_conv_fast_s16(void)
{
    const int32_t len = 64, k = 5, ch = 16, dil = 4;
    for (int32_t i = 0; i < 2 * len * ch; i++)
    {
        dil_input[i] = (int16_t)((i * 97) % 2000 - 1000);
    }
    for (int32_t i = 0; i < k * ch; i++)
    {
        dil_filter[i] = (int8_t)((i * 5 + 1) % 120 - 60);
    }
    for (int32_t i = 0; i < ch; i++)
    {
        dil_bias[i] = i * 100;
        dil_mult[i] = 0x40000000;
        dil_shift[i] = -5;
    }
    cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                      .output_offset = 0,
                                      .ch_mult = 1,
                                      .stride = {1, 1},
                                      .padding = {((k - 1) * dil) / 2, 0},
                                      .dilation = {dil, 1},
                                      .activation = {-32768, 32767}};
    const cmsis_nn_per_channel_quant_params quant = {dil_mult, dil_shift};
    const cmsis_nn_dims input_dims = {1, 1, len, ch}, filter_dims = {1, 1, k, ch}, bias_dims = {1, 1, 1, ch},
                        output_dims = {1, 1, len, ch};
    const cmsis_nn_context none = {NULL, 0};
    const arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s16(&none,
                                                                      &params,
                                                                      &quant,
                                                                      &input_dims,
                                                                      dil_input,
                                                                      &filter_dims,
                                                                      dil_filter,
                                                                      &bias_dims,
                                                                      dil_bias,
                                                                      &output_dims,
                                                                      dil_output);
#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(arm_depthwise_conv_wrapper_s16_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims) >
                     0);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, status);
#else
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
#endif

    /* 2D and dilated in both dimensions, then 1D with vertical dilation: both stay on the reference route. */
    params.dilation.h = 2;
    const cmsis_nn_dims input_2d = {1, 2, len, ch}, output_2d = {1, 2, len, ch};
    TEST_ASSERT_EQUAL(0, arm_depthwise_conv_wrapper_s16_get_buffer_size(&params, &input_2d, &filter_dims, &output_2d));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_s16(&none,
                                                     &params,
                                                     &quant,
                                                     &input_2d,
                                                     dil_input,
                                                     &filter_dims,
                                                     dil_filter,
                                                     &bias_dims,
                                                     dil_bias,
                                                     &output_2d,
                                                     dil_output));
    TEST_ASSERT_EQUAL(0,
                      arm_depthwise_conv_wrapper_s16_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_s16(&none,
                                             &params,
                                             &quant,
                                             &input_dims,
                                             dil_input,
                                             &filter_dims,
                                             dil_filter,
                                             &bias_dims,
                                             dil_bias,
                                             &output_dims,
                                             dil_reference));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_wrapper_s16(&none,
                                                     &params,
                                                     &quant,
                                                     &input_dims,
                                                     dil_input,
                                                     &filter_dims,
                                                     dil_filter,
                                                     &bias_dims,
                                                     dil_bias,
                                                     &output_dims,
                                                     dil_output));
    TEST_ASSERT_EQUAL_INT16_ARRAY(dil_reference, dil_output, len * ch);
}

/* arm_depthwise_conv_fast_s16() steps only the horizontal tap index by dilation, so it rejects vertical dilation
   and a non-positive horizontal dilation instead of computing a wrong result. */
void dilation_arg_check_arm_depthwise_conv_fast_s16(void)
{
    const cmsis_nn_dims input_dims = {1, 4, 8, 4}, filter_dims = {1, 3, 3, 4}, bias_dims = {1, 1, 1, 4},
                        output_dims = {1, 4, 8, 4};
    const cmsis_nn_per_channel_quant_params quant = {dil_mult, dil_shift};
    const cmsis_nn_context ctx = {dil_scratch, (int32_t)sizeof(dil_scratch)};
    const cmsis_nn_tile bad_dilations[] = {{2, 2}, {1, 2}, {0, 1}, {-1, 1}};
    for (size_t i = 0; i < sizeof(bad_dilations) / sizeof(bad_dilations[0]); i++)
    {
        const cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                                .output_offset = 0,
                                                .ch_mult = 1,
                                                .stride = {1, 1},
                                                .padding = {1, 1},
                                                .dilation = bad_dilations[i],
                                                .activation = {-32768, 32767}};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_depthwise_conv_fast_s16(&ctx,
                                                      &params,
                                                      &quant,
                                                      &input_dims,
                                                      dil_input,
                                                      &filter_dims,
                                                      dil_filter,
                                                      &bias_dims,
                                                      dil_bias,
                                                      &output_dims,
                                                      dil_output));
    }
}

/* With a channel count that is not a multiple of 4, the MVE kernel must stay inside every operand, both in the
   four-pixel blocks and in the leftover pixels. Each operand in turn is placed against an MPU gap. */
void operand_bounds_arm_depthwise_conv_fast_s16(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    const int32_t k = 3;
    enum
    {
        op_input,
        op_filter,
        op_bias,
        op_mult,
        op_shift,
        op_scratch,
        op_output,
        op_end
    };
    const int32_t channels[] = {1, 2, 3, 5, 7, 17};
    const int32_t lengths[] = {1, 3, 4, 5};
    for (size_t i_ch = 0; i_ch < sizeof(channels) / sizeof(channels[0]); i_ch++)
    {
        for (size_t i_len = 0; i_len < sizeof(lengths) / sizeof(lengths[0]); i_len++)
        {
            const int32_t ch = channels[i_ch], len = lengths[i_len];
            uint32_t seed = (uint32_t)(len * 131 + ch * 7);
            for (int32_t i = 0; i < len * ch; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                dil_input[i] = (int16_t)(seed >> 16);
            }
            for (int32_t i = 0; i < k * ch; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                dil_filter[i] = (int8_t)(seed >> 24);
            }
            for (int32_t i = 0; i < ch; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                dil_bias[i] = (int64_t)(int32_t)seed / 4096;
                dil_mult[i] = 0x40000000 + i * 0x800000;
                dil_shift[i] = -7 - (i % 3);
            }
            const cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                                    .output_offset = 0,
                                                    .ch_mult = 1,
                                                    .stride = {1, 1},
                                                    .padding = {1, 0},
                                                    .dilation = {1, 1},
                                                    .activation = {-32768, 32767}};
            const cmsis_nn_dims input_dims = {1, 1, len, ch}, filter_dims = {1, 1, k, ch}, bias_dims = {1, 1, 1, ch},
                                output_dims = {1, 1, len, ch};
            const cmsis_nn_per_channel_quant_params ref_quant = {dil_mult, dil_shift};
            const cmsis_nn_context none = {NULL, 0};
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                              arm_depthwise_conv_s16(&none,
                                                     &params,
                                                     &ref_quant,
                                                     &input_dims,
                                                     dil_input,
                                                     &filter_dims,
                                                     dil_filter,
                                                     &bias_dims,
                                                     dil_bias,
                                                     &output_dims,
                                                     dil_reference));
            const int32_t scratch_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
            /* The kernel uses only the im2col rows for four pixels, not the trailing slack in the reported size. */
            const int32_t scratch_used = 4 * k * ch * (int32_t)sizeof(int16_t);
            TEST_ASSERT_TRUE(scratch_used <= scratch_size && scratch_used <= GUARD_OFFSET);

            for (int op = op_input; op < op_end; op++)
            {
                const int16_t *input = op == op_input ? guard_place(dil_input, len * ch * sizeof(int16_t)) : dil_input;
                const int8_t *filter = op == op_filter ? guard_place(dil_filter, k * ch) : dil_filter;
                const int64_t *bias = op == op_bias ? guard_place(dil_bias, ch * sizeof(int64_t)) : dil_bias;
                int32_t *mult = op == op_mult ? guard_place(dil_mult, ch * sizeof(int32_t)) : dil_mult;
                int32_t *shift = op == op_shift ? guard_place(dil_shift, ch * sizeof(int32_t)) : dil_shift;
                int16_t *output = op == op_output ? guard_end(len * ch * sizeof(int16_t)) : dil_output;
                void *scratch = op == op_scratch ? guard_end(scratch_used) : dil_scratch;
                const cmsis_nn_per_channel_quant_params quant = {mult, shift};
                const cmsis_nn_context ctx = {scratch, scratch_size};
                memset(output, 0x5A, len * ch * sizeof(int16_t));

                guard_gap_enable();
                const arm_cmsis_nn_status status = arm_depthwise_conv_fast_s16(&ctx,
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
                guard_gap_disable();

                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
                TEST_ASSERT_EQUAL_INT16_ARRAY(dil_reference, output, len * ch);
            }
        }
    }
#endif
}

/*
 * The DSP path keeps the first tap index of an output as int16_t (#727), and builds without DSP run
 * arm_depthwise_conv_s16(), which does too: there an input 32,768 wide, or a padding of 32,768, is an argument error
 * with the output untouched. The MVE path indexes in int32_t and runs it. An input 32,767 wide runs on every build.
 */
#define WIDE_DW_FAST_MAX_W 32768
static int16_t wide_fast_input[WIDE_DW_FAST_MAX_W];
static int16_t wide_fast_output[WIDE_DW_FAST_MAX_W];

static void wide_dw_fast_s16_case(int32_t width, int32_t pad, arm_cmsis_nn_status expected)
{
    static const int8_t weight[1] = {3};
    static const int64_t bias[1] = {-7};
    static int32_t multiplier[1] = {1 << 30};
    static int32_t shift[1] = {0};
    const cmsis_nn_dw_conv_params params = {0, 0, 1, {1, 1}, {pad, 0}, {1, 1}, {-32768, 32767}};
    const cmsis_nn_per_channel_quant_params quant_params = {multiplier, shift};
    const cmsis_nn_dims input_dims = {1, 1, width, 1};
    const cmsis_nn_dims filter_dims = {1, 1, 1, 1};
    const cmsis_nn_dims bias_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, width, 1};
    const int32_t buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    TEST_ASSERT_TRUE(buf_size >= 0 && buf_size <= (int32_t)sizeof(dil_scratch));
    const cmsis_nn_context ctx = {dil_scratch, buf_size};

    for (int32_t x = 0; x < width; x++)
    {
        wide_fast_input[x] = (int16_t)(x % 2001 - 1000);
    }
    memset(wide_fast_output, 0x55, sizeof(wide_fast_output));
    TEST_ASSERT_EQUAL(expected,
                      arm_depthwise_conv_fast_s16(&ctx,
                                                  &params,
                                                  &quant_params,
                                                  &input_dims,
                                                  wide_fast_input,
                                                  &filter_dims,
                                                  weight,
                                                  &bias_dims,
                                                  bias,
                                                  &output_dims,
                                                  wide_fast_output));
    if (expected != ARM_CMSIS_NN_SUCCESS)
    {
        TEST_ASSERT_EQUAL_INT16(0x5555, wide_fast_output[0]);
        return;
    }
    const int32_t reduced_multiplier = REDUCE_MULTIPLIER(multiplier[0]);
    for (int32_t x = 0; x < width; x++)
    {
        /* With padding, output x reads input x - pad, or the zero padding */
        const int32_t in = x >= pad ? wide_fast_input[x - pad] : 0;
        const int32_t expected_x = arm_nn_requantize_s64((int64_t)in * 3 - 7, reduced_multiplier, 0);
        TEST_ASSERT_EQUAL_INT16(expected_x, wide_fast_output[x]);
    }
}

void tap_index_arm_depthwise_conv_fast_s16(void)
{
#if defined(ARM_MATH_MVEI)
    const arm_cmsis_nn_status wide = ARM_CMSIS_NN_SUCCESS;
#else
    const arm_cmsis_nn_status wide = ARM_CMSIS_NN_ARG_ERROR;
#endif
    wide_dw_fast_s16_case(INT16_MAX, 0, ARM_CMSIS_NN_SUCCESS);
    wide_dw_fast_s16_case(INT16_MAX + 1, 0, wide);
    wide_dw_fast_s16_case(INT16_MAX, INT16_MAX + 1, wide);
}
