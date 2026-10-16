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
#include <string.h>
#include <unity.h>

#include "../TestData/dw_int16xint8/test_data.h"
#include "../TestData/dw_int16xint8_dilation/test_data.h"
#include "../TestData/dw_int16xint8_mult4/test_data.h"
#include "../Utils/validate.h"

void dw_int16xint8_arm_depthwise_conv_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = dw_int16xint8_biases;
    const int16_t *input_data = dw_int16xint8_input;
    const int8_t *kernel_data = dw_int16xint8_weights;
    const int16_t *output_ref = dw_int16xint8_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_DST_SIZE;

    input_dims.n = DW_INT16XINT8_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_INPUT_W;
    input_dims.h = DW_INT16XINT8_INPUT_H;
    input_dims.c = DW_INT16XINT8_IN_CH;
    filter_dims.w = DW_INT16XINT8_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FILTER_Y;
    output_dims.w = DW_INT16XINT8_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_depthwise_conv_s16(&ctx,
                                                        &dw_conv_params,
                                                        &quant_params,
                                                        &input_dims,
                                                        input_data,
                                                        &filter_dims,
                                                        dw_int16xint8_weights,
                                                        &bias_dims,
                                                        bias_data,
                                                        &output_dims,
                                                        output);
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    int buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(buf_size, 0);

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

void dw_int16xint8_dilation_arm_depthwise_conv_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = dw_int16xint8_dilation_biases;
    const int16_t *input_data = dw_int16xint8_dilation_input;
    const int8_t *kernel_data = dw_int16xint8_dilation_weights;
    const int16_t *output_ref = dw_int16xint8_dilation_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_DILATION_DST_SIZE;

    input_dims.n = DW_INT16XINT8_DILATION_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_DILATION_INPUT_W;
    input_dims.h = DW_INT16XINT8_DILATION_INPUT_H;
    input_dims.c = DW_INT16XINT8_DILATION_IN_CH;
    filter_dims.w = DW_INT16XINT8_DILATION_FILTER_X;
    filter_dims.h = DW_INT16XINT8_DILATION_FILTER_Y;
    output_dims.w = DW_INT16XINT8_DILATION_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_DILATION_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_DILATION_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_DILATION_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_DILATION_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_DILATION_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_DILATION_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_DILATION_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_DILATION_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_DILATION_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_DILATION_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_DILATION_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_DILATION_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_dilation_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_dilation_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_depthwise_conv_s16(&ctx,
                                                        &dw_conv_params,
                                                        &quant_params,
                                                        &input_dims,
                                                        input_data,
                                                        &filter_dims,
                                                        dw_int16xint8_dilation_weights,
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
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    int buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(buf_size, 0);

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

void dw_int16xint8_mult4_arm_depthwise_conv_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_MULT4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = dw_int16xint8_mult4_biases;
    const int16_t *input_data = dw_int16xint8_mult4_input;
    const int8_t *kernel_data = dw_int16xint8_mult4_weights;
    const int16_t *output_ref = dw_int16xint8_mult4_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_MULT4_DST_SIZE;

    input_dims.n = DW_INT16XINT8_MULT4_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_MULT4_INPUT_W;
    input_dims.h = DW_INT16XINT8_MULT4_INPUT_H;
    input_dims.c = DW_INT16XINT8_MULT4_IN_CH;
    filter_dims.w = DW_INT16XINT8_MULT4_FILTER_X;
    filter_dims.h = DW_INT16XINT8_MULT4_FILTER_Y;
    output_dims.w = DW_INT16XINT8_MULT4_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_MULT4_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_MULT4_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_MULT4_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_MULT4_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_MULT4_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_MULT4_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_MULT4_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_MULT4_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_MULT4_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_MULT4_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_MULT4_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_MULT4_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_MULT4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_mult4_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_mult4_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_depthwise_conv_s16(&ctx,
                                                        &dw_conv_params,
                                                        &quant_params,
                                                        &input_dims,
                                                        input_data,
                                                        &filter_dims,
                                                        dw_int16xint8_mult4_weights,
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
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    int buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(buf_size, 0);

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

void arm_depthwise_conv_wrapper_s16_buffer(void)
{
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    cmsis_nn_dw_conv_params dw_conv_params;
    input_dims.n = DW_INT16XINT8_MULT4_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_MULT4_INPUT_W;
    input_dims.h = DW_INT16XINT8_MULT4_INPUT_H;
    input_dims.c = DW_INT16XINT8_MULT4_IN_CH;
    filter_dims.w = DW_INT16XINT8_MULT4_FILTER_X;
    filter_dims.h = DW_INT16XINT8_MULT4_FILTER_Y;

    output_dims.w = DW_INT16XINT8_MULT4_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_MULT4_OUTPUT_H;
    output_dims.c = input_dims.c;

    dw_conv_params.padding.w = DW_INT16XINT8_MULT4_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_MULT4_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_MULT4_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_MULT4_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_MULT4_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_MULT4_DILATION_Y;
    dw_conv_params.ch_mult = output_dims.c / input_dims.c;

    int32_t size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(size > 0);
#else
    TEST_ASSERT_TRUE(size == 0);
#endif
    input_dims.c = 513;
    output_dims.c = input_dims.c;
    dw_conv_params.ch_mult = output_dims.c / input_dims.c;
    size = arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

#if defined(ARM_MATH_DSP)
    TEST_ASSERT_TRUE(size > 0);
#else
    TEST_ASSERT_TRUE(size == 0);
#endif
}

void buffer_size_mve_arm_depthwise_conv_s16(void)
{
#if defined(ARM_MATH_MVEI)
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DW_INT16XINT8_MULT4_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_MULT4_INPUT_W;
    input_dims.h = DW_INT16XINT8_MULT4_INPUT_H;
    input_dims.c = DW_INT16XINT8_MULT4_IN_CH;
    filter_dims.w = DW_INT16XINT8_MULT4_FILTER_X;
    filter_dims.h = DW_INT16XINT8_MULT4_FILTER_Y;
    output_dims.w = DW_INT16XINT8_MULT4_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_MULT4_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_MULT4_OUT_CH;

    conv_params.padding.w = DW_INT16XINT8_MULT4_PAD_X;
    conv_params.padding.h = DW_INT16XINT8_MULT4_PAD_Y;
    conv_params.stride.w = DW_INT16XINT8_MULT4_STRIDE_X;
    conv_params.stride.h = DW_INT16XINT8_MULT4_STRIDE_Y;
    conv_params.dilation.w = DW_INT16XINT8_MULT4_DILATION_X;
    conv_params.dilation.h = DW_INT16XINT8_MULT4_DILATION_Y;
    conv_params.ch_mult = DW_INT16XINT8_MULT4_CH_MULT;
    conv_params.input_offset = DW_INT16XINT8_MULT4_INPUT_OFFSET;
    conv_params.output_offset = DW_INT16XINT8_MULT4_OUTPUT_OFFSET;
    conv_params.activation.min = DW_INT16XINT8_MULT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DW_INT16XINT8_MULT4_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t mve_wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size_mve(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, mve_wrapper_buf_size);
#endif
}

void buffer_size_dsp_arm_depthwise_conv_s16(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    cmsis_nn_dw_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = DW_INT16XINT8_MULT4_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_MULT4_INPUT_W;
    input_dims.h = DW_INT16XINT8_MULT4_INPUT_H;
    input_dims.c = DW_INT16XINT8_MULT4_IN_CH;
    filter_dims.w = DW_INT16XINT8_MULT4_FILTER_X;
    filter_dims.h = DW_INT16XINT8_MULT4_FILTER_Y;
    output_dims.w = DW_INT16XINT8_MULT4_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_MULT4_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_MULT4_OUT_CH;

    conv_params.padding.w = DW_INT16XINT8_MULT4_PAD_X;
    conv_params.padding.h = DW_INT16XINT8_MULT4_PAD_Y;
    conv_params.stride.w = DW_INT16XINT8_MULT4_STRIDE_X;
    conv_params.stride.h = DW_INT16XINT8_MULT4_STRIDE_Y;
    conv_params.dilation.w = DW_INT16XINT8_MULT4_DILATION_X;
    conv_params.dilation.h = DW_INT16XINT8_MULT4_DILATION_Y;

    conv_params.ch_mult = DW_INT16XINT8_MULT4_CH_MULT;

    conv_params.input_offset = DW_INT16XINT8_MULT4_INPUT_OFFSET;
    conv_params.output_offset = DW_INT16XINT8_MULT4_OUTPUT_OFFSET;
    conv_params.activation.min = DW_INT16XINT8_MULT4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = DW_INT16XINT8_MULT4_OUT_ACTIVATION_MAX;

    const int32_t wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    const int32_t dsp_wrapper_buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size_dsp(&conv_params, &input_dims, &filter_dims, &output_dims);

    TEST_ASSERT_EQUAL(wrapper_buf_size, dsp_wrapper_buf_size);
#endif
}

/* A 1 x W x 1 layer with a 1 x 1 kernel, so output pixel x is input pixel x scaled by the weight and requantized */
#define WIDE_DW_S16_MAX_W 32768
static const int32_t unit_multiplier = 1 << 30;
static int16_t wide_dw_input[WIDE_DW_S16_MAX_W];
static int16_t wide_dw_output[WIDE_DW_S16_MAX_W];

static arm_cmsis_nn_status wide_dw_s16(const cmsis_nn_dw_conv_params *dw_conv_params,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *filter_dims,
                                       const cmsis_nn_dims *output_dims)
{
    static const int8_t weight[1] = {3};
    static const int64_t bias[1] = {-7};
    static int32_t multiplier[1] = {1 << 30};
    static int32_t shift[1] = {0};
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_per_channel_quant_params quant_params = {multiplier, shift};
    const cmsis_nn_dims bias_dims = {1, 1, 1, 1};
    return arm_depthwise_conv_s16(&ctx,
                                  dw_conv_params,
                                  &quant_params,
                                  input_dims,
                                  wide_dw_input,
                                  filter_dims,
                                  weight,
                                  &bias_dims,
                                  bias,
                                  output_dims,
                                  wide_dw_output);
}

/*
 * arm_depthwise_conv_s16() forms tensor indices in int32_t (#707). Its generic kernel, used without MVE and for 512
 * or more taps, keeps every dimension, padding, stride and dilation as uint16_t and the first tap index of an output
 * as int16_t (#727); the MVE kernel holds each value up to INT32_MAX / 4 (#787). Each value a kernel cannot hold is an
 * argument error with the output untouched; a dilation of 65,535 on a 1x1 kernel and an input and output 32,767 wide
 * still run.
 */
void dims_arg_errors_arm_depthwise_conv_s16(void)
{
    const cmsis_nn_dw_conv_params unit_params = {0, 0, 1, {1, 1}, {0, 0}, {1, 1}, {-32768, 32767}};
    const cmsis_nn_dims unit_dims = {1, 1, 1, 1};
    for (int c = 0; c < 56; c++)
    {
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
        /* The MVE kernel indexes taps in int32_t and takes these 1x1 layers; case 34 tests the generic kernel's rule */
        if (c >= 15 && c <= 21)
        {
            continue;
        }
#endif
        cmsis_nn_dw_conv_params params = unit_params;
        cmsis_nn_dims input_dims = unit_dims;
        cmsis_nn_dims filter_dims = unit_dims;
        cmsis_nn_dims output_dims = unit_dims;
        int32_t *const wide[15] = {&input_dims.n,
                                   &input_dims.w,
                                   &input_dims.h,
                                   &input_dims.c,
                                   &params.ch_mult,
                                   &filter_dims.w,
                                   &filter_dims.h,
                                   &params.padding.w,
                                   &params.padding.h,
                                   &params.stride.w,
                                   &params.stride.h,
                                   &output_dims.w,
                                   &output_dims.h,
                                   &params.dilation.w,
                                   &params.dilation.h};
        if (c < 15)
        {
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
            *wide[c] = INT32_MAX / 4 + 1;
#else
            *wide[c] = UINT16_MAX + 1;
#endif
        }
        else if (c < 19)
        {
            /* Padding or stride of 32,768: the first tap index leaves int16_t */
            int32_t *const tap[4] = {&params.padding.w, &params.padding.h, &params.stride.w, &params.stride.h};
            *tap[c - 15] = INT16_MAX + 1;
        }
        else if (c == 19)
        {
            output_dims.w = INT16_MAX + 2; /* (32,769 - 1) * 1 - 0 */
        }
        else if (c == 20)
        {
            output_dims.h = INT16_MAX + 2;
        }
        else if (c == 21)
        {
            /* An output wider than the input supports: 10 pixels, stride 2, 32,771 outputs */
            input_dims.w = 10;
            params.stride.w = 2;
            output_dims.w = 32771;
        }
        else if (c == 22)
        {
            input_dims.c = UINT16_MAX; /* a filter of 65,535 * 32,769 channels */
            params.ch_mult = 32769;
        }
        else if (c == 23)
        {
            input_dims.w = UINT16_MAX; /* an input of 65,535 * 65,535 elements */
            input_dims.h = UINT16_MAX;
        }
        else if (c == 24)
        {
            output_dims.w = INT16_MAX; /* an output of 32,767 * 32,767 * 4 elements */
            output_dims.h = INT16_MAX;
            params.ch_mult = 4;
        }
        else if (c == 25)
        {
            filter_dims.w = UINT16_MAX; /* KW times dilation W past INT32_MAX / 2 */
            params.dilation.w = 16385;
        }
        else if (c == 26)
        {
            filter_dims.h = UINT16_MAX;
            params.dilation.h = 16385;
        }
        else if (c == 27)
        {
            params.padding.h = -1;
        }
        else if (c == 28)
        {
            input_dims.c = -1;
        }
        else if (c == 29)
        {
            params.dilation.w = 0; /* would read before the input */
        }
        else if (c == 30)
        {
            /* A 2x2 filter of 65,535 * 8,193 channels is past INT32_MAX; the 1x1 output is not */
            input_dims.c = UINT16_MAX;
            params.ch_mult = 8193;
            filter_dims.w = 2;
            filter_dims.h = 2;
        }
        else if (c == 31)
        {
            input_dims = (cmsis_nn_dims){1, UINT16_MAX, UINT16_MAX, 0}; /* an input plane past INT32_MAX */
        }
        else if (c == 32)
        {
            input_dims = (cmsis_nn_dims){1, 32768, 32768, 2}; /* the input plane fits, the tensor does not */
            params.ch_mult = 1;
        }
        else if (c == 33)
        {
            /* An output plane past INT32_MAX */
            input_dims.c = 0;
            output_dims = (cmsis_nn_dims){1, UINT16_MAX, UINT16_MAX, 0};
        }
        else if (c == 35)
        {
            params.stride.w = 0; /* the kernels divide by the stride */
        }
        else if (c == 36)
        {
            /* Output W times stride past INT32_MAX / 2: 40,000 * 65,535, so the reach would wrap int32_t */
            input_dims = (cmsis_nn_dims){1, 1, 4, 1};
            filter_dims = (cmsis_nn_dims){1, 1, 2, 1};
            params.stride.w = UINT16_MAX;
            params.padding.w = INT16_MAX + 1;
            output_dims = (cmsis_nn_dims){1, 1, 40000, 1};
        }
        else if (c == 37)
        {
            input_dims = (cmsis_nn_dims){1, 4, 1, 1};
            filter_dims = (cmsis_nn_dims){1, 2, 1, 1};
            params.stride.h = UINT16_MAX;
            params.padding.h = INT16_MAX + 1;
            output_dims = (cmsis_nn_dims){1, 40000, 1, 1};
        }
        else if (c == 38)
        {
            /* A horizontal tap step, dilation W * C, past INT32_MAX; no outputs, so only the check rejects it */
            input_dims.c = 32769;
            params.dilation.w = UINT16_MAX;
            output_dims.w = 0;
        }
        else if (c == 39)
        {
            /* A vertical tap step, dilation H * W * C, past INT32_MAX */
            input_dims = (cmsis_nn_dims){1, 1, 256, 256};
            params.dilation.h = UINT16_MAX;
            output_dims.w = 0;
        }
        else if (c == 40)
        {
            params.stride.h = 0;
        }
        else if (c == 41)
        {
            params.dilation.h = 0;
        }
        else if (c == 42)
        {
            /* A pixel step, stride W * C, past INT32_MAX */
            input_dims.c = UINT16_MAX;
            params.stride.w = UINT16_MAX;
            output_dims.w = 0;
        }
        else if (c == 43)
        {
            /* C * ch_mult past INT32_MAX with an empty filter and output plane */
            input_dims.c = UINT16_MAX;
            params.ch_mult = UINT16_MAX;
            filter_dims.w = 0;
            output_dims.w = 0;
        }
        else if (c == 44)
        {
            /* A filter plane past INT32_MAX with no channels */
            input_dims.c = 0;
            filter_dims = (cmsis_nn_dims){1, UINT16_MAX, UINT16_MAX, 0};
            output_dims.w = 0;
        }
        else if (c == 45)
        {
            /* A filter row, KW * C * ch_mult, past INT32_MAX with an empty filter plane */
            params.ch_mult = UINT16_MAX;
            filter_dims = (cmsis_nn_dims){1, 0, UINT16_MAX, 0};
            output_dims.w = 0;
        }
        else if (c == 46)
        {
            /* A tap row, KW * dilation W * C, past INT32_MAX while dilation W * C is not */
            input_dims.c = 32768;
            filter_dims.w = 2;
            params.dilation.w = 32768;
            output_dims.w = 0;
        }
        else if (c == 47)
        {
            /* A tap column, KH * dilation H * W * C, past INT32_MAX while dilation H * W * C is not */
            input_dims.c = 32768;
            filter_dims.h = 2;
            params.dilation.h = 32768;
            output_dims.w = 0;
        }
        else if (c == 48)
        {
            /* Dilation H * W past INT32_MAX with no channels */
            input_dims = (cmsis_nn_dims){1, 1, UINT16_MAX, 0};
            params.dilation.h = 48104;
            output_dims.w = 0;
        }
        else if (c == 49)
        {
            /* C * ch_mult past INT32_MAX / 2 with an empty filter and output plane */
            input_dims.c = 32768;
            params.ch_mult = 32768;
            filter_dims.w = 0;
            output_dims.w = 0;
        }
        else if (c == 50)
        {
            /* Values above 65,535 whose products are past INT32_MAX, which the MVE kernel would otherwise take: an
               input plane of 65,536 * 65,536, with no channels */
            input_dims = (cmsis_nn_dims){1, 65536, 65536, 0};
        }
        else if (c == 51)
        {
            input_dims.c = 0; /* an output plane of 65,536 * 65,536, with no channels */
            output_dims = (cmsis_nn_dims){1, 65536, 65536, 0};
        }
        else if (c == 52)
        {
            /* An empty filter still spans one column: dilation W * C, 32,769 * 65,536, past INT32_MAX */
            input_dims.c = 65536;
            filter_dims.w = 0;
            params.dilation.w = 32769;
            output_dims.w = 0;
        }
        else if (c == 53)
        {
            input_dims.c = 65536; /* and one row: dilation H * W * C, 32,769 * 65,536 */
            filter_dims.h = 0;
            params.dilation.h = 32769;
            output_dims.w = 0;
        }
        else if (c == 54)
        {
            /* An input row, W * C = 2^56, past INT32_MAX with no rows; times dilation H it would wrap 64 bits */
            input_dims = (cmsis_nn_dims){1, 0, 1 << 28, 1 << 28};
            params.dilation.h = 1 << 28;
            output_dims.w = 0;
        }
        else if (c == 55)
        {
            /* 512 taps take the generic kernel on every build, and it takes an input W of 65,537 as 1 */
            filter_dims.w = 32;
            filter_dims.h = 16;
            input_dims.w = 65537;
            output_dims.w = 0;
        }
        else if (c == 34)
        {
            /* 512 taps take the generic kernel on every build, and its tap index cannot hold a padding of 32,768 */
            filter_dims.w = 32;
            filter_dims.h = 16;
            params.padding.w = INT16_MAX + 1;
        }
        wide_dw_output[0] = 0x5555;
        wide_dw_output[1] = 0x5555;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, wide_dw_s16(&params, &input_dims, &filter_dims, &output_dims));
        TEST_ASSERT_EQUAL_INT16(0x5555, wide_dw_output[0]);
        TEST_ASSERT_EQUAL_INT16(0x5555, wide_dw_output[1]);
    }

    /* A dilation of 65,535 on a 1x1 kernel */
    cmsis_nn_dw_conv_params dilated = unit_params;
    dilated.dilation = (cmsis_nn_tile){UINT16_MAX, UINT16_MAX};
    wide_dw_input[0] = 100;
    wide_dw_output[0] = 0x5555;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, wide_dw_s16(&dilated, &unit_dims, &unit_dims, &unit_dims));
    TEST_ASSERT_EQUAL_INT16(arm_nn_requantize_s64(100 * 3 - 7, REDUCE_MULTIPLIER(unit_multiplier), 0),
                            wide_dw_output[0]);

    /* The largest padding and stride the int16_t tap index allows: padding 32,767 puts the one output's only tap in
       the padding, and stride 32,767 over two outputs reads input pixels 0 and 32,767. */
    for (int32_t x = 0; x < INT16_MAX + 1; x++)
    {
        wide_dw_input[x] = (int16_t)(x % 2001 - 1000);
    }
    cmsis_nn_dw_conv_params padded = unit_params;
    padded.padding = (cmsis_nn_tile){INT16_MAX, INT16_MAX};
    wide_dw_output[0] = 0x5555;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, wide_dw_s16(&padded, &unit_dims, &unit_dims, &unit_dims));
    TEST_ASSERT_EQUAL_INT16(arm_nn_requantize_s64(-7, REDUCE_MULTIPLIER(unit_multiplier), 0), wide_dw_output[0]);
    cmsis_nn_dw_conv_params strided = unit_params;
    strided.stride = (cmsis_nn_tile){INT16_MAX, INT16_MAX};
    const cmsis_nn_dims strided_input = {1, 1, INT16_MAX + 1, 1};
    const cmsis_nn_dims strided_output = {1, 1, 2, 1};
    wide_dw_output[1] = 0x5555;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, wide_dw_s16(&strided, &strided_input, &unit_dims, &strided_output));
    for (int32_t x = 0; x < 2; x++)
    {
        TEST_ASSERT_EQUAL_INT16(
            arm_nn_requantize_s64((int64_t)wide_dw_input[x * INT16_MAX] * 3 - 7, REDUCE_MULTIPLIER(unit_multiplier), 0),
            wide_dw_output[x]);
    }

    const cmsis_nn_dims wide_dims = {1, 1, INT16_MAX, 1};
    for (int32_t x = 0; x < INT16_MAX; x++)
    {
        wide_dw_input[x] = (int16_t)(x % 2001 - 1000);
    }
    memset(wide_dw_output, 0x55, sizeof(wide_dw_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, wide_dw_s16(&unit_params, &wide_dims, &unit_dims, &wide_dims));
    const int32_t reduced_multiplier = REDUCE_MULTIPLIER(unit_multiplier);
    for (int32_t x = 0; x < INT16_MAX; x++)
    {
        const int32_t expected = arm_nn_requantize_s64((int64_t)wide_dw_input[x] * 3 - 7, reduced_multiplier, 0);
        TEST_ASSERT_EQUAL_INT16(expected, wide_dw_output[x]);
    }
    TEST_ASSERT_EQUAL_INT16(0x5555, wide_dw_output[INT16_MAX]);
}

/* Edge cases checked against a scalar reference */
#define EDGE_MAX_IN 512
#define EDGE_MAX_OUT 512
#define EDGE_MAX_CH 16
#define EDGE_MAX_KER 512
static int16_t edge_input[EDGE_MAX_IN];
static int8_t edge_filter[EDGE_MAX_KER];
static int16_t edge_output[EDGE_MAX_OUT];
static int16_t edge_reference[EDGE_MAX_OUT];
static int64_t edge_bias[EDGE_MAX_CH];
static int32_t edge_mult[EDGE_MAX_CH];
static int32_t edge_shift[EDGE_MAX_CH];

typedef struct
{
    int32_t in_h, in_w, in_ch, ch_mult;
    int32_t k_h, k_w, stride, pad, dil;
    int32_t in_mag, w_mag; /* input and weight magnitudes */
} edge_case;

static uint32_t edge_seed;
/* Fill every input with -32768 and every weight with -128, the largest product */
static int edge_extreme;
/* Pass a NULL bias; the reference then reads the zeros left in edge_bias */
static int edge_null_bias;

static int32_t edge_rand(int32_t mag)
{
    edge_seed = edge_seed * 1664525u + 1013904223u;
    return (int32_t)((edge_seed >> 8) % (uint32_t)(2 * mag + 1)) - mag;
}

/* Scalar depthwise with s64 requantization */
static void edge_reference_dw(const edge_case *c, int32_t out_h, int32_t out_w)
{
    const int32_t out_ch = c->in_ch * c->ch_mult;
    for (int32_t oy = 0; oy < out_h; oy++)
    {
        for (int32_t ox = 0; ox < out_w; ox++)
        {
            for (int32_t oc = 0; oc < out_ch; oc++)
            {
                int64_t acc = edge_bias[oc];
                for (int32_t ky = 0; ky < c->k_h; ky++)
                {
                    for (int32_t kx = 0; kx < c->k_w; kx++)
                    {
                        const int32_t iy = oy * c->stride - c->pad + ky * c->dil;
                        const int32_t ix = ox * c->stride - c->pad + kx * c->dil;
                        if (iy >= 0 && iy < c->in_h && ix >= 0 && ix < c->in_w)
                        {
                            acc += edge_input[(iy * c->in_w + ix) * c->in_ch + oc / c->ch_mult] *
                                edge_filter[(ky * c->k_w + kx) * out_ch + oc];
                        }
                    }
                }
                int32_t r = arm_nn_requantize_s64(acc, REDUCE_MULTIPLIER(edge_mult[oc]), edge_shift[oc]);
                r = ARM_NN_MIN(ARM_NN_MAX(r, INT16_MIN), INT16_MAX);
                edge_reference[(oy * out_w + ox) * out_ch + oc] = (int16_t)r;
            }
        }
    }
}

/* Run one case; return unclamped output count */
static int32_t edge_run(const edge_case *c)
{
    const int32_t out_ch = c->in_ch * c->ch_mult;
    const int32_t out_h = (c->in_h + 2 * c->pad - (c->k_h - 1) * c->dil - 1) / c->stride + 1;
    const int32_t out_w = (c->in_w + 2 * c->pad - (c->k_w - 1) * c->dil - 1) / c->stride + 1;
    const int32_t out_size = out_h * out_w * out_ch;
    TEST_ASSERT_TRUE(c->in_h * c->in_w * c->in_ch <= EDGE_MAX_IN && out_size <= EDGE_MAX_OUT);
    TEST_ASSERT_TRUE(out_ch <= EDGE_MAX_CH && c->k_h * c->k_w * out_ch <= EDGE_MAX_KER);
    for (int32_t i = 0; i < c->in_h * c->in_w * c->in_ch; i++)
    {
        edge_input[i] = edge_extreme ? INT16_MIN : (int16_t)edge_rand(c->in_mag);
    }
    for (int32_t i = 0; i < c->k_h * c->k_w * out_ch; i++)
    {
        edge_filter[i] = edge_extreme ? INT8_MIN : (int8_t)edge_rand(c->w_mag);
    }
    edge_reference_dw(c, out_h, out_w);

    const cmsis_nn_dw_conv_params params = {.input_offset = 0,
                                            .output_offset = 0,
                                            .ch_mult = c->ch_mult,
                                            .stride = {c->stride, c->stride},
                                            .padding = {c->pad, c->pad},
                                            .dilation = {c->dil, c->dil},
                                            .activation = {INT16_MIN, INT16_MAX}};
    const cmsis_nn_per_channel_quant_params quant = {edge_mult, edge_shift};
    const cmsis_nn_dims input_dims = {1, c->in_h, c->in_w, c->in_ch};
    const cmsis_nn_dims filter_dims = {1, c->k_h, c->k_w, out_ch};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_ch};
    const cmsis_nn_dims output_dims = {1, out_h, out_w, out_ch};
    const cmsis_nn_context ctx = {NULL, 0};
    memset(edge_output, 0x5A, sizeof(edge_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_s16(&ctx,
                                             &params,
                                             &quant,
                                             &input_dims,
                                             edge_input,
                                             &filter_dims,
                                             edge_filter,
                                             &bias_dims,
                                             edge_null_bias ? NULL : edge_bias,
                                             &output_dims,
                                             edge_output));
    TEST_ASSERT_EQUAL_INT16_ARRAY(edge_reference, edge_output, out_size);

    int32_t unclamped = 0;
    for (int32_t i = 0; i < out_size; i++)
    {
        unclamped += (edge_reference[i] != INT16_MAX && edge_reference[i] != INT16_MIN);
    }
    return unclamped;
}

static void edge_quant(int32_t out_ch, int64_t bias_mag, int32_t shift)
{
    for (int32_t i = 0; i < out_ch; i++)
    {
        edge_bias[i] = edge_rand(1 << 12) * bias_mag;
        edge_mult[i] = 0x40000000 + i * 0x01000000;
        edge_shift[i] = shift - (i % 3);
    }
}

/* Run a table with default quantization */
static void edge_run_cases(const edge_case *cases, size_t count, uint32_t seed)
{
    edge_seed = seed;
    for (size_t i = 0; i < count; i++)
    {
        edge_quant(cases[i].in_ch * cases[i].ch_mult, 1, -14);
        TEST_ASSERT_TRUE(edge_run(&cases[i]) > 0);
    }
}

void channel_tail_arm_depthwise_conv_s16(void)
{
    /* Under four outputs, gathers and partial blocks */
    const edge_case cases[] = {
        {6, 5, 1, 3, 3, 3, 1, 1, 1, 16384, 127},
        {5, 5, 1, 2, 3, 3, 2, 2, 2, 16384, 127},
        {4, 7, 3, 1, 2, 3, 1, 1, 1, 16384, 127},
        {4, 4, 3, 3, 3, 2, 1, 1, 1, 16384, 127},
        {4, 4, 2, 4, 2, 2, 1, 1, 1, 16384, 127},
        {6, 9, 4, 1, 3, 3, 2, 1, 1, 16384, 127},
    };
    edge_run_cases(cases, sizeof(cases) / sizeof(cases[0]), 1);
}

void zero_taps_arm_depthwise_conv_s16(void)
{
    /* Padding and dilation leave rows tapless */
    const edge_case cases[] = {
        {2, 2, 1, 3, 2, 2, 1, 3, 5, 16384, 127},
        {4, 3, 5, 1, 2, 2, 1, 3, 1, 16384, 127},
        {3, 3, 4, 2, 2, 2, 2, 4, 3, 16384, 127},
    };
    edge_run_cases(cases, sizeof(cases) / sizeof(cases[0]), 2);

    /* Again with a bias that requantizes to a non-zero output where a pixel has no taps */
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        edge_quant(cases[i].in_ch * cases[i].ch_mult, 1, -6);
        (void)edge_run(&cases[i]);
    }

    /* Empty tensors must bypass the MVE address setup and retain the generic bias-only result. */
    for (int empty = 0; empty < 4; ++empty)
    {
        cmsis_nn_dims input_dims = {1, 1, 1, 1};
        cmsis_nn_dims filter_dims = {1, 1, 1, 1};
        const cmsis_nn_dims output_dims = {1, 1, 1, 1};
        if (empty == 0) filter_dims.w = 0;
        if (empty == 1) filter_dims.h = 0;
        if (empty == 2) input_dims.w = 0;
        if (empty == 3) input_dims.h = 0;
        cmsis_nn_dw_conv_params params = {0};
        params.ch_mult = 1;
        params.stride.w = params.stride.h = params.dilation.w = params.dilation.h = 1;
        params.activation.min = -32768;
        params.activation.max = 32767;
        int32_t multiplier = 1073741824, shift = 1;
        const cmsis_nn_per_channel_quant_params quant = {&multiplier, &shift};
        const int16_t input = 0;
        const int8_t weight = 0;
        const int64_t bias = 12;
        int16_t output = -1;
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
        TEST_ASSERT_FALSE(arm_nn_depthwise_conv_s16_mve(
            &params, &quant, &input_dims, &input, &filter_dims, &weight, &bias, &output_dims, &output));
        TEST_ASSERT_EQUAL(-1, output);
#endif
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_depthwise_conv_s16(NULL, &params, &quant, &input_dims, &input, &filter_dims, &weight,
                                                 NULL, &bias, &output_dims, &output));
        TEST_ASSERT_EQUAL(12, output);
    }

}

void wide_bias_arm_depthwise_conv_s16(void)
{
    /* Bias beyond int32 forces exact s64 path */
    const edge_case c = {5, 6, 5, 1, 3, 3, 1, 1, 1, 16384, 127};
    edge_seed = 3;
    edge_quant(5, 1, -14);
    edge_bias[1] = (int64_t)1 << 40;
    edge_bias[3] = -((int64_t)1 << 39) - 12345;
    edge_shift[1] = -27;
    edge_shift[3] = -26;
    TEST_ASSERT_TRUE(2 * edge_run(&c) >= 5 * 6 * 5);
}

void positive_shift_arm_depthwise_conv_s16(void)
{
    /* Large left shifts force exact s64 path */
    const edge_case cases[] = {
        {5, 6, 5, 1, 3, 3, 1, 1, 1, 16, 7},
        {5, 5, 2, 3, 3, 3, 1, 1, 1, 4, 3},
    };
    const int32_t shifts[] = {4, 10};
    edge_seed = 4;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        const int32_t out_ch = cases[i].in_ch * cases[i].ch_mult;
        edge_quant(out_ch, 0, shifts[i]);
        TEST_ASSERT_TRUE(edge_run(&cases[i]) > 0);
    }

    /* Full-range products: shifted left in 32 bits they would wrap, not saturate */
    const edge_case wide = {5, 6, 5, 1, 3, 3, 1, 1, 1, 16384, 127};
    edge_quant(5, 0, 10);
    (void)edge_run(&wide);

    /* The s64 path with no bias */
    edge_quant(5, 0, 10);
    edge_null_bias = 1;
    (void)edge_run(&cases[0]);
    edge_null_bias = 0;
}

/* 511 taps run the s32 kernel and 512 must not: with every product at its largest, 512 taps sum to 2^31 */
void tap_limit_arm_depthwise_conv_s16(void)
{
    const edge_case cases[] = {
        {1, 511, 1, 1, 1, 511, 1, 0, 1, 0, 0},
        {16, 32, 1, 1, 16, 32, 1, 0, 1, 0, 0},
    };
    edge_extreme = 1;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        edge_quant(1, 0, -20);
        TEST_ASSERT_TRUE(edge_run(&cases[i]) > 0);
    }
    edge_extreme = 0;
}

/* Shifts of -2 to 6 keep a 1x1 layer on the s32 requantization; one channel has the largest multiplier */
void s32_shift_arm_depthwise_conv_s16(void)
{
    const edge_case cases[] = {
        {3, 4, 4, 1, 1, 1, 1, 0, 1, 16, 7},
        {3, 4, 2, 2, 1, 1, 1, 0, 1, 16, 7},
    };
    const int32_t shifts[] = {-2, -1, 0, 1, 2, 4, 6};
    edge_seed = 5;
    for (size_t s = 0; s < sizeof(shifts) / sizeof(shifts[0]); s++)
    {
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        {
            edge_quant(cases[i].in_ch * cases[i].ch_mult, shifts[s] < 2 ? 1 : 0, shifts[s]);
            edge_mult[0] = 0x7FFF8000;
            TEST_ASSERT_TRUE(edge_run(&cases[i]) > 0);
        }
    }
}

/* The s32 bound: shift 0 allows |bias| + taps * 2^22 below 2^29. One tap of the largest product with a bias one
   below, at and above that limit; at and above it, a 32-bit left shift would wrap. */
void s32_bound_arm_depthwise_conv_s16(void)
{
    const edge_case c = {1, 4, 1, 1, 1, 1, 1, 0, 1, 0, 0};
    edge_extreme = 1;
    for (int32_t d = -1; d <= 1; d++)
    {
        edge_quant(1, 0, 0);
        edge_bias[0] = ((int64_t)1 << 29) - ((int64_t)1 << 22) + d;
        (void)edge_run(&c);
    }
    edge_extreme = 0;
}
