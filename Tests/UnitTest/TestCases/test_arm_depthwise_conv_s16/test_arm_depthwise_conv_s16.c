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
 * arm_depthwise_conv_s16() keeps every dimension, padding, stride and dilation as uint16_t (#707) and the first tap
 * index of an output as int16_t (#727). A value of 65,536, or an input W or H or a padding of 32,768, is an argument
 * error with the output untouched; an input 32,767 wide still runs.
 */
void dims_arg_errors_arm_depthwise_conv_s16(void)
{
    const cmsis_nn_dw_conv_params unit_params = {0, 0, 1, {1, 1}, {0, 0}, {1, 1}, {-32768, 32767}};
    const cmsis_nn_dims unit_dims = {1, 1, 1, 1};
    for (int field = 0; field < 19; field++)
    {
        cmsis_nn_dw_conv_params params = unit_params;
        cmsis_nn_dims input_dims = unit_dims;
        cmsis_nn_dims filter_dims = unit_dims;
        cmsis_nn_dims output_dims = unit_dims;
        int32_t *const values[19] = {&input_dims.n,
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
                                     &params.dilation.h,
                                     &input_dims.w,
                                     &input_dims.h,
                                     &params.padding.w,
                                     &params.padding.h};
        *values[field] = field < 15 ? UINT16_MAX + 1 : INT16_MAX + 1;
        wide_dw_output[0] = 0x5555;
        wide_dw_output[1] = 0x5555;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, wide_dw_s16(&params, &input_dims, &filter_dims, &output_dims));
        TEST_ASSERT_EQUAL_INT16(0x5555, wide_dw_output[0]);
        TEST_ASSERT_EQUAL_INT16(0x5555, wide_dw_output[1]);
    }

    const cmsis_nn_dims wide_dims = {1, 1, INT16_MAX, 1};
    for (int32_t x = 0; x < INT16_MAX; x++)
    {
        wide_dw_input[x] = (int16_t)(x % 2001 - 1000);
    }
    memset(wide_dw_output, 0x55, sizeof(wide_dw_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, wide_dw_s16(&unit_params, &wide_dims, &unit_dims, &wide_dims));
    const int32_t multiplier = 1 << 30;
    const int32_t reduced_multiplier = REDUCE_MULTIPLIER(multiplier);
    for (int32_t x = 0; x < INT16_MAX; x++)
    {
        const int32_t expected = arm_nn_requantize_s64((int64_t)wide_dw_input[x] * 3 - 7, reduced_multiplier, 0);
        TEST_ASSERT_EQUAL_INT16(expected, wide_dw_output[x]);
    }
    TEST_ASSERT_EQUAL_INT16(0x5555, wide_dw_output[INT16_MAX]);
}
