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

/* Edge cases checked against a scalar reference */
#define EDGE_MAX_IN 256
#define EDGE_MAX_OUT 512
#define EDGE_MAX_CH 16
#define EDGE_MAX_KER 64
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
        edge_input[i] = (int16_t)edge_rand(c->in_mag);
    }
    for (int32_t i = 0; i < c->k_h * c->k_w * out_ch; i++)
    {
        edge_filter[i] = (int8_t)edge_rand(c->w_mag);
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
                                             edge_bias,
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
}
