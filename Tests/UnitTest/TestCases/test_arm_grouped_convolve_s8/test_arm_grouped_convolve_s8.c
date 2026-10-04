/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "../TestData/grouped_conv_1/test_data.h"
#include "../TestData/grouped_conv_2/test_data.h"
#include "../TestData/grouped_conv_3/test_data.h"
#include "../TestData/grouped_conv_4/test_data.h"
#include "../Utils/validate.h"

void grouped_conv_arm_grouped_convolve_1_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[GROUPED_CONV_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = grouped_conv_1_biases;
    const int8_t *kernel_data = grouped_conv_1_weights;
    const int8_t *input_data = grouped_conv_1_input;
    const int8_t *output_ref = grouped_conv_1_output_ref;
    const int32_t output_ref_size = GROUPED_CONV_1_DST_SIZE;

    input_dims.n = GROUPED_CONV_1_INPUT_BATCHES;
    input_dims.w = GROUPED_CONV_1_INPUT_W;
    input_dims.h = GROUPED_CONV_1_INPUT_H;
    input_dims.c = GROUPED_CONV_1_IN_CH;

    filter_dims.n = GROUPED_CONV_1_OUT_CH;
    filter_dims.w = GROUPED_CONV_1_FILTER_X;
    filter_dims.h = GROUPED_CONV_1_FILTER_Y;
    filter_dims.c = GROUPED_CONV_1_FILTER_CH;

    output_dims.n = GROUPED_CONV_1_INPUT_BATCHES;
    output_dims.w = GROUPED_CONV_1_OUTPUT_W;
    output_dims.h = GROUPED_CONV_1_OUTPUT_H;
    output_dims.c = GROUPED_CONV_1_OUT_CH;

    conv_params.padding.w = GROUPED_CONV_1_PAD_X;
    conv_params.padding.h = GROUPED_CONV_1_PAD_Y;
    conv_params.stride.w = GROUPED_CONV_1_STRIDE_X;
    conv_params.stride.h = GROUPED_CONV_1_STRIDE_Y;
    conv_params.dilation.w = GROUPED_CONV_1_DILATION_X;
    conv_params.dilation.h = GROUPED_CONV_1_DILATION_Y;

    conv_params.input_offset = GROUPED_CONV_1_INPUT_OFFSET;
    conv_params.output_offset = GROUPED_CONV_1_OUTPUT_OFFSET;
    conv_params.activation.min = GROUPED_CONV_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = GROUPED_CONV_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)grouped_conv_1_output_mult;
    quant_params.shift = (int32_t *)grouped_conv_1_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);
    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
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
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

void grouped_conv_arm_grouped_convolve_2_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[GROUPED_CONV_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = grouped_conv_2_biases;
    const int8_t *kernel_data = grouped_conv_2_weights;
    const int8_t *input_data = grouped_conv_2_input;
    const int8_t *output_ref = grouped_conv_2_output_ref;
    const int32_t output_ref_size = GROUPED_CONV_2_DST_SIZE;

    input_dims.n = GROUPED_CONV_2_INPUT_BATCHES;
    input_dims.w = GROUPED_CONV_2_INPUT_W;
    input_dims.h = GROUPED_CONV_2_INPUT_H;
    input_dims.c = GROUPED_CONV_2_IN_CH;

    filter_dims.n = GROUPED_CONV_2_OUT_CH;
    filter_dims.w = GROUPED_CONV_2_FILTER_X;
    filter_dims.h = GROUPED_CONV_2_FILTER_Y;
    filter_dims.c = GROUPED_CONV_2_FILTER_CH;

    input_dims.n = GROUPED_CONV_2_INPUT_BATCHES;
    output_dims.w = GROUPED_CONV_2_OUTPUT_W;
    output_dims.h = GROUPED_CONV_2_OUTPUT_H;
    output_dims.c = GROUPED_CONV_2_OUT_CH;

    conv_params.padding.w = GROUPED_CONV_2_PAD_X;
    conv_params.padding.h = GROUPED_CONV_2_PAD_Y;
    conv_params.stride.w = GROUPED_CONV_2_STRIDE_X;
    conv_params.stride.h = GROUPED_CONV_2_STRIDE_Y;
    conv_params.dilation.w = GROUPED_CONV_2_DILATION_X;
    conv_params.dilation.h = GROUPED_CONV_2_DILATION_Y;

    conv_params.input_offset = GROUPED_CONV_2_INPUT_OFFSET;
    conv_params.output_offset = GROUPED_CONV_2_OUTPUT_OFFSET;
    conv_params.activation.min = GROUPED_CONV_2_OUT_ACTIVATION_MIN;
    conv_params.activation.max = GROUPED_CONV_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)grouped_conv_2_output_mult;
    quant_params.shift = (int32_t *)grouped_conv_2_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);
    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
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
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

void grouped_conv_arm_grouped_convolve_3_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[GROUPED_CONV_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = grouped_conv_3_biases;
    const int8_t *kernel_data = grouped_conv_3_weights;
    const int8_t *input_data = grouped_conv_3_input;
    const int8_t *output_ref = grouped_conv_3_output_ref;
    const int32_t output_ref_size = GROUPED_CONV_3_DST_SIZE;

    input_dims.n = GROUPED_CONV_3_INPUT_BATCHES;
    input_dims.w = GROUPED_CONV_3_INPUT_W;
    input_dims.h = GROUPED_CONV_3_INPUT_H;
    input_dims.c = GROUPED_CONV_3_IN_CH;

    filter_dims.n = GROUPED_CONV_3_OUT_CH;
    filter_dims.w = GROUPED_CONV_3_FILTER_X;
    filter_dims.h = GROUPED_CONV_3_FILTER_Y;
    filter_dims.c = GROUPED_CONV_3_FILTER_CH;

    output_dims.n = GROUPED_CONV_3_INPUT_BATCHES;
    output_dims.w = GROUPED_CONV_3_OUTPUT_W;
    output_dims.h = GROUPED_CONV_3_OUTPUT_H;
    output_dims.c = GROUPED_CONV_3_OUT_CH;

    conv_params.padding.w = GROUPED_CONV_3_PAD_X;
    conv_params.padding.h = GROUPED_CONV_3_PAD_Y;
    conv_params.stride.w = GROUPED_CONV_3_STRIDE_X;
    conv_params.stride.h = GROUPED_CONV_3_STRIDE_Y;
    conv_params.dilation.w = GROUPED_CONV_3_DILATION_X;
    conv_params.dilation.h = GROUPED_CONV_3_DILATION_Y;

    conv_params.input_offset = GROUPED_CONV_3_INPUT_OFFSET;
    conv_params.output_offset = GROUPED_CONV_3_OUTPUT_OFFSET;
    conv_params.activation.min = GROUPED_CONV_3_OUT_ACTIVATION_MIN;
    conv_params.activation.max = GROUPED_CONV_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)grouped_conv_3_output_mult;
    quant_params.shift = (int32_t *)grouped_conv_3_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);
    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
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
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

void grouped_conv_arm_grouped_convolve_4_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[GROUPED_CONV_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = grouped_conv_4_biases;
    const int8_t *kernel_data = grouped_conv_4_weights;
    const int8_t *input_data = grouped_conv_4_input;
    const int8_t *output_ref = grouped_conv_4_output_ref;
    const int32_t output_ref_size = GROUPED_CONV_4_DST_SIZE;

    input_dims.n = GROUPED_CONV_4_INPUT_BATCHES;
    input_dims.w = GROUPED_CONV_4_INPUT_W;
    input_dims.h = GROUPED_CONV_4_INPUT_H;
    input_dims.c = GROUPED_CONV_4_IN_CH;

    filter_dims.n = GROUPED_CONV_4_OUT_CH;
    filter_dims.w = GROUPED_CONV_4_FILTER_X;
    filter_dims.h = GROUPED_CONV_4_FILTER_Y;
    filter_dims.c = GROUPED_CONV_4_FILTER_CH;

    output_dims.n = GROUPED_CONV_4_INPUT_BATCHES;
    output_dims.w = GROUPED_CONV_4_OUTPUT_W;
    output_dims.h = GROUPED_CONV_4_OUTPUT_H;
    output_dims.c = GROUPED_CONV_4_OUT_CH;

    conv_params.padding.w = GROUPED_CONV_4_PAD_X;
    conv_params.padding.h = GROUPED_CONV_4_PAD_Y;
    conv_params.stride.w = GROUPED_CONV_4_STRIDE_X;
    conv_params.stride.h = GROUPED_CONV_4_STRIDE_Y;
    conv_params.dilation.w = GROUPED_CONV_4_DILATION_X;
    conv_params.dilation.h = GROUPED_CONV_4_DILATION_Y;

    conv_params.input_offset = GROUPED_CONV_4_INPUT_OFFSET;
    conv_params.output_offset = GROUPED_CONV_4_OUTPUT_OFFSET;
    conv_params.activation.min = GROUPED_CONV_4_OUT_ACTIVATION_MIN;
    conv_params.activation.max = GROUPED_CONV_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)grouped_conv_4_output_mult;
    quant_params.shift = (int32_t *)grouped_conv_4_output_shift;

    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_context weights_sum_ctx;
    int32_t weights_sum_buf_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    weights_sum_ctx.buf = malloc(weights_sum_buf_size);
    weights_sum_ctx.size = weights_sum_buf_size;
    uint32_t lhs_offset = conv_params.input_offset; 
    arm_convolve_weight_sum(weights_sum_ctx.buf, kernel_data,&input_dims,&filter_dims, &output_dims, lhs_offset,  bias_data);
    arm_cmsis_nn_status result = arm_convolve_s8(&ctx,
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
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    memset(output, 0, sizeof(output));
}

/* A NULL bias must act as an all-zero bias in every group (#697): the per-group bias pointer is formed and advanced
 * only for a real bias. */
static void grouped_conv_1_run(const int32_t *bias_data, int8_t *output)
{
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    const cmsis_nn_dims input_dims = {
        GROUPED_CONV_1_INPUT_BATCHES, GROUPED_CONV_1_INPUT_H, GROUPED_CONV_1_INPUT_W, GROUPED_CONV_1_IN_CH};
    const cmsis_nn_dims filter_dims = {
        GROUPED_CONV_1_OUT_CH, GROUPED_CONV_1_FILTER_Y, GROUPED_CONV_1_FILTER_X, GROUPED_CONV_1_FILTER_CH};
    const cmsis_nn_dims bias_dims = {1, 1, 1, GROUPED_CONV_1_OUT_CH};
    const cmsis_nn_dims output_dims = {
        GROUPED_CONV_1_INPUT_BATCHES, GROUPED_CONV_1_OUTPUT_H, GROUPED_CONV_1_OUTPUT_W, GROUPED_CONV_1_OUT_CH};
    conv_params.padding.w = GROUPED_CONV_1_PAD_X;
    conv_params.padding.h = GROUPED_CONV_1_PAD_Y;
    conv_params.stride.w = GROUPED_CONV_1_STRIDE_X;
    conv_params.stride.h = GROUPED_CONV_1_STRIDE_Y;
    conv_params.dilation.w = GROUPED_CONV_1_DILATION_X;
    conv_params.dilation.h = GROUPED_CONV_1_DILATION_Y;
    conv_params.input_offset = GROUPED_CONV_1_INPUT_OFFSET;
    conv_params.output_offset = GROUPED_CONV_1_OUTPUT_OFFSET;
    conv_params.activation.min = GROUPED_CONV_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = GROUPED_CONV_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)grouped_conv_1_output_mult;
    quant_params.shift = (int32_t *)grouped_conv_1_output_shift;

    const int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    cmsis_nn_context ctx = {buf_size > 0 ? malloc(buf_size) : NULL, buf_size};
    const int32_t weights_sum_size = arm_convolve_s8_get_weights_sum_size(&output_dims);
    cmsis_nn_context weights_sum_ctx = {weights_sum_size > 0 ? malloc(weights_sum_size) : NULL, weights_sum_size};
    arm_convolve_weight_sum(weights_sum_ctx.buf,
                            grouped_conv_1_weights,
                            &input_dims,
                            &filter_dims,
                            &output_dims,
                            conv_params.input_offset,
                            bias_data);
    const arm_cmsis_nn_status status = arm_convolve_s8(&ctx,
                                                       &weights_sum_ctx,
                                                       &conv_params,
                                                       &quant_params,
                                                       &input_dims,
                                                       grouped_conv_1_input,
                                                       &filter_dims,
                                                       grouped_conv_1_weights,
                                                       &bias_dims,
                                                       bias_data,
                                                       NULL,
                                                       &output_dims,
                                                       output);
    free(weights_sum_ctx.buf);
    free(ctx.buf);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
}

void grouped_conv_null_bias_arm_grouped_convolve_s8(void)
{
    TEST_ASSERT_TRUE(GROUPED_CONV_1_IN_CH / GROUPED_CONV_1_FILTER_CH > 1);
    static const int32_t zero_bias[GROUPED_CONV_1_OUT_CH] = {0};
    static int8_t with_zero[GROUPED_CONV_1_DST_SIZE];
    static int8_t with_null[GROUPED_CONV_1_DST_SIZE];
    memset(with_zero, 0x55, sizeof(with_zero));
    memset(with_null, 0x5A, sizeof(with_null));
    grouped_conv_1_run(zero_bias, with_zero);
    grouped_conv_1_run(NULL, with_null);
    TEST_ASSERT_EQUAL_INT8_ARRAY(with_zero, with_null, GROUPED_CONV_1_DST_SIZE);
}

/* arm_convolve_s8() on a 1x2x2xC_IN layer with a 1x1 filter of depth filter_c, returning its status; anything it
 * writes lands in output. */
static arm_cmsis_nn_status grouped_conv_status(int32_t in_c,
                                               int32_t filter_c,
                                               int32_t out_c,
                                               const cmsis_nn_dims *upscale_dims,
                                               int8_t *output)
{
    static int8_t input[2 * 2 * 8];
    static int8_t weights[8 * 8];
    static int32_t mult[8];
    static int32_t shift[8];
    static int32_t weight_sum[8];
    static const int32_t bias[8] = {0};
    static int16_t scratch[256];
    for (int32_t i = 0; i < (int32_t)sizeof(input); i++)
    {
        input[i] = (int8_t)(i * 7 - 50);
    }
    for (int32_t i = 0; i < (int32_t)sizeof(weights); i++)
    {
        weights[i] = (int8_t)(i * 5 - 31);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        mult[i] = 1 << 30;
        shift[i] = -1;
        weight_sum[i] = 0;
    }
    const cmsis_nn_dims input_dims = {1, 2, 2, in_c};
    const cmsis_nn_dims filter_dims = {out_c, 1, 1, filter_c};
    const cmsis_nn_dims bias_dims = {1, 1, 1, out_c};
    const cmsis_nn_dims output_dims = {1, 2, 2, out_c};
    const cmsis_nn_conv_params conv_params = {
        .input_offset = 3,
        .output_offset = -2,
        .stride = {1, 1},
        .padding = {0, 0},
        .dilation = {1, 1},
        .activation = {-128, 127},
    };
    const cmsis_nn_per_channel_quant_params quant_params = {.multiplier = mult, .shift = shift};
    const cmsis_nn_context ctx = {scratch, (int32_t)sizeof(scratch)};
    const cmsis_nn_context weight_sum_ctx = {weight_sum, (int32_t)sizeof(weight_sum)};
    return arm_convolve_s8(&ctx,
                           &weight_sum_ctx,
                           &conv_params,
                           &quant_params,
                           &input_dims,
                           input,
                           &filter_dims,
                           weights,
                           &bias_dims,
                           bias,
                           upscale_dims,
                           &output_dims,
                           output);
}

/* Shapes arm_convolve_s8() cannot compute are argument errors that write nothing (#700, #702): a grouped layer with
 * an upscale factor of 2 on either axis, an input depth that is not a whole number of filter depths, an output
 * depth that is not a whole number of groups, and a depth of 0. A grouped layer with an upscale of 1 and an ungrouped
 * layer stay valid. */
void grouped_conv_arg_errors_arm_grouped_convolve_s8(void)
{
    const cmsis_nn_dims up_hw = {0, 2, 2, 0};
    const cmsis_nn_dims up_h = {0, 2, 1, 0};
    const cmsis_nn_dims up_w = {0, 1, 2, 0};
    const cmsis_nn_dims up_none = {0, 1, 1, 0};
    const struct
    {
        int32_t in_c, filter_c, out_c;
        const cmsis_nn_dims *upscale;
        arm_cmsis_nn_status expected;
    } cases[] = {
        {4, 2, 4, &up_hw, ARM_CMSIS_NN_ARG_ERROR},
        {4, 2, 4, &up_h, ARM_CMSIS_NN_ARG_ERROR},
        {4, 2, 4, &up_w, ARM_CMSIS_NN_ARG_ERROR},
        {6, 4, 4, NULL, ARM_CMSIS_NN_ARG_ERROR},
        {7, 4, 4, NULL, ARM_CMSIS_NN_ARG_ERROR},
        {4, 0, 4, NULL, ARM_CMSIS_NN_ARG_ERROR},
        {0, 2, 4, NULL, ARM_CMSIS_NN_ARG_ERROR},
        /* depths kept as uint16_t: 65,536 would read as 0, a negative output depth as 65,532 */
        {65536, 65536, 1, NULL, ARM_CMSIS_NN_ARG_ERROR},
        {4, 2, -4, NULL, ARM_CMSIS_NN_ARG_ERROR},
        {4, 2, 3, NULL, ARM_CMSIS_NN_ARG_ERROR},
        {4, 2, 4, &up_none, ARM_CMSIS_NN_SUCCESS},
        {4, 2, 4, NULL, ARM_CMSIS_NN_SUCCESS},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    {
        int8_t output[2 * 2 * 8];
        memset(output, 0x5A, sizeof(output));
        TEST_ASSERT_EQUAL_MESSAGE(cases[i].expected,
                                  grouped_conv_status(
                                      cases[i].in_c, cases[i].filter_c, cases[i].out_c, cases[i].upscale, output),
                                  "case");
        if (cases[i].expected != ARM_CMSIS_NN_SUCCESS)
        {
            for (size_t j = 0; j < sizeof(output); j++)
            {
                TEST_ASSERT_EQUAL_INT8(0x5A, output[j]);
            }
        }
    }
}
