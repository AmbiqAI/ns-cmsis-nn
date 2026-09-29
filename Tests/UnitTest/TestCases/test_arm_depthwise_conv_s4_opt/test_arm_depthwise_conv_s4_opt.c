/*
 * SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "../TestData/depthwise_int4_1/test_data.h"
#include "../TestData/depthwise_int4_2/test_data.h"
#include "../TestData/depthwise_int4_3/test_data.h"
#include "../TestData/depthwise_int4_4/test_data.h"
#include "../Utils/mpu_guard.h"
#include "../Utils/utils.h"
#include "../Utils/validate.h"

void depthwise_int4_1_arm_depthwise_conv_s4_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_1_biases;
    const int8_t *kernel_data = depthwise_int4_1_weights;
    const int8_t *input_data = depthwise_int4_1_input;

    input_dims.n = DEPTHWISE_INT4_1_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_1_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_1_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_1_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_1_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_1_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_1_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_1_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_1_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_1_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_1_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_1_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_1_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_1_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_1_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_1_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_1_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_1_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_1_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_1_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_1_output_shift;

    ctx.size = arm_depthwise_conv_s4_opt_get_buffer_size(&input_dims, &filter_dims);

    TEST_ASSERT_TRUE(ctx.size > 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_1_output_ref, DEPTHWISE_INT4_1_DST_SIZE));

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_1_output_ref, DEPTHWISE_INT4_1_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_1_DST_SIZE);

    ctx.size = 0;
    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_s4(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_1_output_ref, DEPTHWISE_INT4_1_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_1_DST_SIZE);
}

void depthwise_int4_2_arm_depthwise_conv_s4_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_2_DST_SIZE] = {};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_2_biases;
    const int8_t *kernel_data = depthwise_int4_2_weights;
    const int8_t *input_data = depthwise_int4_2_input;

    input_dims.n = DEPTHWISE_INT4_2_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_2_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_2_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_2_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_2_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_2_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_2_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_2_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_2_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_2_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_2_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_2_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_2_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_2_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_2_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_2_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_2_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_2_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_2_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_2_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_2_output_shift;

    ctx.size = arm_depthwise_conv_s4_opt_get_buffer_size(&input_dims, &filter_dims);

    TEST_ASSERT_TRUE(ctx.size > 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_2_output_ref, DEPTHWISE_INT4_2_DST_SIZE));

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_2_output_ref, DEPTHWISE_INT4_2_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_2_DST_SIZE);

    ctx.size = 0;
    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_s4(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_2_output_ref, DEPTHWISE_INT4_2_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_2_DST_SIZE);
}

void depthwise_int4_3_arm_depthwise_conv_s4_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_3_DST_SIZE] = {};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_3_biases;
    const int8_t *kernel_data = depthwise_int4_3_weights;
    const int8_t *input_data = depthwise_int4_3_input;

    input_dims.n = DEPTHWISE_INT4_3_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_3_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_3_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_3_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_3_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_3_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_3_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_3_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_3_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_3_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_3_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_3_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_3_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_3_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_3_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_3_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_3_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_3_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_3_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_3_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_3_output_shift;

    ctx.size = arm_depthwise_conv_s4_opt_get_buffer_size(&input_dims, &filter_dims);

    TEST_ASSERT_TRUE(ctx.size > 0);

    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result = arm_depthwise_conv_s4_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_3_output_ref, DEPTHWISE_INT4_3_DST_SIZE));

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_3_output_ref, DEPTHWISE_INT4_3_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_3_DST_SIZE);

    ctx.size = 0;
    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_s4(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_3_output_ref, DEPTHWISE_INT4_3_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_3_DST_SIZE);
}

void depthwise_int4_4_arm_depthwise_conv_s4_opt(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_INT4_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_int4_4_biases;
    const int8_t *kernel_data = depthwise_int4_4_weights;
    const int8_t *input_data = depthwise_int4_4_input;

    input_dims.n = DEPTHWISE_INT4_4_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_INT4_4_INPUT_W;
    input_dims.h = DEPTHWISE_INT4_4_INPUT_H;
    input_dims.c = DEPTHWISE_INT4_4_IN_CH;
    filter_dims.w = DEPTHWISE_INT4_4_FILTER_X;
    filter_dims.h = DEPTHWISE_INT4_4_FILTER_Y;
    output_dims.w = DEPTHWISE_INT4_4_OUTPUT_W;
    output_dims.h = DEPTHWISE_INT4_4_OUTPUT_H;
    output_dims.c = DEPTHWISE_INT4_4_OUT_CH;

    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_dims.c;

    dw_conv_params.padding.w = DEPTHWISE_INT4_4_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_INT4_4_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_INT4_4_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_INT4_4_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_INT4_4_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_INT4_4_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_INT4_4_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_INT4_4_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_INT4_4_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_INT4_4_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_INT4_4_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_int4_4_output_mult;
    quant_params.shift = (int32_t *)depthwise_int4_4_output_shift;

    ctx.size = arm_depthwise_conv_s4_opt_get_buffer_size(&input_dims, &filter_dims);
    TEST_ASSERT_TRUE(ctx.size > 0);

    ctx.buf = malloc(ctx.size);
    arm_cmsis_nn_status result = arm_depthwise_conv_s4_opt(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_4_output_ref, DEPTHWISE_INT4_4_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_4_DST_SIZE);

    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_wrapper_s4(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_4_output_ref, DEPTHWISE_INT4_4_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_4_DST_SIZE);

    ctx.size = 0;
    ctx.buf = malloc(ctx.size);
    result = arm_depthwise_conv_s4(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_int4_4_output_ref, DEPTHWISE_INT4_4_DST_SIZE));
    memset(output, 0, DEPTHWISE_INT4_4_DST_SIZE);
}

// Issue #318: every s4 depthwise sizer routes straight to the s8 _mve/_dsp legs, so before the Helium leg carried
// the dispatcher's dimension gate a negative input_dims->c came back from this family as a plausible positive
// size on a Helium build. Not gated on ARM_MATH_MVEI: the leg variants compile on every build target.
void buffer_size_out_of_range_mve_arm_depthwise_conv_s4_opt(void)
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

    input_dims.n = 1;
    input_dims.h = 65536;
    input_dims.w = 2;
    input_dims.c = -1;
    filter_dims = input_dims;
    output_dims = input_dims;

    TEST_ASSERT_EQUAL(-1, arm_depthwise_conv_s4_opt_get_buffer_size(&input_dims, &filter_dims));
    TEST_ASSERT_EQUAL(
        -1,
        arm_depthwise_conv_wrapper_s4_get_buffer_size_mve(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        -1,
        arm_depthwise_conv_wrapper_s4_get_buffer_size_dsp(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_EQUAL(
        -1, arm_depthwise_conv_wrapper_s4_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims));
}

#if defined(MPU_GUARD_AVAILABLE)
/* Weight t * C + c of a packed int4 filter, low nibble first. */
static int32_t s4_weight(const int8_t *filter, int32_t index)
{
    const uint8_t byte = (uint8_t)filter[index >> 1];
    const int32_t nibble = (index & 1) ? (byte >> 4) : (byte & 0x0f);
    return (nibble ^ 8) - 8;
}
#endif

/* With a channel count that is not a multiple of 4, odd or even, the MVE kernel must stay inside every operand, in
   the four-pixel blocks and in the leftover pixels. Each operand in turn is placed against an MPU gap and the result
   is compared with a scalar reference. Even tap counts end on a row that starts at a high nibble, 131 channels adds
   a second channel block, and the bias is also left out. */
void operand_bounds_arm_depthwise_conv_s4_opt(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    enum
    {
        op_none,
        op_input,
        op_filter,
        op_bias,
        op_mult,
        op_shift,
        op_scratch,
        op_scratch_rows,
        op_output,
        op_end
    };
    enum
    {
        max_ch = 131,
        max_len = 5,
        max_k = 4
    };
    static int8_t input[max_len * max_ch], filter[(max_k * max_ch + 1) / 2], output[max_len * max_ch],
        reference[max_len * max_ch];
    static int32_t bias[max_ch], mult[max_ch], shift[max_ch];
    static int8_t scratch[2048];
    const int32_t taps[] = {2, 3, 4};
    const int32_t channels[] = {1, 2, 3, 5, 6, 7, 17, 131};
    const int32_t lengths[] = {1, 3, 4, 5};
    for (int32_t no_bias = 0; no_bias < 2; no_bias++)
    {
        for (size_t i_k = 0; i_k < sizeof(taps) / sizeof(taps[0]); i_k++)
        {
            for (size_t i_ch = 0; i_ch < sizeof(channels) / sizeof(channels[0]); i_ch++)
            {
                for (size_t i_len = 0; i_len < sizeof(lengths) / sizeof(lengths[0]); i_len++)
                {
                    const int32_t k = taps[i_k], ch = channels[i_ch], len = lengths[i_len];
                    const int32_t filter_bytes = (k * ch + 1) / 2;
                    uint32_t seed = (uint32_t)(len * 131 + ch * 7 + k * 7919);
                    for (int32_t i = 0; i < len * ch; i++)
                    {
                        seed = seed * 1664525u + 1013904223u;
                        input[i] = (int8_t)(seed >> 24);
                    }
                    for (int32_t i = 0; i < filter_bytes; i++)
                    {
                        seed = seed * 1664525u + 1013904223u;
                        filter[i] = (int8_t)(seed >> 24);
                    }
                    for (int32_t i = 0; i < ch; i++)
                    {
                        seed = seed * 1664525u + 1013904223u;
                        bias[i] = no_bias ? 0 : (int32_t)(seed >> 20) - 2048;
                        mult[i] = 0x40000000 + (i % 7) * 0x4000000;
                        shift[i] = -5 - (i % 3);
                    }
                    const cmsis_nn_dw_conv_params params = {.input_offset = 3,
                                                            .output_offset = -2,
                                                            .ch_mult = 1,
                                                            .stride = {1, 1},
                                                            .padding = {k / 2, 0},
                                                            .dilation = {1, 1},
                                                            .activation = {-128, 127}};
                    const cmsis_nn_dims input_dims = {1, 1, len, ch}, filter_dims = {1, 1, k, ch},
                                        bias_dims = {1, 1, 1, ch}, output_dims = {1, 1, len, ch};
                    for (int32_t x = 0; x < len; x++)
                    {
                        for (int32_t c = 0; c < ch; c++)
                        {
                            int32_t acc = bias[c];
                            for (int32_t t = 0; t < k; t++)
                            {
                                const int32_t ix = x - params.padding.w + t;
                                if (ix >= 0 && ix < len)
                                {
                                    acc += (input[ix * ch + c] + params.input_offset) * s4_weight(filter, t * ch + c);
                                }
                            }
                            int32_t r = arm_nn_requantize(acc, mult[c], shift[c]) + params.output_offset;
                            r = r < -128 ? -128 : (r > 127 ? 127 : r);
                            reference[x * ch + c] = (int8_t)r;
                        }
                    }
                    const int32_t scratch_size = arm_depthwise_conv_s4_opt_get_buffer_size(&input_dims, &filter_dims);
                    TEST_ASSERT_TRUE(scratch_size > 0 && scratch_size <= GUARD_OFFSET);
                    /* The im2col rows in use end with the last live channel of the last row written. */
                    const int32_t rows = (len < 4 ? len : 4) * k;
                    const int32_t scratch_rows = (rows - 1) * S4_CH_IN_BLOCK_MVE + ch;

                    for (int op = op_none; op < op_end; op++)
                    {
                        /* The rows-in-use bound above holds for a single channel block only. */
                        if ((op == op_bias && no_bias) || (op == op_scratch_rows && ch > S4_CH_IN_BLOCK_MVE))
                        {
                            continue;
                        }
                        const int8_t *in = op == op_input ? guard_place(input, len * ch) : input;
                        const int8_t *ker = op == op_filter ? guard_place(filter, filter_bytes) : filter;
                        const int32_t *b =
                            no_bias ? NULL : (op == op_bias ? guard_place(bias, ch * sizeof(int32_t)) : bias);
                        int32_t *m = op == op_mult ? guard_place(mult, ch * sizeof(int32_t)) : mult;
                        int32_t *sh = op == op_shift ? guard_place(shift, ch * sizeof(int32_t)) : shift;
                        int8_t *out = op == op_output ? guard_end(len * ch) : output;
                        void *buf = op == op_scratch ? guard_end(scratch_size)
                                                     : (op == op_scratch_rows ? guard_end(scratch_rows) : scratch);
                        const cmsis_nn_per_channel_quant_params quant = {m, sh};
                        const cmsis_nn_context ctx = {buf, scratch_size};
                        memset(out, 0x5A, len * ch);

                        guard_gap_enable();
                        const arm_cmsis_nn_status status = arm_depthwise_conv_s4_opt(&ctx,
                                                                                     &params,
                                                                                     &quant,
                                                                                     &input_dims,
                                                                                     in,
                                                                                     &filter_dims,
                                                                                     ker,
                                                                                     &bias_dims,
                                                                                     b,
                                                                                     &output_dims,
                                                                                     out);
                        guard_gap_disable();

                        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
                        TEST_ASSERT_EQUAL_INT8_ARRAY(reference, out, len * ch);
                    }
                }
            }
        }
    }
#endif
}
