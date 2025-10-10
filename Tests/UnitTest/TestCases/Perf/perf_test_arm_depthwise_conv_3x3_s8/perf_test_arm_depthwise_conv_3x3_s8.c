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

#include "unity.h"
#include "arm_nnfunctions.h"
#include "cmsis_perf_profile.h"
#include "perf_estimated_macs.h"

#include "../../TestData/depthwise_kernel_3x3/test_data.h"
#include "../../TestData/depthwise_kernel_3x3_null_bias/test_data.h"
#include "../../TestData/stride2pad1/test_data.h"
#include "../../Utils/utils.h"
#include "../../Utils/validate.h"

void perf_arm_depthwise_conv_3x3_s8_depthwise_kernel_3x3(void)
{
    int8_t output[DEPTHWISE_KERNEL_3X3_PERF_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = depthwise_kernel_3x3_biases;
    const int8_t *kernel_data = depthwise_kernel_3x3_weights;
    const int8_t *input_data = depthwise_kernel_3x3_input;

    input_dims.n = DEPTHWISE_KERNEL_3x3_PERF_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_KERNEL_3X3_PERF_INPUT_W;
    input_dims.h = DEPTHWISE_KERNEL_3X3_PERF_INPUT_H;
    input_dims.c = DEPTHWISE_KERNEL_3X3_PERF_IN_CH;
    filter_dims.w = DEPTHWISE_KERNEL_3X3_FILTER_X;
    filter_dims.h = DEPTHWISE_KERNEL_3X3_FILTER_Y;
    output_dims.w = DEPTHWISE_KERNEL_3X3_PERF_OUTPUT_W;
    output_dims.h = DEPTHWISE_KERNEL_3X3_PERF_OUTPUT_H;
    output_dims.c = DEPTHWISE_KERNEL_3X3_PERF_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_KERNEL_3X3_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_KERNEL_3X3_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_KERNEL_3X3_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_KERNEL_3X3_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_KERNEL_3X3_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_KERNEL_3X3_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_KERNEL_3X3_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_KERNEL_3X3_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_KERNEL_3X3_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_KERNEL_3X3_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_KERNEL_3X3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_kernel_3x3_output_mult;
    quant_params.shift = (int32_t *)depthwise_kernel_3x3_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_depthwise_conv_3x3_s8(&ctx, &dw_conv_params, &quant_params, &input_dims, input_data, &filter_dims, 
                              kernel_data, &bias_dims, bias_data, &output_dims, output);
    
    PERF_DWT_CHARACTERIZE(arm_depthwise_conv_3x3_s8(&ctx, &dw_conv_params, &quant_params, &input_dims, input_data, &filter_dims, 
                                                    kernel_data, &bias_dims, bias_data, &output_dims, output));

    PERF_PMU_CHARACTERIZE(arm_depthwise_conv_3x3_s8(&ctx, &dw_conv_params, &quant_params, &input_dims, input_data, &filter_dims, 
                                                    kernel_data, &bias_dims, bias_data, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
}

/*
void depthwise_kernel_3x3_null_bias_arm_depthwise_conv_3x3_null_bias_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_KERNEL_3X3_NULL_BIAS_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_KERNEL_3X3_NULL_BIAS_DST_SIZE;
    const int32_t *bias_data = depthwise_kernel_3x3_null_bias_biases;
    // get_bias_address(depthwise_kernel_3x3_null_bias_biases, DEPTHWISE_KERNEL_3X3_NULL_BIAS_OUT_CH);
    const int8_t *kernel_data = depthwise_kernel_3x3_null_bias_weights;
    const int8_t *input_data = depthwise_kernel_3x3_null_bias_input;

    input_dims.n = DEPTHWISE_KERNEL_3X3_NULL_BIAS_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_KERNEL_3X3_NULL_BIAS_INPUT_W;
    input_dims.h = DEPTHWISE_KERNEL_3X3_NULL_BIAS_INPUT_H;
    input_dims.c = DEPTHWISE_KERNEL_3X3_NULL_BIAS_IN_CH;
    filter_dims.w = DEPTHWISE_KERNEL_3X3_NULL_BIAS_FILTER_X;
    filter_dims.h = DEPTHWISE_KERNEL_3X3_NULL_BIAS_FILTER_Y;
    output_dims.w = DEPTHWISE_KERNEL_3X3_NULL_BIAS_OUTPUT_W;
    output_dims.h = DEPTHWISE_KERNEL_3X3_NULL_BIAS_OUTPUT_H;
    output_dims.c = DEPTHWISE_KERNEL_3X3_NULL_BIAS_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_KERNEL_3X3_NULL_BIAS_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_KERNEL_3X3_NULL_BIAS_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_KERNEL_3X3_NULL_BIAS_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_KERNEL_3X3_NULL_BIAS_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_KERNEL_3X3_NULL_BIAS_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_KERNEL_3X3_NULL_BIAS_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_KERNEL_3X3_NULL_BIAS_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_KERNEL_3X3_NULL_BIAS_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_KERNEL_3X3_NULL_BIAS_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_KERNEL_3X3_NULL_BIAS_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_KERNEL_3X3_NULL_BIAS_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_kernel_3x3_null_bias_output_mult;
    quant_params.shift = (int32_t *)depthwise_kernel_3x3_null_bias_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_depthwise_conv_3x3_s8(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_kernel_3x3_null_bias_output_ref, output_ref_size));
    memset(output, 0, sizeof(output));

    const arm_cmsis_nn_status expected_wrapper = ARM_CMSIS_NN_SUCCESS;
    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

#if defined(ARM_MATH_MVEI)
    TEST_ASSERT_TRUE(buf_size > 0);
#else
    TEST_ASSERT_EQUAL(buf_size, 0);
#endif

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    result = arm_depthwise_conv_wrapper_s8(&ctx,
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
    TEST_ASSERT_EQUAL(expected_wrapper, result);
    TEST_ASSERT_TRUE(validate(output, depthwise_kernel_3x3_null_bias_output_ref, output_ref_size));
}

void stride2pad1_arm_depthwise_conv_3x3_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[STRIDE2PAD1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = STRIDE2PAD1_DST_SIZE;
    const int32_t *bias_data = get_bias_address(stride2pad1_biases, STRIDE2PAD1_OUT_CH);
    const int8_t *kernel_data = stride2pad1_weights;
    const int8_t *input_data = stride2pad1_input;

    input_dims.n = STRIDE2PAD1_INPUT_BATCHES;
    input_dims.w = STRIDE2PAD1_INPUT_W;
    input_dims.h = STRIDE2PAD1_INPUT_H;
    input_dims.c = STRIDE2PAD1_IN_CH;
    filter_dims.w = STRIDE2PAD1_FILTER_X;
    filter_dims.h = STRIDE2PAD1_FILTER_Y;
    output_dims.w = STRIDE2PAD1_OUTPUT_W;
    output_dims.h = STRIDE2PAD1_OUTPUT_H;
    output_dims.c = STRIDE2PAD1_OUT_CH;

    dw_conv_params.padding.w = STRIDE2PAD1_PAD_X;
    dw_conv_params.padding.h = STRIDE2PAD1_PAD_Y;
    dw_conv_params.stride.w = STRIDE2PAD1_STRIDE_X;
    dw_conv_params.stride.h = STRIDE2PAD1_STRIDE_Y;
    dw_conv_params.dilation.w = STRIDE2PAD1_DILATION_X;
    dw_conv_params.dilation.h = STRIDE2PAD1_DILATION_Y;

    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = STRIDE2PAD1_INPUT_OFFSET;
    dw_conv_params.output_offset = STRIDE2PAD1_OUTPUT_OFFSET;
    dw_conv_params.activation.min = STRIDE2PAD1_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = STRIDE2PAD1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)stride2pad1_output_mult;
    quant_params.shift = (int32_t *)stride2pad1_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_cmsis_nn_status result = arm_depthwise_conv_3x3_s8(&ctx,
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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, stride2pad1_output_ref, output_ref_size));

    memset(output, 0, sizeof(output));

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

#if defined(ARM_MATH_MVEI)
    TEST_ASSERT_TRUE(buf_size > 0);
#else
    TEST_ASSERT_EQUAL(buf_size, 0);
#endif

    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    result = arm_depthwise_conv_wrapper_s8(&ctx,
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
    TEST_ASSERT_TRUE(validate(output, stride2pad1_output_ref, output_ref_size));
}
*/