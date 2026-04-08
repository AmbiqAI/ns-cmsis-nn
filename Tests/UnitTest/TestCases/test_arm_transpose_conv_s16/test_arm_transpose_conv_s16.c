/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include <string.h>

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/transpose_conv_s16_1/test_data.h"
#include "../TestData/transpose_conv_s16_2/test_data.h"
#include "../TestData/transpose_conv_s16_3/test_data.h"
#include "../Utils/validate.h"

void transpose_conv_s16_1_arm_transpose_conv_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[TRANSPOSE_CONV_S16_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int64_t *bias_ptr = transpose_conv_s16_1_biases;
    const cmsis_nn_bias_data bias_data = {bias_ptr, false};
    const int8_t *kernel_data = transpose_conv_s16_1_weights;
    const int16_t *input_data = transpose_conv_s16_1_input;
    const int16_t *output_ref = transpose_conv_s16_1_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_S16_1_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_S16_1_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_S16_1_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_S16_1_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_S16_1_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_S16_1_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_S16_1_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_S16_1_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_S16_1_IN_CH;
    output_dims.n = TRANSPOSE_CONV_S16_1_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_S16_1_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_S16_1_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_S16_1_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_S16_1_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_S16_1_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_S16_1_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_S16_1_PAD_Y_WITH_OFFSET;
    transpose_conv_params.stride.w = TRANSPOSE_CONV_S16_1_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_S16_1_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_S16_1_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_S16_1_DILATION_Y;
    transpose_conv_params.input_offset = 0;
    transpose_conv_params.output_offset = 0;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_S16_1_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_S16_1_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t *)transpose_conv_s16_1_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_s16_1_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s16_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_cmsis_nn_status result = arm_transpose_conv_s16(&ctx,
                                                        NULL,
                                                        &transpose_conv_params,
                                                        &quant_params,
                                                        &input_dims,
                                                        input_data,
                                                        &filter_dims,
                                                        kernel_data,
                                                        &bias_dims,
                                                        &bias_data,
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

void transpose_conv_s16_2_arm_transpose_conv_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[TRANSPOSE_CONV_S16_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    /* No bias for test case 2 */
    const int8_t *kernel_data = transpose_conv_s16_2_weights;
    const int16_t *input_data = transpose_conv_s16_2_input;
    const int16_t *output_ref = transpose_conv_s16_2_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_S16_2_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_S16_2_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_S16_2_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_S16_2_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_S16_2_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_S16_2_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_S16_2_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_S16_2_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_S16_2_IN_CH;
    output_dims.n = TRANSPOSE_CONV_S16_2_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_S16_2_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_S16_2_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_S16_2_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_S16_2_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_S16_2_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_S16_2_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_S16_2_PAD_Y_WITH_OFFSET;
    transpose_conv_params.stride.w = TRANSPOSE_CONV_S16_2_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_S16_2_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_S16_2_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_S16_2_DILATION_Y;
    transpose_conv_params.input_offset = 0;
    transpose_conv_params.output_offset = 0;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_S16_2_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_S16_2_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t *)transpose_conv_s16_2_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_s16_2_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s16_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_cmsis_nn_status result = arm_transpose_conv_s16(&ctx,
                                                        NULL,
                                                        &transpose_conv_params,
                                                        &quant_params,
                                                        &input_dims,
                                                        input_data,
                                                        &filter_dims,
                                                        kernel_data,
                                                        &bias_dims,
                                                        NULL,
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

void transpose_conv_s16_3_arm_transpose_conv_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[TRANSPOSE_CONV_S16_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int64_t *bias_ptr = transpose_conv_s16_3_biases;
    const cmsis_nn_bias_data bias_data = {bias_ptr, false};
    const int8_t *kernel_data = transpose_conv_s16_3_weights;
    const int16_t *input_data = transpose_conv_s16_3_input;
    const int16_t *output_ref = transpose_conv_s16_3_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_S16_3_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_S16_3_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_S16_3_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_S16_3_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_S16_3_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_S16_3_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_S16_3_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_S16_3_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_S16_3_IN_CH;
    output_dims.n = TRANSPOSE_CONV_S16_3_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_S16_3_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_S16_3_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_S16_3_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_S16_3_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_S16_3_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_S16_3_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_S16_3_PAD_Y_WITH_OFFSET;
    transpose_conv_params.stride.w = TRANSPOSE_CONV_S16_3_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_S16_3_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_S16_3_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_S16_3_DILATION_Y;
    transpose_conv_params.input_offset = 0;
    transpose_conv_params.output_offset = 0;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_S16_3_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_S16_3_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t *)transpose_conv_s16_3_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_s16_3_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s16_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_cmsis_nn_status result = arm_transpose_conv_s16(&ctx,
                                                        NULL,
                                                        &transpose_conv_params,
                                                        &quant_params,
                                                        &input_dims,
                                                        input_data,
                                                        &filter_dims,
                                                        kernel_data,
                                                        &bias_dims,
                                                        &bias_data,
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
