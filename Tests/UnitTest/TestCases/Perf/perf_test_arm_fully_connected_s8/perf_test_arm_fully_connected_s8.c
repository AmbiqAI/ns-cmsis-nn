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

#include <arm_nnfunctions.h>
#include <stdlib.h>
#include <unity.h>
#include "cmsis_perf_profile.h"

#include "../../TestData/perf_fully_connected_int8_tiny_out/test_data.h"
#include "../../TestData/perf_fully_connected_int8_tiny_in/test_data.h"
#include "../../TestData/perf_fully_connected_int8_tiny/test_data.h"
#include "../../TestData/perf_fully_connected_int8_avg/test_data.h"
#include "../../TestData/perf_fully_connected_int8_mobnet/test_data.h"
#include "../../TestData/perf_fully_connected_int8_large/test_data.h"

void perf_arm_fully_connected_s8_tiny_out(void)
{
    int8_t output[PERF_FULLY_CONNECTED_INT8_TINY_OUT_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = perf_fully_connected_int8_tiny_out_biases;
    const int8_t *kernel_data = perf_fully_connected_int8_tiny_out_weights;
    const int8_t *input_data = perf_fully_connected_int8_tiny_out_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_FULLY_CONNECTED_INT8_TINY_OUT_INPUT_BATCHES, PERF_FULLY_CONNECTED_INT8_TINY_OUT_INPUT_W,
                                              PERF_FULLY_CONNECTED_INT8_TINY_OUT_INPUT_H, PERF_FULLY_CONNECTED_INT8_TINY_OUT_IN_CH};

    input_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_OUT_INPUT_BATCHES;
    input_dims.w = PERF_FULLY_CONNECTED_INT8_TINY_OUT_INPUT_W;
    input_dims.h = PERF_FULLY_CONNECTED_INT8_TINY_OUT_INPUT_H;
    input_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_OUT_IN_CH;
    filter_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_OUT_ACCUMULATION_DEPTH;
    filter_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_OUT_OUT_CH;
    output_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_OUT_INPUT_BATCHES;
    output_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_OUT_OUT_CH;

    fc_params.input_offset = PERF_FULLY_CONNECTED_INT8_TINY_OUT_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = PERF_FULLY_CONNECTED_INT8_TINY_OUT_OUTPUT_OFFSET;
    fc_params.activation.min = PERF_FULLY_CONNECTED_INT8_TINY_OUT_OUT_ACTIVATION_MIN;
    fc_params.activation.max = PERF_FULLY_CONNECTED_INT8_TINY_OUT_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t*)perf_fully_connected_int8_tiny_out_output_mult;
    quant_params.shift = (int32_t*)perf_fully_connected_int8_tiny_out_output_shift;
    quant_params.is_per_channel = 1;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(buf, filter_dims.n, output_dims.c, kernel_data, fc_params.input_offset, 
                      fc_params.filter_offset, bias_data);
#endif

    arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    PERF_PMU_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

void perf_arm_fully_connected_s8_tiny_in(void)
{
    int8_t output[PERF_FULLY_CONNECTED_INT8_TINY_IN_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = perf_fully_connected_int8_tiny_in_biases;
    const int8_t *kernel_data = perf_fully_connected_int8_tiny_in_weights;
    const int8_t *input_data = perf_fully_connected_int8_tiny_in_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_FULLY_CONNECTED_INT8_TINY_IN_INPUT_BATCHES, PERF_FULLY_CONNECTED_INT8_TINY_IN_INPUT_W,
                                              PERF_FULLY_CONNECTED_INT8_TINY_IN_INPUT_H, PERF_FULLY_CONNECTED_INT8_TINY_IN_IN_CH};

    input_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_IN_INPUT_BATCHES;
    input_dims.w = PERF_FULLY_CONNECTED_INT8_TINY_IN_INPUT_W;
    input_dims.h = PERF_FULLY_CONNECTED_INT8_TINY_IN_INPUT_H;
    input_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_IN_IN_CH;
    filter_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_IN_ACCUMULATION_DEPTH;
    filter_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_IN_OUT_CH;
    output_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_IN_INPUT_BATCHES;
    output_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_IN_OUT_CH;

    fc_params.input_offset = PERF_FULLY_CONNECTED_INT8_TINY_IN_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = PERF_FULLY_CONNECTED_INT8_TINY_IN_OUTPUT_OFFSET;
    fc_params.activation.min = PERF_FULLY_CONNECTED_INT8_TINY_IN_OUT_ACTIVATION_MIN;
    fc_params.activation.max = PERF_FULLY_CONNECTED_INT8_TINY_IN_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t*)perf_fully_connected_int8_tiny_in_output_mult;
    quant_params.shift = (int32_t*)perf_fully_connected_int8_tiny_in_output_shift;
    quant_params.is_per_channel = 1;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(buf, filter_dims.n, output_dims.c, kernel_data, fc_params.input_offset, 
                      fc_params.filter_offset, bias_data);
#endif

    arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    PERF_PMU_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

void perf_arm_fully_connected_s8_tiny(void)
{
    int8_t output[PERF_FULLY_CONNECTED_INT8_TINY_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = perf_fully_connected_int8_tiny_biases;
    const int8_t *kernel_data = perf_fully_connected_int8_tiny_weights;
    const int8_t *input_data = perf_fully_connected_int8_tiny_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_FULLY_CONNECTED_INT8_TINY_INPUT_BATCHES, PERF_FULLY_CONNECTED_INT8_TINY_INPUT_W,
                                              PERF_FULLY_CONNECTED_INT8_TINY_INPUT_H, PERF_FULLY_CONNECTED_INT8_TINY_IN_CH};

    input_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_INPUT_BATCHES;
    input_dims.w = PERF_FULLY_CONNECTED_INT8_TINY_INPUT_W;
    input_dims.h = PERF_FULLY_CONNECTED_INT8_TINY_INPUT_H;
    input_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_IN_CH;
    filter_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_ACCUMULATION_DEPTH;
    filter_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_OUT_CH;
    output_dims.n = PERF_FULLY_CONNECTED_INT8_TINY_INPUT_BATCHES;
    output_dims.c = PERF_FULLY_CONNECTED_INT8_TINY_OUT_CH;

    fc_params.input_offset = PERF_FULLY_CONNECTED_INT8_TINY_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = PERF_FULLY_CONNECTED_INT8_TINY_OUTPUT_OFFSET;
    fc_params.activation.min = PERF_FULLY_CONNECTED_INT8_TINY_OUT_ACTIVATION_MIN;
    fc_params.activation.max = PERF_FULLY_CONNECTED_INT8_TINY_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t*)perf_fully_connected_int8_tiny_output_mult;
    quant_params.shift = (int32_t*)perf_fully_connected_int8_tiny_output_shift;
    quant_params.is_per_channel = 1;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(buf, filter_dims.n, output_dims.c, kernel_data, fc_params.input_offset, 
                      fc_params.filter_offset, bias_data);
#endif

    arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    PERF_PMU_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

void perf_arm_fully_connected_s8_avg(void)
{
    int8_t output[PERF_FULLY_CONNECTED_INT8_AVG_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = perf_fully_connected_int8_avg_biases;
    const int8_t *kernel_data = perf_fully_connected_int8_avg_weights;
    const int8_t *input_data = perf_fully_connected_int8_avg_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_FULLY_CONNECTED_INT8_AVG_INPUT_BATCHES, PERF_FULLY_CONNECTED_INT8_AVG_INPUT_W,
                                              PERF_FULLY_CONNECTED_INT8_AVG_INPUT_H, PERF_FULLY_CONNECTED_INT8_AVG_IN_CH};

    input_dims.n = PERF_FULLY_CONNECTED_INT8_AVG_INPUT_BATCHES;
    input_dims.w = PERF_FULLY_CONNECTED_INT8_AVG_INPUT_W;
    input_dims.h = PERF_FULLY_CONNECTED_INT8_AVG_INPUT_H;
    input_dims.c = PERF_FULLY_CONNECTED_INT8_AVG_IN_CH;
    filter_dims.n = PERF_FULLY_CONNECTED_INT8_AVG_ACCUMULATION_DEPTH;
    filter_dims.c = PERF_FULLY_CONNECTED_INT8_AVG_OUT_CH;
    output_dims.n = PERF_FULLY_CONNECTED_INT8_AVG_INPUT_BATCHES;
    output_dims.c = PERF_FULLY_CONNECTED_INT8_AVG_OUT_CH;

    fc_params.input_offset = PERF_FULLY_CONNECTED_INT8_AVG_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = PERF_FULLY_CONNECTED_INT8_AVG_OUTPUT_OFFSET;
    fc_params.activation.min = PERF_FULLY_CONNECTED_INT8_AVG_OUT_ACTIVATION_MIN;
    fc_params.activation.max = PERF_FULLY_CONNECTED_INT8_AVG_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t*)perf_fully_connected_int8_avg_output_mult;
    quant_params.shift = (int32_t*)perf_fully_connected_int8_avg_output_shift;
    quant_params.is_per_channel = 1;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(buf, filter_dims.n, output_dims.c, kernel_data, fc_params.input_offset, 
                      fc_params.filter_offset, bias_data);
#endif

    arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    PERF_PMU_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

void perf_arm_fully_connected_s8_mobnet(void)
{
    int8_t output[PERF_FULLY_CONNECTED_INT8_MOBNET_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = perf_fully_connected_int8_mobnet_biases;
    const int8_t *kernel_data = perf_fully_connected_int8_mobnet_weights;
    const int8_t *input_data = perf_fully_connected_int8_mobnet_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_FULLY_CONNECTED_INT8_MOBNET_INPUT_BATCHES, PERF_FULLY_CONNECTED_INT8_MOBNET_INPUT_W,
                                              PERF_FULLY_CONNECTED_INT8_MOBNET_INPUT_H, PERF_FULLY_CONNECTED_INT8_MOBNET_IN_CH};

    input_dims.n = PERF_FULLY_CONNECTED_INT8_MOBNET_INPUT_BATCHES;
    input_dims.w = PERF_FULLY_CONNECTED_INT8_MOBNET_INPUT_W;
    input_dims.h = PERF_FULLY_CONNECTED_INT8_MOBNET_INPUT_H;
    input_dims.c = PERF_FULLY_CONNECTED_INT8_MOBNET_IN_CH;
    filter_dims.n = PERF_FULLY_CONNECTED_INT8_MOBNET_ACCUMULATION_DEPTH;
    filter_dims.c = PERF_FULLY_CONNECTED_INT8_MOBNET_OUT_CH;
    output_dims.n = PERF_FULLY_CONNECTED_INT8_MOBNET_INPUT_BATCHES;
    output_dims.c = PERF_FULLY_CONNECTED_INT8_MOBNET_OUT_CH;

    fc_params.input_offset = PERF_FULLY_CONNECTED_INT8_MOBNET_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = PERF_FULLY_CONNECTED_INT8_MOBNET_OUTPUT_OFFSET;
    fc_params.activation.min = PERF_FULLY_CONNECTED_INT8_MOBNET_OUT_ACTIVATION_MIN;
    fc_params.activation.max = PERF_FULLY_CONNECTED_INT8_MOBNET_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t*)perf_fully_connected_int8_mobnet_output_mult;
    quant_params.shift = (int32_t*)perf_fully_connected_int8_mobnet_output_shift;
    quant_params.is_per_channel = 1;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(buf, filter_dims.n, output_dims.c, kernel_data, fc_params.input_offset, 
                      fc_params.filter_offset, bias_data);
#endif

    arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    PERF_PMU_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

void perf_arm_fully_connected_s8_large(void)
{
    int8_t output[PERF_FULLY_CONNECTED_INT8_LARGE_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = perf_fully_connected_int8_large_biases;
    const int8_t *kernel_data = perf_fully_connected_int8_large_weights;
    const int8_t *input_data = perf_fully_connected_int8_large_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_FULLY_CONNECTED_INT8_LARGE_INPUT_BATCHES, PERF_FULLY_CONNECTED_INT8_LARGE_INPUT_W,
                                              PERF_FULLY_CONNECTED_INT8_LARGE_INPUT_H, PERF_FULLY_CONNECTED_INT8_LARGE_IN_CH};

    input_dims.n = PERF_FULLY_CONNECTED_INT8_LARGE_INPUT_BATCHES;
    input_dims.w = PERF_FULLY_CONNECTED_INT8_LARGE_INPUT_W;
    input_dims.h = PERF_FULLY_CONNECTED_INT8_LARGE_INPUT_H;
    input_dims.c = PERF_FULLY_CONNECTED_INT8_LARGE_IN_CH;
    filter_dims.n = PERF_FULLY_CONNECTED_INT8_LARGE_ACCUMULATION_DEPTH;
    filter_dims.c = PERF_FULLY_CONNECTED_INT8_LARGE_OUT_CH;
    output_dims.n = PERF_FULLY_CONNECTED_INT8_LARGE_INPUT_BATCHES;
    output_dims.c = PERF_FULLY_CONNECTED_INT8_LARGE_OUT_CH;

    fc_params.input_offset = PERF_FULLY_CONNECTED_INT8_LARGE_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = PERF_FULLY_CONNECTED_INT8_LARGE_OUTPUT_OFFSET;
    fc_params.activation.min = PERF_FULLY_CONNECTED_INT8_LARGE_OUT_ACTIVATION_MIN;
    fc_params.activation.max = PERF_FULLY_CONNECTED_INT8_LARGE_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t*)perf_fully_connected_int8_large_output_mult;
    quant_params.shift = (int32_t*)perf_fully_connected_int8_large_output_shift;
    quant_params.is_per_channel = 1;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(buf, filter_dims.n, output_dims.c, kernel_data, fc_params.input_offset, 
                      fc_params.filter_offset, bias_data);
#endif

    arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    PERF_PMU_CHARACTERIZE(arm_fully_connected_wrapper_s8(&ctx, &fc_params, &quant_params, &input_dims, input_data, 
                           &filter_dims, kernel_data, &bias_dims, bias_data, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
}

void perf_arm_fully_connected_s8(void)
{
    perf_arm_fully_connected_s8_tiny_out();
    printf(",\n");
    perf_arm_fully_connected_s8_tiny_in();
    printf(",\n");
    perf_arm_fully_connected_s8_tiny();
    printf(",\n");
    perf_arm_fully_connected_s8_avg();
    printf(",\n");
    perf_arm_fully_connected_s8_mobnet();
    printf(",\n");
    perf_arm_fully_connected_s8_large();
    printf(",\n");
}