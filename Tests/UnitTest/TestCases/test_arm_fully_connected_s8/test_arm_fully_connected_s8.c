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
#include <arm_nnsupportfunctions.h>
#include <stdlib.h>
#include <unity.h>

#include "../TestData/fc_per_ch/test_data.h"
#include "../TestData/fully_connected/test_data.h"
#include "../TestData/fully_connected_mve_0/test_data.h"
#include "../TestData/fully_connected_mve_1/test_data.h"
#include "../TestData/fully_connected_null_bias_0/test_data.h"
#include "../TestData/fully_connected_out_activation/test_data.h"
#include "../TestData/fully_connected_w_zp/test_data.h"
#include "../Utils/validate.h"
#include "../Utils/mpu_guard.h"

void fully_connected_arm_fully_connected_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FULLY_CONNECTED_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fully_connected_biases;
    const int8_t *kernel_data = fully_connected_weights;
    const int8_t *input_data = fully_connected_input;
    const int8_t *output_ref = fully_connected_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_DST_SIZE;

    input_dims.n = FULLY_CONNECTED_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_INPUT_W;
    input_dims.h = FULLY_CONNECTED_INPUT_H;
    input_dims.c = FULLY_CONNECTED_IN_CH;
    filter_dims.n = FULLY_CONNECTED_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_OUT_CH;
    output_dims.n = FULLY_CONNECTED_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_OUT_CH;

    fc_params.input_offset = FULLY_CONNECTED_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FULLY_CONNECTED_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_OUTPUT_SHIFT;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    TEST_ASSERT_EQUAL(expected,
                      arm_vector_sum_s8(buf,
                                        filter_dims.n,
                                        output_dims.c,
                                        kernel_data,
                                        fc_params.input_offset,
                                        fc_params.filter_offset,
                                        bias_data));
#endif

    arm_cmsis_nn_status result = arm_fully_connected_s8(&ctx,
                                                        &fc_params,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_w_zp_arm_fully_connected_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FULLY_CONNECTED_W_ZP_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fully_connected_w_zp_biases;
    const int8_t *kernel_data = fully_connected_w_zp_weights;
    const int8_t *input_data = fully_connected_w_zp_input;
    const int8_t *output_ref = fully_connected_w_zp_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_W_ZP_DST_SIZE;

    input_dims.n = FULLY_CONNECTED_W_ZP_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_W_ZP_INPUT_W;
    input_dims.h = FULLY_CONNECTED_W_ZP_INPUT_H;
    input_dims.c = FULLY_CONNECTED_W_ZP_IN_CH;
    filter_dims.n = FULLY_CONNECTED_W_ZP_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_W_ZP_OUT_CH;
    output_dims.n = FULLY_CONNECTED_W_ZP_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_W_ZP_OUT_CH;

    fc_params.input_offset = FULLY_CONNECTED_W_ZP_INPUT_OFFSET;
    fc_params.filter_offset = FULLY_CONNECTED_W_ZP_FILTER_OFFSET;
    fc_params.output_offset = FULLY_CONNECTED_W_ZP_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_W_ZP_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_W_ZP_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FULLY_CONNECTED_W_ZP_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_W_ZP_OUTPUT_SHIFT;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    TEST_ASSERT_EQUAL(expected,
                      arm_vector_sum_s8(buf,
                                        filter_dims.n,
                                        output_dims.c,
                                        kernel_data,
                                        fc_params.input_offset,
                                        fc_params.filter_offset,
                                        bias_data));
#endif

    arm_cmsis_nn_status result = arm_fully_connected_s8(&ctx,
                                                        &fc_params,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_mve_0_arm_fully_connected_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FULLY_CONNECTED_MVE_0_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    const int32_t *bias_data = fully_connected_mve_0_biases;
    const int8_t *kernel_data = fully_connected_mve_0_weights;
    const int8_t *input_data = fully_connected_mve_0_input;
    const int8_t *output_ref = fully_connected_mve_0_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_MVE_0_DST_SIZE;
    input_dims.n = FULLY_CONNECTED_MVE_0_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_MVE_0_INPUT_W;
    input_dims.h = FULLY_CONNECTED_MVE_0_INPUT_H;
    input_dims.c = FULLY_CONNECTED_MVE_0_IN_CH;
    filter_dims.n = FULLY_CONNECTED_MVE_0_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_MVE_0_OUT_CH;
    output_dims.n = FULLY_CONNECTED_MVE_0_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_MVE_0_OUT_CH;
    fc_params.input_offset = FULLY_CONNECTED_MVE_0_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_MVE_0_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_MVE_0_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_MVE_0_OUT_ACTIVATION_MAX;
    quant_params.multiplier = FULLY_CONNECTED_MVE_0_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_MVE_0_OUTPUT_SHIFT;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    TEST_ASSERT_EQUAL(expected,
                      arm_vector_sum_s8(buf,
                                        filter_dims.n,
                                        output_dims.c,
                                        kernel_data,
                                        fc_params.input_offset,
                                        fc_params.filter_offset,
                                        bias_data));
#endif

    arm_cmsis_nn_status result = arm_fully_connected_s8(&ctx,
                                                        &fc_params,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_mve_1_arm_fully_connected_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FULLY_CONNECTED_MVE_1_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    const int32_t *bias_data = fully_connected_mve_1_biases;
    const int8_t *kernel_data = fully_connected_mve_1_weights;
    const int8_t *input_data = fully_connected_mve_1_input;
    const int8_t *output_ref = fully_connected_mve_1_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_MVE_1_DST_SIZE;
    input_dims.n = FULLY_CONNECTED_MVE_1_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_MVE_1_INPUT_W;
    input_dims.h = FULLY_CONNECTED_MVE_1_INPUT_H;
    input_dims.c = FULLY_CONNECTED_MVE_1_IN_CH;
    filter_dims.n = FULLY_CONNECTED_MVE_1_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_MVE_1_OUT_CH;
    output_dims.n = FULLY_CONNECTED_MVE_1_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_MVE_1_OUT_CH;
    fc_params.input_offset = FULLY_CONNECTED_MVE_1_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_MVE_1_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_MVE_1_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_MVE_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = FULLY_CONNECTED_MVE_1_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_MVE_1_OUTPUT_SHIFT;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    TEST_ASSERT_EQUAL(expected,
                      arm_vector_sum_s8(buf,
                                        filter_dims.n,
                                        output_dims.c,
                                        kernel_data,
                                        fc_params.input_offset,
                                        fc_params.filter_offset,
                                        bias_data));
#endif

    arm_cmsis_nn_status result = arm_fully_connected_s8(&ctx,
                                                        &fc_params,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_null_bias_0_arm_fully_connected_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FULLY_CONNECTED_NULL_BIAS_0_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    const int32_t *bias_data = fully_connected_null_bias_0_biases;
    const int8_t *kernel_data = fully_connected_null_bias_0_weights;
    const int8_t *input_data = fully_connected_null_bias_0_input;
    const int8_t *output_ref = fully_connected_null_bias_0_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_NULL_BIAS_0_DST_SIZE;
    input_dims.n = FULLY_CONNECTED_NULL_BIAS_0_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_NULL_BIAS_0_INPUT_W;
    input_dims.h = FULLY_CONNECTED_NULL_BIAS_0_INPUT_H;
    input_dims.c = FULLY_CONNECTED_NULL_BIAS_0_IN_CH;
    filter_dims.n = FULLY_CONNECTED_NULL_BIAS_0_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_NULL_BIAS_0_OUT_CH;
    output_dims.n = FULLY_CONNECTED_NULL_BIAS_0_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_NULL_BIAS_0_OUT_CH;
    fc_params.input_offset = FULLY_CONNECTED_NULL_BIAS_0_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_NULL_BIAS_0_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_NULL_BIAS_0_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_NULL_BIAS_0_OUT_ACTIVATION_MAX;
    quant_params.multiplier = FULLY_CONNECTED_NULL_BIAS_0_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_NULL_BIAS_0_OUTPUT_SHIFT;

    arm_cmsis_nn_status ip_check = ARM_CMSIS_NN_SUCCESS;
    for (int i = 0; i < FULLY_CONNECTED_NULL_BIAS_0_OUT_CH; i++)
    {
        if (bias_data && (bias_data[i] != 0))
        {
            ip_check = ARM_CMSIS_NN_ARG_ERROR;
            break;
        }
    }
    TEST_ASSERT_EQUAL(expected, ip_check);

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    TEST_ASSERT_EQUAL(expected,
                      arm_vector_sum_s8(buf,
                                        filter_dims.n,
                                        output_dims.c,
                                        kernel_data,
                                        fc_params.input_offset,
                                        fc_params.filter_offset,
                                        bias_data));
#endif

    arm_cmsis_nn_status result = arm_fully_connected_s8(&ctx,
                                                        &fc_params,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_out_activation_arm_fully_connected_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FULLY_CONNECTED_OUT_ACTIVATION_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    const int32_t *bias_data = fully_connected_out_activation_biases;
    const int8_t *kernel_data = fully_connected_out_activation_weights;
    const int8_t *input_data = fully_connected_out_activation_input;
    const int8_t *output_ref = fully_connected_out_activation_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_OUT_ACTIVATION_DST_SIZE;
    input_dims.n = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_W;
    input_dims.h = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_H;
    input_dims.c = FULLY_CONNECTED_OUT_ACTIVATION_IN_CH;
    filter_dims.n = FULLY_CONNECTED_OUT_ACTIVATION_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_OUT_ACTIVATION_OUT_CH;
    output_dims.n = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_OUT_ACTIVATION_OUT_CH;
    fc_params.input_offset = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_OUT_ACTIVATION_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_OUT_ACTIVATION_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_OUT_ACTIVATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = FULLY_CONNECTED_OUT_ACTIVATION_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_OUT_ACTIVATION_OUTPUT_SHIFT;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    TEST_ASSERT_EQUAL(expected,
                      arm_vector_sum_s8(buf,
                                        filter_dims.n,
                                        output_dims.c,
                                        kernel_data,
                                        fc_params.input_offset,
                                        fc_params.filter_offset,
                                        bias_data));
#endif

    arm_cmsis_nn_status result = arm_fully_connected_s8(&ctx,
                                                        &fc_params,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fc_per_ch_arm_fully_connected_s8(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FC_PER_CH_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fc_per_ch_biases;
    const int8_t *kernel_data = fc_per_ch_weights;
    const int8_t *input_data = fc_per_ch_input;
    const int8_t *output_ref = fc_per_ch_output_ref;
    const int32_t output_ref_size = FC_PER_CH_DST_SIZE;

    input_dims.n = FC_PER_CH_INPUT_BATCHES;
    input_dims.w = FC_PER_CH_INPUT_W;
    input_dims.h = FC_PER_CH_INPUT_H;
    input_dims.c = FC_PER_CH_IN_CH;
    filter_dims.n = FC_PER_CH_ACCUMULATION_DEPTH;
    filter_dims.c = FC_PER_CH_OUT_CH;
    output_dims.n = FC_PER_CH_INPUT_BATCHES;
    output_dims.c = FC_PER_CH_OUT_CH;

    fc_params.input_offset = FC_PER_CH_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FC_PER_CH_OUTPUT_OFFSET;
    fc_params.activation.min = FC_PER_CH_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FC_PER_CH_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t *)fc_per_ch_output_mult;
    quant_params.shift = (int32_t *)fc_per_ch_output_shift;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(
        buf, filter_dims.n, output_dims.c, kernel_data, fc_params.input_offset, fc_params.filter_offset, bias_data);
#endif

    arm_cmsis_nn_status result = arm_fully_connected_per_channel_s8(&ctx,
                                                                    &fc_params,
                                                                    &quant_params,
                                                                    &input_dims,
                                                                    input_data,
                                                                    &filter_dims,
                                                                    kernel_data,
                                                                    &bias_dims,
                                                                    bias_data,
                                                                    &output_dims,
                                                                    output);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));

    cmsis_nn_quant_params generic_quant_params;
    generic_quant_params.multiplier = quant_params.multiplier;
    generic_quant_params.shift = quant_params.shift;
    generic_quant_params.is_per_channel = 1;

    result = arm_fully_connected_wrapper_s8(&ctx,
                                            &fc_params,
                                            &generic_quant_params,
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
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

// Issue #349: the _dsp leg is a public entry point the Python bindings call directly, so an out-of-range channel
// count has to come back as the same -1 the dispatcher and the _mve leg return, not as a 0 that a caller reads as
// "no buffer needed". Deliberately not gated on ARM_MATH_DSP: the leg variants are plain C and are compiled and
// callable on every build target, which is the whole reason a caller can be misled by one.
void buffer_size_out_of_range_dsp_arm_fully_connected_s8(void)
{
    // Negative channel count.
    const cmsis_nn_dims negative_c = {1, 1, 1, -1};
    TEST_ASSERT_EQUAL(-1, arm_fully_connected_s8_get_buffer_size_dsp(&negative_c));
    // Byte count that does not fit in an int32_t.
    const cmsis_nn_dims overflowing_c = {1, 1, 1, 1073741823};
    TEST_ASSERT_EQUAL(-1, arm_fully_connected_s8_get_buffer_size_dsp(&overflowing_c));
    // The other two entry points already answered this way; pin them alongside.
    TEST_ASSERT_EQUAL(-1, arm_fully_connected_s8_get_buffer_size(&negative_c));
    TEST_ASSERT_EQUAL(-1, arm_fully_connected_s8_get_buffer_size_mve(&negative_c));
    TEST_ASSERT_EQUAL(-1, arm_fully_connected_s8_get_buffer_size(&overflowing_c));
    TEST_ASSERT_EQUAL(-1, arm_fully_connected_s8_get_buffer_size_mve(&overflowing_c));
    // An in-range shape is undisturbed: the DSP leg needs no kernel-sum buffer.
    const cmsis_nn_dims valid_c = {1, 1, 1, FULLY_CONNECTED_IN_CH};
    TEST_ASSERT_EQUAL(0, arm_fully_connected_s8_get_buffer_size_dsp(&valid_c));
}

#if defined(MPU_GUARD_AVAILABLE)
/* Runs one layer with its input, then its weights, copied to end at an unmapped MPU gap. The MVE vector-matrix kernels
   must not load past either operand (#605); a load into the gap faults. */
static void fc_operands_at_gap(const cmsis_nn_fc_params *fc_params,
                               const int32_t *per_tensor_mult_shift,
                               const cmsis_nn_per_channel_quant_params *per_channel,
                               const cmsis_nn_dims *input_dims,
                               const int8_t *input_data,
                               const cmsis_nn_dims *filter_dims,
                               const int8_t *kernel_data,
                               const int32_t *bias_data,
                               const cmsis_nn_dims *output_dims,
                               const int8_t *output_ref,
                               int32_t output_ref_size)
{
    const int32_t input_size = input_dims->n * filter_dims->n;
    const int32_t weights_size = filter_dims->n * output_dims->c;
    cmsis_nn_dims bias_dims = {0};
    int8_t output[64];
    TEST_ASSERT_TRUE(output_ref_size <= (int32_t)sizeof(output));
    TEST_ASSERT_TRUE(input_size <= GUARD_OFFSET && weights_size <= GUARD_OFFSET);

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(filter_dims);
    cmsis_nn_context ctx = {malloc(buf_size), buf_size};
    #if defined(ARM_MATH_MVEI)
    arm_vector_sum_s8(ctx.buf,
                      filter_dims->n,
                      output_dims->c,
                      kernel_data,
                      fc_params->input_offset,
                      fc_params->filter_offset,
                      bias_data);
    #endif
    for (int at_gap = 0; at_gap < 2; at_gap++)
    {
        const int8_t *input = at_gap == 0 ? guard_place(input_data, (size_t)input_size) : input_data;
        const int8_t *kernel = at_gap == 1 ? guard_place(kernel_data, (size_t)weights_size) : kernel_data;
        arm_cmsis_nn_status result;
        memset(output, 0, sizeof(output));
        guard_gap_enable();
        if (per_channel)
        {
            result = arm_fully_connected_per_channel_s8(&ctx,
                                                        fc_params,
                                                        per_channel,
                                                        input_dims,
                                                        input,
                                                        filter_dims,
                                                        kernel,
                                                        &bias_dims,
                                                        bias_data,
                                                        output_dims,
                                                        output);
        }
        else
        {
            const cmsis_nn_per_tensor_quant_params quant = {per_tensor_mult_shift[0], per_tensor_mult_shift[1]};
            result = arm_fully_connected_s8(&ctx,
                                            fc_params,
                                            &quant,
                                            input_dims,
                                            input,
                                            filter_dims,
                                            kernel,
                                            &bias_dims,
                                            bias_data,
                                            output_dims,
                                            output);
        }
        guard_gap_disable();
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
    }
    memset(ctx.buf, 0, buf_size);
    free(ctx.buf);
}
#endif

void operands_at_gap_arm_fully_connected_s8(void)
{
#if defined(MPU_GUARD_AVAILABLE)
    {
        const cmsis_nn_fc_params fc_params = {FULLY_CONNECTED_MVE_0_INPUT_OFFSET,
                                              0,
                                              FULLY_CONNECTED_MVE_0_OUTPUT_OFFSET,
                                              {FULLY_CONNECTED_MVE_0_OUT_ACTIVATION_MIN,
                                               FULLY_CONNECTED_MVE_0_OUT_ACTIVATION_MAX}};
        const int32_t mult_shift[2] = {FULLY_CONNECTED_MVE_0_OUTPUT_MULTIPLIER, FULLY_CONNECTED_MVE_0_OUTPUT_SHIFT};
        const cmsis_nn_dims input_dims = {FULLY_CONNECTED_MVE_0_INPUT_BATCHES,
                                          FULLY_CONNECTED_MVE_0_INPUT_H,
                                          FULLY_CONNECTED_MVE_0_INPUT_W,
                                          FULLY_CONNECTED_MVE_0_IN_CH};
        const cmsis_nn_dims filter_dims = {FULLY_CONNECTED_MVE_0_ACCUMULATION_DEPTH, 0, 0, FULLY_CONNECTED_MVE_0_OUT_CH};
        const cmsis_nn_dims output_dims = {FULLY_CONNECTED_MVE_0_INPUT_BATCHES, 0, 0, FULLY_CONNECTED_MVE_0_OUT_CH};
        fc_operands_at_gap(&fc_params,
                           mult_shift,
                           NULL,
                           &input_dims,
                           fully_connected_mve_0_input,
                           &filter_dims,
                           fully_connected_mve_0_weights,
                           fully_connected_mve_0_biases,
                           &output_dims,
                           fully_connected_mve_0_output_ref,
                           FULLY_CONNECTED_MVE_0_DST_SIZE);
    }
    {
        /* A filter zero point takes the vector-matrix loops that also sum the input. */
        const cmsis_nn_fc_params fc_params = {FULLY_CONNECTED_W_ZP_INPUT_OFFSET,
                                              FULLY_CONNECTED_W_ZP_FILTER_OFFSET,
                                              FULLY_CONNECTED_W_ZP_OUTPUT_OFFSET,
                                              {FULLY_CONNECTED_W_ZP_OUT_ACTIVATION_MIN,
                                               FULLY_CONNECTED_W_ZP_OUT_ACTIVATION_MAX}};
        const int32_t mult_shift[2] = {FULLY_CONNECTED_W_ZP_OUTPUT_MULTIPLIER, FULLY_CONNECTED_W_ZP_OUTPUT_SHIFT};
        const cmsis_nn_dims input_dims = {FULLY_CONNECTED_W_ZP_INPUT_BATCHES,
                                          FULLY_CONNECTED_W_ZP_INPUT_H,
                                          FULLY_CONNECTED_W_ZP_INPUT_W,
                                          FULLY_CONNECTED_W_ZP_IN_CH};
        const cmsis_nn_dims filter_dims = {FULLY_CONNECTED_W_ZP_ACCUMULATION_DEPTH, 0, 0, FULLY_CONNECTED_W_ZP_OUT_CH};
        const cmsis_nn_dims output_dims = {FULLY_CONNECTED_W_ZP_INPUT_BATCHES, 0, 0, FULLY_CONNECTED_W_ZP_OUT_CH};
        fc_operands_at_gap(&fc_params,
                           mult_shift,
                           NULL,
                           &input_dims,
                           fully_connected_w_zp_input,
                           &filter_dims,
                           fully_connected_w_zp_weights,
                           fully_connected_w_zp_biases,
                           &output_dims,
                           fully_connected_w_zp_output_ref,
                           FULLY_CONNECTED_W_ZP_DST_SIZE);
    }
    {
        const cmsis_nn_fc_params fc_params = {
            FC_PER_CH_INPUT_OFFSET, 0, FC_PER_CH_OUTPUT_OFFSET, {FC_PER_CH_OUT_ACTIVATION_MIN, FC_PER_CH_OUT_ACTIVATION_MAX}};
        const cmsis_nn_per_channel_quant_params per_channel = {(int32_t *)fc_per_ch_output_mult,
                                                               (int32_t *)fc_per_ch_output_shift};
        const cmsis_nn_dims input_dims = {FC_PER_CH_INPUT_BATCHES, FC_PER_CH_INPUT_H, FC_PER_CH_INPUT_W, FC_PER_CH_IN_CH};
        const cmsis_nn_dims filter_dims = {FC_PER_CH_ACCUMULATION_DEPTH, 0, 0, FC_PER_CH_OUT_CH};
        const cmsis_nn_dims output_dims = {FC_PER_CH_INPUT_BATCHES, 0, 0, FC_PER_CH_OUT_CH};
        fc_operands_at_gap(&fc_params,
                           NULL,
                           &per_channel,
                           &input_dims,
                           fc_per_ch_input,
                           &filter_dims,
                           fc_per_ch_weights,
                           fc_per_ch_biases,
                           &output_dims,
                           fc_per_ch_output_ref,
                           FC_PER_CH_DST_SIZE);
    }
#endif
}

/* arm_fully_connected_per_channel_packed_s8(): layers packed with arm_fully_connected_per_channel_packed_s8_pack()
 * against a scalar per-channel reference, over depth and output tails, batches, offsets and activation ranges. On
 * builds without the kernel the entry returns ARM_CMSIS_NN_NO_IMPL_ERROR and writes nothing. */
#define FC_PACKED_MAX_K (640)
#define FC_PACKED_MAX_N (128)
#define FC_PACKED_MAX_B (3)
#define FC_PACKED_CANARY (16)
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE) && !defined(CMSIS_NN_USE_SINGLE_ROUNDING)
    #define FC_PACKED_RUNS 1
#else
    #define FC_PACKED_RUNS 0
#endif
static int8_t fc_packed_in[FC_PACKED_MAX_B * FC_PACKED_MAX_K];
static int8_t fc_packed_w[FC_PACKED_MAX_N * FC_PACKED_MAX_K];
static int8_t fc_packed_stream[(FC_PACKED_MAX_N / 4) * (4 * FC_PACKED_MAX_K + 48)] __attribute__((aligned(16)));
static int8_t fc_packed_out[FC_PACKED_MAX_B * FC_PACKED_MAX_N + FC_PACKED_CANARY];
static int8_t fc_packed_ref[FC_PACKED_MAX_B * FC_PACKED_MAX_N];
static int32_t fc_packed_bias[FC_PACKED_MAX_N];
static int32_t fc_packed_ksum[FC_PACKED_MAX_N];
static int32_t fc_packed_mult[FC_PACKED_MAX_N];
static int32_t fc_packed_shift[FC_PACKED_MAX_N];
static uint32_t fc_packed_seed = 0x2468aceU;

static int32_t fc_packed_rand(int32_t lo, int32_t hi)
{
    fc_packed_seed = fc_packed_seed * 1664525U + 1013904223U;
    return lo + (int32_t)((fc_packed_seed >> 8) % (uint32_t)(hi - lo + 1));
}

static void fc_packed_case(int32_t k, int32_t n, int32_t batches, int at_gap)
{
    const int32_t in_off = fc_packed_rand(-127, 128);
    const int32_t out_off = fc_packed_rand(-128, 127);
    const int32_t act_min = fc_packed_rand(-128, 0);
    const int32_t act_max = fc_packed_rand(act_min, 127);
    for (int32_t i = 0; i < batches * k; i++)
    {
        fc_packed_in[i] = (int8_t)fc_packed_rand(-128, 127);
    }
    for (int32_t c = 0; c < n; c++)
    {
        int32_t sum = 0;
        for (int32_t i = 0; i < k; i++)
        {
            fc_packed_w[c * k + i] = (int8_t)fc_packed_rand(-127, 127);
            sum += fc_packed_w[c * k + i];
        }
        fc_packed_bias[c] = fc_packed_rand(-20000, 20000);
        fc_packed_ksum[c] = sum * in_off + fc_packed_bias[c];
        fc_packed_mult[c] = (c % 7 == 3) ? 0 : fc_packed_rand(1 << 30, INT32_MAX);
        fc_packed_shift[c] = (c % 5 == 1) ? -1 : fc_packed_rand(-15, -1);
    }
    for (int32_t b = 0; b < batches; b++)
    {
        for (int32_t c = 0; c < n; c++)
        {
            int32_t acc = fc_packed_bias[c];
            for (int32_t i = 0; i < k; i++)
            {
                acc += (fc_packed_in[b * k + i] + in_off) * fc_packed_w[c * k + i];
            }
            acc = arm_nn_requantize(acc, fc_packed_mult[c], fc_packed_shift[c]) + out_off;
            fc_packed_ref[b * n + c] = (int8_t)ARM_NN_MIN(ARM_NN_MAX(acc, act_min), act_max);
        }
    }

    const cmsis_nn_dims filter_dims = {k, 1, 1, n};
    const cmsis_nn_dims input_dims = {batches, 1, 1, k};
    const cmsis_nn_dims output_dims = {batches, 1, 1, n};
    const cmsis_nn_per_channel_quant_params quant_params = {fc_packed_mult, fc_packed_shift};
    const cmsis_nn_fc_params fc_params = {in_off, 0, out_off, {act_min, act_max}};
    const int32_t size = arm_fully_connected_per_channel_packed_s8_get_packed_size(&filter_dims);
    TEST_ASSERT_EQUAL(((n + 3) / 4) * (4 * ((k + 15) / 16 * 16) + 48), size);
    TEST_ASSERT_TRUE(size <= (int32_t)sizeof(fc_packed_stream));
    TEST_ASSERT_EQUAL(1, arm_nn_fc_packed_s8_supported(&fc_params, &quant_params, &filter_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_per_channel_packed_s8_pack(
                          &filter_dims, fc_packed_w, fc_packed_ksum, &quant_params, fc_packed_stream));

    const int8_t *stream = fc_packed_stream;
    const int8_t *input = fc_packed_in;
#if defined(MPU_GUARD_AVAILABLE)
    /* The stream, or the input, ends at an unmapped gap: a read past either faults */
    if (at_gap == 1)
    {
        TEST_ASSERT_TRUE(size <= GUARD_OFFSET);
        stream = guard_place(fc_packed_stream, (size_t)size);
        guard_gap_enable();
    }
    else if (at_gap == 2)
    {
        TEST_ASSERT_TRUE(batches * k <= GUARD_OFFSET);
        input = guard_place(fc_packed_in, (size_t)(batches * k));
        guard_gap_enable();
    }
#else
    (void)at_gap;
#endif
    memset(fc_packed_out, 0x5A, sizeof(fc_packed_out));
    const arm_cmsis_nn_status status = arm_fully_connected_per_channel_packed_s8(
        &fc_params, &input_dims, input, &filter_dims, stream, &output_dims, fc_packed_out);
#if defined(MPU_GUARD_AVAILABLE)
    if (at_gap != 0)
    {
        guard_gap_disable();
    }
#endif
    if (FC_PACKED_RUNS)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
        TEST_ASSERT_EQUAL_INT8_ARRAY(fc_packed_ref, fc_packed_out, batches * n);
    }
    else
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, status);
        for (int32_t i = 0; i < batches * n; i++)
        {
            TEST_ASSERT_EQUAL_INT8(0x5A, fc_packed_out[i]);
        }
    }
    for (int32_t i = batches * n; i < batches * n + FC_PACKED_CANARY; i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, fc_packed_out[i]);
    }
}

void fc_packed_arm_fully_connected_per_channel_packed_s8(void)
{
    for (int32_t k = 1; k <= 48; k++)
    {
        fc_packed_case(k, 1 + k % 9, 1 + k % 3, 0);
    }
    for (int32_t n = 1; n <= 9; n++)
    {
        fc_packed_case(40, n, 2, 0);
        fc_packed_case(40, n, 1, 1);
        fc_packed_case(40, n, 3, 2);
    }
    fc_packed_case(8, 128, 1, 0);
    fc_packed_case(128, 8, 1, 0);
    fc_packed_case(128, 128, 2, 0);
    fc_packed_case(640, 128, 1, 0);
    fc_packed_case(128, 125, 1, 0);
}

/* Argument errors, the declines and the pack rejection; the documented stream layout byte by byte */
void fc_packed_contract_arm_fully_connected_per_channel_packed_s8(void)
{
    int8_t w[5 * 3];
    int32_t ksum[5], mult[5], shift[5];
    for (int32_t i = 0; i < 15; i++)
    {
        w[i] = (int8_t)(i + 1);
    }
    for (int32_t c = 0; c < 5; c++)
    {
        ksum[c] = 1000 + c;
        mult[c] = 0x40000000 + c;
        shift[c] = -1 - c;
    }
    const cmsis_nn_dims filter_dims = {3, 1, 1, 5};
    const cmsis_nn_per_channel_quant_params quant_params = {mult, shift};
    int32_t stream32[(2 * (4 * 16 + 48)) / 4];
    int8_t *stream = (int8_t *)stream32;
    TEST_ASSERT_EQUAL(2 * (4 * 16 + 48), arm_fully_connected_per_channel_packed_s8_get_packed_size(&filter_dims));
    memset(stream, 0x77, sizeof(stream32));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_per_channel_packed_s8_pack(&filter_dims, w, ksum, &quant_params, stream));
    /* Block 0: four rows of one 16-byte group, then four kernel sums, multipliers and shifts */
    for (int32_t r = 0; r < 4; r++)
    {
        for (int32_t j = 0; j < 16; j++)
        {
            TEST_ASSERT_EQUAL_INT8(j < 3 ? w[r * 3 + j] : 0, stream[r * 16 + j]);
        }
        TEST_ASSERT_EQUAL(ksum[r], stream32[16 + r]);
        TEST_ASSERT_EQUAL(mult[r], stream32[20 + r]);
        TEST_ASSERT_EQUAL(shift[r], stream32[24 + r]);
    }
    /* Block 1: row 4, three zero rows, then lane 0 parameters and zero lanes */
    for (int32_t j = 0; j < 64; j++)
    {
        TEST_ASSERT_EQUAL_INT8(j < 3 ? w[12 + j] : 0, stream[112 + j]);
    }
    for (int32_t r = 0; r < 4; r++)
    {
        TEST_ASSERT_EQUAL(r == 0 ? ksum[4] : 0, stream32[44 + r]);
        TEST_ASSERT_EQUAL(r == 0 ? mult[4] : 0, stream32[48 + r]);
        TEST_ASSERT_EQUAL(r == 0 ? shift[4] : 0, stream32[52 + r]);
    }

    /* A channel that needs a left shift is outside the gate and is not packed */
    const cmsis_nn_fc_params fc_params = {0, 0, 0, {-128, 127}};
    shift[2] = 0;
    TEST_ASSERT_EQUAL(0, arm_nn_fc_packed_s8_supported(&fc_params, &quant_params, &filter_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_fully_connected_per_channel_packed_s8_pack(&filter_dims, w, ksum, &quant_params, stream));
    mult[2] = 0;
    TEST_ASSERT_EQUAL(1, arm_nn_fc_packed_s8_supported(&fc_params, &quant_params, &filter_dims));
    const cmsis_nn_fc_params with_filter_offset = {0, 1, 0, {-128, 127}};
    TEST_ASSERT_EQUAL(0, arm_nn_fc_packed_s8_supported(&with_filter_offset, &quant_params, &filter_dims));
    const cmsis_nn_dims no_depth = {0, 1, 1, 5};
    TEST_ASSERT_EQUAL(0, arm_fully_connected_per_channel_packed_s8_get_packed_size(&no_depth));
    TEST_ASSERT_EQUAL(0, arm_fully_connected_per_channel_packed_s8_get_packed_size(NULL));

    const int8_t input[2 * 3] = {1, 2, 3, 4, 5, 6};
    const cmsis_nn_dims input_dims = {2, 1, 1, 3};
    const cmsis_nn_dims output_dims = {2, 1, 1, 5};
    int8_t out[2 * 5];
    memset(out, 0x5A, sizeof(out));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR,
                      arm_fully_connected_per_channel_packed_s8(
                          &with_filter_offset, &input_dims, input, &filter_dims, stream, &output_dims, out));
    for (int32_t i = 0; i < 10; i++)
    {
        TEST_ASSERT_EQUAL_INT8(0x5A, out[i]);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_fully_connected_per_channel_packed_s8(
                          &fc_params, &input_dims, input, &filter_dims, stream + 1, &output_dims, out));
    const cmsis_nn_dims wrong_out = {2, 1, 1, 4};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_fully_connected_per_channel_packed_s8(
                          &fc_params, &input_dims, input, &filter_dims, stream, &wrong_out, out));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_ARG_ERROR,
        arm_fully_connected_per_channel_packed_s8(&fc_params, &input_dims, NULL, &filter_dims, stream, &output_dims, out));
}
