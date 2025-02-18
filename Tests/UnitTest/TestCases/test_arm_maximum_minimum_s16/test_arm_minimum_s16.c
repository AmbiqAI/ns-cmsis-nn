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

#include "stdio.h"
#include <arm_nnfunctions.h>
#include <stdlib.h>
#include <unity.h>

#include "../TestData/minimum_broadcast_batch_int16/test_data.h"
#include "../TestData/minimum_broadcast_ch_int16/test_data.h"
#include "../TestData/minimum_broadcast_height_int16/test_data.h"
#include "../TestData/minimum_broadcast_width_int16/test_data.h"
#include "../TestData/minimum_no_broadcast_int16/test_data.h"
#include "../TestData/minimum_scalar_1_int16/test_data.h"
#include "../TestData/minimum_scalar_2_int16/test_data.h"

#include "../Utils/validate.h"

void minimum_scalar_1_int16(void)
{
    cmsis_nn_context ctx;

    int16_t output[MINIMUM_SCALAR_1_INT16_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_1_data = minimum_scalar_1_int16_input_tensor_1;
    const int16_t *input_2_data = minimum_scalar_1_int16_input_tensor_2;
    const int16_t *output_ref_data = minimum_scalar_1_int16_output;

    input_1_dims.n = MINIMUM_SCALAR_1_INT16_BATCH_1;
    input_1_dims.h = MINIMUM_SCALAR_1_INT16_HEIGHT_1;
    input_1_dims.w = MINIMUM_SCALAR_1_INT16_WIDTH_1;
    input_1_dims.c = MINIMUM_SCALAR_1_INT16_CHANNEL_1;

    input_2_dims.n = MINIMUM_SCALAR_1_INT16_BATCH_2;
    input_2_dims.h = MINIMUM_SCALAR_1_INT16_HEIGHT_2;
    input_2_dims.w = MINIMUM_SCALAR_1_INT16_WIDTH_2;
    input_2_dims.c = MINIMUM_SCALAR_1_INT16_CHANNEL_2;

    output_dims.n = MINIMUM_SCALAR_1_INT16_OUTPUT_BATCH;
    output_dims.h = MINIMUM_SCALAR_1_INT16_OUTPUT_HEIGHT;
    output_dims.w = MINIMUM_SCALAR_1_INT16_OUTPUT_WIDTH;
    output_dims.c = MINIMUM_SCALAR_1_INT16_OUTPUT_CHANNEL;

    arm_cmsis_nn_status result =
        arm_minimum_s16(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    int dst_size = MINIMUM_SCALAR_1_INT16_DST_SIZE;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref_data, dst_size));
}

void minimum_scalar_2_int16(void)
{
    cmsis_nn_context ctx;

    int16_t output[MINIMUM_SCALAR_2_INT16_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_1_data = minimum_scalar_2_int16_input_tensor_1;
    const int16_t *input_2_data = minimum_scalar_2_int16_input_tensor_2;
    const int16_t *output_ref_data = minimum_scalar_2_int16_output;

    input_1_dims.n = MINIMUM_SCALAR_2_INT16_BATCH_1;
    input_1_dims.h = MINIMUM_SCALAR_2_INT16_HEIGHT_1;
    input_1_dims.w = MINIMUM_SCALAR_2_INT16_WIDTH_1;
    input_1_dims.c = MINIMUM_SCALAR_2_INT16_CHANNEL_1;

    input_2_dims.n = MINIMUM_SCALAR_2_INT16_BATCH_2;
    input_2_dims.h = MINIMUM_SCALAR_2_INT16_HEIGHT_2;
    input_2_dims.w = MINIMUM_SCALAR_2_INT16_WIDTH_2;
    input_2_dims.c = MINIMUM_SCALAR_2_INT16_CHANNEL_2;

    output_dims.n = MINIMUM_SCALAR_2_INT16_OUTPUT_BATCH;
    output_dims.h = MINIMUM_SCALAR_2_INT16_OUTPUT_HEIGHT;
    output_dims.w = MINIMUM_SCALAR_2_INT16_OUTPUT_WIDTH;
    output_dims.c = MINIMUM_SCALAR_2_INT16_OUTPUT_CHANNEL;

    arm_cmsis_nn_status result =
        arm_minimum_s16(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    int dst_size = MINIMUM_SCALAR_2_INT16_DST_SIZE;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref_data, dst_size));
}

void minimum_no_broadcast_int16(void)
{
    cmsis_nn_context ctx;
    int16_t output[MINIMUM_NO_BROADCAST_INT16_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_1_data = minimum_no_broadcast_int16_input_tensor_1;
    const int16_t *input_2_data = minimum_no_broadcast_int16_input_tensor_2;
    const int16_t *output_ref_data = minimum_no_broadcast_int16_output;

    input_1_dims.n = MINIMUM_NO_BROADCAST_INT16_BATCH_1;
    input_1_dims.h = MINIMUM_NO_BROADCAST_INT16_HEIGHT_1;
    input_1_dims.w = MINIMUM_NO_BROADCAST_INT16_WIDTH_1;
    input_1_dims.c = MINIMUM_NO_BROADCAST_INT16_CHANNEL_1;

    input_2_dims.n = MINIMUM_NO_BROADCAST_INT16_BATCH_2;
    input_2_dims.h = MINIMUM_NO_BROADCAST_INT16_HEIGHT_2;
    input_2_dims.w = MINIMUM_NO_BROADCAST_INT16_WIDTH_2;
    input_2_dims.c = MINIMUM_NO_BROADCAST_INT16_CHANNEL_2;

    output_dims.n = MINIMUM_NO_BROADCAST_INT16_OUTPUT_BATCH;
    output_dims.h = MINIMUM_NO_BROADCAST_INT16_OUTPUT_HEIGHT;
    output_dims.w = MINIMUM_NO_BROADCAST_INT16_OUTPUT_WIDTH;
    output_dims.c = MINIMUM_NO_BROADCAST_INT16_OUTPUT_CHANNEL;

    arm_cmsis_nn_status result =
        arm_minimum_s16(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    int dst_size = MINIMUM_NO_BROADCAST_INT16_DST_SIZE;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref_data, dst_size));
}

void minimum_broadcast_batch_int16(void)
{
    cmsis_nn_context ctx;
    int16_t output[MINIMUM_BROADCAST_BATCH_INT16_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_1_data = minimum_broadcast_batch_int16_input_tensor_1;
    const int16_t *input_2_data = minimum_broadcast_batch_int16_input_tensor_2;
    const int16_t *output_ref_data = minimum_broadcast_batch_int16_output;

    input_1_dims.n = MINIMUM_BROADCAST_BATCH_INT16_BATCH_1;
    input_1_dims.h = MINIMUM_BROADCAST_BATCH_INT16_HEIGHT_1;
    input_1_dims.w = MINIMUM_BROADCAST_BATCH_INT16_WIDTH_1;
    input_1_dims.c = MINIMUM_BROADCAST_BATCH_INT16_CHANNEL_1;

    input_2_dims.n = MINIMUM_BROADCAST_BATCH_INT16_BATCH_2;
    input_2_dims.h = MINIMUM_BROADCAST_BATCH_INT16_HEIGHT_2;
    input_2_dims.w = MINIMUM_BROADCAST_BATCH_INT16_WIDTH_2;
    input_2_dims.c = MINIMUM_BROADCAST_BATCH_INT16_CHANNEL_2;

    output_dims.n = MINIMUM_BROADCAST_BATCH_INT16_OUTPUT_BATCH;
    output_dims.h = MINIMUM_BROADCAST_BATCH_INT16_OUTPUT_HEIGHT;
    output_dims.w = MINIMUM_BROADCAST_BATCH_INT16_OUTPUT_WIDTH;
    output_dims.c = MINIMUM_BROADCAST_BATCH_INT16_OUTPUT_CHANNEL;

    arm_cmsis_nn_status result =
        arm_minimum_s16(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    int dst_size = MINIMUM_BROADCAST_BATCH_INT16_DST_SIZE;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref_data, dst_size));
}

void minimum_broadcast_height_int16(void)
{
    cmsis_nn_context ctx;
    int16_t output[MINIMUM_BROADCAST_HEIGHT_INT16_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_1_data = minimum_broadcast_height_int16_input_tensor_1;
    const int16_t *input_2_data = minimum_broadcast_height_int16_input_tensor_2;
    const int16_t *output_ref_data = minimum_broadcast_height_int16_output;

    input_1_dims.n = MINIMUM_BROADCAST_HEIGHT_INT16_BATCH_1;
    input_1_dims.h = MINIMUM_BROADCAST_HEIGHT_INT16_HEIGHT_1;
    input_1_dims.w = MINIMUM_BROADCAST_HEIGHT_INT16_WIDTH_1;
    input_1_dims.c = MINIMUM_BROADCAST_HEIGHT_INT16_CHANNEL_1;

    input_2_dims.n = MINIMUM_BROADCAST_HEIGHT_INT16_BATCH_2;
    input_2_dims.h = MINIMUM_BROADCAST_HEIGHT_INT16_HEIGHT_2;
    input_2_dims.w = MINIMUM_BROADCAST_HEIGHT_INT16_WIDTH_2;
    input_2_dims.c = MINIMUM_BROADCAST_HEIGHT_INT16_CHANNEL_2;

    output_dims.n = MINIMUM_BROADCAST_HEIGHT_INT16_OUTPUT_BATCH;
    output_dims.h = MINIMUM_BROADCAST_HEIGHT_INT16_OUTPUT_HEIGHT;
    output_dims.w = MINIMUM_BROADCAST_HEIGHT_INT16_OUTPUT_WIDTH;
    output_dims.c = MINIMUM_BROADCAST_HEIGHT_INT16_OUTPUT_CHANNEL;

    arm_cmsis_nn_status result =
        arm_minimum_s16(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    int dst_size = MINIMUM_BROADCAST_HEIGHT_INT16_DST_SIZE;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref_data, dst_size));
}

void minimum_broadcast_width_int16(void)
{
    cmsis_nn_context ctx;

    int16_t output[MINIMUM_BROADCAST_WIDTH_INT16_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_1_data = minimum_broadcast_width_int16_input_tensor_1;
    const int16_t *input_2_data = minimum_broadcast_width_int16_input_tensor_2;
    const int16_t *output_ref_data = minimum_broadcast_width_int16_output;

    input_1_dims.n = MINIMUM_BROADCAST_WIDTH_INT16_BATCH_1;
    input_1_dims.h = MINIMUM_BROADCAST_WIDTH_INT16_HEIGHT_1;
    input_1_dims.w = MINIMUM_BROADCAST_WIDTH_INT16_WIDTH_1;
    input_1_dims.c = MINIMUM_BROADCAST_WIDTH_INT16_CHANNEL_1;

    input_2_dims.n = MINIMUM_BROADCAST_WIDTH_INT16_BATCH_2;
    input_2_dims.h = MINIMUM_BROADCAST_WIDTH_INT16_HEIGHT_2;
    input_2_dims.w = MINIMUM_BROADCAST_WIDTH_INT16_WIDTH_2;
    input_2_dims.c = MINIMUM_BROADCAST_WIDTH_INT16_CHANNEL_2;

    output_dims.n = MINIMUM_BROADCAST_WIDTH_INT16_OUTPUT_BATCH;
    output_dims.h = MINIMUM_BROADCAST_WIDTH_INT16_OUTPUT_HEIGHT;
    output_dims.w = MINIMUM_BROADCAST_WIDTH_INT16_OUTPUT_WIDTH;
    output_dims.c = MINIMUM_BROADCAST_WIDTH_INT16_OUTPUT_CHANNEL;

    arm_cmsis_nn_status result =
        arm_minimum_s16(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    int dst_size = MINIMUM_BROADCAST_WIDTH_INT16_DST_SIZE;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref_data, dst_size));
}

void minimum_broadcast_ch_int16(void)
{
    cmsis_nn_context ctx;

    int16_t output[MINIMUM_BROADCAST_CH_INT16_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_1_data = minimum_broadcast_ch_int16_input_tensor_1;
    const int16_t *input_2_data = minimum_broadcast_ch_int16_input_tensor_2;
    const int16_t *output_ref_data = minimum_broadcast_ch_int16_output;

    input_1_dims.n = MINIMUM_BROADCAST_CH_INT16_BATCH_1;
    input_1_dims.h = MINIMUM_BROADCAST_CH_INT16_HEIGHT_1;
    input_1_dims.w = MINIMUM_BROADCAST_CH_INT16_WIDTH_1;
    input_1_dims.c = MINIMUM_BROADCAST_CH_INT16_CHANNEL_1;

    input_2_dims.n = MINIMUM_BROADCAST_CH_INT16_BATCH_2;
    input_2_dims.h = MINIMUM_BROADCAST_CH_INT16_HEIGHT_2;
    input_2_dims.w = MINIMUM_BROADCAST_CH_INT16_WIDTH_2;
    input_2_dims.c = MINIMUM_BROADCAST_CH_INT16_CHANNEL_2;

    output_dims.n = MINIMUM_BROADCAST_CH_INT16_OUTPUT_BATCH;
    output_dims.h = MINIMUM_BROADCAST_CH_INT16_OUTPUT_HEIGHT;
    output_dims.w = MINIMUM_BROADCAST_CH_INT16_OUTPUT_WIDTH;
    output_dims.c = MINIMUM_BROADCAST_CH_INT16_OUTPUT_CHANNEL;

    arm_cmsis_nn_status result =
        arm_minimum_s16(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    int dst_size = MINIMUM_BROADCAST_CH_INT16_DST_SIZE;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, output_ref_data, dst_size));
}
