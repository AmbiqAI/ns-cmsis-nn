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

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "unity.h"

#include "../TestData/avgpooling_int16/test_data.h"
#include "../TestData/avgpooling_int16_1/test_data.h"
#include "../TestData/avgpooling_int16_2/test_data.h"
#include "../TestData/avgpooling_int16_3/test_data.h"
#include "../Utils/validate.h"

void avgpooling_int16_arm_avgpool_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[AVGPOOLING_INT16_OUTPUT_C * AVGPOOLING_INT16_OUTPUT_W * AVGPOOLING_INT16_OUTPUT_H *
                   AVGPOOLING_INT16_BATCH_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_data = avgpooling_int16_input_tensor;

    input_dims.n = AVGPOOLING_INT16_BATCH_SIZE;
    input_dims.w = AVGPOOLING_INT16_INPUT_W;
    input_dims.h = AVGPOOLING_INT16_INPUT_H;
    input_dims.c = AVGPOOLING_INT16_INPUT_C;
    filter_dims.w = AVGPOOLING_INT16_FILTER_W;
    filter_dims.h = AVGPOOLING_INT16_FILTER_H;
    output_dims.w = AVGPOOLING_INT16_OUTPUT_W;
    output_dims.h = AVGPOOLING_INT16_OUTPUT_H;
    output_dims.c = AVGPOOLING_INT16_INPUT_C;

    pool_params.padding.w = AVGPOOLING_INT16_PADDING_W;
    pool_params.padding.h = AVGPOOLING_INT16_PADDING_H;
    pool_params.stride.w = AVGPOOLING_INT16_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_INT16_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_INT16_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_INT16_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_OUTPUT_W, AVGPOOLING_INT16_INPUT_C);
    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result =
        arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                                  avgpooling_int16_output,
                                  AVGPOOLING_INT16_OUTPUT_C * AVGPOOLING_INT16_OUTPUT_W * AVGPOOLING_INT16_OUTPUT_H *
                                      AVGPOOLING_INT16_BATCH_SIZE));
}

void avgpooling_int16_1_arm_avgpool_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[AVGPOOLING_INT16_1_OUTPUT_C * AVGPOOLING_INT16_1_OUTPUT_W * AVGPOOLING_INT16_1_OUTPUT_H *
                   AVGPOOLING_INT16_1_BATCH_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_data = avgpooling_int16_1_input_tensor;

    input_dims.n = AVGPOOLING_INT16_1_BATCH_SIZE;
    input_dims.w = AVGPOOLING_INT16_1_INPUT_W;
    input_dims.h = AVGPOOLING_INT16_1_INPUT_H;
    input_dims.c = AVGPOOLING_INT16_1_INPUT_C;
    filter_dims.w = AVGPOOLING_INT16_1_FILTER_W;
    filter_dims.h = AVGPOOLING_INT16_1_FILTER_H;
    output_dims.w = AVGPOOLING_INT16_1_OUTPUT_W;
    output_dims.h = AVGPOOLING_INT16_1_OUTPUT_H;
    output_dims.c = AVGPOOLING_INT16_1_INPUT_C;

    pool_params.padding.w = AVGPOOLING_INT16_1_PADDING_W;
    pool_params.padding.h = AVGPOOLING_INT16_1_PADDING_H;
    pool_params.stride.w = AVGPOOLING_INT16_1_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_INT16_1_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_INT16_1_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_INT16_1_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_1_OUTPUT_W, AVGPOOLING_INT16_1_INPUT_C);
    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result =
        arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                                  avgpooling_int16_1_output,
                                  AVGPOOLING_INT16_1_OUTPUT_C * AVGPOOLING_INT16_1_OUTPUT_W *
                                      AVGPOOLING_INT16_1_OUTPUT_H * AVGPOOLING_INT16_1_BATCH_SIZE));
}

void avgpooling_int16_2_arm_avgpool_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[AVGPOOLING_INT16_2_OUTPUT_C * AVGPOOLING_INT16_2_OUTPUT_W * AVGPOOLING_INT16_2_OUTPUT_H *
                   AVGPOOLING_INT16_2_BATCH_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_data = avgpooling_int16_2_input_tensor;

    input_dims.n = AVGPOOLING_INT16_2_BATCH_SIZE;
    input_dims.w = AVGPOOLING_INT16_2_INPUT_W;
    input_dims.h = AVGPOOLING_INT16_2_INPUT_H;
    input_dims.c = AVGPOOLING_INT16_2_INPUT_C;
    filter_dims.w = AVGPOOLING_INT16_2_FILTER_W;
    filter_dims.h = AVGPOOLING_INT16_2_FILTER_H;
    output_dims.w = AVGPOOLING_INT16_2_OUTPUT_W;
    output_dims.h = AVGPOOLING_INT16_2_OUTPUT_H;
    output_dims.c = AVGPOOLING_INT16_2_INPUT_C;

    pool_params.padding.w = AVGPOOLING_INT16_2_PADDING_W;
    pool_params.padding.h = AVGPOOLING_INT16_2_PADDING_H;
    pool_params.stride.w = AVGPOOLING_INT16_2_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_INT16_2_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_INT16_2_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_INT16_2_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_2_OUTPUT_W, AVGPOOLING_INT16_2_INPUT_C);
    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result =
        arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                                  avgpooling_int16_2_output,
                                  AVGPOOLING_INT16_2_OUTPUT_C * AVGPOOLING_INT16_2_OUTPUT_W *
                                      AVGPOOLING_INT16_2_OUTPUT_H * AVGPOOLING_INT16_2_BATCH_SIZE));
}

void avgpooling_int16_3_arm_avgpool_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[AVGPOOLING_INT16_3_OUTPUT_C * AVGPOOLING_INT16_3_OUTPUT_W * AVGPOOLING_INT16_3_OUTPUT_H *
                   AVGPOOLING_INT16_3_BATCH_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_data = avgpooling_int16_3_input_tensor;

    input_dims.n = AVGPOOLING_INT16_3_BATCH_SIZE;
    input_dims.w = AVGPOOLING_INT16_3_INPUT_W;
    input_dims.h = AVGPOOLING_INT16_3_INPUT_H;
    input_dims.c = AVGPOOLING_INT16_3_INPUT_C;
    filter_dims.w = AVGPOOLING_INT16_3_FILTER_W;
    filter_dims.h = AVGPOOLING_INT16_3_FILTER_H;
    output_dims.w = AVGPOOLING_INT16_3_OUTPUT_W;
    output_dims.h = AVGPOOLING_INT16_3_OUTPUT_H;
    output_dims.c = AVGPOOLING_INT16_3_INPUT_C;

    pool_params.padding.w = AVGPOOLING_INT16_3_PADDING_W;
    pool_params.padding.h = AVGPOOLING_INT16_3_PADDING_H;
    pool_params.stride.w = AVGPOOLING_INT16_3_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_INT16_3_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_INT16_3_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_INT16_3_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_3_OUTPUT_W, AVGPOOLING_INT16_3_INPUT_C);
    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result =
        arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output,
                                  avgpooling_int16_3_output,
                                  AVGPOOLING_INT16_3_OUTPUT_C * AVGPOOLING_INT16_3_OUTPUT_W *
                                      AVGPOOLING_INT16_3_OUTPUT_H * AVGPOOLING_INT16_3_BATCH_SIZE));
}

void buffer_size_mve_arm_avgpool_s16(void)
{
#if defined(ARM_MATH_MVEI)
    const int32_t buf_size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_3_OUTPUT_W, AVGPOOLING_INT16_3_INPUT_C);
    const int32_t mve_buf_size =
        arm_avgpool_s16_get_buffer_size_mve(AVGPOOLING_INT16_3_OUTPUT_W, AVGPOOLING_INT16_3_INPUT_C);

    TEST_ASSERT_EQUAL(buf_size, mve_buf_size);
#endif
}

void buffer_size_dsp_arm_avgpool_s16(void)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    const int32_t buf_size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_3_OUTPUT_W, AVGPOOLING_INT16_3_INPUT_C);
    const int32_t dsp_buf_size =
        arm_avgpool_s16_get_buffer_size_dsp(AVGPOOLING_INT16_3_OUTPUT_W, AVGPOOLING_INT16_3_INPUT_C);

    TEST_ASSERT_EQUAL(buf_size, dsp_buf_size);
#endif
}

void avgpooling_int16_param_fail_arm_avgpool_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_ARG_ERROR;
    int16_t output[AVGPOOLING_INT16_3_OUTPUT_C * AVGPOOLING_INT16_3_OUTPUT_W * AVGPOOLING_INT16_3_OUTPUT_H *
                   AVGPOOLING_INT16_3_BATCH_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_data = avgpooling_int16_3_input_tensor;

    input_dims.n = 0;
    input_dims.w = AVGPOOLING_INT16_3_INPUT_W;
    input_dims.h = AVGPOOLING_INT16_3_INPUT_H;
    input_dims.c = AVGPOOLING_INT16_3_INPUT_C;
    filter_dims.w = AVGPOOLING_INT16_3_FILTER_W;
    filter_dims.h = AVGPOOLING_INT16_3_FILTER_H;
    output_dims.w = AVGPOOLING_INT16_3_OUTPUT_W;
    output_dims.h = AVGPOOLING_INT16_3_OUTPUT_H;
    output_dims.c = AVGPOOLING_INT16_3_INPUT_C;

    pool_params.padding.w = AVGPOOLING_INT16_3_PADDING_W;
    pool_params.padding.h = AVGPOOLING_INT16_3_PADDING_H;
    pool_params.stride.w = AVGPOOLING_INT16_3_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_INT16_3_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_INT16_3_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_INT16_3_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_3_OUTPUT_W, AVGPOOLING_INT16_3_INPUT_C);
    ctx.buf = malloc(ctx.size);

    arm_cmsis_nn_status result =
        arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    TEST_ASSERT_EQUAL(expected, result);
}

// See the s8 twin: issue #318. The _mve leg is public and directly callable, so it has to answer an out-of-range
// channel count with the dispatcher's -1 rather than a 0. Not gated on ARM_MATH_MVEI - the leg variants compile on
// every target.
void buffer_size_out_of_range_mve_arm_avgpool_s16(void)
{
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size_mve(0, -1));
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size_mve(0, -7));
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size_mve(3, -1));
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size_mve(2, 2147483647));
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size_mve(2, 1073741823));
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size(0, -1));
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size_dsp(0, -1));
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size(2, 1073741823));
    TEST_ASSERT_EQUAL(-1, arm_avgpool_s16_get_buffer_size_dsp(2, 1073741823));
    TEST_ASSERT_EQUAL(0, arm_avgpool_s16_get_buffer_size_mve(0, 0));
}

// Issue #623: arm_avgpool_s16 must return ARM_CMSIS_NN_ARG_ERROR without modifying any destination bytes
// when an output position has an empty pooling window (tested for empty-X, empty-Y, and both, with 9 channels
// to exercise the MVE tail path).
void avgpooling_empty_window_arm_avgpool_s16(void)
{
    const int16_t input_data[2 * 2 * 9] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
    int16_t output[3 * 3 * 9];

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    input_dims.n = 1;
    input_dims.w = 2;
    input_dims.h = 2;
    input_dims.c = 9;

    filter_dims.w = 1;
    filter_dims.h = 1;

    pool_params.padding.w = 0;
    pool_params.padding.h = 0;
    pool_params.stride.w = 1;
    pool_params.stride.h = 1;

    pool_params.activation.min = -32768;
    pool_params.activation.max = 32767;

    // Case 1: Issue #623 reproducer (both X and Y empty at (2, 2)).
    {
        memset(output, 0x55, sizeof(output));
        output_dims.w = 3;
        output_dims.h = 3;
        output_dims.c = 9;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_HEX8(0x55, ((uint8_t *)output)[i]);
        }
    }

    // Case 2: Empty X intersection with valid Y (output_w = 3, output_h = 2).
    {
        memset(output, 0x55, sizeof(output));
        output_dims.w = 3;
        output_dims.h = 2;
        output_dims.c = 9;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_HEX8(0x55, ((uint8_t *)output)[i]);
        }
    }

    // Case 3: Empty Y intersection with valid X (output_w = 2, output_h = 3).
    {
        memset(output, 0x55, sizeof(output));
        output_dims.w = 2;
        output_dims.h = 3;
        output_dims.c = 9;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_HEX8(0x55, ((uint8_t *)output)[i]);
        }
    }

    // Case 4: Extreme integer-overflow bound leading to an empty window at i_x = 3.
    // Preflight validation must reject it upfront without any output writes.
    {
        memset(output, 0x55, sizeof(output));
        input_dims.w = 2;
        input_dims.h = 1;
        input_dims.c = 9;
        filter_dims.w = INT32_MAX;
        filter_dims.h = 1;
        pool_params.padding.w = 1;
        pool_params.padding.h = 0;
        pool_params.stride.w = 1;
        pool_params.stride.h = 1;
        output_dims.w = 4;
        output_dims.h = 1;
        output_dims.c = 9;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_HEX8(0x55, ((uint8_t *)output)[i]);
        }
    }

    // Case 5: Extreme integer-overflow bound leading to an empty window at i_y = 3.
    {
        memset(output, 0x55, sizeof(output));
        input_dims.w = 1;
        input_dims.h = 2;
        input_dims.c = 9;
        filter_dims.w = 1;
        filter_dims.h = INT32_MAX;
        pool_params.padding.w = 0;
        pool_params.padding.h = 1;
        pool_params.stride.w = 1;
        pool_params.stride.h = 1;
        output_dims.w = 1;
        output_dims.h = 4;
        output_dims.c = 9;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, result);
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_HEX8(0x55, ((uint8_t *)output)[i]);
        }
    }

    // Case 6: Extreme integer-overflow bound with valid windows in X across all output positions.
    // Widened int64_t calculations must process it successfully.
    {
        const int16_t test_input[2] = {100, 200};
        const int16_t expected_output[3] = {150, 150, 200};
        int16_t out_buf[3];
        memset(out_buf, 0x55, sizeof(out_buf));
        input_dims.w = 2;
        input_dims.h = 1;
        input_dims.c = 1;
        filter_dims.w = INT32_MAX;
        filter_dims.h = 1;
        pool_params.padding.w = 1;
        pool_params.padding.h = 0;
        pool_params.stride.w = 1;
        pool_params.stride.h = 1;
        output_dims.w = 3;
        output_dims.h = 1;
        output_dims.c = 1;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, test_input, &filter_dims, &output_dims, out_buf);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        for (int i = 0; i < 3; i++)
        {
            TEST_ASSERT_EQUAL_INT16(expected_output[i], out_buf[i]);
        }
    }

    // Case 7: Extreme integer-overflow bound with valid windows in Y across all output positions.
    {
        const int16_t test_input[2] = {100, 200};
        const int16_t expected_output[3] = {150, 150, 200};
        int16_t out_buf[3];
        memset(out_buf, 0x55, sizeof(out_buf));
        input_dims.w = 1;
        input_dims.h = 2;
        input_dims.c = 1;
        filter_dims.w = 1;
        filter_dims.h = INT32_MAX;
        pool_params.padding.w = 0;
        pool_params.padding.h = 1;
        pool_params.stride.w = 1;
        pool_params.stride.h = 1;
        output_dims.w = 1;
        output_dims.h = 3;
        output_dims.c = 1;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, test_input, &filter_dims, &output_dims, out_buf);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        for (int i = 0; i < 3; i++)
        {
            TEST_ASSERT_EQUAL_INT16(expected_output[i], out_buf[i]);
        }
    }
}

void avgpooling_asymmetric_arm_avgpool_s16(void)
{
    const int32_t in_w = 5;
    const int32_t in_h = 4;
    const int32_t in_c = 9;
    const int32_t out_w = 3;
    const int32_t out_h = 3;
    const int32_t ker_w = 3;
    const int32_t ker_h = 2;
    const int32_t pad_w = 1;
    const int32_t pad_h = 0;
    const int32_t str_w = 2;
    const int32_t str_h = 1;
    const int32_t act_min = -32768;
    const int32_t act_max = 32767;

    int16_t input[5 * 4 * 9];
    int16_t output[3 * 3 * 9] = {0};
    int16_t ref_output[3 * 3 * 9] = {0};

    for (int i = 0; i < in_w * in_h * in_c; i++)
    {
        input[i] = (int16_t)((i * 37) % 500 - 250);
    }

    // In-test reference computation
    for (int i_y = 0; i_y < out_h; i_y++)
    {
        for (int i_x = 0; i_x < out_w; i_x++)
        {
            const int32_t k_y_start = ARM_NN_MAX(0, i_y * str_h - pad_h);
            const int32_t k_y_end = ARM_NN_MIN(i_y * str_h - pad_h + ker_h, in_h);
            const int32_t k_x_start = ARM_NN_MAX(0, i_x * str_w - pad_w);
            const int32_t k_x_end = ARM_NN_MIN(i_x * str_w - pad_w + ker_w, in_w);

            for (int c = 0; c < in_c; c++)
            {
                int32_t sum = 0;
                int32_t count = 0;
                for (int ky = k_y_start; ky < k_y_end; ky++)
                {
                    for (int kx = k_x_start; kx < k_x_end; kx++)
                    {
                        sum += input[c + in_c * (kx + ky * in_w)];
                        count++;
                    }
                }
                sum = sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
                sum = ARM_NN_MAX(sum, act_min);
                sum = ARM_NN_MIN(sum, act_max);
                ref_output[c + in_c * (i_x + i_y * out_w)] = (int16_t)sum;
            }
        }
    }

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims = {1, in_h, in_w, in_c};
    cmsis_nn_dims filter_dims = {0, ker_h, ker_w, 0};
    cmsis_nn_dims output_dims = {0, out_h, out_w, in_c};

    pool_params.padding.w = pad_w;
    pool_params.padding.h = pad_h;
    pool_params.stride.w = str_w;
    pool_params.stride.h = str_h;
    pool_params.activation.min = act_min;
    pool_params.activation.max = act_max;

    ctx.size = arm_avgpool_s16_get_buffer_size(out_w, in_c);
    ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

    arm_cmsis_nn_status result =
        arm_avgpool_s16(&ctx, &pool_params, &input_dims, input, &filter_dims, &output_dims, output);

    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_EQUAL_INT16_ARRAY(ref_output, output, out_w * out_h * in_c);
}

void avgpooling_zero_size_output_arm_avgpool_s16(void)
{
    const int16_t input_data[4] = {1, 2, 3, 4};
    int16_t output[4];

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims = {1, 2, 2, 1};
    cmsis_nn_dims filter_dims = {0, 1, 1, 0};
    cmsis_nn_dims output_dims;

    pool_params.padding.w = 0;
    pool_params.padding.h = 0;
    pool_params.stride.w = 1;
    pool_params.stride.h = 1;
    pool_params.activation.min = -32768;
    pool_params.activation.max = 32767;

    // Case 1: output_w = 0, output_h = 2
    {
        memset(output, 0x55, sizeof(output));
        output_dims.n = 1;
        output_dims.h = 2;
        output_dims.w = 0;
        output_dims.c = 1;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_HEX8(0x55, ((uint8_t *)output)[i]);
        }
    }

    // Case 2: output_w = 2, output_h = 0
    {
        memset(output, 0x55, sizeof(output));
        output_dims.n = 1;
        output_dims.h = 0;
        output_dims.w = 2;
        output_dims.c = 1;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_HEX8(0x55, ((uint8_t *)output)[i]);
        }
    }

    // Case 3: output_w = 0, output_h = 0
    {
        memset(output, 0x55, sizeof(output));
        output_dims.n = 1;
        output_dims.h = 0;
        output_dims.w = 0;
        output_dims.c = 1;

        ctx.size = arm_avgpool_s16_get_buffer_size(output_dims.w, input_dims.c);
        ctx.buf = ctx.size > 0 ? malloc(ctx.size) : NULL;

        arm_cmsis_nn_status result =
            arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        if (ctx.buf)
        {
            memset(ctx.buf, 0, ctx.size);
            free(ctx.buf);
        }

        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
        for (size_t i = 0; i < sizeof(output); i++)
        {
            TEST_ASSERT_EQUAL_HEX8(0x55, ((uint8_t *)output)[i]);
        }
    }
}

void avgpooling_null_ctx_arm_avgpool_s16(void)
{
    const int16_t input_data[4] = {1, 2, 3, 4};
    int16_t output[4] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims = {1, 2, 2, 1};
    cmsis_nn_dims filter_dims = {0, 1, 1, 0};
    cmsis_nn_dims output_dims = {1, 2, 2, 1};

    pool_params.padding.w = 0;
    pool_params.padding.h = 0;
    pool_params.stride.w = 1;
    pool_params.stride.h = 1;
    pool_params.activation.min = -32768;
    pool_params.activation.max = 32767;

#if !defined(ARM_MATH_MVEI)
    // Non-MVE DSP / scalar builds require valid ctx and ctx->buf when buffer is needed
    arm_cmsis_nn_status null_ctx_result =
        arm_avgpool_s16(NULL, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, null_ctx_result);
#endif

    // Negative buffer size check with non-NULL buffer sentinel
    ctx.size = -1;
    ctx.buf = (void *)1;
    input_dims.c = -1;

    arm_cmsis_nn_status neg_buf_result =
        arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, neg_buf_result);
}
