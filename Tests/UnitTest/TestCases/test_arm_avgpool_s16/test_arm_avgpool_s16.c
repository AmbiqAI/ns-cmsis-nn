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

/* Runs arm_avgpool_s16() on a layer with the scratch its sizer gives, writing into output, which the caller has filled
   with a known pattern. Returns the status. */
static arm_cmsis_nn_status avgpool_s16_run(const int32_t in_h,
                                           const int32_t in_w,
                                           const int32_t ch,
                                           const int16_t *input,
                                           const int32_t k_h,
                                           const int32_t k_w,
                                           const int32_t stride_h,
                                           const int32_t stride_w,
                                           const int32_t pad_h,
                                           const int32_t pad_w,
                                           const int32_t out_h,
                                           const int32_t out_w,
                                           int16_t *output)
{
    const cmsis_nn_dims input_dims = {1, in_h, in_w, ch};
    const cmsis_nn_dims filter_dims = {1, k_h, k_w, 1};
    const cmsis_nn_dims output_dims = {1, out_h, out_w, ch};
    const cmsis_nn_pool_params pool_params = {
        .stride = {stride_w, stride_h}, .padding = {pad_w, pad_h}, .activation = {-32768, 32767}};
    const int32_t size = arm_avgpool_s16_get_buffer_size(out_w, ch);
    TEST_ASSERT_TRUE(size >= 0);
    int32_t scratch[16];
    TEST_ASSERT_TRUE(size <= (int32_t)sizeof(scratch));
    const cmsis_nn_context ctx = {scratch, size};
    return arm_avgpool_s16(&ctx, &pool_params, &input_dims, input, &filter_dims, &output_dims, output);
}

/* An output position whose pooling window misses the input (along x, along y, or both) is rejected with
   ARM_CMSIS_NN_ARG_ERROR before anything is written. Nine channels run the MVE path's 8-lane block and its tail. */
void empty_window_arm_avgpool_s16(void)
{
    enum
    {
        ch = 9
    };
    int16_t input[2 * 2 * ch];
    int16_t output[3 * 3 * ch];
    for (int i = 0; i < (int)(sizeof(input) / sizeof(input[0])); i++)
    {
        input[i] = (int16_t)(i * 311 - 4000);
    }
    const int32_t out_hw[3][2] = {{3, 3}, {2, 3}, {3, 2}};
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < (int)(sizeof(output) / sizeof(output[0])); i++)
        {
            output[i] = 0x5555;
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          avgpool_s16_run(2, 2, ch, input, 1, 1, 1, 1, 0, 0, out_hw[c][0], out_hw[c][1], output));
        TEST_ASSERT_EACH_EQUAL_INT16(0x5555, output, sizeof(output) / sizeof(output[0]));
    }
}

/* A filter extent of INT32_MAX overflows i * stride - pad + filter in 32 bits. With padding 1 over a 2-element input,
   three output positions all overlap the input and must be computed; a fourth has an empty window and must be
   rejected with the output untouched. Checked along x and along y. */
void window_bound_overflow_arm_avgpool_s16(void)
{
    const int16_t input[2] = {10, 20};
    const int16_t expected[3] = {15, 15, 20};
    int16_t output[4];
    for (int axis = 0; axis < 2; axis++)
    {
        const int32_t in_h = axis == 0 ? 1 : 2;
        const int32_t in_w = axis == 0 ? 2 : 1;
        const int32_t k_h = axis == 0 ? 1 : INT32_MAX;
        const int32_t k_w = axis == 0 ? INT32_MAX : 1;
        const int32_t pad_h = axis == 0 ? 0 : 1;
        const int32_t pad_w = axis == 0 ? 1 : 0;
        for (int32_t n = 3; n <= 4; n++)
        {
            for (int i = 0; i < 4; i++)
            {
                output[i] = 0x5555;
            }
            const arm_cmsis_nn_status status = avgpool_s16_run(
                in_h, in_w, 1, input, k_h, k_w, 1, 1, pad_h, pad_w, axis == 0 ? 1 : n, axis == 0 ? n : 1, output);
            if (n == 3)
            {
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
                TEST_ASSERT_EQUAL_INT16_ARRAY(expected, output, 3);
                TEST_ASSERT_EQUAL_INT16(0x5555, output[3]);
            }
            else
            {
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, status);
                TEST_ASSERT_EACH_EQUAL_INT16(0x5555, output, 4);
            }
        }
    }
}

/* Different stride, padding and filter size along y and x (input 4x5, filter 3x2, stride 2x1, padding 1x0, output 2x4,
   three channels), against a reference computed here: the mean of the in-bounds taps, rounded half away from zero.
   Catches the two axes being swapped anywhere in the window bounds. */
void asymmetric_axes_arm_avgpool_s16(void)
{
    enum
    {
        in_h = 4,
        in_w = 5,
        ch = 3,
        k_h = 3,
        k_w = 2,
        stride_h = 2,
        stride_w = 1,
        pad_h = 1,
        pad_w = 0,
        out_h = 2,
        out_w = 4
    };
    int16_t input[in_h * in_w * ch];
    int16_t expected[out_h * out_w * ch];
    int16_t output[out_h * out_w * ch];
    for (int i = 0; i < in_h * in_w * ch; i++)
    {
        input[i] = (int16_t)((i * 977) % 20011 - 10005);
    }
    for (int oy = 0; oy < out_h; oy++)
    {
        for (int ox = 0; ox < out_w; ox++)
        {
            for (int c = 0; c < ch; c++)
            {
                int32_t sum = 0;
                int32_t count = 0;
                for (int ky = 0; ky < k_h; ky++)
                {
                    for (int kx = 0; kx < k_w; kx++)
                    {
                        const int y = oy * stride_h - pad_h + ky;
                        const int x = ox * stride_w - pad_w + kx;
                        if (y >= 0 && y < in_h && x >= 0 && x < in_w)
                        {
                            sum += input[(y * in_w + x) * ch + c];
                            count++;
                        }
                    }
                }
                sum = sum > 0 ? (sum + count / 2) / count : (sum - count / 2) / count;
                expected[(oy * out_w + ox) * ch + c] = (int16_t)sum;
            }
        }
    }
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        avgpool_s16_run(in_h, in_w, ch, input, k_h, k_w, stride_h, stride_w, pad_h, pad_w, out_h, out_w, output));
    TEST_ASSERT_EQUAL_INT16_ARRAY(expected, output, out_h * out_w * ch);
}

/* An output with no columns has no window: nothing is written and the call succeeds, whatever the other extent. On
   builds that use the scratch buffer, a NULL ctx is rejected before any output is written. */
void degenerate_arguments_arm_avgpool_s16(void)
{
    const int16_t input[1] = {7};
    int16_t output[2] = {0x5555, 0x5555};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, avgpool_s16_run(1, 1, 1, input, 1, 1, 1, 1, 0, 0, 2, 0, output));
    TEST_ASSERT_EACH_EQUAL_INT16(0x5555, output, 2);
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    const cmsis_nn_dims dims = {1, 1, 1, 1};
    const cmsis_nn_pool_params pool_params = {.stride = {1, 1}, .padding = {0, 0}, .activation = {-32768, 32767}};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_avgpool_s16(NULL, &pool_params, &dims, input, &dims, &dims, output));
    TEST_ASSERT_EACH_EQUAL_INT16(0x5555, output, 2);
#endif
}
