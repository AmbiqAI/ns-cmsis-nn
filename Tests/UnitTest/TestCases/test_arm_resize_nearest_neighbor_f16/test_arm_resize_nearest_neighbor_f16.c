/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <unity.h>

static void assert_f16_array_equal(const float16_t *expected, const float16_t *actual, int32_t length)
{
    for (int32_t index = 0; index < length; ++index)
    {
        TEST_ASSERT_FLOAT_WITHIN(1.0e-3f, (float)expected[index], (float)actual[index]);
    }
}

void resize_nearest_neighbor_f16_width_double(void)
{
    const cmsis_nn_resize_params params = {.align_corners = false, .half_pixel_centers = false};
    const cmsis_nn_dims input_dims = {1, 2, 3, 2};
    const cmsis_nn_dims output_size_dims = {1, 1, 1, 2};
    const int32_t output_size[] = {2, 6};
    const cmsis_nn_dims output_dims = {1, 2, 6, 2};
    const float16_t input[] = {(float16_t)1.0f,
                               (float16_t)2.0f,
                               (float16_t)3.0f,
                               (float16_t)4.0f,
                               (float16_t)5.0f,
                               (float16_t)6.0f,
                               (float16_t)7.0f,
                               (float16_t)8.0f,
                               (float16_t)9.0f,
                               (float16_t)10.0f,
                               (float16_t)11.0f,
                               (float16_t)12.0f};
    const float16_t expected[] = {
        (float16_t)1.0f, (float16_t)2.0f,  (float16_t)1.0f,  (float16_t)2.0f,  (float16_t)3.0f,  (float16_t)4.0f,
        (float16_t)3.0f, (float16_t)4.0f,  (float16_t)5.0f,  (float16_t)6.0f,  (float16_t)5.0f,  (float16_t)6.0f,
        (float16_t)7.0f, (float16_t)8.0f,  (float16_t)7.0f,  (float16_t)8.0f,  (float16_t)9.0f,  (float16_t)10.0f,
        (float16_t)9.0f, (float16_t)10.0f, (float16_t)11.0f, (float16_t)12.0f, (float16_t)11.0f, (float16_t)12.0f};
    float16_t output[24] = {0};
    int32_t scratch[8] = {0};
    const cmsis_nn_context ctx = {.buf = scratch, .size = sizeof(scratch)};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_resize_nearest_neighbor_f16(
                          &ctx, &params, &input_dims, input, &output_size_dims, output_size, &output_dims, output));
    assert_f16_array_equal(expected, output, 24);
}

void resize_nearest_neighbor_f16_half_pixel_downsample(void)
{
    const cmsis_nn_resize_params params = {.align_corners = false, .half_pixel_centers = true};
    const cmsis_nn_dims input_dims = {1, 1, 4, 1};
    const cmsis_nn_dims output_size_dims = {1, 1, 1, 2};
    const int32_t output_size[] = {1, 2};
    const cmsis_nn_dims output_dims = {1, 1, 2, 1};
    const float16_t input[] = {(float16_t)1.0f, (float16_t)2.0f, (float16_t)3.0f, (float16_t)4.0f};
    const float16_t expected[] = {(float16_t)2.0f, (float16_t)4.0f};
    float16_t output[2] = {0};
    int32_t scratch[3] = {0};
    const cmsis_nn_context ctx = {.buf = scratch, .size = sizeof(scratch)};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_resize_nearest_neighbor_f16(
                          &ctx, &params, &input_dims, input, &output_size_dims, output_size, &output_dims, output));
    assert_f16_array_equal(expected, output, 2);
}

void resize_nearest_neighbor_f16_rejects_small_buffer(void)
{
    const cmsis_nn_resize_params params = {.align_corners = false, .half_pixel_centers = false};
    const cmsis_nn_dims input_dims = {1, 1, 2, 1};
    const cmsis_nn_dims output_size_dims = {1, 1, 1, 2};
    const int32_t output_size[] = {1, 4};
    const cmsis_nn_dims output_dims = {1, 1, 4, 1};
    const float16_t input[] = {(float16_t)1.0f, (float16_t)2.0f};
    float16_t output[4] = {0};
    int32_t scratch[4] = {0};
    const cmsis_nn_context ctx = {.buf = scratch, .size = sizeof(scratch)};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_resize_nearest_neighbor_f16(
                          &ctx, &params, &input_dims, input, &output_size_dims, output_size, &output_dims, output));
}
