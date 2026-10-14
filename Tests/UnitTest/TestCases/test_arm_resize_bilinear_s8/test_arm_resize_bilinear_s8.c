/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include "unity.h"
#include <stdlib.h>

/* Unreduced coordinate arithmetic from the generated AOT operator, independent of kernel helpers. */
static int8_t reference_pixel(const int8_t *input,
                              const cmsis_nn_dims *dims,
                              const cmsis_nn_dims *output,
                              const cmsis_nn_resize_params *params,
                              int b,
                              int y,
                              int x,
                              int c)
{
    int64_t yn = (int64_t)y * dims->h, yd = output->h;
    int64_t xn = (int64_t)x * dims->w, xd = output->w;
    if (params->half_pixel_centers)
    {
        yn = (2LL * y + 1) * dims->h - output->h;
        xn = (2LL * x + 1) * dims->w - output->w;
        yd *= 2;
        xd *= 2;
    }
    if (params->align_corners)
    {
        yn = (int64_t)y * (dims->h - 1);
        xn = (int64_t)x * (dims->w - 1);
        yd = output->h > 1 ? output->h - 1 : 1;
        xd = output->w > 1 ? output->w - 1 : 1;
    }
    if (yn < 0)
        yn = 0;
    if (xn < 0)
        xn = 0;
    if (yn > (dims->h - 1) * yd)
        yn = (dims->h - 1) * yd;
    if (xn > (dims->w - 1) * xd)
        xn = (dims->w - 1) * xd;
    int y0 = (int)(yn / yd), x0 = (int)(xn / xd);
    int y1 = y0 + (y0 < dims->h - 1), x1 = x0 + (x0 < dims->w - 1);
    int64_t ry = yn % yd, rx = xn % xd;
    const int8_t *p = input + b * dims->h * dims->w * dims->c + c;
    int64_t value = p[(y0 * dims->w + x0) * dims->c] * (yd - ry) * (xd - rx) +
        p[(y0 * dims->w + x1) * dims->c] * (yd - ry) * rx + p[(y1 * dims->w + x0) * dims->c] * ry * (xd - rx) +
        p[(y1 * dims->w + x1) * dims->c] * ry * rx;
    int64_t denominator = yd * xd;
    return (int8_t)((value + (value < 0 ? -(denominator / 2) : denominator / 2)) / denominator);
}

static void
check_case(cmsis_nn_dims input_dims, cmsis_nn_dims output_dims, cmsis_nn_resize_params params, int specialized)
{
    const int input_count = input_dims.n * input_dims.h * input_dims.w * input_dims.c;
    const int output_count = output_dims.n * output_dims.h * output_dims.w * output_dims.c;
    int8_t *input = malloc((size_t)input_count);
    int8_t *expected = malloc((size_t)output_count);
    int8_t *storage = malloc((size_t)output_count + 2);
    TEST_ASSERT_NOT_NULL(input);
    TEST_ASSERT_NOT_NULL(expected);
    TEST_ASSERT_NOT_NULL(storage);
    int8_t *actual = storage + 1; /* Deliberately unaligned output, with both store guards. */
    uint32_t state = 814;
    for (int i = 0; i < input_count; i++)
    {
        state = 1664525U * state + 1013904223U;
        input[i] = (int8_t)((int32_t)(state >> 24) - 128);
    }
    int index = 0;
    for (int b = 0; b < output_dims.n; b++)
        for (int y = 0; y < output_dims.h; y++)
            for (int x = 0; x < output_dims.w; x++)
                for (int c = 0; c < output_dims.c; c++)
                    expected[index++] = reference_pixel(input, &input_dims, &output_dims, &params, b, y, x, c);
    storage[0] = 73;
    storage[output_count + 1] = 81;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_resize_bilinear_s8(&params, &input_dims, input, &output_dims, actual));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, actual, output_count);
    TEST_ASSERT_EQUAL(73, storage[0]);
    TEST_ASSERT_EQUAL(81, storage[output_count + 1]);
    if (specialized)
    {
        memset(actual, 0x55, (size_t)output_count);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_resize_bilinear_s8_pow2(&params, &input_dims, input, &output_dims, actual));
        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, actual, output_count);
        TEST_ASSERT_EQUAL(73, storage[0]);
        TEST_ASSERT_EQUAL(81, storage[output_count + 1]);
    }
    free(storage);
    free(expected);
    free(input);
}

void resize_bilinear_shapes(void)
{
    const int channels[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 64, 65};
    const cmsis_nn_resize_params half = {.align_corners = 0, .half_pixel_centers = 1};
    for (unsigned i = 0; i < sizeof(channels) / sizeof(channels[0]); i++)
    {
        int c = channels[i];
        check_case((cmsis_nn_dims){2, 3, 5, c}, (cmsis_nn_dims){2, 6, 40, c}, half, 1);
        check_case((cmsis_nn_dims){1, 1, 3, c}, (cmsis_nn_dims){1, 8, 12, c}, half, 1);
        check_case((cmsis_nn_dims){1, 3, 1, c}, (cmsis_nn_dims){1, 3, 2, c}, half, 1);
        check_case((cmsis_nn_dims){1, 3, 5, c}, (cmsis_nn_dims){1, 3, 5, c}, half, 1);
        for (int mode = 0; mode < 3; mode++)
        {
            cmsis_nn_resize_params p = {.align_corners = mode == 2, .half_pixel_centers = mode == 1};
            check_case((cmsis_nn_dims){1, 3, 5, c}, (cmsis_nn_dims){1, 7, 9, c}, p, 0);
            check_case((cmsis_nn_dims){1, 7, 9, c}, (cmsis_nn_dims){1, 2, 3, c}, p, 0);
            check_case((cmsis_nn_dims){1, 7, 9, c}, (cmsis_nn_dims){1, 1, 1, c}, p, 0);
        }
    }
}

void resize_bilinear_measured_shapes(void)
{
    const cmsis_nn_resize_params p = {.align_corners = 0, .half_pixel_centers = 1};
    check_case((cmsis_nn_dims){1, 12, 12, 64}, (cmsis_nn_dims){1, 96, 96, 64}, p, 1);
    check_case((cmsis_nn_dims){1, 96, 96, 2}, (cmsis_nn_dims){1, 192, 192, 2}, p, 1);
}

void resize_bilinear_signed_ties(void)
{
    const cmsis_nn_resize_params p = {.align_corners = 0, .half_pixel_centers = 1};
    const cmsis_nn_dims input_dims = {1, 1, 2, 2}, output_dims = {1, 1, 4, 2};
    const int8_t input[] = {-2, 2, 0, 0};
    const int8_t expected[] = {-2, 2, -2, 2, -1, 1, 0, 0};
    int8_t output[8];
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_resize_bilinear_s8(&p, &input_dims, input, &output_dims, output));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, 8);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_resize_bilinear_s8_pow2(&p, &input_dims, input, &output_dims, output));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, 8);
}

void resize_bilinear_invalid(void)
{
    const cmsis_nn_resize_params p = {.align_corners = 0, .half_pixel_centers = 1};
    const cmsis_nn_dims valid = {1, 2, 2, 1};
    const cmsis_nn_dims invalid[] = {{0, 2, 2, 1},
                                     {1, -1, 2, 1},
                                     {1, 2, 0, 1},
                                     {1, 2, 2, -1},
                                     {INT32_MAX, 2, 2, 1},
                                     {1, INT32_MAX, 2, 1},
                                     {1, 2, 2, INT32_MAX},
                                     {2, 2, 2, 1},
                                     {1, 2, 2, 2}};
    int8_t input[4] = {0}, output[4] = {41, 42, 43, 44};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8(&p, &valid, input, &invalid[i], output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8_pow2(&p, &valid, input, &invalid[i], output));
    }
    const cmsis_nn_dims nonpow = {1, 6, 6, 1}, down = {1, 1, 1, 1}, excessive = {1, 4096, 4096, 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8_pow2(&p, &valid, input, &nonpow, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8_pow2(&p, &valid, input, &down, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8_pow2(&p, &valid, input, &excessive, output));
    const cmsis_nn_resize_params both = {.align_corners = 1, .half_pixel_centers = 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8(&both, &valid, input, &valid, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8_pow2(&both, &valid, input, &valid, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8(NULL, &valid, input, &valid, output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_resize_bilinear_s8_pow2(&p, NULL, input, &valid, output));
    const int8_t untouched[] = {41, 42, 43, 44};
    TEST_ASSERT_EQUAL_INT8_ARRAY(untouched, output, 4);
}
