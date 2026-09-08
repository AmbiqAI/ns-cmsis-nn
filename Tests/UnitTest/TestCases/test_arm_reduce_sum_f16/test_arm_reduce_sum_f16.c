/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <math.h>
#include <string.h>
#include <unity.h>

#include "rsum_f16_data.h"

static const cmsis_nn_dims rsum_f16_input_dims = {2, 3, 4, 5};

static void rsum_f16_check(const cmsis_nn_dims *axis_dims, const cmsis_nn_dims *output_dims, const float16_t *expected)
{
    float16_t output[120] = {0};
    const int32_t out_size = output_dims->n * output_dims->h * output_dims->w * output_dims->c;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_reduce_sum_f16(rsum_f16_input, &rsum_f16_input_dims, axis_dims, output, output_dims));

    for (int i = 0; i < out_size; ++i)
    {
        // Kernel and golden both accumulate in float32; tolerance covers
        // accumulation order plus the final float16 rounding (~1 ulp of the
        // largest output) while staying below the smallest single element.
        TEST_ASSERT_FLOAT_WITHIN(1.0e-2f, (float)expected[i], (float)output[i]);
    }
}

void rsum_f16_axis_c_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 4, 1};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_c);
}

void rsum_f16_axis_hwc_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 1, 1};
    const cmsis_nn_dims output_dims = {2, 1, 1, 1};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_hwc);
}

void rsum_f16_axis_all_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_all);
}

void rsum_f16_axis_hw_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 1, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 5};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_hw);
}

void rsum_f16_axis_h_arm_reduce_sum_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {2, 1, 4, 5};
    rsum_f16_check(&axis_dims, &output_dims, rsum_f16_ref_h);
}

void rsum_f16_identity_arm_reduce_sum_f16(void)
{
    // Empty axis mask reduces nothing: bit-exact identity copy
    float16_t output[120] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_reduce_sum_f16(rsum_f16_input, &rsum_f16_input_dims, &axis_dims, output, &rsum_f16_input_dims));

    for (int i = 0; i < 120; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT((float)rsum_f16_input[i], (float)output[i]);
    }
}

void rsum_f16_size_one_dims_arm_reduce_sum_f16(void)
{
    // Non-contiguous mask (N and C) made flatten-eligible by size-1 H/W
    const float16_t input[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float16_t output[1] = {0};
    const cmsis_nn_dims input_dims = {2, 1, 1, 5};
    const cmsis_nn_dims axis_dims = {1, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f16(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL_FLOAT(55.0f, (float)output[0]);
}

void rsum_f16_nan_inf_arm_reduce_sum_f16(void)
{
    // Row sums: [1, Inf, 2] -> Inf; [Inf, -Inf, 0] -> NaN; [NaN, 1, 2] -> NaN
    const float16_t inf = (float16_t)INFINITY;
    const float16_t input[9] = {(float16_t)1.0f,
                                inf,
                                (float16_t)2.0f,
                                inf,
                                -inf,
                                (float16_t)0.0f,
                                (float16_t)NAN,
                                (float16_t)1.0f,
                                (float16_t)2.0f};
    float16_t output[3] = {0};
    const cmsis_nn_dims input_dims = {1, 1, 3, 3};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 3, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f16(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_FLOAT_IS_INF((float)output[0]);
    TEST_ASSERT_FLOAT_IS_NAN((float)output[1]);
    TEST_ASSERT_FLOAT_IS_NAN((float)output[2]);
}

void rsum_f16_arg_error_arm_reduce_sum_f16(void)
{
    float16_t output[8] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 4, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f16(NULL, &rsum_f16_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f16(rsum_f16_input, NULL, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f16(rsum_f16_input, &rsum_f16_input_dims, NULL, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f16(rsum_f16_input, &rsum_f16_input_dims, &axis_dims, NULL, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f16(rsum_f16_input, &rsum_f16_input_dims, &axis_dims, output, NULL));
}

void rsum_f16_portable_masks_arm_reduce_sum_f16(void)
{
    // LiteRT BUILTIN_REF, FP32 accumulation. Refs #484.
    static const float16_t pattern[7] = {2048, 1, -2048, 1, 2, -2, 0};
    float16_t input[210];
    for (int i = 0; i < 210; ++i)
    {
        input[i] = pattern[i % 7];
    }
    static const float16_t expected0[] = {0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f,
                                          0x1.8000000000000p+2f};
    static const float16_t expected1[] = {0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f,
                                          0x1.0000000000000p+2f};
    static const float16_t expected2[] = {0x1.4000000000000p+14f,
                                          0x1.4000000000000p+3f,
                                          -0x1.4000000000000p+14f,
                                          0x1.4000000000000p+3f,
                                          0x1.4000000000000p+4f,
                                          -0x1.4000000000000p+4f,
                                          0x0.0p+0f,
                                          0x1.4000000000000p+14f,
                                          0x1.4000000000000p+3f,
                                          -0x1.4000000000000p+14f,
                                          0x1.4000000000000p+3f,
                                          0x1.4000000000000p+4f,
                                          -0x1.4000000000000p+4f,
                                          0x0.0p+0f,
                                          0x1.4000000000000p+14f,
                                          0x1.4000000000000p+3f,
                                          -0x1.4000000000000p+14f,
                                          0x1.4000000000000p+3f,
                                          0x1.4000000000000p+4f,
                                          -0x1.4000000000000p+4f,
                                          0x0.0p+0f};
    static const float16_t expected3[] = {0x1.e000000000000p+14f,
                                          0x1.e000000000000p+3f,
                                          -0x1.e000000000000p+14f,
                                          0x1.e000000000000p+3f,
                                          0x1.e000000000000p+4f,
                                          -0x1.e000000000000p+4f,
                                          0x0.0p+0f,
                                          0x1.e000000000000p+14f,
                                          0x1.e000000000000p+3f,
                                          -0x1.e000000000000p+14f,
                                          0x1.e000000000000p+3f,
                                          0x1.e000000000000p+4f,
                                          -0x1.e000000000000p+4f,
                                          0x0.0p+0f};
    const cmsis_nn_dims in = {2, 3, 5, 7};
    float16_t output[107];
    {
        const cmsis_nn_dims axes = {0, 1, 0, 1};
        const cmsis_nn_dims out = {2, 1, 5, 1};
        memset(output, 0xa5, sizeof(output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f16(input, &in, &axes, output + 1, &out));
        TEST_ASSERT_EQUAL_MEMORY(expected0, output + 1, sizeof(expected0));
        const unsigned char *bytes = (const unsigned char *)output;
        for (size_t b = 0; b < sizeof(output[0]); ++b)
        {
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[b]);
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[(10 + 1) * sizeof(output[0]) + b]);
        }
    }
    {
        const cmsis_nn_dims axes = {1, 0, 0, 1};
        const cmsis_nn_dims out = {1, 3, 5, 1};
        memset(output, 0xa5, sizeof(output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f16(input, &in, &axes, output + 1, &out));
        TEST_ASSERT_EQUAL_MEMORY(expected1, output + 1, sizeof(expected1));
        const unsigned char *bytes = (const unsigned char *)output;
        for (size_t b = 0; b < sizeof(output[0]); ++b)
        {
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[b]);
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[(15 + 1) * sizeof(output[0]) + b]);
        }
    }
    {
        const cmsis_nn_dims axes = {1, 0, 1, 0};
        const cmsis_nn_dims out = {1, 3, 1, 7};
        memset(output, 0xa5, sizeof(output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f16(input, &in, &axes, output + 1, &out));
        TEST_ASSERT_EQUAL_MEMORY(expected2, output + 1, sizeof(expected2));
        const unsigned char *bytes = (const unsigned char *)output;
        for (size_t b = 0; b < sizeof(output[0]); ++b)
        {
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[b]);
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[(21 + 1) * sizeof(output[0]) + b]);
        }
    }
    {
        const cmsis_nn_dims axes = {0, 1, 1, 0};
        const cmsis_nn_dims out = {2, 1, 1, 7};
        memset(output, 0xa5, sizeof(output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f16(input, &in, &axes, output + 1, &out));
        TEST_ASSERT_EQUAL_MEMORY(expected3, output + 1, sizeof(expected3));
        const unsigned char *bytes = (const unsigned char *)output;
        for (size_t b = 0; b < sizeof(output[0]); ++b)
        {
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[b]);
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[(14 + 1) * sizeof(output[0]) + b]);
        }
    }
}

void rsum_f16_consumer_spatial_layouts_arm_reduce_sum_f16(void)
{
    // LiteRT BUILTIN_REF fixture for the two consumer layouts. Refs #484.
    static float16_t input[64 * 128];
    static const float16_t expected[7] = {-12.125f, -8.125f, -4.125f, -0.125f, 3.875f, 7.875f, 11.875f};
    const cmsis_nn_dims inputs[2] = {{1, 64, 128, 1}, {1, 1, 64, 128}};
    const cmsis_nn_dims axes[2] = {{0, 1, 0, 0}, {0, 0, 1, 0}};
    const cmsis_nn_dims outputs[2] = {{1, 1, 128, 1}, {1, 1, 1, 128}};
    float16_t output[130];
    int modes = 1;
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    uint32_t saved_fpscr;
    __ASM volatile("vmrs %0, fpscr" : "=r"(saved_fpscr));
    modes = 2;
#endif
    for (int r = 0; r < 64; ++r)
    {
        for (int c = 0; c < 128; ++c)
        {
            input[r * 128 + c] = (float16_t)((r % 5) - 2 + (c % 7) - 3) / 16.0f;
        }
    }
    for (int mode = 0; mode < modes; ++mode)
    {
        for (int layout = 0; layout < 2; ++layout)
        {
            output[0] = output[129] = 1234.0f;
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            const uint32_t fpscr = (saved_fpscr & ~((1u << 24) | (3u << 22))) | ((uint32_t)mode << 24);
            __ASM volatile("vmsr fpscr, %0" : : "r"(fpscr));
#endif
            const arm_cmsis_nn_status status =
                arm_reduce_sum_f16(input, &inputs[layout], &axes[layout], output + 1, &outputs[layout]);
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            __ASM volatile("vmsr fpscr, %0" : : "r"(saved_fpscr));
#endif
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
            for (int c = 0; c < 128; ++c)
            {
                TEST_ASSERT_EQUAL_MEMORY(&expected[c % 7], &output[c + 1], sizeof(float16_t));
            }
            TEST_ASSERT_EQUAL_FLOAT(1234.0f, output[0]);
            TEST_ASSERT_EQUAL_FLOAT(1234.0f, output[129]);
        }
    }
}

void rsum_f16_legacy_shapes_arm_reduce_sum_f16(void)
{
    // Preserve defined legacy SUM behavior. Refs #484.
    const float16_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const cmsis_nn_dims inputs[5] = {{1, 0, 2, 2}, {1, -1, 2, 2}, {2, 65536, 65536, 1}, {1, 2, 2, 2}, {1, 1, 2, 2}};
    const cmsis_nn_dims axes[5] = {{0, 1, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};
    const cmsis_nn_dims outputs[5] = {{1, 1, 2, 2}, {1, 1, 2, 2}, {0, 1, 1, 1}, {1, 1, 1, 1}, {0, 1, 2, 1}};
    const float16_t expected[5][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {91, 91, 91, 91}, {6, 91, 91, 91}, {3, 7, 91, 91}};
    for (int i = 0; i < 5; ++i)
    {
        float16_t output[5] = {91, 91, 91, 91, 91};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f16(input, &inputs[i], &axes[i], output, &outputs[i]));
        TEST_ASSERT_EQUAL_MEMORY(expected[i], output, sizeof(expected[i]));
        TEST_ASSERT_EQUAL_FLOAT(91.0f, (float)output[4]);
    }
}

void rsum_f16_spatial_order_arm_reduce_sum_f16(void)
{
    // LiteRT BUILTIN_REF cancellation fixture. Refs #484.
    const float16_t pattern[4] = {2048, 1, -2048, 1};
    for (int channels = 1; channels <= 3; channels += 2)
    {
        float16_t input[24], output[5];
        const cmsis_nn_dims in = {1, 2, 4, channels};
        const cmsis_nn_dims axes = {0, 1, 1, 0};
        const cmsis_nn_dims out = {1, 1, 1, channels};
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < channels; ++c)
                input[r * channels + c] = pattern[r % 4];
        output[0] = output[channels + 1] = 91;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f16(input, &in, &axes, output + 1, &out));
        for (int c = 0; c < channels; ++c)
            TEST_ASSERT_EQUAL_FLOAT(4.0f, (float)output[c + 1]);
        TEST_ASSERT_EQUAL_FLOAT(91.0f, (float)output[0]);
        TEST_ASSERT_EQUAL_FLOAT(91.0f, (float)output[channels + 1]);
    }
}
