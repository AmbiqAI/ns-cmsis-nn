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

#include "rsum_f32_data.h"

static const cmsis_nn_dims rsum_f32_input_dims = {2, 3, 4, 5};

static void rsum_f32_check(const cmsis_nn_dims *axis_dims, const cmsis_nn_dims *output_dims, const float32_t *expected)
{
    float32_t output[120] = {0};
    const int32_t out_size = output_dims->n * output_dims->h * output_dims->w * output_dims->c;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_reduce_sum_f32(rsum_f32_input, &rsum_f32_input_dims, axis_dims, output, output_dims));

    for (int i = 0; i < out_size; ++i)
    {
        // Accumulation-order tolerance: scalar vs MVE partial sums
        TEST_ASSERT_FLOAT_WITHIN(1.0e-4f, expected[i], output[i]);
    }
}

void rsum_f32_axis_c_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 4, 1};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_c);
}

void rsum_f32_axis_hwc_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 1, 1};
    const cmsis_nn_dims output_dims = {2, 1, 1, 1};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_hwc);
}

void rsum_f32_axis_all_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_all);
}

void rsum_f32_axis_hw_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 1, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 5};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_hw);
}

void rsum_f32_axis_h_arm_reduce_sum_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {2, 1, 4, 5};
    rsum_f32_check(&axis_dims, &output_dims, rsum_f32_ref_h);
}

void rsum_f32_identity_arm_reduce_sum_f32(void)
{
    // Empty axis mask reduces nothing: bit-exact identity copy
    float32_t output[120] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 0};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_reduce_sum_f32(rsum_f32_input, &rsum_f32_input_dims, &axis_dims, output, &rsum_f32_input_dims));

    for (int i = 0; i < 120; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT(rsum_f32_input[i], output[i]);
    }
}

void rsum_f32_size_one_dims_arm_reduce_sum_f32(void)
{
    // Non-contiguous mask (N and C) made flatten-eligible by size-1 H/W
    const float32_t input[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float32_t output[1] = {0};
    const cmsis_nn_dims input_dims = {2, 1, 1, 5};
    const cmsis_nn_dims axis_dims = {1, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL_FLOAT(55.0f, output[0]);
}

void rsum_f32_nan_inf_arm_reduce_sum_f32(void)
{
    // Row sums: [1, Inf, 2] -> Inf; [Inf, -Inf, 0] -> NaN; [NaN, 1, 2] -> NaN
    const float32_t inf = (float32_t)INFINITY;
    const float32_t input[9] = {1.0f, inf, 2.0f, inf, -inf, 0.0f, (float32_t)NAN, 1.0f, 2.0f};
    float32_t output[3] = {0};
    const cmsis_nn_dims input_dims = {1, 1, 3, 3};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 3, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_FLOAT_IS_INF(output[0]);
    TEST_ASSERT_FLOAT_IS_NAN(output[1]);
    TEST_ASSERT_FLOAT_IS_NAN(output[2]);
}

void rsum_f32_arg_error_arm_reduce_sum_f32(void)
{
    float32_t output[8] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 4, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f32(NULL, &rsum_f32_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f32(rsum_f32_input, NULL, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f32(rsum_f32_input, &rsum_f32_input_dims, NULL, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f32(rsum_f32_input, &rsum_f32_input_dims, &axis_dims, NULL, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_reduce_sum_f32(rsum_f32_input, &rsum_f32_input_dims, &axis_dims, output, NULL));
}

void rsum_f32_portable_masks_arm_reduce_sum_f32(void)
{
    // LiteRT BUILTIN_REF, FP32 accumulation. Refs #484.
    static const float32_t input[210] = {
        -0x1.b5d62c0000000p-2f, 0x1.7a323c0000000p-1f,  0x1.fa24980000000p-1f,  0x1.55d4140000000p-2f,
        -0x1.684a300000000p-1f, 0x1.b8b2e00000000p-1f,  -0x1.f3d9e40000000p-7f, 0x1.7a66a40000000p-1f,
        0x1.805e980000000p-1f,  -0x1.92f3920000000p-2f, 0x1.2dcdaa0000000p-1f,  0x1.f802420000000p-1f,
        -0x1.6703640000000p-1f, 0x1.2f18de0000000p-1f,  -0x1.1177480000000p-1f, 0x1.fe21440000000p-4f,
        0x1.5e8a740000000p-2f,  0x1.ef56180000000p-1f,  -0x1.fdc40e0000000p-1f, -0x1.2785a60000000p-3f,
        0x1.a9aa760000000p-3f,  -0x1.a36b4a0000000p-2f, 0x1.c8513a0000000p-1f,  0x1.1a01360000000p-6f,
        -0x1.82bb7a0000000p-1f, 0x1.84073a0000000p-2f,  -0x1.c040de0000000p-1f, 0x1.9a25c80000000p-1f,
        0x1.fff6e40000000p-2f,  0x1.c223dc0000000p-3f,  0x1.3670e60000000p-2f,  -0x1.903db60000000p-4f,
        -0x1.5749a60000000p-1f, -0x1.8b69b20000000p-1f, -0x1.7254b80000000p-1f, 0x1.873edc0000000p-1f,
        0x1.3e2a480000000p-3f,  -0x1.a210c00000000p-1f, 0x1.606fa20000000p-4f,  0x1.4f0be60000000p-2f,
        -0x1.2129300000000p-2f, -0x1.d5f8bc0000000p-1f, -0x1.b348d40000000p-2f, 0x1.ae7ecc0000000p-9f,
        0x1.a375260000000p-2f,  -0x1.69e1760000000p-1f, -0x1.6715980000000p-2f, -0x1.bf37240000000p-1f,
        0x1.365b880000000p-1f,  0x1.6bf23a0000000p-1f,  -0x1.de9a4a0000000p-1f, 0x1.7f10d80000000p-1f,
        -0x1.23af740000000p-4f, 0x1.36a7e60000000p-1f,  -0x1.8a99ee0000000p-3f, -0x1.2b7b860000000p-6f,
        -0x1.fd64d20000000p-1f, -0x1.551acc0000000p-1f, -0x1.ddd65e0000000p-2f, 0x1.f6d7fc0000000p-4f,
        0x1.c12b280000000p-1f,  0x1.4e95940000000p-2f,  0x1.f9249c0000000p-1f,  -0x1.4d19400000000p-1f,
        0x1.37b47c0000000p-2f,  0x1.bdc0ba0000000p-2f,  -0x1.ba1d7a0000000p-3f, -0x1.94e01a0000000p-3f,
        0x1.ca63400000000p-1f,  0x1.f741ec0000000p-2f,  0x1.a3a4920000000p-2f,  -0x1.c2dd780000000p-2f,
        -0x1.0500220000000p-1f, -0x1.310c680000000p-2f, 0x1.d240de0000000p-1f,  -0x1.d9b0920000000p-1f,
        -0x1.3b3cb00000000p-1f, -0x1.af801a0000000p-1f, -0x1.e7b0b40000000p-2f, 0x1.dc432a0000000p-2f,
        0x1.7b593e0000000p-1f,  -0x1.a76b100000000p-2f, -0x1.e6096e0000000p-2f, 0x1.b220a40000000p-1f,
        0x1.9499e40000000p-2f,  0x1.1b41420000000p-1f,  -0x1.9c8f3a0000000p-1f, 0x1.0976800000000p-3f,
        0x1.731c2c0000000p-1f,  0x1.8361a00000000p-1f,  0x1.3965440000000p-2f,  0x1.6eef320000000p-1f,
        0x1.e2bd7e0000000p-1f,  0x1.6dda8a0000000p-1f,  0x1.97704e0000000p-1f,  0x1.d631ee0000000p-1f,
        -0x1.d0aa100000000p-2f, -0x1.d10d880000000p-1f, -0x1.2a02120000000p-1f, -0x1.f5977c0000000p-4f,
        0x1.6a027a0000000p-3f,  0x1.d7b5be0000000p-3f,  0x1.2cda280000000p-2f,  -0x1.0d8fd60000000p-2f,
        -0x1.0b62540000000p-2f, -0x1.7e757a0000000p-1f, 0x1.af04ce0000000p-1f,  -0x1.9960a20000000p-1f,
        0x1.4b28860000000p-3f,  0x1.4160340000000p-4f,  0x1.adbe6a0000000p-4f,  -0x1.188c380000000p-2f,
        -0x1.11dcfa0000000p-1f, -0x1.2a619c0000000p-3f, 0x1.2c4f2c0000000p-1f,  0x1.6613400000000p-1f,
        0x1.9731c80000000p-1f,  -0x1.4b1b7e0000000p-2f, -0x1.2acc700000000p-1f, -0x1.2fc56e0000000p-2f,
        0x1.9444a40000000p-2f,  0x1.fcadee0000000p-1f,  0x1.fe710e0000000p-2f,  0x1.3f90500000000p-1f,
        -0x1.b1d8be0000000p-2f, -0x1.104bc60000000p-1f, 0x1.02ecca0000000p-1f,  0x1.b001240000000p-1f,
        -0x1.f63f860000000p-1f, -0x1.fbf62a0000000p-3f, 0x1.21ec600000000p-1f,  0x1.ba6e200000000p-2f,
        -0x1.7580160000000p-4f, -0x1.1fbdb00000000p-2f, 0x1.5dad600000000p-2f,  -0x1.91df500000000p-1f,
        -0x1.dc57d80000000p-2f, 0x1.2592780000000p-4f,  0x1.afdf3c0000000p-1f,  -0x1.aedc4a0000000p-2f,
        -0x1.c94e340000000p-4f, 0x1.b8af420000000p-1f,  0x1.435e0e0000000p-1f,  -0x1.5e7e040000000p-4f,
        0x1.7ba8ca0000000p-2f,  -0x1.07fe060000000p-2f, 0x1.6c19720000000p-1f,  -0x1.73cb700000000p-1f,
        -0x1.a866a80000000p-1f, 0x1.b099ea0000000p-4f,  -0x1.3a4c400000000p-1f, 0x1.af04840000000p-4f,
        -0x1.54bf9e0000000p-1f, -0x1.6009a00000000p-1f, -0x1.8f74da0000000p-2f, -0x1.ef5cba0000000p-1f,
        0x1.376e6c0000000p-2f,  -0x1.f2a23a0000000p-2f, 0x1.008bdc0000000p-2f,  0x1.7efdfe0000000p-1f,
        0x1.1353e80000000p-3f,  0x1.125a460000000p-1f,  -0x1.ba40e00000000p-5f, -0x1.97caa20000000p-1f,
        -0x1.2192500000000p-2f, -0x1.e7e2200000000p-1f, -0x1.0da33c0000000p-1f, -0x1.0cc70a0000000p-1f,
        0x1.27c46e0000000p-1f,  0x1.564d720000000p-1f,  -0x1.5ab8800000000p-3f, 0x1.24b9d80000000p-1f,
        0x1.d6f9160000000p-8f,  -0x1.118bc80000000p-4f, -0x1.00aec20000000p-3f, -0x1.a051f60000000p-1f,
        -0x1.da2f440000000p-1f, -0x1.5f8ab00000000p-1f, -0x1.a8f09e0000000p-3f, 0x1.bc4e840000000p-1f,
        -0x1.25d0ca0000000p-1f, -0x1.55d76c0000000p-1f, -0x1.2246f60000000p-1f, -0x1.babb9e0000000p-1f,
        -0x1.9f17240000000p-1f, -0x1.6b3cc40000000p-2f, -0x1.8fbe2a0000000p-1f, 0x1.3272220000000p-2f,
        0x1.f1dad80000000p-1f,  -0x1.e9cb4a0000000p-2f, -0x1.37141c0000000p-1f, 0x1.9a2fd60000000p-3f,
        0x1.84742e0000000p-2f,  -0x1.395a7e0000000p-1f, 0x1.9482080000000p-1f,  0x1.8ce7cc0000000p-1f,
        -0x1.c4f52a0000000p-6f, 0x1.08a0460000000p-4f,  0x1.811a500000000p-3f,  0x1.d560aa0000000p-1f,
        0x1.11a4360000000p-1f,  -0x1.101ef40000000p-1f, 0x1.8aef1e0000000p-2f,  0x1.6648e60000000p-3f,
        -0x1.008fd60000000p-1f, 0x1.2b609c0000000p-4f,  -0x1.8d9d9a0000000p-1f, -0x1.cc33360000000p-1f,
        -0x1.0c33160000000p-6f, 0x1.0ac9680000000p-1f};
    static const float32_t expected0[] = {-0x1.8241b40000000p-2f,
                                          0x1.121b320000000p+0f,
                                          0x1.7034b00000000p+1f,
                                          0x1.7ada940000000p+1f,
                                          -0x1.69bfb40000000p-1f,
                                          -0x1.85844c0000000p+0f,
                                          -0x1.3b1e300000000p+2f,
                                          0x1.4c60da0000000p+0f,
                                          -0x1.6754100000000p-5f,
                                          -0x1.4f7c240000000p-1f};
    static const float32_t expected1[] = {0x1.2442d00000000p+0f,
                                          0x1.86fd5e0000000p+1f,
                                          0x1.3a15c80000000p+0f,
                                          0x1.135e7a0000000p+0f,
                                          -0x1.efa1740000000p+0f,
                                          0x1.6f60f00000000p+0f,
                                          -0x1.297ae00000000p+2f,
                                          0x1.c4007c0000000p-2f,
                                          -0x1.3547ce0000000p+1f,
                                          0x1.4355220000000p+1f,
                                          -0x1.1e6e1c0000000p+2f,
                                          -0x1.2136640000000p+1f,
                                          0x1.40da2c0000000p+1f,
                                          0x1.106aea0000000p+2f,
                                          -0x1.f3a6c00000000p+0f};
    static const float32_t expected2[] = {-0x1.7c6e940000000p+0f, 0x1.4012b80000000p+2f,  0x1.16ec500000000p-2f,
                                          0x1.af1b860000000p+0f,  0x1.211d640000000p+0f,  -0x1.ff88840000000p-1f,
                                          -0x1.09fbaa0000000p+0f, -0x1.6b82520000000p-1f, -0x1.7586440000000p+0f,
                                          0x1.8abf2c0000000p-2f,  -0x1.aeb91c0000000p+0f, 0x1.0b2e720000000p+0f,
                                          -0x1.ca59880000000p-1f, 0x1.4fbe8e0000000p-1f,  -0x1.9d43de0000000p+0f,
                                          -0x1.305d840000000p+1f, -0x1.fd75d80000000p-1f, 0x1.8e58840000000p+0f,
                                          0x1.8b76ac0000000p+0f,  -0x1.64836e0000000p+0f, 0x1.5b020a0000000p+0f};
    static const float32_t expected3[] = {-0x1.431b120000000p-1f,
                                          0x1.05a4880000000p+1f,
                                          0x1.9bcae20000000p+0f,
                                          0x1.d9f1980000000p+0f,
                                          0x1.5862ac0000000p+1f,
                                          -0x1.8f44520000000p+1f,
                                          0x1.6114c80000000p+0f,
                                          -0x1.96f30c0000000p+1f,
                                          -0x1.c27efc0000000p-1f,
                                          -0x1.f21ae60000000p+0f,
                                          -0x1.2cdabe0000000p-2f,
                                          0x1.06fd340000000p+0f,
                                          -0x1.575e660000000p-3f,
                                          -0x1.a0bc680000000p-2f};
    const cmsis_nn_dims in = {2, 3, 5, 7};
    float32_t output[107];
    {
        const cmsis_nn_dims axes = {0, 1, 0, 1};
        const cmsis_nn_dims out = {2, 1, 5, 1};
        memset(output, 0xa5, sizeof(output));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(input, &in, &axes, output + 1, &out));
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
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(input, &in, &axes, output + 1, &out));
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
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(input, &in, &axes, output + 1, &out));
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
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(input, &in, &axes, output + 1, &out));
        TEST_ASSERT_EQUAL_MEMORY(expected3, output + 1, sizeof(expected3));
        const unsigned char *bytes = (const unsigned char *)output;
        for (size_t b = 0; b < sizeof(output[0]); ++b)
        {
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[b]);
            TEST_ASSERT_EQUAL_HEX8(0xa5, bytes[(14 + 1) * sizeof(output[0]) + b]);
        }
    }
}

void rsum_f32_consumer_spatial_layouts_arm_reduce_sum_f32(void)
{
    // LiteRT BUILTIN_REF fixture for the two consumer layouts. Refs #484.
    static float32_t input[64 * 128];
    static const float32_t expected[7] = {-12.125f, -8.125f, -4.125f, -0.125f, 3.875f, 7.875f, 11.875f};
    const cmsis_nn_dims inputs[2] = {{1, 64, 128, 1}, {1, 1, 64, 128}};
    const cmsis_nn_dims axes[2] = {{0, 1, 0, 0}, {0, 0, 1, 0}};
    const cmsis_nn_dims outputs[2] = {{1, 1, 128, 1}, {1, 1, 1, 128}};
    float32_t output[130];
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
            input[r * 128 + c] = (float32_t)((r % 5) - 2 + (c % 7) - 3) / 16.0f;
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
                arm_reduce_sum_f32(input, &inputs[layout], &axes[layout], output + 1, &outputs[layout]);
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            __ASM volatile("vmsr fpscr, %0" : : "r"(saved_fpscr));
#endif
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
            for (int c = 0; c < 128; ++c)
            {
                TEST_ASSERT_EQUAL_MEMORY(&expected[c % 7], &output[c + 1], sizeof(float32_t));
            }
            TEST_ASSERT_EQUAL_FLOAT(1234.0f, output[0]);
            TEST_ASSERT_EQUAL_FLOAT(1234.0f, output[129]);
        }
    }
}

void rsum_f32_legacy_shapes_arm_reduce_sum_f32(void)
{
    // Preserve defined legacy SUM behavior. Refs #484.
    const float32_t input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const cmsis_nn_dims inputs[5] = {{1, 0, 2, 2}, {1, -1, 2, 2}, {2, 65536, 65536, 1}, {1, 2, 2, 2}, {1, 1, 2, 2}};
    const cmsis_nn_dims axes[5] = {{0, 1, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};
    const cmsis_nn_dims outputs[5] = {{1, 1, 2, 2}, {1, 1, 2, 2}, {0, 1, 1, 1}, {1, 1, 1, 1}, {0, 1, 2, 1}};
    const float32_t expected[5][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {91, 91, 91, 91}, {6, 91, 91, 91}, {3, 7, 91, 91}};
    for (int i = 0; i < 5; ++i)
    {
        float32_t output[5] = {91, 91, 91, 91, 91};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(input, &inputs[i], &axes[i], output, &outputs[i]));
        TEST_ASSERT_EQUAL_MEMORY(expected[i], output, sizeof(expected[i]));
        TEST_ASSERT_EQUAL_FLOAT(91.0f, (float)output[4]);
    }
}

void rsum_f32_spatial_order_arm_reduce_sum_f32(void)
{
    // LiteRT BUILTIN_REF cancellation fixture. Refs #484.
    const float32_t pattern[4] = {1.0e20f, 1.0f, -1.0e20f, 1.0f};
    for (int channels = 1; channels <= 3; channels += 2)
    {
        float32_t input[24], output[5];
        const cmsis_nn_dims in = {1, 2, 4, channels};
        const cmsis_nn_dims axes = {0, 1, 1, 0};
        const cmsis_nn_dims out = {1, 1, 1, channels};
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < channels; ++c)
                input[r * channels + c] = pattern[r % 4];
        output[0] = output[channels + 1] = 91;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_reduce_sum_f32(input, &in, &axes, output + 1, &out));
        for (int c = 0; c < channels; ++c)
            TEST_ASSERT_EQUAL_FLOAT(1.0f, (float)output[c + 1]);
        TEST_ASSERT_EQUAL_FLOAT(91.0f, (float)output[0]);
        TEST_ASSERT_EQUAL_FLOAT(91.0f, (float)output[channels + 1]);
    }
}
