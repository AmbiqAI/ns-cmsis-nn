/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "resize_bilinear_flt_data.h"
#include <string.h>

static void resize_bilinear_reference(void)
{
    const int values[] = {2, -4, 8, -1, 6, -10};
    enum
    {
        batches = 2,
        channels = 21
    };
    RESIZE_TYPE input[batches * 6 * channels];
    RESIZE_TYPE output[batches * 15 * channels + 1];
    for (size_t k = 0; k < sizeof(resize_cases) / sizeof(*resize_cases); ++k)
    {
        const cmsis_nn_dims in = {batches, resize_cases[k].h, resize_cases[k].w, channels};
        const cmsis_nn_dims out = {batches, resize_cases[k].out_h, resize_cases[k].out_w, channels};
        const cmsis_nn_resize_params params = {resize_cases[k].align_corners, resize_cases[k].half_pixel_centers};
        for (int b = 0; b < batches; ++b)
            for (int i = 0; i < in.h * in.w; ++i)
                for (int c = 0; c < channels; ++c)
                    input[(b * in.h * in.w + i) * channels + c] = (RESIZE_TYPE)(values[i] + b * 16 + c);
        const int count = batches * out.h * out.w * channels;
        output[count] = (RESIZE_TYPE)-99;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RESIZE_ENTRY(&params, &in, input, &out, output));
        for (int b = 0; b < batches; ++b)
            for (int i = 0; i < out.h * out.w; ++i)
                for (int c = 0; c < channels; ++c)
                    TEST_ASSERT_FLOAT_WITHIN(RESIZE_TOLERANCE,
                                             resize_cases[k].expected[i] + b * 16 + c,
                                             (float)output[(b * out.h * out.w + i) * channels + c]);
        TEST_ASSERT_EQUAL_FLOAT(-99, (float)output[count]);
    }
}

static void resize_bilinear_copies(void)
{
    const cmsis_nn_dims in = {1, 1, 1, 3}, out = {1, 2, 3, 3};
    const cmsis_nn_resize_params params = {false, true};
    RESIZE_TYPE input[3], output[18];
    // Include a NaN payload, negative zero and the smallest positive subnormal.
    memcpy(input, resize_special_bits, sizeof(input));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RESIZE_ENTRY(&params, &in, input, &in, output));
    TEST_ASSERT_EQUAL_MEMORY(input, output, sizeof(input));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RESIZE_ENTRY(&params, &in, input, &out, output));
    for (int i = 0; i < 6; ++i)
        TEST_ASSERT_EQUAL_MEMORY(input, output + 3 * i, sizeof(input));
}

static void resize_bilinear_rejects(void)
{
    const cmsis_nn_dims in = {1, 1, 1, 1};
    const cmsis_nn_dims bad[] = {
        {0, 1, 1, 1}, {1, 1, -1, 1}, {2, 1, 1, 1}, {1, 1, 1, 2}, {1, INT32_MAX, 2, 1}, {1, INT32_MAX, 1, 1}};
    const cmsis_nn_resize_params both = {true, true}, valid = {false, false};
    RESIZE_TYPE src = 1, dst = -2;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_ENTRY(&both, &in, &src, &in, &dst));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_ENTRY(NULL, &in, &src, &in, &dst));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_ENTRY(&valid, &in, NULL, &in, &dst));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_ENTRY(&valid, &in, &src, &in, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_ENTRY(&valid, NULL, &src, &in, &dst));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_ENTRY(&valid, &in, &src, NULL, &dst));
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); ++i)
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_ENTRY(&valid, &in, &src, bad + i, &dst));
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RESIZE_ENTRY(&valid, bad + i, &src, &in, &dst));
    }
    TEST_ASSERT_EQUAL_FLOAT(-2, (float)dst);
}
