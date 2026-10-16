/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* Define RESIZE_NN_ROUND_TYPE and RESIZE_NN_ROUND_FUNC before including this
 * dtype-independent exact-copy regression. */
#include <stdint.h>
#include <string.h>

/* Frozen float32 reference maps: round the scale division, then the shifted
 * coordinate, then their product before floor/round. These include boundaries
 * where reassociating the division changes which pixel is selected (#852).
 * Expectations contain no floating-point arithmetic that fast-math can alter. */
static const struct
{
    int32_t input_size;
    int32_t output_size;
    bool align;
    bool half;
    uint8_t map[75];
} resize_nn_round_cases[] = {
    {26, 22, false, false, {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24}},
    {6, 37, false, true, {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                          3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5}},
    {4, 75, true, false, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}},
    {27, 23, true, true, {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26}},
    {94, 62, true, true, {1,  2,  4,  5,  7,  8,  10, 11, 13, 14, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31,
                          33, 34, 36, 37, 39, 40, 42, 43, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63,
                          65, 66, 68, 69, 71, 72, 74, 75, 77, 79, 80, 82, 83, 85, 86, 88, 89, 91, 92, 93}},
};

static void resize_nn_rounding_boundaries(void)
{
    enum
    {
        channels = 3,
        batches = 2,
        guard = 8
    };
    static RESIZE_NN_ROUND_TYPE input[batches * 94 * channels];
    static RESIZE_NN_ROUND_TYPE output[batches * 75 * channels + 2 * guard];
    int32_t scratch[76 + 2 * guard];
    const cmsis_nn_dims size_dims = {1, 1, 1, 2};
    for (size_t i = 0; i < sizeof(input); ++i)
    {
        ((uint8_t *)input)[i] = (uint8_t)(37U * i + 17U);
    }
    for (size_t k = 0; k < sizeof(resize_nn_round_cases) / sizeof(resize_nn_round_cases[0]); ++k)
    {
        const int32_t in_size = resize_nn_round_cases[k].input_size;
        const int32_t out_size = resize_nn_round_cases[k].output_size;
        const cmsis_nn_resize_params params = {.align_corners = resize_nn_round_cases[k].align,
                                               .half_pixel_centers = resize_nn_round_cases[k].half};
        for (int32_t axis = 0; axis < 2; ++axis)
        {
            const cmsis_nn_dims input_dims = {batches, axis ? in_size : 1, axis ? 1 : in_size, channels};
            const cmsis_nn_dims output_dims = {batches, axis ? out_size : 1, axis ? 1 : out_size, channels};
            const int32_t sizes[2] = {output_dims.h, output_dims.w};
            cmsis_nn_context ctx = {scratch + guard, (out_size + 1) * (int32_t)sizeof(int32_t)};
            memset(output, 0xa5, sizeof(output));
            memset(scratch, 0xa5, sizeof(scratch));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                              RESIZE_NN_ROUND_FUNC(
                                  &ctx, &params, &input_dims, input, &size_dims, sizes, &output_dims, output + guard));
            const int32_t *map = scratch + guard + (axis ? 1 : 0);
            for (int32_t i = 0; i < out_size; ++i)
            {
                TEST_ASSERT_EQUAL_INT32(resize_nn_round_cases[k].map[i], map[i]);
                for (int32_t b = 0; b < batches; ++b)
                {
                    const RESIZE_NN_ROUND_TYPE *expected =
                        input + (b * in_size + resize_nn_round_cases[k].map[i]) * channels;
                    const RESIZE_NN_ROUND_TYPE *actual = output + guard + (b * out_size + i) * channels;
                    TEST_ASSERT_EQUAL_MEMORY(expected, actual, channels * sizeof(*input));
                }
            }
            TEST_ASSERT_EQUAL_INT32(0, scratch[guard + (axis ? 0 : out_size)]);
            for (size_t i = 0; i < sizeof(output); ++i)
            {
                if (i < guard * sizeof(*output) || i >= (guard + batches * out_size * channels) * sizeof(*output))
                {
                    TEST_ASSERT_EQUAL_HEX8(0xa5, ((uint8_t *)output)[i]);
                }
            }
            for (size_t i = 0; i < sizeof(scratch) / sizeof(scratch[0]); ++i)
            {
                if (i < guard || i >= (size_t)(guard + out_size + 1))
                {
                    TEST_ASSERT_EQUAL_HEX32(0xa5a5a5a5U, (uint32_t)scratch[i]);
                }
            }
        }
    }
}

#undef RESIZE_NN_ROUND_TYPE
#undef RESIZE_NN_ROUND_FUNC
