/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__extension__ typedef __int128 exact_int;

static void require(int condition, const char *message)
{
    if (!condition)
    {
        fprintf(stderr, "%s\n", message);
        exit(EXIT_FAILURE);
    }
}

/* Unreduced half-pixel coordinates for a two-element input axis. */
static int64_t position(int32_t output_index, int32_t output_size)
{
    const int64_t numerator = (2LL * output_index + 1) * 2 - output_size;
    const int64_t denominator = 2LL * output_size;
    return numerator < 0 ? 0 : numerator > denominator ? denominator : numerator;
}

static void check_case(int32_t height, int32_t width, int expect_wide)
{
    const int8_t input[16] = {-128, 127, -128, -126,
                            -128, 127, 127, 127,
                            -128, 127, 127, 127,
                            -128, 127, -128, -126};
    const cmsis_nn_dims input_dims = {.n = 1, .h = 2, .w = 2, .c = 4};
    const cmsis_nn_dims output_dims = {.n = 1, .h = height, .w = width, .c = 4};
    const cmsis_nn_resize_params params = {.align_corners = 0, .half_pixel_centers = 1};
    const size_t count = (size_t)height * width * 4;
    int8_t *allocation = malloc(count + 64);
    require(allocation != NULL, "allocation failed");
    memset(allocation, 0x5a, count + 64);
    int8_t *output = allocation + 32;
    require(arm_resize_bilinear_s8(&params, &input_dims, input, &output_dims, output) == ARM_CMSIS_NN_SUCCESS,
            "resize failed");

    const int64_t dy = 2LL * height;
    const int64_t dx = 2LL * width;
    const exact_int denominator = (exact_int)dy * dx;
    require((denominator > INT32_MAX / 129) == expect_wide, "case does not straddle the accumulator cutoff");
    int positive_tie = 0;
    int negative_tie = 0;
    int exceeds_int32 = 0;
    size_t index = 0;
    for (int32_t y = 0; y < height; y++)
    {
        const int64_t yn = position(y, height);
        const int32_t y0 = (int32_t)(yn / dy);
        const int32_t y1 = y0 < 1 ? y0 + 1 : y0;
        const int64_t ry = yn % dy;
        for (int32_t x = 0; x < width; x++)
        {
            const int64_t xn = position(x, width);
            const int32_t x0 = (int32_t)(xn / dx);
            const int32_t x1 = x0 < 1 ? x0 + 1 : x0;
            const int64_t rx = xn % dx;
            const exact_int weights[4] = {(exact_int)(dy - ry) * (dx - rx),
                                          (exact_int)(dy - ry) * rx,
                                          (exact_int)ry * (dx - rx),
                                          (exact_int)ry * rx};
            for (int32_t c = 0; c < 4; c++, index++)
            {
                const exact_int value = input[(y0 * 2 + x0) * 4 + c] * weights[0] +
                    input[(y0 * 2 + x1) * 4 + c] * weights[1] + input[(y1 * 2 + x0) * 4 + c] * weights[2] +
                    input[(y1 * 2 + x1) * 4 + c] * weights[3];
                const exact_int magnitude = value < 0 ? -value : value;
                const int32_t rounded = (int32_t)((2 * magnitude + denominator) / (2 * denominator));
                const int32_t expected = value < 0 ? -rounded : rounded;
                if (output[index] != expected)
                {
                    fprintf(stderr, "%dx%d at (%d,%d,%d): expected %d, got %d\n",
                            height, width, y, x, c, expected, output[index]);
                    exit(EXIT_FAILURE);
                }
                const int tie = 2 * (magnitude % denominator) == denominator;
                positive_tie |= tie && value > 0;
                negative_tie |= tie && value < 0;
                exceeds_int32 |= value > INT32_MAX || value < INT32_MIN;
            }
        }
    }
    require(positive_tie && negative_tie, "both signed rounding ties must be exercised");
    require(exceeds_int32 == expect_wide, "expected INT32 overflow boundary was not exercised");
    for (size_t i = 0; i < 32; i++)
    {
        require(allocation[i] == 0x5a && allocation[32 + count + i] == 0x5a, "output guard changed");
    }
    free(allocation);
    printf("PASS %dx%d: %zu exact outputs, signed ties, extrema, guards, wide=%d\n",
           height, width, count, expect_wide);
}

int main(void)
{
    check_case(2039, 2041, 0);
    check_case(2049, 2051, 1);
    /* Far enough past the cutoff that narrowing the sum cannot alias the same final INT8 value. */
    check_case(2053, 2057, 1);
    return EXIT_SUCCESS;
}
