/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_resize_bilinear.h"

typedef struct
{
    int64_t step;
    int64_t offset;
    int64_t denominator;
} resize_axis;

static resize_axis make_axis(int32_t input, int32_t output, const cmsis_nn_resize_params *params)
{
    int32_t numerator = params->align_corners ? input - 1 : input;
    int32_t denominator = params->align_corners && output > 1 ? output - 1 : output;
    if (input == 1 || (params->align_corners && output == 1))
    {
        return (resize_axis){0, 0, 1};
    }
    int32_t a = numerator;
    int32_t b = denominator;
    while (b)
    {
        const int32_t remainder = a % b;
        a = b;
        b = remainder;
    }
    numerator /= a;
    denominator /= a;
    if (params->half_pixel_centers)
    {
        return (resize_axis){2LL * numerator, (int64_t)numerator - denominator, 2LL * denominator};
    }
    return (resize_axis){numerator, 0, denominator};
}

static int32_t coordinate(int32_t position, int32_t input_size, resize_axis axis, int64_t *remainder)
{
    int64_t numerator = position * axis.step + axis.offset;
    const int64_t upper = (input_size - 1) * axis.denominator;
    numerator = numerator < 0 ? 0 : numerator;
    numerator = numerator > upper ? upper : numerator;
    const int32_t lower = (int32_t)(numerator / axis.denominator);
    *remainder = numerator - lower * axis.denominator;
    return lower;
}

arm_cmsis_nn_status arm_resize_bilinear_s8(const cmsis_nn_resize_params *resize_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *input_data,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output_data)
{
    if (!arm_nn_resize_bilinear_valid(resize_params, input_dims, input_data, output_dims, output_data))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const resize_axis ay = make_axis(input_dims->h, output_dims->h, resize_params);
    const resize_axis ax = make_axis(input_dims->w, output_dims->w, resize_params);
    /* Shape validation bounds this by 4 * INT32_MAX, including half-pixel coordinates. */
    const int64_t denominator = ay.denominator * ax.denominator;
    const int32_t channels = input_dims->c;
    const int32_t row_stride = input_dims->w * channels;
    const int32_t batch_stride = input_dims->h * row_stride;
    for (int32_t batch = 0; batch < input_dims->n; batch++)
    {
        const int8_t *input = input_data + batch * batch_stride;
        for (int32_t y = 0; y < output_dims->h; y++)
        {
            int64_t ry;
            const int32_t y0 = coordinate(y, input_dims->h, ay, &ry);
            const int32_t y1 = y0 + (y0 < input_dims->h - 1);
            for (int32_t x = 0; x < output_dims->w; x++)
            {
                int64_t rx;
                const int32_t x0 = coordinate(x, input_dims->w, ax, &rx);
                const int32_t x1 = x0 + (x0 < input_dims->w - 1);
                const int8_t *p00 = input + y0 * row_stride + x0 * channels;
                const int8_t *p01 = input + y0 * row_stride + x1 * channels;
                const int8_t *p10 = input + y1 * row_stride + x0 * channels;
                const int8_t *p11 = input + y1 * row_stride + x1 * channels;
                const int64_t w00 = (ay.denominator - ry) * (ax.denominator - rx);
                const int64_t w01 = (ay.denominator - ry) * rx;
                const int64_t w10 = ry * (ax.denominator - rx);
                const int64_t w11 = ry * rx;
                if (denominator <= INT32_MAX / 129)
                {
                    const int32_t d = (int32_t)denominator;
                    for (int32_t c = 0; c < channels; c++)
                    {
                        const int32_t value = p00[c] * (int32_t)w00 + p01[c] * (int32_t)w01 + p10[c] * (int32_t)w10 +
                            p11[c] * (int32_t)w11;
                        output_data[c] = (int8_t)((value + (value < 0 ? -(d / 2) : d / 2)) / d);
                    }
                }
                else
                {
                    for (int32_t c = 0; c < channels; c++)
                    {
                        const int64_t value = p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11;
                        output_data[c] =
                            (int8_t)((value + (value < 0 ? -(denominator / 2) : denominator / 2)) / denominator);
                    }
                }
                output_data += channels;
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
