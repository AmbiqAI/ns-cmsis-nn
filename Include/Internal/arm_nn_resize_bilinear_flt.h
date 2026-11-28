/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_RESIZE_BILINEAR_FLT_H
#define ARM_NN_RESIZE_BILINEAR_FLT_H

#include "Internal/arm_nn_resize_bilinear.h"
#include "arm_nnsupportfunctions.h"
#include <string.h>

#if ARM_NN_FLOAT_API_ENABLED

typedef struct
{
    int32_t lo;
    int32_t hi;
    float fraction;
} arm_nn_bilinear_coordinate;

// The next output position that lands exactly on an input pixel, and its source index.
typedef struct
{
    int32_t output;
    int32_t input;
} arm_nn_bilinear_sample;

typedef struct
{
    float scale;
    int32_t output_step;
    int32_t input_step;
    arm_nn_bilinear_sample first;
} arm_nn_bilinear_grid;

static inline arm_nn_bilinear_grid
arm_nn_bilinear_make_grid(int32_t in, int32_t out, const cmsis_nn_resize_params *params)
{
    const bool align = params->align_corners && out > 1;
    const int32_t numerator = align ? in - 1 : in;
    const int32_t denominator = align ? out - 1 : out;
    int32_t a = numerator, b = denominator;
    while (b != 0)
    {
        const int32_t remainder = a % b;
        a = b;
        b = remainder;
    }
    // With reduced scale p/q, ordinary coordinates map output k*q to input k*p.
    arm_nn_bilinear_grid grid = {(float)numerator / (float)denominator, denominator / a, numerator / a, {0, 0}};
    if (params->half_pixel_centers)
    {
        // (2*x+1)*p = (2*i+1)*q has a solution only when both reduced steps are odd.
        if ((grid.input_step & grid.output_step & 1) != 0)
        {
            grid.first.output = grid.output_step / 2;
            grid.first.input = grid.input_step / 2;
        }
        else
        {
            grid.first.output = -1;
        }
    }
    return grid;
}

static inline arm_nn_bilinear_coordinate arm_nn_bilinear_axis(int32_t pos,
                                                              int32_t extent,
                                                              bool half,
                                                              const arm_nn_bilinear_grid *grid,
                                                              arm_nn_bilinear_sample *next)
{
    if (pos == next->output)
    {
        const int32_t sample = next->input;
        // The byte-count guard bounds each dimension to INT32_MAX/2.
        next->output += grid->output_step;
        next->input += grid->input_step;
        return (arm_nn_bilinear_coordinate){sample, sample, 0.0f};
    }
    // For fractional positions evaluate the same FP32 coordinate as the LiteRT reference.
    const float x = half ? ((float)pos + 0.5f) * grid->scale - 0.5f : (float)pos * grid->scale;
    if (x <= 0.0f)
    {
        return (arm_nn_bilinear_coordinate){0, 0, 0.0f};
    }
    const int32_t lo = (int32_t)x;
    if (lo >= extent - 1)
    {
        return (arm_nn_bilinear_coordinate){extent - 1, extent - 1, 0.0f};
    }
    const float fraction = x - (float)lo;
    return (arm_nn_bilinear_coordinate){lo, lo + (fraction > 0.0f), fraction};
}

typedef void (*arm_nn_bilinear_pixel_fn)(const void *,
                                         const void *,
                                         const void *,
                                         const void *,
                                         float,
                                         float,
                                         float,
                                         float,
                                         int32_t,
                                         void *,
                                         bool);

// Both callers pass a compile-time element width and pixel function.
__STATIC_FORCEINLINE arm_cmsis_nn_status arm_nn_bilinear_resize(const cmsis_nn_resize_params *params,
                                                                const cmsis_nn_dims *in,
                                                                const void *src,
                                                                const cmsis_nn_dims *out,
                                                                void *dst,
                                                                size_t width,
                                                                arm_nn_bilinear_pixel_fn pixel)
{
    if (!arm_nn_resize_bilinear_valid(params, in, src, out, dst))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    // The shared validator already bounded each element product to INT32_MAX.
    const size_t inputs = (size_t)in->n * in->h * in->w * in->c;
    const size_t outputs = (size_t)out->n * out->h * out->w * out->c;
    if (inputs > INT32_MAX / width || outputs > INT32_MAX / width)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (in->h == out->h && in->w == out->w)
    {
        memcpy(dst, src, inputs * width);
        return ARM_CMSIS_NN_SUCCESS;
    }

    const size_t pixel_bytes = in->c * width;
    if (in->h == 1 && in->w == 1)
    {
        const int32_t pixels = out->h * out->w;
        uint8_t *output = (uint8_t *)dst;
        for (int32_t b = 0; b < in->n; ++b)
        {
            const uint8_t *sample = (const uint8_t *)src + b * pixel_bytes;
            for (int32_t i = 0; i < pixels; ++i, output += pixel_bytes)
            {
                memcpy(output, sample, pixel_bytes);
            }
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
    const size_t row_bytes = in->w * pixel_bytes;
    const arm_nn_bilinear_grid rows = arm_nn_bilinear_make_grid(in->h, out->h, params);
    const arm_nn_bilinear_grid cols = arm_nn_bilinear_make_grid(in->w, out->w, params);
    // Exact scales 0, 1/2, 1, and 2 yield only 0, 1/4, 1/2, or 3/4 as fractions,
    // including half-pixel coordinates. Both precisions represent their weights exactly.
    const bool quarter_grid =
        rows.input_step <= 2 && rows.output_step <= 2 && cols.input_step <= 2 && cols.output_step <= 2;
    uint8_t *output = (uint8_t *)dst;
    for (int32_t b = 0; b < in->n; ++b)
    {
        const uint8_t *batch = (const uint8_t *)src + b * in->h * row_bytes;
        arm_nn_bilinear_sample row = rows.first;
        for (int32_t y = 0; y < out->h; ++y)
        {
            const arm_nn_bilinear_coordinate v =
                arm_nn_bilinear_axis(y, in->h, params->half_pixel_centers, &rows, &row);
            const uint8_t *top = batch + v.lo * row_bytes;
            const uint8_t *bottom = batch + v.hi * row_bytes;
            arm_nn_bilinear_sample col = cols.first;
            for (int32_t x = 0; x < out->w; ++x, output += pixel_bytes)
            {
                const arm_nn_bilinear_coordinate u =
                    arm_nn_bilinear_axis(x, in->w, params->half_pixel_centers, &cols, &col);
                const size_t left = u.lo * pixel_bytes;
                const size_t right = u.hi * pixel_bytes;
                if (u.lo == u.hi && v.lo == v.hi)
                {
                    memcpy(output, top + left, pixel_bytes);
                    continue;
                }
                pixel(top + left,
                      bottom + left,
                      top + right,
                      bottom + right,
                      (1.0f - v.fraction) * (1.0f - u.fraction),
                      v.fraction * (1.0f - u.fraction),
                      (1.0f - v.fraction) * u.fraction,
                      v.fraction * u.fraction,
                      in->c,
                      output,
                      quarter_grid);
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif
#endif
