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

// Track the sampling grid as an integer index plus a fractional remainder. Incrementing
// the remainder preserves exact sample positions even when the FP32 scale would round.
typedef struct
{
    int32_t index;
    uint32_t remainder;
} arm_nn_bilinear_position;

typedef struct
{
    int32_t stride;
    uint32_t increment;
    uint32_t denominator;
    float inverse_denominator;
    arm_nn_bilinear_position start;
} arm_nn_bilinear_grid;

static inline arm_nn_bilinear_grid
arm_nn_bilinear_make_grid(int32_t in, int32_t out, const cmsis_nn_resize_params *params)
{
    const bool align = params->align_corners && out > 1;
    const uint32_t numerator = align ? in - 1 : in;
    const uint32_t denominator = align ? out - 1 : out;
    arm_nn_bilinear_grid grid;
    grid.stride = (int32_t)(numerator / denominator);
    grid.increment = numerator % denominator;
    grid.denominator = denominator;
    grid.start = (arm_nn_bilinear_position){0, 0};
    if (params->half_pixel_centers)
    {
        // (pos + 1/2) * in/out - 1/2 starts at (in - out)/(2*out).
        grid.denominator *= 2;
        grid.increment *= 2;
        const int32_t offset = in - out;
        if (offset < 0)
        {
            grid.start.index = -1;
            grid.start.remainder = (uint32_t)(in + out);
        }
        else
        {
            grid.start.index = (int32_t)((uint32_t)offset / grid.denominator);
            grid.start.remainder = (uint32_t)offset % grid.denominator;
        }
    }
    grid.inverse_denominator = 1.0f / (float)grid.denominator;
    return grid;
}

static inline void arm_nn_bilinear_advance(arm_nn_bilinear_position *pos, const arm_nn_bilinear_grid *grid)
{
    // The byte-count guard bounds both dimensions to INT32_MAX/2 even for FP16.
    pos->index += grid->stride;
    pos->remainder += grid->increment;
    if (pos->remainder >= grid->denominator)
    {
        pos->remainder -= grid->denominator;
        ++pos->index;
    }
}

static inline arm_nn_bilinear_coordinate
arm_nn_bilinear_axis(arm_nn_bilinear_position pos, int32_t extent, float inverse_denominator)
{
    if (pos.index < 0)
    {
        return (arm_nn_bilinear_coordinate){0, 0, 0.0f};
    }
    if (pos.index >= extent - 1)
    {
        return (arm_nn_bilinear_coordinate){extent - 1, extent - 1, 0.0f};
    }
    return (arm_nn_bilinear_coordinate){
        pos.index, pos.index + (pos.remainder != 0), (float)pos.remainder * inverse_denominator};
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
                                         void *);

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
    uint8_t *output = (uint8_t *)dst;
    for (int32_t b = 0; b < in->n; ++b)
    {
        const uint8_t *batch = (const uint8_t *)src + b * in->h * row_bytes;
        arm_nn_bilinear_position row = rows.start;
        for (int32_t y = 0; y < out->h; ++y, arm_nn_bilinear_advance(&row, &rows))
        {
            const arm_nn_bilinear_coordinate v = arm_nn_bilinear_axis(row, in->h, rows.inverse_denominator);
            const uint8_t *top = batch + v.lo * row_bytes;
            const uint8_t *bottom = batch + v.hi * row_bytes;
            arm_nn_bilinear_position col = cols.start;
            for (int32_t x = 0; x < out->w; ++x, output += pixel_bytes, arm_nn_bilinear_advance(&col, &cols))
            {
                const arm_nn_bilinear_coordinate u = arm_nn_bilinear_axis(col, in->w, cols.inverse_denominator);
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
                      output);
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif
#endif
