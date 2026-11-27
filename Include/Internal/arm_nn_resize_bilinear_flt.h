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

static inline float arm_nn_bilinear_scale(int32_t in, int32_t out, bool align)
{
    return align && out > 1 ? (float)(in - 1) / (float)(out - 1) : (float)in / (float)out;
}

static inline arm_nn_bilinear_coordinate arm_nn_bilinear_axis(int32_t pos, float scale, int32_t extent, bool half)
{
    const float x = half ? ((float)pos + 0.5f) * scale - 0.5f : (float)pos * scale;
    // Clamp the integer index itself: (float)(extent - 1) may round up on long axes.
    int32_t lo = (int32_t)x;
    if (x <= 0.0f)
    {
        return (arm_nn_bilinear_coordinate){0, 0, 0.0f};
    }
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
    const size_t row_bytes = in->w * pixel_bytes;
    const float sy = arm_nn_bilinear_scale(in->h, out->h, params->align_corners);
    const float sx = arm_nn_bilinear_scale(in->w, out->w, params->align_corners);
    uint8_t *output = (uint8_t *)dst;
    for (int32_t b = 0; b < in->n; ++b)
    {
        const uint8_t *batch = (const uint8_t *)src + b * in->h * row_bytes;
        for (int32_t y = 0; y < out->h; ++y)
        {
            const arm_nn_bilinear_coordinate v = arm_nn_bilinear_axis(y, sy, in->h, params->half_pixel_centers);
            const uint8_t *top = batch + v.lo * row_bytes;
            const uint8_t *bottom = batch + v.hi * row_bytes;
            for (int32_t x = 0; x < out->w; ++x, output += pixel_bytes)
            {
                const arm_nn_bilinear_coordinate u = arm_nn_bilinear_axis(x, sx, in->w, params->half_pixel_centers);
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
