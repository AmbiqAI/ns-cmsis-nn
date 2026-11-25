/*
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Copyright (c) 2026 Ambiq Micro, Inc.
 */

#ifndef ARM_NN_CONV_GROUPED_F16_H
#define ARM_NN_CONV_GROUPED_F16_H

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16 && defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
/* OHWI filters place the same tap of adjacent output channels filter_area halves apart. */
__STATIC_INLINE int32_t arm_nn_conv_f16_weight_gather_lanes(size_t filter_area)
{
    return (int32_t)ARM_NN_MIN(8u, 1u + ARM_NN_MVE_F16_GATHER_OFFSET_MAX / filter_area);
}

/* For a validated, unpadded convolution whose last window fits in the input.
 * The last tap/channel is the largest offset; validation bounds the whole input
 * tensor in size_t. Build offsets once and load them with a predicated vector load. */
__STATIC_INLINE bool arm_nn_conv_f16_gather_offsets(uint16_t *offsets,
                                                    const cmsis_nn_dims *input,
                                                    const cmsis_nn_dims *filter,
                                                    const cmsis_nn_tile *dilation)
{
    const size_t last_spatial =
        (size_t)(filter->h - 1) * dilation->h * input->w + (size_t)(filter->w - 1) * dilation->w;
    if (last_spatial * input->c + filter->c - 1 > ARM_NN_MVE_F16_GATHER_OFFSET_MAX)
    {
        return false;
    }

    for (int32_t y = 0; y < filter->h; ++y)
    {
        const size_t row = (size_t)y * dilation->h * input->w;
        for (int32_t x = 0; x < filter->w; ++x)
        {
            const size_t tap = (row + (size_t)x * dilation->w) * input->c;
            for (int32_t c = 0; c < filter->c; ++c)
            {
                *offsets++ = (uint16_t)(tap + c);
            }
        }
    }
    return true;
}
#endif

#if ARM_NN_ENABLE_F16
/* Clip wide grouped coordinates before taking any tensor offsets. */
__STATIC_FORCEINLINE void arm_conv_f16_tap_span(int64_t base,
                                                int32_t dilation,
                                                int32_t kernel,
                                                int32_t input,
                                                int32_t *first,
                                                int32_t *end,
                                                uint32_t *coordinate)
{
    /* Validated nonnegative int32 padding gives base >= -INT32_MAX. Thus both
     * clipping distances fit uint32, even when coordinate construction needs int64. */
    const uint32_t lo = base < 0 ? ((uint32_t)(-base) - 1u) / (uint32_t)dilation + 1u : 0;
    const uint32_t hi = input > base ? (uint32_t)((int64_t)input - 1 - base) / (uint32_t)dilation + 1u : 0;
    *first = (int32_t)(lo < (uint32_t)kernel ? lo : (uint32_t)kernel);
    *end = (int32_t)(hi < (uint32_t)kernel ? hi : (uint32_t)kernel);
    *coordinate = *first < *end ? (uint32_t)(base + (int64_t)*first * dilation) : 0;
}

#endif

#endif /* ARM_NN_CONV_GROUPED_F16_H */
