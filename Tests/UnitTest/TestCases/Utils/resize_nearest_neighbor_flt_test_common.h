/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* Element-width-agnostic helpers shared by the f32 and f16 nearest-neighbor
 * resize suites. The kernels are pure copies, so every check is bit-exact on
 * raw bytes: the expected output is gathered from the TFLite-derived index maps
 * in resize_nearest_neighbor_flt_cases.h, and the index formula below is an
 * independent transcription of the TFLite reference that the suites hold to
 * those same maps. */
#pragma once

#include "resize_nearest_neighbor_flt_cases.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* TFLite reference_ops::GetNearestNeighbor with TF's out == 1 scale guard. */
static int32_t resize_nn_ref_index(int32_t out_idx, int32_t in_size, int32_t out_size, bool align, bool half)
{
    const float scale =
        (align && out_size > 1) ? (float)(in_size - 1) / (float)(out_size - 1) : (float)in_size / (float)out_size;
    const float pos = ((float)out_idx + (half ? 0.5f : 0.0f)) * scale;
    int32_t idx = align ? (int32_t)roundf(pos) : (int32_t)floorf(pos);
    if (idx > in_size - 1)
    {
        idx = in_size - 1;
    }
    if (half && idx < 0)
    {
        idx = 0;
    }
    return idx;
}

/* xorshift32 raw bit fill: ~0.4 % (f32) / ~3 % (f16) of the lanes come out as NaN, Inf or subnormal, so the
 * bit-exact compare covers those classes on every case without hand placement. */
static void resize_nn_fill_bits(void *buf, size_t elements, size_t elem_size, uint32_t seed)
{
    uint8_t *bytes = (uint8_t *)buf;
    uint32_t state = seed;
    for (size_t i = 0; i < elements; ++i)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        memcpy(bytes + i * elem_size, &state, elem_size);
    }
}

/* Expected output gathered straight from the case's index maps. */
static void resize_nn_gather(void *out, const void *in, size_t elem_size, const resize_nn_flt_case *c)
{
    const uint8_t *src = (const uint8_t *)in;
    uint8_t *dst = (uint8_t *)out;
    const size_t pixel = (size_t)c->c * elem_size;
    for (int32_t b = 0; b < c->n; ++b)
    {
        for (int32_t y = 0; y < c->out_h; ++y)
        {
            for (int32_t x = 0; x < c->out_w; ++x)
            {
                const size_t src_pixel = ((size_t)b * c->h + c->y_map[y]) * c->w + c->x_map[x];
                memcpy(dst, src + src_pixel * pixel, pixel);
                dst += pixel;
            }
        }
    }
}

/* f32 bit patterns: qNaN, -NaN with payload, sNaN, +Inf, -Inf, -0.0, min subnormal, max finite. */
static const uint32_t resize_nn_special_bits_f32[8] =
    {0x7FC00000U, 0xFFC01234U, 0x7FA00001U, 0x7F800000U, 0xFF800000U, 0x80000000U, 0x00000001U, 0x7F7FFFFFU};
/* f16 equivalents. */
static const uint16_t resize_nn_special_bits_f16[8] =
    {0x7E00U, 0xFE12U, 0x7D01U, 0x7C00U, 0xFC00U, 0x8000U, 0x0001U, 0x7BFFU};

#define RESIZE_NN_CANARY_WORD 0xA5C3F00DU
