/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_ARG_EXTREMA_FLT_H
#define ARM_NN_ARG_EXTREMA_FLT_H

#include "arm_nnfunctions.h"
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if ARM_NN_FLOAT_API_ENABLED

/* The encodings are read through these types, which may alias the float
 * storage they are laid over, so the load is one instruction and not a call
 * to memcpy in a freestanding build. Both are the storage's own width. */
typedef uint32_t arm_nn_arg_bits32_t __attribute__((__may_alias__));
typedef uint16_t arm_nn_arg_bits16_t __attribute__((__may_alias__));

/* The sort key of one encoding, without a branch: the sign bit is flipped
 * for a non-negative value and every bit for a negative one, which orders
 * the keys as the values, subnormals included; both zero signs share the
 * key `sign`; and a NaN takes `lose`, a key no finite or infinite value's
 * comparison can select, so it never wins. `sign` is the storage's sign
 * bit and `lose` 0 for a maximum and UINT32_MAX for a minimum. The
 * ternaries compile to conditional selects on Cortex-M. */
static inline uint32_t arm_nn_arg_key(uint32_t bits, uint32_t sign, uint32_t lose)
{
    const uint32_t magnitude_mask = sign - 1;
    const uint32_t infinity = sign == UINT32_C(0x8000) ? UINT32_C(0x7c00) : UINT32_C(0x7f800000);
    const uint32_t magnitude = bits & magnitude_mask;
    const uint32_t flipped = (bits & sign) ? (~bits & (sign | magnitude_mask)) : (bits | sign);
    const uint32_t key = magnitude == 0 ? sign : flipped;
    return magnitude > infinity ? lose : key;
}

/* One reduced line of binary32 encodings, `stride` elements apart: the
 * first extremum's index, the first non-NaN value always replacing the
 * starting key and an all-NaN line keeping index 0. `line` is the float32_t
 * pointer the entry took plus a whole number of elements, so it keeps the
 * alignment the entry's contract requires; the cast through void * says so
 * to -Wcast-align and proves nothing more. */
static inline int32_t arm_nn_arg_line_32(const uint8_t *line, size_t stride, size_t reduction, bool maximum)
{
    const uint32_t sign = UINT32_C(0x80000000);
    const uint32_t lose = maximum ? 0 : UINT32_MAX;
    const arm_nn_arg_bits32_t *element = (const arm_nn_arg_bits32_t *)(const void *)line;
    uint32_t best_key = lose;
    int32_t best_index = 0;
    for (size_t k = 0; k < reduction; ++k)
    {
        const uint32_t key = arm_nn_arg_key(element[k * stride], sign, lose);
        const bool better = maximum ? key > best_key : key < best_key;
        best_key = better ? key : best_key;
        best_index = better ? (int32_t)k : best_index;
    }
    return best_index;
}

/* The binary16 twin of arm_nn_arg_line_32, `line` keeping a float16_t's alignment. */
static inline int32_t arm_nn_arg_line_16(const uint8_t *line, size_t stride, size_t reduction, bool maximum)
{
    const uint32_t sign = UINT32_C(0x8000);
    const uint32_t lose = maximum ? 0 : UINT32_MAX;
    const arm_nn_arg_bits16_t *element = (const arm_nn_arg_bits16_t *)(const void *)line;
    uint32_t best_key = lose;
    int32_t best_index = 0;
    for (size_t k = 0; k < reduction; ++k)
    {
        const uint32_t key = arm_nn_arg_key(element[k * stride], sign, lose);
        const bool better = maximum ? key > best_key : key < best_key;
        best_key = better ? key : best_key;
        best_index = better ? (int32_t)k : best_index;
    }
    return best_index;
}

/* The line counts of a 4-D shape reduced along `axis`, or ARM_CMSIS_NN_ARG_ERROR where the
 * shape is refused: no shape, an axis out of range, a negative extent, an empty reduced axis,
 * or a byte count above INT32_MAX. outer: the extents before the axis; inner: those after it;
 * the output holds one index per (outer, inner) line, and outer * inner is that count. Nothing
 * here divides.
 *
 * Small extents take a 32-bit bound. Where every extent is below 2^7 (the OR of the four,
 * which are non-negative here, is below 128), the element count is at most 127^4 =
 * 260,144,641, below 2^28, so at a width of at most four bytes the input's byte count is below
 * 2^30, the output's (at most 127^3 indices of four bytes) below 2^23, and each pair below
 * 2^14: the 64-bit chain below refuses none of these shapes. Its counts, products of at most
 * three extents, are below 2^21 and are formed exactly in 32 bits.
 *
 * Otherwise the 64-bit chain decides. A zero extent gives an empty array whatever the other
 * extents are; otherwise the input's byte count and the output's (four bytes an index) are
 * bounded by INT32_MAX. Two int32 extents multiply below 2^62 in 64 bits, but three or four
 * can wrap it (2^30 x 2^30 x 16 reads as 0). So the extents are first taken in two pairs:
 * where neither pair is zero, every extent is at least 1 and the element count is at least
 * either pair, so a pair above INT32_MAX is refused here, and every product below then stays
 * under 2^62 and cannot wrap. A zero pair leaves the empty array to the products, which a zero
 * factor keeps exact; where the output count is zero, outer or inner is zero. */
static inline arm_cmsis_nn_status arm_nn_arg_extrema_lines(const cmsis_nn_dims *input_dims,
                                                           int32_t axis,
                                                           size_t width,
                                                           size_t *outer,
                                                           size_t *inner,
                                                           size_t *reduction)
{
    if (!input_dims || axis < 0 || axis > 3)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t n = input_dims->n;
    const int32_t h = input_dims->h;
    const int32_t w = input_dims->w;
    const int32_t c = input_dims->c;
    if ((n | h | w | c) < 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t dims[4] = {n, h, w, c};
    *reduction = (size_t)dims[axis];
    /* There is no valid index for an empty reduced axis, even if output is empty. */
    if (*reduction == 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (width <= sizeof(uint32_t) && (uint32_t)(n | h | w | c) < (UINT32_C(1) << 7))
    {
        *outer = (uint32_t)(axis > 0 ? n : 1) * (uint32_t)(axis > 1 ? h : 1) * (uint32_t)(axis > 2 ? w : 1);
        *inner = (uint32_t)(axis < 1 ? h : 1) * (uint32_t)(axis < 2 ? w : 1) * (uint32_t)(axis < 3 ? c : 1);
        return ARM_CMSIS_NN_SUCCESS;
    }
    const uint64_t pair_nh = (uint64_t)n * (uint64_t)h;
    const uint64_t pair_wc = (uint64_t)w * (uint64_t)c;
    if (pair_nh && pair_wc && (pair_nh > (uint64_t)INT32_MAX || pair_wc > (uint64_t)INT32_MAX))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const uint64_t outer64 = (uint64_t)(axis > 0 ? n : 1) * (uint64_t)(axis > 1 ? h : 1) * (uint64_t)(axis > 2 ? w : 1);
    const uint64_t inner64 = (uint64_t)(axis < 1 ? h : 1) * (uint64_t)(axis < 2 ? w : 1) * (uint64_t)(axis < 3 ? c : 1);
    const uint64_t output_count = outer64 * inner64;
    const uint64_t input_count = output_count * (uint64_t)*reduction;
    if (input_count * (uint64_t)width > (uint64_t)INT32_MAX ||
        output_count * (uint64_t)sizeof(int32_t) > (uint64_t)INT32_MAX)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    /* A zero output count has a zero factor, whose cast stays zero. */
    *outer = (size_t)outer64;
    *inner = (size_t)inner64;
    return ARM_CMSIS_NN_SUCCESS;
}

/* The extremum's index along `axis` of a 4-D array of binary16 or binary32
 * encodings. The argument checks are straight-line and divide nothing
 * (arm_nn_arg_extrema_lines), the storage width and the sense of the
 * comparison are fixed outside the element loop, and the element loop
 * carries one branch, its own. Measured on the farm's Apollo510B before
 * the small-extent bound, the entry took 196 cycles for one line of K = 4
 * scores; what the bound changes is unmeasured. Width and `maximum` are
 * literals at every call, which is what makes each wrapper's body one
 * specialised loop. */
static inline arm_cmsis_nn_status arm_nn_arg_extrema(const void *input_data,
                                                     const cmsis_nn_dims *input_dims,
                                                     int32_t axis,
                                                     int32_t *output_data,
                                                     size_t width,
                                                     bool maximum)
{
    size_t outer;
    size_t inner;
    size_t reduction;
    if (arm_nn_arg_extrema_lines(input_dims, axis, width, &outer, &inner, &reduction) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    /* The reduced extent is at least 1 and the input's count is the output's times it, without
     * a wrap, so the input is empty exactly when the output is. */
    const size_t lines = outer * inner;
    if (lines && (!input_data || !output_data))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!lines)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
    const uint8_t *input = (const uint8_t *)input_data;
    const size_t line_bytes = reduction * inner * width;
    for (size_t o = 0; o < outer; ++o)
    {
        const uint8_t *block = input + o * line_bytes;
        int32_t *out = output_data + o * inner;
        for (size_t i = 0; i < inner; ++i)
        {
            const uint8_t *line = block + i * width;
            out[i] = width == sizeof(uint16_t) ? arm_nn_arg_line_16(line, inner, reduction, maximum)
                                               : arm_nn_arg_line_32(line, inner, reduction, maximum);
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_FLOAT_API_ENABLED */
#endif /* ARM_NN_ARG_EXTREMA_FLT_H */
