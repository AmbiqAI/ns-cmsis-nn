/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_TOPK_FLT_H
#define ARM_NN_TOPK_FLT_H

#include "Internal/arm_nn_arg_extrema_flt.h"
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if ARM_NN_FLOAT_API_ENABLED

/* All signs are checked before counting. A zero extent suppresses the product,
 * even if a preceding nonzero prefix would exceed the byte limit. Each step
 * bounds the next factor by a quotient before it multiplies, so no product
 * is formed that could exceed INT32_MAX, let alone wrap. */
static inline bool arm_nn_topk_count(const int32_t dims[4], size_t width, size_t *count)
{
    for (int32_t d = 0; d < 4; ++d)
    {
        if (dims[d] == 0)
        {
            *count = 0;
            return true;
        }
    }
    *count = 1;
    for (int32_t d = 0; d < 4; ++d)
    {
        if ((size_t)dims[d] > ((size_t)INT32_MAX / width) / *count)
        {
            return false;
        }
        *count *= (size_t)dims[d];
    }
    return true;
}

static inline uint32_t arm_nn_topk_load(const uint8_t *input, size_t width)
{
    if (width == sizeof(uint16_t))
    {
        uint16_t bits;
        memcpy(&bits, input, sizeof(bits));
        return bits;
    }
    uint32_t bits;
    memcpy(&bits, input, sizeof(bits));
    return bits;
}

static inline void arm_nn_topk_store(uint8_t *output, uint32_t bits, size_t width)
{
    if (width == sizeof(uint16_t))
    {
        const uint16_t half = (uint16_t)bits;
        memcpy(output, &half, sizeof(half));
        return;
    }
    memcpy(output, &bits, sizeof(bits));
}

/* One unsigned rank carries the whole order: arm_nn_arg_key for a number, so both zeros
 * share a rank, and 0 for a NaN, the key a maximum's NaN loses with. The smallest number's
 * key, -inf's, is 0x007fffff in binary32 and 0x03ff in binary16, so every number outranks
 * every NaN. The key is the arg reductions' own, so top-k at k = 1 and argmax keep one order. */
static inline uint32_t arm_nn_topk_rank(uint32_t bits, size_t width)
{
    const uint32_t sign = width == sizeof(uint16_t) ? UINT32_C(0x8000) : UINT32_C(0x80000000);
    return arm_nn_arg_key(bits, sign, 0);
}

/* Places `bits`, of rank `rank`, among the kept elements of a binary32 line, which hold the
 * line's best so far in descending rank order. `position` is the free slot: the next one
 * while fewer than k are kept, the worst's once k are. Each kept element of a lower rank
 * moves down one slot, and the first of an equal or greater rank stops the candidate, so
 * equal ranks keep ascending index order. It is the insertion the binary16 line in
 * arm_nn_topk spells inline. */
static inline void arm_nn_topk_insert(uint8_t *row_values,
                                      int32_t *row_indices,
                                      size_t position,
                                      uint32_t bits,
                                      uint32_t rank,
                                      size_t index,
                                      size_t width)
{
    while (position > 0)
    {
        const uint32_t above = arm_nn_topk_load(row_values + (position - 1) * width, width);
        if (arm_nn_topk_rank(above, width) >= rank)
        {
            break;
        }
        arm_nn_topk_store(row_values + position * width, above, width);
        row_indices[position] = row_indices[position - 1];
        --position;
    }
    arm_nn_topk_store(row_values + position * width, bits, width);
    row_indices[position] = (int32_t)index;
}

/* The rank of the worst of the k kept elements, the bar a later candidate must clear. */
static inline uint32_t arm_nn_topk_worst(const uint8_t *row_values, size_t kept_max, size_t width)
{
    return arm_nn_topk_rank(arm_nn_topk_load(row_values + (kept_max - 1) * width, width), width);
}

/* One element of the steady phase of a binary32 line, k already kept: it enters only on a
 * rank strictly above the worst, and the new worst is returned. */
static inline uint32_t arm_nn_topk_admit_32(const uint8_t *element,
                                            uint32_t rank,
                                            uint32_t worst,
                                            size_t index,
                                            uint8_t *row_values,
                                            int32_t *row_indices,
                                            size_t kept_max)
{
    if (rank <= worst)
    {
        return worst;
    }
    const size_t width = sizeof(uint32_t);
    arm_nn_topk_insert(row_values, row_indices, kept_max - 1, arm_nn_topk_load(element, width), rank, index, width);
    return arm_nn_topk_worst(row_values, kept_max, width);
}

/* arm_nn_topk_rank of one binary32 encoding without a branch, by the integer masks of
 * arm_nn_topk_rank_x4: magnitude - 1 has its top bit set only for a zero, which clears the
 * sign of -0; the key flips every bit of a negative encoding and the sign bit of any other;
 * infinity - magnitude is negative only for a NaN, whose key is then cleared to 0. It
 * equals arm_nn_arg_key(bits, 0x80000000, 0) on every encoding. */
static inline uint32_t arm_nn_topk_rank_32(uint32_t bits)
{
    const uint32_t sign = UINT32_C(0x80000000);
    const uint32_t magnitude = bits & ~sign;
    const uint32_t number = bits & ~((magnitude - 1) & sign);
    const uint32_t key = number ^ ((0u - (number >> 31)) | sign);
    return key & ~(0u - ((UINT32_C(0x7f800000) - magnitude) >> 31));
}

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
/* arm_nn_topk_rank_32 for four binary32 encodings, lane for lane, with no lane select, so
 * the only predicate live in the steady loop is the tail's. */
static inline uint32x4_t arm_nn_topk_rank_x4(uint32x4_t bits)
{
    const uint32x4_t sign = vdupq_n_u32(UINT32_C(0x80000000));
    const uint32x4_t magnitude = vandq_u32(bits, vdupq_n_u32(UINT32_C(0x7fffffff)));
    const uint32x4_t number = vbicq_u32(bits, vandq_u32(vsubq_n_u32(magnitude, 1), sign));
    const uint32x4_t negative = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(number), 31));
    const uint32x4_t key = veorq_u32(number, vorrq_u32(negative, sign));
    const uint32x4_t above_infinity = vsubq_u32(vdupq_n_u32(UINT32_C(0x7f800000)), magnitude);
    const uint32x4_t nan = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(above_infinity), 31));
    return vbicq_u32(key, nan);
}
    #endif

/* A binary32 line in two phases. The first k elements are kept whatever their ranks (the
 * fill phase). After them (the steady phase) the ranks of four elements are formed at once
 * and compared, unsigned, with the worst kept; a block with no rank above it is passed over
 * whole, and any other goes through the insertion element by element in index order, each
 * against the worst its block's earlier insertions left. In the steady phase the worst never
 * falls, so a block with no rank above the worst at its start holds no element the
 * element-wise test would keep, and the outputs are those of that test. The comparison is
 * on the integer keys, not on floats, so a fast-math build cannot fold the NaN rule away. */
static inline void
arm_nn_topk_line_32(const uint8_t *row, size_t line, size_t kept_max, uint8_t *row_values, int32_t *row_indices)
{
    const size_t width = sizeof(uint32_t);
    for (size_t i = 0; i < kept_max; ++i)
    {
        const uint32_t bits = arm_nn_topk_load(row + i * width, width);
        arm_nn_topk_insert(row_values, row_indices, i, bits, arm_nn_topk_rank(bits, width), i, width);
    }
    uint32_t worst = arm_nn_topk_worst(row_values, kept_max, width);
    size_t i = kept_max;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* `row` is the float32_t pointer the entry took plus whole elements; the cast through
     * void * says so to -Wcast-align. A lane past the line is neither read nor admitted. */
    const float32_t *elements = (const float32_t *)(const void *)row;
    for (; i < line; i += 4)
    {
        const mve_pred16_t tail = vctp32q((uint32_t)(line - i));
        const uint32x4_t ranks = arm_nn_topk_rank_x4(vreinterpretq_u32_f32(vldrwq_z_f32(elements + i, tail)));
        if (!vcmphiq_m_n_u32(ranks, worst, tail))
        {
            continue;
        }
        uint32_t lane_ranks[4];
        arm_nn_mve_store_word_bits(lane_ranks, ranks, tail);
        const size_t lanes = line - i < 4 ? line - i : 4;
        for (size_t j = 0; j < lanes; ++j)
        {
            worst = arm_nn_topk_admit_32(
                row + (i + j) * width, lane_ranks[j], worst, i + j, row_values, row_indices, kept_max);
        }
    }
    #else
    for (; i + 4 <= line; i += 4)
    {
        const uint8_t *block = row + i * width;
        const uint32_t rank0 = arm_nn_topk_rank_32(arm_nn_topk_load(block, width));
        const uint32_t rank1 = arm_nn_topk_rank_32(arm_nn_topk_load(block + width, width));
        const uint32_t rank2 = arm_nn_topk_rank_32(arm_nn_topk_load(block + 2 * width, width));
        const uint32_t rank3 = arm_nn_topk_rank_32(arm_nn_topk_load(block + 3 * width, width));
        if ((rank0 > worst) | (rank1 > worst) | (rank2 > worst) | (rank3 > worst))
        {
            const uint32_t ranks[4] = {rank0, rank1, rank2, rank3};
            for (size_t j = 0; j < 4; ++j)
            {
                worst = arm_nn_topk_admit_32(
                    row + (i + j) * width, ranks[j], worst, i + j, row_values, row_indices, kept_max);
            }
        }
    }
    for (; i < line; ++i)
    {
        const uint32_t rank = arm_nn_topk_rank(arm_nn_topk_load(row + i * width, width), width);
        worst = arm_nn_topk_admit_32(row + i * width, rank, worst, i, row_values, row_indices, kept_max);
    }
    #endif
}

/* TOPK_V2 over the innermost extent. Each line is an insertion into the caller's outputs
 * in index order, so the outputs hold the line's best so far and no scratch is needed.
 * A candidate displaces a kept element only on a strictly greater rank, so equal ranks,
 * and NaNs among themselves, keep ascending index order. A binary32 line is
 * arm_nn_topk_line_32, which keeps the same order with a test of four candidates at once. */
static inline arm_cmsis_nn_status arm_nn_topk(const void *input_data,
                                              const cmsis_nn_dims *input_dims,
                                              int32_t k,
                                              void *output_values,
                                              int32_t *output_indices,
                                              size_t width)
{
    if (!input_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t dims[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    for (int32_t d = 0; d < 4; ++d)
    {
        if (dims[d] < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    if (k < 0 || k > dims[3])
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t out[4] = {dims[0], dims[1], dims[2], k};
    size_t input_count;
    size_t value_count;
    size_t index_count;
    if (!arm_nn_topk_count(dims, width, &input_count) || !arm_nn_topk_count(out, width, &value_count) ||
        !arm_nn_topk_count(out, sizeof(*output_indices), &index_count))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!index_count)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
    if (!input_data || !output_values || !output_indices)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const size_t rows = (size_t)dims[0] * (size_t)dims[1] * (size_t)dims[2];
    const size_t line = (size_t)dims[3];
    const size_t kept_max = (size_t)k;
    const uint8_t *input = (const uint8_t *)input_data;
    uint8_t *values = (uint8_t *)output_values;
    if (width == sizeof(uint32_t))
    {
        for (size_t r = 0; r < rows; ++r)
        {
            arm_nn_topk_line_32(
                input + r * line * width, line, kept_max, values + r * kept_max * width, output_indices + r * kept_max);
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
    for (size_t r = 0; r < rows; ++r)
    {
        const uint8_t *row = input + r * line * width;
        uint8_t *row_values = values + r * kept_max * width;
        int32_t *row_indices = output_indices + r * kept_max;
        size_t kept = 0;
        uint32_t worst = 0;
        for (size_t i = 0; i < line; ++i)
        {
            const uint32_t bits = arm_nn_topk_load(row + i * width, width);
            const uint32_t rank = arm_nn_topk_rank(bits, width);
            if (kept == kept_max && rank <= worst)
            {
                continue;
            }
            size_t position = kept < kept_max ? kept++ : kept_max - 1;
            while (position > 0)
            {
                const uint32_t above = arm_nn_topk_load(row_values + (position - 1) * width, width);
                if (arm_nn_topk_rank(above, width) >= rank)
                {
                    break;
                }
                arm_nn_topk_store(row_values + position * width, above, width);
                row_indices[position] = row_indices[position - 1];
                --position;
            }
            arm_nn_topk_store(row_values + position * width, bits, width);
            row_indices[position] = (int32_t)i;
            if (kept == kept_max)
            {
                worst = arm_nn_topk_rank(arm_nn_topk_load(row_values + (kept_max - 1) * width, width), width);
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_FLOAT_API_ENABLED */
#endif /* ARM_NN_TOPK_FLT_H */
