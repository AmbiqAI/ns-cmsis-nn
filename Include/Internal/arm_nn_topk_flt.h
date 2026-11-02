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

/* TOPK_V2 over the innermost extent. Each line is an insertion into the caller's outputs
 * in index order, so the outputs hold the line's best so far and no scratch is needed.
 * A candidate displaces a kept element only on a strictly greater rank, so equal ranks,
 * and NaNs among themselves, keep ascending index order. */
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
