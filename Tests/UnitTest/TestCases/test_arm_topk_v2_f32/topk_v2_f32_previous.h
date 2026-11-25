/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The TOPK_V2 body as it was at 98505c5e (Include/Internal/arm_nn_topk_flt.h, with the rank
 * key of Include/Internal/arm_nn_arg_extrema_flt.h), every helper renamed and copied so that
 * no later change to the library reaches it. It is compiled only by this suite, as the
 * reference the differential cases hold the entry to: values, indices and status. */

#include <arm_nnfunctions.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline uint32_t tp_key(uint32_t bits, uint32_t sign, uint32_t lose)
{
    const uint32_t magnitude_mask = sign - 1;
    const uint32_t infinity = sign == UINT32_C(0x8000) ? UINT32_C(0x7c00) : UINT32_C(0x7f800000);
    const uint32_t magnitude = bits & magnitude_mask;
    const uint32_t flipped = (bits & sign) ? (~bits & (sign | magnitude_mask)) : (bits | sign);
    const uint32_t key = magnitude == 0 ? sign : flipped;
    return magnitude > infinity ? lose : key;
}

static inline bool tp_count(const int32_t dims[4], size_t width, size_t *count)
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

static inline uint32_t tp_load(const uint8_t *input, size_t width)
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

static inline void tp_store(uint8_t *output, uint32_t bits, size_t width)
{
    if (width == sizeof(uint16_t))
    {
        const uint16_t half = (uint16_t)bits;
        memcpy(output, &half, sizeof(half));
        return;
    }
    memcpy(output, &bits, sizeof(bits));
}

static inline uint32_t tp_rank(uint32_t bits, size_t width)
{
    const uint32_t sign = width == sizeof(uint16_t) ? UINT32_C(0x8000) : UINT32_C(0x80000000);
    return tp_key(bits, sign, 0);
}

static inline arm_cmsis_nn_status tp_topk(const void *input_data,
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
    if (!tp_count(dims, width, &input_count) || !tp_count(out, width, &value_count) ||
        !tp_count(out, sizeof(*output_indices), &index_count))
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
            const uint32_t bits = tp_load(row + i * width, width);
            const uint32_t rank = tp_rank(bits, width);
            if (kept == kept_max && rank <= worst)
            {
                continue;
            }
            size_t position = kept < kept_max ? kept++ : kept_max - 1;
            while (position > 0)
            {
                const uint32_t above = tp_load(row_values + (position - 1) * width, width);
                if (tp_rank(above, width) >= rank)
                {
                    break;
                }
                tp_store(row_values + position * width, above, width);
                row_indices[position] = row_indices[position - 1];
                --position;
            }
            tp_store(row_values + position * width, bits, width);
            row_indices[position] = (int32_t)i;
            if (kept == kept_max)
            {
                worst = tp_rank(tp_load(row_values + (kept_max - 1) * width, width), width);
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
