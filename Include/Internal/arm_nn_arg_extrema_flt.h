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
#include <string.h>

#if ARM_NN_FLOAT_API_ENABLED

/* All signs are checked before counting. A zero extent suppresses the product,
 * even if a preceding nonzero prefix would exceed the byte limit. */
static inline bool arm_nn_arg_count(const int32_t dims[4], size_t width, size_t *count)
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

static inline uint32_t arm_nn_arg_load(const uint8_t *input, size_t width)
{
    if (width == 2)
    {
        uint16_t bits;
        memcpy(&bits, input, sizeof(bits));
        return bits;
    }
    uint32_t bits;
    memcpy(&bits, input, sizeof(bits));
    return bits;
}

/* Both zero signs share a key. Reversing negative encodings preserves numeric
 * ordering, including subnormals, without floating-point comparisons. */
static inline uint32_t arm_nn_arg_key(uint32_t bits, uint32_t sign)
{
    if ((bits & (sign - 1)) == 0)
    {
        return sign;
    }
    return (bits & sign) ? (~bits & (sign | (sign - 1))) : (bits | sign);
}

static inline arm_cmsis_nn_status arm_nn_arg_extrema(const void *input_data,
                                                     const cmsis_nn_dims *input_dims,
                                                     int32_t axis,
                                                     int32_t *output_data,
                                                     size_t width,
                                                     bool maximum)
{
    if (!input_dims || axis < 0 || axis > 3)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t dims[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    int32_t out[4] = {dims[0], dims[1], dims[2], dims[3]};
    for (int32_t d = 0; d < 4; ++d)
    {
        if (dims[d] < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    /* There is no valid index for an empty reduced axis, even if output is empty. */
    if (dims[axis] == 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    out[axis] = 1;
    size_t input_count;
    size_t output_count;
    if (!arm_nn_arg_count(dims, width, &input_count) || !arm_nn_arg_count(out, sizeof(*output_data), &output_count) ||
        (input_count && !input_data) || (output_count && !output_data))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (!output_count)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }

    size_t outer = 1;
    size_t inner = 1;
    for (int32_t d = 0; d < axis; ++d)
    {
        outer *= (size_t)dims[d];
    }
    for (int32_t d = axis + 1; d < 4; ++d)
    {
        inner *= (size_t)dims[d];
    }
    const size_t reduction = (size_t)dims[axis];
    const uint32_t sign = width == 2 ? UINT32_C(0x8000) : UINT32_C(0x80000000);
    const uint32_t infinity = width == 2 ? UINT32_C(0x7c00) : UINT32_C(0x7f800000);
    const uint8_t *input = (const uint8_t *)input_data;
    for (size_t o = 0; o < outer; ++o)
    {
        for (size_t i = 0; i < inner; ++i)
        {
            const size_t base = o * reduction * inner + i;
            uint32_t best_key = 0;
            int32_t best_index = 0;
            for (size_t k = 0; k < reduction; ++k)
            {
                const uint32_t bits = arm_nn_arg_load(input + (base + k * inner) * width, width);
                if ((bits & (sign - 1)) > infinity)
                {
                    best_index = (int32_t)k;
                    break;
                }
                const uint32_t key = arm_nn_arg_key(bits, sign);
                if (k == 0 || (maximum ? key > best_key : key < best_key))
                {
                    best_key = key;
                    best_index = (int32_t)k;
                }
            }
            output_data[o * inner + i] = best_index;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_FLOAT_API_ENABLED */
#endif /* ARM_NN_ARG_EXTREMA_FLT_H */
