/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_REDUCE_BOOL_H
#define ARM_NN_REDUCE_BOOL_H

#include "Internal/arm_nn_logical_common.h"

static inline bool arm_nn_reduce_bool_count(const int32_t dims[4], int32_t *count)
{
    bool empty = false;
    for (int i = 0; i < 4; ++i)
    {
        if (dims[i] < 0)
        {
            return false;
        }
        empty |= dims[i] == 0;
    }
    *count = empty ? 0 : 1;
    if (!empty)
    {
        for (int i = 0; i < 4; ++i)
        {
            if (*count > INT32_MAX / dims[i])
            {
                return false;
            }
            *count *= dims[i];
        }
    }
    return true;
}

static inline bool arm_nn_reduce_bool_dims_valid(const cmsis_nn_dims *input,
                                                 const cmsis_nn_dims *axis,
                                                 const cmsis_nn_dims *output,
                                                 int32_t *input_count,
                                                 int32_t *output_count)
{
    if (!input || !axis || !output)
    {
        return false;
    }
    const int32_t d[4] = {input->n, input->h, input->w, input->c};
    const int32_t a[4] = {axis->n, axis->h, axis->w, axis->c};
    const int32_t o[4] = {output->n, output->h, output->w, output->c};
    for (int i = 0; i < 4; ++i)
    {
        if ((a[i] != 0 && a[i] != 1) || o[i] != (a[i] ? 1 : d[i]))
        {
            return false;
        }
    }
    return arm_nn_reduce_bool_count(d, input_count) && arm_nn_reduce_bool_count(o, output_count);
}

/* Stop once a value differs from the reduction identity. */
__STATIC_FORCEINLINE bool arm_nn_reduce_bool_block(const bool *input, int32_t count, bool is_all)
{
    uint32_t i = 0;
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (; i < (uint32_t)count; i += 16)
    {
        const mve_pred16_t p = vctp8q((uint32_t)count - i);
        const uint8x16_t v = vldrbq_z_u8((const uint8_t *)input + i, p);
        if (vcmpneq_n_u8(v, is_all ? 1 : 0) & p)
        {
            return !is_all;
        }
    }
#else
    #if defined(ARM_MATH_DSP)
    for (; i + 4 <= (uint32_t)count; i += 4)
    {
        uint32_t v;
        memcpy(&v, input + i, sizeof(v));
        if (v != (is_all ? 0x01010101u : 0))
        {
            return !is_all;
        }
    }
    #endif
    for (; i < (uint32_t)count; ++i)
    {
        if (input[i] != is_all)
        {
            return !is_all;
        }
    }
#endif
    return is_all;
}

static inline void arm_nn_reduce_bool(const bool *input,
                                      const cmsis_nn_dims *input_dims,
                                      const cmsis_nn_dims *axis,
                                      bool *output,
                                      const cmsis_nn_dims *output_dims,
                                      int32_t input_count,
                                      int32_t output_count,
                                      bool is_all)
{
    if (output_count == 0)
    {
        return;
    }
    if (!(axis->n || axis->h || axis->w || axis->c))
    {
        memcpy(output, input, (size_t)output_count);
        return;
    }
    memset(output, is_all ? 1 : 0, (size_t)output_count);
    if (input_count == 0)
    {
        return;
    }
    int32_t d[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t a[4] = {axis->n, axis->h, axis->w, axis->c};
    int32_t block = d[3];
    if (axis->c)
    {
        /* Merge contiguous reduced trailing axes, retaining the same output layout. */
        for (int i = 2; i >= 0 && a[i]; --i)
        {
            block *= d[i];
            d[i] = 1;
        }
    }
    for (int32_t n = 0; n < d[0]; ++n)
    {
        for (int32_t h = 0; h < d[1]; ++h)
        {
            for (int32_t w = 0; w < d[2]; ++w)
            {
                const int32_t in_offset = ((n * d[1] + h) * d[2] + w) * block;
                const int32_t out_offset =
                    (((axis->n ? 0 : n) * output_dims->h + (axis->h ? 0 : h)) * output_dims->w + (axis->w ? 0 : w)) *
                    output_dims->c;
                if (axis->c)
                {
                    if (output[out_offset] == is_all)
                    {
                        output[out_offset] = arm_nn_reduce_bool_block(input + in_offset, block, is_all);
                    }
                }
                else
                {
                    arm_nn_logical_binary(input + in_offset, output + out_offset, output + out_offset, block, !is_all);
                }
            }
        }
    }
}

#endif /* ARM_NN_REDUCE_BOOL_H */
