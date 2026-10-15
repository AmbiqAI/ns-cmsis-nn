/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_LOGICAL_COMMON_H
#define ARM_NN_LOGICAL_COMMON_H

#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnsupportfunctions.h"
#include <limits.h>
#include <stdbool.h>
#include <string.h>

/* Boolean object representations are one-byte 0/1 on the supported Arm ABI. */
typedef char arm_nn_logical_bool_is_one_byte[(sizeof(bool) == 1) ? 1 : -1];

static inline bool arm_nn_logical_dims_valid(const cmsis_nn_dims *a, const cmsis_nn_dims *b, const cmsis_nn_dims *out)
{
    if (!a || !b || !out || !arm_nn_broadcast_dims_valid(a, b, out))
    {
        return false;
    }
    int32_t count = out->n;
    const int32_t dims[3] = {out->h, out->w, out->c};
    for (int i = 0; i < 3; ++i)
    {
        if (count > INT32_MAX / dims[i])
        {
            return false;
        }
        count *= dims[i];
    }
    return true;
}

/* is_or is constant at each direct entry; no per-element operation dispatch. */
__STATIC_FORCEINLINE void arm_nn_logical_binary(const bool *a, const bool *b, bool *out, int32_t count, bool is_or)
{
    uint32_t i = 0;
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (; i < (uint32_t)count; i += 16)
    {
        const mve_pred16_t p = vctp8q((uint32_t)count - i);
        const uint8x16_t va = vldrbq_z_u8((const uint8_t *)a + i, p);
        const uint8x16_t vb = vldrbq_z_u8((const uint8_t *)b + i, p);
        const uint8x16_t result = is_or ? vorrq_u8(va, vb) : vandq_u8(va, vb);
        vstrbq_p_u8((uint8_t *)out + i, result, p);
    }
#else
    #if defined(ARM_MATH_DSP)
    for (; i + 4 <= (uint32_t)count; i += 4)
    {
        uint32_t va, vb;
        memcpy(&va, a + i, sizeof(va));
        memcpy(&vb, b + i, sizeof(vb));
        const uint32_t result = is_or ? (va | vb) : (va & vb);
        memcpy(out + i, &result, sizeof(result));
    }
    #endif
    for (; i < (uint32_t)count; ++i)
    {
        out[i] = is_or ? (a[i] || b[i]) : (a[i] && b[i]);
    }
#endif
}

__STATIC_FORCEINLINE void
arm_nn_logical_scalar(const bool *scalar, const bool *vec, bool *out, int32_t count, bool is_or)
{
    if (*scalar == is_or)
    {
        memset(out, is_or ? 1 : 0, (size_t)count);
    }
    else
    {
        memcpy(out, vec, (size_t)count);
    }
}

/* Both choices are constant at each direct entry/callback. */
__STATIC_FORCEINLINE void
arm_nn_logical_compare(const bool *a, const bool *b, bool *out, int32_t count, bool is_equal, bool scalar_b)
{
    uint32_t i = 0;
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    const uint8x16_t invert = vdupq_n_u8(is_equal ? 1 : 0);
    /* A signed countdown permits low-overhead tail-predicated loop formation. */
    for (int32_t remaining = count; remaining > 0; remaining -= 16, i += 16)
    {
        const mve_pred16_t p = vctp8q((uint32_t)remaining);
        const uint8x16_t va = vldrbq_z_u8((const uint8_t *)a + i, p);
        const uint8x16_t vb = scalar_b ? vdupq_n_u8(*b) : vldrbq_z_u8((const uint8_t *)b + i, p);
        vstrbq_p_u8((uint8_t *)out + i, veorq_u8(veorq_u8(va, vb), invert), p);
    }
#else
    #if defined(ARM_MATH_DSP)
    for (; i + 4 <= (uint32_t)count; i += 4)
    {
        uint32_t va, vb;
        memcpy(&va, a + i, sizeof(va));
        if (scalar_b)
        {
            vb = *b ? 0x01010101u : 0;
        }
        else
        {
            memcpy(&vb, b + i, sizeof(vb));
        }
        const uint32_t result = va ^ vb ^ (is_equal ? 0x01010101u : 0);
        memcpy(out + i, &result, sizeof(result));
    }
    #endif
    for (; i < (uint32_t)count; ++i)
    {
        out[i] = (a[i] == b[scalar_b ? 0 : i]) == is_equal;
    }
#endif
}

#endif /* ARM_NN_LOGICAL_COMMON_H */
