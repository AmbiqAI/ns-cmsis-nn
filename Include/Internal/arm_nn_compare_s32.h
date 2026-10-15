/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#ifndef ARM_NN_COMPARE_S32_H
#define ARM_NN_COMPARE_S32_H

#include "Internal/arm_nn_broadcast_walk.h"
#include "arm_nnsupportfunctions.h"
#include <limits.h>
#include <stdbool.h>

typedef char arm_nn_compare_bool_is_one_byte[(sizeof(bool) == 1) ? 1 : -1];

static inline bool
arm_nn_compare_s32_dims_valid(const cmsis_nn_dims *a, const cmsis_nn_dims *b, const cmsis_nn_dims *out)
{
    if (!a || !b || !out || !arm_nn_broadcast_dims_valid(a, b, out))
    {
        return false;
    }
    int32_t count = 1;
    const int32_t dims[4] = {out->n, out->h, out->w, out->c};
    for (int i = 0; i < 4; ++i)
    {
        if (count > (INT32_MAX / (int32_t)sizeof(int32_t)) / dims[i])
        {
            return false;
        }
        count *= dims[i];
    }
    return true;
}

/* operation and scalar choices are fixed by each direct entry's callbacks. */
__STATIC_FORCEINLINE void arm_nn_compare_s32(const int32_t *a,
                                             const int32_t *b,
                                             bool *out,
                                             int32_t count,
                                             arm_nn_compare_operation operation,
                                             bool scalar_a,
                                             bool scalar_b)
{
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
    uint32_t i = 0;
    for (int32_t remaining = count; remaining > 0; remaining -= 4, i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)remaining);
        const int32x4_t va = scalar_a ? vdupq_n_s32(*a) : vld1q_z(a + i, p);
        const int32x4_t vb = scalar_b ? vdupq_n_s32(*b) : vld1q_z(b + i, p);
        mve_pred16_t match;
        switch (operation)
        {
        case ARM_COMPARE_EQUAL:
            match = vcmpeqq_s32(va, vb);
            break;
        case ARM_COMPARE_NOT_EQUAL:
            match = vcmpneq_s32(va, vb);
            break;
        case ARM_COMPARE_GREATER:
            match = vcmpgtq_s32(va, vb);
            break;
        case ARM_COMPARE_GREATER_EQUAL:
            match = vcmpgeq_s32(va, vb);
            break;
        case ARM_COMPARE_LESS:
            match = vcmpltq_s32(va, vb);
            break;
        case ARM_COMPARE_LESS_EQUAL:
            match = vcmpleq_s32(va, vb);
            break;
        default:
            match = 0;
            break;
        }
        const uint32x4_t result = vpselq(vdupq_n_u32(1), vdupq_n_u32(0), match);
        vstrbq_p_u32((uint8_t *)out + i, result, p);
    }
#else
    for (int32_t i = 0; i < count; ++i)
    {
        const int32_t va = a[scalar_a ? 0 : i];
        const int32_t vb = b[scalar_b ? 0 : i];
        switch (operation)
        {
        case ARM_COMPARE_EQUAL:
            out[i] = va == vb;
            break;
        case ARM_COMPARE_NOT_EQUAL:
            out[i] = va != vb;
            break;
        case ARM_COMPARE_GREATER:
            out[i] = va > vb;
            break;
        case ARM_COMPARE_GREATER_EQUAL:
            out[i] = va >= vb;
            break;
        case ARM_COMPARE_LESS:
            out[i] = va < vb;
            break;
        case ARM_COMPARE_LESS_EQUAL:
            out[i] = va <= vb;
            break;
        default:
            out[i] = false;
            break;
        }
    }
#endif
}
#endif
