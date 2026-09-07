/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_axis_copy_common.h
 * Description:  Rank-agnostic axis slice walker shared by the float
 *               split / unpack / concatenation / pack kernels
 *
 * $Date:        6 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_AXIS_COPY_COMMON_H
#define ARM_NN_AXIS_COPY_COMMON_H

#include "arm_nn_types.h"
#include "arm_nnsupportfunctions.h"

#include <stdint.h>

/*
 * A row-major tensor of `dims` dimensions viewed at `axis` is
 *   outer = prod(shape[0..axis))  rows of  axis_len * inner  contiguous elements,
 *   inner = prod(shape[inner_begin..dims)),
 * and slice s of a row (sizes[s] wide along the axis, or one element wide when
 * sizes == NULL) is sizes[s] * inner contiguous elements. Every copy is
 * therefore one memcpy run and the same two loops serve every rank and axis.
 * Scatter (split, unpack) and gather (concatenation, pack) are the two
 * directions of that walk; pack/unpack pass inner_begin == axis (the axis is
 * inserted rather than present in `shape`). See AmbiqAI/ns-cmsis-nn#411.
 *
 * Validation: shape entries >= 0, num >= 1, sizes entries >= 0, sum(sizes) (or
 * num) == axis_len, and the element count fits in int32_t. Returns
 * ARM_CMSIS_NN_ARG_ERROR without writing outer/inner on failure.
 */
__STATIC_FORCEINLINE arm_cmsis_nn_status arm_nn_axis_copy_plan(const int32_t *shape,
                                                               const int32_t dims,
                                                               const int32_t axis,
                                                               const int32_t inner_begin,
                                                               const int32_t axis_len,
                                                               const int32_t num,
                                                               const int32_t *sizes,
                                                               int32_t *outer,
                                                               int32_t *inner)
{
    if ((shape == NULL && dims > 0) || dims < 0 || axis < 0 || axis > dims || inner_begin < axis ||
        inner_begin > dims || axis_len < 0 || num < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int64_t o = 1;
    for (int32_t d = 0; d < axis; d++)
    {
        if (shape[d] < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        o *= shape[d];
        if (o > INT32_MAX)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    int64_t in = 1;
    for (int32_t d = inner_begin; d < dims; d++)
    {
        if (shape[d] < 0)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        in *= shape[d];
        if (in > INT32_MAX)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    int64_t sum = 0;
    if (sizes != NULL)
    {
        for (int32_t s = 0; s < num; s++)
        {
            if (sizes[s] < 0)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            sum += sizes[s];
        }
    }
    else
    {
        sum = num;
    }
    if (sum != axis_len)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    /* Stepwise: each factor is <= INT32_MAX, so every partial product fits int64. */
    int64_t total = o * axis_len;
    if (total > INT32_MAX)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    total *= in;
    if (total > INT32_MAX)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    *outer = (int32_t)o;
    *inner = (int32_t)in;
    return ARM_CMSIS_NN_SUCCESS;
}

/* Non-NULL `num` entries; NULL is tolerated only when there is nothing to copy. */
__STATIC_FORCEINLINE int32_t arm_nn_axis_copy_ptrs_ok(const void *const *ptrs, const int32_t num, const int32_t total)
{
    if (ptrs == NULL)
    {
        return total == 0;
    }
    for (int32_t s = 0; s < num; s++)
    {
        if (ptrs[s] == NULL && total != 0)
        {
            return 0;
        }
    }
    return 1;
}

/*
 * arm_nn_axis_scatter_<SUFFIX>: packed -> slices (split, unpack).
 * arm_nn_axis_gather_<SUFFIX>:  slices -> packed (concatenation, pack).
 * Both are bit copies through arm_memcpy_<SUFFIX>; NaN/Inf/-0/subnormal
 * payloads are preserved. Operands must not overlap.
 */
#define ARM_NN_AXIS_COPY_DEFINE(SUFFIX, TYPE)                                                                          \
    __STATIC_FORCEINLINE void arm_nn_axis_scatter_##SUFFIX(const TYPE *packed,                                         \
                                                           const int32_t outer,                                        \
                                                           const int32_t num,                                          \
                                                           const int32_t *sizes,                                       \
                                                           const int32_t inner,                                        \
                                                           TYPE *const *slices)                                        \
    {                                                                                                                  \
        for (int32_t k = 0; k < outer; k++)                                                                            \
        {                                                                                                              \
            for (int32_t s = 0; s < num; s++)                                                                          \
            {                                                                                                          \
                const int32_t run = (sizes != NULL ? sizes[s] : 1) * inner;                                            \
                arm_memcpy_##SUFFIX(slices[s] + (size_t)k * (size_t)run, packed, (uint32_t)run);                       \
                packed += run;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    __STATIC_FORCEINLINE void arm_nn_axis_gather_##SUFFIX(const TYPE *const *slices,                                   \
                                                          const int32_t outer,                                         \
                                                          const int32_t num,                                           \
                                                          const int32_t *sizes,                                        \
                                                          const int32_t inner,                                         \
                                                          TYPE *packed)                                                \
    {                                                                                                                  \
        for (int32_t k = 0; k < outer; k++)                                                                            \
        {                                                                                                              \
            for (int32_t s = 0; s < num; s++)                                                                          \
            {                                                                                                          \
                const int32_t run = (sizes != NULL ? sizes[s] : 1) * inner;                                            \
                arm_memcpy_##SUFFIX(packed, slices[s] + (size_t)k * (size_t)run, (uint32_t)run);                       \
                packed += run;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
    }

#if ARM_NN_ENABLE_F32
ARM_NN_AXIS_COPY_DEFINE(f32, float32_t)
#endif
#if ARM_NN_ENABLE_F16
ARM_NN_AXIS_COPY_DEFINE(f16, float16_t)
#endif

#endif /* ARM_NN_AXIS_COPY_COMMON_H */
