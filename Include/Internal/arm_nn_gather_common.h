/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_GATHER_COMMON_H
#define ARM_NN_GATHER_COMMON_H

#include "arm_nnsupportfunctions.h"

#if ARM_NN_FLOAT_API_ENABLED

/* Checked plans and atomic index validation. Refs #493. */
typedef struct
{
    size_t output_count;
    size_t batches;
    size_t outer;
    size_t inner;
    size_t coords;
    int32_t axis_size;
} arm_nn_gather_plan;

typedef struct
{
    size_t output_count;
    size_t batches;
    size_t slices;
    size_t inner;
    size_t batch_stride;
    size_t strides[4];
    int32_t width;
    int32_t batch_dims;
} arm_nn_gather_nd_plan;

__STATIC_FORCEINLINE int32_t arm_nn_gather_shape(const cmsis_nn_dims *dims, int32_t rank, int32_t *shape)
{
    for (int32_t d = 0; d < rank; ++d)
    {
        shape[d] = arm_cmsis_nn_dim_at(dims, d);
        if (shape[d] < 0)
        {
            return 0;
        }
    }
    return 1;
}

/* The element count of a shape whose extents are non-negative, or 0 where its byte count would
 * pass INT32_MAX; `rank` is at most 4, the length of every shape here. Small extents take no
 * division: where every extent is below 2^7 (the OR of the extents is below 128) and an element
 * is at most four bytes, the count is at most 127^4 = 260,144,641, below 2^28 and so below
 * INT32_MAX / 4, which the bound below accepts; the product is then the count, zero where an
 * extent is zero. Otherwise the bound divides per extent. */
__STATIC_FORCEINLINE int32_t arm_nn_gather_count(const int32_t *shape, int32_t rank, size_t element_size, size_t *count)
{
    int32_t any = 0;
    size_t product = 1;
    for (int32_t d = 0; d < rank; ++d)
    {
        any |= shape[d];
        product *= (size_t)shape[d];
    }
    if (element_size <= sizeof(int32_t) && (uint32_t)any < (UINT32_C(1) << 7))
    {
        *count = product;
        return 1;
    }
    *count = 1;
    for (int32_t d = 0; d < rank; ++d)
    {
        if (shape[d] == 0)
        {
            *count = 0;
            return 1;
        }
    }
    const size_t limit = INT32_MAX / element_size;
    for (int32_t d = 0; d < rank; ++d)
    {
        if ((size_t)shape[d] > limit / *count)
        {
            return 0;
        }
        *count *= (size_t)shape[d];
    }
    return 1;
}

/* Used only after nonempty buffer counts have bounded every factor. */
__STATIC_FORCEINLINE size_t arm_nn_gather_product(const int32_t *shape, int32_t begin, int32_t end)
{
    size_t count = 1;
    for (int32_t d = begin; d < end; ++d)
    {
        count *= (size_t)shape[d];
    }
    return count;
}

__STATIC_FORCEINLINE arm_cmsis_nn_status arm_nn_gather_prepare(const void *input_data,
                                                               const cmsis_nn_dims *input_dims,
                                                               const int32_t *indices_data,
                                                               const cmsis_nn_dims *indices_dims,
                                                               const cmsis_nn_gather_params *params,
                                                               void *output_data,
                                                               const cmsis_nn_dims *output_dims,
                                                               size_t element_size,
                                                               arm_nn_gather_plan *plan)
{
    if (input_dims == NULL || indices_dims == NULL || params == NULL || output_dims == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t rank = params->input_rank;
    const int32_t coords_rank = params->coords_rank;
    if (rank < 1 || rank > 4 || coords_rank < 0 || coords_rank > 4)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int32_t axis = params->axis;
    int32_t batch = params->batch_dims;
    if (axis < 0)
    {
        axis += rank;
    }
    if (batch < 0)
    {
        batch += coords_rank;
    }
    if (axis < 0 || axis >= rank || batch < 0 || batch > coords_rank || batch > axis)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t output_rank = rank + coords_rank - batch - 1;
    if (output_rank > 4)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int32_t input_shape[4];
    int32_t indices_shape[4];
    int32_t output_shape[4] = {0};
    if (!arm_nn_gather_shape(input_dims, rank, input_shape) ||
        !arm_nn_gather_shape(indices_dims, coords_rank, indices_shape) ||
        !arm_nn_gather_shape(output_dims, output_rank, output_shape))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    for (int32_t d = 0; d < batch; ++d)
    {
        if (input_shape[d] != indices_shape[d])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    int32_t out = 0;
    for (int32_t d = 0; d < axis; ++d)
    {
        if (output_shape[out++] != input_shape[d])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    for (int32_t d = batch; d < coords_rank; ++d)
    {
        if (output_shape[out++] != indices_shape[d])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    for (int32_t d = axis + 1; d < rank; ++d)
    {
        if (output_shape[out++] != input_shape[d])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    size_t input_count;
    size_t indices_count;
    if (!arm_nn_gather_count(input_shape, rank, element_size, &input_count) ||
        !arm_nn_gather_count(indices_shape, coords_rank, sizeof(int32_t), &indices_count) ||
        !arm_nn_gather_count(output_shape, output_rank, element_size, &plan->output_count))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if ((input_data == NULL && input_count != 0) || (indices_data == NULL && indices_count != 0) ||
        (output_data == NULL && plan->output_count != 0))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    for (size_t i = 0; i < indices_count; ++i)
    {
        if (indices_data[i] < 0 || indices_data[i] >= input_shape[axis])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    if (plan->output_count == 0)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
    plan->batches = arm_nn_gather_product(input_shape, 0, batch);
    plan->outer = arm_nn_gather_product(input_shape, batch, axis);
    plan->inner = arm_nn_gather_product(input_shape, axis + 1, rank);
    plan->coords = arm_nn_gather_product(indices_shape, batch, coords_rank);
    plan->axis_size = input_shape[axis];
    return ARM_CMSIS_NN_SUCCESS;
}

__STATIC_FORCEINLINE arm_cmsis_nn_status arm_nn_gather_nd_prepare(const void *params_data,
                                                                  const cmsis_nn_dims *params_dims,
                                                                  const int32_t *indices_data,
                                                                  const cmsis_nn_dims *indices_dims,
                                                                  const cmsis_nn_gather_nd_params *params,
                                                                  void *output_data,
                                                                  const cmsis_nn_dims *output_dims,
                                                                  size_t element_size,
                                                                  arm_nn_gather_nd_plan *plan)
{
    if (params_dims == NULL || indices_dims == NULL || params == NULL || output_dims == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t rank = params->params_rank;
    const int32_t indices_rank = params->indices_rank;
    const int32_t batch = params->batch_dims;
    if (rank < 1 || rank > 4 || indices_rank < 1 || indices_rank > 4 || batch < 0 || batch >= indices_rank ||
        batch >= rank)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int32_t input_shape[4];
    int32_t indices_shape[4];
    int32_t output_shape[4] = {0};
    if (!arm_nn_gather_shape(params_dims, rank, input_shape) ||
        !arm_nn_gather_shape(indices_dims, indices_rank, indices_shape))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t width = indices_shape[indices_rank - 1];
    if (width < 1 || width > rank - batch)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t output_rank = indices_rank - 1 + rank - batch - width;
    if (output_rank > 4 || !arm_nn_gather_shape(output_dims, output_rank, output_shape))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    for (int32_t d = 0; d < batch; ++d)
    {
        if (input_shape[d] != indices_shape[d])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    int32_t out = 0;
    for (int32_t d = 0; d < indices_rank - 1; ++d)
    {
        if (output_shape[out++] != indices_shape[d])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    for (int32_t d = batch + width; d < rank; ++d)
    {
        if (output_shape[out++] != input_shape[d])
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    size_t input_count;
    size_t indices_count;
    if (!arm_nn_gather_count(input_shape, rank, element_size, &input_count) ||
        !arm_nn_gather_count(indices_shape, indices_rank, sizeof(int32_t), &indices_count) ||
        !arm_nn_gather_count(output_shape, output_rank, element_size, &plan->output_count))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if ((input_count == 0 && indices_count != 0) || (params_data == NULL && input_count != 0) ||
        (indices_data == NULL && indices_count != 0) || (output_data == NULL && plan->output_count != 0))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    for (size_t i = 0; i < indices_count; i += (size_t)width)
    {
        for (int32_t d = 0; d < width; ++d)
        {
            const int32_t coord = indices_data[i + (size_t)d];
            if (coord < 0 || coord >= input_shape[batch + d])
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
        }
    }
    if (plan->output_count == 0)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
    plan->batches = arm_nn_gather_product(input_shape, 0, batch);
    plan->slices = arm_nn_gather_product(indices_shape, batch, indices_rank - 1);
    plan->inner = arm_nn_gather_product(input_shape, batch + width, rank);
    plan->batch_stride = input_count / plan->batches;
    size_t stride = 1;
    for (int32_t d = rank - 1; d >= 0; --d)
    {
        plan->strides[d] = stride;
        stride *= (size_t)input_shape[d];
    }
    plan->width = width;
    plan->batch_dims = batch;
    return ARM_CMSIS_NN_SUCCESS;
}

    #if ARM_NN_ENABLE_F32
/* binary32 encodings read and written through this type, which may alias the float storage. */
typedef uint32_t arm_nn_gather_bits32_t __attribute__((__may_alias__));

/* One gathered run of `count` binary32 elements, the encodings' bits unchanged. A run under one
 * vector is copied directly: on Helium by one predicated load and store (a run of four
 * included), otherwise word by word; a longer run goes to arm_memcpy_f32. */
__STATIC_FORCEINLINE void arm_nn_gather_row_f32(float32_t *dst, const float32_t *src, uint32_t count)
{
        #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (count <= 4U)
    {
        const mve_pred16_t p = vctp32q(count);
        vstrwq_p(dst, vld1q_z(src, p), p);
        return;
    }
        #else
    if (count < 4U)
    {
        arm_nn_gather_bits32_t *to = (arm_nn_gather_bits32_t *)(void *)dst;
        const arm_nn_gather_bits32_t *from = (const arm_nn_gather_bits32_t *)(const void *)src;
        if (count > 0U)
        {
            to[0] = from[0];
        }
        if (count > 1U)
        {
            to[1] = from[1];
        }
        if (count > 2U)
        {
            to[2] = from[2];
        }
        return;
    }
        #endif
    arm_memcpy_f32(dst, src, count);
}
    #endif

    /* ROW copies one gathered run of GATHER; GATHER_ND's runs go to arm_memcpy_##SUFFIX. */
    #define ARM_NN_GATHER_COPY_DEFINE(SUFFIX, TYPE, ROW)                                                               \
        __STATIC_FORCEINLINE void arm_nn_gather_copy_##SUFFIX(                                                         \
            const TYPE *input, const int32_t *indices, TYPE *output, const arm_nn_gather_plan *plan)                   \
        {                                                                                                              \
            for (size_t batch = 0; batch < plan->batches; ++batch)                                                     \
            {                                                                                                          \
                for (size_t outer = 0; outer < plan->outer; ++outer)                                                   \
                {                                                                                                      \
                    const size_t base = (batch * plan->outer + outer) * (size_t)plan->axis_size * plan->inner;         \
                    for (size_t c = 0; c < plan->coords; ++c)                                                          \
                    {                                                                                                  \
                        const size_t offset = base + (size_t)indices[batch * plan->coords + c] * plan->inner;          \
                        ROW(output, input + offset, (uint32_t)plan->inner);                                            \
                        output += plan->inner;                                                                         \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        __STATIC_FORCEINLINE void arm_nn_gather_nd_copy_##SUFFIX(                                                      \
            const TYPE *input, const int32_t *indices, TYPE *output, const arm_nn_gather_nd_plan *plan)                \
        {                                                                                                              \
            for (size_t batch = 0; batch < plan->batches; ++batch)                                                     \
            {                                                                                                          \
                for (size_t slice = 0; slice < plan->slices; ++slice)                                                  \
                {                                                                                                      \
                    size_t offset = batch * plan->batch_stride;                                                        \
                    for (int32_t d = 0; d < plan->width; ++d)                                                          \
                    {                                                                                                  \
                        offset += (size_t) * indices++ * plan->strides[plan->batch_dims + d];                          \
                    }                                                                                                  \
                    arm_memcpy_##SUFFIX(output, input + offset, (uint32_t)plan->inner);                                \
                    output += plan->inner;                                                                             \
                }                                                                                                      \
            }                                                                                                          \
        }

    #if ARM_NN_ENABLE_F32
ARM_NN_GATHER_COPY_DEFINE(f32, float32_t, arm_nn_gather_row_f32)
    #endif
    #if ARM_NN_ENABLE_F16
ARM_NN_GATHER_COPY_DEFINE(f16, float16_t, arm_memcpy_f16)
    #endif

#endif /* ARM_NN_FLOAT_API_ENABLED */
#endif /* ARM_NN_GATHER_COMMON_H */
