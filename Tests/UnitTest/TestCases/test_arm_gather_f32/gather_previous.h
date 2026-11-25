/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The float GATHER and GATHER_ND bodies as they were at 84bdc0c3
 * (Include/Internal/arm_nn_gather_common.h and the four entries of Source/GatherFunctions/),
 * every helper renamed from arm_nn_gather_ to gp_gather_ and every entry from arm_gather_ to a
 * static gp_gather_, copied so that no later change to the library reaches them. They are
 * compiled only by this suite, as the reference the differential cases hold arm_gather_f32,
 * arm_gather_f16, arm_gather_nd_f32 and arm_gather_nd_f16 to: status, every output byte, and
 * the plans of the two prepare steps. The row copies call the library's arm_memcpy_f32 and
 * arm_memcpy_f16, which this change leaves alone. */

#include "arm_nnfunctions.h"
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
} gp_gather_plan;

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
} gp_gather_nd_plan;

__STATIC_FORCEINLINE int32_t gp_gather_shape(const cmsis_nn_dims *dims, int32_t rank, int32_t *shape)
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

__STATIC_FORCEINLINE int32_t gp_gather_count(const int32_t *shape, int32_t rank, size_t element_size, size_t *count)
{
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
__STATIC_FORCEINLINE size_t gp_gather_product(const int32_t *shape, int32_t begin, int32_t end)
{
    size_t count = 1;
    for (int32_t d = begin; d < end; ++d)
    {
        count *= (size_t)shape[d];
    }
    return count;
}

__STATIC_FORCEINLINE arm_cmsis_nn_status gp_gather_prepare(const void *input_data,
                                                           const cmsis_nn_dims *input_dims,
                                                           const int32_t *indices_data,
                                                           const cmsis_nn_dims *indices_dims,
                                                           const cmsis_nn_gather_params *params,
                                                           void *output_data,
                                                           const cmsis_nn_dims *output_dims,
                                                           size_t element_size,
                                                           gp_gather_plan *plan)
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
    if (!gp_gather_shape(input_dims, rank, input_shape) || !gp_gather_shape(indices_dims, coords_rank, indices_shape) ||
        !gp_gather_shape(output_dims, output_rank, output_shape))
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
    if (!gp_gather_count(input_shape, rank, element_size, &input_count) ||
        !gp_gather_count(indices_shape, coords_rank, sizeof(int32_t), &indices_count) ||
        !gp_gather_count(output_shape, output_rank, element_size, &plan->output_count))
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
    plan->batches = gp_gather_product(input_shape, 0, batch);
    plan->outer = gp_gather_product(input_shape, batch, axis);
    plan->inner = gp_gather_product(input_shape, axis + 1, rank);
    plan->coords = gp_gather_product(indices_shape, batch, coords_rank);
    plan->axis_size = input_shape[axis];
    return ARM_CMSIS_NN_SUCCESS;
}

__STATIC_FORCEINLINE arm_cmsis_nn_status gp_gather_nd_prepare(const void *params_data,
                                                              const cmsis_nn_dims *params_dims,
                                                              const int32_t *indices_data,
                                                              const cmsis_nn_dims *indices_dims,
                                                              const cmsis_nn_gather_nd_params *params,
                                                              void *output_data,
                                                              const cmsis_nn_dims *output_dims,
                                                              size_t element_size,
                                                              gp_gather_nd_plan *plan)
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
    if (!gp_gather_shape(params_dims, rank, input_shape) || !gp_gather_shape(indices_dims, indices_rank, indices_shape))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t width = indices_shape[indices_rank - 1];
    if (width < 1 || width > rank - batch)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t output_rank = indices_rank - 1 + rank - batch - width;
    if (output_rank > 4 || !gp_gather_shape(output_dims, output_rank, output_shape))
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
    if (!gp_gather_count(input_shape, rank, element_size, &input_count) ||
        !gp_gather_count(indices_shape, indices_rank, sizeof(int32_t), &indices_count) ||
        !gp_gather_count(output_shape, output_rank, element_size, &plan->output_count))
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
    plan->batches = gp_gather_product(input_shape, 0, batch);
    plan->slices = gp_gather_product(indices_shape, batch, indices_rank - 1);
    plan->inner = gp_gather_product(input_shape, batch + width, rank);
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

    #define GP_GATHER_COPY_DEFINE(SUFFIX, TYPE)                                                                        \
        __STATIC_FORCEINLINE void gp_gather_copy_##SUFFIX(                                                             \
            const TYPE *input, const int32_t *indices, TYPE *output, const gp_gather_plan *plan)                       \
        {                                                                                                              \
            for (size_t batch = 0; batch < plan->batches; ++batch)                                                     \
            {                                                                                                          \
                for (size_t outer = 0; outer < plan->outer; ++outer)                                                   \
                {                                                                                                      \
                    const size_t base = (batch * plan->outer + outer) * (size_t)plan->axis_size * plan->inner;         \
                    for (size_t c = 0; c < plan->coords; ++c)                                                          \
                    {                                                                                                  \
                        const size_t offset = base + (size_t)indices[batch * plan->coords + c] * plan->inner;          \
                        arm_memcpy_##SUFFIX(output, input + offset, (uint32_t)plan->inner);                            \
                        output += plan->inner;                                                                         \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        __STATIC_FORCEINLINE void gp_gather_nd_copy_##SUFFIX(                                                          \
            const TYPE *input, const int32_t *indices, TYPE *output, const gp_gather_nd_plan *plan)                    \
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
GP_GATHER_COPY_DEFINE(f32, float32_t)
    #endif
    #if ARM_NN_ENABLE_F16
GP_GATHER_COPY_DEFINE(f16, float16_t)
    #endif

    #if ARM_NN_ENABLE_F32
static arm_cmsis_nn_status gp_gather_f32(const float32_t *input_data,
                                         const cmsis_nn_dims *input_dims,
                                         const int32_t *indices_data,
                                         const cmsis_nn_dims *indices_dims,
                                         const cmsis_nn_gather_params *params,
                                         float32_t *output_data,
                                         const cmsis_nn_dims *output_dims)
{
    gp_gather_plan plan;
    const arm_cmsis_nn_status status = gp_gather_prepare(
        input_data, input_dims, indices_data, indices_dims, params, output_data, output_dims, sizeof(float32_t), &plan);
    if (status != ARM_CMSIS_NN_SUCCESS || plan.output_count == 0)
    {
        return status;
    }
    gp_gather_copy_f32(input_data, indices_data, output_data, &plan);
    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status gp_gather_nd_f32(const float32_t *params_data,
                                            const cmsis_nn_dims *params_dims,
                                            const int32_t *indices_data,
                                            const cmsis_nn_dims *indices_dims,
                                            const cmsis_nn_gather_nd_params *params,
                                            float32_t *output_data,
                                            const cmsis_nn_dims *output_dims)
{
    gp_gather_nd_plan plan;
    const arm_cmsis_nn_status status = gp_gather_nd_prepare(params_data,
                                                            params_dims,
                                                            indices_data,
                                                            indices_dims,
                                                            params,
                                                            output_data,
                                                            output_dims,
                                                            sizeof(float32_t),
                                                            &plan);
    if (status != ARM_CMSIS_NN_SUCCESS || plan.output_count == 0)
    {
        return status;
    }
    gp_gather_nd_copy_f32(params_data, indices_data, output_data, &plan);
    return ARM_CMSIS_NN_SUCCESS;
}

    #endif
    #if ARM_NN_ENABLE_F16
static arm_cmsis_nn_status gp_gather_f16(const float16_t *input_data,
                                         const cmsis_nn_dims *input_dims,
                                         const int32_t *indices_data,
                                         const cmsis_nn_dims *indices_dims,
                                         const cmsis_nn_gather_params *params,
                                         float16_t *output_data,
                                         const cmsis_nn_dims *output_dims)
{
    gp_gather_plan plan;
    const arm_cmsis_nn_status status = gp_gather_prepare(
        input_data, input_dims, indices_data, indices_dims, params, output_data, output_dims, sizeof(float16_t), &plan);
    if (status != ARM_CMSIS_NN_SUCCESS || plan.output_count == 0)
    {
        return status;
    }
    gp_gather_copy_f16(input_data, indices_data, output_data, &plan);
    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status gp_gather_nd_f16(const float16_t *params_data,
                                            const cmsis_nn_dims *params_dims,
                                            const int32_t *indices_data,
                                            const cmsis_nn_dims *indices_dims,
                                            const cmsis_nn_gather_nd_params *params,
                                            float16_t *output_data,
                                            const cmsis_nn_dims *output_dims)
{
    gp_gather_nd_plan plan;
    const arm_cmsis_nn_status status = gp_gather_nd_prepare(params_data,
                                                            params_dims,
                                                            indices_data,
                                                            indices_dims,
                                                            params,
                                                            output_data,
                                                            output_dims,
                                                            sizeof(float16_t),
                                                            &plan);
    if (status != ARM_CMSIS_NN_SUCCESS || plan.output_count == 0)
    {
        return status;
    }
    gp_gather_nd_copy_f16(params_data, indices_data, output_data, &plan);
    return ARM_CMSIS_NN_SUCCESS;
}

    #endif

#endif /* ARM_NN_FLOAT_API_ENABLED */
