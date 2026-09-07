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
 * Title:        arm_resize_nearest_neighbor_common.h
 * Description:  Shared float nearest-neighbor resize walker
 *
 * $Date:        6 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_RESIZE_NEAREST_NEIGHBOR_COMMON_H
#define ARM_RESIZE_NEAREST_NEIGHBOR_COMMON_H

#include "arm_nn_types.h"
#include "arm_nnsupportfunctions.h"

#include <stdint.h>

/*
 * Nearest-neighbor resize is pure data movement, so the f32 and f16 kernels
 * share one index walker; only the scalar type and the copy helper differ.
 * The index maps follow the TFLite RESIZE_NEAREST_NEIGHBOR reference
 * (GetNearestNeighbor in arm_nnsupportfunctions.h, same float32 scale math),
 * verified against TFLite 2.20 in the PR that added the float kernels.
 */

/* Scratch bytes for the x/y index maps; -1 family (see arm_nn_size_mul): a
 * negative return must never size a buffer, the kernels reject { NULL, 0 }. */
static inline int32_t arm_nn_resize_nearest_neighbor_scratch_bytes(const cmsis_nn_dims *output_dims)
{
    if (output_dims == NULL || output_dims->h < 1 || output_dims->w < 1)
    {
        return -1;
    }
    const int64_t elements = arm_nn_size_add((int64_t)output_dims->h, (int64_t)output_dims->w);
    return (int32_t)arm_nn_size_mul(elements, (int64_t)sizeof(int32_t));
}

/* Validates the argument set and fills the x/y index maps in ctx->buf. */
static inline arm_cmsis_nn_status arm_nn_resize_nearest_neighbor_prepare(const cmsis_nn_context *ctx,
                                                                         const cmsis_nn_resize_params *resize_params,
                                                                         const cmsis_nn_dims *input_shape,
                                                                         const cmsis_nn_dims *output_size_shape,
                                                                         const int32_t *output_size_data,
                                                                         const cmsis_nn_dims *output_shape,
                                                                         int32_t **x_map_out,
                                                                         int32_t **y_map_out)
{
    if (!ctx || !resize_params || !input_shape || !output_size_shape || !output_size_data || !output_shape)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (input_shape->n <= 0 || input_shape->h <= 0 || input_shape->w <= 0 || input_shape->c <= 0 ||
        input_shape->n != output_shape->n || input_shape->c != output_shape->c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t output_size_elements =
        output_size_shape->n * output_size_shape->h * output_size_shape->w * output_size_shape->c;
    if (output_size_elements != 2)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t output_height = output_size_data[0];
    const int32_t output_width = output_size_data[1];
    if (output_height <= 0 || output_width <= 0 || output_shape->h != output_height || output_shape->w != output_width)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t required_buffer_size = arm_nn_resize_nearest_neighbor_scratch_bytes(output_shape);
    if (required_buffer_size < 0 || ctx->buf == NULL || ctx->size < required_buffer_size ||
        (((uintptr_t)ctx->buf) & (sizeof(int32_t) - 1U)) != 0U)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const float32_t y_scale = (resize_params->align_corners && output_height > 1)
        ? (float32_t)(input_shape->h - 1) / (float32_t)(output_height - 1)
        : (float32_t)input_shape->h / (float32_t)output_height;
    const float32_t x_scale = (resize_params->align_corners && output_width > 1)
        ? (float32_t)(input_shape->w - 1) / (float32_t)(output_width - 1)
        : (float32_t)input_shape->w / (float32_t)output_width;
    const float32_t offset = resize_params->half_pixel_centers ? 0.5f : 0.0f;

    int32_t *x_map = (int32_t *)ctx->buf;
    int32_t *y_map = x_map + output_width;
    for (int32_t y = 0; y < output_height; ++y)
    {
        y_map[y] = GetNearestNeighbor(
            y, input_shape->h, y_scale, offset, resize_params->align_corners, resize_params->half_pixel_centers);
    }
    for (int32_t x = 0; x < output_width; ++x)
    {
        x_map[x] = GetNearestNeighbor(
            x, input_shape->w, x_scale, offset, resize_params->align_corners, resize_params->half_pixel_centers);
    }

    *x_map_out = x_map;
    *y_map_out = y_map;
    return ARM_CMSIS_NN_SUCCESS;
}

/* MEMCPY_FUNC takes an element count (arm_memcpy_f32 / arm_memcpy_f16). */
#define ARM_RESIZE_NEAREST_NEIGHBOR_DEFINE(FUNC_NAME, SCALAR_T, MEMCPY_FUNC)                                           \
    arm_cmsis_nn_status FUNC_NAME(const cmsis_nn_context *ctx,                                                         \
                                  const cmsis_nn_resize_params *resize_params,                                         \
                                  const cmsis_nn_dims *input_shape,                                                    \
                                  const SCALAR_T *input_data,                                                          \
                                  const cmsis_nn_dims *output_size_shape,                                              \
                                  const int32_t *output_size_data,                                                     \
                                  const cmsis_nn_dims *output_shape,                                                   \
                                  SCALAR_T *output_data)                                                               \
    {                                                                                                                  \
        if (!input_data || !output_data)                                                                               \
        {                                                                                                              \
            return ARM_CMSIS_NN_ARG_ERROR;                                                                             \
        }                                                                                                              \
                                                                                                                       \
        int32_t *x_map = NULL;                                                                                         \
        int32_t *y_map = NULL;                                                                                         \
        const arm_cmsis_nn_status status = arm_nn_resize_nearest_neighbor_prepare(                                     \
            ctx, resize_params, input_shape, output_size_shape, output_size_data, output_shape, &x_map, &y_map);       \
        if (status != ARM_CMSIS_NN_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
                                                                                                                       \
        const int32_t depth = input_shape->c;                                                                          \
        const int32_t row_stride = input_shape->w * depth;                                                             \
        const int32_t batch_stride = input_shape->h * row_stride;                                                      \
        const int32_t output_height = output_shape->h;                                                                 \
        const int32_t output_width = output_shape->w;                                                                  \
        for (int32_t batch = 0; batch < input_shape->n; ++batch)                                                       \
        {                                                                                                              \
            const SCALAR_T *batch_input = input_data + batch * batch_stride;                                           \
            for (int32_t y = 0; y < output_height; ++y)                                                                \
            {                                                                                                          \
                const SCALAR_T *row_input = batch_input + y_map[y] * row_stride;                                       \
                for (int32_t x = 0; x < output_width; ++x)                                                             \
                {                                                                                                      \
                    MEMCPY_FUNC(output_data, row_input + x_map[x] * depth, (uint32_t)depth);                           \
                    output_data += depth;                                                                              \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        return ARM_CMSIS_NN_SUCCESS;                                                                                   \
    }

#endif /* ARM_RESIZE_NEAREST_NEIGHBOR_COMMON_H */
