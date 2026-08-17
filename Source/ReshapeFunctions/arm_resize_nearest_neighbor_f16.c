/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_resize_nearest_neighbor_f16.c
 * Description:  Resize nearest neighbor for float16 tensors
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup Reshape
 * @{
 */

/* Refer header file for details. */
arm_cmsis_nn_status arm_resize_nearest_neighbor_f16(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_resize_params *resize_params,
                                                    const cmsis_nn_dims *input_shape,
                                                    const float16_t *input_data,
                                                    const cmsis_nn_dims *output_size_shape,
                                                    const int32_t *output_size_data,
                                                    const cmsis_nn_dims *output_shape,
                                                    float16_t *output_data)
{
    if (!ctx || !resize_params || !input_shape || !input_data || !output_size_shape || !output_size_data ||
        !output_shape || !output_data)
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

    const int32_t required_buffer_size = (output_height + output_width) * (int32_t)sizeof(int32_t);
    if (ctx->buf == NULL || ctx->size < required_buffer_size)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const float y_scale = (resize_params->align_corners && output_height > 1)
        ? (float)(input_shape->h - 1) / (float)(output_height - 1)
        : (float)input_shape->h / (float)output_height;
    const float x_scale = (resize_params->align_corners && output_width > 1)
        ? (float)(input_shape->w - 1) / (float)(output_width - 1)
        : (float)input_shape->w / (float)output_width;
    const float offset = resize_params->half_pixel_centers ? 0.5f : 0.0f;

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

    const int32_t row_stride = input_shape->w * input_shape->c;
    const int32_t batch_stride = input_shape->h * row_stride;
    for (int32_t batch = 0; batch < input_shape->n; ++batch)
    {
        const float16_t *batch_input = input_data + batch * batch_stride;
        for (int32_t y = 0; y < output_height; ++y)
        {
            const float16_t *row_input = batch_input + y_map[y] * row_stride;
            for (int32_t x = 0; x < output_width; ++x)
            {
                arm_memcpy_f16(output_data, row_input + x_map[x] * input_shape->c, (uint32_t)input_shape->c);
                output_data += input_shape->c;
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Reshape group
 */

#endif /* ARM_NN_ENABLE_F16 */
