/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_resize_nearest_neighbor_s16.c
 * Description:  Resize nearest neighbor for s16 tensors
 *
 * $Date:        5 January 2025
 * $Revision:    V.1.0.2
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Reshape
 * @{
 */

/*
 * Basic s16 resize nearest neighbor function.
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_resize_nearest_neighbor_s16(const cmsis_nn_context *ctx,
                                                    const cmsis_nn_resize_params *resize_params,
                                                    const cmsis_nn_dims *input_shape,
                                                    const int16_t *input_data,
                                                    const cmsis_nn_dims *output_size_shape,
                                                    const int32_t *output_size_data,
                                                    const cmsis_nn_dims *output_shape,
                                                    int16_t *output_data)
{
    if (!ctx || !resize_params || !input_shape || !input_data || !output_size_shape || !output_size_data ||
        !output_shape || !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (input_shape->n != output_shape->n || input_shape->c != output_shape->c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int32_t batches = input_shape->n;
    int32_t input_height = input_shape->h;
    int32_t input_width = input_shape->w;
    int32_t depth = input_shape->c;

    int32_t flat_size = output_size_shape->n * output_size_shape->h * output_size_shape->w * output_size_shape->c;

    if (flat_size != 2)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t output_height = output_size_data[0];
    int32_t output_width = output_size_data[1];
    if (output_height <= 0 || output_width <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const float y_scale = (resize_params->align_corners && output_height > 1)
        ? (float)(input_height - 1) / (float)(output_height - 1)
        : (float)input_height / (float)output_height;
    const float x_scale = (resize_params->align_corners && output_width > 1)
        ? (float)(input_width - 1) / (float)(output_width - 1)
        : (float)input_width / (float)output_width;
    const float y_offset = resize_params->half_pixel_centers ? 0.5f : 0.0f;
    const float x_offset = resize_params->half_pixel_centers ? 0.5f : 0.0f;

    const int col_offset = input_shape->c;
    const int row_offset = input_shape->w * col_offset;
    const int batch_offset = input_shape->h * row_offset;

    const int16_t *input_ptr = input_data;
    int16_t *output_ptr = output_data;

    const int32_t required_buffer_size = (output_height + output_width) * (int32_t)sizeof(int32_t);
    if (ctx->buf == NULL || ctx->size < required_buffer_size)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // scratch buffer must be 4-byte aligned by caller
    int32_t *x_map = (int32_t *)ctx->buf;
    int32_t *y_map = x_map + output_width;

    for (int y = 0; y < output_height; ++y)
    {
        y_map[y] = GetNearestNeighbor(
            y, input_height, y_scale, y_offset, resize_params->align_corners, resize_params->half_pixel_centers);
    }
    for (int x = 0; x < output_width; ++x)
    {
        x_map[x] = GetNearestNeighbor(
            x, input_width, x_scale, x_offset, resize_params->align_corners, resize_params->half_pixel_centers);
    }
    for (int b = 0; b < batches; ++b)
    {
        for (int y = 0; y < output_height; ++y)
        {
            int32_t in_y = y_map[y];
            const int16_t *y_input_ptr = input_ptr + in_y * row_offset;
            for (int x = 0; x < output_width; ++x)
            {
                int32_t in_x = x_map[x];
                const int16_t *x_input_ptr = y_input_ptr + in_x * col_offset;
                arm_memcpy_s16(output_ptr, x_input_ptr, depth);
                output_ptr += depth;
            }
        }
        input_ptr += batch_offset;
    }
    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of Reshape group
 */
