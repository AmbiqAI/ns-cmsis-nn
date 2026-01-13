/*
 * SPDX-FileCopyrightText: 2025 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "unity.h"

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <stdlib.h>

#include "../TestData/resize_nn_basic_s16/test_data.h"
#include "../TestData/resize_nn_half_pixel_s16/test_data.h"
#include "../TestData/resize_nn_align_corners_s16/test_data.h"
#include "../TestData/resize_nn_downsample_s16/test_data.h"
#include "../TestData/resize_nn_batch_channels_s16/test_data.h"
#include "../Utils/validate.h"

void resize_nn_basic_s16_arm_resize_nearest_neighbor_s16(void)
{
    const cmsis_nn_resize_params resize_params = {
        .align_corners = RESIZE_NN_BASIC_S16_ALIGN_CORNERS,
        .half_pixel_centers = RESIZE_NN_BASIC_S16_HALF_PIXEL_CENTERS,
    };
    const cmsis_nn_dims input_shape = {RESIZE_NN_BASIC_S16_INPUT_N,
                                       RESIZE_NN_BASIC_S16_INPUT_H,
                                       RESIZE_NN_BASIC_S16_INPUT_W,
                                       RESIZE_NN_BASIC_S16_INPUT_C};
    const cmsis_nn_dims output_size_shape = {RESIZE_NN_BASIC_S16_OUTPUT_SIZE_N,
                                             RESIZE_NN_BASIC_S16_OUTPUT_SIZE_H,
                                             RESIZE_NN_BASIC_S16_OUTPUT_SIZE_W,
                                             RESIZE_NN_BASIC_S16_OUTPUT_SIZE_C};
    const int32_t output_size_data[] = RESIZE_NN_BASIC_S16_OUTPUT_SIZE_DATA;
    const cmsis_nn_dims output_shape = {RESIZE_NN_BASIC_S16_OUTPUT_N,
                                        RESIZE_NN_BASIC_S16_OUTPUT_H,
                                        RESIZE_NN_BASIC_S16_OUTPUT_W,
                                        RESIZE_NN_BASIC_S16_OUTPUT_C};
    int16_t output_data[RESIZE_NN_BASIC_S16_OUTPUT_SIZE] = {0};
    TEST_ASSERT_TRUE(output_size_data != NULL);
    TEST_ASSERT_TRUE(output_size_data[0] >= 0);
    TEST_ASSERT_TRUE(output_size_data[1] >= 0);
    int32_t scratch_elements = output_size_data[0] * output_size_data[1];
    int32_t *scratch_buffer = NULL;
    if (scratch_elements > 0)
    {
        scratch_buffer = (int32_t *)malloc((size_t)scratch_elements * sizeof(int32_t));
    }
    TEST_ASSERT_TRUE((scratch_elements == 0) || (scratch_buffer != NULL));
    cmsis_nn_context ctx = { .buf = scratch_buffer, .size = scratch_elements * (int32_t)sizeof(int32_t) };

    const arm_cmsis_nn_status result = arm_resize_nearest_neighbor_s16(&ctx,
                                                                       &resize_params,
                                                                       &input_shape,
                                                                       resize_nn_basic_s16_input_tensor,
                                                                       &output_size_shape,
                                                                       output_size_data,
                                                                       &output_shape,
                                                                       output_data);

    if (result == ARM_CMSIS_NN_SUCCESS)
    {
        TEST_ASSERT_TRUE(validate_s16(output_data, resize_nn_basic_s16_output, RESIZE_NN_BASIC_S16_OUTPUT_SIZE));
    }
    else
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, result);
    }
    free(scratch_buffer);
}

void resize_nn_half_pixel_s16_arm_resize_nearest_neighbor_s16(void)
{
    const cmsis_nn_resize_params resize_params = {
        .align_corners = RESIZE_NN_HALF_PIXEL_S16_ALIGN_CORNERS,
        .half_pixel_centers = RESIZE_NN_HALF_PIXEL_S16_HALF_PIXEL_CENTERS,
    };
    const cmsis_nn_dims input_shape = {RESIZE_NN_HALF_PIXEL_S16_INPUT_N,
                                       RESIZE_NN_HALF_PIXEL_S16_INPUT_H,
                                       RESIZE_NN_HALF_PIXEL_S16_INPUT_W,
                                       RESIZE_NN_HALF_PIXEL_S16_INPUT_C};
    const cmsis_nn_dims output_size_shape = {RESIZE_NN_HALF_PIXEL_S16_OUTPUT_SIZE_N,
                                             RESIZE_NN_HALF_PIXEL_S16_OUTPUT_SIZE_H,
                                             RESIZE_NN_HALF_PIXEL_S16_OUTPUT_SIZE_W,
                                             RESIZE_NN_HALF_PIXEL_S16_OUTPUT_SIZE_C};
    const int32_t output_size_data[] = RESIZE_NN_HALF_PIXEL_S16_OUTPUT_SIZE_DATA;
    const cmsis_nn_dims output_shape = {RESIZE_NN_HALF_PIXEL_S16_OUTPUT_N,
                                        RESIZE_NN_HALF_PIXEL_S16_OUTPUT_H,
                                        RESIZE_NN_HALF_PIXEL_S16_OUTPUT_W,
                                        RESIZE_NN_HALF_PIXEL_S16_OUTPUT_C};
    int16_t output_data[RESIZE_NN_HALF_PIXEL_S16_OUTPUT_SIZE] = {0};
    TEST_ASSERT_TRUE(output_size_data != NULL);
    TEST_ASSERT_TRUE(output_size_data[0] >= 0);
    TEST_ASSERT_TRUE(output_size_data[1] >= 0);
    int32_t scratch_elements = output_size_data[0] * output_size_data[1];
    int32_t *scratch_buffer = NULL;
    if (scratch_elements > 0)
    {
        scratch_buffer = (int32_t *)malloc((size_t)scratch_elements * sizeof(int32_t));
    }
    TEST_ASSERT_TRUE((scratch_elements == 0) || (scratch_buffer != NULL));
    cmsis_nn_context ctx = { .buf = scratch_buffer, .size = scratch_elements * (int32_t)sizeof(int32_t) };

    const arm_cmsis_nn_status result = arm_resize_nearest_neighbor_s16(&ctx,
                                                                       &resize_params,
                                                                       &input_shape,
                                                                       resize_nn_half_pixel_s16_input_tensor,
                                                                       &output_size_shape,
                                                                       output_size_data,
                                                                       &output_shape,
                                                                       output_data);

    if (result == ARM_CMSIS_NN_SUCCESS)
    {
        TEST_ASSERT_TRUE(validate_s16(output_data, resize_nn_half_pixel_s16_output, RESIZE_NN_HALF_PIXEL_S16_OUTPUT_SIZE));
    }
    else
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, result);
    }
    free(scratch_buffer);
}

void resize_nn_align_corners_s16_arm_resize_nearest_neighbor_s16(void)
{
    const cmsis_nn_resize_params resize_params = {
        .align_corners = RESIZE_NN_ALIGN_CORNERS_S16_ALIGN_CORNERS,
        .half_pixel_centers = RESIZE_NN_ALIGN_CORNERS_S16_HALF_PIXEL_CENTERS,
    };
    const cmsis_nn_dims input_shape = {RESIZE_NN_ALIGN_CORNERS_S16_INPUT_N,
                                       RESIZE_NN_ALIGN_CORNERS_S16_INPUT_H,
                                       RESIZE_NN_ALIGN_CORNERS_S16_INPUT_W,
                                       RESIZE_NN_ALIGN_CORNERS_S16_INPUT_C};
    const cmsis_nn_dims output_size_shape = {RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_SIZE_N,
                                             RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_SIZE_H,
                                             RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_SIZE_W,
                                             RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_SIZE_C};
    const int32_t output_size_data[] = RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_SIZE_DATA;
    const cmsis_nn_dims output_shape = {RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_N,
                                        RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_H,
                                        RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_W,
                                        RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_C};
    int16_t output_data[RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_SIZE] = {0};
    TEST_ASSERT_TRUE(output_size_data != NULL);
    TEST_ASSERT_TRUE(output_size_data[0] >= 0);
    TEST_ASSERT_TRUE(output_size_data[1] >= 0);
    int32_t scratch_elements = output_size_data[0] * output_size_data[1];
    int32_t *scratch_buffer = NULL;
    if (scratch_elements > 0)
    {
        scratch_buffer = (int32_t *)malloc((size_t)scratch_elements * sizeof(int32_t));
    }
    TEST_ASSERT_TRUE((scratch_elements == 0) || (scratch_buffer != NULL));
    cmsis_nn_context ctx = { .buf = scratch_buffer, .size = scratch_elements * (int32_t)sizeof(int32_t) };

    const arm_cmsis_nn_status result = arm_resize_nearest_neighbor_s16(&ctx,
                                                                       &resize_params,
                                                                       &input_shape,
                                                                       resize_nn_align_corners_s16_input_tensor,
                                                                       &output_size_shape,
                                                                       output_size_data,
                                                                       &output_shape,
                                                                       output_data);

    if (result == ARM_CMSIS_NN_SUCCESS)
    {
        TEST_ASSERT_TRUE(
            validate_s16(output_data, resize_nn_align_corners_s16_output, RESIZE_NN_ALIGN_CORNERS_S16_OUTPUT_SIZE));
    }
    else
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, result);
    }
    free(scratch_buffer);
}

void resize_nn_downsample_s16_arm_resize_nearest_neighbor_s16(void)
{
    const cmsis_nn_resize_params resize_params = {
        .align_corners = RESIZE_NN_DOWNSAMPLE_S16_ALIGN_CORNERS,
        .half_pixel_centers = RESIZE_NN_DOWNSAMPLE_S16_HALF_PIXEL_CENTERS,
    };
    const cmsis_nn_dims input_shape = {RESIZE_NN_DOWNSAMPLE_S16_INPUT_N,
                                       RESIZE_NN_DOWNSAMPLE_S16_INPUT_H,
                                       RESIZE_NN_DOWNSAMPLE_S16_INPUT_W,
                                       RESIZE_NN_DOWNSAMPLE_S16_INPUT_C};
    const cmsis_nn_dims output_size_shape = {RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_SIZE_N,
                                             RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_SIZE_H,
                                             RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_SIZE_W,
                                             RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_SIZE_C};
    const int32_t output_size_data[] = RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_SIZE_DATA;
    const cmsis_nn_dims output_shape = {RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_N,
                                        RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_H,
                                        RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_W,
                                        RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_C};
    int16_t output_data[RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_SIZE] = {0};
    TEST_ASSERT_TRUE(output_size_data != NULL);
    TEST_ASSERT_TRUE(output_size_data[0] >= 0);
    TEST_ASSERT_TRUE(output_size_data[1] >= 0);
    int32_t scratch_elements = output_size_data[0] * output_size_data[1];
    int32_t *scratch_buffer = NULL;
    if (scratch_elements > 0)
    {
        scratch_buffer = (int32_t *)malloc((size_t)scratch_elements * sizeof(int32_t));
    }
    TEST_ASSERT_TRUE((scratch_elements == 0) || (scratch_buffer != NULL));
    cmsis_nn_context ctx = { .buf = scratch_buffer, .size = scratch_elements * (int32_t)sizeof(int32_t) };

    const arm_cmsis_nn_status result = arm_resize_nearest_neighbor_s16(&ctx,
                                                                       &resize_params,
                                                                       &input_shape,
                                                                       resize_nn_downsample_s16_input_tensor,
                                                                       &output_size_shape,
                                                                       output_size_data,
                                                                       &output_shape,
                                                                       output_data);

    if (result == ARM_CMSIS_NN_SUCCESS)
    {
        TEST_ASSERT_TRUE(validate_s16(output_data, resize_nn_downsample_s16_output, RESIZE_NN_DOWNSAMPLE_S16_OUTPUT_SIZE));
    }
    else
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, result);
    }
    free(scratch_buffer);
}

void resize_nn_batch_channels_s16_arm_resize_nearest_neighbor_s16(void)
{
    const cmsis_nn_resize_params resize_params = {
        .align_corners = RESIZE_NN_BATCH_CHANNELS_S16_ALIGN_CORNERS,
        .half_pixel_centers = RESIZE_NN_BATCH_CHANNELS_S16_HALF_PIXEL_CENTERS,
    };
    const cmsis_nn_dims input_shape = {RESIZE_NN_BATCH_CHANNELS_S16_INPUT_N,
                                       RESIZE_NN_BATCH_CHANNELS_S16_INPUT_H,
                                       RESIZE_NN_BATCH_CHANNELS_S16_INPUT_W,
                                       RESIZE_NN_BATCH_CHANNELS_S16_INPUT_C};
    const cmsis_nn_dims output_size_shape = {RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_SIZE_N,
                                             RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_SIZE_H,
                                             RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_SIZE_W,
                                             RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_SIZE_C};
    const int32_t output_size_data[] = RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_SIZE_DATA;
    const cmsis_nn_dims output_shape = {RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_N,
                                        RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_H,
                                        RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_W,
                                        RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_C};
    int16_t output_data[RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_SIZE] = {0};
    TEST_ASSERT_TRUE(output_size_data != NULL);
    TEST_ASSERT_TRUE(output_size_data[0] >= 0);
    TEST_ASSERT_TRUE(output_size_data[1] >= 0);
    int32_t scratch_elements = output_size_data[0] * output_size_data[1];
    int32_t *scratch_buffer = NULL;
    if (scratch_elements > 0)
    {
        scratch_buffer = (int32_t *)malloc((size_t)scratch_elements * sizeof(int32_t));
    }
    TEST_ASSERT_TRUE((scratch_elements == 0) || (scratch_buffer != NULL));
    cmsis_nn_context ctx = { .buf = scratch_buffer, .size = scratch_elements * (int32_t)sizeof(int32_t) };

    const arm_cmsis_nn_status result = arm_resize_nearest_neighbor_s16(&ctx,
                                                                       &resize_params,
                                                                       &input_shape,
                                                                       resize_nn_batch_channels_s16_input_tensor,
                                                                       &output_size_shape,
                                                                       output_size_data,
                                                                       &output_shape,
                                                                       output_data);

    if (result == ARM_CMSIS_NN_SUCCESS)
    {
        TEST_ASSERT_TRUE(
            validate_s16(output_data, resize_nn_batch_channels_s16_output, RESIZE_NN_BATCH_CHANNELS_S16_OUTPUT_SIZE));
    }
    else
    {
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_NO_IMPL_ERROR, result);
    }
    free(scratch_buffer);
}
