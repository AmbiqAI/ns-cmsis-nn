/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "stdio.h"
#include <arm_nnfunctions.h>
#include <stdlib.h>
#include <unity.h>
#include "cmsis_perf_profile.h"

#include "../../TestData/perf_maximum_scalar_1st_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_maximum_scalar_1st_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_maximum_scalar_1st_int8_tiny/test_data.h"
#include "../../TestData/perf_maximum_scalar_1st_int8_avg/test_data.h"
#include "../../TestData/perf_maximum_scalar_1st_int8_mobnet/test_data.h"
#include "../../TestData/perf_maximum_scalar_1st_int8_large/test_data.h"
#include "../../TestData/perf_maximum_scalar_2nd_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_maximum_scalar_2nd_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_maximum_scalar_2nd_int8_tiny/test_data.h"
#include "../../TestData/perf_maximum_scalar_2nd_int8_avg/test_data.h"
#include "../../TestData/perf_maximum_scalar_2nd_int8_mobnet/test_data.h"
#include "../../TestData/perf_maximum_scalar_2nd_int8_large/test_data.h"
#include "../../TestData/perf_maximum_no_broadcast_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_maximum_no_broadcast_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_maximum_no_broadcast_int8_tiny/test_data.h"
#include "../../TestData/perf_maximum_no_broadcast_int8_avg/test_data.h"
#include "../../TestData/perf_maximum_no_broadcast_int8_mobnet/test_data.h"
#include "../../TestData/perf_maximum_no_broadcast_int8_large/test_data.h"
#include "../../TestData/perf_maximum_broadcast_batch_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_maximum_broadcast_batch_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_maximum_broadcast_batch_int8_tiny/test_data.h"
#include "../../TestData/perf_maximum_broadcast_batch_int8_avg/test_data.h"
#include "../../TestData/perf_maximum_broadcast_batch_int8_mobnet/test_data.h"
#include "../../TestData/perf_maximum_broadcast_batch_int8_large/test_data.h"
#include "../../TestData/perf_maximum_broadcast_height_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_maximum_broadcast_height_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_maximum_broadcast_height_int8_tiny/test_data.h"
#include "../../TestData/perf_maximum_broadcast_height_int8_avg/test_data.h"
#include "../../TestData/perf_maximum_broadcast_height_int8_mobnet/test_data.h"
#include "../../TestData/perf_maximum_broadcast_height_int8_large/test_data.h"
#include "../../TestData/perf_maximum_broadcast_width_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_maximum_broadcast_width_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_maximum_broadcast_width_int8_tiny/test_data.h"
#include "../../TestData/perf_maximum_broadcast_width_int8_avg/test_data.h"
#include "../../TestData/perf_maximum_broadcast_width_int8_mobnet/test_data.h"
#include "../../TestData/perf_maximum_broadcast_width_int8_large/test_data.h"
#include "../../TestData/perf_maximum_broadcast_ch_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_maximum_broadcast_ch_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_maximum_broadcast_ch_int8_tiny/test_data.h"
#include "../../TestData/perf_maximum_broadcast_ch_int8_avg/test_data.h"
#include "../../TestData/perf_maximum_broadcast_ch_int8_mobnet/test_data.h"
#include "../../TestData/perf_maximum_broadcast_ch_int8_large/test_data.h"

void perf_arm_maximum_scalar_1st_s8_tiny_edge1(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_1st_int8_tiny_edge1_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_1st_int8_tiny_edge1_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_BATCH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_WIDTH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_BATCH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_WIDTH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE1_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_1st_s8_tiny_edge2(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_1st_int8_tiny_edge2_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_1st_int8_tiny_edge2_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_BATCH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_WIDTH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_BATCH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_WIDTH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_EDGE2_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_1st_s8_tiny(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_1st_int8_tiny_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_1st_int8_tiny_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_BATCH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_WIDTH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_BATCH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_WIDTH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_TINY_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_1st_s8_avg(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_1st_int8_avg_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_1st_int8_avg_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_BATCH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_WIDTH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_BATCH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_WIDTH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_AVG_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_1st_s8_mobnet(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_1st_int8_mobnet_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_1st_int8_mobnet_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_BATCH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_WIDTH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_BATCH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_WIDTH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_MOBNET_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_1st_s8_large(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_1st_int8_large_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_1st_int8_large_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_BATCH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_WIDTH_1, PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_BATCH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_WIDTH_2, PERF_MAXIMUM_SCALAR_1ST_INT8_LARGE_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_1st_s8(void)
{
    perf_arm_maximum_scalar_1st_s8_tiny_edge1();
    printf(",\n");
    perf_arm_maximum_scalar_1st_s8_tiny_edge2();
    printf(",\n");
    perf_arm_maximum_scalar_1st_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_maximum_scalar_1st_s8_avg();
    // printf(",\n");
    // perf_arm_maximum_scalar_1st_s8_mobnet();
    // printf(",\n");
    // perf_arm_maximum_scalar_1st_s8_large();
    // printf(",\n");
}

void perf_arm_maximum_scalar_2nd_s8_tiny_edge1(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_2nd_int8_tiny_edge1_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_2nd_int8_tiny_edge1_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_BATCH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_WIDTH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_BATCH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_WIDTH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE1_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_2nd_s8_tiny_edge2(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_2nd_int8_tiny_edge2_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_2nd_int8_tiny_edge2_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_BATCH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_WIDTH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_BATCH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_WIDTH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_EDGE2_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_2nd_s8_tiny(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_2nd_int8_tiny_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_2nd_int8_tiny_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_BATCH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_WIDTH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_BATCH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_WIDTH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_TINY_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_2nd_s8_avg(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_2nd_int8_avg_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_2nd_int8_avg_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_BATCH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_WIDTH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_BATCH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_WIDTH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_AVG_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_2nd_s8_mobnet(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_2nd_int8_mobnet_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_2nd_int8_mobnet_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_BATCH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_WIDTH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_BATCH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_WIDTH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_MOBNET_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_2nd_s8_large(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_scalar_2nd_int8_large_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_scalar_2nd_int8_large_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_BATCH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_HEIGHT_1,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_WIDTH_1, PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_BATCH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_HEIGHT_2,
                                              PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_WIDTH_2, PERF_MAXIMUM_SCALAR_2ND_INT8_LARGE_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_scalar_2nd_s8(void)
{
    perf_arm_maximum_scalar_2nd_s8_tiny_edge1();
    printf(",\n");
    perf_arm_maximum_scalar_2nd_s8_tiny_edge2();
    printf(",\n");
    perf_arm_maximum_scalar_2nd_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_maximum_scalar_2nd_s8_avg();
    // printf(",\n");
    // perf_arm_maximum_scalar_2nd_s8_mobnet();
    // printf(",\n");
    // perf_arm_maximum_scalar_2nd_s8_large();
    // printf(",\n");
}

void perf_arm_maximum_no_broadcast_s8_tiny_edge1(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_no_broadcast_int8_tiny_edge1_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_no_broadcast_int8_tiny_edge1_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_BATCH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_HEIGHT_1,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_WIDTH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_BATCH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_HEIGHT_2,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_WIDTH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE1_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_no_broadcast_s8_tiny_edge2(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_no_broadcast_int8_tiny_edge2_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_no_broadcast_int8_tiny_edge2_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_BATCH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_HEIGHT_1,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_WIDTH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_BATCH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_HEIGHT_2,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_WIDTH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_EDGE2_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_no_broadcast_s8_tiny(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_no_broadcast_int8_tiny_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_no_broadcast_int8_tiny_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_BATCH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_HEIGHT_1,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_WIDTH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_BATCH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_HEIGHT_2,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_WIDTH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_TINY_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_no_broadcast_s8_avg(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_no_broadcast_int8_avg_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_no_broadcast_int8_avg_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_BATCH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_HEIGHT_1,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_WIDTH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_BATCH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_HEIGHT_2,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_WIDTH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_AVG_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_no_broadcast_s8_mobnet(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_no_broadcast_int8_mobnet_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_no_broadcast_int8_mobnet_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_BATCH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_HEIGHT_1,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_WIDTH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_BATCH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_HEIGHT_2,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_WIDTH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_MOBNET_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_no_broadcast_s8_large(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_no_broadcast_int8_large_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_no_broadcast_int8_large_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_BATCH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_HEIGHT_1,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_WIDTH_1, PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_BATCH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_HEIGHT_2,
                                              PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_WIDTH_2, PERF_MAXIMUM_NO_BROADCAST_INT8_LARGE_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_no_broadcast_s8(void)
{
    perf_arm_maximum_no_broadcast_s8_tiny_edge1();
    printf(",\n");
    perf_arm_maximum_no_broadcast_s8_tiny_edge2();
    printf(",\n");
    perf_arm_maximum_no_broadcast_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_maximum_no_broadcast_s8_avg();
    // printf(",\n");
    // perf_arm_maximum_no_broadcast_s8_mobnet();
    // printf(",\n");
    // perf_arm_maximum_no_broadcast_s8_large();
    // printf(",\n");
}

void perf_arm_maximum_broadcast_batch_s8_tiny_edge1(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_batch_int8_tiny_edge1_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_batch_int8_tiny_edge1_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_BATCH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_WIDTH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_BATCH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_WIDTH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE1_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_batch_s8_tiny_edge2(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_batch_int8_tiny_edge2_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_batch_int8_tiny_edge2_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_BATCH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_WIDTH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_BATCH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_WIDTH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_EDGE2_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_batch_s8_tiny(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_batch_int8_tiny_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_batch_int8_tiny_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_BATCH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_WIDTH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_BATCH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_WIDTH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_TINY_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_batch_s8_avg(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_batch_int8_avg_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_batch_int8_avg_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_BATCH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_WIDTH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_BATCH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_WIDTH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_AVG_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_batch_s8_mobnet(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_batch_int8_mobnet_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_batch_int8_mobnet_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_BATCH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_WIDTH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_BATCH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_WIDTH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_MOBNET_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_batch_s8_large(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_batch_int8_large_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_batch_int8_large_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_BATCH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_WIDTH_1, PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_BATCH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_WIDTH_2, PERF_MAXIMUM_BROADCAST_BATCH_INT8_LARGE_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_batch_s8(void)
{
    perf_arm_maximum_broadcast_batch_s8_tiny_edge1();
    printf(",\n");
    perf_arm_maximum_broadcast_batch_s8_tiny_edge2();
    printf(",\n");
    perf_arm_maximum_broadcast_batch_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_maximum_broadcast_batch_s8_avg();
    // printf(",\n");
    // perf_arm_maximum_broadcast_batch_s8_mobnet();
    // printf(",\n");
    // perf_arm_maximum_broadcast_batch_s8_large();
    // printf(",\n");
}

void perf_arm_maximum_broadcast_height_s8_tiny_edge1(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_height_int8_tiny_edge1_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_height_int8_tiny_edge1_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_BATCH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_WIDTH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_BATCH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_WIDTH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE1_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_height_s8_tiny_edge2(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_height_int8_tiny_edge2_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_height_int8_tiny_edge2_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_BATCH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_WIDTH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_BATCH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_WIDTH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_EDGE2_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_height_s8_tiny(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_height_int8_tiny_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_height_int8_tiny_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_BATCH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_WIDTH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_BATCH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_WIDTH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_TINY_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_height_s8_avg(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_height_int8_avg_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_height_int8_avg_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_BATCH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_WIDTH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_BATCH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_WIDTH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_AVG_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_height_s8_mobnet(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_height_int8_mobnet_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_height_int8_mobnet_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_BATCH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_WIDTH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_BATCH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_WIDTH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_MOBNET_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_height_s8_large(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_height_int8_large_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_height_int8_large_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_BATCH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_WIDTH_1, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_BATCH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_WIDTH_2, PERF_MAXIMUM_BROADCAST_HEIGHT_INT8_LARGE_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_height_s8(void)
{
    perf_arm_maximum_broadcast_height_s8_tiny_edge1();
    printf(",\n");
    perf_arm_maximum_broadcast_height_s8_tiny_edge2();
    printf(",\n");
    perf_arm_maximum_broadcast_height_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_maximum_broadcast_height_s8_avg();
    // printf(",\n");
    // perf_arm_maximum_broadcast_height_s8_mobnet();
    // printf(",\n");
    // perf_arm_maximum_broadcast_height_s8_large();
    // printf(",\n");
}

void perf_arm_maximum_broadcast_width_s8_tiny_edge1(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_width_int8_tiny_edge1_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_width_int8_tiny_edge1_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_BATCH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_WIDTH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_BATCH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_WIDTH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE1_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_width_s8_tiny_edge2(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_width_int8_tiny_edge2_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_width_int8_tiny_edge2_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_BATCH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_WIDTH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_BATCH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_WIDTH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_EDGE2_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_width_s8_tiny(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_width_int8_tiny_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_width_int8_tiny_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_BATCH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_WIDTH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_BATCH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_WIDTH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_TINY_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_width_s8_avg(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_width_int8_avg_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_width_int8_avg_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_BATCH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_WIDTH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_BATCH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_WIDTH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_AVG_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_width_s8_mobnet(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_width_int8_mobnet_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_width_int8_mobnet_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_BATCH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_WIDTH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_BATCH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_WIDTH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_MOBNET_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_width_s8_large(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_width_int8_large_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_width_int8_large_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_BATCH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_WIDTH_1, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_BATCH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_WIDTH_2, PERF_MAXIMUM_BROADCAST_WIDTH_INT8_LARGE_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_width_s8(void)
{
    perf_arm_maximum_broadcast_width_s8_tiny_edge1();
    printf(",\n");
    perf_arm_maximum_broadcast_width_s8_tiny_edge2();
    printf(",\n");
    perf_arm_maximum_broadcast_width_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_maximum_broadcast_width_s8_avg();
    // printf(",\n");
    // perf_arm_maximum_broadcast_width_s8_mobnet();
    // printf(",\n");
    // perf_arm_maximum_broadcast_width_s8_large();
    // printf(",\n");
}

void perf_arm_maximum_broadcast_ch_s8_tiny_edge1(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_ch_int8_tiny_edge1_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_ch_int8_tiny_edge1_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_BATCH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_WIDTH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_BATCH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_WIDTH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE1_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_ch_s8_tiny_edge2(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_ch_int8_tiny_edge2_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_ch_int8_tiny_edge2_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_BATCH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_WIDTH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_BATCH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_WIDTH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_EDGE2_CHANNEL_2};
    

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_ch_s8_tiny(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_ch_int8_tiny_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_ch_int8_tiny_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_BATCH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_WIDTH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_BATCH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_WIDTH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_TINY_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_ch_s8_avg(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_ch_int8_avg_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_ch_int8_avg_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_BATCH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_WIDTH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_BATCH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_WIDTH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_AVG_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_ch_s8_mobnet(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_ch_int8_mobnet_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_ch_int8_mobnet_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_BATCH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_WIDTH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_BATCH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_WIDTH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_MOBNET_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_ch_s8_large(void)
{
    cmsis_nn_context ctx;

    int8_t output[PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_DST_SIZE] = {0};

    int buf_size = 0;
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    cmsis_nn_dims input_1_dims;
    cmsis_nn_dims input_2_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_1_data = perf_maximum_broadcast_ch_int8_large_input_tensor_1;
    const int8_t *input_2_data = perf_maximum_broadcast_ch_int8_large_input_tensor_2;

    input_1_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_BATCH_1;
    input_1_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_HEIGHT_1;
    input_1_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_WIDTH_1;
    input_1_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_CHANNEL_1;

    input_2_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_BATCH_2;
    input_2_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_HEIGHT_2;
    input_2_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_WIDTH_2;
    input_2_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_CHANNEL_2;

    output_dims.n = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_OUTPUT_BATCH;
    output_dims.h = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_OUTPUT_HEIGHT;
    output_dims.w = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_OUTPUT_WIDTH;
    output_dims.c = PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_OUTPUT_CHANNEL;

    const int32_t perf_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_BATCH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_HEIGHT_1,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_WIDTH_1, PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_CHANNEL_1};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_BATCH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_HEIGHT_2,
                                              PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_WIDTH_2, PERF_MAXIMUM_BROADCAST_CH_INT8_LARGE_CHANNEL_2};

    arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims);

    PERF_DWT_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    PERF_PMU_CHARACTERIZE(arm_maximum_s8(&ctx, input_1_data, &input_1_dims, input_2_data, &input_2_dims, output, &output_dims));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    
}

void perf_arm_maximum_broadcast_ch_s8(void)
{
    perf_arm_maximum_broadcast_ch_s8_tiny_edge1();
    printf(",\n");
    perf_arm_maximum_broadcast_ch_s8_tiny_edge2();
    printf(",\n");
    perf_arm_maximum_broadcast_ch_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_maximum_broadcast_ch_s8_avg();
    // printf(",\n");
    // perf_arm_maximum_broadcast_ch_s8_mobnet();
    // printf(",\n");
    // perf_arm_maximum_broadcast_ch_s8_large();
    // printf(",\n");
}