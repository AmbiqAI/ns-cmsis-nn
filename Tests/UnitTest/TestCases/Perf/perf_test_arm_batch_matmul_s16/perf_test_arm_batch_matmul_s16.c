/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "../../TestData/perf_batch_matmul_no_transpose_int16_tiny_edge/test_data.h"
#include "../../TestData/perf_batch_matmul_no_transpose_int16_tiny/test_data.h"
#include "../../TestData/perf_batch_matmul_no_transpose_int16_avg/test_data.h"
#include "../../TestData/perf_batch_matmul_no_transpose_int16_mobnet/test_data.h"
#include "../../TestData/perf_batch_matmul_no_transpose_int16_large/test_data.h"
#include "../../TestData/perf_batch_matmul_x_transpose_int16_tiny_edge/test_data.h"
#include "../../TestData/perf_batch_matmul_x_transpose_int16_tiny/test_data.h"
#include "../../TestData/perf_batch_matmul_x_transpose_int16_avg/test_data.h"
#include "../../TestData/perf_batch_matmul_x_transpose_int16_mobnet/test_data.h"
#include "../../TestData/perf_batch_matmul_x_transpose_int16_large/test_data.h"
#include "../../TestData/perf_batch_matmul_x_y_transpose_int16_tiny_edge/test_data.h"
#include "../../TestData/perf_batch_matmul_x_y_transpose_int16_tiny/test_data.h"
#include "../../TestData/perf_batch_matmul_x_y_transpose_int16_avg/test_data.h"
#include "../../TestData/perf_batch_matmul_x_y_transpose_int16_mobnet/test_data.h"
#include "../../TestData/perf_batch_matmul_x_y_transpose_int16_large/test_data.h"
#include "../../TestData/perf_batch_matmul_y_transpose_int16_tiny_edge/test_data.h"
#include "../../TestData/perf_batch_matmul_y_transpose_int16_tiny/test_data.h"
#include "../../TestData/perf_batch_matmul_y_transpose_int16_avg/test_data.h"
#include "../../TestData/perf_batch_matmul_y_transpose_int16_mobnet/test_data.h"
#include "../../TestData/perf_batch_matmul_y_transpose_int16_large/test_data.h"
#include <arm_nnfunctions.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unity.h>
#include "cmsis_perf_profile.h"

/*
 * We want to transpose LHS as usual if adj_x is 1. When adj_y is 1, we actually already have the rhs in the desired
 * shape, as our matmul kernel is expecting the rhs to be transposed. We have 2 versions of each input tensor a regular
 * and transposed version. So we just replace _input with _transposed in the tensor name and use that. We also switch
 * the ROWS and COLS macro in the shape.
 */

// Adj_x = 0, Adj_y=0
void perf_arm_batch_matmul_no_transpose_s16_tiny_edge(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_ACTIVATION_MIN, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_ROWS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_COLS};
    // Adj_y = 0, but we actually want to transpose rhs.
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_BATCH,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_ROWS,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_COLS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_ROWS};

    int16_t output[PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_no_transpose_int16_tiny_edge_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_no_transpose_int16_tiny_edge_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_EDGE_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_no_transpose_s16_tiny(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_ACTIVATION_MIN, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_ROWS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_COLS};
    // Adj_y = 0, but we actually want to transpose rhs.
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_BATCH,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_ROWS,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_OUTPUT_COLS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_OUTPUT_ROWS};

    int16_t output[PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_no_transpose_int16_tiny_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_no_transpose_int16_tiny_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_TINY_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_no_transpose_s16_avg(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_ACTIVATION_MIN, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_ROWS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_COLS};
    // Adj_y = 0, but we actually want to transpose rhs.
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_BATCH,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_ROWS,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_OUTPUT_COLS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_OUTPUT_ROWS};

    int16_t output[PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_no_transpose_int16_avg_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_no_transpose_int16_avg_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_AVG_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_no_transpose_s16_mobnet(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_ACTIVATION_MIN, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_ROWS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_COLS};
    // Adj_y = 0, but we actually want to transpose rhs.
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_BATCH,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_ROWS,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_OUTPUT_COLS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_OUTPUT_ROWS};

    int16_t output[PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_no_transpose_int16_mobnet_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_no_transpose_int16_mobnet_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_MOBNET_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_no_transpose_s16_large(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_ACTIVATION_MIN, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_ROWS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_COLS};
    // Adj_y = 0, but we actually want to transpose rhs.
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_BATCH,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_ROWS,
                                 PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_OUTPUT_COLS,
                                  PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_OUTPUT_ROWS};

    int16_t output[PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_no_transpose_int16_large_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_no_transpose_int16_large_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_BATCH, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_ROWS, PERF_BATCH_MATMUL_NO_TRANSPOSE_INT16_LARGE_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_no_transpose_s16(void)
{
    perf_arm_batch_matmul_no_transpose_s16_tiny_edge();
    printf(",\n");
    perf_arm_batch_matmul_no_transpose_s16_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_batch_matmul_no_transpose_s16_avg();
    printf(",\n");
    perf_arm_batch_matmul_no_transpose_s16_mobnet();

    // needs debugging
    // printf(",\n");
    // perf_arm_batch_matmul_no_transpose_s16_large();

    printf(",\n");
}

// Adj_x = 0, Adj_y=1
void perf_arm_batch_matmul_x_transpose_s16_tiny_edge(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_COLS};
    // Adj_y = 1, but we do not want to transpose rhs.
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_transpose_int16_tiny_edge_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_transpose_int16_tiny_edge_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_EDGE_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_transpose_s16_tiny(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_COLS};
    // Adj_y = 1, but we do not want to transpose rhs.
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_transpose_int16_tiny_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_transpose_int16_tiny_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_TINY_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_transpose_s16_avg(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_COLS};
    // Adj_y = 1, but we do not want to transpose rhs.
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_transpose_int16_avg_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_transpose_int16_avg_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_AVG_RHS_COLS};    

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_transpose_s16_mobnet(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_COLS};
    // Adj_y = 1, but we do not want to transpose rhs.
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_transpose_int16_mobnet_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_transpose_int16_mobnet_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_MOBNET_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_transpose_s16_large(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_OUTPUT_SHIFT};
    cmsis_nn_dims lhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_COLS};
    // Adj_y = 1, but we do not want to transpose rhs.
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_transpose_int16_large_lhs_input_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_transpose_int16_large_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_BATCH, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_ROWS, PERF_BATCH_MATMUL_X_TRANSPOSE_INT16_LARGE_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_nt, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_transpose_s16(void)
{
    perf_arm_batch_matmul_x_transpose_s16_tiny_edge();
    printf(",\n");
    perf_arm_batch_matmul_x_transpose_s16_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_batch_matmul_x_transpose_s16_avg();

    printf(",\n");
    perf_arm_batch_matmul_x_transpose_s16_mobnet();

    // needs debugging
    // printf(",\n");
    // perf_arm_batch_matmul_x_transpose_s16_large();

    printf(",\n");
}

// Adj_x = 1, Adj_y=0
void perf_arm_batch_matmul_y_transpose_s16_tiny_edge(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_ACTIVATION_MIN, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_ROWS};
    // Adj_y = 0, but we want to transpose rhs
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_ROWS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_y_transpose_int16_tiny_edge_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_y_transpose_int16_tiny_edge_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_y_transpose_s16_tiny(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_ACTIVATION_MIN, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_ROWS};
    // Adj_y = 0, but we want to transpose rhs
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_ROWS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_y_transpose_int16_tiny_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_y_transpose_int16_tiny_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_TINY_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_y_transpose_s16_avg(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_ACTIVATION_MIN, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_ROWS};
    // Adj_y = 0, but we want to transpose rhs
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_ROWS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_y_transpose_int16_avg_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_y_transpose_int16_avg_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_AVG_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_y_transpose_s16_mobnet(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_ACTIVATION_MIN, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_ROWS};
    // Adj_y = 0, but we want to transpose rhs
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_ROWS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_y_transpose_int16_mobnet_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_y_transpose_int16_mobnet_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_MOBNET_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_y_transpose_s16_large(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_ACTIVATION_MIN, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_ROWS};
    // Adj_y = 0, but we want to transpose rhs
    cmsis_nn_dims rhs_shape_t = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_BATCH,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_HEIGHT,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_COLS,
                                 PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_ROWS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_y_transpose_int16_large_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_y_transpose_int16_large_rhs_transposed_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_BATCH, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_ROWS, PERF_BATCH_MATMUL_Y_TRANSPOSE_INT16_LARGE_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_t, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_y_transpose_s16(void)
{
    perf_arm_batch_matmul_y_transpose_s16_tiny_edge();
    printf(",\n");
    perf_arm_batch_matmul_y_transpose_s16_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_batch_matmul_y_transpose_s16_avg();

    printf(",\n");
    perf_arm_batch_matmul_y_transpose_s16_mobnet();

    // needs debugging
    // printf(",\n");
    // perf_arm_batch_matmul_y_transpose_s16_large();

    printf(",\n");
}

// Adj_x = 1, Adj_y=1
void perf_arm_batch_matmul_x_y_transpose_s16_tiny_edge(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_BATCH,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_COLS,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_ROWS};
    // Adj_y = 1, but we do not want to transpose rhs
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_y_transpose_int16_tiny_edge_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_y_transpose_int16_tiny_edge_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_EDGE_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_y_transpose_s16_tiny(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_BATCH,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_COLS,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_ROWS};
    // Adj_y = 1, but we do not want to transpose rhs
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_y_transpose_int16_tiny_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_y_transpose_int16_tiny_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_TINY_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_y_transpose_s16_avg(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_BATCH,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_COLS,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_ROWS};
    // Adj_y = 1, but we do not want to transpose rhs
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_y_transpose_int16_avg_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_y_transpose_int16_avg_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_AVG_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_y_transpose_s16_mobnet(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_BATCH,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_COLS,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_ROWS};
    // Adj_y = 1, but we do not want to transpose rhs
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_y_transpose_int16_mobnet_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_y_transpose_int16_mobnet_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_MOBNET_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_y_transpose_s16_large(void)
{
    cmsis_nn_context ctx;
    cmsis_nn_bmm_params bmm_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_ADJ_X, // adj_x
                                      PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_ADJ_Y, // adj_y
                                      {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_OFFSET,
                                       PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_OUTPUT_OFFSET,
                                       {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_ACTIVATION_MIN, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_ACTIVATION_MAX}}};
    cmsis_nn_per_tensor_quant_params quant_params = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_OUTPUT_MULTIPLIER,
                                                     PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_OUTPUT_SHIFT};
    // Adj_x = 1, so we transpose lhs
    cmsis_nn_dims lhs_shape_t = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_BATCH,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_HEIGHT,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_COLS,
                                 PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_ROWS};
    // Adj_y = 1, but we do not want to transpose rhs
    cmsis_nn_dims rhs_shape_nt = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_COLS};
    cmsis_nn_dims output_shape = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_OUTPUT_BATCH,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_OUTPUT_HEIGHT,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_OUTPUT_ROWS,
                                  PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_OUTPUT_COLS};

    int16_t output[PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_DST_SIZE] = {0};
    const int16_t *lhs_input = perf_batch_matmul_x_y_transpose_int16_large_lhs_transposed_tensor;
    const int16_t *rhs_input = perf_batch_matmul_x_y_transpose_int16_large_rhs_input_tensor;

    const int32_t perf_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_BATCH, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_HEIGHT,
                                              PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_ROWS, PERF_BATCH_MATMUL_X_Y_TRANSPOSE_INT16_LARGE_RHS_COLS};

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&output_shape);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output);
    PERF_DWT_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

    PERF_PMU_CHARACTERIZE(arm_batch_matmul_s16(
        &ctx, &bmm_params, &quant_params, &lhs_shape_t, lhs_input, &rhs_shape_nt, rhs_input, &output_shape, output));

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

void perf_arm_batch_matmul_x_y_transpose_s16(void)
{
    perf_arm_batch_matmul_x_y_transpose_s16_tiny_edge();
    printf(",\n");
    perf_arm_batch_matmul_x_y_transpose_s16_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_batch_matmul_x_y_transpose_s16_avg();

    printf(",\n");
    perf_arm_batch_matmul_x_y_transpose_s16_mobnet();

    // needs debugging
    // printf(",\n");
    // perf_arm_batch_matmul_x_y_transpose_s16_large();
    
    printf(",\n");
}