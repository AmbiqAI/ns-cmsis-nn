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

#include "arm_nnfunctions.h"
#include "unity.h"
#include "cmsis_perf_profile.h"

#include "../../TestData/perf_maxpooling_1x1_filter_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_maxpooling_1x1_filter_int8_tiny/test_data.h"
#include "../../TestData/perf_maxpooling_1x1_filter_int8_avg/test_data.h"
#include "../../TestData/perf_maxpooling_1x1_filter_int8_mobnet/test_data.h"
#include "../../TestData/perf_maxpooling_1x1_filter_int8_large/test_data.h"
#include "../../TestData/perf_maxpooling_1xn_filter_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_maxpooling_1xn_filter_int8_tiny/test_data.h"
#include "../../TestData/perf_maxpooling_1xn_filter_int8_avg/test_data.h"
#include "../../TestData/perf_maxpooling_1xn_filter_int8_mobnet/test_data.h"
#include "../../TestData/perf_maxpooling_1xn_filter_int8_large/test_data.h"
#include "../../TestData/perf_maxpooling_nx1_filter_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_maxpooling_nx1_filter_int8_tiny/test_data.h"
#include "../../TestData/perf_maxpooling_nx1_filter_int8_avg/test_data.h"
#include "../../TestData/perf_maxpooling_nx1_filter_int8_mobnet/test_data.h"
#include "../../TestData/perf_maxpooling_nx1_filter_int8_large/test_data.h"
#include "../../TestData/perf_maxpooling_gap_filter_int8_tiny/test_data.h"
#include "../../TestData/perf_maxpooling_gap_filter_int8_avg/test_data.h"
#include "../../TestData/perf_maxpooling_gap_filter_int8_mobnet/test_data.h"
#include "../../TestData/perf_maxpooling_gap_filter_int8_large/test_data.h"
#include "../../TestData/perf_maxpooling_activ_bound_int8_tiny/test_data.h"
#include "../../TestData/perf_maxpooling_activ_bound_int8_avg/test_data.h"
#include "../../TestData/perf_maxpooling_activ_bound_int8_mobnet/test_data.h"
#include "../../TestData/perf_maxpooling_activ_bound_int8_large/test_data.h"
#include "../../TestData/perf_maxpooling_max_filter_int8_tiny/test_data.h"
#include "../../TestData/perf_maxpooling_max_filter_int8_avg/test_data.h"
#include "../../TestData/perf_maxpooling_max_filter_int8_mobnet/test_data.h"
#include "../../TestData/perf_maxpooling_max_filter_int8_large/test_data.h"
#include "../../TestData/perf_maxpooling_no_gap_no_overlap_filter_int8_tiny/test_data.h"
#include "../../TestData/perf_maxpooling_no_gap_no_overlap_filter_int8_avg/test_data.h"
#include "../../TestData/perf_maxpooling_no_gap_no_overlap_filter_int8_mobnet/test_data.h"
#include "../../TestData/perf_maxpooling_no_gap_no_overlap_filter_int8_large/test_data.h"
#include "../../TestData/perf_maxpooling_overlap_filter_int8_tiny/test_data.h"
#include "../../TestData/perf_maxpooling_overlap_filter_int8_avg/test_data.h"
#include "../../TestData/perf_maxpooling_overlap_filter_int8_mobnet/test_data.h"
#include "../../TestData/perf_maxpooling_overlap_filter_int8_large/test_data.h"

void perf_arm_maxpool_1x1_filter_s8_tiny_edge(void)
{
    int8_t output[PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_OUTPUT_W * PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_OUTPUT_H 
       * PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_BATCH_SIZE * PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1x1_filter_int8_tiny_edge_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_INPUT_N, PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_INPUT_W,
                                              PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_INPUT_H, PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_EDGE_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);

    
}

void perf_arm_maxpool_1x1_filter_s8_tiny(void)
{
    int8_t output[PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_OUTPUT_W * PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_OUTPUT_H 
       * PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_BATCH_SIZE * PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1x1_filter_int8_tiny_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_INPUT_N, PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_INPUT_W,
                                              PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_INPUT_H, PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1X1_FILTER_INT8_TINY_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_1x1_filter_s8_avg(void)
{
    int8_t output[PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_OUTPUT_W * PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_OUTPUT_H 
       * PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_BATCH_SIZE * PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1x1_filter_int8_avg_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_INPUT_N, PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_INPUT_W,
                                              PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_INPUT_H, PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1X1_FILTER_INT8_AVG_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_1x1_filter_s8_mobnet(void)
{
    int8_t output[PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_OUTPUT_W * PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_OUTPUT_H 
       * PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_BATCH_SIZE * PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1x1_filter_int8_mobnet_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_INPUT_N, PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_INPUT_W,
                                              PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_INPUT_H, PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1X1_FILTER_INT8_MOBNET_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_1x1_filter_s8_large(void)
{
    int8_t output[PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_OUTPUT_W * PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_OUTPUT_H 
       * PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_BATCH_SIZE * PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1x1_filter_int8_large_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_INPUT_N, PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_INPUT_W,
                                              PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_INPUT_H, PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1X1_FILTER_INT8_LARGE_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_1x1_filter_s8(void)
{
    perf_arm_maxpool_1x1_filter_s8_tiny_edge();
    printf(",\n");
    perf_arm_maxpool_1x1_filter_s8_tiny();
    printf(",\n");
    perf_arm_maxpool_1x1_filter_s8_avg();
    printf(",\n");
    perf_arm_maxpool_1x1_filter_s8_mobnet();

    // needs debugging
    // printf(",\n");
    // perf_arm_maxpool_1x1_filter_s8_large();

    printf(",\n");
}

void perf_arm_maxpool_1xn_filter_s8_tiny_edge(void)
{
    int8_t output[PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_OUTPUT_W * PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_OUTPUT_H 
       * PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_BATCH_SIZE * PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1xn_filter_int8_tiny_edge_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_INPUT_N, PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_INPUT_W,
                                              PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_INPUT_H, PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_EDGE_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_1xn_filter_s8_tiny(void)
{
    int8_t output[PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_OUTPUT_W * PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_OUTPUT_H 
       * PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_BATCH_SIZE * PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1xn_filter_int8_tiny_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_INPUT_N, PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_INPUT_W,
                                              PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_INPUT_H, PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1XN_FILTER_INT8_TINY_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_1xn_filter_s8_avg(void)
{
    int8_t output[PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_OUTPUT_W * PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_OUTPUT_H 
       * PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_BATCH_SIZE * PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1xn_filter_int8_avg_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_INPUT_N, PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_INPUT_W,
                                              PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_INPUT_H, PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1XN_FILTER_INT8_AVG_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
    
}

void perf_arm_maxpool_1xn_filter_s8_mobnet(void)
{
    int8_t output[PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_OUTPUT_W * PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_OUTPUT_H 
       * PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_BATCH_SIZE * PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1xn_filter_int8_mobnet_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_INPUT_N, PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_INPUT_W,
                                              PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_INPUT_H, PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1XN_FILTER_INT8_MOBNET_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
    
}

void perf_arm_maxpool_1xn_filter_s8_large(void)
{
    int8_t output[PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_OUTPUT_W * PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_OUTPUT_H 
       * PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_BATCH_SIZE * PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_1xn_filter_int8_large_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_INPUT_N, PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_INPUT_W,
                                              PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_INPUT_H, PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_1XN_FILTER_INT8_LARGE_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_1xn_filter_s8(void)
{
    perf_arm_maxpool_1xn_filter_s8_tiny_edge();
    printf(",\n");
    perf_arm_maxpool_1xn_filter_s8_tiny();
    printf(",\n");
    perf_arm_maxpool_1xn_filter_s8_avg();
    printf(",\n");
    perf_arm_maxpool_1xn_filter_s8_mobnet();

    // needs debugging
    // printf(",\n");
    // perf_arm_maxpool_1xn_filter_s8_large();

    printf(",\n");
}

void perf_arm_maxpool_nx1_filter_s8_tiny_edge(void)
{
    int8_t output[PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_OUTPUT_W * PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_OUTPUT_H 
       * PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_BATCH_SIZE * PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_nx1_filter_int8_tiny_edge_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_INPUT_N, PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_INPUT_W,
                                              PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_INPUT_H, PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_EDGE_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_nx1_filter_s8_tiny(void)
{
    int8_t output[PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_OUTPUT_W * PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_OUTPUT_H 
       * PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_BATCH_SIZE * PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_nx1_filter_int8_tiny_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_INPUT_N, PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_INPUT_W,
                                              PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_INPUT_H, PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NX1_FILTER_INT8_TINY_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_nx1_filter_s8_avg(void)
{
    int8_t output[PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_OUTPUT_W * PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_OUTPUT_H 
       * PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_BATCH_SIZE * PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_nx1_filter_int8_avg_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_INPUT_N, PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_INPUT_W,
                                              PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_INPUT_H, PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NX1_FILTER_INT8_AVG_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_nx1_filter_s8_mobnet(void)
{
    int8_t output[PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_OUTPUT_W * PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_OUTPUT_H 
       * PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_BATCH_SIZE * PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_nx1_filter_int8_mobnet_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_INPUT_N, PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_INPUT_W,
                                              PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_INPUT_H, PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NX1_FILTER_INT8_MOBNET_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_nx1_filter_s8_large(void)
{
    int8_t output[PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_OUTPUT_W * PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_OUTPUT_H 
       * PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_BATCH_SIZE * PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_nx1_filter_int8_large_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_INPUT_N, PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_INPUT_W,
                                              PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_INPUT_H, PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NX1_FILTER_INT8_LARGE_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_nx1_filter_s8(void)
{
    perf_arm_maxpool_nx1_filter_s8_tiny_edge();
    printf(",\n");
    perf_arm_maxpool_nx1_filter_s8_tiny();
    printf(",\n");
    perf_arm_maxpool_nx1_filter_s8_avg();
    printf(",\n");
    perf_arm_maxpool_nx1_filter_s8_mobnet();

    // needs debugging
    // printf(",\n");
    // perf_arm_maxpool_nx1_filter_s8_large();

    printf(",\n");
}

void perf_arm_maxpool_activ_bound_s8_tiny(void)
{
    int8_t output[PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_OUTPUT_W * PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_OUTPUT_H 
       * PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_BATCH_SIZE * PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_activ_bound_int8_tiny_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_INPUT_N, PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_INPUT_W,
                                              PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_INPUT_H, PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_ACTIV_BOUND_INT8_TINY_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_activ_bound_s8_avg(void)
{
    int8_t output[PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_OUTPUT_W * PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_OUTPUT_H 
       * PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_BATCH_SIZE * PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_activ_bound_int8_avg_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_INPUT_N, PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_INPUT_W,
                                              PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_INPUT_H, PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_ACTIV_BOUND_INT8_AVG_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_activ_bound_s8_mobnet(void)
{
    int8_t output[PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_OUTPUT_W * PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_OUTPUT_H 
       * PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_BATCH_SIZE * PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_activ_bound_int8_mobnet_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_INPUT_N, PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_INPUT_W,
                                              PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_INPUT_H, PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_ACTIV_BOUND_INT8_MOBNET_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_activ_bound_s8_large(void)
{
    int8_t output[PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_OUTPUT_W * PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_OUTPUT_H 
       * PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_BATCH_SIZE * PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_activ_bound_int8_large_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_INPUT_N, PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_INPUT_W,
                                              PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_INPUT_H, PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_ACTIV_BOUND_INT8_LARGE_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_activ_bound_s8(void)
{
    perf_arm_maxpool_activ_bound_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_maxpool_activ_bound_s8_avg();
    // printf(",\n");
    // perf_arm_maxpool_activ_bound_s8_mobnet();
    // printf(",\n");
    // perf_arm_maxpool_activ_bound_s8_large();
    
    printf(",\n");
}

void perf_arm_maxpool_max_filter_s8_tiny(void)
{
    int8_t output[PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_OUTPUT_W * PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_OUTPUT_H 
       * PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_BATCH_SIZE * PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_max_filter_int8_tiny_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_INPUT_N, PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_INPUT_W,
                                              PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_INPUT_H, PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_MAX_FILTER_INT8_TINY_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_max_filter_s8_avg(void)
{
    int8_t output[PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_OUTPUT_W * PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_OUTPUT_H 
       * PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_BATCH_SIZE * PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_max_filter_int8_avg_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_INPUT_N, PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_INPUT_W,
                                              PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_INPUT_H, PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_MAX_FILTER_INT8_AVG_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_max_filter_s8_mobnet(void)
{
    int8_t output[PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_OUTPUT_W * PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_OUTPUT_H 
       * PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_BATCH_SIZE * PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_max_filter_int8_mobnet_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_INPUT_N, PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_INPUT_W,
                                              PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_INPUT_H, PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_MAX_FILTER_INT8_MOBNET_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_max_filter_s8_large(void)
{
    int8_t output[PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_OUTPUT_W * PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_OUTPUT_H 
       * PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_BATCH_SIZE * PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_max_filter_int8_large_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_INPUT_N, PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_INPUT_W,
                                              PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_INPUT_H, PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_MAX_FILTER_INT8_LARGE_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_max_filter_s8(void)
{
    perf_arm_maxpool_max_filter_s8_tiny();
    printf(",\n");
    perf_arm_maxpool_max_filter_s8_avg();
    printf(",\n");
    perf_arm_maxpool_max_filter_s8_mobnet();
    printf(",\n");
    perf_arm_maxpool_max_filter_s8_large();
    printf(",\n");
}

void perf_arm_maxpool_no_gap_no_overlap_filter_s8_tiny(void)
{
    int8_t output[PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_OUTPUT_W * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_OUTPUT_H 
       * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_BATCH_SIZE * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_no_gap_no_overlap_filter_int8_tiny_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_INPUT_N, PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_INPUT_W,
                                              PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_INPUT_H, PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_TINY_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_no_gap_no_overlap_filter_s8_avg(void)
{
    int8_t output[PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_OUTPUT_W * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_OUTPUT_H 
       * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_BATCH_SIZE * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_no_gap_no_overlap_filter_int8_avg_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_INPUT_N, PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_INPUT_W,
                                              PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_INPUT_H, PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_AVG_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_no_gap_no_overlap_filter_s8_mobnet(void)
{
    int8_t output[PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_W * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_H 
       * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_BATCH_SIZE * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_no_gap_no_overlap_filter_int8_mobnet_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_INPUT_N, PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_INPUT_W,
                                              PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_INPUT_H, PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_MOBNET_ACTIVATION_MAX;
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_no_gap_no_overlap_filter_s8_large(void)
{
    int8_t output[PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_OUTPUT_W * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_OUTPUT_H 
       * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_BATCH_SIZE * PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_no_gap_no_overlap_filter_int8_large_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_INPUT_N, PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_INPUT_W,
                                              PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_INPUT_H, PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_NO_GAP_NO_OVERLAP_FILTER_INT8_LARGE_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_no_gap_no_overlap_filter_s8(void)
{
    perf_arm_maxpool_no_gap_no_overlap_filter_s8_tiny();
    printf(",\n");
    perf_arm_maxpool_no_gap_no_overlap_filter_s8_avg();
    printf(",\n");
    perf_arm_maxpool_no_gap_no_overlap_filter_s8_mobnet();
    printf(",\n");
    perf_arm_maxpool_no_gap_no_overlap_filter_s8_large();
    printf(",\n");
}

void perf_arm_maxpool_overlap_filter_s8_tiny(void)
{
    int8_t output[PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_OUTPUT_W * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_OUTPUT_H 
       * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_BATCH_SIZE * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_overlap_filter_int8_tiny_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_INPUT_N, PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_INPUT_W,
                                              PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_INPUT_H, PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_TINY_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_overlap_filter_s8_avg(void)
{
    int8_t output[PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_OUTPUT_W * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_OUTPUT_H 
       * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_BATCH_SIZE * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_overlap_filter_int8_avg_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_INPUT_N, PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_INPUT_W,
                                              PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_INPUT_H, PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_AVG_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_overlap_filter_s8_mobnet(void)
{
    int8_t output[PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_W * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_H 
       * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_BATCH_SIZE * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_overlap_filter_int8_mobnet_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_INPUT_N, PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_INPUT_W,
                                              PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_INPUT_H, PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_MOBNET_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_overlap_filter_s8_large(void)
{
    int8_t output[PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_OUTPUT_W * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_OUTPUT_H 
       * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_BATCH_SIZE * PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_OUTPUT_C] = {0};

    cmsis_nn_context ctx = {NULL, 0};
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = perf_maxpooling_overlap_filter_int8_large_input_tensor;
    const int32_t perf_input_tensor_dims[] = {PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_INPUT_N, PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_INPUT_W,
                                              PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_INPUT_H, PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_INPUT_C};

    input_dims.n = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_BATCH_SIZE;
    input_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_INPUT_W;
    input_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_INPUT_H;
    input_dims.c = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_INPUT_C;
    filter_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_FILTER_W;
    filter_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_FILTER_H;
    output_dims.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_OUTPUT_W;
    output_dims.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_OUTPUT_H;
    output_dims.c = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_OUTPUT_C;

    pool_params.padding.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_PADDING_W;
    pool_params.padding.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_PADDING_H;
    pool_params.stride.w = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_STRIDE_W;
    pool_params.stride.h = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_STRIDE_H;

    pool_params.activation.min = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_ACTIVATION_MIN;
    pool_params.activation.max = PERF_MAXPOOLING_OVERLAP_FILTER_INT8_LARGE_ACTIVATION_MAX;

    arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output);
    PERF_DWT_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));
    
    PERF_PMU_CHARACTERIZE(arm_max_pool_s8(&ctx, &pool_params, &input_dims, 
                          input_data, &filter_dims, &output_dims, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_maxpool_overlap_filter_s8(void)
{
    perf_arm_maxpool_overlap_filter_s8_tiny();
    printf(",\n");
    perf_arm_maxpool_overlap_filter_s8_avg();
    printf(",\n");
    perf_arm_maxpool_overlap_filter_s8_mobnet();
    printf(",\n");
    perf_arm_maxpool_overlap_filter_s8_large();
    printf(",\n");
}