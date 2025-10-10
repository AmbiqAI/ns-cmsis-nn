/*
 * Copyright (C) 2022 Arm Limited or its affiliates.
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

#include "../../TestData/perf_elementwise_mul_int16_tiny_edge/test_data.h"
#include "../../TestData/perf_elementwise_mul_int16_tiny/test_data.h"
#include "../../TestData/perf_elementwise_mul_int16_avg/test_data.h"
#include "../../TestData/perf_elementwise_mul_int16_mobnet/test_data.h"
#include "../../TestData/perf_elementwise_mul_int16_large/test_data.h"

void perf_arm_elementwise_mul_s16_tiny_edge(void)
{
    int16_t output[PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_DST_SIZE] = {0};

    const int16_t *input_data1 = perf_elementwise_mul_int16_tiny_edge_lhs_input_tensor;
    const int16_t *input_data2 = perf_elementwise_mul_int16_tiny_edge_rhs_input_tensor;
    
    const int32_t input_1_offset = PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_LHS_OFFSET;
    const int32_t input_2_offset = PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_RHS_OFFSET;

    const int32_t out_offset = PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_OUTPUT_OFFSET;
    const int32_t out_mult = PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_OUTPUT_MULTIPLIER;
    const int32_t out_shift = PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_OUTPUT_SHIFT;

    const int32_t out_activation_min = PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_ACTIVATION_MIN;
    const int32_t out_activation_max = PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_ACTIVATION_MAX;

    const int32_t perf_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_LHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_LHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_LHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_RHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_RHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_RHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_RHS_COLS};

    PERF_DWT_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_DST_SIZE));

    PERF_PMU_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_TINY_EDGE_DST_SIZE));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_elementwise_mul_s16_tiny(void)
{
    int16_t output[PERF_ELEMENTWISE_MUL_INT16_TINY_DST_SIZE] = {0};

    const int16_t *input_data1 = perf_elementwise_mul_int16_tiny_lhs_input_tensor;
    const int16_t *input_data2 = perf_elementwise_mul_int16_tiny_rhs_input_tensor;
    
    const int32_t input_1_offset = PERF_ELEMENTWISE_MUL_INT16_TINY_LHS_OFFSET;
    const int32_t input_2_offset = PERF_ELEMENTWISE_MUL_INT16_TINY_RHS_OFFSET;

    const int32_t out_offset = PERF_ELEMENTWISE_MUL_INT16_TINY_OUTPUT_OFFSET;
    const int32_t out_mult = PERF_ELEMENTWISE_MUL_INT16_TINY_OUTPUT_MULTIPLIER;
    const int32_t out_shift = PERF_ELEMENTWISE_MUL_INT16_TINY_OUTPUT_SHIFT;

    const int32_t out_activation_min = PERF_ELEMENTWISE_MUL_INT16_TINY_ACTIVATION_MIN;
    const int32_t out_activation_max = PERF_ELEMENTWISE_MUL_INT16_TINY_ACTIVATION_MAX;

    const int32_t perf_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_TINY_LHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_TINY_LHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_TINY_LHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_TINY_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_TINY_RHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_TINY_RHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_TINY_RHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_TINY_RHS_COLS};

    PERF_DWT_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_TINY_DST_SIZE));

    PERF_PMU_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_TINY_DST_SIZE));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_elementwise_mul_s16_avg(void)
{
    int16_t output[PERF_ELEMENTWISE_MUL_INT16_AVG_DST_SIZE] = {0};

    const int16_t *input_data1 = perf_elementwise_mul_int16_avg_lhs_input_tensor;
    const int16_t *input_data2 = perf_elementwise_mul_int16_avg_rhs_input_tensor;
    
    const int32_t input_1_offset = PERF_ELEMENTWISE_MUL_INT16_AVG_LHS_OFFSET;
    const int32_t input_2_offset = PERF_ELEMENTWISE_MUL_INT16_AVG_RHS_OFFSET;

    const int32_t out_offset = PERF_ELEMENTWISE_MUL_INT16_AVG_OUTPUT_OFFSET;
    const int32_t out_mult = PERF_ELEMENTWISE_MUL_INT16_AVG_OUTPUT_MULTIPLIER;
    const int32_t out_shift = PERF_ELEMENTWISE_MUL_INT16_AVG_OUTPUT_SHIFT;

    const int32_t out_activation_min = PERF_ELEMENTWISE_MUL_INT16_AVG_ACTIVATION_MIN;
    const int32_t out_activation_max = PERF_ELEMENTWISE_MUL_INT16_AVG_ACTIVATION_MAX;

    const int32_t perf_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_AVG_LHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_AVG_LHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_AVG_LHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_AVG_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_AVG_RHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_AVG_RHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_AVG_RHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_AVG_RHS_COLS};

    PERF_DWT_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_AVG_DST_SIZE));

    PERF_PMU_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_AVG_DST_SIZE));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_elementwise_mul_s16_mobnet(void)
{
    int16_t output[PERF_ELEMENTWISE_MUL_INT16_MOBNET_DST_SIZE] = {0};

    const int16_t *input_data1 = perf_elementwise_mul_int16_mobnet_lhs_input_tensor;
    const int16_t *input_data2 = perf_elementwise_mul_int16_mobnet_rhs_input_tensor;
    
    const int32_t input_1_offset = PERF_ELEMENTWISE_MUL_INT16_MOBNET_LHS_OFFSET;
    const int32_t input_2_offset = PERF_ELEMENTWISE_MUL_INT16_MOBNET_RHS_OFFSET;

    const int32_t out_offset = PERF_ELEMENTWISE_MUL_INT16_MOBNET_OUTPUT_OFFSET;
    const int32_t out_mult = PERF_ELEMENTWISE_MUL_INT16_MOBNET_OUTPUT_MULTIPLIER;
    const int32_t out_shift = PERF_ELEMENTWISE_MUL_INT16_MOBNET_OUTPUT_SHIFT;

    const int32_t out_activation_min = PERF_ELEMENTWISE_MUL_INT16_MOBNET_ACTIVATION_MIN;
    const int32_t out_activation_max = PERF_ELEMENTWISE_MUL_INT16_MOBNET_ACTIVATION_MAX;

    const int32_t perf_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_MOBNET_LHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_MOBNET_LHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_MOBNET_LHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_MOBNET_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_MOBNET_RHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_MOBNET_RHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_MOBNET_RHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_MOBNET_RHS_COLS};

    PERF_DWT_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_MOBNET_DST_SIZE));

    PERF_PMU_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_MOBNET_DST_SIZE));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_elementwise_mul_s16_large(void)
{
    int16_t output[PERF_ELEMENTWISE_MUL_INT16_LARGE_DST_SIZE] = {0};

    const int16_t *input_data1 = perf_elementwise_mul_int16_large_lhs_input_tensor;
    const int16_t *input_data2 = perf_elementwise_mul_int16_large_rhs_input_tensor;
    
    const int32_t input_1_offset = PERF_ELEMENTWISE_MUL_INT16_LARGE_LHS_OFFSET;
    const int32_t input_2_offset = PERF_ELEMENTWISE_MUL_INT16_LARGE_RHS_OFFSET;

    const int32_t out_offset = PERF_ELEMENTWISE_MUL_INT16_LARGE_OUTPUT_OFFSET;
    const int32_t out_mult = PERF_ELEMENTWISE_MUL_INT16_LARGE_OUTPUT_MULTIPLIER;
    const int32_t out_shift = PERF_ELEMENTWISE_MUL_INT16_LARGE_OUTPUT_SHIFT;

    const int32_t out_activation_min = PERF_ELEMENTWISE_MUL_INT16_LARGE_ACTIVATION_MIN;
    const int32_t out_activation_max = PERF_ELEMENTWISE_MUL_INT16_LARGE_ACTIVATION_MAX;

    const int32_t perf_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_LARGE_LHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_LARGE_LHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_LARGE_LHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_LARGE_LHS_COLS};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_ELEMENTWISE_MUL_INT16_LARGE_RHS_BATCH, PERF_ELEMENTWISE_MUL_INT16_LARGE_RHS_HEIGHT,
                                              PERF_ELEMENTWISE_MUL_INT16_LARGE_RHS_ROWS, PERF_ELEMENTWISE_MUL_INT16_LARGE_RHS_COLS};

    PERF_DWT_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_LARGE_DST_SIZE));

    PERF_PMU_CHARACTERIZE(arm_elementwise_mul_s16(       input_data1,
                                                        input_data2,
                                                        input_1_offset,
                                                        input_2_offset,
                                                        output,
                                                        out_offset,
                                                        out_mult,
                                                        out_shift,
                                                        out_activation_min,
                                                        out_activation_max,
                                                        PERF_ELEMENTWISE_MUL_INT16_LARGE_DST_SIZE));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_elementwise_mul_s16(void)
{
    perf_arm_elementwise_mul_s16_tiny_edge();
    printf(",\n");
    perf_arm_elementwise_mul_s16_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_elementwise_mul_s16_avg();
    // printf(",\n");
    // perf_arm_elementwise_mul_s16_mobnet();
    // printf(",\n");
    // perf_arm_elementwise_mul_s16_large();
    
    printf(",\n");
}