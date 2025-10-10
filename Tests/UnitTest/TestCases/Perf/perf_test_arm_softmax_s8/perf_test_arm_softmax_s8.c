/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "unity.h"
#include "arm_nnfunctions.h"
#include "cmsis_perf_profile.h"

#include "../../TestData/perf_softmax_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_softmax_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_softmax_int8_tiny/test_data.h"
#include "../../TestData/perf_softmax_int8_avg/test_data.h"
#include "../../TestData/perf_softmax_int8_mobnet/test_data.h"
#include "../../TestData/perf_softmax_int8_large/test_data.h"

void perf_arm_softmax_s8_tiny_edge1(void)
{
    const int32_t num_rows = PERF_SOFTMAX_INT8_TINY_EDGE1_NUM_ROWS;
    const int32_t row_size = PERF_SOFTMAX_INT8_TINY_EDGE1_ROW_SIZE;
    const int32_t mult = PERF_SOFTMAX_INT8_TINY_EDGE1_INPUT_MULT;
    const int32_t shift = PERF_SOFTMAX_INT8_TINY_EDGE1_INPUT_LEFT_SHIFT;
    const int32_t diff_min = PERF_SOFTMAX_INT8_TINY_EDGE1_DIFF_MIN;
    const int8_t *input_data = perf_softmax_int8_tiny_edge1_input_tensor;
    int8_t output[PERF_SOFTMAX_INT8_TINY_EDGE1_DST_SIZE];

    const int32_t perf_input_tensor_dims[] = {PERF_SOFTMAX_INT8_TINY_EDGE1_NUM_ROWS,
                                              PERF_SOFTMAX_INT8_TINY_EDGE1_ROW_SIZE};

    // Print JSON for function being tested and dimension of test inputs

    // Initial function call to populate the cache
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);

    // DWT Cycle Count measurement
    PERF_DWT_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    // PMU MACs count
    PERF_PMU_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_softmax_s8_tiny_edge2(void)
{
    const int32_t num_rows = PERF_SOFTMAX_INT8_TINY_EDGE2_NUM_ROWS;
    const int32_t row_size = PERF_SOFTMAX_INT8_TINY_EDGE2_ROW_SIZE;
    const int32_t mult = PERF_SOFTMAX_INT8_TINY_EDGE2_INPUT_MULT;
    const int32_t shift = PERF_SOFTMAX_INT8_TINY_EDGE2_INPUT_LEFT_SHIFT;
    const int32_t diff_min = PERF_SOFTMAX_INT8_TINY_EDGE2_DIFF_MIN;
    const int8_t *input_data = perf_softmax_int8_tiny_edge2_input_tensor;
    int8_t output[PERF_SOFTMAX_INT8_TINY_EDGE2_DST_SIZE];

    const int32_t perf_input_tensor_dims[] = {PERF_SOFTMAX_INT8_TINY_EDGE2_NUM_ROWS,
                                              PERF_SOFTMAX_INT8_TINY_EDGE2_ROW_SIZE};

    // Print JSON for function being tested and dimension of test inputs

    // Initial function call to populate the cache
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);

    // DWT Cycle Count measurement
    PERF_DWT_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    // PMU MACs count
    PERF_PMU_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_softmax_s8_tiny(void)
{
    const int32_t num_rows = PERF_SOFTMAX_INT8_TINY_NUM_ROWS;
    const int32_t row_size = PERF_SOFTMAX_INT8_TINY_ROW_SIZE;
    const int32_t mult = PERF_SOFTMAX_INT8_TINY_INPUT_MULT;
    const int32_t shift = PERF_SOFTMAX_INT8_TINY_INPUT_LEFT_SHIFT;
    const int32_t diff_min = PERF_SOFTMAX_INT8_TINY_DIFF_MIN;
    const int8_t *input_data = perf_softmax_int8_tiny_input_tensor;
    int8_t output[PERF_SOFTMAX_INT8_TINY_DST_SIZE];

    const int32_t perf_input_tensor_dims[] = {PERF_SOFTMAX_INT8_TINY_NUM_ROWS,
                                              PERF_SOFTMAX_INT8_TINY_ROW_SIZE};

    // Print JSON for function being tested and dimension of test inputs

    // Initial function call to populate the cache
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);

    // DWT Cycle Count measurement
    PERF_DWT_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    // PMU MACs count
    PERF_PMU_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_softmax_s8_avg(void)
{
    const int32_t num_rows = PERF_SOFTMAX_INT8_AVG_NUM_ROWS;
    const int32_t row_size = PERF_SOFTMAX_INT8_AVG_ROW_SIZE;
    const int32_t mult = PERF_SOFTMAX_INT8_AVG_INPUT_MULT;
    const int32_t shift = PERF_SOFTMAX_INT8_AVG_INPUT_LEFT_SHIFT;
    const int32_t diff_min = PERF_SOFTMAX_INT8_AVG_DIFF_MIN;
    const int8_t *input_data = perf_softmax_int8_avg_input_tensor;
    int8_t output[PERF_SOFTMAX_INT8_AVG_DST_SIZE];

    const int32_t perf_input_tensor_dims[] = {PERF_SOFTMAX_INT8_AVG_NUM_ROWS,
                                              PERF_SOFTMAX_INT8_AVG_ROW_SIZE};

    // Print JSON for function being tested and dimension of test inputs

    // Initial function call to populate the cache
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);

    // DWT Cycle Count measurement
    PERF_DWT_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    // PMU MACs count
    PERF_PMU_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_softmax_s8_mobnet(void)
{
    const int32_t num_rows = PERF_SOFTMAX_INT8_MOBNET_NUM_ROWS;
    const int32_t row_size = PERF_SOFTMAX_INT8_MOBNET_ROW_SIZE;
    const int32_t mult = PERF_SOFTMAX_INT8_MOBNET_INPUT_MULT;
    const int32_t shift = PERF_SOFTMAX_INT8_MOBNET_INPUT_LEFT_SHIFT;
    const int32_t diff_min = PERF_SOFTMAX_INT8_MOBNET_DIFF_MIN;
    const int8_t *input_data = perf_softmax_int8_mobnet_input_tensor;
    int8_t output[PERF_SOFTMAX_INT8_MOBNET_DST_SIZE];

    const int32_t perf_input_tensor_dims[] = {PERF_SOFTMAX_INT8_MOBNET_NUM_ROWS,
                                              PERF_SOFTMAX_INT8_MOBNET_ROW_SIZE};

    // Print JSON for function being tested and dimension of test inputs

    // Initial function call to populate the cache
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);

    // DWT Cycle Count measurement
    PERF_DWT_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    // PMU MACs count
    PERF_PMU_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_softmax_s8_large(void)
{
    printf("Before init any\n");
    const int32_t num_rows = PERF_SOFTMAX_INT8_LARGE_NUM_ROWS;
    const int32_t row_size = PERF_SOFTMAX_INT8_LARGE_ROW_SIZE;
    const int32_t mult = PERF_SOFTMAX_INT8_LARGE_INPUT_MULT;
    const int32_t shift = PERF_SOFTMAX_INT8_LARGE_INPUT_LEFT_SHIFT;
    const int32_t diff_min = PERF_SOFTMAX_INT8_LARGE_DIFF_MIN;
    const int8_t *input_data = perf_softmax_int8_large_input_tensor;
    int8_t output[PERF_SOFTMAX_INT8_LARGE_DST_SIZE];

    const int32_t perf_input_tensor_dims[] = {PERF_SOFTMAX_INT8_LARGE_NUM_ROWS,
                                              PERF_SOFTMAX_INT8_LARGE_ROW_SIZE};

    // Print JSON for function being tested and dimension of test inputs

    printf("Cache populate call:\n");
    // Initial function call to populate the cache
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);
        
    printf("DWT perf call:\n");
    // DWT Cycle Count measurement
    PERF_DWT_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    printf("PMU perf call:\n");
    // PMU MACs count
    PERF_PMU_CHARACTERIZE(arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_softmax_s8(void)
{
    perf_arm_softmax_s8_tiny_edge1();
    printf(",\n");
    perf_arm_softmax_s8_tiny_edge2();
    printf(",\n");
    perf_arm_softmax_s8_tiny();
    printf(",\n");
    perf_arm_softmax_s8_avg();
    printf(",\n");
    perf_arm_softmax_s8_mobnet();

    // needs debugging
    // printf(",\n");
    // perf_arm_softmax_s8_large();
    
    printf(",\n");
}

