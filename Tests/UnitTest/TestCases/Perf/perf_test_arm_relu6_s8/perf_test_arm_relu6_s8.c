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
#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include "cmsis_perf_profile.h"

#include "../../TestData/perf_relu6_basic_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_relu6_basic_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_relu6_basic_int8_tiny/test_data.h"
#include "../../TestData/perf_relu6_basic_int8_avg/test_data.h"
#include "../../TestData/perf_relu6_basic_int8_mobnet/test_data.h"
#include "../../TestData/perf_relu6_basic_int8_large/test_data.h"

#ifndef RELU6_BASIC_LOWER
#define RELU6_BASIC_LOWER 0
#endif

#ifndef RELU6_BASIC_UPPER
#define RELU6_BASIC_UPPER 6
#endif

void perf_arm_relu6_basic_s8_tiny_edge1(void)
{
    const int8_t *input_data = perf_relu6_basic_int8_tiny_edge1_input_tensor;
    int8_t output[PERF_RELU6_BASIC_INT8_TINY_EDGE1_OUTPUT_LEN];

    const int32_t perf_input_tensor_dims[] = {PERF_RELU6_BASIC_INT8_TINY_EDGE1_BATCH_1, PERF_RELU6_BASIC_INT8_TINY_EDGE1_HEIGHT_1,
                                              PERF_RELU6_BASIC_INT8_TINY_EDGE1_WIDTH_1, PERF_RELU6_BASIC_INT8_TINY_EDGE1_CHANNEL_1};

    arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_EDGE1_OUTPUT_LEN
    );

    PERF_DWT_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_EDGE1_OUTPUT_LEN
    ));

    PERF_PMU_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_EDGE1_OUTPUT_LEN
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_relu6_basic_s8_tiny_edge2(void)
{
    const int8_t *input_data = perf_relu6_basic_int8_tiny_edge2_input_tensor;
    int8_t output[PERF_RELU6_BASIC_INT8_TINY_EDGE2_OUTPUT_LEN];

    const int32_t perf_input_tensor_dims[] = {PERF_RELU6_BASIC_INT8_TINY_EDGE2_BATCH_1, PERF_RELU6_BASIC_INT8_TINY_EDGE2_HEIGHT_1,
                                              PERF_RELU6_BASIC_INT8_TINY_EDGE2_WIDTH_1, PERF_RELU6_BASIC_INT8_TINY_EDGE2_CHANNEL_1};

    arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_EDGE2_OUTPUT_LEN
    );

    PERF_DWT_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_EDGE2_OUTPUT_LEN
    ));

    PERF_PMU_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_EDGE2_OUTPUT_LEN
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_relu6_basic_s8_tiny(void)
{
    const int8_t *input_data = perf_relu6_basic_int8_tiny_input_tensor;
    int8_t output[PERF_RELU6_BASIC_INT8_TINY_OUTPUT_LEN];

    const int32_t perf_input_tensor_dims[] = {PERF_RELU6_BASIC_INT8_TINY_BATCH_1, PERF_RELU6_BASIC_INT8_TINY_HEIGHT_1,
                                              PERF_RELU6_BASIC_INT8_TINY_WIDTH_1, PERF_RELU6_BASIC_INT8_TINY_CHANNEL_1};

    arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_OUTPUT_LEN
    );

    PERF_DWT_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_OUTPUT_LEN
    ));

    PERF_PMU_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_TINY_OUTPUT_LEN
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_relu6_basic_s8_avg(void)
{
    const int8_t *input_data = perf_relu6_basic_int8_avg_input_tensor;
    int8_t output[PERF_RELU6_BASIC_INT8_AVG_OUTPUT_LEN];

    const int32_t perf_input_tensor_dims[] = {PERF_RELU6_BASIC_INT8_AVG_BATCH_1, PERF_RELU6_BASIC_INT8_AVG_HEIGHT_1,
                                              PERF_RELU6_BASIC_INT8_AVG_WIDTH_1, PERF_RELU6_BASIC_INT8_AVG_CHANNEL_1};

    arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_AVG_OUTPUT_LEN
    );

    PERF_DWT_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_AVG_OUTPUT_LEN
    ));

    PERF_PMU_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_AVG_OUTPUT_LEN
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_relu6_basic_s8_mobnet(void)
{
    const int8_t *input_data = perf_relu6_basic_int8_mobnet_input_tensor;
    int8_t output[PERF_RELU6_BASIC_INT8_MOBNET_OUTPUT_LEN];

    const int32_t perf_input_tensor_dims[] = {PERF_RELU6_BASIC_INT8_MOBNET_BATCH_1, PERF_RELU6_BASIC_INT8_MOBNET_HEIGHT_1,
                                              PERF_RELU6_BASIC_INT8_MOBNET_WIDTH_1, PERF_RELU6_BASIC_INT8_MOBNET_CHANNEL_1};

    arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_MOBNET_OUTPUT_LEN
    );

    PERF_DWT_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_MOBNET_OUTPUT_LEN
    ));

    PERF_PMU_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_MOBNET_OUTPUT_LEN
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_relu6_basic_s8_large(void)
{
    const int8_t *input_data = perf_relu6_basic_int8_large_input_tensor;
    int8_t output[PERF_RELU6_BASIC_INT8_LARGE_OUTPUT_LEN];

    const int32_t perf_input_tensor_dims[] = {PERF_RELU6_BASIC_INT8_LARGE_BATCH_1, PERF_RELU6_BASIC_INT8_LARGE_HEIGHT_1,
                                              PERF_RELU6_BASIC_INT8_LARGE_WIDTH_1, PERF_RELU6_BASIC_INT8_LARGE_CHANNEL_1};

    arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_LARGE_OUTPUT_LEN
    );

    PERF_DWT_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_LARGE_OUTPUT_LEN
    ));

    PERF_PMU_CHARACTERIZE(arm_relu6_s8(
        input_data,
        RELU6_BASIC_LOWER,
        RELU6_BASIC_UPPER,
        output,
        PERF_RELU6_BASIC_INT8_LARGE_OUTPUT_LEN
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);
    
    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_relu6_basic_s8(void)
{
    perf_arm_relu6_basic_s8_tiny_edge1();
    printf(",\n");
    perf_arm_relu6_basic_s8_tiny_edge2();
    printf(",\n");
    perf_arm_relu6_basic_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_relu6_basic_s8_avg();
    // printf(",\n");
    // perf_arm_relu6_basic_s8_mobnet();
    // printf(",\n");
    // perf_arm_relu6_basic_s8_large();
    
    printf(",\n");
}