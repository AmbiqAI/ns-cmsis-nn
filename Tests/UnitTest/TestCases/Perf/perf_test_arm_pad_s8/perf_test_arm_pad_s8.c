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

#include "../../TestData/perf_pad_int8_tiny/test_data.h"
#include "../../TestData/perf_pad_int8_tiny_edge1/test_data.h"
#include "../../TestData/perf_pad_int8_tiny_edge2/test_data.h"
#include "../../TestData/perf_pad_int8_avg/test_data.h"
#include "../../TestData/perf_pad_int8_mobnet/test_data.h"
#include "../../TestData/perf_pad_int8_large/test_data.h"

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"
#include "cmsis_perf_profile.h"

void perf_arm_pad_s8_tiny(void)
{
    const int8_t *input_ptr = perf_pad_int8_tiny_input_tensor;
    int8_t output_ptr[PERF_PAD_INT8_TINY_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_size = {PERF_PAD_INT8_TINY_INPUT_N, PERF_PAD_INT8_TINY_INPUT_H, 
                                      PERF_PAD_INT8_TINY_INPUT_W, PERF_PAD_INT8_TINY_INPUT_C};
    const cmsis_nn_dims pre_pad = {
        PERF_PAD_INT8_TINY_PRE_PAD_N, PERF_PAD_INT8_TINY_PRE_PAD_H, 
        PERF_PAD_INT8_TINY_PRE_PAD_W, PERF_PAD_INT8_TINY_PRE_PAD_C};
    const cmsis_nn_dims post_pad = {
        PERF_PAD_INT8_TINY_POST_PAD_N, PERF_PAD_INT8_TINY_POST_PAD_H, 
        PERF_PAD_INT8_TINY_POST_PAD_W, PERF_PAD_INT8_TINY_POST_PAD_C};
    const int32_t perf_input_tensor_dims[] = {PERF_PAD_INT8_TINY_INPUT_N, PERF_PAD_INT8_TINY_INPUT_W,
                                              PERF_PAD_INT8_TINY_INPUT_H, PERF_PAD_INT8_TINY_INPUT_C};

    arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_PAD_VALUE, &input_size, &pre_pad, &post_pad);
    PERF_DWT_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    PERF_PMU_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_pad_s8_tiny_edge1(void)
{
    const int8_t *input_ptr = perf_pad_int8_tiny_edge1_input_tensor;
    int8_t output_ptr[PERF_PAD_INT8_TINY_EDGE1_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_size = {PERF_PAD_INT8_TINY_EDGE1_INPUT_N, PERF_PAD_INT8_TINY_EDGE1_INPUT_H, 
                                      PERF_PAD_INT8_TINY_EDGE1_INPUT_W, PERF_PAD_INT8_TINY_EDGE1_INPUT_C};
    const cmsis_nn_dims pre_pad = {
        PERF_PAD_INT8_TINY_EDGE1_PRE_PAD_N, PERF_PAD_INT8_TINY_EDGE1_PRE_PAD_H, 
        PERF_PAD_INT8_TINY_EDGE1_PRE_PAD_W, PERF_PAD_INT8_TINY_EDGE1_PRE_PAD_C};
    const cmsis_nn_dims post_pad = {
        PERF_PAD_INT8_TINY_EDGE1_POST_PAD_N, PERF_PAD_INT8_TINY_EDGE1_POST_PAD_H, 
        PERF_PAD_INT8_TINY_EDGE1_POST_PAD_W, PERF_PAD_INT8_TINY_EDGE1_POST_PAD_C};
    const int32_t perf_input_tensor_dims[] = {PERF_PAD_INT8_TINY_EDGE1_INPUT_N, PERF_PAD_INT8_TINY_EDGE1_INPUT_W,
                                              PERF_PAD_INT8_TINY_EDGE1_INPUT_H, PERF_PAD_INT8_TINY_EDGE1_INPUT_C};

    arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_EDGE1_PAD_VALUE, &input_size, &pre_pad, &post_pad);
    PERF_DWT_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_EDGE1_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    PERF_PMU_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_EDGE1_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_pad_s8_tiny_edge2(void)
{
    const int8_t *input_ptr = perf_pad_int8_tiny_edge2_input_tensor;
    int8_t output_ptr[PERF_PAD_INT8_TINY_EDGE2_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_size = {PERF_PAD_INT8_TINY_EDGE2_INPUT_N, PERF_PAD_INT8_TINY_EDGE2_INPUT_H, 
                                      PERF_PAD_INT8_TINY_EDGE2_INPUT_W, PERF_PAD_INT8_TINY_EDGE2_INPUT_C};
    const cmsis_nn_dims pre_pad = {
        PERF_PAD_INT8_TINY_EDGE2_PRE_PAD_N, PERF_PAD_INT8_TINY_EDGE2_PRE_PAD_H, 
        PERF_PAD_INT8_TINY_EDGE2_PRE_PAD_W, PERF_PAD_INT8_TINY_EDGE2_PRE_PAD_C};
    const cmsis_nn_dims post_pad = {
        PERF_PAD_INT8_TINY_EDGE2_POST_PAD_N, PERF_PAD_INT8_TINY_EDGE2_POST_PAD_H, 
        PERF_PAD_INT8_TINY_EDGE2_POST_PAD_W, PERF_PAD_INT8_TINY_EDGE2_POST_PAD_C};
    const int32_t perf_input_tensor_dims[] = {PERF_PAD_INT8_TINY_EDGE2_INPUT_N, PERF_PAD_INT8_TINY_EDGE2_INPUT_W,
                                              PERF_PAD_INT8_TINY_EDGE2_INPUT_H, PERF_PAD_INT8_TINY_EDGE2_INPUT_C};

    arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_EDGE2_PAD_VALUE, &input_size, &pre_pad, &post_pad);
    PERF_DWT_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_EDGE2_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    PERF_PMU_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_TINY_EDGE2_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_pad_s8_avg(void)
{
    const int8_t *input_ptr = perf_pad_int8_avg_input_tensor;
    int8_t output_ptr[PERF_PAD_INT8_AVG_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_size = {PERF_PAD_INT8_AVG_INPUT_N, PERF_PAD_INT8_AVG_INPUT_H, 
                                      PERF_PAD_INT8_AVG_INPUT_W, PERF_PAD_INT8_AVG_INPUT_C};
    const cmsis_nn_dims pre_pad = {
        PERF_PAD_INT8_AVG_PRE_PAD_N, PERF_PAD_INT8_AVG_PRE_PAD_H, 
        PERF_PAD_INT8_AVG_PRE_PAD_W, PERF_PAD_INT8_AVG_PRE_PAD_C};
    const cmsis_nn_dims post_pad = {
        PERF_PAD_INT8_AVG_POST_PAD_N, PERF_PAD_INT8_AVG_POST_PAD_H, 
        PERF_PAD_INT8_AVG_POST_PAD_W, PERF_PAD_INT8_AVG_POST_PAD_C};
    const int32_t perf_input_tensor_dims[] = {PERF_PAD_INT8_AVG_INPUT_N, PERF_PAD_INT8_AVG_INPUT_W,
                                              PERF_PAD_INT8_AVG_INPUT_H, PERF_PAD_INT8_AVG_INPUT_C};

    arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_AVG_PAD_VALUE, &input_size, &pre_pad, &post_pad);
    PERF_DWT_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_AVG_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    PERF_PMU_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_AVG_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_pad_s8_mobnet(void)
{
    const int8_t *input_ptr = perf_pad_int8_mobnet_input_tensor;
    int8_t output_ptr[PERF_PAD_INT8_MOBNET_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_size = {PERF_PAD_INT8_MOBNET_INPUT_N, PERF_PAD_INT8_MOBNET_INPUT_H, 
                                      PERF_PAD_INT8_MOBNET_INPUT_W, PERF_PAD_INT8_MOBNET_INPUT_C};
    const cmsis_nn_dims pre_pad = {
        PERF_PAD_INT8_MOBNET_PRE_PAD_N, PERF_PAD_INT8_MOBNET_PRE_PAD_H, 
        PERF_PAD_INT8_MOBNET_PRE_PAD_W, PERF_PAD_INT8_MOBNET_PRE_PAD_C};
    const cmsis_nn_dims post_pad = {
        PERF_PAD_INT8_MOBNET_POST_PAD_N, PERF_PAD_INT8_MOBNET_POST_PAD_H, 
        PERF_PAD_INT8_MOBNET_POST_PAD_W, PERF_PAD_INT8_MOBNET_POST_PAD_C};
    const int32_t perf_input_tensor_dims[] = {PERF_PAD_INT8_MOBNET_INPUT_N, PERF_PAD_INT8_MOBNET_INPUT_W,
                                              PERF_PAD_INT8_MOBNET_INPUT_H, PERF_PAD_INT8_MOBNET_INPUT_C};

    arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_MOBNET_PAD_VALUE, &input_size, &pre_pad, &post_pad);
    PERF_DWT_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_MOBNET_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    PERF_PMU_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_MOBNET_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_pad_s8_large(void)
{
    const int8_t *input_ptr = perf_pad_int8_large_input_tensor;
    int8_t output_ptr[PERF_PAD_INT8_LARGE_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_size = {PERF_PAD_INT8_LARGE_INPUT_N, PERF_PAD_INT8_LARGE_INPUT_H, 
                                      PERF_PAD_INT8_LARGE_INPUT_W, PERF_PAD_INT8_LARGE_INPUT_C};
    const cmsis_nn_dims pre_pad = {
        PERF_PAD_INT8_LARGE_PRE_PAD_N, PERF_PAD_INT8_LARGE_PRE_PAD_H, 
        PERF_PAD_INT8_LARGE_PRE_PAD_W, PERF_PAD_INT8_LARGE_PRE_PAD_C};
    const cmsis_nn_dims post_pad = {
        PERF_PAD_INT8_LARGE_POST_PAD_N, PERF_PAD_INT8_LARGE_POST_PAD_H, 
        PERF_PAD_INT8_LARGE_POST_PAD_W, PERF_PAD_INT8_LARGE_POST_PAD_C};
    const int32_t perf_input_tensor_dims[] = {PERF_PAD_INT8_LARGE_INPUT_N, PERF_PAD_INT8_LARGE_INPUT_W,
                                              PERF_PAD_INT8_LARGE_INPUT_H, PERF_PAD_INT8_LARGE_INPUT_C};

    arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_LARGE_PAD_VALUE, &input_size, &pre_pad, &post_pad);
    PERF_DWT_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_LARGE_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    PERF_PMU_CHARACTERIZE(arm_pad_s8(input_ptr, output_ptr, PERF_PAD_INT8_LARGE_PAD_VALUE, &input_size, &pre_pad, &post_pad));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_pad_s8(void)
{
    perf_arm_pad_s8_tiny();
    printf(",\n");
    perf_arm_pad_s8_tiny_edge1();
    printf(",\n");
    perf_arm_pad_s8_tiny_edge2();

    // needs debugging
    // printf(",\n");
    // perf_arm_pad_s8_avg();
    // printf(",\n");
    // perf_arm_pad_s8_mobnet();
    // printf(",\n");
    // perf_arm_pad_s8_large();
    
    printf(",\n");
}