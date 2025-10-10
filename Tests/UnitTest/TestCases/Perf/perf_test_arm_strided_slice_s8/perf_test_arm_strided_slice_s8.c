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

#include "../../TestData/perf_strided_slice_even_stride_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_strided_slice_even_stride_int8_tiny/test_data.h"
#include "../../TestData/perf_strided_slice_even_stride_int8_avg/test_data.h"
#include "../../TestData/perf_strided_slice_even_stride_int8_mobnet/test_data.h"
#include "../../TestData/perf_strided_slice_even_stride_int8_large/test_data.h"
#include "../../TestData/perf_strided_slice_2x3_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_strided_slice_2x3_int8_tiny/test_data.h"
#include "../../TestData/perf_strided_slice_2x3_int8_avg/test_data.h"
#include "../../TestData/perf_strided_slice_2x3_int8_mobnet/test_data.h"
#include "../../TestData/perf_strided_slice_2x3_int8_large/test_data.h"

#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "unity.h"
#include "cmsis_perf_profile.h"

void perf_arm_strided_slice_even_stride_s8_tiny_edge(void)
{
    const int8_t *input_ptr = perf_strided_slice_even_stride_int8_tiny_edge_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_BEGIN_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_BEGIN_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_BEGIN_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_OUTPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_OUTPUT_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_OUTPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_EDGE_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_even_stride_s8_tiny(void)
{
    const int8_t *input_ptr = perf_strided_slice_even_stride_int8_tiny_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_BEGIN_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_BEGIN_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_BEGIN_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_OUTPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_OUTPUT_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_OUTPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_TINY_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_even_stride_s8_avg(void)
{
    const int8_t *input_ptr = perf_strided_slice_even_stride_int8_avg_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_BEGIN_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_BEGIN_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_BEGIN_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_OUTPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_OUTPUT_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_OUTPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_AVG_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_even_stride_s8_mobnet(void)
{
    const int8_t *input_ptr = perf_strided_slice_even_stride_int8_mobnet_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_BEGIN_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_BEGIN_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_BEGIN_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_OUTPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_OUTPUT_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_OUTPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_MOBNET_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_even_stride_s8_large(void)
{
    const int8_t *input_ptr = perf_strided_slice_even_stride_int8_large_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_BEGIN_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_BEGIN_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_BEGIN_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_OUTPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_OUTPUT_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_OUTPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_INPUT_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_INPUT_H, 
                                      PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_INPUT_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_STRIDES_N, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_STRIDES_H, 
        PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_STRIDES_W, PERF_STRIDED_SLICE_EVEN_STRIDE_INT8_LARGE_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_even_stride_s8(void)
{
    perf_arm_strided_slice_even_stride_s8_tiny_edge();
    printf(",\n");
    perf_arm_strided_slice_even_stride_s8_tiny();
    printf(",\n");
    perf_arm_strided_slice_even_stride_s8_avg();
    printf(",\n");
    perf_arm_strided_slice_even_stride_s8_mobnet();
    printf(",\n");
    //perf_arm_strided_slice_even_stride_s8_large();
    //printf(",\n");
}

void perf_arm_strided_slice_2x3_s8_tiny_edge(void)
{
    const int8_t *input_ptr = perf_strided_slice_2x3_int8_tiny_edge_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_BEGIN_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_BEGIN_H, 
        PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_BEGIN_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_OUTPUT_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_OUTPUT_H, 
        PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_OUTPUT_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_EDGE_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_2x3_s8_tiny(void)
{
    const int8_t *input_ptr = perf_strided_slice_2x3_int8_tiny_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_2X3_INT8_TINY_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_2X3_INT8_TINY_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_TINY_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_TINY_BEGIN_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_BEGIN_H, 
        PERF_STRIDED_SLICE_2X3_INT8_TINY_BEGIN_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_TINY_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_TINY_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_TINY_OUTPUT_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_OUTPUT_H, 
        PERF_STRIDED_SLICE_2X3_INT8_TINY_OUTPUT_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_2X3_INT8_TINY_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_TINY_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_2X3_INT8_TINY_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_TINY_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_TINY_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_TINY_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_2x3_s8_avg(void)
{
    const int8_t *input_ptr = perf_strided_slice_2x3_int8_avg_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_2X3_INT8_AVG_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_2X3_INT8_AVG_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_AVG_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_AVG_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_AVG_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_AVG_BEGIN_N, PERF_STRIDED_SLICE_2X3_INT8_AVG_BEGIN_H, 
        PERF_STRIDED_SLICE_2X3_INT8_AVG_BEGIN_W, PERF_STRIDED_SLICE_2X3_INT8_AVG_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_AVG_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_AVG_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_AVG_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_AVG_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_AVG_OUTPUT_N, PERF_STRIDED_SLICE_2X3_INT8_AVG_OUTPUT_H, 
        PERF_STRIDED_SLICE_2X3_INT8_AVG_OUTPUT_W, PERF_STRIDED_SLICE_2X3_INT8_AVG_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_2X3_INT8_AVG_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_AVG_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_AVG_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_AVG_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_2X3_INT8_AVG_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_AVG_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_AVG_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_AVG_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_2x3_s8_mobnet(void)
{
    const int8_t *input_ptr = perf_strided_slice_2x3_int8_mobnet_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_2X3_INT8_MOBNET_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_2X3_INT8_MOBNET_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_MOBNET_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_MOBNET_BEGIN_N, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_BEGIN_H, 
        PERF_STRIDED_SLICE_2X3_INT8_MOBNET_BEGIN_W, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_MOBNET_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_MOBNET_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_MOBNET_OUTPUT_N, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_OUTPUT_H, 
        PERF_STRIDED_SLICE_2X3_INT8_MOBNET_OUTPUT_W, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_2X3_INT8_MOBNET_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_MOBNET_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_2X3_INT8_MOBNET_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_MOBNET_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_MOBNET_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_2x3_s8_large(void)
{
    const int8_t *input_ptr = perf_strided_slice_2x3_int8_large_input_tensor;
    int8_t output_ptr[PERF_STRIDED_SLICE_2X3_INT8_LARGE_OUTPUT_SIZE] = {0};

    const cmsis_nn_dims input_dims = {PERF_STRIDED_SLICE_2X3_INT8_LARGE_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_LARGE_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_LARGE_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_LARGE_INPUT_C};
    const cmsis_nn_dims begin_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_LARGE_BEGIN_N, PERF_STRIDED_SLICE_2X3_INT8_LARGE_BEGIN_H, 
        PERF_STRIDED_SLICE_2X3_INT8_LARGE_BEGIN_W, PERF_STRIDED_SLICE_2X3_INT8_LARGE_BEGIN_C};
    const cmsis_nn_dims stride_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_LARGE_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_LARGE_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_LARGE_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_LARGE_STRIDES_C};
    const cmsis_nn_dims output_dims = {
        PERF_STRIDED_SLICE_2X3_INT8_LARGE_OUTPUT_N, PERF_STRIDED_SLICE_2X3_INT8_LARGE_OUTPUT_H, 
        PERF_STRIDED_SLICE_2X3_INT8_LARGE_OUTPUT_W, PERF_STRIDED_SLICE_2X3_INT8_LARGE_OUTPUT_C};
    
    const int32_t perf_input_tensor_dims[] = {PERF_STRIDED_SLICE_2X3_INT8_LARGE_INPUT_N, PERF_STRIDED_SLICE_2X3_INT8_LARGE_INPUT_H, 
                                      PERF_STRIDED_SLICE_2X3_INT8_LARGE_INPUT_W, PERF_STRIDED_SLICE_2X3_INT8_LARGE_INPUT_C};

    const int32_t perf_rhs_input_tensor_dims[] = {
        PERF_STRIDED_SLICE_2X3_INT8_LARGE_STRIDES_N, PERF_STRIDED_SLICE_2X3_INT8_LARGE_STRIDES_H, 
        PERF_STRIDED_SLICE_2X3_INT8_LARGE_STRIDES_W, PERF_STRIDED_SLICE_2X3_INT8_LARGE_STRIDES_C};


    arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    );

    PERF_DWT_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    PERF_PMU_CHARACTERIZE(arm_strided_slice_s8(
        input_ptr,
        output_ptr,
        &input_dims,
        &begin_dims,
        &stride_dims,
        &output_dims
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_strided_slice_2x3_s8(void)
{
    perf_arm_strided_slice_2x3_s8_tiny_edge();
    printf(",\n");
    perf_arm_strided_slice_2x3_s8_tiny();
    printf(",\n");
    perf_arm_strided_slice_2x3_s8_avg();
    printf(",\n");
    perf_arm_strided_slice_2x3_s8_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_strided_slice_2x3_s8_large();
    // printf(",\n");
}