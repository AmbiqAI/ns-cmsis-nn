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

#include <stdlib.h>

#include <arm_nnfunctions.h>
#include <unity.h>
#include "cmsis_perf_profile.h"

#include "../../TestData/perf_mean_axis_c_int16_tiny_edge1/test_data.h"
#include "../../TestData/perf_mean_axis_c_int16_tiny_edge2/test_data.h"
#include "../../TestData/perf_mean_axis_c_int16_tiny/test_data.h"
#include "../../TestData/perf_mean_axis_c_int16_avg/test_data.h"
#include "../../TestData/perf_mean_axis_c_int16_mobnet/test_data.h"
#include "../../TestData/perf_mean_axis_c_int16_large/test_data.h"
#include "../../TestData/perf_mean_axis_h_int16_tiny_edge1/test_data.h"
#include "../../TestData/perf_mean_axis_h_int16_tiny_edge2/test_data.h"
#include "../../TestData/perf_mean_axis_h_int16_tiny/test_data.h"
#include "../../TestData/perf_mean_axis_h_int16_avg/test_data.h"
#include "../../TestData/perf_mean_axis_h_int16_mobnet/test_data.h"
#include "../../TestData/perf_mean_axis_h_int16_large/test_data.h"
#include "../../TestData/perf_mean_axis_n_int16_tiny_edge1/test_data.h"
#include "../../TestData/perf_mean_axis_n_int16_tiny_edge2/test_data.h"
#include "../../TestData/perf_mean_axis_n_int16_tiny/test_data.h"
#include "../../TestData/perf_mean_axis_n_int16_avg/test_data.h"
#include "../../TestData/perf_mean_axis_n_int16_mobnet/test_data.h"
#include "../../TestData/perf_mean_axis_n_int16_large/test_data.h"
#include "../../TestData/perf_mean_axis_hw_int16_tiny_edge1/test_data.h"
#include "../../TestData/perf_mean_axis_hw_int16_tiny_edge2/test_data.h"
#include "../../TestData/perf_mean_axis_hw_int16_tiny/test_data.h"
#include "../../TestData/perf_mean_axis_hw_int16_avg/test_data.h"
#include "../../TestData/perf_mean_axis_hw_int16_mobnet/test_data.h"
#include "../../TestData/perf_mean_axis_hw_int16_large/test_data.h"
#include "../../TestData/perf_mean_axis_hwc_int16_tiny_edge1/test_data.h"
#include "../../TestData/perf_mean_axis_hwc_int16_tiny_edge2/test_data.h"
#include "../../TestData/perf_mean_axis_hwc_int16_tiny/test_data.h"
#include "../../TestData/perf_mean_axis_hwc_int16_avg/test_data.h"
#include "../../TestData/perf_mean_axis_hwc_int16_mobnet/test_data.h"
#include "../../TestData/perf_mean_axis_hwc_int16_large/test_data.h"
#include "../../TestData/perf_mean_axis_nhwc_int16_tiny_edge1/test_data.h"
#include "../../TestData/perf_mean_axis_nhwc_int16_tiny_edge2/test_data.h"
#include "../../TestData/perf_mean_axis_nhwc_int16_tiny/test_data.h"
#include "../../TestData/perf_mean_axis_nhwc_int16_avg/test_data.h"
#include "../../TestData/perf_mean_axis_nhwc_int16_mobnet/test_data.h"
#include "../../TestData/perf_mean_axis_nhwc_int16_large/test_data.h"
#include "../../TestData/perf_mean_axis_w_int16_tiny_edge1/test_data.h"
#include "../../TestData/perf_mean_axis_w_int16_tiny_edge2/test_data.h"
#include "../../TestData/perf_mean_axis_w_int16_tiny/test_data.h"
#include "../../TestData/perf_mean_axis_w_int16_avg/test_data.h"
#include "../../TestData/perf_mean_axis_w_int16_mobnet/test_data.h"
#include "../../TestData/perf_mean_axis_w_int16_large/test_data.h"
#include "../../TestData/perf_mean_axis_wc_int16_tiny_edge1/test_data.h"
#include "../../TestData/perf_mean_axis_wc_int16_tiny_edge2/test_data.h"
#include "../../TestData/perf_mean_axis_wc_int16_tiny/test_data.h"
#include "../../TestData/perf_mean_axis_wc_int16_avg/test_data.h"
#include "../../TestData/perf_mean_axis_wc_int16_mobnet/test_data.h"
#include "../../TestData/perf_mean_axis_wc_int16_large/test_data.h"

void perf_arm_mean_axis_n_s16_tiny_edge1(void)
{
    int16_t output_data[PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_n_int16_tiny_edge1_input_tensor;

    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_n_s16_tiny_edge2(void)
{
    int16_t output_data[PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_n_int16_tiny_edge2_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_n_s16_tiny(void)
{
    int16_t output_data[PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_N_INT16_TINY_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_N_INT16_TINY_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_N_INT16_TINY_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_n_int16_tiny_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_N_INT16_TINY_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_TINY_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_n_s16_avg(void)
{
    int16_t output_data[PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_N_INT16_AVG_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_N_INT16_AVG_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_N_INT16_AVG_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_n_int16_avg_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_N_INT16_AVG_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_AVG_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_n_s16_mobnet(void)
{
    int16_t output_data[PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_N_INT16_MOBNET_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_N_INT16_MOBNET_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_N_INT16_MOBNET_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_n_int16_mobnet_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_N_INT16_MOBNET_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_MOBNET_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_n_s16_large(void)
{
    int16_t output_data[PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_N_INT16_LARGE_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_N_INT16_LARGE_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_N_INT16_LARGE_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_n_int16_large_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_N_INT16_LARGE_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_N_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_N_INT16_LARGE_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_n_s16(void)
{
    perf_arm_mean_axis_n_s16_tiny_edge1();
    printf(",\n");
    perf_arm_mean_axis_n_s16_tiny_edge2();
    printf(",\n");
    perf_arm_mean_axis_n_s16_tiny();
    printf(",\n");
    perf_arm_mean_axis_n_s16_avg();
    printf(",\n");
    perf_arm_mean_axis_n_s16_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_mean_axis_n_s16_large();
    // printf(",\n");
}

void perf_arm_mean_axis_c_s16_tiny_edge1(void)
{
    int16_t output_data[PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_c_int16_tiny_edge1_input_tensor;

    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_c_s16_tiny_edge2(void)
{
    int16_t output_data[PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_c_int16_tiny_edge2_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_c_s16_tiny(void)
{
    int16_t output_data[PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_C_INT16_TINY_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_C_INT16_TINY_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_C_INT16_TINY_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_c_int16_tiny_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_C_INT16_TINY_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_TINY_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_c_s16_avg(void)
{
    int16_t output_data[PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_C_INT16_AVG_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_C_INT16_AVG_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_C_INT16_AVG_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_c_int16_avg_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_C_INT16_AVG_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_AVG_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_c_s16_mobnet(void)
{
    int16_t output_data[PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_C_INT16_MOBNET_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_C_INT16_MOBNET_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_C_INT16_MOBNET_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_c_int16_mobnet_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_C_INT16_MOBNET_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_MOBNET_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_c_s16_large(void)
{
    int16_t output_data[PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_C_INT16_LARGE_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_C_INT16_LARGE_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_C_INT16_LARGE_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_c_int16_large_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_C_INT16_LARGE_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_C_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_C_INT16_LARGE_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_c_s16(void)
{
    perf_arm_mean_axis_c_s16_tiny_edge1();
    printf(",\n");
    perf_arm_mean_axis_c_s16_tiny_edge2();
    printf(",\n");
    perf_arm_mean_axis_c_s16_tiny();
    printf(",\n");
    perf_arm_mean_axis_c_s16_avg();
    printf(",\n");
    perf_arm_mean_axis_c_s16_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_mean_axis_c_s16_large();
    // printf(",\n");
}

void perf_arm_mean_axis_h_s16_tiny_edge1(void)
{
    int16_t output_data[PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_h_int16_tiny_edge1_input_tensor;

    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_h_s16_tiny_edge2(void)
{
    int16_t output_data[PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_h_int16_tiny_edge2_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_h_s16_tiny(void)
{
    int16_t output_data[PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_H_INT16_TINY_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_H_INT16_TINY_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_H_INT16_TINY_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_h_int16_tiny_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_H_INT16_TINY_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_TINY_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_h_s16_avg(void)
{
    int16_t output_data[PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_H_INT16_AVG_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_H_INT16_AVG_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_H_INT16_AVG_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_h_int16_avg_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_H_INT16_AVG_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_AVG_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_h_s16_mobnet(void)
{
    int16_t output_data[PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_H_INT16_MOBNET_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_H_INT16_MOBNET_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_H_INT16_MOBNET_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_h_int16_mobnet_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_H_INT16_MOBNET_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_MOBNET_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_h_s16_large(void)
{
    int16_t output_data[PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_H_INT16_LARGE_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_H_INT16_LARGE_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_H_INT16_LARGE_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_h_int16_large_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_H_INT16_LARGE_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_H_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_H_INT16_LARGE_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_h_s16(void)
{
    perf_arm_mean_axis_h_s16_tiny_edge1();
    printf(",\n");
    perf_arm_mean_axis_h_s16_tiny_edge2();
    printf(",\n");
    perf_arm_mean_axis_h_s16_tiny();
    printf(",\n");
    perf_arm_mean_axis_h_s16_avg();
    printf(",\n");
    perf_arm_mean_axis_h_s16_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_mean_axis_h_s16_large();
    // printf(",\n");
}

void perf_arm_mean_axis_w_s16_tiny_edge1(void)
{
    int16_t output_data[PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_w_int16_tiny_edge1_input_tensor;

    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_w_s16_tiny_edge2(void)
{
    int16_t output_data[PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_w_int16_tiny_edge2_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_w_s16_tiny(void)
{
    int16_t output_data[PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_W_INT16_TINY_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_W_INT16_TINY_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_W_INT16_TINY_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_w_int16_tiny_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_W_INT16_TINY_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_TINY_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_w_s16_avg(void)
{
    int16_t output_data[PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_W_INT16_AVG_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_W_INT16_AVG_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_W_INT16_AVG_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_w_int16_avg_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_W_INT16_AVG_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_AVG_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_w_s16_mobnet(void)
{
    int16_t output_data[PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_W_INT16_MOBNET_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_W_INT16_MOBNET_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_W_INT16_MOBNET_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_w_int16_mobnet_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_W_INT16_MOBNET_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_MOBNET_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_w_s16_large(void)
{
    int16_t output_data[PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_W_INT16_LARGE_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_W_INT16_LARGE_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_W_INT16_LARGE_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_w_int16_large_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_W_INT16_LARGE_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_W_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_W_INT16_LARGE_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_w_s16(void)
{
    perf_arm_mean_axis_w_s16_tiny_edge1();
    printf(",\n");
    perf_arm_mean_axis_w_s16_tiny_edge2();
    printf(",\n");
    perf_arm_mean_axis_w_s16_tiny();
    printf(",\n");
    perf_arm_mean_axis_w_s16_avg();
    printf(",\n");
    perf_arm_mean_axis_w_s16_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_mean_axis_w_s16_large();
    // printf(",\n");
}

void perf_arm_mean_axis_hw_s16_tiny_edge1(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hw_int16_tiny_edge1_input_tensor;

    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hw_s16_tiny_edge2(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hw_int16_tiny_edge2_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hw_s16_tiny(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HW_INT16_TINY_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HW_INT16_TINY_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HW_INT16_TINY_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hw_int16_tiny_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HW_INT16_TINY_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_TINY_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hw_s16_avg(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HW_INT16_AVG_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HW_INT16_AVG_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HW_INT16_AVG_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hw_int16_avg_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HW_INT16_AVG_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_AVG_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hw_s16_mobnet(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HW_INT16_MOBNET_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HW_INT16_MOBNET_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HW_INT16_MOBNET_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hw_int16_mobnet_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HW_INT16_MOBNET_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_MOBNET_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hw_s16_large(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HW_INT16_LARGE_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HW_INT16_LARGE_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HW_INT16_LARGE_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hw_int16_large_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HW_INT16_LARGE_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HW_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HW_INT16_LARGE_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hw_s16(void)
{
    perf_arm_mean_axis_hw_s16_tiny_edge1();
    printf(",\n");
    perf_arm_mean_axis_hw_s16_tiny_edge2();
    printf(",\n");
    perf_arm_mean_axis_hw_s16_tiny();
    printf(",\n");
    perf_arm_mean_axis_hw_s16_avg();
    printf(",\n");
    perf_arm_mean_axis_hw_s16_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_mean_axis_hw_s16_large();
    // printf(",\n");
}

void perf_arm_mean_axis_hwc_s16_tiny_edge1(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hwc_int16_tiny_edge1_input_tensor;

    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hwc_s16_tiny_edge2(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hwc_int16_tiny_edge2_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hwc_s16_tiny(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HWC_INT16_TINY_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hwc_int16_tiny_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HWC_INT16_TINY_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_TINY_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hwc_s16_avg(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HWC_INT16_AVG_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HWC_INT16_AVG_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HWC_INT16_AVG_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hwc_int16_avg_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HWC_INT16_AVG_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_AVG_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hwc_s16_mobnet(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HWC_INT16_MOBNET_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HWC_INT16_MOBNET_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hwc_int16_mobnet_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HWC_INT16_MOBNET_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_MOBNET_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hwc_s16_large(void)
{
    int16_t output_data[PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_HWC_INT16_LARGE_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_HWC_INT16_LARGE_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_HWC_INT16_LARGE_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_hwc_int16_large_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_HWC_INT16_LARGE_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_HWC_INT16_LARGE_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_hwc_s16(void)
{
    perf_arm_mean_axis_hwc_s16_tiny_edge1();
    printf(",\n");
    perf_arm_mean_axis_hwc_s16_tiny_edge2();
    printf(",\n");
    perf_arm_mean_axis_hwc_s16_tiny();
    printf(",\n");
    perf_arm_mean_axis_hwc_s16_avg();
    printf(",\n");
    perf_arm_mean_axis_hwc_s16_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_mean_axis_hwc_s16_large();
    // printf(",\n");
}

void perf_arm_mean_axis_nhwc_s16_tiny_edge1(void)
{
    int16_t output_data[PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_nhwc_int16_tiny_edge1_input_tensor;

    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_nhwc_s16_tiny_edge2(void)
{
    int16_t output_data[PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_nhwc_int16_tiny_edge2_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_nhwc_s16_tiny(void)
{
    int16_t output_data[PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_NHWC_INT16_TINY_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_nhwc_int16_tiny_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_NHWC_INT16_TINY_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_TINY_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_nhwc_s16_avg(void)
{
    int16_t output_data[PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_NHWC_INT16_AVG_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_NHWC_INT16_AVG_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_NHWC_INT16_AVG_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_nhwc_int16_avg_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_NHWC_INT16_AVG_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_AVG_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_nhwc_s16_mobnet(void)
{
    int16_t output_data[PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_NHWC_INT16_MOBNET_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_NHWC_INT16_MOBNET_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_nhwc_int16_mobnet_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_NHWC_INT16_MOBNET_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_MOBNET_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_nhwc_s16_large(void)
{
    int16_t output_data[PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_NHWC_INT16_LARGE_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_NHWC_INT16_LARGE_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_nhwc_int16_large_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_NHWC_INT16_LARGE_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_NHWC_INT16_LARGE_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_nhwc_s16(void)
{
    perf_arm_mean_axis_nhwc_s16_tiny_edge1();
    printf(",\n");
    perf_arm_mean_axis_nhwc_s16_tiny_edge2();
    printf(",\n");
    perf_arm_mean_axis_nhwc_s16_tiny();
    printf(",\n");
    perf_arm_mean_axis_nhwc_s16_avg();
    printf(",\n");
    perf_arm_mean_axis_nhwc_s16_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_mean_axis_nhwc_s16_large();
    // printf(",\n");
}

void perf_arm_mean_axis_wc_s16_tiny_edge1(void)
{
    int16_t output_data[PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_wc_int16_tiny_edge1_input_tensor;

    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE1_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_wc_s16_tiny_edge2(void)
{
    int16_t output_data[PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_wc_int16_tiny_edge2_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_EDGE2_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_wc_s16_tiny(void)
{
    int16_t output_data[PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_WC_INT16_TINY_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_WC_INT16_TINY_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_WC_INT16_TINY_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_wc_int16_tiny_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_WC_INT16_TINY_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_TINY_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_wc_s16_avg(void)
{
    int16_t output_data[PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_WC_INT16_AVG_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_WC_INT16_AVG_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_WC_INT16_AVG_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_wc_int16_avg_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_WC_INT16_AVG_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_AVG_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_AVG_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_wc_s16_mobnet(void)
{
    int16_t output_data[PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_WC_INT16_MOBNET_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_WC_INT16_MOBNET_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_WC_INT16_MOBNET_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_wc_int16_mobnet_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_WC_INT16_MOBNET_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_MOBNET_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_wc_s16_large(void)
{
    int16_t output_data[PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_SIZE] = {0};
    int16_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_MEAN_AXIS_WC_INT16_LARGE_IN_DIM;
    const cmsis_nn_dims axis_dims = PERF_MEAN_AXIS_WC_INT16_LARGE_AXIS_DIM;
    const cmsis_nn_dims output_dims = PERF_MEAN_AXIS_WC_INT16_LARGE_OUT_DIM;

    const int16_t *input_data = perf_mean_axis_wc_int16_large_input_tensor;
        
    const int32_t perf_input_tensor_dims[] = PERF_MEAN_AXIS_WC_INT16_LARGE_IN_DIM;

    arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_SHIFT
    );

    PERF_DWT_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_SHIFT
    ));

    PERF_PMU_CHARACTERIZE(arm_mean_s16(
        input_data,
        &input_dims,
        PERF_MEAN_AXIS_WC_INT16_LARGE_INPUT_OFFSET,
        &axis_dims,
        output_ptr,
        &output_dims,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_OFFSET,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_MULTIPLIER,
        PERF_MEAN_AXIS_WC_INT16_LARGE_OUTPUT_SHIFT
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_mean_axis_wc_s16(void)
{
    perf_arm_mean_axis_wc_s16_tiny_edge1();
    printf(",\n");
    perf_arm_mean_axis_wc_s16_tiny_edge2();
    printf(",\n");
    perf_arm_mean_axis_wc_s16_tiny();
    printf(",\n");
    perf_arm_mean_axis_wc_s16_avg();
    printf(",\n");
    perf_arm_mean_axis_wc_s16_mobnet();
    printf(",\n");

    // needs debugging
    // perf_arm_mean_axis_wc_s16_large();
    // printf(",\n");
}