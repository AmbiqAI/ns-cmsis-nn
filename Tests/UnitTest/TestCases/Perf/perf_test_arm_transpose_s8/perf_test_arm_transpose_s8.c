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

#include "../../TestData/perf_transpose_int8_3dim_tiny_edge1/test_data.h"
#include "../../TestData/perf_transpose_int8_3dim_tiny_edge2/test_data.h"
#include "../../TestData/perf_transpose_int8_default_tiny_edge1/test_data.h"
#include "../../TestData/perf_transpose_int8_default_tiny_edge2/test_data.h"
#include "../../TestData/perf_transpose_int8_default_tiny/test_data.h"
#include "../../TestData/perf_transpose_int8_default_avg/test_data.h"
#include "../../TestData/perf_transpose_int8_default_mobnet/test_data.h"
#include "../../TestData/perf_transpose_int8_default_large/test_data.h"
#include "../../TestData/perf_transpose_int8_chwn_tiny_edge1/test_data.h"
#include "../../TestData/perf_transpose_int8_chwn_tiny_edge2/test_data.h"
#include "../../TestData/perf_transpose_int8_chwn_tiny/test_data.h"
#include "../../TestData/perf_transpose_int8_chwn_avg/test_data.h"
#include "../../TestData/perf_transpose_int8_chwn_mobnet/test_data.h"
#include "../../TestData/perf_transpose_int8_chwn_large/test_data.h"
#include "../../TestData/perf_transpose_int8_matrix/test_data.h"
#include "../../TestData/perf_transpose_int8_nchw_tiny_edge1/test_data.h"
#include "../../TestData/perf_transpose_int8_nchw_tiny_edge2/test_data.h"
#include "../../TestData/perf_transpose_int8_nchw_tiny/test_data.h"
#include "../../TestData/perf_transpose_int8_nchw_avg/test_data.h"
#include "../../TestData/perf_transpose_int8_nchw_mobnet/test_data.h"
#include "../../TestData/perf_transpose_int8_nchw_large/test_data.h"
#include "../../TestData/perf_transpose_int8_ncwh_tiny_edge1/test_data.h"
#include "../../TestData/perf_transpose_int8_ncwh_tiny_edge2/test_data.h"
#include "../../TestData/perf_transpose_int8_ncwh_tiny/test_data.h"
#include "../../TestData/perf_transpose_int8_ncwh_avg/test_data.h"
#include "../../TestData/perf_transpose_int8_ncwh_mobnet/test_data.h"
#include "../../TestData/perf_transpose_int8_ncwh_large/test_data.h"
#include "../../TestData/perf_transpose_int8_nhcw_tiny_edge1/test_data.h"
#include "../../TestData/perf_transpose_int8_nhcw_tiny_edge2/test_data.h"
#include "../../TestData/perf_transpose_int8_nhcw_tiny/test_data.h"
#include "../../TestData/perf_transpose_int8_nhcw_avg/test_data.h"
#include "../../TestData/perf_transpose_int8_nhcw_mobnet/test_data.h"
#include "../../TestData/perf_transpose_int8_nhcw_large/test_data.h"
#include "../../TestData/perf_transpose_int8_nwhc_tiny_edge1/test_data.h"
#include "../../TestData/perf_transpose_int8_nwhc_tiny_edge2/test_data.h"
#include "../../TestData/perf_transpose_int8_nwhc_tiny/test_data.h"
#include "../../TestData/perf_transpose_int8_nwhc_avg/test_data.h"
#include "../../TestData/perf_transpose_int8_nwhc_mobnet/test_data.h"
#include "../../TestData/perf_transpose_int8_nwhc_large/test_data.h"
#include "../../TestData/perf_transpose_int8_wchn_tiny_edge1/test_data.h"
#include "../../TestData/perf_transpose_int8_wchn_tiny_edge2/test_data.h"
#include "../../TestData/perf_transpose_int8_wchn_tiny/test_data.h"
#include "../../TestData/perf_transpose_int8_wchn_avg/test_data.h"
#include "../../TestData/perf_transpose_int8_wchn_mobnet/test_data.h"
#include "../../TestData/perf_transpose_int8_wchn_large/test_data.h"

void perf_arm_transpose_default_s8_tiny_edge1(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE1_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE1_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE1_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_default_tiny_edge1_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE1_PERM_SIZE] = PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE1_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE1_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_default_s8_tiny_edge2(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE2_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE2_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_default_tiny_edge2_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE2_PERM_SIZE] = PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE2_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_DEFAULT_TINY_EDGE2_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_default_s8_tiny(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_DEFAULT_TINY_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_DEFAULT_TINY_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_DEFAULT_TINY_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_DEFAULT_TINY_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_default_tiny_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_DEFAULT_TINY_PERM_SIZE] = PERF_TRANSPOSE_INT8_DEFAULT_TINY_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_DEFAULT_TINY_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_default_s8_avg(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_DEFAULT_AVG_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_DEFAULT_AVG_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_DEFAULT_AVG_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_DEFAULT_AVG_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_default_avg_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_DEFAULT_AVG_PERM_SIZE] = PERF_TRANSPOSE_INT8_DEFAULT_AVG_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_DEFAULT_AVG_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_default_s8_mobnet(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_DEFAULT_MOBNET_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_DEFAULT_MOBNET_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_DEFAULT_MOBNET_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_DEFAULT_MOBNET_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_default_mobnet_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_DEFAULT_MOBNET_PERM_SIZE] = PERF_TRANSPOSE_INT8_DEFAULT_MOBNET_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_DEFAULT_MOBNET_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_default_s8_large(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_DEFAULT_LARGE_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_DEFAULT_LARGE_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_DEFAULT_LARGE_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_DEFAULT_LARGE_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_default_large_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_DEFAULT_LARGE_PERM_SIZE] = PERF_TRANSPOSE_INT8_DEFAULT_LARGE_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_DEFAULT_LARGE_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_default_s8(void)
{
    perf_arm_transpose_default_s8_tiny_edge1();
    printf(",\n");
    perf_arm_transpose_default_s8_tiny_edge2();
    printf(",\n");
    perf_arm_transpose_default_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_transpose_default_s8_avg();
    // printf(",\n");
    // perf_arm_transpose_default_s8_mobnet();
    // printf(",\n");
    // perf_arm_transpose_default_s8_large();

    printf(",\n");
}

void perf_arm_transpose_nhcw_s8_tiny_edge1(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE1_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE1_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE1_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nhcw_tiny_edge1_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE1_PERM_SIZE] = PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE1_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE1_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nhcw_s8_tiny_edge2(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE2_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE2_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nhcw_tiny_edge2_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE2_PERM_SIZE] = PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE2_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NHCW_TINY_EDGE2_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nhcw_s8_tiny(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NHCW_TINY_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NHCW_TINY_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NHCW_TINY_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NHCW_TINY_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nhcw_tiny_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NHCW_TINY_PERM_SIZE] = PERF_TRANSPOSE_INT8_NHCW_TINY_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NHCW_TINY_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nhcw_s8_avg(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NHCW_AVG_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NHCW_AVG_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NHCW_AVG_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NHCW_AVG_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nhcw_avg_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NHCW_AVG_PERM_SIZE] = PERF_TRANSPOSE_INT8_NHCW_AVG_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NHCW_AVG_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nhcw_s8_mobnet(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NHCW_MOBNET_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NHCW_MOBNET_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NHCW_MOBNET_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NHCW_MOBNET_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nhcw_mobnet_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NHCW_MOBNET_PERM_SIZE] = PERF_TRANSPOSE_INT8_NHCW_MOBNET_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NHCW_MOBNET_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nhcw_s8_large(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NHCW_LARGE_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NHCW_LARGE_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NHCW_LARGE_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NHCW_LARGE_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nhcw_large_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NHCW_LARGE_PERM_SIZE] = PERF_TRANSPOSE_INT8_NHCW_LARGE_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NHCW_LARGE_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nhcw_s8(void)
{
    perf_arm_transpose_nhcw_s8_tiny_edge1();
    printf(",\n");
    perf_arm_transpose_nhcw_s8_tiny_edge2();
    printf(",\n");
    perf_arm_transpose_nhcw_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_transpose_nhcw_s8_avg();
    // printf(",\n");
    // perf_arm_transpose_nhcw_s8_mobnet();
    // printf(",\n");
    // perf_arm_transpose_nhcw_s8_large();

    printf(",\n");
}

void perf_arm_transpose_wchn_s8_tiny_edge1(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE1_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE1_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE1_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_wchn_tiny_edge1_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE1_PERM_SIZE] = PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE1_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE1_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_wchn_s8_tiny_edge2(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE2_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE2_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_wchn_tiny_edge2_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE2_PERM_SIZE] = PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE2_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_WCHN_TINY_EDGE2_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_wchn_s8_tiny(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_WCHN_TINY_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_WCHN_TINY_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_WCHN_TINY_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_WCHN_TINY_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_wchn_tiny_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_WCHN_TINY_PERM_SIZE] = PERF_TRANSPOSE_INT8_WCHN_TINY_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_WCHN_TINY_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_wchn_s8_avg(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_WCHN_AVG_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_WCHN_AVG_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_WCHN_AVG_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_WCHN_AVG_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_wchn_avg_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_WCHN_AVG_PERM_SIZE] = PERF_TRANSPOSE_INT8_WCHN_AVG_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_WCHN_AVG_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_wchn_s8_mobnet(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_WCHN_MOBNET_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_WCHN_MOBNET_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_WCHN_MOBNET_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_WCHN_MOBNET_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_wchn_mobnet_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_WCHN_MOBNET_PERM_SIZE] = PERF_TRANSPOSE_INT8_WCHN_MOBNET_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_WCHN_MOBNET_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_wchn_s8_large(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_WCHN_LARGE_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_WCHN_LARGE_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_WCHN_LARGE_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_WCHN_LARGE_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_wchn_large_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_WCHN_LARGE_PERM_SIZE] = PERF_TRANSPOSE_INT8_WCHN_LARGE_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_WCHN_LARGE_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_wchn_s8(void)
{
    perf_arm_transpose_wchn_s8_tiny_edge1();
    printf(",\n");
    perf_arm_transpose_wchn_s8_tiny_edge2();
    printf(",\n");
    perf_arm_transpose_wchn_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_transpose_wchn_s8_avg();
    // printf(",\n");
    // perf_arm_transpose_wchn_s8_mobnet();
    // printf(",\n");
    // perf_arm_transpose_wchn_s8_large();

    printf(",\n");
}

void perf_arm_transpose_nchw_s8_tiny_edge1(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE1_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE1_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE1_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nchw_tiny_edge1_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE1_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE1_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE1_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nchw_s8_tiny_edge2(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE2_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE2_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nchw_tiny_edge2_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE2_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE2_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCHW_TINY_EDGE2_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nchw_s8_tiny(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCHW_TINY_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCHW_TINY_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCHW_TINY_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCHW_TINY_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nchw_tiny_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCHW_TINY_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCHW_TINY_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCHW_TINY_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nchw_s8_avg(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCHW_AVG_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCHW_AVG_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCHW_AVG_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCHW_AVG_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nchw_avg_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCHW_AVG_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCHW_AVG_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCHW_AVG_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nchw_s8_mobnet(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCHW_MOBNET_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCHW_MOBNET_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCHW_MOBNET_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCHW_MOBNET_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nchw_mobnet_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCHW_MOBNET_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCHW_MOBNET_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCHW_MOBNET_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nchw_s8_large(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCHW_LARGE_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCHW_LARGE_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCHW_LARGE_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCHW_LARGE_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nchw_large_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCHW_LARGE_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCHW_LARGE_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCHW_LARGE_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nchw_s8(void)
{
    perf_arm_transpose_nchw_s8_tiny_edge1();
    printf(",\n");
    perf_arm_transpose_nchw_s8_tiny_edge2();
    printf(",\n");
    perf_arm_transpose_nchw_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_transpose_nchw_s8_avg();
    // printf(",\n");
    // perf_arm_transpose_nchw_s8_mobnet();
    // printf(",\n");
    // perf_arm_transpose_nchw_s8_large();

    printf(",\n");
}

void perf_arm_transpose_chwn_s8_tiny_edge1(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE1_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE1_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE1_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_chwn_tiny_edge1_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE1_PERM_SIZE] = PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE1_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE1_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_chwn_s8_tiny_edge2(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE2_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE2_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_chwn_tiny_edge2_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE2_PERM_SIZE] = PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE2_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_CHWN_TINY_EDGE2_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_chwn_s8_tiny(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_CHWN_TINY_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_CHWN_TINY_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_CHWN_TINY_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_CHWN_TINY_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_chwn_tiny_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_CHWN_TINY_PERM_SIZE] = PERF_TRANSPOSE_INT8_CHWN_TINY_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_CHWN_TINY_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_chwn_s8_avg(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_CHWN_AVG_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_CHWN_AVG_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_CHWN_AVG_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_CHWN_AVG_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_chwn_avg_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_CHWN_AVG_PERM_SIZE] = PERF_TRANSPOSE_INT8_CHWN_AVG_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_CHWN_AVG_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_chwn_s8_mobnet(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_CHWN_MOBNET_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_CHWN_MOBNET_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_CHWN_MOBNET_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_CHWN_MOBNET_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_chwn_mobnet_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_CHWN_MOBNET_PERM_SIZE] = PERF_TRANSPOSE_INT8_CHWN_MOBNET_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_CHWN_MOBNET_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_chwn_s8_large(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_CHWN_LARGE_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_CHWN_LARGE_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_CHWN_LARGE_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_CHWN_LARGE_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_chwn_large_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_CHWN_LARGE_PERM_SIZE] = PERF_TRANSPOSE_INT8_CHWN_LARGE_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_CHWN_LARGE_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_chwn_s8(void)
{
    perf_arm_transpose_chwn_s8_tiny_edge1();
    printf(",\n");
    perf_arm_transpose_chwn_s8_tiny_edge2();
    printf(",\n");
    perf_arm_transpose_chwn_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_transpose_chwn_s8_avg();
    // printf(",\n");
    // perf_arm_transpose_chwn_s8_mobnet();
    // printf(",\n");
    // perf_arm_transpose_chwn_s8_large();

    printf(",\n");
}

void perf_arm_transpose_matrix_s8_matrix_size(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_MATRIX_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_MATRIX_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_MATRIX_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_MATRIX_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_matrix_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_MATRIX_PERM_SIZE] = PERF_TRANSPOSE_INT8_MATRIX_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_MATRIX_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_matrix_s8(void)
{
    perf_arm_transpose_matrix_s8_matrix_size();
    printf(",\n");
}

void perf_arm_transpose_ncwh_s8_tiny_edge1(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE1_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE1_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE1_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_ncwh_tiny_edge1_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE1_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE1_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE1_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_ncwh_s8_tiny_edge2(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE2_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE2_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_ncwh_tiny_edge2_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE2_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE2_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCWH_TINY_EDGE2_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_ncwh_s8_tiny(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCWH_TINY_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCWH_TINY_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCWH_TINY_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCWH_TINY_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_ncwh_tiny_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCWH_TINY_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCWH_TINY_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCWH_TINY_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_ncwh_s8_avg(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCWH_AVG_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCWH_AVG_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCWH_AVG_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCWH_AVG_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_ncwh_avg_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCWH_AVG_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCWH_AVG_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCWH_AVG_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_ncwh_s8_mobnet(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCWH_MOBNET_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCWH_MOBNET_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCWH_MOBNET_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCWH_MOBNET_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_ncwh_mobnet_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCWH_MOBNET_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCWH_MOBNET_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCWH_MOBNET_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_ncwh_s8_large(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NCWH_LARGE_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NCWH_LARGE_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NCWH_LARGE_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NCWH_LARGE_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_ncwh_large_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NCWH_LARGE_PERM_SIZE] = PERF_TRANSPOSE_INT8_NCWH_LARGE_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NCWH_LARGE_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_ncwh_s8(void)
{
    perf_arm_transpose_ncwh_s8_tiny_edge1();
    printf(",\n");
    perf_arm_transpose_ncwh_s8_tiny_edge2();
    printf(",\n");
    perf_arm_transpose_ncwh_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_transpose_ncwh_s8_avg();
    // printf(",\n");
    // perf_arm_transpose_ncwh_s8_mobnet();
    // printf(",\n");
    // perf_arm_transpose_ncwh_s8_large();

    printf(",\n");
}

void perf_arm_transpose_nwhc_s8_tiny_edge1(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE1_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE1_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE1_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nwhc_tiny_edge1_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE1_PERM_SIZE] = PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE1_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE1_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nwhc_s8_tiny_edge2(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE2_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE2_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nwhc_tiny_edge2_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE2_PERM_SIZE] = PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE2_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NWHC_TINY_EDGE2_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nwhc_s8_tiny(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NWHC_TINY_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NWHC_TINY_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NWHC_TINY_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NWHC_TINY_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nwhc_tiny_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NWHC_TINY_PERM_SIZE] = PERF_TRANSPOSE_INT8_NWHC_TINY_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NWHC_TINY_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nwhc_s8_avg(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NWHC_AVG_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NWHC_AVG_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NWHC_AVG_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NWHC_AVG_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nwhc_avg_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NWHC_AVG_PERM_SIZE] = PERF_TRANSPOSE_INT8_NWHC_AVG_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NWHC_AVG_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nwhc_s8_mobnet(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NWHC_MOBNET_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NWHC_MOBNET_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NWHC_MOBNET_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NWHC_MOBNET_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nwhc_mobnet_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NWHC_MOBNET_PERM_SIZE] = PERF_TRANSPOSE_INT8_NWHC_MOBNET_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NWHC_MOBNET_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nwhc_s8_large(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_NWHC_LARGE_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_NWHC_LARGE_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_NWHC_LARGE_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_NWHC_LARGE_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_nwhc_large_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_NWHC_LARGE_PERM_SIZE] = PERF_TRANSPOSE_INT8_NWHC_LARGE_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_NWHC_LARGE_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_nwhc_s8(void)
{
    perf_arm_transpose_nwhc_s8_tiny_edge1();
    printf(",\n");
    perf_arm_transpose_nwhc_s8_tiny_edge2();
    printf(",\n");
    perf_arm_transpose_nwhc_s8_tiny();

    // needs debugging
    // printf(",\n");
    // perf_arm_transpose_nwhc_s8_avg();
    // printf(",\n");
    // perf_arm_transpose_nwhc_s8_mobnet();
    // printf(",\n");
    // perf_arm_transpose_nwhc_s8_large();
    
    printf(",\n");
}

void perf_arm_transpose_3dim_s8_tiny_edge1(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE1_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE1_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE1_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE1_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_3dim_tiny_edge1_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE1_PERM_SIZE] = PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE1_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE1_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_3dim_s8_tiny_edge2(void)
{
    int8_t output_data[PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE2_IN_DIM;
    const cmsis_nn_dims output_dims = PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE2_OUT_DIM;
    const int32_t perf_input_tensor_dims[] = PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE2_IN_DIM;

    const int8_t *input_data = perf_transpose_int8_3dim_tiny_edge2_input_tensor;

    const uint32_t perm[PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE2_PERM_SIZE] = PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE2_PERM;
    const cmsis_nn_transpose_params transpose_params = {PERF_TRANSPOSE_INT8_3DIM_TINY_EDGE2_PERM_SIZE, perm};

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

    PERF_DWT_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    PERF_PMU_CHARACTERIZE(arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_transpose_3dim_s8(void)
{
    perf_arm_transpose_3dim_s8_tiny_edge1();
    printf(",\n");
    perf_arm_transpose_3dim_s8_tiny_edge2();
    printf(",\n");
}