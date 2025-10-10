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

#include "../../TestData/perf_mul_scalar_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_mul_scalar_int8_tiny/test_data.h"
#include "../../TestData/perf_mul_scalar_int8_avg/test_data.h"
#include "../../TestData/perf_mul_scalar_int8_mobnet/test_data.h"
#include "../../TestData/perf_mul_scalar_int8_large/test_data.h"
#include "../../TestData/perf_mul_broadcast_c_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_mul_broadcast_c_int8_tiny/test_data.h"
#include "../../TestData/perf_mul_broadcast_c_int8_avg/test_data.h"
#include "../../TestData/perf_mul_broadcast_c_int8_mobnet/test_data.h"
#include "../../TestData/perf_mul_broadcast_c_int8_large/test_data.h"
#include "../../TestData/perf_mul_broadcast_h_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_mul_broadcast_h_int8_tiny/test_data.h"
#include "../../TestData/perf_mul_broadcast_h_int8_avg/test_data.h"
#include "../../TestData/perf_mul_broadcast_h_int8_mobnet/test_data.h"
#include "../../TestData/perf_mul_broadcast_h_int8_large/test_data.h"
#include "../../TestData/perf_mul_broadcast_hc_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_mul_broadcast_hc_int8_tiny/test_data.h"
#include "../../TestData/perf_mul_broadcast_hc_int8_avg/test_data.h"
#include "../../TestData/perf_mul_broadcast_hc_int8_mobnet/test_data.h"
#include "../../TestData/perf_mul_broadcast_hc_int8_large/test_data.h"
#include "../../TestData/perf_mul_broadcast_w_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_mul_broadcast_w_int8_tiny/test_data.h"
#include "../../TestData/perf_mul_broadcast_w_int8_avg/test_data.h"
#include "../../TestData/perf_mul_broadcast_w_int8_mobnet/test_data.h"
#include "../../TestData/perf_mul_broadcast_w_int8_large/test_data.h"
#include "../../TestData/perf_mul_ident_int8_tiny_edge/test_data.h"
#include "../../TestData/perf_mul_ident_int8_tiny/test_data.h"
#include "../../TestData/perf_mul_ident_int8_avg/test_data.h"
#include "../../TestData/perf_mul_ident_int8_mobnet/test_data.h"
#include "../../TestData/perf_mul_ident_int8_large/test_data.h"

void perf_arm_mul_scalar_s8_tiny_edge(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_N;
    lhs_dims.h = PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_H;
    lhs_dims.w = PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_W;
    lhs_dims.c = PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_C;

    rhs_dims.n = PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_N;
    rhs_dims.h = PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_H;
    rhs_dims.w = PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_W;
    rhs_dims.c = PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_C;

    out_dims.n = PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_N;
    out_dims.h = PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_H;
    out_dims.w = PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_W;
    out_dims.c = PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_scalar_int8_tiny_edge_lhs_input_tensor;
    const int8_t *rhs = perf_mul_scalar_int8_tiny_edge_rhs_input_tensor;
    int8_t output[PERF_MUL_SCALAR_INT8_TINY_EDGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_N, PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_H,
                                              PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_W, PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_N, PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_H,
                                              PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_W, PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_scalar_s8_tiny(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_SCALAR_INT8_TINY_LHS_N;
    lhs_dims.h = PERF_MUL_SCALAR_INT8_TINY_LHS_H;
    lhs_dims.w = PERF_MUL_SCALAR_INT8_TINY_LHS_W;
    lhs_dims.c = PERF_MUL_SCALAR_INT8_TINY_LHS_C;

    rhs_dims.n = PERF_MUL_SCALAR_INT8_TINY_RHS_N;
    rhs_dims.h = PERF_MUL_SCALAR_INT8_TINY_RHS_H;
    rhs_dims.w = PERF_MUL_SCALAR_INT8_TINY_RHS_W;
    rhs_dims.c = PERF_MUL_SCALAR_INT8_TINY_RHS_C;

    out_dims.n = PERF_MUL_SCALAR_INT8_TINY_OUTPUT_N;
    out_dims.h = PERF_MUL_SCALAR_INT8_TINY_OUTPUT_H;
    out_dims.w = PERF_MUL_SCALAR_INT8_TINY_OUTPUT_W;
    out_dims.c = PERF_MUL_SCALAR_INT8_TINY_OUTPUT_C;

    const int8_t *lhs = perf_mul_scalar_int8_tiny_lhs_input_tensor;
    const int8_t *rhs = perf_mul_scalar_int8_tiny_rhs_input_tensor;
    int8_t output[PERF_MUL_SCALAR_INT8_TINY_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_TINY_LHS_N, PERF_MUL_SCALAR_INT8_TINY_LHS_H,
                                              PERF_MUL_SCALAR_INT8_TINY_LHS_W, PERF_MUL_SCALAR_INT8_TINY_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_TINY_RHS_N, PERF_MUL_SCALAR_INT8_TINY_RHS_H,
                                              PERF_MUL_SCALAR_INT8_TINY_RHS_W, PERF_MUL_SCALAR_INT8_TINY_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_TINY_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_TINY_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_TINY_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_TINY_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_TINY_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_TINY_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_scalar_s8_avg(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_SCALAR_INT8_AVG_LHS_N;
    lhs_dims.h = PERF_MUL_SCALAR_INT8_AVG_LHS_H;
    lhs_dims.w = PERF_MUL_SCALAR_INT8_AVG_LHS_W;
    lhs_dims.c = PERF_MUL_SCALAR_INT8_AVG_LHS_C;

    rhs_dims.n = PERF_MUL_SCALAR_INT8_AVG_RHS_N;
    rhs_dims.h = PERF_MUL_SCALAR_INT8_AVG_RHS_H;
    rhs_dims.w = PERF_MUL_SCALAR_INT8_AVG_RHS_W;
    rhs_dims.c = PERF_MUL_SCALAR_INT8_AVG_RHS_C;

    out_dims.n = PERF_MUL_SCALAR_INT8_AVG_OUTPUT_N;
    out_dims.h = PERF_MUL_SCALAR_INT8_AVG_OUTPUT_H;
    out_dims.w = PERF_MUL_SCALAR_INT8_AVG_OUTPUT_W;
    out_dims.c = PERF_MUL_SCALAR_INT8_AVG_OUTPUT_C;

    const int8_t *lhs = perf_mul_scalar_int8_avg_lhs_input_tensor;
    const int8_t *rhs = perf_mul_scalar_int8_avg_rhs_input_tensor;
    int8_t output[PERF_MUL_SCALAR_INT8_AVG_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_AVG_LHS_N, PERF_MUL_SCALAR_INT8_AVG_LHS_H,
                                              PERF_MUL_SCALAR_INT8_AVG_LHS_W, PERF_MUL_SCALAR_INT8_AVG_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_AVG_RHS_N, PERF_MUL_SCALAR_INT8_AVG_RHS_H,
                                              PERF_MUL_SCALAR_INT8_AVG_RHS_W, PERF_MUL_SCALAR_INT8_AVG_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_AVG_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_AVG_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_AVG_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_AVG_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_AVG_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_AVG_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_scalar_s8_mobnet(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_SCALAR_INT8_MOBNET_LHS_N;
    lhs_dims.h = PERF_MUL_SCALAR_INT8_MOBNET_LHS_H;
    lhs_dims.w = PERF_MUL_SCALAR_INT8_MOBNET_LHS_W;
    lhs_dims.c = PERF_MUL_SCALAR_INT8_MOBNET_LHS_C;

    rhs_dims.n = PERF_MUL_SCALAR_INT8_MOBNET_RHS_N;
    rhs_dims.h = PERF_MUL_SCALAR_INT8_MOBNET_RHS_H;
    rhs_dims.w = PERF_MUL_SCALAR_INT8_MOBNET_RHS_W;
    rhs_dims.c = PERF_MUL_SCALAR_INT8_MOBNET_RHS_C;

    out_dims.n = PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_N;
    out_dims.h = PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_H;
    out_dims.w = PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_W;
    out_dims.c = PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_C;

    const int8_t *lhs = perf_mul_scalar_int8_mobnet_lhs_input_tensor;
    const int8_t *rhs = perf_mul_scalar_int8_mobnet_rhs_input_tensor;
    int8_t output[PERF_MUL_SCALAR_INT8_MOBNET_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_MOBNET_LHS_N, PERF_MUL_SCALAR_INT8_MOBNET_LHS_H,
                                              PERF_MUL_SCALAR_INT8_MOBNET_LHS_W, PERF_MUL_SCALAR_INT8_MOBNET_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_MOBNET_RHS_N, PERF_MUL_SCALAR_INT8_MOBNET_RHS_H,
                                              PERF_MUL_SCALAR_INT8_MOBNET_RHS_W, PERF_MUL_SCALAR_INT8_MOBNET_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_MOBNET_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_MOBNET_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_MOBNET_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_scalar_s8_large(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_SCALAR_INT8_LARGE_LHS_N;
    lhs_dims.h = PERF_MUL_SCALAR_INT8_LARGE_LHS_H;
    lhs_dims.w = PERF_MUL_SCALAR_INT8_LARGE_LHS_W;
    lhs_dims.c = PERF_MUL_SCALAR_INT8_LARGE_LHS_C;

    rhs_dims.n = PERF_MUL_SCALAR_INT8_LARGE_RHS_N;
    rhs_dims.h = PERF_MUL_SCALAR_INT8_LARGE_RHS_H;
    rhs_dims.w = PERF_MUL_SCALAR_INT8_LARGE_RHS_W;
    rhs_dims.c = PERF_MUL_SCALAR_INT8_LARGE_RHS_C;

    out_dims.n = PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_N;
    out_dims.h = PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_H;
    out_dims.w = PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_W;
    out_dims.c = PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_scalar_int8_large_lhs_input_tensor;
    const int8_t *rhs = perf_mul_scalar_int8_large_rhs_input_tensor;
    int8_t output[PERF_MUL_SCALAR_INT8_LARGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_LARGE_LHS_N, PERF_MUL_SCALAR_INT8_LARGE_LHS_H,
                                              PERF_MUL_SCALAR_INT8_LARGE_LHS_W, PERF_MUL_SCALAR_INT8_LARGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_SCALAR_INT8_LARGE_RHS_N, PERF_MUL_SCALAR_INT8_LARGE_RHS_H,
                                              PERF_MUL_SCALAR_INT8_LARGE_RHS_W, PERF_MUL_SCALAR_INT8_LARGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_LARGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_LARGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_SCALAR_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_SCALAR_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_SCALAR_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_SCALAR_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_SCALAR_INT8_LARGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_scalar_s8(void)
{
    perf_arm_mul_scalar_s8_tiny_edge();
    printf(",\n");
    perf_arm_mul_scalar_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_mul_scalar_s8_avg();
    // printf(",\n");
    // perf_arm_mul_scalar_s8_mobnet();
    // printf(",\n");
    // perf_arm_mul_scalar_s8_large();
    // printf(",\n");
}

void perf_arm_mul_broadcast_c_s8_tiny_edge(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_c_int8_tiny_edge_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_c_int8_tiny_edge_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_N, PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_W, PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_N, PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_W, PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_c_s8_tiny(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_C_INT8_TINY_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_C_INT8_TINY_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_C_INT8_TINY_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_C_INT8_TINY_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_C_INT8_TINY_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_C_INT8_TINY_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_C_INT8_TINY_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_C_INT8_TINY_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_c_int8_tiny_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_c_int8_tiny_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_C_INT8_TINY_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_TINY_LHS_N, PERF_MUL_BROADCAST_C_INT8_TINY_LHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_TINY_LHS_W, PERF_MUL_BROADCAST_C_INT8_TINY_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_TINY_RHS_N, PERF_MUL_BROADCAST_C_INT8_TINY_RHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_TINY_RHS_W, PERF_MUL_BROADCAST_C_INT8_TINY_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_TINY_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_TINY_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_TINY_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_c_s8_avg(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_C_INT8_AVG_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_C_INT8_AVG_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_C_INT8_AVG_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_C_INT8_AVG_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_C_INT8_AVG_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_C_INT8_AVG_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_C_INT8_AVG_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_C_INT8_AVG_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_c_int8_avg_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_c_int8_avg_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_C_INT8_AVG_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_AVG_LHS_N, PERF_MUL_BROADCAST_C_INT8_AVG_LHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_AVG_LHS_W, PERF_MUL_BROADCAST_C_INT8_AVG_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_AVG_RHS_N, PERF_MUL_BROADCAST_C_INT8_AVG_RHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_AVG_RHS_W, PERF_MUL_BROADCAST_C_INT8_AVG_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_AVG_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_AVG_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_AVG_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_c_s8_mobnet(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_c_int8_mobnet_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_c_int8_mobnet_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_C_INT8_MOBNET_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_N, PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_W, PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_N, PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_W, PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_MOBNET_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_c_s8_large(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_c_int8_large_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_c_int8_large_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_C_INT8_LARGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_N, PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_W, PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_N, PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_H,
                                              PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_W, PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_LARGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_LARGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_C_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_C_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_C_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_C_INT8_LARGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_c_s8(void)
{
    perf_arm_mul_broadcast_c_s8_tiny_edge();
    printf(",\n");
    perf_arm_mul_broadcast_c_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_mul_broadcast_c_s8_avg();
    // printf(",\n");
    // perf_arm_mul_broadcast_c_s8_mobnet();
    // printf(",\n");
    // perf_arm_mul_broadcast_c_s8_large();
    // printf(",\n");
}

void perf_arm_mul_broadcast_h_s8_tiny_edge(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_h_int8_tiny_edge_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_h_int8_tiny_edge_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_N, PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_W, PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_N, PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_W, PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_h_s8_tiny(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_H_INT8_TINY_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_H_INT8_TINY_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_H_INT8_TINY_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_H_INT8_TINY_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_H_INT8_TINY_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_H_INT8_TINY_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_H_INT8_TINY_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_H_INT8_TINY_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_h_int8_tiny_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_h_int8_tiny_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_H_INT8_TINY_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_TINY_LHS_N, PERF_MUL_BROADCAST_H_INT8_TINY_LHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_TINY_LHS_W, PERF_MUL_BROADCAST_H_INT8_TINY_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_TINY_RHS_N, PERF_MUL_BROADCAST_H_INT8_TINY_RHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_TINY_RHS_W, PERF_MUL_BROADCAST_H_INT8_TINY_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_TINY_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_TINY_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_TINY_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_h_s8_avg(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_H_INT8_AVG_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_H_INT8_AVG_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_H_INT8_AVG_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_H_INT8_AVG_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_H_INT8_AVG_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_H_INT8_AVG_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_H_INT8_AVG_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_H_INT8_AVG_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_h_int8_avg_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_h_int8_avg_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_H_INT8_AVG_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_AVG_LHS_N, PERF_MUL_BROADCAST_H_INT8_AVG_LHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_AVG_LHS_W, PERF_MUL_BROADCAST_H_INT8_AVG_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_AVG_RHS_N, PERF_MUL_BROADCAST_H_INT8_AVG_RHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_AVG_RHS_W, PERF_MUL_BROADCAST_H_INT8_AVG_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_AVG_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_AVG_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_AVG_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_h_s8_mobnet(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_h_int8_mobnet_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_h_int8_mobnet_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_H_INT8_MOBNET_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_N, PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_W, PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_N, PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_W, PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_MOBNET_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_h_s8_large(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_h_int8_large_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_h_int8_large_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_H_INT8_LARGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_N, PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_W, PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_N, PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_H,
                                              PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_W, PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_LARGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_LARGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_H_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_H_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_H_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_H_INT8_LARGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_h_s8(void)
{
    perf_arm_mul_broadcast_h_s8_tiny_edge();
    printf(",\n");
    perf_arm_mul_broadcast_h_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_mul_broadcast_h_s8_avg();
    // printf(",\n");
    // perf_arm_mul_broadcast_h_s8_mobnet();
    // printf(",\n");
    // perf_arm_mul_broadcast_h_s8_large();
    // printf(",\n");
}

void perf_arm_mul_broadcast_hc_s8_tiny_edge(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_hc_int8_tiny_edge_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_hc_int8_tiny_edge_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_N, PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_W, PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_N, PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_W, PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_hc_s8_tiny(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_hc_int8_tiny_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_hc_int8_tiny_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_HC_INT8_TINY_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_N, PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_W, PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_N, PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_W, PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_TINY_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_TINY_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_TINY_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_hc_s8_avg(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_hc_int8_avg_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_hc_int8_avg_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_HC_INT8_AVG_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_N, PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_W, PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_N, PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_W, PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_AVG_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_AVG_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_AVG_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_hc_s8_mobnet(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_hc_int8_mobnet_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_hc_int8_mobnet_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_HC_INT8_MOBNET_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_N, PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_W, PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_N, PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_W, PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_MOBNET_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_hc_s8_large(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_hc_int8_large_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_hc_int8_large_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_HC_INT8_LARGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_N, PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_W, PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_N, PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_H,
                                              PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_W, PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_HC_INT8_LARGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_hc_s8(void)
{
    perf_arm_mul_broadcast_hc_s8_tiny_edge();
    printf(",\n");
    perf_arm_mul_broadcast_hc_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_mul_broadcast_hc_s8_avg();
    // printf(",\n");
    // perf_arm_mul_broadcast_hc_s8_mobnet();
    // printf(",\n");
    // perf_arm_mul_broadcast_hc_s8_large();
    // printf(",\n");
}

void perf_arm_mul_broadcast_w_s8_tiny_edge(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_w_int8_tiny_edge_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_w_int8_tiny_edge_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_N, PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_W, PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_N, PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_W, PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_w_s8_tiny(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_W_INT8_TINY_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_W_INT8_TINY_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_W_INT8_TINY_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_W_INT8_TINY_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_W_INT8_TINY_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_W_INT8_TINY_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_W_INT8_TINY_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_W_INT8_TINY_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_w_int8_tiny_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_w_int8_tiny_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_W_INT8_TINY_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_TINY_LHS_N, PERF_MUL_BROADCAST_W_INT8_TINY_LHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_TINY_LHS_W, PERF_MUL_BROADCAST_W_INT8_TINY_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_TINY_RHS_N, PERF_MUL_BROADCAST_W_INT8_TINY_RHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_TINY_RHS_W, PERF_MUL_BROADCAST_W_INT8_TINY_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_TINY_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_TINY_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_TINY_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_w_s8_avg(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_W_INT8_AVG_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_W_INT8_AVG_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_W_INT8_AVG_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_W_INT8_AVG_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_W_INT8_AVG_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_W_INT8_AVG_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_W_INT8_AVG_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_W_INT8_AVG_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_w_int8_avg_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_w_int8_avg_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_W_INT8_AVG_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_AVG_LHS_N, PERF_MUL_BROADCAST_W_INT8_AVG_LHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_AVG_LHS_W, PERF_MUL_BROADCAST_W_INT8_AVG_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_AVG_RHS_N, PERF_MUL_BROADCAST_W_INT8_AVG_RHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_AVG_RHS_W, PERF_MUL_BROADCAST_W_INT8_AVG_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_AVG_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_AVG_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_AVG_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_AVG_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_w_s8_mobnet(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_w_int8_mobnet_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_w_int8_mobnet_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_W_INT8_MOBNET_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_N, PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_W, PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_N, PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_W, PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_MOBNET_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_w_s8_large(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_N;
    lhs_dims.h = PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_H;
    lhs_dims.w = PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_W;
    lhs_dims.c = PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_C;

    rhs_dims.n = PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_N;
    rhs_dims.h = PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_H;
    rhs_dims.w = PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_W;
    rhs_dims.c = PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_C;

    out_dims.n = PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_N;
    out_dims.h = PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_H;
    out_dims.w = PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_W;
    out_dims.c = PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_broadcast_w_int8_large_lhs_input_tensor;
    const int8_t *rhs = perf_mul_broadcast_w_int8_large_rhs_input_tensor;
    int8_t output[PERF_MUL_BROADCAST_W_INT8_LARGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_N, PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_W, PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_N, PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_H,
                                              PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_W, PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_LARGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_LARGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_BROADCAST_W_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_BROADCAST_W_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_BROADCAST_W_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_BROADCAST_W_INT8_LARGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_broadcast_w_s8(void)
{
    perf_arm_mul_broadcast_w_s8_tiny_edge();
    printf(",\n");
    perf_arm_mul_broadcast_w_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_mul_broadcast_w_s8_avg();
    // printf(",\n");
    // perf_arm_mul_broadcast_w_s8_mobnet();
    // printf(",\n");
    // perf_arm_mul_broadcast_w_s8_large();
    // printf(",\n");
}

void perf_arm_mul_ident_s8_tiny_edge(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_N;
    lhs_dims.h = PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_H;
    lhs_dims.w = PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_W;
    lhs_dims.c = PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_C;

    rhs_dims.n = PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_N;
    rhs_dims.h = PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_H;
    rhs_dims.w = PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_W;
    rhs_dims.c = PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_C;

    out_dims.n = PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_N;
    out_dims.h = PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_H;
    out_dims.w = PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_W;
    out_dims.c = PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_ident_int8_tiny_edge_lhs_input_tensor;
    const int8_t *rhs = perf_mul_ident_int8_tiny_edge_rhs_input_tensor;
    int8_t output[PERF_MUL_IDENT_INT8_TINY_EDGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_N, PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_H,
                                              PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_W, PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_N, PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_H,
                                              PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_W, PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_TINY_EDGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_TINY_EDGE_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_EDGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_TINY_EDGE_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_TINY_EDGE_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_TINY_EDGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_ident_s8_tiny(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_IDENT_INT8_TINY_LHS_N;
    lhs_dims.h = PERF_MUL_IDENT_INT8_TINY_LHS_H;
    lhs_dims.w = PERF_MUL_IDENT_INT8_TINY_LHS_W;
    lhs_dims.c = PERF_MUL_IDENT_INT8_TINY_LHS_C;

    rhs_dims.n = PERF_MUL_IDENT_INT8_TINY_RHS_N;
    rhs_dims.h = PERF_MUL_IDENT_INT8_TINY_RHS_H;
    rhs_dims.w = PERF_MUL_IDENT_INT8_TINY_RHS_W;
    rhs_dims.c = PERF_MUL_IDENT_INT8_TINY_RHS_C;

    out_dims.n = PERF_MUL_IDENT_INT8_TINY_OUTPUT_N;
    out_dims.h = PERF_MUL_IDENT_INT8_TINY_OUTPUT_H;
    out_dims.w = PERF_MUL_IDENT_INT8_TINY_OUTPUT_W;
    out_dims.c = PERF_MUL_IDENT_INT8_TINY_OUTPUT_C;

    const int8_t *lhs = perf_mul_ident_int8_tiny_lhs_input_tensor;
    const int8_t *rhs = perf_mul_ident_int8_tiny_rhs_input_tensor;
    int8_t output[PERF_MUL_IDENT_INT8_TINY_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_TINY_LHS_N, PERF_MUL_IDENT_INT8_TINY_LHS_H,
                                              PERF_MUL_IDENT_INT8_TINY_LHS_W, PERF_MUL_IDENT_INT8_TINY_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_TINY_RHS_N, PERF_MUL_IDENT_INT8_TINY_RHS_H,
                                              PERF_MUL_IDENT_INT8_TINY_RHS_W, PERF_MUL_IDENT_INT8_TINY_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_TINY_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_TINY_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_TINY_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_TINY_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_TINY_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_TINY_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_TINY_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_TINY_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_ident_s8_avg(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_IDENT_INT8_AVG_LHS_N;
    lhs_dims.h = PERF_MUL_IDENT_INT8_AVG_LHS_H;
    lhs_dims.w = PERF_MUL_IDENT_INT8_AVG_LHS_W;
    lhs_dims.c = PERF_MUL_IDENT_INT8_AVG_LHS_C;

    rhs_dims.n = PERF_MUL_IDENT_INT8_AVG_RHS_N;
    rhs_dims.h = PERF_MUL_IDENT_INT8_AVG_RHS_H;
    rhs_dims.w = PERF_MUL_IDENT_INT8_AVG_RHS_W;
    rhs_dims.c = PERF_MUL_IDENT_INT8_AVG_RHS_C;

    out_dims.n = PERF_MUL_IDENT_INT8_AVG_OUTPUT_N;
    out_dims.h = PERF_MUL_IDENT_INT8_AVG_OUTPUT_H;
    out_dims.w = PERF_MUL_IDENT_INT8_AVG_OUTPUT_W;
    out_dims.c = PERF_MUL_IDENT_INT8_AVG_OUTPUT_C;

    const int8_t *lhs = perf_mul_ident_int8_avg_lhs_input_tensor;
    const int8_t *rhs = perf_mul_ident_int8_avg_rhs_input_tensor;
    int8_t output[PERF_MUL_IDENT_INT8_AVG_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_AVG_LHS_N, PERF_MUL_IDENT_INT8_AVG_LHS_H,
                                              PERF_MUL_IDENT_INT8_AVG_LHS_W, PERF_MUL_IDENT_INT8_AVG_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_AVG_RHS_N, PERF_MUL_IDENT_INT8_AVG_RHS_H,
                                              PERF_MUL_IDENT_INT8_AVG_RHS_W, PERF_MUL_IDENT_INT8_AVG_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_AVG_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_AVG_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_AVG_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_AVG_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_AVG_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_AVG_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_AVG_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_AVG_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_AVG_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_ident_s8_mobnet(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_IDENT_INT8_MOBNET_LHS_N;
    lhs_dims.h = PERF_MUL_IDENT_INT8_MOBNET_LHS_H;
    lhs_dims.w = PERF_MUL_IDENT_INT8_MOBNET_LHS_W;
    lhs_dims.c = PERF_MUL_IDENT_INT8_MOBNET_LHS_C;

    rhs_dims.n = PERF_MUL_IDENT_INT8_MOBNET_RHS_N;
    rhs_dims.h = PERF_MUL_IDENT_INT8_MOBNET_RHS_H;
    rhs_dims.w = PERF_MUL_IDENT_INT8_MOBNET_RHS_W;
    rhs_dims.c = PERF_MUL_IDENT_INT8_MOBNET_RHS_C;

    out_dims.n = PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_N;
    out_dims.h = PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_H;
    out_dims.w = PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_W;
    out_dims.c = PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_C;

    const int8_t *lhs = perf_mul_ident_int8_mobnet_lhs_input_tensor;
    const int8_t *rhs = perf_mul_ident_int8_mobnet_rhs_input_tensor;
    int8_t output[PERF_MUL_IDENT_INT8_MOBNET_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_MOBNET_LHS_N, PERF_MUL_IDENT_INT8_MOBNET_LHS_H,
                                              PERF_MUL_IDENT_INT8_MOBNET_LHS_W, PERF_MUL_IDENT_INT8_MOBNET_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_MOBNET_RHS_N, PERF_MUL_IDENT_INT8_MOBNET_RHS_H,
                                              PERF_MUL_IDENT_INT8_MOBNET_RHS_W, PERF_MUL_IDENT_INT8_MOBNET_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_MOBNET_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_MOBNET_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_MOBNET_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_MOBNET_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_MOBNET_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_MOBNET_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_MOBNET_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_ident_s8_large(void)
{
    cmsis_nn_dims lhs_dims;
    cmsis_nn_dims rhs_dims;
    cmsis_nn_dims out_dims;

    lhs_dims.n = PERF_MUL_IDENT_INT8_LARGE_LHS_N;
    lhs_dims.h = PERF_MUL_IDENT_INT8_LARGE_LHS_H;
    lhs_dims.w = PERF_MUL_IDENT_INT8_LARGE_LHS_W;
    lhs_dims.c = PERF_MUL_IDENT_INT8_LARGE_LHS_C;

    rhs_dims.n = PERF_MUL_IDENT_INT8_LARGE_RHS_N;
    rhs_dims.h = PERF_MUL_IDENT_INT8_LARGE_RHS_H;
    rhs_dims.w = PERF_MUL_IDENT_INT8_LARGE_RHS_W;
    rhs_dims.c = PERF_MUL_IDENT_INT8_LARGE_RHS_C;

    out_dims.n = PERF_MUL_IDENT_INT8_LARGE_OUTPUT_N;
    out_dims.h = PERF_MUL_IDENT_INT8_LARGE_OUTPUT_H;
    out_dims.w = PERF_MUL_IDENT_INT8_LARGE_OUTPUT_W;
    out_dims.c = PERF_MUL_IDENT_INT8_LARGE_OUTPUT_C;

    const int8_t *lhs = perf_mul_ident_int8_large_lhs_input_tensor;
    const int8_t *rhs = perf_mul_ident_int8_large_rhs_input_tensor;
    int8_t output[PERF_MUL_IDENT_INT8_LARGE_DST_SIZE] = {0};

    const int32_t perf_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_LARGE_LHS_N, PERF_MUL_IDENT_INT8_LARGE_LHS_H,
                                              PERF_MUL_IDENT_INT8_LARGE_LHS_W, PERF_MUL_IDENT_INT8_LARGE_LHS_C};

    const int32_t perf_rhs_input_tensor_dims[] = {PERF_MUL_IDENT_INT8_LARGE_RHS_N, PERF_MUL_IDENT_INT8_LARGE_RHS_H,
                                              PERF_MUL_IDENT_INT8_LARGE_RHS_W, PERF_MUL_IDENT_INT8_LARGE_RHS_C};
    
    arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_LARGE_ACTIVATION_MAX
    );

    PERF_DWT_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_LARGE_ACTIVATION_MAX
    ));

    PERF_PMU_CHARACTERIZE(arm_mul_s8(
        lhs,
        &lhs_dims,
        rhs,
        &rhs_dims,
        PERF_MUL_IDENT_INT8_LARGE_LHS_OFFSET,
        PERF_MUL_IDENT_INT8_LARGE_RHS_OFFSET,
        output,
        &out_dims,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_OFFSET,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_MULT,
        PERF_MUL_IDENT_INT8_LARGE_OUTPUT_SHIFT,
        PERF_MUL_IDENT_INT8_LARGE_ACTIVATION_MIN,
        PERF_MUL_IDENT_INT8_LARGE_ACTIVATION_MAX
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    const int32_t rhs_input_dims_len = sizeof(perf_rhs_input_tensor_dims) / sizeof(perf_rhs_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, perf_rhs_input_tensor_dims, rhs_input_dims_len);
}

void perf_arm_mul_ident_s8(void)
{
    perf_arm_mul_ident_s8_tiny_edge();
    printf(",\n");
    perf_arm_mul_ident_s8_tiny();
    printf(",\n");

    // needs debugging
    // perf_arm_mul_ident_s8_avg();
    // printf(",\n");
    // perf_arm_mul_ident_s8_mobnet();
    // printf(",\n");
    // perf_arm_mul_ident_s8_large();
    // printf(",\n");
}