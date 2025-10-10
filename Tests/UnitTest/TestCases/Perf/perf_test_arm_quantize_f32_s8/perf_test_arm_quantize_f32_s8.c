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
 
 #include "../../TestData/perf_quantize_float32_int8_tiny_edge1/test_data.h"
 #include "../../TestData/perf_quantize_float32_int8_tiny_edge2/test_data.h"
 #include "../../TestData/perf_quantize_float32_int8_tiny/test_data.h"
 #include "../../TestData/perf_quantize_float32_int8_avg/test_data.h"
 #include "../../TestData/perf_quantize_float32_int8_mobnet/test_data.h"

void perf_arm_quantize_f32_s8_tiny_edge1(void)
{
    const float   scale      = PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_QUANT_OUTPUT_SCALE_INT8_T;
    const int32_t zero_point = PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_QUANT_OUTPUT_ZERO_POINT_INT8_T;

    const int32_t perf_input_tensor_dims[] = {PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_BATCH_1, PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_WIDTH_1,
                                              PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_HEIGHT_1, PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_CHANNEL_1};

    int8_t output[PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_OUTPUT_LEN];

    arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_edge1_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    );
    PERF_DWT_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_edge1_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    PERF_PMU_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_edge1_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE1_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_quantize_f32_s8_tiny_edge2(void)
{
    const float   scale      = PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_QUANT_OUTPUT_SCALE_INT8_T;
    const int32_t zero_point = PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_QUANT_OUTPUT_ZERO_POINT_INT8_T;

    const int32_t perf_input_tensor_dims[] = {PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_BATCH_1, PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_WIDTH_1,
                                              PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_HEIGHT_1, PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_CHANNEL_1};

    int8_t output[PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_OUTPUT_LEN];

    arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_edge2_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    );
    PERF_DWT_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_edge2_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    PERF_PMU_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_edge2_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_EDGE2_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_quantize_f32_s8_tiny(void)
{
    const float   scale      = PERF_QUANTIZE_FLOAT32_INT8_TINY_QUANT_OUTPUT_SCALE_INT8_T;
    const int32_t zero_point = PERF_QUANTIZE_FLOAT32_INT8_TINY_QUANT_OUTPUT_ZERO_POINT_INT8_T;

    const int32_t perf_input_tensor_dims[] = {PERF_QUANTIZE_FLOAT32_INT8_TINY_BATCH_1, PERF_QUANTIZE_FLOAT32_INT8_TINY_WIDTH_1,
                                              PERF_QUANTIZE_FLOAT32_INT8_TINY_HEIGHT_1, PERF_QUANTIZE_FLOAT32_INT8_TINY_CHANNEL_1};

    int8_t output[PERF_QUANTIZE_FLOAT32_INT8_TINY_OUTPUT_LEN];

    arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    );
    PERF_DWT_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    PERF_PMU_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_tiny_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_TINY_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_quantize_f32_s8_avg(void)
{
    const float   scale      = PERF_QUANTIZE_FLOAT32_INT8_AVG_QUANT_OUTPUT_SCALE_INT8_T;
    const int32_t zero_point = PERF_QUANTIZE_FLOAT32_INT8_AVG_QUANT_OUTPUT_ZERO_POINT_INT8_T;

    const int32_t perf_input_tensor_dims[] = {PERF_QUANTIZE_FLOAT32_INT8_AVG_BATCH_1, PERF_QUANTIZE_FLOAT32_INT8_AVG_WIDTH_1,
                                              PERF_QUANTIZE_FLOAT32_INT8_AVG_HEIGHT_1, PERF_QUANTIZE_FLOAT32_INT8_AVG_CHANNEL_1};

    int8_t output[PERF_QUANTIZE_FLOAT32_INT8_AVG_OUTPUT_LEN];

    arm_quantize_f32_s8(
        perf_quantize_float32_int8_avg_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_AVG_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    );
    PERF_DWT_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_avg_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_AVG_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    PERF_PMU_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_avg_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_AVG_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_quantize_f32_s8_mobnet(void)
{
    const float   scale      = PERF_QUANTIZE_FLOAT32_INT8_MOBNET_QUANT_OUTPUT_SCALE_INT8_T;
    const int32_t zero_point = PERF_QUANTIZE_FLOAT32_INT8_MOBNET_QUANT_OUTPUT_ZERO_POINT_INT8_T;

    const int32_t perf_input_tensor_dims[] = {PERF_QUANTIZE_FLOAT32_INT8_MOBNET_BATCH_1, PERF_QUANTIZE_FLOAT32_INT8_MOBNET_WIDTH_1,
                                              PERF_QUANTIZE_FLOAT32_INT8_MOBNET_HEIGHT_1, PERF_QUANTIZE_FLOAT32_INT8_MOBNET_CHANNEL_1};

    int8_t output[PERF_QUANTIZE_FLOAT32_INT8_MOBNET_OUTPUT_LEN];

    arm_quantize_f32_s8(
        perf_quantize_float32_int8_mobnet_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_MOBNET_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    );
    PERF_DWT_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_mobnet_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_MOBNET_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    PERF_PMU_CHARACTERIZE(arm_quantize_f32_s8(
        perf_quantize_float32_int8_mobnet_input_tensor_1,  // float[8]
        output,                          // int8_t[8]
        PERF_QUANTIZE_FLOAT32_INT8_MOBNET_OUTPUT_LEN,                               // number of elements
        zero_point,                      // from config_data.h
        scale                            // from config_data.h
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_quantize_f32_s8(void)
{
    perf_arm_quantize_f32_s8_tiny_edge1();
    printf(",\n");
    perf_arm_quantize_f32_s8_tiny_edge2();
    printf(",\n");
    perf_arm_quantize_f32_s8_tiny();
    printf(",\n");
    perf_arm_quantize_f32_s8_avg();
    printf(",\n");
    perf_arm_quantize_f32_s8_mobnet();
    printf(",\n");
}