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
 
 #include "../../TestData/perf_dequantize_int16_float32_tiny_edge1/test_data.h"
 #include "../../TestData/perf_dequantize_int16_float32_tiny_edge2/test_data.h"
 #include "../../TestData/perf_dequantize_int16_float32_tiny/test_data.h"
 #include "../../TestData/perf_dequantize_int16_float32_avg/test_data.h"
 #include "../../TestData/perf_dequantize_int16_float32_mobnet/test_data.h"
 
void perf_arm_dequantize_s16_f32_tiny_edge1(void)
{
    // from config_data.h
    float   scale      = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE1_QUANT_INPUT_SCALE_FLOAT32_T;      // e.g. 2.725261e-05
    int32_t zero_point = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE1_QUANT_INPUT_ZERO_POINT_FLOAT32_T; // e.g. 0

    const int size = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE1_OUTPUT_LEN;

    const int32_t perf_input_tensor_dims[] = {PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE1_BATCH_1, PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE1_WIDTH_1,
                                              PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE1_HEIGHT_1, PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE1_CHANNEL_1};

    // Prepare a buffer for the function output
    float output_f32[PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE1_OUTPUT_LEN];

    // Call your custom function
    arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_edge1_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    );

    PERF_DWT_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_edge1_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    PERF_PMU_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_edge1_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_dequantize_s16_f32_tiny_edge2(void)
{
    // from config_data.h
    float   scale      = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE2_QUANT_INPUT_SCALE_FLOAT32_T;      // e.g. 2.725261e-05
    int32_t zero_point = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE2_QUANT_INPUT_ZERO_POINT_FLOAT32_T; // e.g. 0

    const int size = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE2_OUTPUT_LEN;

    const int32_t perf_input_tensor_dims[] = {PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE2_BATCH_1, PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE2_WIDTH_1,
                                              PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE2_HEIGHT_1, PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE2_CHANNEL_1};

    // Prepare a buffer for the function output
    float output_f32[PERF_DEQUANTIZE_INT16_FLOAT32_TINY_EDGE2_OUTPUT_LEN];

    // Call your custom function
    arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_edge2_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    );

    PERF_DWT_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_edge2_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    PERF_PMU_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_edge2_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_dequantize_s16_f32_tiny(void)
{
    // from config_data.h
    float   scale      = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_QUANT_INPUT_SCALE_FLOAT32_T;      // e.g. 2.725261e-05
    int32_t zero_point = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_QUANT_INPUT_ZERO_POINT_FLOAT32_T; // e.g. 0

    const int size = PERF_DEQUANTIZE_INT16_FLOAT32_TINY_OUTPUT_LEN;

    const int32_t perf_input_tensor_dims[] = {PERF_DEQUANTIZE_INT16_FLOAT32_TINY_BATCH_1, PERF_DEQUANTIZE_INT16_FLOAT32_TINY_WIDTH_1,
                                              PERF_DEQUANTIZE_INT16_FLOAT32_TINY_HEIGHT_1, PERF_DEQUANTIZE_INT16_FLOAT32_TINY_CHANNEL_1};

    // Prepare a buffer for the function output
    float output_f32[PERF_DEQUANTIZE_INT16_FLOAT32_TINY_OUTPUT_LEN];

    // Call your custom function
    arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    );

    PERF_DWT_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    PERF_PMU_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_tiny_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_dequantize_s16_f32_avg(void)
{
    // from config_data.h
    float   scale      = PERF_DEQUANTIZE_INT16_FLOAT32_AVG_QUANT_INPUT_SCALE_FLOAT32_T;      // e.g. 2.725261e-05
    int32_t zero_point = PERF_DEQUANTIZE_INT16_FLOAT32_AVG_QUANT_INPUT_ZERO_POINT_FLOAT32_T; // e.g. 0

    const int size = PERF_DEQUANTIZE_INT16_FLOAT32_AVG_OUTPUT_LEN;

    const int32_t perf_input_tensor_dims[] = {PERF_DEQUANTIZE_INT16_FLOAT32_AVG_BATCH_1, PERF_DEQUANTIZE_INT16_FLOAT32_AVG_WIDTH_1,
                                              PERF_DEQUANTIZE_INT16_FLOAT32_AVG_HEIGHT_1, PERF_DEQUANTIZE_INT16_FLOAT32_AVG_CHANNEL_1};

    // Prepare a buffer for the function output
    float output_f32[PERF_DEQUANTIZE_INT16_FLOAT32_AVG_OUTPUT_LEN];

    // Call your custom function
    arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_avg_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    );

    PERF_DWT_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_avg_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    PERF_PMU_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_avg_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_dequantize_s16_f32_mobnet(void)
{
    // from config_data.h
    float   scale      = PERF_DEQUANTIZE_INT16_FLOAT32_MOBNET_QUANT_INPUT_SCALE_FLOAT32_T;      // e.g. 2.725261e-05
    int32_t zero_point = PERF_DEQUANTIZE_INT16_FLOAT32_MOBNET_QUANT_INPUT_ZERO_POINT_FLOAT32_T; // e.g. 0

    const int size = PERF_DEQUANTIZE_INT16_FLOAT32_MOBNET_OUTPUT_LEN;

    const int32_t perf_input_tensor_dims[] = {PERF_DEQUANTIZE_INT16_FLOAT32_MOBNET_BATCH_1, PERF_DEQUANTIZE_INT16_FLOAT32_MOBNET_WIDTH_1,
                                              PERF_DEQUANTIZE_INT16_FLOAT32_MOBNET_HEIGHT_1, PERF_DEQUANTIZE_INT16_FLOAT32_MOBNET_CHANNEL_1};

    // Prepare a buffer for the function output
    float output_f32[PERF_DEQUANTIZE_INT16_FLOAT32_MOBNET_OUTPUT_LEN];

    // Call your custom function
    arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_mobnet_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    );

    PERF_DWT_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_mobnet_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    PERF_PMU_CHARACTERIZE(arm_dequantize_s16_f32(
        perf_dequantize_int16_float32_mobnet_input_tensor_1, // the int16[] input
        output_f32,                        // float[] output
        size,
        scale,
        zero_point
    ));

    const int32_t input_dims_len = sizeof(perf_input_tensor_dims) / sizeof(perf_input_tensor_dims[0]);

    perf_print_all(__func__ + 9, perf_input_tensor_dims, input_dims_len, NULL, 0);
}

void perf_arm_dequantize_s16_f32(void)
{
    perf_arm_dequantize_s16_f32_tiny_edge1();
    printf(",\n");
    perf_arm_dequantize_s16_f32_tiny_edge2();
    printf(",\n");
    perf_arm_dequantize_s16_f32_tiny();
    printf(",\n");
    perf_arm_dequantize_s16_f32_avg();
    printf(",\n");
    perf_arm_dequantize_s16_f32_mobnet();
    printf(",\n");
}