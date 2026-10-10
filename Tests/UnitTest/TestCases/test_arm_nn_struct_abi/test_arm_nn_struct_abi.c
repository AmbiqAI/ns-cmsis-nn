/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

/*
 * This suite is built with -fshort-enums, so against a library built with int-sized enums it is the mixed-convention
 * caller of AmbiqAI/ns-cmsis-nn#693. Public struct fields that name an enum are int32_t (#764), so their size and the
 * value the library reads do not depend on either side's enum convention.
 */

#include <arm_nnfunctions.h>
#include <stddef.h>
#include <string.h>
#include <unity.h>

#define NN_ABI_FIELD_IS_WORD(type, field) _Static_assert(sizeof(((type *)0)->field) == 4, #type "." #field " is not 4 B")

NN_ABI_FIELD_IS_WORD(cmsis_nn_lstm_gate, activation_type);
#if ARM_NN_ENABLE_F32
NN_ABI_FIELD_IS_WORD(cmsis_nn_conv_params_f32, weight_format);
NN_ABI_FIELD_IS_WORD(cmsis_nn_fc_params_f32, weight_format);
NN_ABI_FIELD_IS_WORD(cmsis_nn_bmm_params_f32, rhs_format);
NN_ABI_FIELD_IS_WORD(cmsis_nn_transpose_params_f32, layout);
NN_ABI_FIELD_IS_WORD(cmsis_nn_lstm_gate_f32, activation_type);
#endif
#if ARM_NN_ENABLE_F16
NN_ABI_FIELD_IS_WORD(cmsis_nn_conv_params_f16, weight_format);
NN_ABI_FIELD_IS_WORD(cmsis_nn_fc_params_f16, weight_format);
NN_ABI_FIELD_IS_WORD(cmsis_nn_bmm_params_f16, rhs_format);
NN_ABI_FIELD_IS_WORD(cmsis_nn_transpose_params_f16, layout);
NN_ABI_FIELD_IS_WORD(cmsis_nn_lstm_gate_f16, activation_type);
#endif

/* The params struct starts as 0xFF bytes, so any byte of weight_format this caller does not write is visibly wrong. */
void struct_abi_weight_format_f32(void)
{
#if ARM_NN_ENABLE_F32
    cmsis_nn_conv_params_f32 params;
    memset(&params, 0xFF, sizeof(params));
    params.stride = (cmsis_nn_tile){1, 1};
    params.padding = (cmsis_nn_tile){0, 0};
    params.dilation = (cmsis_nn_tile){1, 1};
    params.activation = (cmsis_nn_activation_f32){-1000.0f, 1000.0f};
    params.weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD;

    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {1, 1, 2, 3};
    const cmsis_nn_dims filter_dims = {2, 1, 1, 3};
    const cmsis_nn_dims bias_dims = {1, 1, 1, 2};
    const cmsis_nn_dims output_dims = {1, 1, 2, 2};
    const float32_t input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f};
    const float32_t filter[6] = {1.0f, 0.0f, 2.0f, -1.0f, 1.0f, 0.5f};
    const float32_t bias[2] = {0.25f, -0.5f};
    float32_t output[4] = {0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_1x1_nhwc_ohwi_f32(
                          &ctx, &params, &input_dims, input, &filter_dims, filter, &bias_dims, bias, &output_dims, output));
    const float32_t expected[4] = {7.25f, 2.0f, 7.25f, 3.0f};
    for (int i = 0; i < 4; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT(expected[i], output[i]);
    }
#else
    TEST_IGNORE_MESSAGE("needs ARM_NN_ENABLE_F32");
#endif
}
