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
#include <stdlib.h>
#include <string.h>
#include <unity.h>

#if ARM_NN_ENABLE_F16
    #include "../Common/float_packed_test_utils.h"
#endif

#define NN_ABI_FIELD_IS_WORD(type, field)                                                                              \
    _Static_assert(sizeof(((type *)0)->field) == 4, #type "." #field " is not 4 B")

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

/* Each params struct starts as 0xFF bytes, so any byte of a format or layout field this caller does not write is
   visibly wrong. */
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

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
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

#if ARM_NN_ENABLE_F16
/* in[2][3] times w[2][3] (each output channel one row), plus bias */
static const float16_t abi_in_f16[6] = {1, 2, 3, -1, 0, 2};
static const float16_t abi_w_f16[6] = {1, 0, 2, -1, 1, 1};
static const float16_t abi_bias_f16[2] = {1, -2};

static void abi_expect_f16(const float16_t *output, const float16_t *bias)
{
    for (int b = 0; b < 2; ++b)
    {
        for (int j = 0; j < 2; ++j)
        {
            float acc = bias ? (float)bias[j] : 0.0f;
            for (int k = 0; k < 3; ++k)
            {
                acc += (float)abi_in_f16[b * 3 + k] * (float)abi_w_f16[j * 3 + k];
            }
            TEST_ASSERT_EQUAL_FLOAT(acc, (float)output[b * 2 + j]);
        }
    }
}
#endif

/* Packed weights: a library that misreads the format field as the standard format reads them in the wrong order. */
void struct_abi_weight_format_f16(void)
{
#if ARM_NN_ENABLE_F16
    float16_t *packed = pack_rhs_nt_n_from_nt_t_f16(abi_w_f16, 2, 3);
    const cmsis_nn_context ctx = {NULL, 0};
    float16_t output[4] = {0};

    cmsis_nn_fc_params_f16 fc_params;
    memset(&fc_params, 0xFF, sizeof(fc_params));
    fc_params.activation = (cmsis_nn_activation_f16){(float16_t)-1000.0f, (float16_t)1000.0f};
    fc_params.weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED;
    const cmsis_nn_dims input_dims = {2, 1, 1, 3};
    const cmsis_nn_dims filter_dims = {3, 1, 1, 2};
    const cmsis_nn_dims bias_dims = {1, 1, 1, 2};
    const cmsis_nn_dims output_dims = {2, 1, 1, 2};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_fully_connected_f16(&ctx,
                                              &fc_params,
                                              &input_dims,
                                              abi_in_f16,
                                              &filter_dims,
                                              packed,
                                              &bias_dims,
                                              abi_bias_f16,
                                              &output_dims,
                                              output,
                                              ARM_NN_LAYOUT_NHWC));
    abi_expect_f16(output, abi_bias_f16);

    /* adj_x and adj_y are const, so these params are initialised rather than pre-filled */
    const cmsis_nn_bmm_params_f16 bmm_params = {.adj_x = false,
                                                .adj_y = false,
                                                .activation = {(float16_t)-1000.0f, (float16_t)1000.0f},
                                                .rhs_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims lhs_dims = {1, 1, 3, 2};
    const cmsis_nn_dims rhs_dims = {1, 1, 3, 2};
    const cmsis_nn_dims bmm_output_dims = {1, 1, 2, 2};
    memset(output, 0, sizeof(output));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_batch_matmul_f16(&ctx, &bmm_params, &lhs_dims, abi_in_f16, &rhs_dims, packed, &bmm_output_dims, output));
    abi_expect_f16(output, NULL);
    free(packed);
#else
    TEST_IGNORE_MESSAGE("needs ARM_NN_ENABLE_F16");
#endif
}

/* A 2 x 3 plane transposed to 3 x 2. The library accepts only the NHWC layout, compared as the full int32_t: 0x100
   would read as NHWC through an 8-bit enum. */
void struct_abi_layout(void)
{
#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims input_dims = {1, 2, 3, 1};
    const cmsis_nn_dims output_dims = {1, 3, 2, 1};
    const int32_t expected[6] = {0, 3, 1, 4, 2, 5};
#endif
#if ARM_NN_ENABLE_F32
    {
        cmsis_nn_transpose_params_f32 params;
        memset(&params, 0xFF, sizeof(params));
        params.num_dims = 4;
        memcpy(params.perm, (const int32_t[4]){0, 2, 1, 3}, sizeof(params.perm));
        params.layout = ARM_NN_LAYOUT_NHWC;
        const float32_t input[6] = {0, 1, 2, 3, 4, 5};
        float32_t output[6] = {0};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_transpose_f32(&ctx, &params, &input_dims, input, &output_dims, output));
        for (int i = 0; i < 6; ++i)
        {
            TEST_ASSERT_EQUAL_FLOAT((float)expected[i], output[i]);
        }
        params.layout = 0x100;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_transpose_f32(&ctx, &params, &input_dims, input, &output_dims, output));
    }
#endif
#if ARM_NN_ENABLE_F16
    {
        cmsis_nn_transpose_params_f16 params;
        memset(&params, 0xFF, sizeof(params));
        params.num_dims = 4;
        memcpy(params.perm, (const int32_t[4]){0, 2, 1, 3}, sizeof(params.perm));
        params.layout = ARM_NN_LAYOUT_NHWC;
        const float16_t input[6] = {0, 1, 2, 3, 4, 5};
        float16_t output[6] = {0};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          arm_transpose_f16(&ctx, &params, &input_dims, input, &output_dims, output));
        for (int i = 0; i < 6; ++i)
        {
            TEST_ASSERT_EQUAL_FLOAT((float)expected[i], (float)output[i]);
        }
        params.layout = 0x100;
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                          arm_transpose_f16(&ctx, &params, &input_dims, input, &output_dims, output));
    }
#endif
#if !ARM_NN_ENABLE_F32 && !ARM_NN_ENABLE_F16
    TEST_IGNORE_MESSAGE("needs ARM_NN_ENABLE_F32 or ARM_NN_ENABLE_F16");
#endif
}
