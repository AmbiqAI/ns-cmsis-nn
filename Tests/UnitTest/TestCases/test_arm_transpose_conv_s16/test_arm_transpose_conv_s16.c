/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <stdlib.h>
#include <unity.h>

#include "../TestData/transpose_conv_s16_1/test_data.h"
#include "../TestData/transpose_conv_s16_2/test_data.h"
#include "../TestData/transpose_conv_s16_3/test_data.h"
#include "../TestData/transpose_conv_s16_4/test_data.h"
#include "../TestData/transpose_conv_s16_5/test_data.h"
#include "../TestData/transpose_conv_s16_6/test_data.h"
#include "../TestData/transpose_conv_s16_7/test_data.h"
#include "../Utils/validate.h"

#define OUTPUT_GUARD (0x5A5A)

typedef struct
{
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;
    cmsis_nn_transpose_conv_params params;
    cmsis_nn_per_channel_quant_params quant;
    const int16_t *input;
    const int8_t *weights;
    const int64_t *bias;
    const int16_t *output_ref;
    int32_t output_size;
} tconv_s16_case;

/* Fill a case from generated macros. */
#define TCONV_S16_CASE(c, P, p)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        (c).input_dims = (cmsis_nn_dims){P##_INPUT_BATCHES, P##_INPUT_H, P##_INPUT_W, P##_IN_CH};                      \
        (c).filter_dims = (cmsis_nn_dims){P##_OUT_CH, P##_FILTER_Y, P##_FILTER_X, P##_IN_CH};                          \
        (c).output_dims = (cmsis_nn_dims){P##_INPUT_BATCHES, P##_OUTPUT_H, P##_OUTPUT_W, P##_OUT_CH};                  \
        (c).params.input_offset = P##_INPUT_OFFSET;                                                                    \
        (c).params.output_offset = P##_OUTPUT_OFFSET;                                                                  \
        (c).params.stride = (cmsis_nn_tile){P##_STRIDE_X, P##_STRIDE_Y};                                               \
        (c).params.padding = (cmsis_nn_tile){P##_PAD_X, P##_PAD_Y};                                                    \
        (c).params.padding_offsets = (cmsis_nn_tile){P##_PAD_X_WITH_OFFSET, P##_PAD_Y_WITH_OFFSET};                    \
        (c).params.dilation = (cmsis_nn_tile){P##_DILATION_X, P##_DILATION_Y};                                         \
        (c).params.activation = (cmsis_nn_activation){P##_OUT_ACTIVATION_MIN, P##_OUT_ACTIVATION_MAX};                 \
        (c).quant.multiplier = (int32_t *)p##_output_mult;                                                             \
        (c).quant.shift = (int32_t *)p##_output_shift;                                                                 \
        (c).input = p##_input;                                                                                         \
        (c).weights = p##_weights;                                                                                     \
        (c).bias = p##_biases;                                                                                         \
        (c).output_ref = p##_output_ref;                                                                               \
        (c).output_size = P##_DST_SIZE;                                                                                \
    } while (0)

/* Run one case and check every output. */
static void run_tconv_s16_case(const tconv_s16_case *c)
{
    cmsis_nn_dims bias_dims = {1, 1, 1, c->output_dims.c};
    const int32_t buf_size =
        arm_transpose_conv_s16_get_buffer_size(&c->params, &c->input_dims, &c->filter_dims, &c->output_dims);
    TEST_ASSERT_EQUAL(0, buf_size);

    cmsis_nn_context ctx = {NULL, buf_size};
    cmsis_nn_context output_ctx = {NULL, 0};

    int16_t *output = malloc((c->output_size + 1) * sizeof(int16_t));
    TEST_ASSERT_NOT_NULL(output);
    output[c->output_size] = OUTPUT_GUARD;

    const arm_cmsis_nn_status result = arm_transpose_conv_s16(&ctx,
                                                              &output_ctx,
                                                              &c->params,
                                                              &c->quant,
                                                              &c->input_dims,
                                                              c->input,
                                                              &c->filter_dims,
                                                              c->weights,
                                                              &bias_dims,
                                                              c->bias,
                                                              &c->output_dims,
                                                              output);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, result);
    TEST_ASSERT_TRUE(validate_s16(output, c->output_ref, c->output_size));
    TEST_ASSERT_EQUAL_HEX16(OUTPUT_GUARD, (uint16_t)output[c->output_size]);
    free(output);
}

void transpose_conv_s16_1_arm_transpose_conv_s16(void)
{
    tconv_s16_case c;
    TCONV_S16_CASE(c, TRANSPOSE_CONV_S16_1, transpose_conv_s16_1);
    run_tconv_s16_case(&c);
}

void transpose_conv_s16_2_arm_transpose_conv_s16(void)
{
    tconv_s16_case c;
    TCONV_S16_CASE(c, TRANSPOSE_CONV_S16_2, transpose_conv_s16_2);
    run_tconv_s16_case(&c);
}

void transpose_conv_s16_3_arm_transpose_conv_s16(void)
{
    tconv_s16_case c;
    TCONV_S16_CASE(c, TRANSPOSE_CONV_S16_3, transpose_conv_s16_3);
    run_tconv_s16_case(&c);
}

void transpose_conv_s16_4_arm_transpose_conv_s16(void)
{
    tconv_s16_case c;
    TCONV_S16_CASE(c, TRANSPOSE_CONV_S16_4, transpose_conv_s16_4);
    run_tconv_s16_case(&c);
}

void transpose_conv_s16_5_arm_transpose_conv_s16(void)
{
    tconv_s16_case c;
    TCONV_S16_CASE(c, TRANSPOSE_CONV_S16_5, transpose_conv_s16_5);
    run_tconv_s16_case(&c);
}

void transpose_conv_s16_6_arm_transpose_conv_s16(void)
{
    tconv_s16_case c;
    TCONV_S16_CASE(c, TRANSPOSE_CONV_S16_6, transpose_conv_s16_6);
    run_tconv_s16_case(&c);
}

void transpose_conv_s16_7_arm_transpose_conv_s16(void)
{
    tconv_s16_case c;
    TCONV_S16_CASE(c, TRANSPOSE_CONV_S16_7, transpose_conv_s16_7);
    run_tconv_s16_case(&c);
}

/* Run a case expecting an argument error. */
static void expect_arg_error(const tconv_s16_case *c)
{
    cmsis_nn_dims bias_dims = {1, 1, 1, c->output_dims.c};
    cmsis_nn_context ctx = {NULL, 0};
    int16_t output[TRANSPOSE_CONV_S16_1_DST_SIZE];

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_transpose_conv_s16(&ctx,
                                             &ctx,
                                             &c->params,
                                             &c->quant,
                                             &c->input_dims,
                                             c->input,
                                             &c->filter_dims,
                                             c->weights,
                                             &bias_dims,
                                             c->bias,
                                             &c->output_dims,
                                             output));
}

void transpose_conv_s16_invalid_params_arm_transpose_conv_s16(void)
{
    tconv_s16_case c;
    TCONV_S16_CASE(c, TRANSPOSE_CONV_S16_1, transpose_conv_s16_1);

    c.params.dilation.w = 2;
    expect_arg_error(&c);

    c.params.dilation.w = 1;
    c.params.stride.h = 0;
    expect_arg_error(&c);
    TEST_ASSERT_EQUAL(-1,
                      arm_transpose_conv_s16_get_buffer_size(&c.params, &c.input_dims, &c.filter_dims, &c.output_dims));
    TEST_ASSERT_EQUAL(
        -1, arm_transpose_conv_s16_get_buffer_size_mve(&c.params, &c.input_dims, &c.filter_dims, &c.output_dims));
}
