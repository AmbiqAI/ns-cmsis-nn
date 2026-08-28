/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <math.h>
#include <string.h>
#include <unity.h>

// Regression for the NHWC broadcast walk shared by arm_maximum_f16 / arm_minimum_f16: both operands
// have width 1, so each row of input 1 is a single element that must advance with the output row.
// The previous walk left input 1 on row 0 and returned SUCCESS with rows 1..h-1 wrong.
static const float16_t minmax_f16_row_scalar_input_1[3] = {(float16_t)10.0f, (float16_t)20.0f, (float16_t)30.0f};
static const float16_t minmax_f16_row_scalar_input_2[12] = {(float16_t)1.0f,
                                                            (float16_t)2.0f,
                                                            (float16_t)3.0f,
                                                            (float16_t)4.0f,
                                                            (float16_t)11.0f,
                                                            (float16_t)12.0f,
                                                            (float16_t)13.0f,
                                                            (float16_t)14.0f,
                                                            (float16_t)21.0f,
                                                            (float16_t)22.0f,
                                                            (float16_t)23.0f,
                                                            (float16_t)24.0f};
static const cmsis_nn_dims minmax_f16_row_scalar_dims_1 = {1, 3, 1, 1};
static const cmsis_nn_dims minmax_f16_row_scalar_dims_2 = {1, 3, 1, 4};
static const cmsis_nn_dims minmax_f16_row_scalar_dims_out = {1, 3, 1, 4};

static void minmax_f16_check_row_scalar(int32_t select_max)
{
    const cmsis_nn_context ctx = {NULL, 0};
    const float16_t *in_1 = minmax_f16_row_scalar_input_1;
    const float16_t *in_2 = minmax_f16_row_scalar_input_2;
    float16_t output[12];

    for (int32_t order = 0; order < 2; order++)
    {
        memset(output, 0, sizeof(output));
        arm_cmsis_nn_status status;
        if (order == 0)
        {
            status = select_max ? arm_maximum_f16(&ctx,
                                                  in_1,
                                                  &minmax_f16_row_scalar_dims_1,
                                                  in_2,
                                                  &minmax_f16_row_scalar_dims_2,
                                                  output,
                                                  &minmax_f16_row_scalar_dims_out)
                                : arm_minimum_f16(&ctx,
                                                  in_1,
                                                  &minmax_f16_row_scalar_dims_1,
                                                  in_2,
                                                  &minmax_f16_row_scalar_dims_2,
                                                  output,
                                                  &minmax_f16_row_scalar_dims_out);
        }
        else
        {
            status = select_max ? arm_maximum_f16(&ctx,
                                                  in_2,
                                                  &minmax_f16_row_scalar_dims_2,
                                                  in_1,
                                                  &minmax_f16_row_scalar_dims_1,
                                                  output,
                                                  &minmax_f16_row_scalar_dims_out)
                                : arm_minimum_f16(&ctx,
                                                  in_2,
                                                  &minmax_f16_row_scalar_dims_2,
                                                  in_1,
                                                  &minmax_f16_row_scalar_dims_1,
                                                  output,
                                                  &minmax_f16_row_scalar_dims_out);
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);

        for (int32_t h = 0; h < 3; h++)
        {
            for (int32_t c = 0; c < 4; c++)
            {
                const float32_t a = (float32_t)in_1[h];
                const float32_t b = (float32_t)in_2[h * 4 + c];
                // fmaxf/fminf, never a ternary -- GCC PR target/118460 (issue #344), the same bug
                // the kernel dodges in arm_minmax_common_f16.c. A float32 select whose arms are round-tripped
                // halves is narrowed back to HFmode, and arm-none-eabi-gcc 14.x then dies with
                // "error: unrecognizable insn: (set (reg:HF ...) (if_then_else:HF ...))" and
                // "internal compiler error: in extract_insn, at recog.cc:2812" during RTL pass
                // vregs. It reproduces at -O3 and at the -Ofast -fno-finite-math-only this harness
                // compiles with, i.e. on the cortex-m55 legacy-tester leg. Every operand below is
                // an exact integer and none is a zero of either sign, so fmaxf/fminf and the
                // ternary agree bit for bit. Do not put the ternary back.
                const float32_t expected = select_max ? fmaxf(a, b) : fminf(a, b);
                TEST_ASSERT_EQUAL_FLOAT(expected, (float32_t)output[h * 4 + c]);
            }
        }
    }
}

void maximum_broadcast_row_scalar_f16(void) { minmax_f16_check_row_scalar(1); }

void minimum_broadcast_row_scalar_f16(void) { minmax_f16_check_row_scalar(0); }

// General four-dimensional broadcast, (2,2,3,1) against (1,2,1,4), against an index-modulo reference.
void maximum_broadcast_general_f16(void)
{
    const cmsis_nn_dims dims_1 = {2, 2, 3, 1};
    const cmsis_nn_dims dims_2 = {1, 2, 1, 4};
    const cmsis_nn_dims dims_out = {2, 2, 3, 4};
    float16_t input_1[12];
    float16_t input_2[8];
    float16_t output[48];
    const cmsis_nn_context ctx = {NULL, 0};

    for (int32_t i = 0; i < 12; i++)
    {
        input_1[i] = (float16_t)(float32_t)((i * 7) % 11 - 5);
    }
    for (int32_t i = 0; i < 8; i++)
    {
        input_2[i] = (float16_t)(float32_t)((i * 5) % 13 - 6);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_maximum_f16(&ctx, input_1, &dims_1, input_2, &dims_2, output, &dims_out));

    for (int32_t n = 0; n < 2; n++)
    {
        for (int32_t h = 0; h < 2; h++)
        {
            for (int32_t w = 0; w < 3; w++)
            {
                for (int32_t c = 0; c < 4; c++)
                {
                    const float32_t a = (float32_t)input_1[(n * 2 + h) * 3 + w];
                    const float32_t b = (float32_t)input_2[h * 4 + c];
                    // fmaxf, not a ternary, for the reason spelled out in
                    // minmax_f16_check_row_scalar above. input_1 holds a +0.0 but input_2 never
                    // does, so no comparison here is between two zeros and the two forms still
                    // agree bit for bit.
                    const float32_t expected = fmaxf(a, b);
                    TEST_ASSERT_EQUAL_FLOAT(expected, (float32_t)output[((n * 2 + h) * 3 + w) * 4 + c]);
                }
            }
        }
    }
}

void minmax_arg_error_f16(void)
{
    float16_t data[12] = {0};
    float16_t output[12] = {0};
    const cmsis_nn_context ctx = {NULL, 0};
    const cmsis_nn_dims dims_3x1 = {1, 3, 1, 1};
    const cmsis_nn_dims dims_2x4 = {1, 2, 1, 4};
    const cmsis_nn_dims dims_3x4 = {1, 3, 1, 4};

    // h = 3 against h = 2 is not broadcastable
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_maximum_f16(&ctx, data, &dims_3x1, data, &dims_2x4, output, &dims_3x4));
    // the output shape must be the broadcast shape
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_minimum_f16(&ctx, data, &dims_3x1, data, &dims_3x4, output, &dims_3x1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_maximum_f16(&ctx, NULL, &dims_3x1, data, &dims_3x4, output, &dims_3x4));
}
