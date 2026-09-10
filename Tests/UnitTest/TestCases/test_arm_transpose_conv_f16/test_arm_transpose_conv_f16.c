/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arm_nnfunctions.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#include "transpose_conv_basic_f16_data.h"
#include "transpose_conv_basic_nhwc_f16_data.h"

#define RUN_TRANSPOSE_CONV_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                 \
    void case_name##_arm_transpose_conv_f16(void)                                                                      \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        const cmsis_nn_context ctx = {0};                                                                              \
        const cmsis_nn_context output_ctx = {0};                                                                       \
        const cmsis_nn_transpose_conv_params_f16 params = {                                                            \
            .stride = {.h = CASE_PREFIX##_STRIDE_H, .w = CASE_PREFIX##_STRIDE_W},                                      \
            .padding = {.h = CASE_PREFIX##_PADDING_H, .w = CASE_PREFIX##_PADDING_W},                                   \
            .padding_offsets = {.h = CASE_PREFIX##_PADDING_OFFSET_H, .w = CASE_PREFIX##_PADDING_OFFSET_W},             \
            .dilation = {.h = CASE_PREFIX##_DILATION_H, .w = CASE_PREFIX##_DILATION_W},                                \
            .activation = {.min = CASE_PREFIX##_OUT_ACTIVATION_MIN, .max = CASE_PREFIX##_OUT_ACTIVATION_MAX}};         \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .c = CASE_PREFIX##_IN_CH};                                                   \
        const cmsis_nn_dims filter_dims = {.n = CASE_PREFIX##_OUT_CH,                                                  \
                                           .h = CASE_PREFIX##_FILTER_H,                                                \
                                           .w = CASE_PREFIX##_FILTER_W,                                                \
                                           .c = CASE_PREFIX##_IN_CH};                                                  \
        const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = CASE_PREFIX##_OUT_CH};                           \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                           \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
        arm_cmsis_nn_status status;                                                                                    \
                                                                                                                       \
        if (CASE_PREFIX##_USE_WRAPPER)                                                                                 \
        {                                                                                                              \
            status = arm_transpose_conv_wrapper_f16(&ctx,                                                              \
                                                    &output_ctx,                                                       \
                                                    &params,                                                           \
                                                    &input_dims,                                                       \
                                                    case_name##_input,                                                 \
                                                    &filter_dims,                                                      \
                                                    case_name##_weights,                                               \
                                                    &bias_dims,                                                        \
                                                    case_name##_biases,                                                \
                                                    &output_dims,                                                      \
                                                    output,                                                            \
                                                    CASE_PREFIX##_LAYOUT);                                             \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            status = arm_transpose_conv_f16(&ctx,                                                                      \
                                            &output_ctx,                                                               \
                                            &params,                                                                   \
                                            &input_dims,                                                               \
                                            case_name##_input,                                                         \
                                            &filter_dims,                                                              \
                                            case_name##_weights,                                                       \
                                            &bias_dims,                                                                \
                                            case_name##_biases,                                                        \
                                            &output_dims,                                                              \
                                            output,                                                                    \
                                            CASE_PREFIX##_LAYOUT);                                                     \
        }                                                                                                              \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);                                                               \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref[i], (float)output[i]);                 \
        }                                                                                                              \
    }

RUN_TRANSPOSE_CONV_F16_CASE(TRANSPOSE_CONV_BASIC_F16, transpose_conv_basic_f16, 5.0e-3f)
RUN_TRANSPOSE_CONV_F16_CASE(TRANSPOSE_CONV_BASIC_NHWC_F16, transpose_conv_basic_nhwc_f16, 5.0e-3f)

// Rebuild a float16 from bits held in a volatile so the NaN input is created at run time; the
// bit-pattern NaN check below survives even a standalone -Ofast build of this TU, where
// -ffinite-math-only licenses folding isnan() to false. Same idiom as the elementwise NaN suites (#380).
static float16_t tc_f16_from_bits(volatile const uint16_t *bits)
{
    const uint16_t b = *bits;
    float16_t x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static inline bool tc_f16_bits_are_nan(float16_t x)
{
    uint16_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return (uint16_t)(bits & 0x7FFFu) > 0x7C00u;
}

// NaN through arm_nn_vector_clamp_f16, the output clamp shared by the conv/depthwise/transpose-conv f16
// family. On the scalar (non-MVE) build path the clamp is the bit-classified clamp of #380, so the NaN
// element must come back NaN at every optimization level; the MVE leg clamps with vmaxnmq/vminnmq and no
// NaN restore, so there the NaN resolves to the lower clamp bound instead. A 1x1 kernel with unit weight
// makes the affected output element exactly the NaN input element.
void transpose_conv_nan_f16_arm_transpose_conv_f16(void)
{
    volatile uint16_t nan_bits = 0x7E00u;
    const float16_t nan = tc_f16_from_bits(&nan_bits);
    const float16_t input[4] = {nan, (float16_t)2.0f, (float16_t)-9.0f, (float16_t)0.5f};
    const float16_t weights[1] = {(float16_t)1.0f};
    float16_t output[4] = {0};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_context output_ctx = {0};
    const cmsis_nn_transpose_conv_params_f16 params = {.stride = {.h = 1, .w = 1},
                                                       .padding = {.h = 0, .w = 0},
                                                       .padding_offsets = {.h = 0, .w = 0},
                                                       .dilation = {.h = 1, .w = 1},
                                                       .activation = {.min = (float16_t)-6.0f, .max = (float16_t)6.0f}};
    const cmsis_nn_dims input_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
    const cmsis_nn_dims filter_dims = {.n = 1, .h = 1, .w = 1, .c = 1};
    const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = 1};
    const cmsis_nn_dims output_dims = {.n = 1, .h = 2, .w = 2, .c = 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_transpose_conv_f16(&ctx,
                                             &output_ctx,
                                             &params,
                                             &input_dims,
                                             input,
                                             &filter_dims,
                                             weights,
                                             &bias_dims,
                                             NULL,
                                             &output_dims,
                                             output,
                                             ARM_NN_LAYOUT_NHWC));

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, (float32_t)output[0]);
#else
    TEST_ASSERT_TRUE_MESSAGE(tc_f16_bits_are_nan(output[0]), "Expected NaN through the scalar clamp");
#endif
    TEST_ASSERT_EQUAL_FLOAT(2.0f, (float32_t)output[1]);
    TEST_ASSERT_EQUAL_FLOAT(-6.0f, (float32_t)output[2]);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, (float32_t)output[3]);
}
