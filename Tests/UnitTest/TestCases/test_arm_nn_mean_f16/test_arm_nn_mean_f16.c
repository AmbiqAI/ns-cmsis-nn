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
#include <unity.h>

#include "mean_f16_data.h"

static const cmsis_nn_dims mean_f16_input_dims = {2, 3, 1, 5};

static void mean_f16_check(const cmsis_nn_dims *axis_dims, const cmsis_nn_dims *output_dims, const float16_t *expected)
{
    float16_t output[MEAN_F16_INPUT_SIZE] = {0};
    const int32_t output_size = output_dims->n * output_dims->h * output_dims->w * output_dims->c;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_mean_f16(mean_f16_input, &mean_f16_input_dims, axis_dims, output, output_dims));
    for (int32_t i = 0; i < output_size; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-3f, (float)expected[i], (float)output[i]);
    }
}


// float64 reference: accumulate in double, divide in double, round once to
// float16. Used by the adversarial cases to prove the kernel's
// accumulate-in-float32-round-once behaviour numerically against a wider
// accumulator. static (not static inline in a header) so every build,
// including MVE target builds where some cases stress only one path, still
// compiles it without -Wunused-function noise.
static void mean_f16_check_vs_f64(const float16_t *input,
                                  const cmsis_nn_dims *input_dims,
                                  const cmsis_nn_dims *axis_dims,
                                  const cmsis_nn_dims *output_dims)
{
    float16_t output[MEAN_F16_INPUT_SIZE] = {0};
    const int32_t in_shape[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t mask[4] = {axis_dims->n ? 1 : 0, axis_dims->h ? 1 : 0, axis_dims->w ? 1 : 0, axis_dims->c ? 1 : 0};
    const int32_t out_shape[4] = {output_dims->n, output_dims->h, output_dims->w, output_dims->c};
    const int32_t output_size = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f16(input, input_dims, axis_dims, output, output_dims));

    for (int32_t n = 0; n < out_shape[0]; ++n)
        for (int32_t h = 0; h < out_shape[1]; ++h)
            for (int32_t w = 0; w < out_shape[2]; ++w)
                for (int32_t c = 0; c < out_shape[3]; ++c)
                {
                    double sum = 0.0;
                    int32_t count = 0;
                    const int32_t n_limit = mask[0] ? in_shape[0] : 1;
                    const int32_t h_limit = mask[1] ? in_shape[1] : 1;
                    const int32_t w_limit = mask[2] ? in_shape[2] : 1;
                    const int32_t c_limit = mask[3] ? in_shape[3] : 1;
                    for (int32_t ni = 0; ni < n_limit; ++ni)
                        for (int32_t hi = 0; hi < h_limit; ++hi)
                            for (int32_t wi = 0; wi < w_limit; ++wi)
                                for (int32_t ci = 0; ci < c_limit; ++ci)
                                {
                                    const int32_t in_n = mask[0] ? ni : n;
                                    const int32_t in_h = mask[1] ? hi : h;
                                    const int32_t in_w = mask[2] ? wi : w;
                                    const int32_t in_c = mask[3] ? ci : c;
                                    const int32_t idx =
                                        ((in_n * in_shape[1] + in_h) * in_shape[2] + in_w) * in_shape[3] + in_c;
                                    sum += (double)input[idx];
                                    ++count;
                                }
                    const int32_t out_idx = ((n * out_shape[1] + h) * out_shape[2] + w) * out_shape[3] + c;
                    const float expected = (float)(float16_t)(sum / (double)count);
                    TEST_ASSERT_FLOAT_WITHIN(4.0e-3f, expected, (float)output[out_idx]);
                    (void)output_size;
                }
}

// {2,2,3,2} strided fixture shared by the axis-combination cases.
static const float16_t mean_f16_strided_input[24] = {
    -3.5f, 1.25f, 2.0f,  4.5f, -0.75f, 6.0f, -2.5f, 0.5f,  3.25f, 1.75f, -1.0f, 5.5f,
    -4.0f, 2.25f, 7.0f,  0.25f, -6.5f, 3.5f, 1.0f,  -2.0f, 8.0f,  0.75f, -1.5f, 4.0f};
static const cmsis_nn_dims mean_f16_strided_dims = {2, 2, 3, 2};

void mean_f16_axis_c_arm_nn_mean_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 1, 1};
    mean_f16_check(&axis_dims, &output_dims, mean_f16_ref_c);
}

void mean_f16_axis_h_arm_nn_mean_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 5};
    mean_f16_check(&axis_dims, &output_dims, mean_f16_ref_h);
}

void mean_f16_axis_hc_arm_nn_mean_f16(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 1};
    const cmsis_nn_dims output_dims = {2, 1, 1, 1};
    mean_f16_check(&axis_dims, &output_dims, mean_f16_ref_hc);
}

void mean_f16_axis_all_arm_nn_mean_f16(void)
{
    const cmsis_nn_dims axis_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    mean_f16_check(&axis_dims, &output_dims, mean_f16_ref_all);
}

void mean_f16_identity_arm_nn_mean_f16(void)
{
    float16_t output[MEAN_F16_INPUT_SIZE] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_mean_f16(mean_f16_input, &mean_f16_input_dims, &axis_dims, output, &mean_f16_input_dims));
    for (int32_t i = 0; i < MEAN_F16_INPUT_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT((float)mean_f16_input[i], (float)output[i]);
    }
}

void mean_f16_large_sum_arm_nn_mean_f16(void)
{
    float16_t input[17];
    float16_t output = 0.0f;
    const cmsis_nn_dims input_dims = {1, 1, 1, 17};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    for (int32_t i = 0; i < 17; ++i)
    {
        input[i] = (float16_t)65504.0f;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f16(input, &input_dims, &axis_dims, &output, &output_dims));
    TEST_ASSERT_EQUAL_FLOAT(65504.0f, (float)output);
}

void mean_f16_nan_inf_arm_nn_mean_f16(void)
{
    const float16_t input[6] = {1.0f, (float16_t)INFINITY, 2.0f, (float16_t)NAN, 1.0f, 2.0f};
    float16_t output[2] = {0};
    const cmsis_nn_dims input_dims = {1, 1, 2, 3};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 2, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f16(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_FLOAT_IS_INF((float)output[0]);
    TEST_ASSERT_FLOAT_IS_NAN((float)output[1]);
}

void mean_f16_arg_error_arm_nn_mean_f16(void)
{
    float16_t output[6] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 1, 1};
    const cmsis_nn_dims invalid_input_dims = {2, 0, 1, 5};
    const cmsis_nn_dims invalid_unreduced_output_dims = {2, 4, 1, 1};
    const cmsis_nn_dims invalid_reduced_output_dims = {2, 3, 1, 2};
    const cmsis_nn_dims overflow_input_dims = {INT32_MAX, 2, 1, 1};
    const cmsis_nn_dims overflow_axis_dims = {1, 1, 0, 0};
    const cmsis_nn_dims overflow_output_dims = {1, 1, 1, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f16(NULL, &mean_f16_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_mean_f16(mean_f16_input, NULL, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f16(mean_f16_input, &mean_f16_input_dims, NULL, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f16(mean_f16_input, &mean_f16_input_dims, &axis_dims, NULL, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f16(mean_f16_input, &mean_f16_input_dims, &axis_dims, output, NULL));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f16(mean_f16_input, &invalid_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f16(
                          mean_f16_input, &mean_f16_input_dims, &axis_dims, output, &invalid_unreduced_output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f16(
                          mean_f16_input, &mean_f16_input_dims, &axis_dims, output, &invalid_reduced_output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f16(
                          mean_f16_input, &overflow_input_dims, &overflow_axis_dims, output, &overflow_output_dims));
}

void mean_f16_axis_n_arm_nn_mean_f16(void)
{
    // Reduce the batch axis only: generic path (h, w, c untouched, h > 1).
    const cmsis_nn_dims axis_dims = {1, 0, 0, 0};
    const cmsis_nn_dims output_dims = {1, 3, 1, 5};
    mean_f16_check_vs_f64(mean_f16_input, &mean_f16_input_dims, &axis_dims, &output_dims);
}

void mean_f16_axis_w_arm_nn_mean_f16(void)
{
    // Reduce W with C > 1 unreduced: strided generic path.
    const cmsis_nn_dims axis_dims = {0, 0, 1, 0};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};
    mean_f16_check_vs_f64(mean_f16_strided_input, &mean_f16_strided_dims, &axis_dims, &output_dims);
}

void mean_f16_axis_nw_arm_nn_mean_f16(void)
{
    // PR #294 review item 4: leading axis reduced while a later axis of
    // size > 1 is not -- the only class where the generic path's output
    // index compression is non-trivial.
    const cmsis_nn_dims axis_dims = {1, 0, 1, 0};
    const cmsis_nn_dims output_dims = {1, 2, 1, 2};
    mean_f16_check_vs_f64(mean_f16_strided_input, &mean_f16_strided_dims, &axis_dims, &output_dims);
}

void mean_f16_axis_hw_arm_nn_mean_f16(void)
{
    // H and W reduced, C > 1 unreduced: strided generic path.
    const cmsis_nn_dims axis_dims = {0, 1, 1, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 2};
    mean_f16_check_vs_f64(mean_f16_strided_input, &mean_f16_strided_dims, &axis_dims, &output_dims);
}

void mean_f16_axis_wc_arm_nn_mean_f16(void)
{
    // W and C form a contiguous suffix: flatten fast path with outer > 1.
    const cmsis_nn_dims axis_dims = {0, 0, 1, 1};
    const cmsis_nn_dims output_dims = {2, 2, 1, 1};
    mean_f16_check_vs_f64(mean_f16_strided_input, &mean_f16_strided_dims, &axis_dims, &output_dims);
}

void mean_f16_dim1_axis_arm_nn_mean_f16(void)
{
    // Reducing an axis whose size is already 1 must be an exact identity.
    float16_t output[MEAN_F16_INPUT_SIZE] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 1, 0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_mean_f16(mean_f16_input, &mean_f16_input_dims, &axis_dims, output, &mean_f16_input_dims));
    for (int32_t i = 0; i < MEAN_F16_INPUT_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT((float)mean_f16_input[i], (float)output[i]);
    }
}

void mean_f16_generic_large_sum_arm_nn_mean_f16(void)
{
    // PR #294 review item 2 (blocking): the float32 accumulator on the
    // GENERIC path. Reducing H with C > 1 unreduced forces the generic
    // path; two f16-max values per reduction overflow a float16
    // accumulator (65504 + 65504 -> inf) but are exact in float32, and
    // (65504 + 65504) / 2 == 65504 exactly. Kills the
    // float32_t-sum -> float16_t-sum mutation the review proved alive.
    float16_t input[8];
    float16_t output[4] = {0};
    const cmsis_nn_dims input_dims = {1, 2, 1, 4};
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {1, 1, 1, 4};
    for (int32_t i = 0; i < 8; ++i)
    {
        input[i] = (float16_t)65504.0f;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f16(input, &input_dims, &axis_dims, output, &output_dims));
    for (int32_t i = 0; i < 4; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT(65504.0f, (float)output[i]);
    }
}

void mean_f16_long_accumulation_arm_nn_mean_f16(void)
{
    // 3000 copies of 1.0 reduced over the whole tensor: a float16
    // accumulator saturates at 2048 (1.0 is half an ulp there, ties round
    // to even) and would return about 0.68; float32 accumulation is exact
    // (3000.0 / 3000 == 1.0) and matches the float64 reference bit for
    // bit. Proves the accumulate-in-f32-round-once claim numerically on
    // the flatten path (MVE on target, scalar elsewhere) with a reduction
    // long enough that per-step rounding would otherwise accumulate.
    static float16_t input[3000];
    float16_t output = (float16_t)0.0f;
    const cmsis_nn_dims input_dims = {1, 3, 8, 125};
    const cmsis_nn_dims axis_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    for (int32_t i = 0; i < 3000; ++i)
    {
        input[i] = (float16_t)1.0f;
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f16(input, &input_dims, &axis_dims, &output, &output_dims));
    TEST_ASSERT_EQUAL_FLOAT(1.0f, (float)output);
}

void mean_f16_generic_nan_inf_arm_nn_mean_f16(void)
{
    // NaN and Inf propagation on the GENERIC path (Paul's nan_inf case
    // reduces C and takes the flatten path). Reducing H with W = C = 2
    // unreduced stays generic. NaN poisons only output[0]; +Inf poisons
    // only output[3]; the two clean columns stay exact.
    const float16_t input[8] = {
        (float16_t)NAN, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, (float16_t)INFINITY};
    float16_t output[4] = {0};
    const cmsis_nn_dims input_dims = {1, 2, 2, 2};
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {1, 1, 2, 2};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f16(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_FLOAT_IS_NAN((float)output[0]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, (float)output[1]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, (float)output[2]);
    TEST_ASSERT_FLOAT_IS_INF((float)output[3]);
}
