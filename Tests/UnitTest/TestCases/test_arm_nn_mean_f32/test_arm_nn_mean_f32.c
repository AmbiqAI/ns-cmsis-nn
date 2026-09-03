/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include <arm_nnfunctions.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <unity.h>

#include "mean_f32_data.h"

static const cmsis_nn_dims mean_f32_input_dims = {2, 3, 1, 5};

// Bit-pattern non-finite constructors: armclang rejects the NAN/INFINITY
// literals outright under -Ofast (-Werror,-Wnan-infinity-disabled), and
// bit patterns survive any finite-math assumption on the producing side.
// static inline (PR #390 lesson): MVE target builds may leave a helper
// uncalled on one path, and a plain static function then trips
// -Werror=unused-function.
static inline float32_t mean_f32_make_nan(void)
{
    const uint32_t bits = 0x7FC00000U;
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline float32_t mean_f32_make_inf(void)
{
    const uint32_t bits = 0x7F800000U;
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void mean_f32_check(const cmsis_nn_dims *axis_dims, const cmsis_nn_dims *output_dims, const float32_t *expected)
{
    float32_t output[MEAN_F32_INPUT_SIZE] = {0};
    const int32_t output_size = output_dims->n * output_dims->h * output_dims->w * output_dims->c;

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_mean_f32(mean_f32_input, &mean_f32_input_dims, axis_dims, output, output_dims));
    for (int32_t i = 0; i < output_size; ++i)
    {
        // Golden refs use sequential float32 accumulation; MVE folds four
        // lanes, so allow a few ulps of reassociation slack.
        const float32_t tolerance = 8.0f * FLT_EPSILON * fmaxf(fabsf(expected[i]), 1.0f);
        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected[i], output[i]);
    }
}

// float64 reference: accumulate in double, divide in double, round once to
// float32. The kernel accumulates in float32 (no cheap wider type on
// M-profile), so the comparison cannot be bit-exact for long sums: the
// classic sequential-summation bound is |err| <= n * eps * sum(|x_i|),
// i.e. n * FLT_EPSILON * mean(|x|) after the divide, and the tolerance is
// scaled to the reduction length accordingly. static (not static inline
// in a header) so every build, including MVE target builds where some
// cases stress only one path, still compiles it without
// -Wunused-function noise.
static void mean_f32_check_vs_f64(const float32_t *input,
                                  const cmsis_nn_dims *input_dims,
                                  const cmsis_nn_dims *axis_dims,
                                  const cmsis_nn_dims *output_dims)
{
    static float32_t output[4096];
    const int32_t in_shape[4] = {input_dims->n, input_dims->h, input_dims->w, input_dims->c};
    const int32_t mask[4] = {axis_dims->n ? 1 : 0, axis_dims->h ? 1 : 0, axis_dims->w ? 1 : 0, axis_dims->c ? 1 : 0};
    const int32_t out_shape[4] = {output_dims->n, output_dims->h, output_dims->w, output_dims->c};
    const int32_t output_size = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];

    TEST_ASSERT_TRUE(output_size <= 4096);
    memset(output, 0, sizeof(output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f32(input, input_dims, axis_dims, output, output_dims));

    for (int32_t n = 0; n < out_shape[0]; ++n)
        for (int32_t h = 0; h < out_shape[1]; ++h)
            for (int32_t w = 0; w < out_shape[2]; ++w)
                for (int32_t c = 0; c < out_shape[3]; ++c)
                {
                    double sum = 0.0;
                    double sum_abs = 0.0;
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
                                    sum_abs += fabs((double)input[idx]);
                                    ++count;
                                }
                    const int32_t out_idx = ((n * out_shape[1] + h) * out_shape[2] + w) * out_shape[3] + c;
                    const float expected = (float)(sum / (double)count);
                    const double mean_abs = sum_abs / (double)count;
                    const float tolerance =
                        (float)((double)count * (double)FLT_EPSILON * mean_abs) + 4.0f * FLT_EPSILON;
                    TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, output[out_idx]);
                }
}

// {2,2,3,2} strided fixture shared by the axis-combination cases.
static const float32_t mean_f32_strided_input[24] = {
    -3.5f, 1.25f, 2.0f,  4.5f,  -0.75f, 6.0f, -2.5f, 0.5f,  3.25f, 1.75f, -1.0f, 5.5f,
    -4.0f, 2.25f, 7.0f,  0.25f, -6.5f,  3.5f, 1.0f,  -2.0f, 8.0f,  0.75f, -1.5f, 4.0f};
static const cmsis_nn_dims mean_f32_strided_dims = {2, 2, 3, 2};

void mean_f32_axis_c_arm_nn_mean_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 1, 1};
    mean_f32_check(&axis_dims, &output_dims, mean_f32_ref_c);
}

void mean_f32_axis_h_arm_nn_mean_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 5};
    mean_f32_check(&axis_dims, &output_dims, mean_f32_ref_h);
}

void mean_f32_axis_hc_arm_nn_mean_f32(void)
{
    const cmsis_nn_dims axis_dims = {0, 1, 0, 1};
    const cmsis_nn_dims output_dims = {2, 1, 1, 1};
    mean_f32_check(&axis_dims, &output_dims, mean_f32_ref_hc);
}

void mean_f32_axis_all_arm_nn_mean_f32(void)
{
    const cmsis_nn_dims axis_dims = {1, 1, 1, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    mean_f32_check(&axis_dims, &output_dims, mean_f32_ref_all);
}

void mean_f32_identity_arm_nn_mean_f32(void)
{
    float32_t output[MEAN_F32_INPUT_SIZE] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_mean_f32(mean_f32_input, &mean_f32_input_dims, &axis_dims, output, &mean_f32_input_dims));
    for (int32_t i = 0; i < MEAN_F32_INPUT_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT(mean_f32_input[i], output[i]);
    }
}

void mean_f32_nan_inf_arm_nn_mean_f32(void)
{
    const float32_t input[6] = {1.0f, mean_f32_make_inf(), 2.0f, mean_f32_make_nan(), 1.0f, 2.0f};
    float32_t output[2] = {0};
    const cmsis_nn_dims input_dims = {1, 1, 2, 3};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 2, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f32(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_FLOAT_IS_INF(output[0]);
    TEST_ASSERT_FLOAT_IS_NAN(output[1]);
}

void mean_f32_arg_error_arm_nn_mean_f32(void)
{
    float32_t output[6] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {2, 3, 1, 1};
    const cmsis_nn_dims invalid_input_dims = {2, 0, 1, 5};
    const cmsis_nn_dims invalid_unreduced_output_dims = {2, 4, 1, 1};
    const cmsis_nn_dims invalid_reduced_output_dims = {2, 3, 1, 2};
    const cmsis_nn_dims overflow_input_dims = {INT32_MAX, 2, 1, 1};
    const cmsis_nn_dims overflow_axis_dims = {1, 1, 0, 0};
    const cmsis_nn_dims overflow_output_dims = {1, 1, 1, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f32(NULL, &mean_f32_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, arm_nn_mean_f32(mean_f32_input, NULL, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f32(mean_f32_input, &mean_f32_input_dims, NULL, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f32(mean_f32_input, &mean_f32_input_dims, &axis_dims, NULL, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f32(mean_f32_input, &mean_f32_input_dims, &axis_dims, output, NULL));
    // Zero-size reduced dimension: the reduction set is empty, so the mean
    // is undefined and reduced_count == 0 must reject, not divide by zero.
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f32(mean_f32_input, &invalid_input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f32(
                          mean_f32_input, &mean_f32_input_dims, &axis_dims, output, &invalid_unreduced_output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f32(
                          mean_f32_input, &mean_f32_input_dims, &axis_dims, output, &invalid_reduced_output_dims));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_nn_mean_f32(
                          mean_f32_input, &overflow_input_dims, &overflow_axis_dims, output, &overflow_output_dims));
}

void mean_f32_axis_n_arm_nn_mean_f32(void)
{
    // Reduce the batch axis only: generic path (h, w, c untouched, h > 1).
    const cmsis_nn_dims axis_dims = {1, 0, 0, 0};
    const cmsis_nn_dims output_dims = {1, 3, 1, 5};
    mean_f32_check_vs_f64(mean_f32_input, &mean_f32_input_dims, &axis_dims, &output_dims);
}

void mean_f32_axis_w_arm_nn_mean_f32(void)
{
    // Reduce W with C > 1 unreduced: strided generic path.
    const cmsis_nn_dims axis_dims = {0, 0, 1, 0};
    const cmsis_nn_dims output_dims = {2, 2, 1, 2};
    mean_f32_check_vs_f64(mean_f32_strided_input, &mean_f32_strided_dims, &axis_dims, &output_dims);
}

void mean_f32_axis_nw_arm_nn_mean_f32(void)
{
    // PR #294 review item 4 (inherited via #412): leading axis reduced
    // while a later axis of size > 1 is not -- the only class where the
    // generic path's output index compression is non-trivial.
    const cmsis_nn_dims axis_dims = {1, 0, 1, 0};
    const cmsis_nn_dims output_dims = {1, 2, 1, 2};
    mean_f32_check_vs_f64(mean_f32_strided_input, &mean_f32_strided_dims, &axis_dims, &output_dims);
}

void mean_f32_axis_hw_arm_nn_mean_f32(void)
{
    // H and W reduced, C > 1 unreduced: strided generic path.
    const cmsis_nn_dims axis_dims = {0, 1, 1, 0};
    const cmsis_nn_dims output_dims = {2, 1, 1, 2};
    mean_f32_check_vs_f64(mean_f32_strided_input, &mean_f32_strided_dims, &axis_dims, &output_dims);
}

void mean_f32_axis_wc_arm_nn_mean_f32(void)
{
    // W and C form a contiguous suffix: flatten fast path with outer > 1.
    const cmsis_nn_dims axis_dims = {0, 0, 1, 1};
    const cmsis_nn_dims output_dims = {2, 2, 1, 1};
    mean_f32_check_vs_f64(mean_f32_strided_input, &mean_f32_strided_dims, &axis_dims, &output_dims);
}

void mean_f32_dim1_axis_arm_nn_mean_f32(void)
{
    // Reducing an axis whose size is already 1 must be an exact identity.
    float32_t output[MEAN_F32_INPUT_SIZE] = {0};
    const cmsis_nn_dims axis_dims = {0, 0, 1, 0};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_nn_mean_f32(mean_f32_input, &mean_f32_input_dims, &axis_dims, output, &mean_f32_input_dims));
    for (int32_t i = 0; i < MEAN_F32_INPUT_SIZE; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT(mean_f32_input[i], output[i]);
    }
}

void mean_f32_flatten_long_accumulation_arm_nn_mean_f32(void)
{
    // Accumulation-order/precision on the FLATTEN path, with a reduction
    // length that is NOT a multiple of the MVE vector width (2999 = 4*749
    // + 3): the last three elements only reach the sum through the
    // predicated tail. They carry the largest values in the tensor, so a
    // dropped vctp32q predication (reading past the row) or an
    // off-by-one row stride shifts the result far outside the scaled
    // tolerance. 0.1f is not representable in binary float, so every one
    // of the 2999 additions rounds and the float64 reference genuinely
    // exercises the length-scaled tolerance.
    static float32_t input[2999];
    const cmsis_nn_dims input_dims = {1, 1, 1, 2999};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};
    for (int32_t i = 0; i < 2999; ++i)
    {
        input[i] = 0.1f * (float32_t)(1 + (i % 7));
    }
    input[2996] = 1000.25f;
    input[2997] = -750.5f;
    input[2998] = 500.125f;

    mean_f32_check_vs_f64(input, &input_dims, &axis_dims, &output_dims);
}

void mean_f32_generic_long_accumulation_arm_nn_mean_f32(void)
{
    // Accumulation-order/precision on the GENERIC path: reduce H with
    // C > 1 unreduced (suffix test fails at C). 50 rounding additions per
    // output column against the float64 reference with the tolerance
    // scaled to the reduction length.
    static float32_t input[200];
    const cmsis_nn_dims input_dims = {1, 50, 1, 4};
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {1, 1, 1, 4};
    for (int32_t i = 0; i < 200; ++i)
    {
        input[i] = 0.1f * (float32_t)(1 + (i % 17)) - 0.7f;
    }

    mean_f32_check_vs_f64(input, &input_dims, &axis_dims, &output_dims);
}

void mean_f32_finite_overflow_arm_nn_mean_f32(void)
{
    // Contract pin (not an accident of implementation): four copies of
    // 3e38 overflow the float32 intermediate sum to +Inf even though the
    // true mean, 3e38, is representable. With all values same-signed,
    // EVERY accumulation order overflows -- any two partial sums of two
    // 3e38 terms already exceed FLT_MAX -- so unlike the mixed-sign
    // divergence case below this expectation is leg-agnostic: sequential
    // scalar goes Inf at element 2, and the MVE flatten path goes Inf in
    // the lane fold (3e38 + 3e38). Verified on host gcc-12 (-O0/-Ofast)
    // and FVP Corstone-300 MVE and forced-scalar (-Ofast).
    const float32_t input[4] = {3e38f, 3e38f, 3e38f, 3e38f};
    float32_t output = 0.0f;
    const cmsis_nn_dims input_dims = {1, 1, 1, 4};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f32(input, &input_dims, &axis_dims, &output, &output_dims));
    TEST_ASSERT_FLOAT_IS_INF(output);
    TEST_ASSERT_TRUE(output > 0.0f);
}

void mean_f32_finite_overflow_divergence_arm_nn_mean_f32(void)
{
    // On mixed-sign inputs whose PARTIAL sums exceed FLT_MAX the legs
    // genuinely disagree (finite vs Inf), so pin each leg's own behaviour
    // rather than a false universal Inf contract. Measured (bit-exact) on
    // FVP Corstone-300 at -Ofast and host gcc-12 at -O0/-Ofast:
    //  - scalar_inf_input {3e38, 3e38, 0, 0, -3e38, -3e38, 0, 0}: the
    //    sequential scalar sum saturates to +Inf at element 2 and Inf is
    //    sticky (0x7F800000); the MVE flatten path's four per-lane sums
    //    each cancel to 0 and the fold returns exactly 0 (0x00000000).
    //  - vector_inf_input {3e38, 0, -3e38, 0, 0, 3e38, 0, -3e38}: the
    //    sequential sum never leaves [-3e38, 3e38] and returns exactly 0;
    //    the MVE per-lane sums are {3e38, 3e38, -3e38, -3e38} and the
    //    fold saturates at 3e38 + 3e38 -> +Inf.
    //  - alternating_input {3e38, -3e38, ...} (x4): the sequential sum
    //    oscillates between 3e38 and 0 and returns exactly 0; the MVE
    //    per-lane sums are {+Inf, -Inf, +Inf, -Inf} (each lane adds two
    //    same-signed 3e38-magnitude values) and the fold's Inf + -Inf
    //    yields NaN (0x7FC00000) from all-finite inputs (final-gate
    //    review finding, reproduced here on the FVP).
    // ARM_MATH_AUTOVECTORIZE builds compile the scalar source under the
    // vector leg's -Ofast, which hands the accumulation order to the
    // compiler -- gcc 14.2.1 autovectorizes it into the same 4-lane split
    // as MVE (a genuinely sequential target build needs
    // -fno-tree-vectorize) -- so only the call contract is asserted there
    // (the test TU's own -fno-finite-math-only strips __FAST_MATH__, so
    // that macro cannot probe how the kernel library was compiled).
    const float32_t scalar_inf_input[8] = {3e38f, 3e38f, 0.0f, 0.0f, -3e38f, -3e38f, 0.0f, 0.0f};
    const float32_t vector_inf_input[8] = {3e38f, 0.0f, -3e38f, 0.0f, 0.0f, 3e38f, 0.0f, -3e38f};
    const float32_t alternating_input[8] = {3e38f, -3e38f, 3e38f, -3e38f, 3e38f, -3e38f, 3e38f, -3e38f};
    float32_t scalar_inf_output = 42.0f;
    float32_t vector_inf_output = 42.0f;
    float32_t alternating_output = 42.0f;
    const cmsis_nn_dims input_dims = {1, 1, 1, 8};
    const cmsis_nn_dims axis_dims = {0, 0, 0, 1};
    const cmsis_nn_dims output_dims = {1, 1, 1, 1};

    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mean_f32(scalar_inf_input, &input_dims, &axis_dims, &scalar_inf_output, &output_dims));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mean_f32(vector_inf_input, &input_dims, &axis_dims, &vector_inf_output, &output_dims));
    TEST_ASSERT_EQUAL(
        ARM_CMSIS_NN_SUCCESS,
        arm_nn_mean_f32(alternating_input, &input_dims, &axis_dims, &alternating_output, &output_dims));

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    TEST_ASSERT_EQUAL_FLOAT(0.0f, scalar_inf_output);
    TEST_ASSERT_FLOAT_IS_INF(vector_inf_output);
    TEST_ASSERT_TRUE(vector_inf_output > 0.0f);
    TEST_ASSERT_FLOAT_IS_NAN(alternating_output);
#elif !defined(ARM_MATH_AUTOVECTORIZE) && !defined(__FAST_MATH__)
    TEST_ASSERT_FLOAT_IS_INF(scalar_inf_output);
    TEST_ASSERT_TRUE(scalar_inf_output > 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, vector_inf_output);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, alternating_output);
#endif
}

void mean_f32_generic_nan_inf_arm_nn_mean_f32(void)
{
    // NaN and Inf propagation on the GENERIC path (the nan_inf case above
    // reduces C and takes the flatten path). Reducing H with W = C = 2
    // unreduced stays generic. NaN poisons only output[0]; +Inf poisons
    // only output[3]; the two clean columns stay exact.
    const float32_t input[8] = {mean_f32_make_nan(), 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, mean_f32_make_inf()};
    float32_t output[4] = {0};
    const cmsis_nn_dims input_dims = {1, 2, 2, 2};
    const cmsis_nn_dims axis_dims = {0, 1, 0, 0};
    const cmsis_nn_dims output_dims = {1, 1, 2, 2};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, arm_nn_mean_f32(input, &input_dims, &axis_dims, output, &output_dims));
    TEST_ASSERT_FLOAT_IS_NAN(output[0]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, output[1]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, output[2]);
    TEST_ASSERT_FLOAT_IS_INF(output[3]);
}
