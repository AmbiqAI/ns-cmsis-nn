/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Focused tests for the shared float32 scalar activation helpers in
 * Include/Internal/arm_nn_activation_flt.h (issue #250).
 *
 * This suite is deliberately self-contained: references are computed inline
 * against libm (tanhf/expf) or against a verbatim copy of the pre-#250 helper,
 * so there is no generated ../TestData dependency.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <unity.h>

/* Quoted relative form on purpose: check #10 in scripts/check_pdsc.py only
 * resolves path-shaped *quoted* includes, so this spelling is what puts the
 * suite under that guard and makes a header move surface as a CI failure
 * rather than as a silently unbuildable suite (see #256). */
#include "../../../../Include/Internal/arm_nn_activation_flt.h"

/* Sweep of the old table window: [0, 3.99] at 1e-4, both signs. */
#define TANH_SWEEP_STEP (1.0e-4f)
#define TANH_SWEEP_LAST (39900)

/* Newly covered band and tail. */
#define TANH_BAND_TOL (5.0e-5f)
#define TANH_TAIL_TOL (1.5e-5f)
#define SIGMOID_TOL (5.0e-6f)

static uint32_t f32_bits(float32_t x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

/*
 * Verbatim copy of arm_nn_tanh_scalar_ref_f32 as it stood before #250: a
 * 256-segment LUT over [0, 4]. It reads the first 257 entries of the shared
 * table, which #250 kept bit-identical (6/384 == 4/256 == 1/64), so for
 * |x| <= 4 the current helper must reproduce this bit for bit.
 */
static float32_t tanh_pre_250_ref_f32(float32_t x)
{
    float32_t ax = (x < 0.0f) ? -x : x;
    const float32_t xmax = 4.0f;

    if (ax > xmax)
    {
        return (x < 0.0f) ? -1.0f : 1.0f;
    }

    const float32_t t = ax * (256.0f / xmax);
    int32_t idx = (int32_t)t;
    idx = CLAMP(idx, 255, 0);
    const float32_t frac = t - (float32_t)idx;
    const float32_t y0 = arm_nn_tanh_lut384_f32[idx];
    const float32_t y1 = arm_nn_tanh_lut384_f32[idx + 1];
    const float32_t y = y0 + (y1 - y0) * frac;
    return (x < 0.0f) ? -y : y;
}

/* (a) The old domain is untouched: bit-for-bit equality with the old algorithm. */
void nn_activation_helpers_f32_tanh_old_domain_bit_identical(void)
{
    for (int32_t i = 0; i <= TANH_SWEEP_LAST; ++i)
    {
        const float32_t x = (float32_t)i * TANH_SWEEP_STEP;

        TEST_ASSERT_EQUAL_UINT32(f32_bits(tanh_pre_250_ref_f32(x)), f32_bits(arm_nn_tanh_scalar_ref_f32(x)));
        TEST_ASSERT_EQUAL_UINT32(f32_bits(tanh_pre_250_ref_f32(-x)), f32_bits(arm_nn_tanh_scalar_ref_f32(-x)));
    }

    /* Exact grid points, where interpolation degenerates to a table read. */
    for (int32_t i = 0; i <= 256; ++i)
    {
        const float32_t x = (float32_t)i / 64.0f;

        TEST_ASSERT_EQUAL_UINT32(f32_bits(tanh_pre_250_ref_f32(x)), f32_bits(arm_nn_tanh_scalar_ref_f32(x)));
    }
}

/* (b) The band the old helper clamped away is now interpolated. */
void nn_activation_helpers_f32_tanh_extended_band_accuracy(void)
{
    for (int32_t i = 0; i <= 20000; ++i)
    {
        const float32_t x = 4.0f + (float32_t)i * 1.0e-4f;

        TEST_ASSERT_FLOAT_WITHIN(TANH_BAND_TOL, tanhf(x), arm_nn_tanh_scalar_ref_f32(x));
        TEST_ASSERT_FLOAT_WITHIN(TANH_BAND_TOL, tanhf(-x), arm_nn_tanh_scalar_ref_f32(-x));
    }
}

/* (c) The residual clamp step above the new boundary is 1 - tanh(6) ~= 1.2e-5. */
void nn_activation_helpers_f32_tanh_tail_clamp_step(void)
{
    for (int32_t i = 0; i <= 14000; ++i)
    {
        const float32_t x = 6.0f + (float32_t)i * 1.0e-3f;

        TEST_ASSERT_FLOAT_WITHIN(TANH_TAIL_TOL, tanhf(x), arm_nn_tanh_scalar_ref_f32(x));
        TEST_ASSERT_FLOAT_WITHIN(TANH_TAIL_TOL, tanhf(-x), arm_nn_tanh_scalar_ref_f32(-x));
    }

    /* Saturation is exact, and continuous with the last interpolated sample. */
    TEST_ASSERT_EQUAL_FLOAT(1.0f, arm_nn_tanh_scalar_ref_f32(100.0f));
    TEST_ASSERT_EQUAL_FLOAT(-1.0f, arm_nn_tanh_scalar_ref_f32(-100.0f));
}

/* (d) Documented non-finite contract of the scalar leg. */
void nn_activation_helpers_f32_tanh_non_finite_contract(void)
{
    const float32_t y_nan = arm_nn_tanh_scalar_ref_f32((float32_t)NAN);

    /* NaN propagates rather than saturating. Note this pins the contract only:
     * the pre-#250 helper also returned NaN on every target we can measure, so
     * this assertion does not observe the removal of the undefined float-to-int
     * conversion -- that fix is a source-level guarantee, not a behaviour
     * change. */
    TEST_ASSERT_TRUE(y_nan != y_nan);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, arm_nn_tanh_scalar_ref_f32((float32_t)INFINITY));
    TEST_ASSERT_EQUAL_FLOAT(-1.0f, arm_nn_tanh_scalar_ref_f32(-(float32_t)INFINITY));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, arm_nn_tanh_scalar_ref_f32(0.0f));
}

/* (e) Sigmoid: defined (not undefined) on NaN, unchanged accuracy on finites. */
void nn_activation_helpers_f32_sigmoid_contract_and_accuracy(void)
{
    /* Documented value: the exp domain clamp maps NaN to its upper bound, so
     * the result is 1.0f. See arm_nn_softmax_exp_lut_f32(). */
    TEST_ASSERT_EQUAL_FLOAT(1.0f, arm_nn_sigmoid_scalar_f32((float32_t)NAN));

    TEST_ASSERT_EQUAL_FLOAT(1.0f, arm_nn_sigmoid_scalar_f32((float32_t)INFINITY));

    /* -Inf clamps to the exp domain's lower bound, so the result is
     * exp(-80)/(1 + exp(-80)) ~= 1.8e-35, not an exact zero. */
    TEST_ASSERT_FLOAT_WITHIN(1.0e-30f, 0.0f, arm_nn_sigmoid_scalar_f32(-(float32_t)INFINITY));
    TEST_ASSERT_FLOAT_WITHIN(SIGMOID_TOL, 0.5f, arm_nn_sigmoid_scalar_f32(0.0f));

    for (int32_t i = -20000; i <= 20000; ++i)
    {
        const float32_t x = (float32_t)i * 1.0e-3f;
        const float32_t ref = 1.0f / (1.0f + expf(-x));

        TEST_ASSERT_FLOAT_WITHIN(SIGMOID_TOL, ref, arm_nn_sigmoid_scalar_f32(x));
    }
}
