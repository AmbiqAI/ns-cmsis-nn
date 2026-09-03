/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Focused tests for the shared float16 scalar tanh helper in
 * Include/Internal/arm_nn_activation_flt.h (issue #407).
 *
 * Self-contained like the float32 twin: references are computed inline against
 * libm's double tanh(), so there is no generated ../TestData dependency.
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <unity.h>

/* Quoted relative form on purpose: check #10 in scripts/check_pdsc.py only
 * resolves path-shaped *quoted* includes, so this spelling is what puts the
 * suite under that guard (see #256). */
#include "../../../../Include/Internal/arm_nn_activation_flt.h"

/* Exhaustive bound, measured over all 2^16 patterns: 4.57e-4 inside |x| <= 4
 * and 6.65e-4 overall (the ax > 4 saturation step, shared with the MVE leg). */
#define TANH_F16_IN_RANGE_TOL (5.0e-4f)
#define TANH_F16_GLOBAL_TOL (7.0e-4f)

/* The two tolerances that gated helia-rt's cortex-m55 scalar (+nomve) legs
 * (helia-rt #239), pinned with a 4x margin so the fix holds with headroom. */
#define TANH_F16_HELIA_X2_TOL (2.5e-3f)  /* helia-rt asserts 1e-2 at x = 2 */
#define TANH_F16_HELIA_X05_TOL (7.5e-4f) /* helia-rt asserts 3e-3 at x = 0.5 */

static uint16_t f16_bits(float16_t x)
{
    uint16_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static float16_t f16_from_bits(uint16_t u)
{
    float16_t x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static float f16_val(float16_t x) { return (float)(_Float16)x; }

/* (a) Exhaustive sweep of every finite float16 against double tanh. */
void nn_activation_helpers_f16_tanh_exhaustive_accuracy(void)
{
    for (uint32_t u = 0; u < 0x10000U; ++u)
    {
        const uint16_t b = (uint16_t)u;
        if ((uint16_t)(b & 0x7FFFu) >= 0x7C00u)
        {
            continue; /* Inf and NaN are pinned in (c). */
        }
        const float16_t x = f16_from_bits(b);
        const float xf = f16_val(x);
        const float ref = (float)tanh((double)xf);
        const float y = f16_val(arm_nn_tanh_scalar_ref_f16(x));

        TEST_ASSERT_FLOAT_WITHIN(TANH_F16_GLOBAL_TOL, ref, y);
        if (xf >= -4.0f && xf <= 4.0f)
        {
            TEST_ASSERT_FLOAT_WITHIN(TANH_F16_IN_RANGE_TOL, ref, y);
        }
    }
}

/* (b) The exact points whose drift kept helia-rt's ATfE contexts red (#407). */
void nn_activation_helpers_f16_tanh_helia_rt_points(void)
{
    const float16_t x2 = (float16_t)(_Float16)2.0f;
    const float16_t x05 = (float16_t)(_Float16)0.5f;

    TEST_ASSERT_FLOAT_WITHIN(TANH_F16_HELIA_X2_TOL, (float)tanh(2.0), f16_val(arm_nn_tanh_scalar_ref_f16(x2)));
    TEST_ASSERT_FLOAT_WITHIN(TANH_F16_HELIA_X2_TOL,
                             (float)tanh(-2.0),
                             f16_val(arm_nn_tanh_scalar_ref_f16((float16_t)(-(_Float16)x2))));
    TEST_ASSERT_FLOAT_WITHIN(TANH_F16_HELIA_X05_TOL, (float)tanh(0.5), f16_val(arm_nn_tanh_scalar_ref_f16(x05)));
    TEST_ASSERT_FLOAT_WITHIN(TANH_F16_HELIA_X05_TOL,
                             (float)tanh(-0.5),
                             f16_val(arm_nn_tanh_scalar_ref_f16((float16_t)(-(_Float16)x05))));
}

/* (c) Non-finite and boundary contract: NaN propagates by bit pattern (the
 * guard is integer-domain, so it holds under -Ofast too, #333 / #334);
 * +/-Inf and ax > 4 saturate to exactly +/-1; ax == 4 interpolates to
 * lut[256]; +/-0 pass through with sign. */
void nn_activation_helpers_f16_tanh_non_finite_and_boundary(void)
{
    const uint16_t nan_patterns[3] = {0x7E00u, 0xFE00u, 0x7C01u};
    for (int32_t i = 0; i < 3; ++i)
    {
        const uint16_t y_bits = f16_bits(arm_nn_tanh_scalar_ref_f16(f16_from_bits(nan_patterns[i])));
        TEST_ASSERT_TRUE((uint16_t)(y_bits & 0x7FFFu) > 0x7C00u);
    }

    TEST_ASSERT_EQUAL_UINT16(0x3C00u, f16_bits(arm_nn_tanh_scalar_ref_f16(f16_from_bits(0x7C00u)))); /* +Inf */
    TEST_ASSERT_EQUAL_UINT16(0xBC00u, f16_bits(arm_nn_tanh_scalar_ref_f16(f16_from_bits(0xFC00u)))); /* -Inf */
    TEST_ASSERT_EQUAL_FLOAT(1.0f, f16_val(arm_nn_tanh_scalar_ref_f16((float16_t)(_Float16)100.0f)));
    TEST_ASSERT_EQUAL_FLOAT(-1.0f, f16_val(arm_nn_tanh_scalar_ref_f16((float16_t)(_Float16)-100.0f)));

    /* First value above the table window saturates; the boundary itself interpolates to lut[256]. */
    TEST_ASSERT_EQUAL_UINT16(0x3C00u, f16_bits(arm_nn_tanh_scalar_ref_f16(f16_from_bits(0x4401u)))); /* 4.00390625 */
    TEST_ASSERT_EQUAL_UINT16(arm_nn_tanh_lut_f16[256],
                             f16_bits(arm_nn_tanh_scalar_ref_f16((float16_t)(_Float16)4.0f)));

    /* Zero returns zero. The helper copies x's sign bit (so -0 stays -0 at
     * source level, as the rational helper did), but -Ofast's -fno-signed-zeros
     * licenses the compiler to hand back either zero, so only magnitude is
     * asserted for -0. */
    TEST_ASSERT_EQUAL_UINT16(0x0000u, f16_bits(arm_nn_tanh_scalar_ref_f16(f16_from_bits(0x0000u))));
    TEST_ASSERT_EQUAL_UINT16(0x0000u,
                             (uint16_t)(f16_bits(arm_nn_tanh_scalar_ref_f16(f16_from_bits(0x8000u))) & 0x7FFFu));
}

/* (d) Scalar-vs-MVE agreement (#315 / #407): on MVE builds the two legs must
 * be bit-identical for every finite input except -0.0, where vabsq gives the
 * vector leg +0 while the scalar leg keeps IEEE tanh(-0) == -0. */
void nn_activation_helpers_f16_tanh_scalar_vs_mve_agreement(void)
{
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (uint32_t u = 0; u < 0x10000U; u += 8U)
    {
        float16_t in[8];
        for (uint32_t lane = 0; lane < 8U; ++lane)
        {
            in[lane] = f16_from_bits((uint16_t)(u + lane));
        }
        float16_t out[8];
        vst1q(out, arm_nn_vtanh_lut_direct_mve_f16(vld1q(in)));
        for (uint32_t lane = 0; lane < 8U; ++lane)
        {
            const uint16_t b = (uint16_t)(u + lane);
            if ((uint16_t)(b & 0x7FFFu) >= 0x7C00u || b == 0x8000u)
            {
                continue; /* NaN/Inf divergence is documented at the helpers; -0 noted above. */
            }
            TEST_ASSERT_EQUAL_UINT16(f16_bits(arm_nn_tanh_scalar_ref_f16(in[lane])), f16_bits(out[lane]));
        }
    }
#else
    TEST_PASS();
#endif
}
