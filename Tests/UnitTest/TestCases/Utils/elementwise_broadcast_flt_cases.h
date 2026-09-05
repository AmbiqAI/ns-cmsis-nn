/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Shared body of the test_arm_elementwise_{sub,add,mul}_broadcast_{f32,f16} suites (#415).
 *
 * The including suite defines, before this header:
 *   EW_BCAST_OP      EW_BCAST_OP_SUB, EW_BCAST_OP_ADD or EW_BCAST_OP_MUL
 *   EW_BCAST_F16     0 for float32_t, 1 for float16_t
 *   EW_BCAST_PREFIX  token prefix of the suite's test functions, e.g. sub_broadcast_f32
 *   EW_BCAST_KERNEL  the broadcast kernel under test
 *   EW_BCAST_FLAT    the flat kernel whose numerics it must reproduce bit for bit
 *
 * Every expectation is computed here, in double, from an explicit NumPy-style index walk
 * (ew_ref_index); nothing is derived from the kernel under test. The clamp and NaN rules are
 * the ones arm_nnfunctions_flt.h documents for the flat kernels: NaN in either operand or
 * from the arithmetic propagates, non-NaN infinities clamp to the bounds.
 *
 * The fuzz case also runs the flat kernel over the materialised broadcast operands and
 * requires the two outputs to agree bit for bit (NaN payload aside), which is the proof
 * that the scalar-broadcast leaves are the flat kernel's arithmetic.
 *
 * Signed zeros: the shipped -Ofast implies -fno-signed-zeros for the library and for this TU
 * alike, so neither a kernel leg nor a floating-point reference is entitled to a particular
 * sign on a zero result. The reference therefore derives the sign of every zero result from
 * the operand bit patterns with integer logic (ew_zero_sign_bits), and the checks demand that
 * sign only when ew_probe_signed_zero() finds the flat kernel itself honouring it on this
 * build; otherwise a zero of either sign is accepted. The rsub leaf is still checked to be
 * scalar - element rather than -(element - scalar) wherever the build lets it show.
 *
 * Subnormals: MVE floating-point instructions run with FZ=1 hard-wired (Armv8-M
 * StandardFPSCRValue), so on a cortex-m55 MVE build f32 subnormal inputs and results flush
 * to zero while the scalar legs (and every f16 leg, FZ16 passing through) keep them. The
 * reference does not guess which leg it faces: ew_probe_flush() runs the flat kernel on a
 * subnormal once and the reference applies the same flush.
 */

#pragma once

#include <arm_nnfunctions.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#define EW_BCAST_OP_SUB 0
#define EW_BCAST_OP_ADD 1
#define EW_BCAST_OP_MUL 2

#ifndef EW_BCAST_FUZZ_ITERS
    #define EW_BCAST_FUZZ_ITERS 2000
#endif

#define EW_CAT_(a, b) a##_##b
#define EW_CAT(a, b) EW_CAT_(a, b)
#define EW_FN(name) EW_CAT(EW_BCAST_PREFIX, name)
#define EW_STR_(x) #x
/* Sign-exact only when the build honours signed zeros (see the file comment). */
#define EW_ASSERT_BITS_OR_ZERO(strict, expected, actual)                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        const ew_bits_t ew_exp_ = (expected);                                                                          \
        const ew_bits_t ew_act_ = (actual);                                                                            \
        if ((strict) || !ew_bits_is_zero(ew_exp_) || !ew_bits_is_zero(ew_act_))                                        \
        {                                                                                                              \
            TEST_ASSERT_EQUAL_HEX32((uint32_t)ew_exp_, (uint32_t)ew_act_);                                             \
        }                                                                                                              \
    } while (0)
#define EW_STRINGIFY(x) EW_STR_(x)

#if EW_BCAST_F16
typedef float16_t ew_t;
typedef uint16_t ew_bits_t;
    #define EW_LANES 8
    #define EW_EXP_MASK 0x7C00u
    #define EW_MANT_MASK 0x03FFu
    #define EW_SIGN_MASK 0x8000u
    #define EW_MIN_NORMAL 6.103515625e-05 /* 2^-14 */
    #define EW_QNAN_BITS 0x7E00u
    #define EW_INF_BITS 0x7C00u
    #define EW_MAX_FINITE_BITS 0x7BFFu
#else
typedef float32_t ew_t;
typedef uint32_t ew_bits_t;
    #define EW_LANES 4
    #define EW_EXP_MASK 0x7F800000u
    #define EW_MANT_MASK 0x007FFFFFu
    #define EW_SIGN_MASK 0x80000000u
    #define EW_MIN_NORMAL 1.17549435082228750797e-38 /* 2^-126 */
    #define EW_QNAN_BITS 0x7FC00000u
    #define EW_INF_BITS 0x7F800000u
    #define EW_MAX_FINITE_BITS 0x7F7FFFFFu
#endif

/* Largest shape the fuzz draws: every NHWC dimension in 1..9. */
#define EW_FUZZ_MAX_DIM 9
#define EW_MAX_ELEMENTS (EW_FUZZ_MAX_DIM * EW_FUZZ_MAX_DIM * EW_FUZZ_MAX_DIM * EW_FUZZ_MAX_DIM)

/* ---------------------------------------------------------------- bit helpers */

static ew_bits_t ew_bits(ew_t x)
{
    ew_bits_t b;
    memcpy(&b, &x, sizeof(b));
    return b;
}

/* The bits pass through a volatile so that a value built here is never a compile-time
 * constant: under this TU's -fno-signed-zeros (implied by the harness's -Ofast) a folded
 * -0.0 is not guaranteed to keep its sign, and under -ffinite-math-only a folded NaN or Inf
 * is not guaranteed to reach the kernel at all. */
static ew_t ew_from_bits(ew_bits_t b)
{
    volatile ew_bits_t staged = b;
    const ew_bits_t bits = staged;
    ew_t x;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static bool ew_bits_is_nan(ew_bits_t b) { return ((b & EW_EXP_MASK) == EW_EXP_MASK) && ((b & EW_MANT_MASK) != 0u); }

static bool ew_bits_is_zero(ew_bits_t b) { return (b & (ew_bits_t)~EW_SIGN_MASK) == 0u; }

static bool ew_bits_is_subnormal(ew_bits_t b) { return ((b & EW_EXP_MASK) == 0u) && ((b & EW_MANT_MASK) != 0u); }

static bool ew_double_is_nan(double d)
{
    /* Bit test rather than d != d: the harness compiles this TU with -fno-finite-math-only,
     * but a standalone -Ofast build would fold the self-compare. */
    uint64_t b;
    memcpy(&b, &d, sizeof(b));
    return ((b & 0x7FF0000000000000ull) == 0x7FF0000000000000ull) && ((b & 0x000FFFFFFFFFFFFFull) != 0ull);
}

static double ew_to_double(ew_t x)
{
#if EW_BCAST_F16
    return (double)(float32_t)x;
#else
    return (double)x;
#endif
}

/* Round a double to the element type. The double is the exact (f16, f32 multiply) or once-
 * rounded (f32 add/sub) result of a single operation on two element-type operands, so the
 * conversion chain below is correctly rounded: 53 >= 2 * 24 + 2 and 24 >= 2 * 11 + 2 make
 * each intermediate rounding innocuous. */
static ew_t ew_from_double(double d)
{
#if EW_BCAST_F16
    return (float16_t)(float32_t)d;
#else
    return (float32_t)d;
#endif
}

/* Sign-magnitude bit pattern mapped onto a monotonic integer, so the distance between two
 * finite values is their ulp distance. */
static int64_t ew_bits_ordinal(ew_bits_t b)
{
    const int64_t mag = (int64_t)(b & (ew_bits_t)~EW_SIGN_MASK);
    return (b & EW_SIGN_MASK) ? -mag : mag;
}

static int64_t ew_ulp_distance(ew_bits_t a, ew_bits_t b)
{
    const int64_t d = ew_bits_ordinal(a) - ew_bits_ordinal(b);
    return d < 0 ? -d : d;
}

/* NumPy-style broadcast index: a dimension of 1 is read at 0 for every output coordinate. */
static int32_t ew_ref_index(const cmsis_nn_dims *dims, int32_t n, int32_t h, int32_t w, int32_t c)
{
    const int32_t nn = (dims->n == 1) ? 0 : n;
    const int32_t hh = (dims->h == 1) ? 0 : h;
    const int32_t ww = (dims->w == 1) ? 0 : w;
    const int32_t cc = (dims->c == 1) ? 0 : c;
    return ((nn * dims->h + hh) * dims->w + ww) * dims->c + cc;
}

static int32_t ew_dims_count(const cmsis_nn_dims *dims) { return dims->n * dims->h * dims->w * dims->c; }

/* ---------------------------------------------------------------- reference */

/* Whether this build flushes subnormals on the flat kernel's path (see the file comment).
 * Probed once, on the kernel the broadcast leaves must match. */
static int ew_flush_state = -1;

static bool ew_probe_flush(void)
{
    if (ew_flush_state < 0)
    {
        ew_t in1[EW_LANES * 2];
        ew_t in2[EW_LANES * 2];
        ew_t out[EW_LANES * 2];
        for (int32_t i = 0; i < EW_LANES * 2; i++)
        {
            in1[i] = ew_from_bits((ew_bits_t)1u); /* smallest subnormal */
#if EW_BCAST_OP == EW_BCAST_OP_MUL
            in2[i] = ew_from_double(1.0);
#else
            in2[i] = ew_from_double(0.0);
#endif
            out[i] = ew_from_double(-1.0);
        }
        const ew_t no_min = ew_from_bits((ew_bits_t)(EW_INF_BITS | EW_SIGN_MASK));
        const ew_t no_max = ew_from_bits(EW_INF_BITS);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, EW_BCAST_FLAT(in1, in2, out, no_min, no_max, EW_LANES * 2));
        ew_flush_state = ew_bits_is_zero(ew_bits(out[0])) ? 1 : 0;
    }
    return ew_flush_state == 1;
}

/* Whether the flat kernel returns a -0 where IEEE 754 requires one on this build; under
 * -fno-signed-zeros the compiler need not, on any leg. Probed once. */
static int ew_signed_zero_state = -1;

static bool ew_probe_signed_zero(void)
{
    if (ew_signed_zero_state < 0)
    {
        ew_t in1[EW_LANES * 2];
        ew_t in2[EW_LANES * 2];
        ew_t out[EW_LANES * 2];
        for (int32_t i = 0; i < EW_LANES * 2; i++)
        {
#if EW_BCAST_OP == EW_BCAST_OP_SUB
            in1[i] = ew_from_bits(EW_SIGN_MASK); /* -0 - +0 = -0 */
            in2[i] = ew_from_bits(0u);
#elif EW_BCAST_OP == EW_BCAST_OP_ADD
            in1[i] = ew_from_bits(EW_SIGN_MASK); /* -0 + -0 = -0 */
            in2[i] = ew_from_bits(EW_SIGN_MASK);
#else
            in1[i] = ew_from_double(-1.0); /* -1 * +0 = -0 */
            in2[i] = ew_from_bits(0u);
#endif
            out[i] = ew_from_double(1.0);
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                          EW_BCAST_FLAT(in1, in2, out, ew_from_double(-6.0), ew_from_double(6.0), EW_LANES * 2));
        ew_signed_zero_state = (ew_bits(out[0]) == EW_SIGN_MASK) ? 1 : 0;
    }
    return ew_signed_zero_state == 1;
}

static ew_bits_t ew_flush_bits(ew_bits_t b, bool flush)
{
    return (flush && ew_bits_is_subnormal(b)) ? (ew_bits_t)(b & EW_SIGN_MASK) : b;
}

static double ew_op(double a, double b)
{
#if EW_BCAST_OP == EW_BCAST_OP_SUB
    return a - b;
#elif EW_BCAST_OP == EW_BCAST_OP_ADD
    return a + b;
#else
    return a * b;
#endif
}

/* IEEE 754 sign of an exact zero result of a op b (round-to-nearest), from the operand bits
 * alone so that this TU's own -fno-signed-zeros cannot touch it: a difference or sum of
 * non-zero operands that cancels is +0; a difference of zeros is -0 only for -0 - +0; a sum of
 * zeros is -0 only for -0 + -0; a zero product carries the XOR of the signs. */
static ew_bits_t ew_zero_sign_bits(ew_bits_t a_bits, ew_bits_t b_bits)
{
#if EW_BCAST_OP == EW_BCAST_OP_SUB
    if (ew_bits_is_zero(a_bits) && ew_bits_is_zero(b_bits))
    {
        return (ew_bits_t)(a_bits & (ew_bits_t)~b_bits & EW_SIGN_MASK);
    }
    return 0u;
#elif EW_BCAST_OP == EW_BCAST_OP_ADD
    if (ew_bits_is_zero(a_bits) && ew_bits_is_zero(b_bits))
    {
        return (ew_bits_t)(a_bits & b_bits & EW_SIGN_MASK);
    }
    return 0u;
#else
    return (ew_bits_t)((a_bits ^ b_bits) & EW_SIGN_MASK);
#endif
}

/* Expected output bits for one element: a op b, rounded to the element type, then clamped
 * with NaN propagating. A NaN expectation is returned as EW_QNAN_BITS and matched on
 * NaN-ness only. */
static ew_bits_t ew_expected_bits(ew_t a, ew_t b, ew_t min_v, ew_t max_v, bool flush)
{
    const ew_bits_t a_bits = ew_flush_bits(ew_bits(a), flush);
    const ew_bits_t b_bits = ew_flush_bits(ew_bits(b), flush);
    const ew_bits_t min_bits = ew_flush_bits(ew_bits(min_v), flush);
    const ew_bits_t max_bits = ew_flush_bits(ew_bits(max_v), flush);

    if (ew_bits_is_nan(a_bits) || ew_bits_is_nan(b_bits))
    {
        return EW_QNAN_BITS;
    }

    const double d = ew_op(ew_to_double(ew_from_bits(a_bits)), ew_to_double(ew_from_bits(b_bits)));
    if (ew_double_is_nan(d))
    {
        return EW_QNAN_BITS; /* Inf - Inf, Inf + -Inf, 0 * Inf */
    }

    ew_bits_t r_bits;
    if (flush && d != 0.0 && d > -EW_MIN_NORMAL && d < EW_MIN_NORMAL)
    {
        /* Tiny before rounding: FZ flushes the result to a zero of the same sign. d is not
         * zero here, so its sign is a plain compare. */
        r_bits = (d < 0.0) ? EW_SIGN_MASK : 0u;
    }
    else
    {
        r_bits = ew_bits(ew_from_double(d));
        if (ew_bits_is_zero(r_bits))
        {
            r_bits = ew_zero_sign_bits(a_bits, b_bits);
        }
    }

    const double r = ew_to_double(ew_from_bits(r_bits));
    if (r < ew_to_double(ew_from_bits(min_bits)))
    {
        return min_bits;
    }
    if (r > ew_to_double(ew_from_bits(max_bits)))
    {
        return max_bits;
    }
    return r_bits;
}

/* Zero leniency: a zero of the wrong sign is accepted when the build does not honour signed
 * zeros (ew_probe_signed_zero), and at a tie against a zero bound, where the scalar clamp
 * keeps the operand while VMAXNM / VMINNM pick by sign. Both answers compare equal to zero
 * and the flat kernels' contract pins neither. */
static bool ew_zero_tie(ew_bits_t expected, ew_bits_t actual, ew_t min_v, ew_t max_v)
{
    return ew_bits_is_zero(expected) && ew_bits_is_zero(actual) &&
        (!ew_probe_signed_zero() || ew_bits_is_zero(ew_bits(min_v)) || ew_bits_is_zero(ew_bits(max_v)));
}

static bool ew_bits_match(ew_bits_t expected, ew_bits_t actual, ew_t min_v, ew_t max_v)
{
    if (ew_bits_is_nan(expected))
    {
        return ew_bits_is_nan(actual);
    }
    return expected == actual || ew_zero_tie(expected, actual, min_v, max_v);
}

/* ---------------------------------------------------------------- checker */

static ew_t ew_in1[EW_MAX_ELEMENTS];
static ew_t ew_in2[EW_MAX_ELEMENTS];
static ew_t ew_out[EW_MAX_ELEMENTS];

#define EW_SENTINEL_BITS ((ew_bits_t)(EW_QNAN_BITS | 0x55u))

static void ew_fill_sentinel(ew_t *out, int32_t count)
{
    for (int32_t i = 0; i < count; i++)
    {
        out[i] = ew_from_bits(EW_SENTINEL_BITS);
    }
}

/* Run the kernel over ew_in1 / ew_in2 into ew_out and check every element against the
 * reference walk. Returns the number of zero-tie leniencies used. */
static int32_t ew_check_case(const cmsis_nn_dims *dims_1,
                             const cmsis_nn_dims *dims_2,
                             const cmsis_nn_dims *dims_out,
                             ew_t min_v,
                             ew_t max_v)
{
    const bool flush = ew_probe_flush();
    const int32_t count = ew_dims_count(dims_out);
    int32_t ties = 0;
    char msg[320];

    TEST_ASSERT_TRUE(count <= EW_MAX_ELEMENTS);
    ew_fill_sentinel(ew_out, count);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      EW_BCAST_KERNEL(ew_in1, dims_1, ew_in2, dims_2, ew_out, dims_out, min_v, max_v));

    for (int32_t n = 0; n < dims_out->n; n++)
    {
        for (int32_t h = 0; h < dims_out->h; h++)
        {
            for (int32_t w = 0; w < dims_out->w; w++)
            {
                for (int32_t c = 0; c < dims_out->c; c++)
                {
                    const ew_t a = ew_in1[ew_ref_index(dims_1, n, h, w, c)];
                    const ew_t b = ew_in2[ew_ref_index(dims_2, n, h, w, c)];
                    const int32_t o = ((n * dims_out->h + h) * dims_out->w + w) * dims_out->c + c;
                    const ew_bits_t expected = ew_expected_bits(a, b, min_v, max_v, flush);
                    const ew_bits_t actual = ew_bits(ew_out[o]);
                    if (!ew_bits_match(expected, actual, min_v, max_v))
                    {
                        snprintf(
                            msg,
                            sizeof(msg),
                            "[%ld,%ld,%ld,%ld] op [%ld,%ld,%ld,%ld] out[%ld] (n%ld h%ld w%ld c%ld): a 0x%lx b 0x%lx "
                            "expected 0x%lx got 0x%lx",
                            (long)dims_1->n,
                            (long)dims_1->h,
                            (long)dims_1->w,
                            (long)dims_1->c,
                            (long)dims_2->n,
                            (long)dims_2->h,
                            (long)dims_2->w,
                            (long)dims_2->c,
                            (long)o,
                            (long)n,
                            (long)h,
                            (long)w,
                            (long)c,
                            (unsigned long)ew_bits(a),
                            (unsigned long)ew_bits(b),
                            (unsigned long)expected,
                            (unsigned long)actual);
                        TEST_FAIL_MESSAGE(msg);
                    }
                    if (expected != actual && !ew_bits_is_nan(expected))
                    {
                        ties++;
                    }
                }
            }
        }
    }
    return ties;
}

static void ew_expect_arg_error(const cmsis_nn_dims *dims_1,
                                const cmsis_nn_dims *dims_2,
                                const cmsis_nn_dims *dims_out,
                                int32_t out_count)
{
    const ew_t min_v = ew_from_double(-6.0);
    const ew_t max_v = ew_from_double(6.0);
    ew_fill_sentinel(ew_out, out_count);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      EW_BCAST_KERNEL(ew_in1, dims_1, ew_in2, dims_2, ew_out, dims_out, min_v, max_v));
    for (int32_t i = 0; i < out_count; i++)
    {
        TEST_ASSERT_EQUAL_MESSAGE(EW_SENTINEL_BITS, ew_bits(ew_out[i]), "output written on ARG_ERROR");
    }
}

/* Deterministic, mildly irregular test data: every element distinct, both signs, magnitudes
 * that make the -6..6 clamp bite on some elements. */
static void ew_fill_pattern(ew_t *dst, int32_t count, int32_t salt)
{
    for (int32_t i = 0; i < count; i++)
    {
        const int32_t k = (i * 37 + salt * 11) % 97;
        dst[i] = ew_from_double((double)(k - 48) * 0.125 + (double)((i + salt) % 5) * 0.0625);
    }
}

static void ew_fill_inputs(const cmsis_nn_dims *dims_1, const cmsis_nn_dims *dims_2)
{
    ew_fill_pattern(ew_in1, ew_dims_count(dims_1), 1);
    ew_fill_pattern(ew_in2, ew_dims_count(dims_2), 2);
}

static void ew_run_shapes(const cmsis_nn_dims *dims_1, const cmsis_nn_dims *dims_2, const cmsis_nn_dims *dims_out)
{
    ew_fill_inputs(dims_1, dims_2);
    ew_check_case(dims_1, dims_2, dims_out, ew_from_double(-6.0), ew_from_double(6.0));
}

/* ---------------------------------------------------------------- shape cases */

/* C = 13 is not a multiple of 4 or 8, so every contiguous run ends in a predicated tail. */
#define EW_C 13

void EW_FN(identical_shapes)(void)
{
    const cmsis_nn_dims dims = {2, 3, 4, EW_C};
    ew_run_shapes(&dims, &dims, &dims);
}

void EW_FN(per_channel_right)(void)
{
    const cmsis_nn_dims dims_1 = {1, 3, 5, EW_C};
    const cmsis_nn_dims dims_2 = {1, 1, 1, EW_C};
    ew_run_shapes(&dims_1, &dims_2, &dims_1);
}

void EW_FN(per_channel_left)(void)
{
    const cmsis_nn_dims dims_1 = {1, 1, 1, EW_C};
    const cmsis_nn_dims dims_2 = {1, 3, 5, EW_C};
    ew_run_shapes(&dims_1, &dims_2, &dims_2);
}

void EW_FN(scalar_right)(void)
{
    const cmsis_nn_dims dims_1 = {1, 3, 5, EW_C};
    const cmsis_nn_dims dims_2 = {1, 1, 1, 1};
    ew_run_shapes(&dims_1, &dims_2, &dims_1);
}

void EW_FN(scalar_left)(void)
{
    const cmsis_nn_dims dims_1 = {1, 1, 1, 1};
    const cmsis_nn_dims dims_2 = {1, 3, 5, EW_C};
    ew_run_shapes(&dims_1, &dims_2, &dims_2);
}

void EW_FN(both_broadcast)(void)
{
    const cmsis_nn_dims dims_1 = {1, 3, 1, EW_C};
    const cmsis_nn_dims dims_2 = {1, 1, 5, EW_C};
    const cmsis_nn_dims dims_out = {1, 3, 5, EW_C};
    ew_run_shapes(&dims_1, &dims_2, &dims_out);
    ew_run_shapes(&dims_2, &dims_1, &dims_out);
}

void EW_FN(batch)(void)
{
    const cmsis_nn_dims dims_1 = {3, 1, 1, EW_C};
    const cmsis_nn_dims dims_2 = {1, 4, 2, EW_C};
    const cmsis_nn_dims dims_out = {3, 4, 2, EW_C};
    ew_run_shapes(&dims_1, &dims_2, &dims_out);
    ew_run_shapes(&dims_2, &dims_1, &dims_out);
    /* Same batch operand against a full tensor: FULL runs per batch on the left, per-row scalar
     * runs on the right. */
    const cmsis_nn_dims dims_full = {3, 4, 2, EW_C};
    ew_run_shapes(&dims_1, &dims_full, &dims_out);
    const cmsis_nn_dims dims_row = {3, 4, 2, 1};
    ew_run_shapes(&dims_full, &dims_row, &dims_out);
}

/* Channel counts on both sides of the lane count, including runs shorter than one vector
 * (the predicated tail is then the whole run) and the C = 1 operand, which turns each row
 * into a scalar leaf call of length W * C. */
void EW_FN(tail_lengths)(void)
{
    static const int32_t lengths[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33};
    for (uint32_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++)
    {
        const int32_t c = lengths[i];
        const cmsis_nn_dims dims_full = {1, 2, 3, c};
        const cmsis_nn_dims dims_chan = {1, 1, 1, c};
        const cmsis_nn_dims dims_one = {1, 1, 1, 1};
        const cmsis_nn_dims dims_col = {1, 2, 3, 1};
        ew_run_shapes(&dims_full, &dims_chan, &dims_full);
        ew_run_shapes(&dims_chan, &dims_full, &dims_full);
        ew_run_shapes(&dims_full, &dims_one, &dims_full);
        ew_run_shapes(&dims_one, &dims_full, &dims_full);
        ew_run_shapes(&dims_full, &dims_col, &dims_full);
        ew_run_shapes(&dims_col, &dims_full, &dims_full);
    }
}

void EW_FN(tight_clamp)(void)
{
    const cmsis_nn_dims dims_1 = {1, 2, 3, EW_C};
    const cmsis_nn_dims dims_2 = {1, 1, 1, EW_C};
    ew_fill_inputs(&dims_1, &dims_2);
    ew_check_case(&dims_1, &dims_2, &dims_1, ew_from_double(-0.5), ew_from_double(0.5));
    ew_check_case(&dims_2, &dims_1, &dims_1, ew_from_double(-0.5), ew_from_double(0.5));
    /* min == max: every element becomes the bound. */
    ew_check_case(&dims_1, &dims_2, &dims_1, ew_from_double(0.25), ew_from_double(0.25));
    /* Bounds far outside the data: nothing clamps. */
    ew_check_case(&dims_2, &dims_1, &dims_1, ew_from_double(-1.0e4), ew_from_double(1.0e4));
}

/* ---------------------------------------------------------------- non-finite operands */

static ew_t ew_inf(bool negative) { return ew_from_bits((ew_bits_t)(EW_INF_BITS | (negative ? EW_SIGN_MASK : 0u))); }

static ew_t ew_nan(void) { return ew_from_bits(EW_QNAN_BITS); }

/* NaN and Inf in each operand, as a vector element and as a broadcast scalar, against the
 * contract in arm_nnfunctions_flt.h: NaN in either operand (or Inf - Inf, 0 * Inf) propagates
 * through the clamp; a non-NaN infinity clamps to the bound. Checked both element by element
 * against the reference walk and by explicit expectation here. */
void EW_FN(nan_inf)(void)
{
    const cmsis_nn_dims dims_vec = {1, 1, 1, 8};
    const cmsis_nn_dims dims_one = {1, 1, 1, 1};
    const ew_t min_v = ew_from_double(-6.0);
    const ew_t max_v = ew_from_double(6.0);

    /* Vector against vector: NaN and Inf on each side. */
    const ew_t vec_a[8] = {ew_nan(),
                           ew_from_double(1.0),
                           ew_inf(false),
                           ew_inf(true),
                           ew_inf(false),
                           ew_from_double(2.0),
                           ew_from_double(0.0),
                           ew_from_double(-3.0)};
    const ew_t vec_b[8] = {ew_from_double(1.0),
                           ew_nan(),
                           ew_from_double(1.0),
                           ew_from_double(1.0),
                           ew_inf(false),
                           ew_inf(true),
                           ew_inf(false),
                           ew_from_double(0.5)};
    memcpy(ew_in1, vec_a, sizeof(vec_a));
    memcpy(ew_in2, vec_b, sizeof(vec_b));
    ew_check_case(&dims_vec, &dims_vec, &dims_vec, min_v, max_v);

    /* Explicit contract on the vector result: lanes 0 and 1 carry a NaN operand, lane 2 and
     * 3 an infinite left operand, lane 4 Inf op Inf. */
    TEST_ASSERT_TRUE_MESSAGE(ew_bits_is_nan(ew_bits(ew_out[0])), "NaN op x must be NaN");
    TEST_ASSERT_TRUE_MESSAGE(ew_bits_is_nan(ew_bits(ew_out[1])), "x op NaN must be NaN");
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(max_v), ew_bits(ew_out[2]), "+Inf op 1 must clamp to max");
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(min_v), ew_bits(ew_out[3]), "-Inf op 1 must clamp to min");
#if EW_BCAST_OP == EW_BCAST_OP_SUB
    TEST_ASSERT_TRUE_MESSAGE(ew_bits_is_nan(ew_bits(ew_out[4])), "Inf - Inf must be NaN");
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(max_v), ew_bits(ew_out[5]), "2 - -Inf must clamp to max");
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(min_v), ew_bits(ew_out[6]), "0 - Inf must clamp to min");
#elif EW_BCAST_OP == EW_BCAST_OP_ADD
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(max_v), ew_bits(ew_out[4]), "Inf + Inf must clamp to max");
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(min_v), ew_bits(ew_out[5]), "2 + -Inf must clamp to min");
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(max_v), ew_bits(ew_out[6]), "0 + Inf must clamp to max");
#else
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(max_v), ew_bits(ew_out[4]), "Inf * Inf must clamp to max");
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(min_v), ew_bits(ew_out[5]), "2 * -Inf must clamp to min");
    TEST_ASSERT_TRUE_MESSAGE(ew_bits_is_nan(ew_bits(ew_out[6])), "0 * Inf must be NaN");
#endif

    /* A NaN broadcast scalar on either side poisons every element. */
    ew_fill_pattern(ew_in1, 8, 3);
    ew_in2[0] = ew_nan();
    ew_check_case(&dims_vec, &dims_one, &dims_vec, min_v, max_v);
    for (int32_t i = 0; i < 8; i++)
    {
        TEST_ASSERT_TRUE_MESSAGE(ew_bits_is_nan(ew_bits(ew_out[i])), "vector op NaN scalar must be NaN");
    }
    ew_in1[0] = ew_nan();
    ew_fill_pattern(ew_in2, 8, 4);
    ew_check_case(&dims_one, &dims_vec, &dims_vec, min_v, max_v);
    for (int32_t i = 0; i < 8; i++)
    {
        TEST_ASSERT_TRUE_MESSAGE(ew_bits_is_nan(ew_bits(ew_out[i])), "NaN scalar op vector must be NaN");
    }

    /* An infinite broadcast scalar on either side against finite data clamps to a bound. */
    ew_fill_pattern(ew_in1, 8, 5);
    ew_in2[0] = ew_inf(false);
    ew_check_case(&dims_vec, &dims_one, &dims_vec, min_v, max_v);
    ew_in1[0] = ew_inf(true);
    ew_fill_pattern(ew_in2, 8, 6);
    ew_check_case(&dims_one, &dims_vec, &dims_vec, min_v, max_v);
    for (int32_t i = 0; i < 8; i++)
    {
        const ew_bits_t o = ew_bits(ew_out[i]);
        TEST_ASSERT_TRUE_MESSAGE(o == ew_bits(min_v) || o == ew_bits(max_v), "-Inf scalar op finite must clamp");
    }

    /* Infinite bounds are the no-clamp idiom: infinities pass through, NaN still propagates. */
    memcpy(ew_in1, vec_a, sizeof(vec_a));
    memcpy(ew_in2, vec_b, sizeof(vec_b));
    ew_check_case(&dims_vec, &dims_vec, &dims_vec, ew_inf(true), ew_inf(false));
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(ew_inf(false)), ew_bits(ew_out[2]), "+Inf must pass an infinite clamp");
    TEST_ASSERT_EQUAL_MESSAGE(ew_bits(ew_inf(true)), ew_bits(ew_out[3]), "-Inf must pass an infinite clamp");
}

/* Signed zero is checked bit for bit with non-zero bounds, where scalar and MVE clamps agree.
 * For subtract the scalar-left leaf must compute scalar - element as written: +0 - +0 is +0,
 * while -(+0 - +0) would be -0. */
void EW_FN(signed_zero)(void)
{
    if (!ew_probe_signed_zero())
    {
        /* Still every value is checked against the reference; only the sign of a zero is not. */
        UnityPrint("signed zeros not honoured by this build, checking magnitudes only");
        UNITY_PRINT_EOL();
    }
    const bool strict = ew_probe_signed_zero();
    const cmsis_nn_dims dims_vec = {1, 1, 1, 6};
    const cmsis_nn_dims dims_one = {1, 1, 1, 1};
    const ew_t min_v = ew_from_double(-6.0);
    const ew_t max_v = ew_from_double(6.0);
    const ew_t pz = ew_from_bits(0u);
    const ew_t nz = ew_from_bits(EW_SIGN_MASK);
    const ew_t vec[6] = {pz, nz, ew_from_double(1.0), ew_from_double(-1.0), pz, nz};

    for (int32_t s = 0; s < 2; s++)
    {
        const ew_t scalar = s ? nz : pz;

        /* scalar op vector */
        ew_in1[0] = scalar;
        memcpy(ew_in2, vec, sizeof(vec));
        ew_check_case(&dims_one, &dims_vec, &dims_vec, min_v, max_v);
#if EW_BCAST_OP == EW_BCAST_OP_SUB
        /* +0 - +0 = +0, +0 - -0 = +0, -0 - +0 = -0, -0 - -0 = +0 */
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(s ? EW_SIGN_MASK : 0u), ew_bits(ew_out[0]));
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(0u), ew_bits(ew_out[1]));
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(ew_bits(ew_from_double(-1.0))), ew_bits(ew_out[2]));
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(ew_bits(ew_from_double(1.0))), ew_bits(ew_out[3]));
#elif EW_BCAST_OP == EW_BCAST_OP_ADD
        /* only -0 + -0 is -0 */
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(0u), ew_bits(ew_out[0]));
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(s ? EW_SIGN_MASK : 0u), ew_bits(ew_out[1]));
#else
        /* the sign of a zero product is the XOR of the operand signs */
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(s ? EW_SIGN_MASK : 0u), ew_bits(ew_out[0]));
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(s ? 0u : EW_SIGN_MASK), ew_bits(ew_out[1]));
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(s ? EW_SIGN_MASK : 0u), ew_bits(ew_out[2]));
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(s ? 0u : EW_SIGN_MASK), ew_bits(ew_out[3]));
#endif

        /* vector op scalar */
        memcpy(ew_in1, vec, sizeof(vec));
        ew_in2[0] = scalar;
        ew_check_case(&dims_vec, &dims_one, &dims_vec, min_v, max_v);
#if EW_BCAST_OP == EW_BCAST_OP_SUB
        /* +0 - s: +0 for s = +0 or -0; -0 - +0 = -0, -0 - -0 = +0 */
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(0u), ew_bits(ew_out[0]));
        EW_ASSERT_BITS_OR_ZERO(strict, (ew_bits_t)(s ? 0u : EW_SIGN_MASK), ew_bits(ew_out[1]));
#endif

        /* vector op vector, identical shapes */
        memcpy(ew_in1, vec, sizeof(vec));
        memcpy(ew_in2, vec, sizeof(vec));
        ew_check_case(&dims_vec, &dims_vec, &dims_vec, min_v, max_v);
    }
}

/* ---------------------------------------------------------------- argument errors */

void EW_FN(arg_error)(void)
{
    const cmsis_nn_dims dims_h2 = {1, 2, 1, EW_C};
    const cmsis_nn_dims dims_h3 = {1, 3, 1, EW_C};
    const cmsis_nn_dims dims_chan = {1, 1, 1, EW_C};
    const cmsis_nn_dims dims_zero = {1, 0, 1, EW_C};
    const cmsis_nn_dims dims_neg = {1, 2, -1, EW_C};
    const ew_t min_v = ew_from_double(-6.0);
    const ew_t max_v = ew_from_double(6.0);
    const int32_t count = 3 * EW_C;

    ew_fill_inputs(&dims_h3, &dims_h3);

    /* h = 2 against h = 3 is not broadcastable, in either order. */
    ew_expect_arg_error(&dims_h2, &dims_h3, &dims_h3, count);
    ew_expect_arg_error(&dims_h3, &dims_h2, &dims_h3, count);
    /* Output shape must be the broadcast shape, not an input shape. */
    ew_expect_arg_error(&dims_h3, &dims_chan, &dims_chan, count);
    ew_expect_arg_error(&dims_chan, &dims_h3, &dims_chan, count);
    /* A dimension of 0 or below is rejected rather than treated as empty. */
    ew_expect_arg_error(&dims_zero, &dims_h3, &dims_h3, count);
    ew_expect_arg_error(&dims_h3, &dims_neg, &dims_h3, count);

    /* NULL pointers. */
    ew_fill_sentinel(ew_out, count);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      EW_BCAST_KERNEL(ew_in1, NULL, ew_in2, &dims_h3, ew_out, &dims_h3, min_v, max_v));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      EW_BCAST_KERNEL(ew_in1, &dims_h3, ew_in2, NULL, ew_out, &dims_h3, min_v, max_v));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      EW_BCAST_KERNEL(ew_in1, &dims_h3, ew_in2, &dims_h3, ew_out, NULL, min_v, max_v));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      EW_BCAST_KERNEL(NULL, &dims_h3, ew_in2, &dims_h3, ew_out, &dims_h3, min_v, max_v));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      EW_BCAST_KERNEL(ew_in1, &dims_h3, NULL, &dims_h3, ew_out, &dims_h3, min_v, max_v));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      EW_BCAST_KERNEL(ew_in1, &dims_h3, ew_in2, &dims_h3, NULL, &dims_h3, min_v, max_v));
    for (int32_t i = 0; i < count; i++)
    {
        TEST_ASSERT_EQUAL_MESSAGE(EW_SENTINEL_BITS, ew_bits(ew_out[i]), "output written on ARG_ERROR");
    }
}

/* ---------------------------------------------------------------- differential fuzz */

static uint32_t ew_rng_state;

static uint32_t ew_rng(void)
{
    /* xorshift32; seeded per suite so every run draws the same sequence. */
    uint32_t x = ew_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ew_rng_state = x;
    return x;
}

static int32_t ew_rng_below(uint32_t n) { return (int32_t)(ew_rng() % n); }

static double ew_rng_uniform(double lo, double hi) { return lo + (hi - lo) * ((double)(ew_rng() >> 8) / 16777216.0); }

/* One random element, drawn from classes that reach every path of the arithmetic and the
 * clamp: ordinary values, exact small integers (ties and zeros), signed zeros, infinities,
 * NaN with a random payload, subnormals, values near the overflow threshold, and arbitrary
 * finite bit patterns across the whole exponent range. */
static ew_t ew_rng_value(void)
{
    const int32_t cls = ew_rng_below(100);
    if (cls < 55)
    {
        return ew_from_double(ew_rng_uniform(-8.0, 8.0));
    }
    if (cls < 70)
    {
        return ew_from_double((double)(ew_rng_below(9) - 4));
    }
    if (cls < 75)
    {
        return ew_from_bits((ew_rng() & 1u) ? EW_SIGN_MASK : 0u);
    }
    if (cls < 80)
    {
        return ew_inf((ew_rng() & 1u) != 0u);
    }
    if (cls < 85)
    {
        const ew_bits_t payload = (ew_bits_t)(ew_rng() & EW_MANT_MASK);
        return ew_from_bits(
            (ew_bits_t)(EW_EXP_MASK | (payload ? payload : 1u) | ((ew_rng() & 1u) ? EW_SIGN_MASK : 0u)));
    }
    if (cls < 90)
    {
        const ew_bits_t mant = (ew_bits_t)(ew_rng() & EW_MANT_MASK);
        return ew_from_bits((ew_bits_t)((mant ? mant : 1u) | ((ew_rng() & 1u) ? EW_SIGN_MASK : 0u)));
    }
    if (cls < 95)
    {
        return ew_from_bits(
            (ew_bits_t)((EW_MAX_FINITE_BITS - (ew_bits_t)ew_rng_below(64)) | ((ew_rng() & 1u) ? EW_SIGN_MASK : 0u)));
    }
    /* Any finite pattern: exponent below all-ones. */
    ew_bits_t b = (ew_bits_t)ew_rng();
    if ((b & EW_EXP_MASK) == EW_EXP_MASK)
    {
        b = (ew_bits_t)(b ^ (EW_EXP_MASK & (ew_bits_t) ~(EW_EXP_MASK >> 1))); /* clear the top exponent bit */
    }
    return ew_from_bits(b);
}

static void ew_rng_clamp(ew_t *min_v, ew_t *max_v)
{
    const int32_t cls = ew_rng_below(100);
    if (cls < 35)
    {
        *min_v = ew_inf(true);
        *max_v = ew_inf(false);
    }
    else if (cls < 55)
    {
        *min_v = ew_from_double(-6.0);
        *max_v = ew_from_double(6.0);
    }
    else if (cls < 75)
    {
        double a = ew_rng_uniform(-8.0, 8.0);
        double b = ew_rng_uniform(-8.0, 8.0);
        if (a > b)
        {
            const double t = a;
            a = b;
            b = t;
        }
        *min_v = ew_from_double(a);
        *max_v = ew_from_double(b);
    }
    else if (cls < 85)
    {
        /* RELU-like, with a zero bound of either sign. */
        *min_v = ew_from_bits((ew_rng() & 1u) ? EW_SIGN_MASK : 0u);
        *max_v = ew_from_double(6.0);
    }
    else if (cls < 90)
    {
        *min_v = ew_from_double(-6.0);
        *max_v = ew_from_bits((ew_rng() & 1u) ? EW_SIGN_MASK : 0u);
    }
    else if (cls < 95)
    {
        const double t = ew_rng_uniform(-2.0, 2.0);
        *min_v = ew_from_double(t);
        *max_v = ew_from_double(t);
    }
    else
    {
        /* Subnormal bounds: only a flushed or subnormal result lands inside them. */
        const ew_bits_t m = (ew_bits_t)((ew_rng() & EW_MANT_MASK) | 1u);
        *min_v = ew_from_bits((ew_bits_t)(m | EW_SIGN_MASK));
        *max_v = ew_from_bits(m);
    }
}

/* Draw a broadcast-compatible shape pair: each NHWC dimension is 1..9 for the output, and
 * each input independently either keeps it or broadcasts it (dimension 1). */
static void ew_rng_shapes(cmsis_nn_dims *dims_1, cmsis_nn_dims *dims_2, cmsis_nn_dims *dims_out)
{
    int32_t *d1 = &dims_1->n;
    int32_t *d2 = &dims_2->n;
    int32_t *dout = &dims_out->n;
    for (int32_t k = 0; k < 4; k++)
    {
        const int32_t d = 1 + ew_rng_below(EW_FUZZ_MAX_DIM);
        const int32_t b1 = ew_rng_below(100) < 35;
        const int32_t b2 = ew_rng_below(100) < 35;
        d1[k] = b1 ? 1 : d;
        d2[k] = b2 ? 1 : d;
        dout[k] = (d1[k] > d2[k]) ? d1[k] : d2[k];
    }
}

static ew_t ew_flat1[EW_MAX_ELEMENTS];
static ew_t ew_flat2[EW_MAX_ELEMENTS];
static ew_t ew_flat_out[EW_MAX_ELEMENTS];

static void ew_print_stat(const char *name, int64_t value)
{
    UnityPrint(name);
    UnityPrint("=");
    UnityPrintNumber((UNITY_INT)value);
    UNITY_PRINT_EOL();
}

/* EW_BCAST_FUZZ_ITERS random shape pairs (about one in ten made invalid), random data and
 * random clamps, each output checked bit for bit against the double reference and against
 * the flat kernel on the materialised broadcast operands. Statistics go to the Unity output. */
void EW_FN(fuzz)(void)
{
    const bool flush = ew_probe_flush();
    int64_t shapes = 0, invalid = 0, elements = 0, ties = 0;
    int64_t max_ulp_ref = 0, max_ulp_flat = 0;
    char msg[320];

    ew_rng_state = 0x415A5A00u ^ ((uint32_t)EW_BCAST_OP << 8) ^ (uint32_t)EW_BCAST_F16;

    for (int32_t iter = 0; iter < EW_BCAST_FUZZ_ITERS; iter++)
    {
        cmsis_nn_dims dims_1, dims_2, dims_out;
        ew_t min_v, max_v;

        ew_rng_shapes(&dims_1, &dims_2, &dims_out);
        ew_rng_clamp(&min_v, &max_v);
        shapes++;

        if (ew_rng_below(10) == 0)
        {
            /* Make the pair invalid: incompatible dimension, wrong output shape, or a zero. */
            int32_t *d1 = &dims_1.n;
            int32_t *d2 = &dims_2.n;
            int32_t *dout = &dims_out.n;
            const int32_t k = ew_rng_below(4);
            switch (ew_rng_below(3))
            {
            case 0:
                d1[k] = 2 + ew_rng_below(3);
                d2[k] = d1[k] + 1;
                dout[k] = d2[k];
                break;
            case 1:
                d1[k] = 2 + ew_rng_below(3);
                d2[k] = 1;
                dout[k] = 1;
                break;
            default:
                d2[k] = 0;
                break;
            }
            invalid++;
            ew_expect_arg_error(&dims_1, &dims_2, &dims_out, ew_dims_count(&dims_out) > 0 ? 8 : 0);
            continue;
        }

        const int32_t count_1 = ew_dims_count(&dims_1);
        const int32_t count_2 = ew_dims_count(&dims_2);
        const int32_t count = ew_dims_count(&dims_out);
        for (int32_t i = 0; i < count_1; i++)
        {
            ew_in1[i] = ew_rng_value();
        }
        for (int32_t i = 0; i < count_2; i++)
        {
            ew_in2[i] = ew_rng_value();
        }

        ties += ew_check_case(&dims_1, &dims_2, &dims_out, min_v, max_v);
        elements += count;

        /* Flat kernel over the materialised operands must agree bit for bit. */
        for (int32_t n = 0; n < dims_out.n; n++)
        {
            for (int32_t h = 0; h < dims_out.h; h++)
            {
                for (int32_t w = 0; w < dims_out.w; w++)
                {
                    for (int32_t c = 0; c < dims_out.c; c++)
                    {
                        const int32_t o = ((n * dims_out.h + h) * dims_out.w + w) * dims_out.c + c;
                        ew_flat1[o] = ew_in1[ew_ref_index(&dims_1, n, h, w, c)];
                        ew_flat2[o] = ew_in2[ew_ref_index(&dims_2, n, h, w, c)];
                    }
                }
            }
        }
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, EW_BCAST_FLAT(ew_flat1, ew_flat2, ew_flat_out, min_v, max_v, count));
        for (int32_t o = 0; o < count; o++)
        {
            const ew_bits_t bcast = ew_bits(ew_out[o]);
            const ew_bits_t flat = ew_bits(ew_flat_out[o]);
            const ew_bits_t expected = ew_expected_bits(ew_flat1[o], ew_flat2[o], min_v, max_v, flush);
            if (ew_bits_is_nan(flat) || ew_bits_is_nan(bcast))
            {
                if (!(ew_bits_is_nan(flat) && ew_bits_is_nan(bcast)))
                {
                    snprintf(msg,
                             sizeof(msg),
                             "iter %ld out[%ld]: broadcast 0x%lx vs flat 0x%lx",
                             (long)iter,
                             (long)o,
                             (unsigned long)bcast,
                             (unsigned long)flat);
                    TEST_FAIL_MESSAGE(msg);
                }
                continue;
            }
            const int64_t ulp_flat = ew_ulp_distance(bcast, flat);
            if (ulp_flat > max_ulp_flat)
            {
                max_ulp_flat = ulp_flat;
            }
            if (bcast != flat && !ew_zero_tie(flat, bcast, min_v, max_v))
            {
                snprintf(msg,
                         sizeof(msg),
                         "iter %ld out[%ld]: broadcast 0x%lx vs flat 0x%lx",
                         (long)iter,
                         (long)o,
                         (unsigned long)bcast,
                         (unsigned long)flat);
                TEST_FAIL_MESSAGE(msg);
            }
            if (!ew_bits_is_nan(expected))
            {
                const int64_t ulp_ref = ew_ulp_distance(bcast, expected);
                if (ulp_ref > max_ulp_ref)
                {
                    max_ulp_ref = ulp_ref;
                }
            }
        }
    }

    UnityPrint("fuzz ");
    UnityPrint(EW_STRINGIFY(EW_BCAST_PREFIX));
    UnityPrint(" flush_subnormals=");
    UnityPrintNumber((UNITY_INT)(flush ? 1 : 0));
    UnityPrint(" signed_zero_honoured=");
    UnityPrintNumber((UNITY_INT)(ew_probe_signed_zero() ? 1 : 0));
    UNITY_PRINT_EOL();
    ew_print_stat("  shapes_drawn", shapes);
    ew_print_stat("  invalid_pairs", invalid);
    ew_print_stat("  elements_checked", elements);
    ew_print_stat("  zero_tie_leniencies", ties);
    ew_print_stat("  max_ulp_vs_reference", max_ulp_ref);
    ew_print_stat("  max_ulp_vs_flat", max_ulp_flat);

    TEST_ASSERT_EQUAL_INT(0, (int)max_ulp_flat);
    /* The reference is exact, and a zero tie is zero ulps from either zero. */
    TEST_ASSERT_EQUAL_INT(0, (int)max_ulp_ref);
    TEST_ASSERT_TRUE(shapes - invalid >= EW_BCAST_FUZZ_ITERS * 8 / 10);
}
