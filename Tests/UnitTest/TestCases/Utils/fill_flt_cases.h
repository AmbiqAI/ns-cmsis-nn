/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/*
 * Shared body of the test_arm_nn_fill_{f32,f16} suites (#411). The including suite defines
 * FL_F16 (0/1), FL_PREFIX and FL_KERNEL. A fill is a bit splat: every output word must equal the
 * value's bits exactly, NaN payloads included, and the guard words around the output must survive.
 */

#pragma once

#include <arm_nnfunctions.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#define FL_CAT_(a, b) a##_##b
#define FL_CAT(a, b) FL_CAT_(a, b)
#define FL_FN(name) FL_CAT(FL_PREFIX, name)

#if FL_F16
typedef float16_t fl_t;
typedef uint16_t fl_bits_t;
    #define FL_ASSERT_BITS(e, a) TEST_ASSERT_EQUAL_HEX16((e), (a))
    #define FL_EXP 0x7C00u
    #define FL_MANT 0x03FFu
    #define FL_SIGN 0x8000u
    #define FL_QUIET 0x0200u
#else
typedef float32_t fl_t;
typedef uint32_t fl_bits_t;
    #define FL_ASSERT_BITS(e, a) TEST_ASSERT_EQUAL_HEX32((e), (a))
    #define FL_EXP 0x7F800000u
    #define FL_MANT 0x007FFFFFu
    #define FL_SIGN 0x80000000u
    #define FL_QUIET 0x00400000u
#endif

#define FL_GUARD 8
#define FL_MAX 1000
#define FL_GUARD_BITS ((fl_bits_t)(0xA5A5A5A5u & (fl_bits_t)~0u))

static fl_t fl_buf[FL_MAX + 2 * FL_GUARD];

static fl_bits_t fl_bits_of(fl_t x)
{
    fl_bits_t b;
    memcpy(&b, &x, sizeof(b));
    return b;
}
static fl_t fl_of_bits(fl_bits_t b)
{
    /* Staged through a volatile so -Ofast cannot fold the sign of a zero away. */
    volatile fl_bits_t staged = b;
    const fl_bits_t bits = staged;
    fl_t x;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static void fl_run(fl_bits_t value_bits, int32_t n)
{
    for (int32_t i = 0; i < FL_MAX + 2 * FL_GUARD; i++)
    {
        fl_buf[i] = fl_of_bits(FL_GUARD_BITS);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, FL_KERNEL(fl_of_bits(value_bits), fl_buf + FL_GUARD, n));
    for (int32_t i = 0; i < FL_GUARD; i++)
    {
        FL_ASSERT_BITS(FL_GUARD_BITS, fl_bits_of(fl_buf[i]));
        FL_ASSERT_BITS(FL_GUARD_BITS, fl_bits_of(fl_buf[FL_GUARD + n + i]));
    }
    for (int32_t i = 0; i < n; i++)
    {
        FL_ASSERT_BITS(value_bits, fl_bits_of(fl_buf[FL_GUARD + i]));
    }
}

/* Block sizes 0, 1, 15, 16, 17, 1000 with an ordinary value. */
void FL_FN(block_sizes)(void)
{
    static const int32_t sizes[] = {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 1000};
    fl_t v;
#if FL_F16
    v = (fl_t)1.5f;
#else
    v = 1.5f;
#endif
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++)
    {
        fl_run(fl_bits_of(v), sizes[k]);
    }
}

/* NaN (quiet and signaling, with payload and sign), +/-Inf, -0, subnormal, max finite: bit-exact. */
void FL_FN(special_values)(void)
{
    static const fl_bits_t specials[] = {
        (fl_bits_t)(FL_EXP | FL_QUIET | (FL_MANT & 0x2AAAAu)),
        (fl_bits_t)(FL_SIGN | FL_EXP | FL_QUIET | (FL_MANT & 0x15555u)),
        (fl_bits_t)(FL_EXP | 1u),
        (fl_bits_t)(FL_SIGN | FL_EXP | ((FL_MANT >> 1) & 0x3F3u)),
        (fl_bits_t)FL_EXP,
        (fl_bits_t)(FL_SIGN | FL_EXP),
        (fl_bits_t)FL_SIGN,
        1u,
        (fl_bits_t)(FL_SIGN | FL_MANT),
        (fl_bits_t)(FL_EXP - 1u),
    };
    static const int32_t sizes[] = {1, 15, 17, 1000};
    for (size_t k = 0; k < sizeof(specials) / sizeof(specials[0]); k++)
    {
        for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++)
        {
            fl_run(specials[k], sizes[s]);
        }
    }
}

/* Negative block size and a NULL output with work to do: ARG_ERROR, output untouched. */
void FL_FN(arg_error)(void)
{
    fl_t v;
#if FL_F16
    v = (fl_t)2.0f;
#else
    v = 2.0f;
#endif
    for (int32_t i = 0; i < FL_MAX + 2 * FL_GUARD; i++)
    {
        fl_buf[i] = fl_of_bits(FL_GUARD_BITS);
    }
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, FL_KERNEL(v, fl_buf + FL_GUARD, -1));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, FL_KERNEL(v, NULL, 4));
    for (int32_t i = 0; i < FL_MAX + 2 * FL_GUARD; i++)
    {
        FL_ASSERT_BITS(FL_GUARD_BITS, fl_bits_of(fl_buf[i]));
    }
    /* A NULL output with nothing to write is a no-op, not an error. */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, FL_KERNEL(v, NULL, 0));
}
