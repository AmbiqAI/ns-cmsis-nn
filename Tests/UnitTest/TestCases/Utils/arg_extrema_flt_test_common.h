/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

#include <arm_nnfunctions.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

/* Format fields are local to this independent binary64 reference.
 * An all-ones exponent means Inf/NaN only under IEEE; alternative half is finite. */
#if AE_HALF
    #define AE_TYPE float16_t
    #define AE_BITS uint16_t
    #define AE_SIGN_MASK UINT32_C(0x8000)
    #define AE_EXPONENT_MASK UINT32_C(0x7c00)
    #define AE_ONE_BITS UINT32_C(0x3c00)
    #define AE_FRAC 10
    #define AE_BIAS 15
    #include "arg_extrema_f16_data.h"
#else
    #define AE_TYPE float32_t
    #define AE_BITS uint32_t
    #define AE_SIGN_MASK UINT32_C(0x80000000)
    #define AE_EXPONENT_MASK UINT32_C(0x7f800000)
    #define AE_ONE_BITS UINT32_C(0x3f800000)
    #define AE_FRAC 23
    #define AE_BIAS 127
    #include "arg_extrema_f32_data.h"
#endif
#if AE_HALF && defined(__ARM_FP16_FORMAT_ALTERNATIVE) && !(defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2))
    #define AE_IEEE_SPECIALS 0
#else
    #define AE_IEEE_SPECIALS 1
#endif
#define AE_MAGNITUDE_MASK (AE_SIGN_MASK - 1)
#define AE_FRACTION_MASK ((UINT32_C(1) << AE_FRAC) - 1)
#define AE_CAP 1024
#define AE_GUARD 8
#define AE_SENTINEL INT32_C(0x5a5a5a5a)

static AE_TYPE ae_storage[AE_CAP + 2 * AE_GUARD];
static AE_BITS ae_bits[AE_CAP];
static int32_t ae_output[AE_CAP + 2 * AE_GUARD];
static int32_t ae_expected[AE_CAP];
static AE_BITS ae_winners[AE_CAP];

static cmsis_nn_dims ae_dims(const int32_t d[4])
{
    const cmsis_nn_dims dims = {d[0], d[1], d[2], d[3]};
    return dims;
}

static int ae_count(const int32_t d[4]) { return d[0] * d[1] * d[2] * d[3]; }

/* Decode directly into binary64, independently of the kernel's integer keys.
 * Alternative half uses exponent 31 for finite values. Every finite value of
 * either half format and binary32 is normal in binary64, including subnormals. */
static double ae_value(uint32_t bits)
{
    const uint32_t magnitude = bits & AE_MAGNITUDE_MASK;
    const uint32_t fraction = magnitude & AE_FRACTION_MASK;
    const int exponent = (int)(magnitude >> AE_FRAC);
    double value;
    if (AE_IEEE_SPECIALS && magnitude == AE_EXPONENT_MASK)
    {
        value = INFINITY;
    }
    else if (exponent == 0)
    {
        value = ldexp((double)fraction, 1 - AE_BIAS - AE_FRAC);
    }
    else
    {
        value = ldexp((double)((UINT32_C(1) << AE_FRAC) + fraction), exponent - AE_BIAS - AE_FRAC);
    }
    return (bits & AE_SIGN_MASK) ? -value : value;
}

static int ae_is_nan(uint32_t bits)
{
    return AE_IEEE_SPECIALS && (bits & AE_EXPONENT_MASK) == AE_EXPONENT_MASK && (bits & AE_FRACTION_MASK) != 0;
}

static void ae_reference(const int32_t d[4], int axis)
{
    const int out_count = ae_count(d) / d[axis];
    for (int i = 0; i < out_count; ++i)
        ae_expected[i] = -1;
    int source = 0;
    for (int n = 0; n < d[0]; ++n)
        for (int h = 0; h < d[1]; ++h)
            for (int w = 0; w < d[2]; ++w)
                for (int c = 0; c < d[3]; ++c, ++source)
                {
                    const int coords[4] = {n, h, w, c};
                    int dest = 0;
                    for (int k = 0; k < 4; ++k)
                        if (k != axis)
                            dest = dest * d[k] + coords[k];
                    const uint32_t next = ae_bits[source];
                    const uint32_t previous = ae_winners[dest];
                    int replace = ae_expected[dest] < 0;
                    if (!replace && !ae_is_nan(previous))
                    {
                        if (ae_is_nan(next))
                            replace = 1;
#if AE_MAX
                        else if (ae_value(next) > ae_value(previous))
                            replace = 1;
#else
                        else if (ae_value(next) < ae_value(previous))
                            replace = 1;
#endif
                    }
                    if (replace)
                    {
                        ae_expected[dest] = coords[axis];
                        ae_winners[dest] = (AE_BITS)next;
                    }
                }
}

static void ae_check(const int32_t d[4], int axis)
{
    const cmsis_nn_dims dims = ae_dims(d);
    const int count = ae_count(d);
    const int out_count = count / d[axis];
    TEST_ASSERT_LESS_OR_EQUAL_INT(AE_CAP, count);
    memset(ae_storage, 0xa5, sizeof(ae_storage));
    memcpy(ae_storage + AE_GUARD, ae_bits, (size_t)count * sizeof(AE_BITS));
    for (int i = 0; i < AE_CAP + 2 * AE_GUARD; ++i)
        ae_output[i] = AE_SENTINEL;
    ae_reference(d, axis);
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AE_KERNEL(ae_storage + AE_GUARD, &dims, axis, ae_output + AE_GUARD));
    TEST_ASSERT_EQUAL_INT32_ARRAY(ae_expected, ae_output + AE_GUARD, out_count);
    TEST_ASSERT_EQUAL_MEMORY(ae_bits, ae_storage + AE_GUARD, (size_t)count * sizeof(AE_BITS));
    const unsigned char *input_bytes = (const unsigned char *)ae_storage;
    for (size_t i = 0; i < AE_GUARD * sizeof(AE_TYPE); ++i)
        TEST_ASSERT_EQUAL_HEX8(0xa5, input_bytes[i]);
    for (size_t i = (AE_GUARD + count) * sizeof(AE_TYPE); i < sizeof(ae_storage); ++i)
        TEST_ASSERT_EQUAL_HEX8(0xa5, input_bytes[i]);
    for (int i = 0; i < AE_GUARD; ++i)
        TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_output[i]);
    for (int i = AE_GUARD + out_count; i < AE_CAP + 2 * AE_GUARD; ++i)
        TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_output[i]);
}

static void ae_golden(void)
{
    const int32_t d[4] = {2, 3, 2, 5};
#if AE_MAX
    const int32_t *golden[4] = {ae_max_axis0, ae_max_axis1, ae_max_axis2, ae_max_axis3};
#else
    const int32_t *golden[4] = {ae_min_axis0, ae_min_axis1, ae_min_axis2, ae_min_axis3};
#endif
    memcpy(ae_bits, ae_golden_input, sizeof(ae_golden_input));
    for (int axis = 0; axis < 4; ++axis)
    {
        ae_check(d, axis);
        TEST_ASSERT_EQUAL_INT32_ARRAY(golden[axis], ae_output + AE_GUARD, ae_count(d) / d[axis]);
    }
}

static void ae_axes(void)
{
    const int32_t shapes[][4] = {{3, 1, 1, 9}, {2, 3, 2, 5}, {2, 1, 3, 9}, {1, 1, 1, 1}, {1, 3, 5, 7}, {2, 2, 3, 17}};
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
    {
        const int count = ae_count(shapes[s]);
        for (int i = 0; i < count; ++i)
            ae_bits[i] = ae_golden_input[(i * 17 + s * 7) % 60];
        for (int axis = 0; axis < 4; ++axis)
            ae_check(shapes[s], axis);
    }
    /* Axis0 winners vary by retained column, including both endpoints. */
    const int32_t d[4] = {3, 1, 1, 9};
    for (int n = 0; n < 3; ++n)
        for (int c = 0; c < 9; ++c)
        {
#if AE_MAX
            ae_bits[n * 9 + c] = n == c % 3 ? AE_ONE_BITS : 0;
#else
            ae_bits[n * 9 + c] = n == c % 3 ? (AE_ONE_BITS | AE_SIGN_MASK) : 0;
#endif
        }
    ae_check(d, 0);
    for (int c = 0; c < 9; ++c)
        TEST_ASSERT_EQUAL_INT32(c % 3, ae_output[AE_GUARD + c]);
}

static void ae_special(void)
{
#if !AE_IEEE_SPECIALS
    /* Exponent-31 finite boundary, next value, and signed extrema of alternative half. */
    TEST_ASSERT_TRUE(ae_value(0x7c00) == 65536.0);
    TEST_ASSERT_TRUE(ae_value(0x7c01) == 65600.0);
    TEST_ASSERT_TRUE(ae_value(0x7fff) == 131008.0);
    TEST_ASSERT_TRUE(ae_value(0xffff) == -131008.0);
#endif
    const uint32_t patterns[] = {AE_EXPONENT_MASK | 1,
                                 AE_ONE_BITS,
                                 AE_SIGN_MASK | AE_ONE_BITS,
                                 AE_EXPONENT_MASK | 2,
                                 0,
                                 AE_SIGN_MASK,
                                 1,
                                 AE_SIGN_MASK | 1,
                                 AE_EXPONENT_MASK,
                                 AE_SIGN_MASK | AE_EXPONENT_MASK,
                                 AE_ONE_BITS,
                                 AE_ONE_BITS,
                                 AE_SIGN_MASK | AE_EXPONENT_MASK | 3};
    const int32_t d[4] = {2, 3, 2, 5};
    for (unsigned shift = 0; shift < sizeof(patterns) / sizeof(patterns[0]); ++shift)
    {
        for (int i = 0; i < 60; ++i)
            ae_bits[i] = (AE_BITS)patterns[(i + shift) % 13];
        for (int axis = 0; axis < 4; ++axis)
            ae_check(d, axis);
    }
    const int32_t line[4] = {1, 1, 1, 9};
    for (int position = 0; position < 9; ++position)
    {
        for (int i = 0; i < 9; ++i)
            ae_bits[i] = (AE_BITS)AE_ONE_BITS;
        /* A NaN wins under IEEE; a signed finite extreme wins under alternative half. */
        ae_bits[position] = (AE_BITS)(AE_EXPONENT_MASK | 1 | ((!AE_IEEE_SPECIALS && !AE_MAX) ? AE_SIGN_MASK : 0));
        ae_check(line, 3);
        TEST_ASSERT_EQUAL_INT32(position, ae_output[AE_GUARD]);
    }
    /* First zero tie crosses a prospective vector boundary. */
    for (int i = 0; i < 9; ++i)
        ae_bits[i] = (AE_BITS)(AE_SIGN_MASK * (i & 1));
    ae_check(line, 3);
    TEST_ASSERT_EQUAL_INT32(0, ae_output[AE_GUARD]);
    ae_bits[7] = (AE_BITS)(AE_EXPONENT_MASK | 1);
    ae_bits[8] = (AE_BITS)(AE_EXPONENT_MASK | 2);
    ae_check(line, 3);
#if AE_IEEE_SPECIALS
    TEST_ASSERT_EQUAL_INT32(7, ae_output[AE_GUARD]);
#else
    TEST_ASSERT_EQUAL_INT32(AE_MAX ? 8 : 0, ae_output[AE_GUARD]);
#endif
}

static void ae_patterns(void)
{
    const int32_t d[4] = {1, 1, 3, 256};
#if AE_HALF
    for (uint32_t start = 0; start < 65536; start += 256)
#else
    uint32_t state = UINT32_C(0x9e3779b9);
    for (uint32_t start = 0; start < 8192; start += 256)
#endif
    {
        for (int rotation = 0; rotation < 3; ++rotation)
        {
            for (uint32_t c = 0; c < 256; ++c)
            {
#if AE_HALF
                const uint32_t bits = start + c;
#else
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                const uint32_t bits = state;
#endif
                const uint32_t competitors[3] = {bits, AE_SIGN_MASK * (c & 1), AE_ONE_BITS | (AE_SIGN_MASK * ((c >> 1) & 1))};
                for (int k = 0; k < 3; ++k)
                    ae_bits[k * 256 + c] = (AE_BITS)competitors[(k + rotation) % 3];
            }
            ae_check(d, 2);
        }
    }
}

static void ae_empty(void)
{
    for (int zero = 0; zero < 4; ++zero)
        for (int axis = 0; axis < 4; ++axis)
        {
            int32_t d[4] = {2, 3, 4, 5};
            d[zero] = 0;
            const cmsis_nn_dims dims = ae_dims(d);
            ae_output[0] = AE_SENTINEL;
            const arm_cmsis_nn_status expected = axis == zero ? ARM_CMSIS_NN_ARG_ERROR : ARM_CMSIS_NN_SUCCESS;
            TEST_ASSERT_EQUAL(expected, AE_KERNEL(NULL, &dims, axis, ae_output));
            TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_output[0]);
            TEST_ASSERT_EQUAL(expected, AE_KERNEL(NULL, &dims, axis, NULL));
        }
    const cmsis_nn_dims zero_product = {INT32_MAX, INT32_MAX, 0, 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, AE_KERNEL(NULL, &zero_product, 0, NULL));
    const cmsis_nn_dims both_empty = {0, 0, 1, 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(NULL, &both_empty, 0, NULL));
}

static void ae_invalid(void)
{
    const cmsis_nn_dims valid = {1, 1, 1, 1};
    ae_output[0] = AE_SENTINEL;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(ae_storage, NULL, 0, ae_output));
    const int32_t axes[] = {-1, 4, INT32_MIN, INT32_MAX};
    for (unsigned i = 0; i < sizeof(axes) / sizeof(axes[0]); ++i)
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(ae_storage, &valid, axes[i], ae_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(NULL, &valid, 0, ae_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(ae_storage, &valid, 0, NULL));
    const cmsis_nn_dims invalid[] = {{-1, 1, 1, 1},
                                     {1, -1, 1, 1},
                                     {1, 1, -1, 1},
                                     {1, 1, 1, -1},
                                     {0, -1, 1, 1},
                                     {1, 1, 1, INT32_MAX},
                                     {1, 1, 536870912, 1}};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, AE_KERNEL(ae_storage, &invalid[i], 3, ae_output));
    TEST_ASSERT_EQUAL_INT32(AE_SENTINEL, ae_output[0]);
}

static void ae_fp_controls(void)
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
    uint32_t original;
    __asm volatile("vmrs %0, fpscr" : "=r"(original));
    /* FPSCR: FZ16[19], RMode[23:22], FZ[24], DN[25], AHP[26]. */
    const uint32_t controls = (UINT32_C(1) << 19) | (UINT32_C(3) << 22) | (UINT32_C(7) << 24);
    /* Cumulative exception flags IOC/DZC/OFC/UFC/IXC[4:0] and IDC[7]. */
    const uint32_t exceptions = UINT32_C(0x9f);
    const cmsis_nn_dims dims = {1, 1, 1, 5};
    AE_TYPE input[5];
    const AE_BITS cases[][5] = {{AE_EXPONENT_MASK | 1, 1, AE_SIGN_MASK | 1, AE_ONE_BITS, 0},
                                {0, 1, AE_SIGN_MASK | 1, AE_SIGN_MASK, 0},
                                {AE_ONE_BITS, AE_SIGN_MASK | AE_ONE_BITS, AE_EXPONENT_MASK, AE_SIGN_MASK | AE_EXPONENT_MASK, 0},
                                {1, AE_SIGN_MASK | 1, AE_ONE_BITS, AE_EXPONENT_MASK | 1, AE_EXPONENT_MASK | 2},
                                {AE_SIGN_MASK, 0, AE_SIGN_MASK, 0, 0}};
    #if AE_IEEE_SPECIALS
        #if AE_MAX
    const int32_t expected[] = {0, 1, 2, 3, 0};
        #else
    const int32_t expected[] = {0, 2, 3, 3, 0};
        #endif
    #elif AE_MAX
    const int32_t expected[] = {0, 1, 2, 4, 0};
    #else
    const int32_t expected[] = {2, 2, 3, 1, 0};
    #endif
    for (unsigned sample = 0; sample < sizeof(cases) / sizeof(cases[0]); ++sample)
    {
        memcpy(input, cases[sample], sizeof(input));
        for (uint32_t c = 0; c < 64; ++c)
        {
            const uint32_t mode = ((c & 1) << 19) | (((c >> 1) & 3) << 22) | (((c >> 3) & 7) << 24);
            const uint32_t wanted = (original & ~(controls | exceptions)) | mode | 2;
            __asm volatile("vmsr fpscr, %0" ::"r"(wanted) : "memory");
            int32_t index = -1;
            const arm_cmsis_nn_status status = AE_KERNEL(input, &dims, 3, &index);
            uint32_t after;
            __asm volatile("vmrs %0, fpscr" : "=r"(after));
            __asm volatile("vmsr fpscr, %0" ::"r"(original) : "memory");
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
            TEST_ASSERT_EQUAL_INT32(expected[sample], index);
            TEST_ASSERT_EQUAL_HEX32(wanted & (controls | exceptions), after & (controls | exceptions));
        }
    }
#endif
}
