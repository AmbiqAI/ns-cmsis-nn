/*
 * SPDX-FileCopyrightText: Copyright 2026 Ambiq
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

/* Raw encodings and a binary64 reference keep FP controls out of the oracle. Refs #498. */
#if RE_HALF
    #define RE_TYPE float16_t
    #define RE_BITS uint16_t
    #define RE_FRAC_BITS 10
    #define RE_BIAS 15
    #define RE_SIGN UINT32_C(0x8000)
    #define RE_INF UINT32_C(0x7c00)
    #define RE_NAN UINT32_C(0x7e00)
    #include "reduce_extrema_f16_data.h"
#else
    #define RE_TYPE float32_t
    #define RE_BITS uint32_t
    #define RE_FRAC_BITS 23
    #define RE_BIAS 127
    #define RE_SIGN UINT32_C(0x80000000)
    #define RE_INF UINT32_C(0x7f800000)
    #define RE_NAN UINT32_C(0x7fc00000)
    #include "reduce_extrema_f32_data.h"
#endif
#define RE_CAPACITY 1024

static RE_TYPE re_input[RE_CAPACITY];
static RE_TYPE re_output[RE_CAPACITY + 16];
static RE_BITS re_expected[RE_CAPACITY];
static RE_BITS re_input_bits[RE_CAPACITY];

static uint32_t re_read(const RE_TYPE *p)
{
    RE_BITS bits;
    memcpy(&bits, p, sizeof(bits));
    return bits;
}

static double re_finite_value(uint32_t bits)
{
    const uint32_t frac_mask = (UINT32_C(1) << RE_FRAC_BITS) - 1;
    const uint32_t frac = bits & frac_mask;
    const int exponent = (int)((bits & ~RE_SIGN) >> RE_FRAC_BITS);
    double result;
    if (exponent == 0)
    {
        result = ldexp((double)frac, 1 - RE_BIAS - RE_FRAC_BITS);
    }
    else
    {
        result = ldexp((double)((UINT32_C(1) << RE_FRAC_BITS) + frac), exponent - RE_BIAS - RE_FRAC_BITS);
    }
    if (bits & RE_SIGN)
    {
        result = -result;
    }
    return result;
}

static uint32_t re_reference_pair(uint32_t first, uint32_t next)
{
    const uint32_t a = first & ~RE_SIGN;
    const uint32_t b = next & ~RE_SIGN;
    if (a > RE_INF || b > RE_INF)
    {
        return RE_NAN;
    }
    if (a == RE_INF || b == RE_INF)
    {
        if (first == next)
        {
            return first;
        }
#if RE_MAX
        if (first == RE_INF || next == (RE_INF | RE_SIGN))
#else
        if (first == (RE_INF | RE_SIGN) || next == RE_INF)
#endif
        {
            return first;
        }
        return next;
    }
    const double av = re_finite_value(first);
    const double bv = re_finite_value(next);
#if RE_MAX
    if (bv > av)
#else
    if (bv < av)
#endif
    {
        return next;
    }
    return first;
}

static cmsis_nn_dims re_dims(const int32_t x[4])
{
    const cmsis_nn_dims result = {x[0], x[1], x[2], x[3]};
    return result;
}

static int re_count(const int32_t x[4]) { return x[0] * x[1] * x[2] * x[3]; }

static void re_shape(const int32_t in[4], int mask, int32_t axes[4], int32_t out[4])
{
    for (int d = 0; d < 4; ++d)
    {
        axes[d] = (mask >> d) & 1;
        out[d] = axes[d] ? 1 : in[d];
    }
}

static void re_reference(const int32_t in[4], int mask, const int32_t out[4])
{
    unsigned char seen[RE_CAPACITY] = {0};
#if RE_MAX
    const RE_BITS identity = RE_INF | RE_SIGN;
#else
    const RE_BITS identity = RE_INF;
#endif
    for (int i = 0; i < re_count(out); ++i)
    {
        re_expected[i] = identity;
    }
    int source = 0;
    for (int n = 0; n < in[0]; ++n)
    {
        for (int h = 0; h < in[1]; ++h)
        {
            for (int w = 0; w < in[2]; ++w)
            {
                for (int c = 0; c < in[3]; ++c, ++source)
                {
                    const int coords[4] = {n, h, w, c};
                    int dest = 0;
                    for (int d = 0; d < 4; ++d)
                    {
                        dest = dest * out[d] + ((mask & (1 << d)) ? 0 : coords[d]);
                    }
                    const uint32_t bits = re_read(&re_input[source]);
                    if (!seen[dest])
                    {
                        re_expected[dest] = (RE_BITS)bits;
                        if (mask && (bits & ~RE_SIGN) > RE_INF)
                        {
                            re_expected[dest] = RE_NAN;
                        }
                        seen[dest] = 1;
                    }
                    else
                    {
                        re_expected[dest] = (RE_BITS)re_reference_pair(re_expected[dest], bits);
                    }
                }
            }
        }
    }
}

static void re_check_result(int count)
{
    if (count != 0)
    {
        TEST_ASSERT_EQUAL_MEMORY(re_expected, re_output + 8, (size_t)count * sizeof(RE_BITS));
    }
    const unsigned char *bytes = (const unsigned char *)re_output;
    TEST_ASSERT_EACH_EQUAL_HEX8(0xa5, bytes, 8 * sizeof(RE_TYPE));
    const size_t tail = (size_t)(count + 8) * sizeof(RE_TYPE);
    TEST_ASSERT_EACH_EQUAL_HEX8(0xa5, bytes + tail, sizeof(re_output) - tail);
}

static void re_run(const int32_t in[4], int mask)
{
    int32_t axes[4], out[4];
    re_shape(in, mask, axes, out);
    const cmsis_nn_dims input_dims = re_dims(in), axis_dims = re_dims(axes), output_dims = re_dims(out);
    TEST_ASSERT_LESS_OR_EQUAL_INT(RE_CAPACITY, re_count(in));
    TEST_ASSERT_LESS_OR_EQUAL_INT(RE_CAPACITY, re_count(out));
    re_reference(in, mask, out);
    memcpy(re_input_bits, re_input, (size_t)re_count(in) * sizeof(RE_TYPE));
    memset(re_output, 0xa5, sizeof(re_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RE_KERNEL(re_input, &input_dims, &axis_dims, re_output + 8, &output_dims));
    re_check_result(re_count(out));
    TEST_ASSERT_EQUAL_MEMORY(re_input_bits, re_input, (size_t)re_count(in) * sizeof(RE_TYPE));
}

static const RE_BITS re_special[] = {0,
                                     RE_SIGN,
                                     1,
                                     RE_SIGN | 1,
                                     (UINT32_C(1) << RE_FRAC_BITS) - 1,
                                     RE_SIGN | ((UINT32_C(1) << RE_FRAC_BITS) - 1),
                                     UINT32_C(1) << RE_FRAC_BITS,
                                     RE_SIGN | (UINT32_C(1) << RE_FRAC_BITS),
                                     (uint32_t)RE_BIAS << RE_FRAC_BITS,
                                     RE_SIGN | ((uint32_t)RE_BIAS << RE_FRAC_BITS),
                                     RE_INF - 1,
                                     RE_SIGN | (RE_INF - 1),
                                     RE_INF,
                                     RE_SIGN | RE_INF,
                                     RE_INF | 1,
                                     RE_SIGN | (RE_INF | 1),
                                     RE_NAN | 3,
                                     RE_SIGN | (RE_NAN | 7)};

static void re_golden(void)
{
    const int32_t in[4] = {2, 3, 5, 7};
    memcpy(re_input, re_golden_input, sizeof(re_golden_input));
    for (int mask = 0; mask < 16; ++mask)
    {
        re_run(in, mask);
#if RE_MAX
        const RE_BITS *golden = re_golden_max + re_golden_offsets[mask];
#else
        const RE_BITS *golden = re_golden_min + re_golden_offsets[mask];
#endif
        TEST_ASSERT_EQUAL_MEMORY(
            golden, re_output + 8, (size_t)(re_golden_offsets[mask + 1] - re_golden_offsets[mask]) * sizeof(RE_BITS));
    }
}

static void re_masks(void)
{
    static const int32_t shapes[][4] = {
        {2, 3, 5, 7}, {1, 8, 17, 1}, {1, 1, 8, 17}, {3, 1, 1, 9}, {2, 1, 3, 1}, {1, 1, 1, 1}};
    uint32_t state = UINT32_C(0xa8310f72);
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
    {
        for (int i = 0; i < re_count(shapes[s]); ++i)
        {
            state = state * UINT32_C(1664525) + UINT32_C(1013904223);
            /* Exclude NaNs so a random NaN cannot conceal axis errors. */
            const RE_BITS bits = (RE_BITS)((state & RE_SIGN) | ((state >> 1) % RE_INF));
            memcpy(re_input + i, &bits, sizeof(bits));
        }
        for (int mask = 0; mask < 16; ++mask)
        {
            re_run(shapes[s], mask);
        }
    }
}

static void re_special_positions(void)
{
    const int32_t shapes[][4] = {{1, 1, 3, 17}, {1, 3, 1, 17}, {3, 1, 1, 17}, {2, 3, 1, 7}};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
    {
        for (size_t p = 0; p < sizeof(re_special) / sizeof(re_special[0]); ++p)
        {
            for (int i = 0; i < re_count(shapes[s]); ++i)
            {
                const RE_BITS bits = re_special[(i + p) % (sizeof(re_special) / sizeof(re_special[0]))];
                memcpy(re_input + i, &bits, sizeof(bits));
            }
            for (int mask = 0; mask < 16; ++mask)
            {
                re_run(shapes[s], mask);
            }
        }
    }
    for (int size = 1; size <= 33; ++size)
    {
        const int32_t in[4] = {1, 1, 1, size};
        for (int nan_pos = 0; nan_pos < size; ++nan_pos)
        {
            memset(re_input, 0, sizeof(re_input));
            const RE_BITS snan = RE_SIGN | RE_INF | 1;
            memcpy(re_input + nan_pos, &snan, sizeof(snan));
            re_run(in, 8);
        }
    }
}

static void re_ordered_pairs(void)
{
    const int32_t in[4] = {1, 1, 1, 2};
    for (size_t a = 0; a < sizeof(re_special) / sizeof(re_special[0]); ++a)
    {
        for (size_t b = 0; b < sizeof(re_special) / sizeof(re_special[0]); ++b)
        {
            memcpy(re_input, re_special + a, sizeof(RE_BITS));
            memcpy(re_input + 1, re_special + b, sizeof(RE_BITS));
            re_run(in, 8);
        }
    }
    /* Multiple vector blocks: first zero beats a later opposite zero. */
    for (int n = 2; n <= 33; ++n)
    {
        const int32_t dims[4] = {1, 1, 1, n};
        for (int sign = 0; sign < 2; ++sign)
        {
            for (int i = 0; i < n; ++i)
            {
                const RE_BITS bits = ((i == 0) == sign) ? RE_SIGN : 0;
                memcpy(re_input + i, &bits, sizeof(bits));
            }
            re_run(dims, 8);
        }
    }
}

static void re_random_pairs(void)
{
    uint32_t state = UINT32_C(0x93c16af5);
    const int32_t suffix[4] = {1, 1, 257, 2}, retained[4] = {1, 1, 2, 257};
    for (int batch = 0; batch < 128; ++batch)
    {
        for (int i = 0; i < 514; ++i)
        {
            state = state * UINT32_C(1664525) + UINT32_C(1013904223);
            const RE_BITS raw = (RE_BITS)state;
            memcpy(re_input + i, &raw, sizeof(raw));
        }
        re_run(suffix, 8);
        re_run(retained, 4);
    }
}

static void re_empty(void)
{
    for (int zero_dim = 0; zero_dim < 4; ++zero_dim)
    {
        int32_t in[4] = {2, 3, 5, 7};
        in[zero_dim] = 0;
        for (int mask = 0; mask < 16; ++mask)
        {
            int32_t axes[4], out[4];
            re_shape(in, mask, axes, out);
            const cmsis_nn_dims id = re_dims(in), ax = re_dims(axes), od = re_dims(out);
            re_reference(in, mask, out);
            memset(re_output, 0xa5, sizeof(re_output));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RE_KERNEL(NULL, &id, &ax, re_output + 8, &od));
            re_check_result(re_count(out));
            if (re_count(out) == 0)
            {
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RE_KERNEL(NULL, &id, &ax, NULL, &od));
            }
        }
    }
    for (int zero_dim = 0; zero_dim < 4; ++zero_dim)
    {
        int32_t in[4] = {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
        in[zero_dim] = 0;
        const cmsis_nn_dims id = re_dims(in), no_axes = {0, 0, 0, 0}, all_axes = {1, 1, 1, 1}, one = {1, 1, 1, 1};
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RE_KERNEL(NULL, &id, &no_axes, NULL, &id));
        memset(re_output, 0xa5, sizeof(re_output));
#if RE_MAX
        re_expected[0] = RE_INF | RE_SIGN;
#else
        re_expected[0] = RE_INF;
#endif
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, RE_KERNEL(NULL, &id, &all_axes, re_output + 8, &one));
        re_check_result(1);
    }
}

static void re_bad(const RE_TYPE *input,
                   const cmsis_nn_dims *id,
                   const cmsis_nn_dims *axes,
                   RE_TYPE *output,
                   const cmsis_nn_dims *od)
{
    memset(re_output, 0xa5, sizeof(re_output));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, RE_KERNEL(input, id, axes, output, od));
    const unsigned char *bytes = (const unsigned char *)re_output;
    TEST_ASSERT_EACH_EQUAL_HEX8(0xa5, bytes, sizeof(re_output));
}

static void re_invalid(void)
{
    const cmsis_nn_dims id = {2, 3, 5, 7}, axes = {0, 1, 0, 1}, od = {2, 1, 5, 1};
    re_bad(NULL, &id, &axes, re_output + 8, &od);
    re_bad(re_input, NULL, &axes, re_output + 8, &od);
    re_bad(re_input, &id, NULL, re_output + 8, &od);
    re_bad(re_input, &id, &axes, NULL, &od);
    re_bad(re_input, &id, &axes, re_output + 8, NULL);
    for (int dim = 0; dim < 4; ++dim)
    {
        for (int which = 0; which < 3; ++which)
        {
            int32_t iv[4] = {2, 3, 5, 7}, av[4] = {0, 1, 0, 1}, ov[4] = {2, 1, 5, 1};
            if (which == 0)
                iv[dim] = -1;
            if (which == 1)
                av[dim] = 2;
            if (which == 2)
                ov[dim] = -1;
            cmsis_nn_dims bi = re_dims(iv), ba = re_dims(av), bo = re_dims(ov);
            re_bad(re_input, &bi, &ba, re_output + 8, &bo);
            av[dim] = -1;
            ba = re_dims(av);
            re_bad(re_input, &bi, &ba, re_output + 8, &bo);
        }
        int32_t ov[4] = {2, 1, 5, 1};
        ov[dim] += 1;
        const cmsis_nn_dims wrong = re_dims(ov);
        re_bad(re_input, &id, &axes, re_output + 8, &wrong);
    }
    const cmsis_nn_dims no_axes = {0, 0, 0, 0}, all_axes = {1, 1, 1, 1}, one = {1, 1, 1, 1};
    const cmsis_nn_dims over[] = {{1, 1, 1, INT32_MAX / (int32_t)sizeof(RE_TYPE) + 1},
                                  {1, 65536, 65536, 1},
                                  {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX}};
    for (size_t i = 0; i < sizeof(over) / sizeof(over[0]); ++i)
    {
        re_bad(re_input, &over[i], &all_axes, re_output + 8, &one);
        re_bad(re_input, &over[i], &no_axes, re_output + 8, &over[i]);
    }
    const cmsis_nn_dims empty = {0, INT32_MAX, INT32_MAX, INT32_MAX}, reduce_empty = {1, 0, 0, 0};
    const cmsis_nn_dims overflow_output = {1, INT32_MAX, INT32_MAX, INT32_MAX};
    re_bad(NULL, &empty, &reduce_empty, re_output + 8, &overflow_output);
    const cmsis_nn_dims empty_bad = {0, -1, 3, 1}, invalid_mask = {0, 0, 0, 2};
    re_bad(NULL, &empty_bad, &no_axes, re_output + 8, &empty_bad);
    re_bad(NULL, &empty, &invalid_mask, re_output + 8, &empty);
    re_bad(NULL, &empty, &all_axes, NULL, &one);
}

static void re_fp_controls(void)
{
#if defined(__ARM_FP) && (__ARM_FP != 0)
    uint32_t saved;
    __asm__ volatile("vmrs %0, fpscr" : "=r"(saved) : : "memory");
    const uint32_t clear = (3u << 22) | (1u << 24) | (1u << 25) | (1u << 19);
    const int32_t shapes[][4] = {{1, 1, 3, 17}, {1, 3, 1, 17}, {2, 3, 1, 7}};
    const int masks[] = {8, 2, 9};
    for (unsigned pattern_count = 8; pattern_count <= 18; pattern_count += 5)
    {
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
        {
            for (int i = 0; i < re_count(shapes[s]); ++i)
            {
                const RE_BITS bits = re_special[i % pattern_count];
                memcpy(re_input + i, &bits, sizeof(bits));
            }
            int32_t av[4], ov[4];
            re_shape(shapes[s], masks[s], av, ov);
            const cmsis_nn_dims id = re_dims(shapes[s]), axes = re_dims(av), od = re_dims(ov);
            re_reference(shapes[s], masks[s], ov);
            for (unsigned mode = 0; mode < 32; ++mode)
            {
                const uint32_t selected = (saved & ~clear) | ((mode & 3u) << 22) | (((mode >> 2) & 1u) << 24) |
                    (((mode >> 3) & 1u) << 25) | (((mode >> 4) & 1u) << 19);
                memset(re_output, 0xa5, sizeof(re_output));
                __asm__ volatile("vmsr fpscr, %0" : : "r"(selected) : "memory");
                const arm_cmsis_nn_status status = RE_KERNEL(re_input, &id, &axes, re_output + 8, &od);
                __asm__ volatile("vmsr fpscr, %0" : : "r"(saved) : "memory");
                TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
                re_check_result(re_count(ov));
            }
        }
    }
#else
    TEST_IGNORE_MESSAGE("Target FPSCR controls require Arm floating-point hardware");
#endif
}

#if RE_HALF
static void re_exhaust_half(void)
{
    static const uint16_t constants[] = {0, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7c01};
    for (size_t k = 0; k < 9; ++k)
    {
        for (int order = 0; order < 2; ++order)
        {
            for (uint32_t start = 0; start < 65536; start += 257)
            {
                const int count = (65536 - start < 257) ? (int)(65536 - start) : 257;
                const int32_t shape[4] = {1, 1, count, 2};
                for (int i = 0; i < count; ++i)
                {
                    const uint16_t raw = (uint16_t)(start + (uint32_t)i);
                    uint16_t other;
                    if (k < 7)
                        other = constants[k];
                    else if (k == 7)
                        other = raw ^ UINT16_C(0x8000);
                    else
                        other = (uint16_t)(raw + 1);
                    memcpy(re_input + 2 * i + order, &raw, sizeof(raw));
                    memcpy(re_input + 2 * i + (1 - order), &other, sizeof(raw));
                }
                re_run(shape, 8);
                const int32_t retained_shape[4] = {1, 1, 2, count};
                for (int i = 0; i < count; ++i)
                {
                    const uint16_t raw = (uint16_t)(start + (uint32_t)i);
                    uint16_t other;
                    if (k < 7)
                        other = constants[k];
                    else if (k == 7)
                        other = raw ^ UINT16_C(0x8000);
                    else
                        other = (uint16_t)(raw + 1);
                    memcpy(re_input + count * order + i, &raw, sizeof(raw));
                    memcpy(re_input + count * (1 - order) + i, &other, sizeof(raw));
                }
                re_run(retained_shape, 4);
            }
        }
    }
    /* Every encoding is also an exact no-axis copy and a singleton reduction. */
    for (uint32_t start = 0; start < 65536; start += 257)
    {
        const int count = (65536 - start < 257) ? (int)(65536 - start) : 257;
        const int32_t shape[4] = {1, 1, count, 1};
        for (int i = 0; i < count; ++i)
        {
            const uint16_t raw = (uint16_t)(start + (uint32_t)i);
            memcpy(re_input + i, &raw, sizeof(raw));
        }
        re_run(shape, 0);
        re_run(shape, 8);
    }
}
#endif
