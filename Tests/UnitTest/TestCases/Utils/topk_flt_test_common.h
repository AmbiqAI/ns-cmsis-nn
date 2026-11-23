/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The including suite defines TK_HALF, TK_KERNEL and TK_ARGMAX and includes its
 * generated topk_v2_<width>_data.h first. */

#include <arm_nnfunctions.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

/* Format fields are local to this independent binary64 reference.
 * An all-ones exponent means Inf or NaN. */
#if TK_HALF
    #define TK_TYPE float16_t
    #define TK_BITS uint16_t
    #define TK_SIGN_MASK UINT32_C(0x8000)
    #define TK_EXPONENT_MASK UINT32_C(0x7c00)
    #define TK_ONE_BITS UINT32_C(0x3c00)
    #define TK_TWO_BITS UINT32_C(0x4000)
    #define TK_MAX_BITS UINT32_C(0x7bff)
    #define TK_FRAC 10
    #define TK_BIAS 15
#else
    #define TK_TYPE float32_t
    #define TK_BITS uint32_t
    #define TK_SIGN_MASK UINT32_C(0x80000000)
    #define TK_EXPONENT_MASK UINT32_C(0x7f800000)
    #define TK_ONE_BITS UINT32_C(0x3f800000)
    #define TK_TWO_BITS UINT32_C(0x40000000)
    #define TK_MAX_BITS UINT32_C(0x7f7fffff)
    #define TK_FRAC 23
    #define TK_BIAS 127
#endif
#define TK_MAGNITUDE_MASK (TK_SIGN_MASK - 1)
#define TK_FRACTION_MASK ((UINT32_C(1) << TK_FRAC) - 1)
#define TK_QUIET ((TK_FRACTION_MASK >> 1) + 1)
#define TK_INF TK_EXPONENT_MASK
#define TK_NINF (TK_SIGN_MASK | TK_EXPONENT_MASK)
#define TK_CAP 4096
#define TK_GUARD 8
#define TK_SENTINEL INT32_C(0x5a5a5a5a)
#define TK_BYTE 0xa5

/* The arg-extrema goldens, for k = 1 against arm_argmax_* (their AE_BITS is ours). */
#define AE_BITS TK_BITS
#if TK_HALF
    #include "arg_extrema_f16_data.h"
#else
    #include "arg_extrema_f32_data.h"
#endif

static TK_TYPE tk_storage[TK_CAP + 2 * TK_GUARD];
static TK_BITS tk_bits[TK_CAP];
static TK_TYPE tk_values[TK_CAP + 2 * TK_GUARD];
static int32_t tk_indices[TK_CAP + 2 * TK_GUARD];
static int32_t tk_expected[TK_CAP];
static TK_BITS tk_expected_bits[TK_CAP];
static int32_t tk_order[TK_CAP];
static int32_t tk_merge[TK_CAP];
static double tk_decoded[TK_CAP];
static unsigned char tk_byte_image[sizeof(tk_values)];

static cmsis_nn_dims tk_dims(const int32_t d[4])
{
    const cmsis_nn_dims dims = {d[0], d[1], d[2], d[3]};
    return dims;
}

/* Decode directly into binary64, independently of the kernel's integer keys.
 * Every finite binary16 and binary32 value is normal in binary64, including
 * subnormals. */
static double tk_value(uint32_t bits)
{
    const uint32_t magnitude = bits & TK_MAGNITUDE_MASK;
    const uint32_t fraction = magnitude & TK_FRACTION_MASK;
    const int exponent = (int)(magnitude >> TK_FRAC);
    double value;
    if (magnitude == TK_EXPONENT_MASK)
    {
        value = INFINITY;
    }
    else if (exponent == 0)
    {
        value = ldexp((double)fraction, 1 - TK_BIAS - TK_FRAC);
    }
    else
    {
        value = ldexp((double)((UINT32_C(1) << TK_FRAC) + fraction), exponent - TK_BIAS - TK_FRAC);
    }
    return (bits & TK_SIGN_MASK) ? -value : value;
}

static int tk_is_nan(uint32_t bits)
{
    return (bits & TK_EXPONENT_MASK) == TK_EXPONENT_MASK && (bits & TK_FRACTION_MASK) != 0;
}

/* The order of the contract, on decoded values: numbers descending (+0 equals -0),
 * every NaN after every number, ties and NaNs by ascending index. */
static int tk_before(int a, int b)
{
    const int nan_a = tk_is_nan(tk_bits[a]);
    const int nan_b = tk_is_nan(tk_bits[b]);
    if (nan_a != nan_b)
        return nan_b;
    if (!nan_a)
    {
        if (tk_decoded[a] > tk_decoded[b])
            return 1;
        if (tk_decoded[a] < tk_decoded[b])
            return 0;
    }
    return a < b;
}

/* A stable bottom-up merge sort of one row's element offsets. */
static void tk_sort_row(int first, int c)
{
    for (int i = 0; i < c; ++i)
        tk_order[first + i] = first + i;
    for (int width = 1; width < c; width *= 2)
    {
        for (int lo = 0; lo < c; lo += 2 * width)
        {
            const int mid = lo + width < c ? lo + width : c;
            const int hi = lo + 2 * width < c ? lo + 2 * width : c;
            int i = lo, j = mid, o = lo;
            while (i < mid && j < hi)
                tk_merge[first + o++] =
                    tk_before(tk_order[first + j], tk_order[first + i]) ? tk_order[first + j++] : tk_order[first + i++];
            while (i < mid)
                tk_merge[first + o++] = tk_order[first + i++];
            while (j < hi)
                tk_merge[first + o++] = tk_order[first + j++];
        }
        for (int i = 0; i < c; ++i)
            tk_order[first + i] = tk_merge[first + i];
    }
}

/* Sort every row of tk_bits once; tk_expect then slices any k. */
static void tk_reference(const int32_t d[4])
{
    const int rows = d[0] * d[1] * d[2];
    const int c = d[3];
    for (int i = 0; i < rows * c; ++i)
        tk_decoded[i] = tk_value(tk_bits[i]);
    for (int r = 0; r < rows; ++r)
        tk_sort_row(r * c, c);
}

static void tk_expect(const int32_t d[4], int k)
{
    const int rows = d[0] * d[1] * d[2];
    const int c = d[3];
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j)
        {
            const int source = tk_order[r * c + j];
            tk_expected[r * k + j] = source - r * c;
            tk_expected_bits[r * k + j] = tk_bits[source];
        }
}

/* One call on tk_bits with guards on every buffer: the status, the indices, the value
 * bits (memcmp, never a float compare), an untouched input and untouched sentinels. */
static void tk_call(const int32_t d[4], int k)
{
    const cmsis_nn_dims dims = tk_dims(d);
    const int count = d[0] * d[1] * d[2] * d[3];
    const int out_count = d[0] * d[1] * d[2] * k;
    TEST_ASSERT_LESS_OR_EQUAL_INT(TK_CAP, count);
    TEST_ASSERT_LESS_OR_EQUAL_INT(TK_CAP, out_count);
    memset(tk_storage, TK_BYTE, sizeof(tk_storage));
    memcpy(tk_storage + TK_GUARD, tk_bits, (size_t)count * sizeof(TK_BITS));
    memset(tk_values, TK_BYTE, sizeof(tk_values));
    for (int i = 0; i < TK_CAP + 2 * TK_GUARD; ++i)
        tk_indices[i] = TK_SENTINEL;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      TK_KERNEL(tk_storage + TK_GUARD, &dims, k, tk_values + TK_GUARD, tk_indices + TK_GUARD));
    if (out_count)
    {
        TEST_ASSERT_EQUAL_INT32_ARRAY(tk_expected, tk_indices + TK_GUARD, out_count);
        TEST_ASSERT_EQUAL_MEMORY(tk_expected_bits, tk_values + TK_GUARD, (size_t)out_count * sizeof(TK_BITS));
    }
    TEST_ASSERT_EQUAL_MEMORY(tk_bits, tk_storage + TK_GUARD, (size_t)count * sizeof(TK_BITS));
    const unsigned char *input_bytes = (const unsigned char *)tk_storage;
    for (size_t i = 0; i < TK_GUARD * sizeof(TK_TYPE); ++i)
        TEST_ASSERT_EQUAL_HEX8(TK_BYTE, input_bytes[i]);
    for (size_t i = (TK_GUARD + (size_t)count) * sizeof(TK_TYPE); i < sizeof(tk_storage); ++i)
        TEST_ASSERT_EQUAL_HEX8(TK_BYTE, input_bytes[i]);
    const unsigned char *value_bytes = (const unsigned char *)tk_values;
    for (size_t i = 0; i < TK_GUARD * sizeof(TK_TYPE); ++i)
        TEST_ASSERT_EQUAL_HEX8(TK_BYTE, value_bytes[i]);
    for (size_t i = (TK_GUARD + (size_t)out_count) * sizeof(TK_TYPE); i < sizeof(tk_values); ++i)
        TEST_ASSERT_EQUAL_HEX8(TK_BYTE, value_bytes[i]);
    for (int i = 0; i < TK_GUARD; ++i)
        TEST_ASSERT_EQUAL_INT32(TK_SENTINEL, tk_indices[i]);
    for (int i = TK_GUARD + out_count; i < TK_CAP + 2 * TK_GUARD; ++i)
        TEST_ASSERT_EQUAL_INT32(TK_SENTINEL, tk_indices[i]);
}

/* The reference once per row order, then every k of the list. */
static void tk_check_ks(const int32_t d[4], const int32_t *ks, int count)
{
    tk_reference(d);
    for (int i = 0; i < count; ++i)
    {
        tk_expect(d, ks[i]);
        tk_call(d, ks[i]);
    }
}

static void tk_check_all_k(const int32_t d[4])
{
    tk_reference(d);
    for (int k = 0; k <= d[3]; ++k)
    {
        tk_expect(d, k);
        tk_call(d, k);
    }
}

static uint32_t tk_xorshift(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

/* Values with ties, both zeros, both infinities, subnormals, the largest finite values
 * and NaNs of every kind; a random draw adds arbitrary patterns. */
static const uint32_t tk_pool[] = {TK_ONE_BITS,
                                   TK_ONE_BITS,
                                   TK_TWO_BITS,
                                   TK_SIGN_MASK | TK_ONE_BITS,
                                   0,
                                   TK_SIGN_MASK,
                                   TK_INF,
                                   TK_NINF,
                                   1,
                                   TK_SIGN_MASK | 1,
                                   TK_MAX_BITS,
                                   TK_SIGN_MASK | TK_MAX_BITS,
                                   TK_EXPONENT_MASK | TK_QUIET,
                                   TK_SIGN_MASK | TK_EXPONENT_MASK | 1,
                                   TK_TWO_BITS,
                                   TK_FRACTION_MASK};

static uint32_t tk_draw(uint32_t *state)
{
    const uint32_t r = tk_xorshift(state);
    if ((r & 7) == 0)
        return tk_xorshift(state) & (TK_SIGN_MASK | TK_MAGNITUDE_MASK);
    return tk_pool[(r >> 3) % (sizeof(tk_pool) / sizeof(tk_pool[0]))];
}

/* The generated goldens: the interpreter's rows (as literals, in the f32 suite) and the
 * float64 reference's specials and random rows. */
static void tk_golden(void)
{
    for (int g = 0; g < TK_GOLDEN_CASES; ++g)
    {
        const tk_golden_case *gc = &tk_golden_cases[g];
        const int count = gc->dims[0] * gc->dims[1] * gc->dims[2] * gc->dims[3];
        const int out_count = gc->dims[0] * gc->dims[1] * gc->dims[2] * gc->k;
        memcpy(tk_bits, gc->input, (size_t)count * sizeof(TK_BITS));
        memcpy(tk_expected, gc->indices, (size_t)out_count * sizeof(int32_t));
        memcpy(tk_expected_bits, gc->values, (size_t)out_count * sizeof(TK_BITS));
        tk_call(gc->dims, gc->k);
        /* The independent C reference agrees with the generated data. */
        tk_reference(gc->dims);
        tk_expect(gc->dims, gc->k);
        TEST_ASSERT_EQUAL_INT32_ARRAY(gc->indices, tk_expected, out_count);
        TEST_ASSERT_EQUAL_MEMORY(gc->values, tk_expected_bits, (size_t)out_count * sizeof(TK_BITS));
    }
}

/* Every k from 0 to c for c in 1..9, over rows independent in n, h and w. */
static void tk_every_k(void)
{
    uint32_t state = UINT32_C(0x2545f491);
    for (int c = 1; c <= 9; ++c)
        for (int batch = 0; batch < 16; ++batch)
        {
            const int32_t d[4] = {2, 2, 2, c};
            for (int i = 0; i < 8 * c; ++i)
                tk_bits[i] = (TK_BITS)tk_draw(&state);
            tk_check_all_k(d);
        }
    const int32_t shapes[][4] = {{2, 3, 2, 7}, {3, 2, 2, 5}, {2, 2, 3, 1}, {1, 3, 1, 9}, {5, 1, 2, 3}};
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
    {
        const int count = shapes[s][0] * shapes[s][1] * shapes[s][2] * shapes[s][3];
        for (int i = 0; i < count; ++i)
            tk_bits[i] = (TK_BITS)tk_draw(&state);
        tk_check_all_k(shapes[s]);
    }
}

/* c = 16, 17, 256 and 4 096 at k in {1, 2, 5, 16}. */
static void tk_wide(void)
{
    const int32_t shapes[][4] = {{2, 1, 3, 16}, {2, 1, 3, 17}, {1, 2, 2, 256}, {1, 1, 1, 4096}};
    const int32_t ks[] = {1, 2, 5, 16};
    uint32_t state = UINT32_C(0x9e3779b9);
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
        for (int pass = 0; pass < 2; ++pass)
        {
            const int count = shapes[s][0] * shapes[s][1] * shapes[s][2] * shapes[s][3];
            for (int i = 0; i < count; ++i)
                tk_bits[i] = (TK_BITS)(pass ? tk_xorshift(&state) : tk_draw(&state));
            tk_check_ks(shapes[s], ks, 4);
        }
}

/* Every arrangement of six elements over {2, 1, +0, -0, quiet NaN, negative signalling
 * NaN}: ties at the top, ties straddling every k, both zeros and NaNs in every place. */
static void tk_ties(void)
{
    const uint32_t alphabet[6] = {TK_TWO_BITS,
                                  TK_ONE_BITS,
                                  0,
                                  TK_SIGN_MASK,
                                  TK_EXPONENT_MASK | TK_QUIET,
                                  TK_SIGN_MASK | TK_EXPONENT_MASK | 1};
    const int32_t d[4] = {6, 6, 6, 6};
    int pattern = 0;
    for (int batch = 0; batch < 216; ++batch)
    {
        for (int r = 0; r < 216; ++r, ++pattern)
        {
            int p = pattern;
            for (int i = 0; i < 6; ++i, p /= 6)
                tk_bits[r * 6 + i] = (TK_BITS)alphabet[p % 6];
        }
        tk_check_all_k(d);
    }
    TEST_ASSERT_EQUAL_INT(46656, pattern);
}

/* NaNs of every kind at every position; all-NaN rows; -inf rows with one number;
 * subnormals of both signs; the largest finite values. */
static void tk_special(void)
{
    const uint32_t nans[] = {TK_EXPONENT_MASK | TK_QUIET,
                             TK_SIGN_MASK | TK_EXPONENT_MASK | TK_QUIET,
                             TK_EXPONENT_MASK | 1,
                             TK_SIGN_MASK | TK_EXPONENT_MASK | 1,
                             TK_MAGNITUDE_MASK,
                             TK_SIGN_MASK | TK_MAGNITUDE_MASK,
                             TK_EXPONENT_MASK | TK_QUIET | 1,
                             TK_SIGN_MASK | TK_EXPONENT_MASK | (TK_QUIET - 1),
                             TK_EXPONENT_MASK | 2};
    const int n_nans = (int)(sizeof(nans) / sizeof(nans[0]));
    const uint32_t numbers[9] = {TK_TWO_BITS + 0x40,
                                 TK_ONE_BITS,
                                 TK_TWO_BITS + 0x40,
                                 TK_SIGN_MASK,
                                 0,
                                 TK_NINF,
                                 TK_TWO_BITS,
                                 TK_ONE_BITS,
                                 TK_SIGN_MASK | TK_ONE_BITS};
    const int32_t line[4] = {1, 1, 1, 9};
    for (int n = 0; n < n_nans; ++n)
        for (int position = 0; position < 9; ++position)
        {
            for (int i = 0; i < 9; ++i)
                tk_bits[i] = (TK_BITS)numbers[i];
            tk_bits[position] = (TK_BITS)nans[n];
            tk_check_all_k(line);
            /* Two NaNs: both keep their index order at the end of the row. */
            tk_bits[(position + 4) % 9] = (TK_BITS)nans[(n + 1) % n_nans];
            tk_check_all_k(line);
        }
    /* All-NaN rows, whatever the payloads and signs: indices 0..k-1, bits copied. */
    for (int shift = 0; shift < n_nans; ++shift)
    {
        for (int i = 0; i < 9; ++i)
            tk_bits[i] = (TK_BITS)nans[(i + shift) % n_nans];
        tk_check_all_k(line);
        tk_expect(line, 9);
        for (int j = 0; j < 9; ++j)
            TEST_ASSERT_EQUAL_INT32(j, tk_expected[j]);
    }
    /* -inf everywhere but one number, which comes first; -inf precedes a NaN. */
    const uint32_t lone[] = {0, TK_SIGN_MASK, TK_SIGN_MASK | TK_MAX_BITS, TK_SIGN_MASK | 1, TK_ONE_BITS, TK_NINF};
    for (unsigned v = 0; v < sizeof(lone) / sizeof(lone[0]); ++v)
        for (int position = 0; position < 9; ++position)
        {
            for (int i = 0; i < 9; ++i)
                tk_bits[i] = (TK_BITS)TK_NINF;
            tk_bits[position] = (TK_BITS)lone[v];
            tk_check_all_k(line);
            tk_bits[(position + 3) % 9] = (TK_BITS)nans[v % n_nans];
            tk_check_all_k(line);
        }
    /* Subnormals of both signs against both zeros and the smallest normals, and the
     * largest finite values against the infinities. */
    const uint32_t rows[][9] = {{1,
                                 TK_SIGN_MASK | 1,
                                 TK_FRACTION_MASK,
                                 TK_SIGN_MASK | TK_FRACTION_MASK,
                                 0,
                                 TK_SIGN_MASK,
                                 TK_FRACTION_MASK + 1,
                                 TK_SIGN_MASK | (TK_FRACTION_MASK + 1),
                                 1},
                                {TK_SIGN_MASK | 1, TK_SIGN_MASK, 0, 1, TK_SIGN_MASK | 2, 2, TK_SIGN_MASK | 1,
                                 TK_SIGN_MASK, 2},
                                {TK_MAX_BITS, TK_SIGN_MASK | TK_MAX_BITS, TK_INF, TK_NINF, TK_MAX_BITS - 1, TK_MAX_BITS,
                                 TK_SIGN_MASK | (TK_MAX_BITS - 1), TK_INF, TK_NINF},
                                {TK_NINF, TK_SIGN_MASK | TK_MAX_BITS, TK_EXPONENT_MASK | 1, TK_MAX_BITS, TK_INF, 0,
                                 TK_SIGN_MASK | 1, TK_SIGN_MASK | TK_MAGNITUDE_MASK, TK_SIGN_MASK | TK_MAX_BITS}};
    for (unsigned r = 0; r < sizeof(rows) / sizeof(rows[0]); ++r)
    {
        for (int i = 0; i < 9; ++i)
            tk_bits[i] = (TK_BITS)rows[r][i];
        tk_check_all_k(line);
    }
}

/* Each bit pattern of the width (every binary16 one; random binary32 ones) as one
 * competitor against fixed others, at every position of a five-element row. */
static void tk_patterns(void)
{
    const int32_t d[4] = {1, 1, 256, 5};
    const int32_t ks[] = {1, 2, 3, 5};
#if TK_HALF
    for (uint32_t start = 0; start < 65536; start += 256)
#else
    uint32_t state = UINT32_C(0x9e3779b9);
    for (uint32_t start = 0; start < 8192; start += 256)
#endif
    {
        for (int rotation = 0; rotation < 5; ++rotation)
        {
            for (uint32_t c = 0; c < 256; ++c)
            {
#if TK_HALF
                const uint32_t bits = start + c;
#else
                const uint32_t bits = tk_xorshift(&state);
#endif
                const uint32_t competitors[5] = {bits,
                                                 TK_SIGN_MASK * (c & 1),
                                                 TK_ONE_BITS | (TK_SIGN_MASK * ((c >> 1) & 1)),
                                                 (c & 4) ? (TK_SIGN_MASK | TK_EXPONENT_MASK | 1)
                                                         : (TK_EXPONENT_MASK | TK_QUIET),
                                                 TK_NINF};
                for (int i = 0; i < 5; ++i)
                    tk_bits[c * 5 + i] = (TK_BITS)competitors[(i + rotation) % 5];
            }
            tk_check_ks(d, ks, 4);
        }
    }
}

/* k = 1 writes arm_argmax_*'s index over the innermost axis and the value there. */
static void tk_argmax_one(const int32_t d[4])
{
    const cmsis_nn_dims dims = tk_dims(d);
    const int rows = d[0] * d[1] * d[2];
    const int count = rows * d[3];
    memcpy(tk_storage, tk_bits, (size_t)count * sizeof(TK_BITS));
    for (int i = 0; i < rows; ++i)
        tk_expected[i] = -1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, TK_ARGMAX(tk_storage, &dims, 3, tk_expected));
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, TK_KERNEL(tk_storage, &dims, 1, tk_values, tk_indices));
    TEST_ASSERT_EQUAL_INT32_ARRAY(tk_expected, tk_indices, rows);
    for (int r = 0; r < rows; ++r)
        tk_expected_bits[r] = tk_bits[r * d[3] + tk_expected[r]];
    TEST_ASSERT_EQUAL_MEMORY(tk_expected_bits, tk_values, (size_t)rows * sizeof(TK_BITS));
}

static void tk_argmax(void)
{
    /* The arg-extrema goldens, reduced over their innermost axis. */
    const int32_t golden[4] = {2, 3, 2, 5};
    memcpy(tk_bits, ae_golden_input, sizeof(ae_golden_input));
    tk_argmax_one(golden);
    TEST_ASSERT_EQUAL_INT32_ARRAY(ae_max_axis3, tk_indices, 12);
    /* The arg-extrema suite's ae_patterns triples, laid along the innermost axis. */
    const int32_t d[4] = {1, 1, 256, 3};
#if TK_HALF
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
#if TK_HALF
                const uint32_t bits = start + c;
#else
                const uint32_t bits = tk_xorshift(&state);
#endif
                const uint32_t competitors[3] = {
                    bits, TK_SIGN_MASK * (c & 1), TK_ONE_BITS | (TK_SIGN_MASK * ((c >> 1) & 1))};
                for (int k = 0; k < 3; ++k)
                    tk_bits[c * 3 + k] = (TK_BITS)competitors[(k + rotation) % 3];
            }
            tk_argmax_one(d);
        }
    }
    /* And every golden row of this suite. */
    for (int g = 0; g < TK_GOLDEN_CASES; ++g)
    {
        const tk_golden_case *gc = &tk_golden_cases[g];
        const int count = gc->dims[0] * gc->dims[1] * gc->dims[2] * gc->dims[3];
        memcpy(tk_bits, gc->input, (size_t)count * sizeof(TK_BITS));
        tk_argmax_one(gc->dims);
    }
}

/* Zero rows or k = 0: success, nothing written, NULL pointers accepted. */
static void tk_empty(void)
{
    const int32_t shapes[][4] = {{2, 3, 4, 5}, {0, 3, 4, 5}, {2, 0, 4, 5}, {2, 3, 0, 5}, {2, 3, 4, 0}, {0, 0, 0, 0}};
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s)
    {
        const cmsis_nn_dims dims = tk_dims(shapes[s]);
        const int rows = shapes[s][0] * shapes[s][1] * shapes[s][2];
        for (int k = 0; k <= shapes[s][3]; ++k)
        {
            if (rows && k)
                continue;
            memset(tk_values, TK_BYTE, sizeof(tk_values));
            memset(tk_byte_image, TK_BYTE, sizeof(tk_byte_image));
            tk_indices[0] = TK_SENTINEL;
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, TK_KERNEL(tk_storage, &dims, k, tk_values, tk_indices));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, TK_KERNEL(NULL, &dims, k, NULL, NULL));
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, TK_KERNEL(NULL, &dims, k, tk_values, NULL));
            TEST_ASSERT_EQUAL_INT32(TK_SENTINEL, tk_indices[0]);
            TEST_ASSERT_EQUAL_MEMORY(tk_byte_image, tk_values, sizeof(tk_values));
        }
    }
    /* A zero extent suppresses the count even beside the largest extents. */
    const cmsis_nn_dims zero_product = {INT32_MAX, INT32_MAX, 0, 1};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, TK_KERNEL(NULL, &zero_product, 1, NULL, NULL));
    const cmsis_nn_dims zero_axis = {INT32_MAX, 1, 1, 0};
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, TK_KERNEL(NULL, &zero_axis, 0, NULL, NULL));
}

static void
tk_expect_refused(const cmsis_nn_dims *dims, int32_t k, const TK_TYPE *input, TK_TYPE *values, int32_t *indices)
{
    memset(tk_values, TK_BYTE, sizeof(tk_values));
    memset(tk_byte_image, TK_BYTE, sizeof(tk_byte_image));
    for (int i = 0; i < TK_CAP + 2 * TK_GUARD; ++i)
        tk_indices[i] = TK_SENTINEL;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, TK_KERNEL(input, dims, k, values, indices));
    TEST_ASSERT_EQUAL_MEMORY(tk_byte_image, tk_values, sizeof(tk_values));
    for (int i = 0; i < TK_CAP + 2 * TK_GUARD; ++i)
        TEST_ASSERT_EQUAL_INT32(TK_SENTINEL, tk_indices[i]);
}

static void tk_invalid(void)
{
    const cmsis_nn_dims valid = {1, 2, 1, 4};
    memset(tk_storage, 0, sizeof(tk_storage));
    tk_expect_refused(NULL, 1, tk_storage, tk_values, tk_indices);
    const int32_t ks[] = {5, -1, INT32_MIN, INT32_MAX};
    for (unsigned i = 0; i < sizeof(ks) / sizeof(ks[0]); ++i)
        tk_expect_refused(&valid, ks[i], tk_storage, tk_values, tk_indices);
    tk_expect_refused(&valid, 2, NULL, tk_values, tk_indices);
    tk_expect_refused(&valid, 2, tk_storage, NULL, tk_indices);
    tk_expect_refused(&valid, 2, tk_storage, tk_values, NULL);
    const cmsis_nn_dims negative[] = {{-1, 1, 1, 1},
                                      {1, -1, 1, 1},
                                      {1, 1, -1, 1},
                                      {1, 1, 1, -1},
                                      {0, -1, 1, 1},
                                      {1, 1, 0, -1},
                                      {INT32_MIN, 1, 1, 1}};
    for (unsigned i = 0; i < sizeof(negative) / sizeof(negative[0]); ++i)
    {
        tk_expect_refused(&negative[i], 0, tk_storage, tk_values, tk_indices);
        tk_expect_refused(&negative[i], 1, tk_storage, tk_values, tk_indices);
    }
    /* Byte counts above INT32_MAX: the input's, and at k = c the indices' too (four bytes
     * each, which exceeds a binary16 input's two: the last three shapes). */
    const cmsis_nn_dims overflow[] = {{1, 1, 1, INT32_MAX}, {65536, 32768, 1, 1}, {1, 1, 536870912, 1},
                                      {536870912, 1, 1, 1}, {1, 1, 268435456, 2}};
    for (unsigned i = 0; i < sizeof(overflow) / sizeof(overflow[0]); ++i)
        tk_expect_refused(&overflow[i], overflow[i].c, tk_storage, tk_values, tk_indices);
    /* At k = 0 only the input's count remains: it overflows in both widths for the first
     * two shapes and in binary32 alone for the other three, where binary16 succeeds. */
    for (unsigned i = 0; i < sizeof(overflow) / sizeof(overflow[0]); ++i)
    {
    #if TK_HALF
        if (i >= 2)
        {
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, TK_KERNEL(NULL, &overflow[i], 0, NULL, NULL));
            continue;
        }
    #endif
        tk_expect_refused(&overflow[i], 0, tk_storage, tk_values, tk_indices);
    }
    /* k above an empty innermost extent. */
    const cmsis_nn_dims empty_axis = {2, 3, 4, 0};
    tk_expect_refused(&empty_axis, 1, NULL, NULL, NULL);
}

static void tk_fp_controls(void)
{
#if defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
    uint32_t original;
    __asm volatile("vmrs %0, fpscr" : "=r"(original));
    /* FPSCR: FZ16[19], RMode[23:22], FZ[24], DN[25], AHP[26]. */
    const uint32_t controls = (UINT32_C(1) << 19) | (UINT32_C(3) << 22) | (UINT32_C(7) << 24);
    /* Cumulative exception flags IOC/DZC/OFC/UFC/IXC[4:0] and IDC[7]. */
    const uint32_t exceptions = UINT32_C(0x9f);
    const int32_t d[4] = {1, 1, 4, 6};
    /* Four rows of six: subnormals against both zeros, a NaN first and NaNs last. */
    const uint32_t cases[24] = {TK_EXPONENT_MASK | 1,
                                1,
                                TK_SIGN_MASK | 1,
                                TK_ONE_BITS,
                                0,
                                TK_SIGN_MASK,
                                0,
                                1,
                                TK_SIGN_MASK | 1,
                                TK_SIGN_MASK,
                                0,
                                TK_FRACTION_MASK,
                                TK_ONE_BITS,
                                TK_SIGN_MASK | TK_ONE_BITS,
                                TK_INF,
                                TK_NINF,
                                0,
                                TK_SIGN_MASK | TK_EXPONENT_MASK | 1,
                                1,
                                TK_SIGN_MASK | 1,
                                TK_EXPONENT_MASK | TK_QUIET,
                                TK_EXPONENT_MASK | 1,
                                TK_EXPONENT_MASK | 2,
                                2};
    const int32_t ks[] = {1, 3, 6};
    for (int i = 0; i < 24; ++i)
        tk_bits[i] = (TK_BITS)cases[i];
    /* The reference runs under the original controls, before any is changed. */
    tk_reference(d);
    const cmsis_nn_dims dims = tk_dims(d);
    TK_TYPE input[24];
    memcpy(input, tk_bits, sizeof(input));
    for (unsigned s = 0; s < sizeof(ks) / sizeof(ks[0]); ++s)
    {
        const int k = ks[s];
        tk_expect(d, k);
        for (uint32_t c = 0; c < 64; ++c)
        {
            const uint32_t mode = ((c & 1) << 19) | (((c >> 1) & 3) << 22) | (((c >> 3) & 7) << 24);
            const uint32_t wanted = (original & ~(controls | exceptions)) | mode | 2;
            int32_t indices[24];
            TK_TYPE values[24];
            memset(indices, 0x5a, sizeof(indices));
            memset(values, TK_BYTE, sizeof(values));
            __asm volatile("vmsr fpscr, %0" ::"r"(wanted) : "memory");
            const arm_cmsis_nn_status status = TK_KERNEL(input, &dims, k, values, indices);
            uint32_t after;
            __asm volatile("vmrs %0, fpscr" : "=r"(after));
            __asm volatile("vmsr fpscr, %0" ::"r"(original) : "memory");
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);
            TEST_ASSERT_EQUAL_INT32_ARRAY(tk_expected, indices, 4 * k);
            TEST_ASSERT_EQUAL_MEMORY(tk_expected_bits, values, (size_t)(4 * k) * sizeof(TK_BITS));
            TEST_ASSERT_EQUAL_HEX32(wanted & (controls | exceptions), after & (controls | exceptions));
        }
    }
#endif
}
