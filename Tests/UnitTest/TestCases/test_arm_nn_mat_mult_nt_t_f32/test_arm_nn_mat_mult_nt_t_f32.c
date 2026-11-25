/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_nn_mat_mult_nt_t_f32 after its body moved to Include/Internal/arm_nn_mat_mult_nt_t_flt.h.
 *
 * Differential cases: the entry against its body as it was (mat_mult_nt_t_f32_previous.c, compiled with the library's
 * own options), on the same inputs, each writing into its own guarded output. A case differs when the statuses differ,
 * when any bit of the two outputs differs over the span the case can write and MM_GUARD words either side, or when a
 * guard of the entry's output moved. The move changes no arithmetic, so every count of differences must be 0.
 *
 * Literal cases: the inline body compiled with literal extents, as an emitted operator compiles it, under the library's
 * options and under -std=c11 -O2 (mat_mult_nt_t_f32_literal_*.c), against a portable reference that rounds every
 * product and every sum in the order the scalar leg states (mm_reference). Where that TU holds the scalar leg under GCC
 * in an ISO C mode without fast-math (mml_*_exact) the outputs must equal the reference's bits; elsewhere the Helium
 * leg sums in its own order and the options may contract or reorder, and the cases count the outputs that differ and
 * state the largest difference relative to the output's term scale, |bias| plus the sum of |lhs * rhs|. No tolerance is
 * applied to those counts: they are reported, not judged. */

#include "mat_mult_nt_t_f32_literal.h"
#include "mat_mult_nt_t_f32_previous.h"
#include <arm_nnfunctions.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#define MM_MAX_LHS_ROWS 3
#define MM_MAX_RHS_ROWS 2048
#define MM_MAX_COLS 64
#define MM_MAX_LITERAL_COLS 738
#define MM_LHS_CAP (MM_MAX_LHS_ROWS * MM_MAX_LITERAL_COLS)
#define MM_RHS_CAP (MM_MAX_RHS_ROWS * MM_MAX_COLS)
#define MM_DST_CAP (MM_MAX_LHS_ROWS * (MM_MAX_RHS_ROWS + 8))
#define MM_GUARD 16
#define MM_SENTINEL UINT32_C(0xa5a5a5a5)
#define MM_REPORTED 8

/* 0.5 MB of rhs and the outputs go to SRAM under the GCC Corstone-300 linker script, which places .bss.NoInit there;
 * elsewhere they join .bss. */
#if defined(USING_FVP_CORSTONE_300)
    #define MM_SECTION __attribute__((section(".bss.NoInit")))
#else
    #define MM_SECTION
#endif

static float32_t mm_lhs[MM_LHS_CAP] MM_SECTION;
static float32_t mm_rhs[MM_RHS_CAP > 3 * MM_MAX_LITERAL_COLS ? MM_RHS_CAP : 3 * MM_MAX_LITERAL_COLS] MM_SECTION;
static float32_t mm_bias[MM_MAX_RHS_ROWS] MM_SECTION;
static float32_t mm_new[MM_DST_CAP + 2 * MM_GUARD] MM_SECTION;
static float32_t mm_old[MM_DST_CAP + 2 * MM_GUARD] MM_SECTION;
static float32_t mm_ref[MM_DST_CAP] MM_SECTION;
static float32_t mm_scale[MM_DST_CAP] MM_SECTION;
static unsigned long mm_cases;
static unsigned long mm_differences;
static unsigned long mm_payload_only;
static char mm_message[256];

typedef struct
{
    float32_t min;
    float32_t max;
} mm_clamp;

/* The full range an operator without an activation passes, the infinite one, a ReLU, a narrow band and the two zeros.
 */
static mm_clamp mm_clamps(int i)
{
    switch (i)
    {
    case 0:
        return (mm_clamp){-FLT_MAX, FLT_MAX};
    case 1:
        return (mm_clamp){-INFINITY, INFINITY};
    case 2:
        return (mm_clamp){0.0f, FLT_MAX};
    case 3:
        return (mm_clamp){-1.5f, 1.5f};
    default:
        return (mm_clamp){-0.0f, 0.0f};
    }
}
#define MM_CLAMPS 5

static uint32_t mm_hash(uint32_t i, uint32_t seed)
{
    uint32_t h = i * 2654435761u + seed * 40503u + 0x9e3779b9u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;
    return h;
}

static float32_t mm_from_bits(uint32_t bits)
{
    float32_t x;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static uint32_t mm_bits(float32_t x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static int mm_is_nan(float32_t x) { return (mm_bits(x) & 0x7fffffffu) > 0x7f800000u; }

static int mm_is_finite(float32_t x) { return (mm_bits(x) & 0x7f800000u) != 0x7f800000u; }

/* NaN, both infinities, both zeros, the least and the greatest subnormal, and the extreme normals. */
static const uint32_t mm_specials[] = {0x7fc00000u,
                                       0x7f800000u,
                                       0xff800000u,
                                       0x00000000u,
                                       0x80000000u,
                                       0x00000001u,
                                       0x807fffffu,
                                       0x7f7fffffu,
                                       0xff7fffffu};
#define MM_SPECIALS (sizeof(mm_specials) / sizeof(mm_specials[0]))

/* Patterns: 0, values in [-2, 2) with a full mantissa; 1, the same with one element in eight a special value; 2,
 * random bit patterns, which hold NaN payloads, infinities, subnormals and overflowing products. */
#define MM_PATTERNS 3
static float32_t mm_value(uint32_t i, uint32_t seed, int pattern)
{
    const uint32_t h = mm_hash(i, seed);
    if (pattern == 2)
    {
        return mm_from_bits(h);
    }
    if (pattern == 1 && (h >> 29) == 0)
    {
        return mm_from_bits(mm_specials[(h >> 8) % MM_SPECIALS]);
    }
    return (float32_t)(int32_t)(h & 0xffffffu) * 0x1p-22f - 2.0f;
}

static void mm_fill(int32_t lhs_rows, int32_t rhs_rows, int32_t rhs_cols, int pattern, uint32_t seed)
{
    for (int32_t i = 0; i < lhs_rows * rhs_cols; ++i)
    {
        mm_lhs[i] = mm_value((uint32_t)i, seed * 3u + 1u, pattern);
    }
    for (int32_t i = 0; i < rhs_rows * rhs_cols; ++i)
    {
        mm_rhs[i] = mm_value((uint32_t)i, seed * 3u + 2u, pattern);
    }
    for (int32_t i = 0; i < rhs_rows; ++i)
    {
        mm_bias[i] = mm_value((uint32_t)i, seed * 3u + 3u, pattern);
    }
}

static void mm_reset(float32_t *out, int32_t span)
{
    for (int32_t i = 0; i < span + 2 * MM_GUARD; ++i)
    {
        out[i] = mm_from_bits(MM_SENTINEL);
    }
}

/* Whether the two outputs differ over span + guards, or a guard of the entry's output moved: 0 when every bit is the
 * same, MM_PAYLOAD_ONLY when the only differing words are NaN in both (the declaration guarantees the NaN-ness of an
 * output, not its payload), 1 otherwise. */
#define MM_PAYLOAD_ONLY 2
static int mm_outputs_differ(int32_t span)
{
    for (int32_t i = 0; i < MM_GUARD; ++i)
    {
        if (mm_bits(mm_new[i]) != MM_SENTINEL || mm_bits(mm_new[MM_GUARD + span + i]) != MM_SENTINEL)
        {
            return 1;
        }
    }
    int found = 0;
    for (int32_t i = 0; i < span + 2 * MM_GUARD; ++i)
    {
        if (mm_bits(mm_new[i]) == mm_bits(mm_old[i]))
        {
            continue;
        }
        if (!(mm_is_nan(mm_new[i]) && mm_is_nan(mm_old[i])))
        {
            return 1;
        }
        found = MM_PAYLOAD_ONLY;
    }
    return found;
}

static void mm_report_difference(const char *what, int32_t l, int32_t r, int32_t k, int32_t o, int b, int clamp, int p)
{
    if (mm_differences <= MM_REPORTED)
    {
        snprintf(mm_message,
                 sizeof(mm_message),
                 "%s differs: lhs_rows %ld rhs_rows %ld rhs_cols %ld offset %ld bias %d clamp %d pattern %d",
                 what,
                 (long)l,
                 (long)r,
                 (long)k,
                 (long)o,
                 b,
                 clamp,
                 p);
        TEST_MESSAGE(mm_message);
    }
}

/* One differential case: inputs already filled. */
static void mm_differential_case(int32_t l, int32_t r, int32_t k, int32_t o, int use_bias, int clamp, int pattern)
{
    const mm_clamp range = mm_clamps(clamp);
    const int32_t span = (l - 1) * o + r;
    mm_reset(mm_new, span);
    mm_reset(mm_old, span);
    const arm_cmsis_nn_status s_new = arm_nn_mat_mult_nt_t_f32(
        mm_lhs, mm_rhs, use_bias ? mm_bias : NULL, mm_new + MM_GUARD, l, r, k, o, range.min, range.max);
    const arm_cmsis_nn_status s_old = mp_mat_mult_nt_t_f32(
        mm_lhs, mm_rhs, use_bias ? mm_bias : NULL, mm_old + MM_GUARD, l, r, k, o, range.min, range.max);
    ++mm_cases;
    const int differ = mm_outputs_differ(span);
    if (s_new != s_old || s_new != ARM_CMSIS_NN_SUCCESS || differ == 1)
    {
        ++mm_differences;
        mm_report_difference("entry", l, r, k, o, use_bias, clamp, pattern);
    }
    else if (differ == MM_PAYLOAD_ONLY)
    {
        ++mm_payload_only;
    }
}

/* Rows 1 to 17 (each leg's blocks of four and eight and their remainders), columns 1 to 17, 30 to 33 (either side of
 * the Helium contiguous kernel's K = 32) and 64; one to three lhs rows; a packed and a strided output; with and without
 * bias; every clamp and pattern. */
static const int32_t mm_cols[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 64};
#define MM_COLS_COUNT ((int32_t)(sizeof(mm_cols) / sizeof(mm_cols[0])))

void mm_differential_small(void)
{
    mm_cases = 0;
    mm_differences = 0;
    mm_payload_only = 0;
    for (int32_t r = 1; r <= 17; ++r)
    {
        for (int32_t ki = 0; ki < MM_COLS_COUNT; ++ki)
        {
            const int32_t k = mm_cols[ki];
            for (int32_t l = 1; l <= MM_MAX_LHS_ROWS; ++l)
            {
                for (int pattern = 0; pattern < MM_PATTERNS; ++pattern)
                {
                    mm_fill(l, r, k, pattern, (uint32_t)(r * 131 + k * 7 + l));
                    for (int32_t stride = 0; stride < 2; ++stride)
                    {
                        const int32_t o = r + 3 * stride;
                        for (int use_bias = 0; use_bias < 2; ++use_bias)
                        {
                            for (int clamp = 0; clamp < MM_CLAMPS; ++clamp)
                            {
                                mm_differential_case(l, r, k, o, use_bias, clamp, pattern);
                            }
                        }
                    }
                }
            }
        }
    }
    snprintf(mm_message,
             sizeof(mm_message),
             "differential rows 1-17: %lu cases, %lu differences; %lu cases differ in NaN payloads alone",
             mm_cases,
             mm_differences,
             mm_payload_only);
    TEST_MESSAGE(mm_message);
    TEST_ASSERT_EQUAL_UINT32(0, mm_differences);
}

/* 2,048 rhs rows, the knn probe's product, at every column count; one and two lhs rows, the full-range and ReLU
 * clamps. */
void mm_differential_2048(void)
{
    mm_cases = 0;
    mm_differences = 0;
    mm_payload_only = 0;
    const int32_t r = MM_MAX_RHS_ROWS;
    for (int32_t ki = 0; ki < MM_COLS_COUNT; ++ki)
    {
        const int32_t k = mm_cols[ki];
        for (int32_t l = 1; l <= 2; ++l)
        {
            for (int pattern = 0; pattern < MM_PATTERNS; ++pattern)
            {
                mm_fill(l, r, k, pattern, (uint32_t)(r * 131 + k * 7 + l));
                for (int use_bias = 0; use_bias < 2; ++use_bias)
                {
                    for (int clamp = 0; clamp < 3; clamp += 2)
                    {
                        mm_differential_case(l, r, k, r, use_bias, clamp, pattern);
                    }
                }
            }
        }
    }
    snprintf(mm_message,
             sizeof(mm_message),
             "differential rows 2048: %lu cases, %lu differences; %lu cases differ in NaN payloads alone",
             mm_cases,
             mm_differences,
             mm_payload_only);
    TEST_MESSAGE(mm_message);
    TEST_ASSERT_EQUAL_UINT32(0, mm_differences);
}

/* Every argument the entry refuses: each pointer missing, and each extent and the offset at 0, -1 and INT32_MIN. Both
 * bodies must refuse alike and write nothing. */
void mm_differential_refusals(void)
{
    static const int32_t bad[] = {0, -1, INT32_MIN};
    mm_cases = 0;
    mm_differences = 0;
    mm_payload_only = 0;
    mm_fill(2, 5, 7, 0, 99u);
    for (int which = 0; which < 7; ++which)
    {
        for (int32_t bi = 0; bi < 3; ++bi)
        {
            int32_t ext[4] = {2, 5, 7, 5};
            const float32_t *lhs = mm_lhs;
            const float32_t *rhs = mm_rhs;
            int null_dst = 0;
            if (which < 4)
            {
                ext[which] = bad[bi];
            }
            else if (bi > 0)
            {
                continue;
            }
            else if (which == 4)
            {
                lhs = NULL;
            }
            else if (which == 5)
            {
                rhs = NULL;
            }
            else
            {
                null_dst = 1;
            }
            const int32_t span = 2 * 5 + 5;
            mm_reset(mm_new, span);
            mm_reset(mm_old, span);
            const arm_cmsis_nn_status s_new = arm_nn_mat_mult_nt_t_f32(lhs,
                                                                       rhs,
                                                                       mm_bias,
                                                                       null_dst ? NULL : mm_new + MM_GUARD,
                                                                       ext[0],
                                                                       ext[1],
                                                                       ext[2],
                                                                       ext[3],
                                                                       -FLT_MAX,
                                                                       FLT_MAX);
            const arm_cmsis_nn_status s_old = mp_mat_mult_nt_t_f32(lhs,
                                                                   rhs,
                                                                   mm_bias,
                                                                   null_dst ? NULL : mm_old + MM_GUARD,
                                                                   ext[0],
                                                                   ext[1],
                                                                   ext[2],
                                                                   ext[3],
                                                                   -FLT_MAX,
                                                                   FLT_MAX);
            ++mm_cases;
            int moved = 0;
            for (int32_t i = 0; i < span + 2 * MM_GUARD; ++i)
            {
                moved |= mm_bits(mm_new[i]) != MM_SENTINEL;
            }
            if (s_new != s_old || s_new != ARM_CMSIS_NN_ARG_ERROR || moved || mm_outputs_differ(span) != 0)
            {
                ++mm_differences;
                mm_report_difference("refusal", ext[0], ext[1], ext[2], ext[3], 1, 0, which);
            }
        }
    }
    snprintf(
        mm_message, sizeof(mm_message), "differential refusals: %lu cases, %lu differences", mm_cases, mm_differences);
    TEST_MESSAGE(mm_message);
    TEST_ASSERT_EQUAL_UINT32(0, mm_differences);
}

/* The portable reference, in the scalar leg's stated order: each lhs row's dot product with each rhs row from 0.0f,
 * one product rounded and then one sum rounded at a time in k order, added to the bias (or 0.0f) at the end, then
 * clamped with a NaN passed through. Every rounding point is a volatile, so no option contracts, reorders or vectorises
 * it. mm_scale holds each output's term scale. */
static void
mm_reference(int32_t l, int32_t r, int32_t k, int32_t o, const float32_t *bias, float32_t act_min, float32_t act_max)
{
    volatile float32_t zero = 0.0f;
    for (int32_t i = 0; i < l; ++i)
    {
        for (int32_t c = 0; c < r; ++c)
        {
            volatile float32_t dot = zero;
            volatile float32_t scale = zero;
            for (int32_t j = 0; j < k; ++j)
            {
                volatile float32_t product = mm_lhs[i * k + j] * mm_rhs[c * k + j];
                dot = dot + product;
                scale = scale + fabsf(product);
            }
            volatile float32_t acc = (bias ? bias[c] : zero) + dot;
            scale = scale + (bias ? fabsf(bias[c]) : zero);
            const float32_t a = acc;
            float32_t y = a < act_max ? a : act_max;
            y = y > act_min ? y : act_min;
            mm_ref[i * o + c] = mm_is_nan(a) ? a : y;
            mm_scale[i * o + c] = scale;
        }
    }
}

typedef struct
{
    int32_t l;
    int32_t r;
    int32_t k;
    int32_t o;
    int clamp; /* -1: passed at run time; else the clamp index the literals equal */
    mml_fn library;
    mml_fn c11;
} mm_literal_shape;

#define MM_CLAMP_ARG (-1)
#define MM_CLAMP_FULL 0
#define MM_CLAMP_RELU 2
#define MM_LITERAL_ENTRY(l, r, k, o, clamp)                                                                            \
    {l, r, k, o, MM_CLAMP_##clamp, MML_NAME(library, l, r, k, o, clamp), MML_NAME(c11, l, r, k, o, clamp)},
static const mm_literal_shape mm_literal_shapes[] = {MML_SHAPES(MM_LITERAL_ENTRY)};
#define MM_LITERAL_SHAPES ((int32_t)(sizeof(mm_literal_shapes) / sizeof(mm_literal_shapes[0])))

typedef struct
{
    unsigned long outputs;
    unsigned long bits_differ;
    unsigned long nan_differ;
    unsigned long finite_differ;
    unsigned long entry_differ;
    float32_t largest;
} mm_tally;

static void mm_compare_literal(const mm_literal_shape *s, int list, int use_bias, int clamp, mm_tally *t)
{
    const mm_clamp range = mm_clamps(clamp);
    const int32_t span = (s->l - 1) * s->o + s->r;
    const mml_fn fn = list ? s->c11 : s->library;
    mm_reset(mm_new, span);
    mm_reset(mm_old, span);
    const float32_t *bias = use_bias ? mm_bias : NULL;
    const arm_cmsis_nn_status status = fn(mm_lhs, mm_rhs, bias, mm_new + MM_GUARD, range.min, range.max);
    const arm_cmsis_nn_status entry =
        arm_nn_mat_mult_nt_t_f32(mm_lhs, mm_rhs, bias, mm_old + MM_GUARD, s->l, s->r, s->k, s->o, range.min, range.max);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, entry);
    mm_reference(s->l, s->r, s->k, s->o, bias, range.min, range.max);
    for (int32_t i = 0; i < MM_GUARD; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(MM_SENTINEL, mm_bits(mm_new[i]));
        TEST_ASSERT_EQUAL_HEX32(MM_SENTINEL, mm_bits(mm_new[MM_GUARD + span + i]));
    }
    for (int32_t i = 0; i < s->l; ++i)
    {
        for (int32_t c = s->r; c < s->o; ++c)
        {
            TEST_ASSERT_EQUAL_HEX32(MM_SENTINEL, mm_bits(mm_new[MM_GUARD + i * s->o + c]));
        }
        for (int32_t c = 0; c < s->r; ++c)
        {
            const int32_t at = i * s->o + c;
            const float32_t got = mm_new[MM_GUARD + at];
            const float32_t want = mm_ref[at];
            ++t->outputs;
            t->entry_differ += mm_bits(got) != mm_bits(mm_old[MM_GUARD + at]);
            if (mm_bits(got) == mm_bits(want) || (mm_is_nan(got) && mm_is_nan(want)))
            {
                continue;
            }
            ++t->bits_differ;
            if (mm_is_nan(got) != mm_is_nan(want))
            {
                ++t->nan_differ;
            }
            else if (mm_is_finite(got) != mm_is_finite(want))
            {
                ++t->finite_differ;
            }
            else if (mm_is_finite(mm_scale[at]) && mm_scale[at] > 0.0f)
            {
                const float32_t relative = fabsf(got - want) / mm_scale[at];
                t->largest = relative > t->largest ? relative : t->largest;
            }
        }
    }
}

static const char *const mm_pattern_names[MM_PATTERNS] = {"normal values", "special values", "random bits"};

static void mm_literal(int list)
{
    mm_tally t[MM_PATTERNS];
    memset(t, 0, sizeof(t));
    for (int32_t si = 0; si < MM_LITERAL_SHAPES; ++si)
    {
        const mm_literal_shape *s = &mm_literal_shapes[si];
        for (int pattern = 0; pattern < MM_PATTERNS; ++pattern)
        {
            mm_fill(s->l, s->r, s->k, pattern, (uint32_t)(s->r * 131 + s->k * 7 + s->l + 17));
            for (int use_bias = 0; use_bias < 2; ++use_bias)
            {
                for (int clamp = 0; clamp < MM_CLAMPS; ++clamp)
                {
                    if (s->clamp == MM_CLAMP_ARG || s->clamp == clamp)
                    {
                        mm_compare_literal(s, list, use_bias, clamp, &t[pattern]);
                    }
                }
            }
        }
    }
    const int exact = list ? mml_c11_exact : mml_library_exact;
    unsigned long differ = 0;
    for (int pattern = 0; pattern < MM_PATTERNS; ++pattern)
    {
        snprintf(mm_message,
                 sizeof(mm_message),
                 "literal %s (%s), %s: %ld shapes, %lu outputs; %lu differ from the reference (NaN-ness %lu, "
                 "finiteness %lu), largest finite difference %.3e of the term scale; %lu differ from the entry",
                 list ? "c11" : "library",
                 exact ? "bit-exact expected" : "reported",
                 mm_pattern_names[pattern],
                 (long)MM_LITERAL_SHAPES,
                 t[pattern].outputs,
                 t[pattern].bits_differ,
                 t[pattern].nan_differ,
                 t[pattern].finite_differ,
                 (double)t[pattern].largest,
                 t[pattern].entry_differ);
        TEST_MESSAGE(mm_message);
        differ += t[pattern].bits_differ;
    }
    if (exact)
    {
        TEST_ASSERT_EQUAL_UINT32(0, differ);
    }
}

void mm_literal_library(void) { mm_literal(0); }

void mm_literal_c11(void) { mm_literal(1); }

/* The pointer checks stay when the extents are literal: each literal function refuses a missing lhs, rhs or dst and
 * writes nothing. */
void mm_literal_refusals(void)
{
    mm_fill(1, 4, 9, 0, 5u);
    for (int32_t si = 0; si < MM_LITERAL_SHAPES; ++si)
    {
        const mm_literal_shape *s = &mm_literal_shapes[si];
        for (int list = 0; list < 2; ++list)
        {
            const mml_fn fn = list ? s->c11 : s->library;
            mm_reset(mm_new, 8);
            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, fn(NULL, mm_rhs, mm_bias, mm_new + MM_GUARD, -1.0f, 1.0f));
            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, fn(mm_lhs, NULL, mm_bias, mm_new + MM_GUARD, -1.0f, 1.0f));
            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, fn(mm_lhs, mm_rhs, mm_bias, NULL, -1.0f, 1.0f));
            for (int32_t i = 0; i < 8 + 2 * MM_GUARD; ++i)
            {
                TEST_ASSERT_EQUAL_HEX32(MM_SENTINEL, mm_bits(mm_new[i]));
            }
        }
    }
}
