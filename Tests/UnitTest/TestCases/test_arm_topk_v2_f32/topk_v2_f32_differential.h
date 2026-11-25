/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The differential cases: arm_topk_v2_f32 against the body it replaced
 * (topk_v2_f32_previous.h) on the same input, each writing into its own guarded buffers.
 * A case differs when the statuses differ, when any byte of the two value buffers or the
 * two index buffers differs (guards included), when a guard of the entry's buffers moved,
 * or when the input changed; each buffer is checked over what the case can touch and
 * TD_GUARD words either side (td_case). Every length of td_lengths meets every k of td_ks on one, two
 * and (up to 33) six rows, so the steady phase starts at every lane of a block of four and
 * every tail length is reached. */

#include "topk_v2_f32_previous.h"
#include <stdio.h>

#define TD_CAP 4224
#define TD_GUARD 16
#define TD_BYTE 0xa5
#define TD_SENTINEL INT32_C(0x5a5a5a5a)
#define TD_REPORTED 8

#define TD_SIGN UINT32_C(0x80000000)
#define TD_INF UINT32_C(0x7f800000)
#define TD_NINF UINT32_C(0xff800000)
#define TD_QUIET UINT32_C(0x00400000)
#define TD_FRACTION UINT32_C(0x007fffff)

static float32_t td_input[TD_CAP + 2 * TD_GUARD];
static float32_t td_new_values[TD_CAP + 2 * TD_GUARD];
static float32_t td_old_values[TD_CAP + 2 * TD_GUARD];
static int32_t td_new_indices[TD_CAP + 2 * TD_GUARD];
static int32_t td_old_indices[TD_CAP + 2 * TD_GUARD];
static uint32_t td_line[TD_CAP];
static unsigned long td_cases;
static unsigned long td_differences;
static char td_message[160];

static const int32_t td_lengths[] = {1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 2048, 2049};

/* NaNs of both signs, quiet and signalling, with the smallest and largest payloads. */
static const uint32_t td_nans[] = {TD_INF | TD_QUIET,
                                   TD_SIGN | TD_INF | TD_QUIET,
                                   TD_INF | 1,
                                   TD_SIGN | TD_INF | 1,
                                   TD_INF | TD_FRACTION,
                                   TD_SIGN | TD_INF | TD_FRACTION,
                                   TD_INF | TD_QUIET | 1,
                                   TD_SIGN | TD_INF | (TD_QUIET - 1),
                                   TD_INF | 2};
#define TD_NAN_KINDS ((int)(sizeof(td_nans) / sizeof(td_nans[0])))

static uint32_t td_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static uint32_t td_bits_of(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* An approximately standard normal draw: the sum of twelve uniforms on [0, 1) less six. */
static uint32_t td_normal(uint32_t *state)
{
    float32_t sum = 0.0f;
    for (int i = 0; i < 12; ++i)
        sum += (float32_t)(td_next(state) >> 8) * (1.0f / 16777216.0f);
    return td_bits_of(sum - 6.0f);
}

/* One case on td_line[0, count): the entry, then the previous body, on the same input. */
static void td_case(const char *pattern,
                    const cmsis_nn_dims *dims,
                    int32_t k,
                    int count,
                    int pass_input,
                    int pass_values,
                    int pass_indices)
{
    /* The elements each buffer is prepared and compared over, TD_GUARD words either side of
     * them: where every extent and k are at most TD_CAP, the larger of the input's count and
     * the n * h * w * k a call may write; otherwise (the refusals of large or negative
     * extents) the whole buffer. */
    int span = TD_CAP;
    if (dims && dims->n >= 0 && dims->h >= 0 && dims->w >= 0 && dims->c >= 0 && k >= 0 && dims->n <= TD_CAP &&
        dims->h <= TD_CAP && dims->w <= TD_CAP && dims->c <= TD_CAP && k <= TD_CAP)
    {
        const long long most = (long long)dims->n * dims->h * dims->w * k;
        if (most <= TD_CAP)
            span = most > count ? (int)most : count;
    }
    const size_t words = (size_t)span + 2 * TD_GUARD;
    memset(td_input, TD_BYTE, words * sizeof(float32_t));
    memcpy(td_input + TD_GUARD, td_line, (size_t)count * sizeof(uint32_t));
    memset(td_new_values, TD_BYTE, words * sizeof(float32_t));
    memset(td_old_values, TD_BYTE, words * sizeof(float32_t));
    for (size_t i = 0; i < words; ++i)
    {
        td_new_indices[i] = TD_SENTINEL;
        td_old_indices[i] = TD_SENTINEL;
    }
    const float32_t *input = pass_input ? td_input + TD_GUARD : NULL;
    const arm_cmsis_nn_status got = arm_topk_v2_f32(
        input, dims, k, pass_values ? td_new_values + TD_GUARD : NULL, pass_indices ? td_new_indices + TD_GUARD : NULL);
    const arm_cmsis_nn_status want = tp_topk(input,
                                             dims,
                                             k,
                                             pass_values ? td_old_values + TD_GUARD : NULL,
                                             pass_indices ? td_old_indices + TD_GUARD : NULL,
                                             sizeof(float32_t));
    int differs = got != want;
    differs |= memcmp(td_new_values, td_old_values, words * sizeof(float32_t)) != 0;
    differs |= memcmp(td_new_indices, td_old_indices, words * sizeof(int32_t)) != 0;
    /* The entry's guards and the input, on their own, so a fault both bodies share is caught too. */
    long long written = 0;
    if (got == ARM_CMSIS_NN_SUCCESS && dims)
        written = (long long)dims->n * dims->h * dims->w * k;
    if (written < 0 || written > TD_CAP)
        differs = 1;
    else
    {
        const unsigned char *value_bytes = (const unsigned char *)td_new_values;
        for (size_t i = 0; i < words * sizeof(float32_t); ++i)
        {
            const int inside =
                i >= TD_GUARD * sizeof(float32_t) && i < (TD_GUARD + (size_t)written) * sizeof(float32_t);
            if (!inside && value_bytes[i] != TD_BYTE)
                differs = 1;
        }
        for (size_t i = 0; i < words; ++i)
        {
            const int inside = i >= TD_GUARD && i < TD_GUARD + (size_t)written;
            if (!inside && td_new_indices[i] != TD_SENTINEL)
                differs = 1;
        }
    }
    const unsigned char *input_bytes = (const unsigned char *)td_input;
    for (size_t i = 0; i < words * sizeof(float32_t); ++i)
    {
        const int inside = i >= TD_GUARD * sizeof(float32_t) && i < (TD_GUARD + (size_t)count) * sizeof(float32_t);
        if (!inside && input_bytes[i] != TD_BYTE)
            differs = 1;
    }
    differs |= memcmp(td_input + TD_GUARD, td_line, (size_t)count * sizeof(uint32_t)) != 0;

    ++td_cases;
    if (differs)
    {
        if (td_differences < TD_REPORTED)
        {
            snprintf(td_message,
                     sizeof(td_message),
                     "difference: %s dims {%ld, %ld, %ld, %ld} k %ld status %d against %d",
                     pattern,
                     dims ? (long)dims->n : -1L,
                     dims ? (long)dims->h : -1L,
                     dims ? (long)dims->w : -1L,
                     dims ? (long)dims->c : -1L,
                     (long)k,
                     (int)got,
                     (int)want);
            TEST_MESSAGE(td_message);
        }
        ++td_differences;
    }
}

/* k in {0, 1, 2, 5, N - 1, N, N + 1}, each once; N + 1, and 5 above N, are refused. */
static int td_ks(int32_t line, int32_t ks[7])
{
    const int32_t all[7] = {0, 1, 2, 5, line - 1, line, line + 1};
    int count = 0;
    for (int i = 0; i < 7; ++i)
    {
        int seen = 0;
        for (int j = 0; j < count; ++j)
            seen |= ks[j] == all[i];
        if (!seen)
            ks[count++] = all[i];
    }
    return count;
}

typedef void (*td_fill_fn)(int rows, int line, int variant, uint32_t *state);
typedef int (*td_variants_fn)(int line);

/* Every length on one and two rows, and on six (n, h, w = 1, 2, 3) up to 33; every k. */
static void td_sweep(const char *pattern, td_fill_fn fill, td_variants_fn variants)
{
    uint32_t state = UINT32_C(0x2545f491);
    const unsigned long before_cases = td_cases;
    const unsigned long before_differences = td_differences;
    for (unsigned l = 0; l < sizeof(td_lengths) / sizeof(td_lengths[0]); ++l)
    {
        const int32_t line = td_lengths[l];
        const int32_t shapes[3][4] = {{1, 1, 1, line}, {2, 1, 1, line}, {1, 2, 3, line}};
        const int n_shapes = line <= 33 ? 3 : 2;
        int32_t ks[7];
        const int n_ks = td_ks(line, ks);
        for (int s = 0; s < n_shapes; ++s)
        {
            const int rows = shapes[s][0] * shapes[s][1] * shapes[s][2];
            const cmsis_nn_dims dims = {shapes[s][0], shapes[s][1], shapes[s][2], shapes[s][3]};
            const int n_variants = variants(line);
            for (int v = 0; v < n_variants; ++v)
            {
                fill(rows, line, v, &state);
                for (int i = 0; i < n_ks; ++i)
                    td_case(pattern, &dims, ks[i], rows * line, 1, 1, 1);
            }
        }
    }
    snprintf(td_message,
             sizeof(td_message),
             "differential %s: %lu cases, %lu differences",
             pattern,
             td_cases - before_cases,
             td_differences - before_differences);
    TEST_MESSAGE(td_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(td_differences - before_differences));
}

static int td_four(int line)
{
    (void)line;
    return 4;
}

static int td_three(int line)
{
    (void)line;
    return 3;
}

static void td_fill_normals(int rows, int line, int variant, uint32_t *state)
{
    (void)variant;
    for (int i = 0; i < rows * line; ++i)
        td_line[i] = td_normal(state);
}

/* One value everywhere: a number, -0 and the largest finite value. */
static void td_fill_equal(int rows, int line, int variant, uint32_t *state)
{
    (void)state;
    const uint32_t values[3] = {td_bits_of(1.5f), TD_SIGN, UINT32_C(0x7f7fffff)};
    for (int i = 0; i < rows * line; ++i)
        td_line[i] = values[variant];
}

/* Equal values at neighbouring indices across every boundary of a block of four: pairs at
 * each of four shifts, and draws from an alphabet of two and of three values, which tie
 * with the kept worst at every k. */
static int td_tie_variants(int line)
{
    (void)line;
    return 8;
}

static void td_fill_ties(int rows, int line, int variant, uint32_t *state)
{
    const uint32_t alphabet[2] = {td_bits_of(2.0f), td_bits_of(-1.0f)};
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < line; ++i)
        {
            uint32_t bits;
            if (variant < 4)
            {
                /* Groups of two, shifted: a group's value is a hash of its number. */
                const uint32_t group = (uint32_t)(i + variant) / 2u;
                bits = td_bits_of((float32_t)((group * UINT32_C(2654435761)) >> 28) - 7.0f);
            }
            else if (variant < 6)
                bits = alphabet[td_next(state) % 2u];
            else
                bits = td_next(state) % 3u == 0 ? td_bits_of(-1.0f) : td_bits_of(0.5f);
            td_line[r * line + i] = bits;
        }
}

/* +0 and -0: alone, beside +-1, and alternating. */
static void td_fill_zeros(int rows, int line, int variant, uint32_t *state)
{
    const uint32_t mixed[4] = {0, TD_SIGN, td_bits_of(1.0f), td_bits_of(-1.0f)};
    for (int i = 0; i < rows * line; ++i)
    {
        if (variant == 0)
            td_line[i] = (td_next(state) & 1u) ? TD_SIGN : 0;
        else if (variant == 1)
            td_line[i] = mixed[td_next(state) % 4u];
        else
            td_line[i] = (i % line) & 1 ? 0 : TD_SIGN;
    }
}

/* Positions that carry a NaN: every one up to 33; on the long lines the first ten, ten
 * around the middle and the last ten, which reach every lane of the first blocks after
 * k = 1, 2 and 5, and of the last blocks and the tail after k = N - 1. */
static int td_nan_positions(int line, int p)
{
    if (line <= 33)
        return p;
    if (p < 10)
        return p;
    if (p < 20)
        return line / 2 - 5 + (p - 10);
    return line - 10 + (p - 20);
}

static int td_nan_variants(int line)
{
    /* Three kinds at each position on a short line, one on a long one; and two NaNs. */
    return line <= 33 ? 3 * line + line : 30 + 30;
}

static void td_fill_nans(int rows, int line, int variant, uint32_t *state)
{
    for (int i = 0; i < rows * line; ++i)
        td_line[i] = td_normal(state);
    const int single = line <= 33 ? 3 * line : 30;
    if (variant < single)
    {
        const int position = td_nan_positions(line, line <= 33 ? variant % line : variant);
        const int kind = line <= 33 ? variant / line + 3 * (position % 3) : position;
        for (int r = 0; r < rows; ++r)
            td_line[r * line + position] = td_nans[kind % TD_NAN_KINDS];
        return;
    }
    /* Two NaNs, at a position and four (one block) later, of different kinds. */
    const int index = variant - single;
    const int position = td_nan_positions(line, index);
    for (int r = 0; r < rows; ++r)
    {
        td_line[r * line + position] = td_nans[index % TD_NAN_KINDS];
        td_line[r * line + (position + 4) % line] = td_nans[(index + 1) % TD_NAN_KINDS];
    }
}

/* Every element a NaN, kinds rotated. */
static void td_fill_all_nan(int rows, int line, int variant, uint32_t *state)
{
    (void)state;
    for (int i = 0; i < rows * line; ++i)
        td_line[i] = td_nans[(i + variant) % TD_NAN_KINDS];
}

/* Infinities of both signs: alone, beside numbers, and -inf everywhere but one +inf. */
static void td_fill_infinities(int rows, int line, int variant, uint32_t *state)
{
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < line; ++i)
        {
            uint32_t bits;
            if (variant == 0)
                bits = (td_next(state) & 1u) ? TD_NINF : TD_INF;
            else if (variant == 1)
            {
                const uint32_t pick = td_next(state) % 4u;
                bits = pick == 0 ? TD_INF : pick == 1 ? TD_NINF : td_normal(state);
            }
            else
                bits = i == (r * 7 + 3) % line ? TD_INF : TD_NINF;
            td_line[r * line + i] = bits;
        }
}

/* Strictly ascending: every element is the best so far, so every one is inserted. */
static void td_fill_ascending(int rows, int line, int variant, uint32_t *state)
{
    (void)state;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < line; ++i)
            td_line[r * line + i] = variant == 0 ? td_bits_of(0.25f * (float32_t)(i - line / 2))
                : variant == 1                   ? (uint32_t)(i + 1)
                                                 : TD_SIGN | (uint32_t)(line - i);
}

/* Strictly descending: every element after the first k is rejected. */
static void td_fill_descending(int rows, int line, int variant, uint32_t *state)
{
    (void)state;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < line; ++i)
            td_line[r * line + i] = variant == 0 ? td_bits_of(0.25f * (float32_t)(line / 2 - i))
                : variant == 1                   ? (uint32_t)(line - i)
                                                 : TD_SIGN | (uint32_t)(i + 1);
}

/* Subnormals of both signs; then beside both zeros and the smallest normals. */
static void td_fill_subnormals(int rows, int line, int variant, uint32_t *state)
{
    const uint32_t edges[6] = {0, TD_SIGN, UINT32_C(0x00800000), TD_SIGN | UINT32_C(0x00800000), 1, TD_SIGN | 1};
    for (int i = 0; i < rows * line; ++i)
    {
        const uint32_t r = td_next(state);
        const uint32_t subnormal = (r & TD_SIGN) | (1u + (r >> 9) % TD_FRACTION);
        if (variant == 0)
            td_line[i] = subnormal;
        else if (variant == 1)
            td_line[i] = (r & 3u) ? subnormal : edges[(r >> 2) % 6u];
        else
            td_line[i] = (r & 1u) ? (TD_SIGN | (r >> 31)) : (r >> 31);
    }
}

/* Arbitrary bit patterns, NaNs and infinities among them. */
static void td_fill_bits(int rows, int line, int variant, uint32_t *state)
{
    (void)variant;
    for (int i = 0; i < rows * line; ++i)
        td_line[i] = td_next(state);
}

static void td_normals(void) { td_sweep("normals", td_fill_normals, td_four); }
static void td_equal(void) { td_sweep("equal", td_fill_equal, td_three); }
static void td_ties(void) { td_sweep("ties", td_fill_ties, td_tie_variants); }
static void td_zeros(void) { td_sweep("zeros", td_fill_zeros, td_three); }
static void td_nan_lanes(void) { td_sweep("nan lanes", td_fill_nans, td_nan_variants); }
static void td_all_nan(void) { td_sweep("all nan", td_fill_all_nan, td_three); }
static void td_infinities(void) { td_sweep("infinities", td_fill_infinities, td_three); }
static void td_ascending(void) { td_sweep("ascending", td_fill_ascending, td_three); }
static void td_descending(void) { td_sweep("descending", td_fill_descending, td_three); }
static void td_subnormals(void) { td_sweep("subnormals", td_fill_subnormals, td_three); }
static void td_bits(void) { td_sweep("bits", td_fill_bits, td_four); }

/* Every refusal of the previous body, refused alike, every buffer untouched; and the
 * empty calls that succeed with null pointers. */
static void td_refusals(void)
{
    const unsigned long before_cases = td_cases;
    const unsigned long before_differences = td_differences;
    for (int i = 0; i < 8; ++i)
        td_line[i] = td_bits_of((float32_t)i);
    const cmsis_nn_dims valid = {1, 2, 1, 4};
    td_case("null dims", NULL, 1, 8, 1, 1, 1);
    const int32_t ks[] = {5, -1, INT32_MIN, INT32_MAX, 0, 1, 4};
    for (unsigned i = 0; i < sizeof(ks) / sizeof(ks[0]); ++i)
    {
        td_case("k", &valid, ks[i], 8, 1, 1, 1);
        td_case("null input", &valid, ks[i], 8, 0, 1, 1);
        td_case("null values", &valid, ks[i], 8, 1, 0, 1);
        td_case("null indices", &valid, ks[i], 8, 1, 1, 0);
        td_case("all null", &valid, ks[i], 8, 0, 0, 0);
    }
    const cmsis_nn_dims negative[] = {{-1, 1, 1, 1},
                                      {1, -1, 1, 1},
                                      {1, 1, -1, 1},
                                      {1, 1, 1, -1},
                                      {0, -1, 1, 1},
                                      {1, 1, 0, -1},
                                      {INT32_MIN, 1, 1, 1},
                                      {1, 1, 1, INT32_MIN}};
    for (unsigned i = 0; i < sizeof(negative) / sizeof(negative[0]); ++i)
        for (int32_t k = -1; k <= 1; ++k)
            td_case("negative extent", &negative[i], k, 0, 1, 1, 1);
    /* Byte counts above INT32_MAX: the input's, and at k = c the indices' too. */
    const cmsis_nn_dims oversized[] = {{1, 1, 1, INT32_MAX},
                                       {65536, 32768, 1, 1},
                                       {1, 1, 536870912, 1},
                                       {536870912, 1, 1, 1},
                                       {1, 1, 268435456, 2},
                                       {INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX}};
    for (unsigned i = 0; i < sizeof(oversized) / sizeof(oversized[0]); ++i)
    {
        td_case("oversized", &oversized[i], oversized[i].c, 0, 1, 1, 1);
        td_case("oversized", &oversized[i], 0, 0, 1, 1, 1);
        td_case("oversized", &oversized[i], 1, 0, 1, 1, 1);
        td_case("oversized null", &oversized[i], 1, 0, 0, 0, 0);
    }
    /* Zero extents: success with nothing written, null pointers accepted; k above an
     * empty innermost extent refused. */
    const cmsis_nn_dims empty[] = {
        {0, 3, 4, 5}, {2, 0, 4, 5}, {2, 3, 0, 5}, {2, 3, 4, 0}, {0, 0, 0, 0}, {INT32_MAX, INT32_MAX, 0, 1}};
    for (unsigned i = 0; i < sizeof(empty) / sizeof(empty[0]); ++i)
        for (int32_t k = 0; k <= 6; ++k)
        {
            td_case("empty", &empty[i], k, 0, 1, 1, 1);
            td_case("empty null", &empty[i], k, 0, 0, 0, 0);
        }
    snprintf(td_message,
             sizeof(td_message),
             "differential refusals: %lu cases, %lu differences",
             td_cases - before_cases,
             td_differences - before_differences);
    TEST_MESSAGE(td_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(td_differences - before_differences));
}
