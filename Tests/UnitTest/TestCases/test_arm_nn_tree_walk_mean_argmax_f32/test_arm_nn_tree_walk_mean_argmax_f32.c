/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_nn_tree_walk_mean_argmax_f32 against its portable reference (tree_mean_argmax_reference.h), and its class
 * against scikit-learn's predict on the golden set (tree_mean_argmax_f32_data.h).
 *
 * The sweep builds its tables and leaf rows at run time from a seeded generator: tree shapes (single-leaf trees,
 * stumps, chains to depth 21, random trees to depth 21, roots shared among trees and placed at inner splits),
 * T in {1, 2, 3, 5, 16, 100}, C in {1, 2, 3, 4, 5, 7, 8, 9}, the NaN route of every split random, all left and all
 * right; rows finite, then a NaN of each kind (quiet and signalling, both signs), each infinity and each zero in each
 * feature in turn; leaf rows of nine kinds (tm_kind_names). A case differs when the statuses differ, when on success
 * the class or a score differs (bit for bit, any NaN equal to any NaN, or within the tolerance below), when on a
 * refusal *class_index was written, or when a guard word around scores or class_index changed.
 *
 * Tolerance. Where the M-profile vector extension has floating point (__ARM_FEATURE_MVE & 2) the compiler may put the
 * entry's class loops in Helium, whose float arithmetic on QEMU flushes subnormal operands and results to zero and
 * answers the default NaN. There a finite score may differ from the reference by at most T x 2^-123 (each of the
 * T - 1 adds and the multiply flushes at most one value below 2^-126 in magnitude per operand and result, and a
 * flushed addend moves a rounding by at most an ulp of a sum below 2^-101), and the class may differ only where the
 * reference's two scores are within twice that. Everywhere else the rule is bit for bit. The counts of each kind of
 * difference are reported per build. */

#include <arm_nnfunctions.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "tree_mean_argmax_f32_data.h"
#include "tree_mean_argmax_reference.h"

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2)
    #define TM_HELIUM_FLOAT 1
#else
    #define TM_HELIUM_FLOAT 0
#endif

#if defined(USING_FVP_CORSTONE_300)
    #define TM_LARGE_SECTION __attribute__((section(".bss.NoInit")))
#else
    #define TM_LARGE_SECTION
#endif

#define TM_TREES 100
#define TM_NODES 8192
#define TM_FEATURES 9
#define TM_ROWS 96
#define TM_CLASSES 9
#define TM_LEAVES (TM_NODES + TM_TREES)
#define TM_GUARD 8
#define TM_SCORE_SENTINEL UINT32_C(0x5a5a5a5a)
#define TM_INDEX_SENTINEL INT32_C(0x3c3c3c3c)
#define TM_REPORTED 8
#define TM_MISSING_LEFT UINT16_C(0x8000)

/* The limits, spelled here rather than read from the header, so moving one fails a case. */
#define TM_MOST_NODES 32767
#define TM_MOST_LEAVES 32768
#define TM_MOST_TREES 65535
#define TM_MOST_FEATURES 32767
#define TM_MOST_CLASSES_AT_MOST_LEAVES 16383 /* INT32_MAX / (4 x 32768) */

static cmsis_nn_tree_node_f32 tm_nodes[TM_NODES];
static uint32_t tm_roots[TM_TREES];
static float32_t tm_rows[TM_ROWS][TM_FEATURES];
static int32_t tm_row_count;
static float32_t tm_leaf_rows[TM_LEAVES * TM_CLASSES] TM_LARGE_SECTION;
static int32_t tm_ref_leaves[TM_MOST_TREES] TM_LARGE_SECTION;
static char tm_message[256];

/* The sweep's tally. */
enum
{
    TM_EXACT,      /* status, class and every score bit for bit */
    TM_NAN_BITS,   /* equal but for a NaN score's sign or payload */
    TM_WITHIN,     /* a score within the Helium tolerance, not bit for bit */
    TM_CLASS_NEAR, /* the class differs where the reference's scores are within the tolerance */
    TM_DIFFERENT,  /* anything else: a failure */
    TM_OUTCOMES
};
static unsigned long tm_tally[TM_OUTCOMES];
static unsigned long tm_statuses[3];
static unsigned long tm_cases;

static uint32_t tm_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static float32_t tm_float(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t tm_bits(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int tm_is_nan(float32_t value) { return (tm_bits(value) & 0x7fffffffu) > 0x7f800000u; }

/* The bits of `count` scores, each pair equal. */
static void tm_assert_bits(const float32_t *want, const float32_t *got, int32_t count, const char *what)
{
    for (int32_t i = 0; i < count; ++i)
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(tm_bits(want[i]), tm_bits(got[i]), what);
}

static uint32_t tm_hash(uint32_t hash, const void *data, size_t size)
{
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < size; ++i)
        hash = (hash ^ bytes[i]) * UINT32_C(16777619);
    return hash;
}

/* ---- Tables and rows (the walk sweep's shapes) ---- */

static const uint32_t tm_thresholds[] = {UINT32_C(0xc0000000),
                                         UINT32_C(0xbf800000),
                                         UINT32_C(0xbf000000),
                                         UINT32_C(0x80000000),
                                         UINT32_C(0x00000000),
                                         UINT32_C(0x00000001),
                                         UINT32_C(0x00800000),
                                         UINT32_C(0x3e800000),
                                         UINT32_C(0x3f800000),
                                         UINT32_C(0x3fc00000),
                                         UINT32_C(0x7f7fffff),
                                         UINT32_C(0xff7fffff)};
#define TM_THRESHOLDS ((uint32_t)(sizeof(tm_thresholds) / sizeof(tm_thresholds[0])))

static const uint32_t tm_specials[] = {UINT32_C(0x7fc00000),
                                       UINT32_C(0xffc00000),
                                       UINT32_C(0x7f800001),
                                       UINT32_C(0xff800001),
                                       UINT32_C(0x7fbfffff),
                                       UINT32_C(0x7f800000),
                                       UINT32_C(0xff800000),
                                       UINT32_C(0x00000000),
                                       UINT32_C(0x80000000)};
#define TM_SPECIALS ((int32_t)(sizeof(tm_specials) / sizeof(tm_specials[0])))

static float32_t tm_value(uint32_t *state)
{
    const uint32_t r = tm_next(state);
    const uint32_t t = tm_thresholds[(r >> 4) % TM_THRESHOLDS];
    switch (r & 3u)
    {
    case 0:
        return tm_float(t);
    case 1:
        return (t & 0x7fffffffu) == 0x7f7fffffu || (t & 0x7fffffffu) == 0u ? tm_float(t) : tm_float(t + 1u);
    case 2:
        return (t & 0x7fffffffu) == 0u ? tm_float(UINT32_C(0x80000001)) : tm_float(t - 1u);
    default:
        return (float32_t)(int32_t)((tm_next(state) >> 8) & 0xffffu) * (1.0f / 8192.0f) - 4.0f;
    }
}

typedef struct
{
    int32_t node_count;
    int32_t leaf_count;
    int32_t tree_count;
    int32_t feature_count;
} tm_shape;

enum
{
    TM_SINGLE_LEAF,
    TM_STUMP,
    TM_CHAIN,
    TM_RANDOM,
    TM_SHARED,
    TM_SHAPE_KINDS
};

static uint16_t tm_feature(uint32_t *state, int32_t features, int missing)
{
    const uint16_t feature = (uint16_t)(tm_next(state) % (uint32_t)features);
    const int left = missing == 0 ? (int)(tm_next(state) & 1u) : missing == 1;
    return (uint16_t)(feature | (left ? TM_MISSING_LEFT : 0u));
}

static void tm_split(int32_t index, uint32_t *state, int32_t features, int missing)
{
    memset(&tm_nodes[index], 0, sizeof(tm_nodes[index]));
    tm_nodes[index].threshold = tm_float(tm_thresholds[tm_next(state) % TM_THRESHOLDS]);
    tm_nodes[index].feature = tm_feature(state, features, missing);
}

static int16_t tm_leaf(tm_shape *shape) { return (int16_t)(-(shape->leaf_count++) - 1); }

static void tm_random_tree(tm_shape *shape, uint32_t *state, int32_t depth, int32_t budget, int missing)
{
    static int32_t queue[TM_NODES];
    static int32_t level[TM_NODES];
    int32_t head = 0;
    int32_t tail = 0;
    const int32_t first = shape->node_count;
    tm_split(shape->node_count++, state, shape->feature_count, missing);
    queue[tail] = first;
    level[tail++] = 1;
    while (head < tail)
    {
        const int32_t index = queue[head];
        const int32_t at = level[head++];
        for (int side = 0; side < 2; ++side)
        {
            int16_t child;
            if (at < depth && shape->node_count - first < budget && (tm_next(state) & 3u) != 0u)
            {
                child = (int16_t)shape->node_count;
                tm_split(shape->node_count++, state, shape->feature_count, missing);
                queue[tail] = child;
                level[tail++] = at + 1;
            }
            else
            {
                child = tm_leaf(shape);
            }
            if (side == 0)
                tm_nodes[index].left = child;
            else
                tm_nodes[index].right = child;
        }
    }
}

static tm_shape tm_build(int kind, int32_t trees, int32_t features, int32_t depth, int missing, uint32_t *state)
{
    tm_shape shape = {0, 0, trees, features};
    const int32_t budget = 60;
    for (int32_t t = 0; t < trees; ++t)
    {
        const int32_t root = shape.node_count;
        switch (kind)
        {
        case TM_SINGLE_LEAF: {
            const int16_t leaf = tm_leaf(&shape);
            tm_split(shape.node_count++, state, features, missing);
            tm_nodes[root].left = leaf;
            tm_nodes[root].right = leaf;
            break;
        }
        case TM_STUMP:
            tm_split(shape.node_count++, state, features, missing);
            tm_nodes[root].left = tm_leaf(&shape);
            tm_nodes[root].right = tm_leaf(&shape);
            break;
        case TM_CHAIN:
            for (int32_t d = 0; d < depth; ++d)
            {
                const int32_t index = shape.node_count;
                const int16_t next = d + 1 < depth ? (int16_t)(index + 1) : tm_leaf(&shape);
                const int16_t leaf = tm_leaf(&shape);
                tm_split(shape.node_count++, state, features, missing);
                if (tm_next(state) & 1u)
                {
                    tm_nodes[index].left = next;
                    tm_nodes[index].right = leaf;
                }
                else
                {
                    tm_nodes[index].left = leaf;
                    tm_nodes[index].right = next;
                }
            }
            break;
        default:
            tm_random_tree(&shape, state, depth, budget, missing);
            break;
        }
        tm_roots[t] = (uint32_t)root;
    }
    if (kind == TM_SHARED)
    {
        for (int32_t t = 0; t < trees; ++t)
            tm_roots[t] = tm_next(state) % (uint32_t)shape.node_count;
    }
    return shape;
}

static void tm_make_rows(int32_t features, uint32_t *state)
{
    tm_row_count = 0;
    for (int32_t r = 0; r < 6; ++r, ++tm_row_count)
        for (int32_t f = 0; f < features; ++f)
            tm_rows[tm_row_count][f] = tm_value(state);
    for (int32_t f = 0; f < features; ++f)
        for (int32_t s = 0; s < TM_SPECIALS; ++s, ++tm_row_count)
            for (int32_t g = 0; g < features; ++g)
                tm_rows[tm_row_count][g] = g == f ? tm_float(tm_specials[s]) : tm_value(state);
    for (int32_t f = 0; f < features; ++f)
        tm_rows[tm_row_count][f] = tm_float(tm_specials[f % 5]);
    ++tm_row_count;
}

static cmsis_nn_tree_table_f32 tm_table(const tm_shape *shape)
{
    cmsis_nn_tree_table_f32 table;
    memset(&table, 0, sizeof(table));
    table.nodes = tm_nodes;
    table.roots = tm_roots;
    table.node_count = shape->node_count;
    table.leaf_count = shape->leaf_count;
    table.tree_count = shape->tree_count;
    table.feature_count = shape->feature_count;
    return table;
}

/* ---- Leaf rows ---- */

enum
{
    TM_FRACTIONS,    /* class fractions of small counts */
    TM_VOTES,        /* a weight in [0.5, 1) on one class */
    TM_TIES,         /* two classes equal and at the top in every row */
    TM_ONE_QUOTIENT, /* values one ulp apart, whose sums can meet at one mean */
    TM_NANS,         /* fractions with NaNs of each kind */
    TM_INFINITIES,   /* fractions with infinities of each sign */
    TM_SUBNORMALS,   /* subnormals of each sign, zeros and the smallest normals: sums that cancel */
    TM_ZEROS,        /* +0, -0, +1 and -1 */
    TM_RANGE,        /* +-FLT_MAX, the smallest normals and 1: sums that overflow and cancel */
    TM_LEAF_KINDS
};
static const char *const tm_kind_names[TM_LEAF_KINDS] =
    {"fractions", "votes", "ties", "one quotient", "NaN", "infinities", "subnormals", "zeros", "range"};
static unsigned long tm_kind_cases[TM_LEAF_KINDS][TM_OUTCOMES];

static void tm_make_leaf_rows(int kind, int32_t leaves, int32_t classes, uint32_t *state)
{
    static const uint32_t nans[] = {
        UINT32_C(0x7fc00000), UINT32_C(0xffc00000), UINT32_C(0x7f800001), UINT32_C(0xff800001), UINT32_C(0x7fbfffff)};
    static const uint32_t tiny[] = {UINT32_C(0x00000001),
                                    UINT32_C(0x80000001),
                                    UINT32_C(0x007fffff),
                                    UINT32_C(0x807fffff),
                                    UINT32_C(0x00400000),
                                    UINT32_C(0x80400000),
                                    UINT32_C(0x00800000),
                                    UINT32_C(0x80800000),
                                    UINT32_C(0x00000000),
                                    UINT32_C(0x80000000)};
    static const uint32_t range[] = {UINT32_C(0x7f7fffff),
                                     UINT32_C(0xff7fffff),
                                     UINT32_C(0x7f7ffffe),
                                     UINT32_C(0x00800000),
                                     UINT32_C(0x80c00000),
                                     UINT32_C(0x3f800000)};
    const int32_t a = (int32_t)(tm_next(state) % (uint32_t)classes);
    const int32_t b = classes > 1 ? (a + 1 + (int32_t)(tm_next(state) % (uint32_t)(classes - 1))) % classes : a;
    for (int32_t l = 0; l < leaves; ++l)
    {
        float32_t *const row = &tm_leaf_rows[l * classes];
        switch (kind)
        {
        case TM_FRACTIONS:
        case TM_NANS:
        case TM_INFINITIES: {
            int32_t counts[TM_CLASSES];
            int32_t total = 0;
            for (int32_t c = 0; c < classes; ++c)
                total += counts[c] = (int32_t)(tm_next(state) % 8u);
            counts[tm_next(state) % (uint32_t)classes] += 1;
            total += 1;
            for (int32_t c = 0; c < classes; ++c)
                row[c] = (float32_t)counts[c] / (float32_t)total;
            if (kind != TM_FRACTIONS && (tm_next(state) & 3u) == 0u)
            {
                const uint32_t r = tm_next(state);
                row[r % (uint32_t)classes] = kind == TM_NANS
                    ? tm_float(nans[(r >> 8) % 5u])
                    : tm_float((r & 256u) ? UINT32_C(0x7f800000) : UINT32_C(0xff800000));
            }
            break;
        }
        case TM_VOTES:
            for (int32_t c = 0; c < classes; ++c)
                row[c] = 0.0f;
            row[tm_next(state) % (uint32_t)classes] = 0.5f + (float32_t)(tm_next(state) & 0xffu) * (1.0f / 512.0f);
            break;
        case TM_TIES:
            for (int32_t c = 0; c < classes; ++c)
                row[c] = 0.125f * (float32_t)(tm_next(state) & 1u);
            row[a] = row[b] = 0.25f * (float32_t)(1u + tm_next(state) % 4u);
            break;
        case TM_ONE_QUOTIENT: {
            const float32_t x = 0.5f + (float32_t)(tm_next(state) & 0x3fffffu) * (1.0f / 8388608.0f);
            for (int32_t c = 0; c < classes; ++c)
                row[c] = 0.0625f * (float32_t)(tm_next(state) & 1u);
            row[a] = x;
            row[b] = (tm_next(state) & 3u) == 0u ? tm_float(tm_bits(x) + 1u) : x;
            break;
        }
        case TM_SUBNORMALS:
            for (int32_t c = 0; c < classes; ++c)
                row[c] = tm_float(tiny[tm_next(state) % 10u]);
            break;
        case TM_ZEROS: {
            static const float32_t zeros[] = {0.0f, -0.0f, 1.0f, -1.0f, -0.0f};
            for (int32_t c = 0; c < classes; ++c)
                row[c] = zeros[tm_next(state) % 5u];
            break;
        }
        default:
            for (int32_t c = 0; c < classes; ++c)
                row[c] = tm_float(range[tm_next(state) % 6u]);
            break;
        }
    }
}

/* ---- One case ---- */

typedef struct
{
    float32_t scores[TM_GUARD + TM_CLASSES + TM_GUARD];
    int32_t index[3];
} tm_out;

static void tm_fill(tm_out *out)
{
    for (size_t i = 0; i < sizeof(out->scores) / sizeof(out->scores[0]); ++i)
        out->scores[i] = tm_float(TM_SCORE_SENTINEL);
    out->index[0] = out->index[1] = out->index[2] = TM_INDEX_SENTINEL;
}

static int tm_guards_intact(const tm_out *out, int32_t classes)
{
    for (int32_t i = 0; i < TM_GUARD; ++i)
        if (tm_bits(out->scores[i]) != TM_SCORE_SENTINEL ||
            tm_bits(out->scores[TM_GUARD + classes + i]) != TM_SCORE_SENTINEL)
            return 0;
    return out->index[0] == TM_INDEX_SENTINEL && out->index[2] == TM_INDEX_SENTINEL;
}

static int tm_scores_untouched(const tm_out *out, int32_t classes)
{
    for (int32_t c = 0; c < classes; ++c)
        if (tm_bits(out->scores[TM_GUARD + c]) != TM_SCORE_SENTINEL)
            return 0;
    return 1;
}

/* The outcome of one success: TM_EXACT, TM_NAN_BITS, TM_WITHIN, TM_CLASS_NEAR or TM_DIFFERENT. */
static int tm_compare(const tm_out *got, const tm_out *want, int32_t classes, int32_t trees, int32_t *first_difference)
{
    const float32_t tolerance = (float32_t)trees * 0x1p-123f;
    int outcome = TM_EXACT;
    *first_difference = -1;
    for (int32_t c = 0; c < classes; ++c)
    {
        const float32_t g = got->scores[TM_GUARD + c];
        const float32_t w = want->scores[TM_GUARD + c];
        int here = TM_EXACT;
        if (tm_bits(g) == tm_bits(w))
            continue;
        if (tm_is_nan(g) && tm_is_nan(w))
            here = TM_NAN_BITS;
        else if (TM_HELIUM_FLOAT && !tm_is_nan(g) && !tm_is_nan(w) && isfinite(g) && isfinite(w) &&
                 fabsf(g - w) <= tolerance)
            here = TM_WITHIN;
        else
            here = TM_DIFFERENT;
        if (here > outcome)
        {
            outcome = here;
            *first_difference = c;
        }
    }
    if (got->index[1] != want->index[1])
    {
        int here = TM_DIFFERENT;
        if (TM_HELIUM_FLOAT && got->index[1] >= 0 && got->index[1] < classes)
        {
            const float32_t g = want->scores[TM_GUARD + got->index[1]];
            const float32_t w = want->scores[TM_GUARD + want->index[1]];
            if (isfinite(g) && isfinite(w) && fabsf(g - w) <= 2.0f * tolerance)
                here = TM_CLASS_NEAR;
        }
        if (here > outcome)
            outcome = here;
    }
    return outcome;
}

static tm_out tm_got;
static tm_out tm_want;

/* Both bodies on one row and table; `pass` names the pointers passed (bit 0 input, 1 table, 2 leaf rows, 3 scores,
 * 4 class_index). Returns the entry's status. */
static arm_cmsis_nn_status tm_case(const char *what,
                                   int kind,
                                   const float32_t *row,
                                   const cmsis_nn_tree_table_f32 *table,
                                   const float32_t *leaf_rows,
                                   int32_t classes,
                                   int pass)
{
    tm_fill(&tm_got);
    tm_fill(&tm_want);
    const int32_t written = classes >= 1 && classes <= TM_CLASSES ? classes : 0;
    const float32_t *input = (pass & 1) ? row : NULL;
    const cmsis_nn_tree_table_f32 *tab = (pass & 2) ? table : NULL;
    const float32_t *rows = (pass & 4) ? leaf_rows : NULL;
    const arm_cmsis_nn_status got = arm_nn_tree_walk_mean_argmax_f32(
        input, tab, rows, classes, (pass & 8) ? &tm_got.scores[TM_GUARD] : NULL, (pass & 16) ? &tm_got.index[1] : NULL);
    const arm_cmsis_nn_status want = tr_reference(input,
                                                  tab,
                                                  rows,
                                                  classes,
                                                  (pass & 8) ? &tm_want.scores[TM_GUARD] : NULL,
                                                  (pass & 16) ? &tm_want.index[1] : NULL,
                                                  tm_ref_leaves);
    int outcome = TM_EXACT;
    int32_t at = -1;
    if (got != want || !tm_guards_intact(&tm_got, written))
        outcome = TM_DIFFERENT;
    else if (got == ARM_CMSIS_NN_SUCCESS)
        outcome = tm_compare(&tm_got, &tm_want, classes, table->tree_count, &at);
    else if (tm_got.index[1] != TM_INDEX_SENTINEL)
        outcome = TM_DIFFERENT;
    ++tm_cases;
    ++tm_tally[outcome];
    if (kind >= 0)
        ++tm_kind_cases[kind][outcome];
    tm_statuses[got == ARM_CMSIS_NN_SUCCESS ? 0 : got == ARM_CMSIS_NN_ARG_ERROR ? 1 : 2] += 1;
    if (outcome == TM_DIFFERENT && tm_tally[TM_DIFFERENT] <= TM_REPORTED)
    {
        snprintf(tm_message,
                 sizeof(tm_message),
                 "difference: %s, kind %s, %ld trees, %ld classes, status %d against %d, class %ld against %ld, "
                 "score %ld 0x%08lx against 0x%08lx",
                 what,
                 kind >= 0 ? tm_kind_names[kind] : "-",
                 table ? (long)table->tree_count : -1L,
                 (long)classes,
                 (int)got,
                 (int)want,
                 (long)tm_got.index[1],
                 (long)tm_want.index[1],
                 (long)at,
                 at >= 0 ? (unsigned long)tm_bits(tm_got.scores[TM_GUARD + at]) : 0ul,
                 at >= 0 ? (unsigned long)tm_bits(tm_want.scores[TM_GUARD + at]) : 0ul);
        TEST_MESSAGE(tm_message);
    }
    return got;
}

static void tm_report(const char *what, const unsigned long *before)
{
    snprintf(tm_message,
             sizeof(tm_message),
             "%s: %lu cases: %lu bit for bit, %lu NaN sign or payload, %lu within tolerance, %lu class near a tie, "
             "%lu differences (all so far: %lu success, %lu argument error, %lu failure; Helium float %d)",
             what,
             (tm_tally[0] + tm_tally[1] + tm_tally[2] + tm_tally[3] + tm_tally[4]) -
                 (before[0] + before[1] + before[2] + before[3] + before[4]),
             tm_tally[TM_EXACT] - before[TM_EXACT],
             tm_tally[TM_NAN_BITS] - before[TM_NAN_BITS],
             tm_tally[TM_WITHIN] - before[TM_WITHIN],
             tm_tally[TM_CLASS_NEAR] - before[TM_CLASS_NEAR],
             tm_tally[TM_DIFFERENT] - before[TM_DIFFERENT],
             tm_statuses[0],
             tm_statuses[1],
             tm_statuses[2],
             TM_HELIUM_FLOAT);
    TEST_MESSAGE(tm_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(tm_tally[TM_DIFFERENT] - before[TM_DIFFERENT]));
#if !TM_HELIUM_FLOAT
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(tm_tally[TM_WITHIN] - before[TM_WITHIN]));
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(tm_tally[TM_CLASS_NEAR] - before[TM_CLASS_NEAR]));
#endif
}

static uint32_t tm_inputs_hash(const tm_shape *shape, int32_t classes)
{
    uint32_t hash = UINT32_C(2166136261);
    hash = tm_hash(hash, tm_nodes, (size_t)shape->node_count * sizeof(tm_nodes[0]));
    hash = tm_hash(hash, tm_roots, (size_t)shape->tree_count * sizeof(tm_roots[0]));
    hash = tm_hash(hash, tm_leaf_rows, (size_t)shape->leaf_count * (size_t)classes * sizeof(float32_t));
    return tm_hash(hash, tm_rows, sizeof(tm_rows));
}

static const int32_t tm_tree_counts[] = {1, 2, 3, 5, 16, 100};
#define TM_TREE_COUNTS ((int)(sizeof(tm_tree_counts) / sizeof(tm_tree_counts[0])))
static const int32_t tm_class_counts[] = {1, 2, 3, 4, 5, 7, 8, 9};
#define TM_CLASS_COUNTS ((int)(sizeof(tm_class_counts) / sizeof(tm_class_counts[0])))
static const int32_t tm_feature_counts[] = {1, 4, 9};
#define TM_FEATURE_COUNTS ((int)(sizeof(tm_feature_counts) / sizeof(tm_feature_counts[0])))
static const int32_t tm_chain_depths[] = {1, 2, 7, 14, 21};
#define TM_CHAIN_DEPTHS ((int)(sizeof(tm_chain_depths) / sizeof(tm_chain_depths[0])))

/* ---- The cases ---- */

static void tm_capability(void)
{
    TEST_ASSERT_EQUAL_INT(1, ARM_NN_HAS_TREE_WALK_MEAN_ARGMAX_F32);
    TEST_ASSERT_EQUAL_INT(1, ARM_NN_HAS_TREE_WALK_F32);
}

/* N2: every row of every table, at every class count, each table's leaf rows of one kind, the kind moving on with
 * each table and class count so every (C, kind) pair is met at every T. */
static void tm_sweep(void)
{
    unsigned long before[TM_OUTCOMES];
    memcpy(before, tm_tally, sizeof(before));
    uint32_t state = UINT32_C(0x2545f491);
    int32_t table_number = 0;
    for (int ti = 0; ti < TM_TREE_COUNTS; ++ti)
        for (int fi = 0; fi < TM_FEATURE_COUNTS; ++fi)
            for (int shape_kind = 0; shape_kind < TM_SHAPE_KINDS; ++shape_kind)
                for (int di = 0; di < (shape_kind == TM_CHAIN ? TM_CHAIN_DEPTHS : 2); ++di)
                    for (int missing = 0; missing < 3; ++missing, ++table_number)
                    {
                        const int32_t depth = shape_kind == TM_CHAIN ? tm_chain_depths[di] : (di == 0 ? 5 : 21);
                        const tm_shape shape =
                            tm_build(shape_kind, tm_tree_counts[ti], tm_feature_counts[fi], depth, missing, &state);
                        const cmsis_nn_tree_table_f32 table = tm_table(&shape);
                        tm_make_rows(shape.feature_count, &state);
                        for (int ci = 0; ci < TM_CLASS_COUNTS; ++ci)
                        {
                            const int32_t classes = tm_class_counts[ci];
                            const int kind = (table_number + ci) % TM_LEAF_KINDS;
                            tm_make_leaf_rows(kind, shape.leaf_count, classes, &state);
                            const uint32_t hash = tm_inputs_hash(&shape, classes);
                            for (int32_t r = 0; r < tm_row_count; ++r)
                                tm_case("sweep", kind, tm_rows[r], &table, tm_leaf_rows, classes, 31);
                            TEST_ASSERT_EQUAL_HEX32_MESSAGE(hash, tm_inputs_hash(&shape, classes), "inputs changed");
                        }
                    }
    tm_report("sweep", before);
    for (int kind = 0; kind < TM_LEAF_KINDS; ++kind)
    {
        snprintf(tm_message,
                 sizeof(tm_message),
                 "sweep %s: %lu bit for bit, %lu NaN sign or payload, %lu within tolerance, %lu class near a tie, %lu "
                 "differences",
                 tm_kind_names[kind],
                 tm_kind_cases[kind][TM_EXACT],
                 tm_kind_cases[kind][TM_NAN_BITS],
                 tm_kind_cases[kind][TM_WITHIN],
                 tm_kind_cases[kind][TM_CLASS_NEAR],
                 tm_kind_cases[kind][TM_DIFFERENT]);
        TEST_MESSAGE(tm_message);
    }
}

static void tm_rows_against(const char *what, const cmsis_nn_tree_table_f32 *table, int32_t classes, int must_fail)
{
    for (int32_t r = 0; r < tm_row_count; ++r)
    {
        const arm_cmsis_nn_status status = tm_case(what, -1, tm_rows[r], table, tm_leaf_rows, classes, 31);
        if (must_fail)
        {
            TEST_ASSERT_EQUAL_INT_MESSAGE(ARM_CMSIS_NN_ARG_ERROR, status, what);
            TEST_ASSERT_TRUE_MESSAGE(tm_scores_untouched(&tm_got, classes >= 1 && classes <= TM_CLASSES ? classes : 0),
                                     what);
        }
    }
}

/* N4: each refusal, each from a good table with one thing changed; the reference states the expected status and
 * the case checks the guard words and the untouched class_index. The argument refusals also leave scores untouched. */
static void tm_refusals(void)
{
    static const int32_t odd_counts[] = {INT32_MIN, -1, 0, 1};
    static const int32_t most[4] = {TM_MOST_NODES, TM_MOST_LEAVES, TM_MOST_TREES, TM_MOST_FEATURES};
    static const uint32_t bad_roots[] = {0u, UINT32_C(0x80000000), UINT32_MAX};
    unsigned long before[TM_OUTCOMES];
    memcpy(before, tm_tally, sizeof(before));
    uint32_t state = UINT32_C(0x68e31da4);
    for (int ti = 0; ti < 4; ++ti)
        for (int shape_kind = TM_STUMP; shape_kind < TM_SHAPE_KINDS; ++shape_kind)
        {
            const int32_t trees = tm_tree_counts[ti];
            const int32_t classes = 3;
            const tm_shape shape = tm_build(shape_kind, trees, 4, shape_kind == TM_CHAIN ? 7 : 6, 0, &state);
            const cmsis_nn_tree_table_f32 good = tm_table(&shape);
            cmsis_nn_tree_table_f32 table;
            tm_make_rows(shape.feature_count, &state);
            tm_make_leaf_rows(TM_FRACTIONS, shape.leaf_count, classes, &state);

            /* Each pointer, in every combination, and the table's two. */
            for (int pass = 0; pass < 31; ++pass)
            {
                TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                                      tm_case("pointer", -1, tm_rows[0], &good, tm_leaf_rows, classes, pass));
                TEST_ASSERT_TRUE(tm_scores_untouched(&tm_got, classes));
            }
            table = good;
            table.nodes = NULL;
            tm_rows_against("NULL nodes", &table, classes, 1);
            table = good;
            table.roots = NULL;
            tm_rows_against("NULL roots", &table, classes, 1);

            /* Each count of the table either side of its range and at each end of it. */
            for (int field = 0; field < 4; ++field)
            {
                int32_t values[8];
                int count = 0;
                for (unsigned i = 0; i < sizeof(odd_counts) / sizeof(odd_counts[0]); ++i)
                    values[count++] = odd_counts[i];
                if (field != 2)
                    values[count++] = most[field];
                values[count++] = most[field] + 1;
                values[count++] = INT32_MAX;
                for (int i = 0; i < count; ++i)
                {
                    table = good;
                    if (field == 2 && values[i] > 0 && values[i] <= trees)
                        continue;
                    if (field == 0)
                        table.node_count = values[i];
                    else if (field == 1)
                        table.leaf_count = values[i];
                    else if (field == 2)
                        table.tree_count = values[i];
                    else
                        table.feature_count = values[i];
                    tm_rows_against("count", &table, classes, values[i] < 1 || values[i] > most[field]);
                }
            }

            /* class_count below 1 and past its bound, INT32_MAX / (4 x leaf_count). */
            {
                const int32_t bound = INT32_MAX / (4 * shape.leaf_count);
                const int32_t refused[] = {INT32_MIN, -1, 0, bound + 1, INT32_MAX};
                for (unsigned i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i)
                    tm_rows_against("class_count", &good, refused[i], 1);
            }

            /* A root out of range at tree 0, at a middle tree and at the last. */
            for (unsigned b = 0; b < sizeof(bad_roots) / sizeof(bad_roots[0]); ++b)
                for (int32_t at = 0; at < trees; at += (trees > 2 ? trees / 2 : 1))
                {
                    const uint32_t saved = tm_roots[at];
                    tm_roots[at] = bad_roots[b] == 0u ? (uint32_t)shape.node_count : bad_roots[b];
                    for (int32_t r = 0; r < tm_row_count; ++r)
                        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                                              tm_case("root", -1, tm_rows[r], &good, tm_leaf_rows, classes, 31));
                    tm_roots[at] = saved;
                }

            /* A feature, a child or a leaf out of range at a split on the path, each side in turn, and cycles. */
            for (int site = 0; site < 3; ++site)
            {
                const int32_t index = site == 0 ? (int32_t)tm_roots[0]
                    : site == 1                 ? (int32_t)tm_roots[trees - 1]
                                                : (int32_t)(tm_next(&state) % (uint32_t)shape.node_count);
                const cmsis_nn_tree_node_f32 saved = tm_nodes[index];
                const uint16_t features[] = {4u, (uint16_t)(TM_MISSING_LEFT | 4u), 0x7fffu, 0xffffu};
                const int16_t children[] = {
                    (int16_t)shape.node_count, INT16_MAX, (int16_t)(-shape.leaf_count - 1), INT16_MIN, (int16_t)index};
                for (unsigned i = 0; i < sizeof(features) / sizeof(features[0]); ++i)
                {
                    tm_nodes[index].feature = features[i];
                    tm_rows_against("feature", &good, classes, 0);
                    tm_nodes[index] = saved;
                }
                for (unsigned i = 0; i < sizeof(children) / sizeof(children[0]); ++i)
                    for (int side = 0; side < 2; ++side)
                    {
                        if (side == 0)
                            tm_nodes[index].left = children[i];
                        else
                            tm_nodes[index].right = children[i];
                        tm_rows_against(i == 4 ? "cycle" : "child or leaf", &good, classes, 0);
                        tm_nodes[index] = saved;
                    }
                if (tm_nodes[index].left >= 0)
                {
                    const int32_t child = tm_nodes[index].left;
                    const cmsis_nn_tree_node_f32 child_saved = tm_nodes[child];
                    tm_nodes[child].right = (int16_t)index;
                    tm_rows_against("cycle through a child", &good, classes, 0);
                    tm_nodes[child] = child_saved;
                }
            }
        }
    tm_report("refusals", before);
}

static cmsis_nn_tree_node_f32 tm_node(float32_t threshold, uint16_t feature, int16_t left, int16_t right)
{
    cmsis_nn_tree_node_f32 node;
    memset(&node, 0, sizeof(node));
    node.threshold = threshold;
    node.feature = feature;
    node.left = left;
    node.right = right;
    return node;
}

static cmsis_nn_tree_table_f32 tm_small_table(const cmsis_nn_tree_node_f32 *nodes,
                                              const uint32_t *roots,
                                              int32_t node_count,
                                              int32_t leaf_count,
                                              int32_t tree_count,
                                              int32_t feature_count)
{
    cmsis_nn_tree_table_f32 table;
    memset(&table, 0, sizeof(table));
    table.nodes = nodes;
    table.roots = roots;
    table.node_count = node_count;
    table.leaf_count = leaf_count;
    table.tree_count = tree_count;
    table.feature_count = feature_count;
    return table;
}

/* The precedence the declaration states, where several defects are present at once. */
static void tm_precedence(void)
{
    /* Tree 0 cycles at its root; tree 1's root is out of range; tree 2 reads a feature out of range. */
    const cmsis_nn_tree_node_f32 nodes[3] = {
        tm_node(0.0f, 0u, 0, 0),
        tm_node(0.0f, 0u, -1, -2),
        tm_node(0.0f, 7u, -1, -2),
    };
    const float32_t rows[2 * 3] = {0.25f, 0.5f, 0.25f, 0.5f, 0.25f, 0.25f};
    const float32_t x = 1.0f;
    float32_t scores[3];
    int32_t index = TM_INDEX_SENTINEL;
    uint32_t roots[3] = {0u, 9u, 2u};
    cmsis_nn_tree_table_f32 table = tm_small_table(nodes, roots, 3, 2, 3, 1);

    /* An argument defect comes first, whatever the trees hold. */
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_mean_argmax_f32(&x, &table, rows, 3, NULL, &index));
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          arm_nn_tree_walk_mean_argmax_f32(&x, &table, rows, 0, scores, &index));
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          arm_nn_tree_walk_mean_argmax_f32(&x, &table, NULL, 3, scores, &index));
    /* Then the trees in order: tree 0's cycle before tree 1's root. */
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE, arm_nn_tree_walk_mean_argmax_f32(&x, &table, rows, 3, scores, &index));
    /* Tree 0 good: tree 1's root, before tree 2's feature. */
    roots[0] = 1u;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          arm_nn_tree_walk_mean_argmax_f32(&x, &table, rows, 3, scores, &index));
    /* Trees 0 and 1 good: tree 2's feature. */
    roots[1] = 1u;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          arm_nn_tree_walk_mean_argmax_f32(&x, &table, rows, 3, scores, &index));
    /* Tree 2 cycles where it was out of range: a FAILURE after two good trees, class_index untouched. */
    roots[2] = 0u;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE, arm_nn_tree_walk_mean_argmax_f32(&x, &table, rows, 3, scores, &index));
    TEST_ASSERT_EQUAL_INT32(TM_INDEX_SENTINEL, index);
    /* A defect past the first failing tree, and one off the row's path, are not seen. */
    roots[2] = 1u;
    table.tree_count = 3;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_mean_argmax_f32(&x, &table, rows, 3, scores, &index));
    TEST_ASSERT_EQUAL_INT32(0, index); /* leaf 1 each time: {0.5, 0.25, 0.25} */
}

/* The class rules of the declaration, by hand: ties, NaN, zeros, infinities. One tree, so each score is its row
 * times 1.0f. */
static void tm_class_rules(void)
{
    const cmsis_nn_tree_node_f32 node = tm_node(0.0f, 0u, -1, -1);
    const uint32_t root = 0;
    const cmsis_nn_tree_table_f32 table = tm_small_table(&node, &root, 1, 1, 1, 1);
    const float32_t x = 0.0f;
    const float32_t qnan = tm_float(UINT32_C(0x7fc00000));
    const float32_t nnan = tm_float(UINT32_C(0xffc00000));
    const float32_t snan = tm_float(UINT32_C(0x7f800001));
    const float32_t inf = tm_float(UINT32_C(0x7f800000));
    const struct
    {
        float32_t row[5];
        int32_t classes;
        int32_t expected;
    } cases[] = {
        {{0.5f, 0.75f, 0.75f, 0.25f}, 4, 1}, /* first of an exact tie */
        {{0.75f, 0.75f, 0.75f, 0.75f, 0.75f}, 5, 0},
        {{qnan, 0.125f, nnan}, 3, 1}, /* a NaN never wins */
        {{snan, -inf}, 2, 1},
        {{qnan, nnan, snan}, 3, 0}, /* all NaN */
        {{-0.0f, 0.0f}, 2, 0},      /* the zeros are one value */
        {{0.0f, -0.0f}, 2, 0},
        {{-inf, -inf, -1.0f}, 3, 2},
        {{inf, FLT_MAX, inf}, 3, 0},
        {{-FLT_MAX, -inf}, 2, 0},
        {{0.25f}, 1, 0},
        {{qnan}, 1, 0},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        float32_t scores[5];
        int32_t index = TM_INDEX_SENTINEL;
        TEST_ASSERT_EQUAL_INT(
            ARM_CMSIS_NN_SUCCESS,
            arm_nn_tree_walk_mean_argmax_f32(&x, &table, cases[i].row, cases[i].classes, scores, &index));
        TEST_ASSERT_EQUAL_INT32_MESSAGE(cases[i].expected, index, "class rule");
    }

    /* The order: tree 0's value first, then one add per later tree in tree order; 1e30 + 1 - 1e30 + 1 + 1 is 2, not
     * 3, and its mean is 2 x float32(1/5). */
    {
        const cmsis_nn_tree_node_f32 nodes[5] = {
            tm_node(0.0f, 0u, -1, -1),
            tm_node(0.0f, 0u, -2, -2),
            tm_node(0.0f, 0u, -3, -3),
            tm_node(0.0f, 0u, -2, -2),
            tm_node(0.0f, 0u, -2, -2),
        };
        const uint32_t roots[5] = {0u, 1u, 2u, 3u, 4u};
        const cmsis_nn_tree_table_f32 five = tm_small_table(nodes, roots, 5, 3, 5, 1);
        const float32_t rows[3] = {1e30f, 1.0f, -1e30f};
        float32_t score = 0.0f;
        int32_t index = TM_INDEX_SENTINEL;
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                              arm_nn_tree_walk_mean_argmax_f32(&x, &five, rows, 1, &score, &index));
        TEST_ASSERT_EQUAL_HEX32(tm_bits(2.0f * (1.0f / 5.0f)), tm_bits(score));
        TEST_ASSERT_EQUAL_INT32(0, index);
    }
}

/* N4, the limits, generated here: 32767 splits in one chain reaching 32768 leaves over 32767 features; 65535 trees
 * over three shared stumps; the class_count bound at the most leaves on both sides, and exactly at a small one. */
static void tm_limits(void)
{
    static cmsis_nn_tree_node_f32 nodes[TM_MOST_NODES] TM_LARGE_SECTION;
    static float32_t row[TM_MOST_FEATURES] TM_LARGE_SECTION;
    static float32_t leaf_rows[TM_MOST_LEAVES] TM_LARGE_SECTION;
    static uint32_t roots[TM_MOST_TREES] TM_LARGE_SECTION;
    static float32_t scores[TM_GUARD + TM_MOST_CLASSES_AT_MOST_LEAVES + TM_GUARD] TM_LARGE_SECTION;
    static float32_t want[TM_MOST_CLASSES_AT_MOST_LEAVES] TM_LARGE_SECTION;
    const uint16_t feature = TM_MOST_FEATURES - 1;
    uint32_t root = 0;
    int32_t index;
    int32_t want_index;
    cmsis_nn_tree_table_f32 table;

    /* The chain: split i sends a value at or below i to leaf i, else to split i + 1; the last split's right is the
     * code -32768. Leaf l's one score is l, so the class is 0 and the score the leaf's number. */
    for (int32_t i = 0; i < TM_MOST_NODES; ++i)
    {
        const int16_t right = (i == TM_MOST_NODES - 1) ? INT16_MIN : (int16_t)(i + 1);
        nodes[i] = tm_node((float32_t)i, feature, (int16_t)(-i - 1), right);
    }
    for (int32_t i = 0; i < TM_MOST_FEATURES; ++i)
        row[i] = -1.0f;
    for (int32_t l = 0; l < TM_MOST_LEAVES; ++l)
        leaf_rows[l] = (float32_t)l;
    table = tm_small_table(nodes, &root, TM_MOST_NODES, TM_MOST_LEAVES, 1, TM_MOST_FEATURES);
    row[feature] = 40000.0f;
    index = TM_INDEX_SENTINEL;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_tree_walk_mean_argmax_f32(row, &table, leaf_rows, 1, &scores[TM_GUARD], &index));
    TEST_ASSERT_EQUAL_FLOAT(32767.0f, scores[TM_GUARD]);
    TEST_ASSERT_EQUAL_INT32(0, index);
    row[feature] = 1000.5f;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_tree_walk_mean_argmax_f32(row, &table, leaf_rows, 1, &scores[TM_GUARD], &index));
    TEST_ASSERT_EQUAL_FLOAT(1001.0f, scores[TM_GUARD]);
    /* One leaf fewer: the code -32768 is out of range on the path. */
    row[feature] = 40000.0f;
    table.leaf_count = TM_MOST_LEAVES - 1;
    index = TM_INDEX_SENTINEL;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          arm_nn_tree_walk_mean_argmax_f32(row, &table, leaf_rows, 1, &scores[TM_GUARD], &index));
    TEST_ASSERT_EQUAL_INT32(TM_INDEX_SENTINEL, index);

    /* 65535 trees over three shared stumps, three classes; then a root out of range at the last tree, then one tree
     * past the most. */
    {
        const cmsis_nn_tree_node_f32 stumps[3] = {
            tm_node(0.0f, 0u, -1, -2),
            tm_node(5.0f, 0u, -3, -4),
            tm_node(-5.0f, 0u, -5, -6),
        };
        const float32_t x = 1.0f;
        for (int32_t t = 0; t < TM_MOST_TREES; ++t)
            roots[t] = (uint32_t)(t % 3);
        for (int32_t i = 0; i < 6 * 3; ++i)
            leaf_rows[i] = (float32_t)((i * 7) % 5) * 0.125f;
        table = tm_small_table(stumps, roots, 3, 6, TM_MOST_TREES, 1);
        index = TM_INDEX_SENTINEL;
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                              arm_nn_tree_walk_mean_argmax_f32(&x, &table, leaf_rows, 3, &scores[TM_GUARD], &index));
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                              tr_reference(&x, &table, leaf_rows, 3, want, &want_index, tm_ref_leaves));
        tm_assert_bits(want, &scores[TM_GUARD], 3, "most trees");
        TEST_ASSERT_EQUAL_INT32(want_index, index);
        roots[TM_MOST_TREES - 1] = 3u;
        index = TM_INDEX_SENTINEL;
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                              arm_nn_tree_walk_mean_argmax_f32(&x, &table, leaf_rows, 3, &scores[TM_GUARD], &index));
        TEST_ASSERT_EQUAL_INT32(TM_INDEX_SENTINEL, index);
        roots[TM_MOST_TREES - 1] = (uint32_t)((TM_MOST_TREES - 1) % 3);
        table.tree_count = TM_MOST_TREES + 1;
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                              arm_nn_tree_walk_mean_argmax_f32(&x, &table, leaf_rows, 3, &scores[TM_GUARD], &index));
    }

    /* The class_count bound at 32768 leaves, INT32_MAX / 131072 = 16383, on both sides. Every row reaches leaf 0,
     * so only its row is read: leaf_rows holds that one row. */
    {
        const cmsis_nn_tree_node_f32 stump = tm_node(0.0f, 0u, -1, -1);
        const float32_t x = 0.0f;
        const int32_t most = TM_MOST_CLASSES_AT_MOST_LEAVES;
        TEST_ASSERT_EQUAL_INT32(most, INT32_MAX / (4 * TM_MOST_LEAVES));
        for (int32_t c = 0; c < most; ++c)
            leaf_rows[c] = (float32_t)(c % 1000) * 0.001f;
        for (int32_t i = 0; i < TM_GUARD; ++i)
            scores[i] = scores[TM_GUARD + most + i] = tm_float(TM_SCORE_SENTINEL);
        root = 0;
        table = tm_small_table(&stump, &root, 1, TM_MOST_LEAVES, 1, 1);
        index = TM_INDEX_SENTINEL;
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                              arm_nn_tree_walk_mean_argmax_f32(&x, &table, leaf_rows, most, &scores[TM_GUARD], &index));
        TEST_ASSERT_EQUAL_INT32(999, index);
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                              tr_reference(&x, &table, leaf_rows, most, want, &want_index, tm_ref_leaves));
        tm_assert_bits(want, &scores[TM_GUARD], most, "most classes");
        for (int32_t i = 0; i < TM_GUARD; ++i)
        {
            TEST_ASSERT_EQUAL_HEX32(TM_SCORE_SENTINEL, tm_bits(scores[i]));
            TEST_ASSERT_EQUAL_HEX32(TM_SCORE_SENTINEL, tm_bits(scores[TM_GUARD + most + i]));
        }
        index = TM_INDEX_SENTINEL;
        TEST_ASSERT_EQUAL_INT(
            ARM_CMSIS_NN_ARG_ERROR,
            arm_nn_tree_walk_mean_argmax_f32(&x, &table, leaf_rows, most + 1, &scores[TM_GUARD], &index));
        TEST_ASSERT_EQUAL_INT32(TM_INDEX_SENTINEL, index);
    }

    /* Exactly at the bound of two leaves, INT32_MAX / 8: tree 0 cycles, so nothing is read or written past the
     * arguments and the answer tells the bound's two sides apart, FAILURE at it and ARG_ERROR past it. */
    {
        const cmsis_nn_tree_node_f32 cycle = tm_node(0.0f, 0u, 0, 0);
        const float32_t x = 0.0f;
        const int32_t bound = INT32_MAX / 8;
        float32_t few[2 * TM_GUARD];
        for (int32_t i = 0; i < 2 * TM_GUARD; ++i)
            few[i] = tm_float(TM_SCORE_SENTINEL);
        root = 0;
        table = tm_small_table(&cycle, &root, 1, 2, 1, 1);
        index = TM_INDEX_SENTINEL;
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE,
                              arm_nn_tree_walk_mean_argmax_f32(&x, &table, leaf_rows, bound, &few[TM_GUARD], &index));
        TEST_ASSERT_EQUAL_INT(
            ARM_CMSIS_NN_ARG_ERROR,
            arm_nn_tree_walk_mean_argmax_f32(&x, &table, leaf_rows, bound + 1, &few[TM_GUARD], &index));
        TEST_ASSERT_EQUAL_INT32(TM_INDEX_SENTINEL, index);
        for (int32_t i = 0; i < 2 * TM_GUARD; ++i)
            TEST_ASSERT_EQUAL_HEX32(TM_SCORE_SENTINEL, tm_bits(few[i]));
    }
}

/* N3: the golden set. Each fit's class against scikit-learn's predict, and against the generator's float32 mirror
 * of the entry, with the scores held to the reference as in the sweep. The rows where the class differs from predict
 * must be exactly those the generator traced (tg_differing), each to a tie of the top two classes. */
static void tm_golden(void)
{
    static cmsis_nn_tree_node_f32 nodes[TG_MOST_NODES];
    static uint32_t roots[TG_MOST_TREES];
    static float32_t leaf_rows[TG_MOST_LEAF_VALUES];
    float32_t row[TG_MOST_FEATURES];
    unsigned long before[TM_OUTCOMES];
    long total_rows = 0;
    long total_agree = 0;
    memcpy(before, tm_tally, sizeof(before));
    TEST_ASSERT_TRUE(TG_MOST_CLASSES <= 16);
    for (int f = 0; f < TG_FITS; ++f)
    {
        const tg_fit *const fit = &tg_fits[f];
        long agree = 0;
        long differ = 0;
        long listed = 0;
        for (int32_t i = 0; i < fit->node_count; ++i)
        {
            const int32_t at = fit->node_start + i;
            nodes[i] = tm_node(tm_float(tg_threshold[at]), tg_feature[at], tg_left[at], tg_right[at]);
        }
        for (int32_t t = 0; t < fit->tree_count; ++t)
            roots[t] = tg_roots[fit->root_start + t];
        for (int32_t i = 0; i < fit->leaf_count * fit->class_count; ++i)
            leaf_rows[i] = tm_float(tg_leaf_rows[fit->leaf_row_start + i]);
        const cmsis_nn_tree_table_f32 table =
            tm_small_table(nodes, roots, fit->node_count, fit->leaf_count, fit->tree_count, fit->feature_count);
        for (int32_t r = 0; r < fit->row_count; ++r)
        {
            for (int32_t j = 0; j < fit->feature_count; ++j)
                row[j] = tm_float(tg_rows[fit->row_value_start + r * fit->feature_count + j]);
            float32_t scores[16];
            float32_t want[16];
            int32_t index = TM_INDEX_SENTINEL;
            int32_t want_index = TM_INDEX_SENTINEL;
            TEST_ASSERT_EQUAL_INT(
                ARM_CMSIS_NN_SUCCESS,
                arm_nn_tree_walk_mean_argmax_f32(row, &table, leaf_rows, fit->class_count, scores, &index));
            TEST_ASSERT_EQUAL_INT(
                ARM_CMSIS_NN_SUCCESS,
                tr_reference(row, &table, leaf_rows, fit->class_count, want, &want_index, tm_ref_leaves));
            tm_assert_bits(want, scores, fit->class_count, fit->name);
            TEST_ASSERT_EQUAL_INT32_MESSAGE(want_index, index, fit->name);
            TEST_ASSERT_EQUAL_INT32_MESSAGE(tg_mirror[fit->row_start + r], index, fit->name);
            if (index == tg_predict[fit->row_start + r])
            {
                ++agree;
                continue;
            }
            ++differ;
            for (int32_t d = 0; d < fit->differ_count; ++d)
            {
                const tg_differ *const listed_row = &tg_differing[fit->differ_start + d];
                if (listed_row->row == r && listed_row->predict == tg_predict[fit->row_start + r] &&
                    listed_row->entry == index)
                    ++listed;
            }
        }
        snprintf(tm_message,
                 sizeof(tm_message),
                 "golden %s: %ld trees, %ld classes, %ld rows, %ld agree with predict, %ld differ (%ld traced)",
                 fit->name,
                 (long)fit->tree_count,
                 (long)fit->class_count,
                 (long)fit->row_count,
                 agree,
                 differ,
                 listed);
        TEST_MESSAGE(tm_message);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(fit->differ_count, (int32_t)differ, fit->name);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(fit->differ_count, (int32_t)listed, fit->name);
        total_rows += fit->row_count;
        total_agree += agree;
    }
    snprintf(tm_message,
             sizeof(tm_message),
             "golden: %d fits, %ld rows, %ld agree with predict, %ld differ",
             TG_FITS,
             total_rows,
             total_agree,
             total_rows - total_agree);
    TEST_MESSAGE(tm_message);
}
