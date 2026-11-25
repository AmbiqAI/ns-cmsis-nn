/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The differential cases: arm_nn_tree_walk_f32 against the body it had (tree_walk_previous.h), on
 * the same row and table, each writing into its own guarded leaf buffer. A case differs when the
 * statuses differ, when any byte of the two leaf buffers differs (guards included, so a write the
 * other body did not make is a difference), or when the row or the table changed. The tables are
 * built here at run time from a seeded generator: tree shapes (single-leaf trees, stumps, chains
 * to depth 21, random trees to depth 21, roots shared among trees and placed at inner splits),
 * T in {1, 2, 3, 5, 16, 100}, the NaN route of every split random, all left and all right; and the
 * rows finite, with a NaN of each kind (quiet and signalling, both signs), each infinity and each
 * zero in each feature in turn. The refusals start from good tables and change one thing: each
 * pointer, each count either side of its range, a root at tree 0 and at a later tree, a feature,
 * a child or a leaf out of range at a split on the path, and a cycle.
 *
 * This file must not include Internal/arm_nn_tree_walk_flt.h: the suite compiles a counted copy of
 * the library source after it, and that copy has to be the one to see the walk's body first. */

#include "tree_walk_previous.h"
#include <stdio.h>
#include <string.h>
#include <unity.h>

#define TD_TREES 100
#define TD_NODES 8192
#define TD_FEATURES 9
#define TD_ROWS 96
#define TD_GUARD 8
#define TD_SENTINEL INT32_C(0x3c3c3c3c)
#define TD_REPORTED 8
#define TD_MISSING_LEFT UINT16_C(0x8000)
#define TD_MOST_TREES 65535

#if defined(USING_FVP_CORSTONE_300)
    #define TD_LARGE_SECTION __attribute__((section(".bss.NoInit")))
#else
    #define TD_LARGE_SECTION
#endif

static cmsis_nn_tree_node_f32 td_nodes[TD_NODES];
static cmsis_nn_tree_node_f32 td_nodes_copy[TD_NODES];
static uint32_t td_roots[TD_TREES];
static uint32_t td_roots_copy[TD_TREES];
static float32_t td_rows[TD_ROWS][TD_FEATURES];
static int32_t td_row_count;
static int32_t td_new[TD_TREES + 2 * TD_GUARD];
static int32_t td_old[TD_TREES + 2 * TD_GUARD];
static unsigned long td_cases;
static unsigned long td_differences;
static unsigned long td_statuses[3]; /* the entry's SUCCESS, ARG_ERROR and FAILURE answers */
static char td_message[224];

static uint32_t td_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static float32_t td_float(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* Thresholds: a few values a row can equal, and their neighbours one ulp away, signed zeros
 * among them, and the smallest subnormal and normal. All finite, as a writer holds them. */
static const uint32_t td_thresholds[] = {UINT32_C(0xc0000000),
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
#define TD_THRESHOLDS ((uint32_t)(sizeof(td_thresholds) / sizeof(td_thresholds[0])))

/* The specials put in each feature in turn: quiet and signalling NaNs of both signs, both
 * infinities and both zeros. */
static const uint32_t td_specials[] = {UINT32_C(0x7fc00000),
                                       UINT32_C(0xffc00000),
                                       UINT32_C(0x7f800001),
                                       UINT32_C(0xff800001),
                                       UINT32_C(0x7fbfffff),
                                       UINT32_C(0x7f800000),
                                       UINT32_C(0xff800000),
                                       UINT32_C(0x00000000),
                                       UINT32_C(0x80000000)};
#define TD_SPECIALS ((int32_t)(sizeof(td_specials) / sizeof(td_specials[0])))

/* A finite feature value: a threshold, a threshold one ulp either side (never past a finite
 * extreme), or a value drawn from [-4, 4). */
static float32_t td_value(uint32_t *state)
{
    const uint32_t r = td_next(state);
    const uint32_t t = td_thresholds[(r >> 4) % TD_THRESHOLDS];
    switch (r & 3u)
    {
    case 0:
        return td_float(t);
    case 1:
        return (t & 0x7fffffffu) == 0x7f7fffffu || (t & 0x7fffffffu) == 0u ? td_float(t) : td_float(t + 1u);
    case 2:
        return (t & 0x7fffffffu) == 0u ? td_float(UINT32_C(0x80000001)) : td_float(t - 1u);
    default:
        return (float32_t)(int32_t)((td_next(state) >> 8) & 0xffffu) * (1.0f / 8192.0f) - 4.0f;
    }
}

typedef struct
{
    int32_t node_count;
    int32_t leaf_count;
    int32_t tree_count;
    int32_t feature_count;
} td_shape;

enum
{
    TD_SINGLE_LEAF,
    TD_STUMP,
    TD_CHAIN,
    TD_RANDOM,
    TD_SHARED,
    TD_KINDS
};
static const char *const td_kind_names[TD_KINDS] = {"single leaf", "stumps", "chain", "random", "shared roots"};

static uint16_t td_feature(uint32_t *state, int32_t features, int missing)
{
    const uint16_t feature = (uint16_t)(td_next(state) % (uint32_t)features);
    const int left = missing == 0 ? (int)(td_next(state) & 1u) : missing == 1;
    return (uint16_t)(feature | (left ? TD_MISSING_LEFT : 0u));
}

static void td_split(int32_t index, uint32_t *state, int32_t features, int missing)
{
    memset(&td_nodes[index], 0, sizeof(td_nodes[index]));
    td_nodes[index].threshold = td_float(td_thresholds[td_next(state) % TD_THRESHOLDS]);
    td_nodes[index].feature = td_feature(state, features, missing);
}

static int16_t td_leaf(td_shape *shape) { return (int16_t)(-(shape->leaf_count++) - 1); }

/* One random tree to depth `depth` at most, grown breadth first from a queue of open children,
 * each open child a split with probability 3/4 while splits remain in its budget; its root is
 * the first new split. */
static void td_random_tree(td_shape *shape, uint32_t *state, int32_t depth, int32_t budget, int missing)
{
    static int32_t queue[TD_NODES];
    static int32_t level[TD_NODES];
    int32_t head = 0;
    int32_t tail = 0;
    const int32_t first = shape->node_count;
    td_split(shape->node_count++, state, shape->feature_count, missing);
    queue[tail] = first;
    level[tail++] = 1;
    while (head < tail)
    {
        const int32_t index = queue[head];
        const int32_t at = level[head++];
        for (int side = 0; side < 2; ++side)
        {
            int16_t child;
            if (at < depth && shape->node_count - first < budget && (td_next(state) & 3u) != 0u)
            {
                child = (int16_t)shape->node_count;
                td_split(shape->node_count++, state, shape->feature_count, missing);
                queue[tail] = child;
                level[tail++] = at + 1;
            }
            else
            {
                child = td_leaf(shape);
            }
            if (side == 0)
                td_nodes[index].left = child;
            else
                td_nodes[index].right = child;
        }
    }
}

/* A table of `trees` trees of one kind over `features` features; `depth` is the chain's length
 * and the random trees' bound; `missing` 0 draws each split's NaN route, 1 sends every NaN left
 * and 2 right. Leaves are numbered over the table in the order they are made. */
static td_shape td_build(int kind, int32_t trees, int32_t features, int32_t depth, int missing, uint32_t *state)
{
    td_shape shape = {0, 0, trees, features};
    const int32_t budget = 60; /* splits a random tree may have: 100 trees of 60 fit TD_NODES */
    for (int32_t t = 0; t < trees; ++t)
    {
        const int32_t root = shape.node_count;
        switch (kind)
        {
        case TD_SINGLE_LEAF: {
            const int16_t leaf = td_leaf(&shape);
            td_split(shape.node_count++, state, features, missing);
            td_nodes[root].left = leaf;
            td_nodes[root].right = leaf;
            break;
        }
        case TD_STUMP:
            td_split(shape.node_count++, state, features, missing);
            td_nodes[root].left = td_leaf(&shape);
            td_nodes[root].right = td_leaf(&shape);
            break;
        case TD_CHAIN:
            for (int32_t d = 0; d < depth; ++d)
            {
                const int32_t index = shape.node_count;
                const int16_t next = d + 1 < depth ? (int16_t)(index + 1) : td_leaf(&shape);
                const int16_t leaf = td_leaf(&shape);
                td_split(shape.node_count++, state, features, missing);
                if (td_next(state) & 1u)
                {
                    td_nodes[index].left = next;
                    td_nodes[index].right = leaf;
                }
                else
                {
                    td_nodes[index].left = leaf;
                    td_nodes[index].right = next;
                }
            }
            break;
        default:
            td_random_tree(&shape, state, depth, budget, missing);
            break;
        }
        td_roots[t] = (uint32_t)root;
    }
    if (kind == TD_SHARED)
    {
        /* Every split heads a tree of the splits made after it, so any split is a root: two roots
         * may name one split, and a root may be another tree's inner split. */
        for (int32_t t = 0; t < trees; ++t)
            td_roots[t] = td_next(state) % (uint32_t)shape.node_count;
    }
    return shape;
}

/* The rows of a table: finite rows, then for each feature each special in it with the other
 * features finite, then a row whose every feature is a NaN. */
static void td_make_rows(int32_t features, uint32_t *state)
{
    td_row_count = 0;
    for (int32_t r = 0; r < 6; ++r, ++td_row_count)
        for (int32_t f = 0; f < features; ++f)
            td_rows[td_row_count][f] = td_value(state);
    for (int32_t f = 0; f < features; ++f)
        for (int32_t s = 0; s < TD_SPECIALS; ++s, ++td_row_count)
            for (int32_t g = 0; g < features; ++g)
                td_rows[td_row_count][g] = g == f ? td_float(td_specials[s]) : td_value(state);
    for (int32_t f = 0; f < features; ++f)
        td_rows[td_row_count][f] = td_float(td_specials[f % 5]);
    ++td_row_count;
}

static cmsis_nn_tree_table_f32 td_table(const td_shape *shape)
{
    cmsis_nn_tree_table_f32 table;
    memset(&table, 0, sizeof(table));
    table.nodes = td_nodes;
    table.roots = td_roots;
    table.node_count = shape->node_count;
    table.leaf_count = shape->leaf_count;
    table.tree_count = shape->tree_count;
    table.feature_count = shape->feature_count;
    return table;
}

/* One case: both bodies on one row and table, `pass` naming which pointers are passed (bit 0
 * the row, 1 the table, 2 the leaves). The table's own pointers are the caller's. */
static void td_case(const char *what, const float32_t *row, const cmsis_nn_tree_table_f32 *table, int pass)
{
    float32_t row_copy[TD_FEATURES];
    const size_t words = TD_TREES + 2 * TD_GUARD;
    for (size_t i = 0; i < words; ++i)
    {
        td_new[i] = TD_SENTINEL;
        td_old[i] = TD_SENTINEL;
    }
    memcpy(row_copy, row, sizeof(row_copy));
    memcpy(td_nodes_copy, td_nodes, sizeof(td_nodes));
    memcpy(td_roots_copy, td_roots, sizeof(td_roots));
    const float32_t *input = (pass & 1) ? row : NULL;
    const cmsis_nn_tree_table_f32 *tab = (pass & 2) ? table : NULL;
    const arm_cmsis_nn_status got = arm_nn_tree_walk_f32(input, tab, (pass & 4) ? td_new + TD_GUARD : NULL);
    const arm_cmsis_nn_status want = twp_tree_walk_f32(input, tab, (pass & 4) ? td_old + TD_GUARD : NULL);
    int differs = got != want;
    differs |= memcmp(td_new, td_old, sizeof(td_new)) != 0;
    differs |= memcmp(row_copy, row, sizeof(row_copy)) != 0;
    differs |= memcmp(td_nodes_copy, td_nodes, sizeof(td_nodes)) != 0;
    differs |= memcmp(td_roots_copy, td_roots, sizeof(td_roots)) != 0;
    ++td_cases;
    td_statuses[got == ARM_CMSIS_NN_SUCCESS ? 0 : got == ARM_CMSIS_NN_ARG_ERROR ? 1 : 2] += 1;
    if (differs)
    {
        if (td_differences < TD_REPORTED)
        {
            snprintf(td_message,
                     sizeof(td_message),
                     "difference: %s, %ld trees, %ld splits, status %d against %d",
                     what,
                     table ? (long)table->tree_count : -1L,
                     table ? (long)table->node_count : -1L,
                     (int)got,
                     (int)want);
            TEST_MESSAGE(td_message);
        }
        ++td_differences;
    }
}

static void td_report(const char *what, unsigned long before_cases, unsigned long before_differences)
{
    snprintf(td_message,
             sizeof(td_message),
             "differential %s: %lu cases, %lu differences (all cases so far: %lu success, %lu argument error, %lu "
             "failure)",
             what,
             td_cases - before_cases,
             td_differences - before_differences,
             td_statuses[0],
             td_statuses[1],
             td_statuses[2]);
    TEST_MESSAGE(td_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(td_differences - before_differences));
}

static const int32_t td_tree_counts[] = {1, 2, 3, 5, 16, 100};
#define TD_TREE_COUNTS ((int)(sizeof(td_tree_counts) / sizeof(td_tree_counts[0])))
static const int32_t td_feature_counts[] = {1, 4, 9};
#define TD_FEATURE_COUNTS ((int)(sizeof(td_feature_counts) / sizeof(td_feature_counts[0])))
static const int32_t td_chain_depths[] = {1, 2, 7, 14, 21};
#define TD_CHAIN_DEPTHS ((int)(sizeof(td_chain_depths) / sizeof(td_chain_depths[0])))

/* Every row of every table the sweep builds. */
static void td_shapes(void)
{
    const unsigned long before_cases = td_cases;
    const unsigned long before_differences = td_differences;
    uint32_t state = UINT32_C(0x6b43a9b5);
    for (int ti = 0; ti < TD_TREE_COUNTS; ++ti)
        for (int fi = 0; fi < TD_FEATURE_COUNTS; ++fi)
            for (int kind = 0; kind < TD_KINDS; ++kind)
                for (int di = 0; di < (kind == TD_CHAIN ? TD_CHAIN_DEPTHS : 2); ++di)
                    for (int missing = 0; missing < 3; ++missing)
                    {
                        const int32_t depth = kind == TD_CHAIN ? td_chain_depths[di] : (di == 0 ? 5 : 21);
                        const td_shape shape =
                            td_build(kind, td_tree_counts[ti], td_feature_counts[fi], depth, missing, &state);
                        const cmsis_nn_tree_table_f32 table = td_table(&shape);
                        td_make_rows(shape.feature_count, &state);
                        for (int32_t r = 0; r < td_row_count; ++r)
                            td_case(td_kind_names[kind], td_rows[r], &table, 7);
                    }
    td_report("shapes", before_cases, before_differences);
}

/* Every row of the base table against one changed table, then the base restored. */
static void td_rows_against(const char *what, const cmsis_nn_tree_table_f32 *table)
{
    for (int32_t r = 0; r < td_row_count; ++r)
        td_case(what, td_rows[r], table, 7);
}

static void td_refusals(void)
{
    static const int32_t odd_counts[] = {INT32_MIN, -1, 0, 1};
    static const int32_t most[4] = {32767, 32768, TD_MOST_TREES, 32767};
    static const uint32_t bad_roots[] = {0u, UINT32_C(0x80000000), UINT32_MAX};
    const unsigned long before_cases = td_cases;
    const unsigned long before_differences = td_differences;
    uint32_t state = UINT32_C(0x1b873593);
    for (int ti = 0; ti < 4; ++ti)
        for (int kind = TD_STUMP; kind < TD_KINDS; ++kind)
        {
            const int32_t trees = td_tree_counts[ti];
            const td_shape shape = td_build(kind, trees, 4, kind == TD_CHAIN ? 7 : 6, 0, &state);
            const cmsis_nn_tree_table_f32 good = td_table(&shape);
            cmsis_nn_tree_table_f32 table;
            td_make_rows(shape.feature_count, &state);

            /* Each pointer. */
            for (int pass = 0; pass < 7; ++pass)
                td_case("pointer", td_rows[0], &good, pass);
            table = good;
            table.nodes = NULL;
            td_case("NULL nodes", td_rows[0], &table, 7);
            table = good;
            table.roots = NULL;
            td_case("NULL roots", td_rows[0], &table, 7);

            /* Each count either side of its range, and at each end of it; a count above its
             * range is refused before the table is read, and one at the most a table holds is
             * read only along the path a row takes (tree_count, which reads every root, is the
             * case below). */
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
                        continue; /* in range and within the roots: an ordinary table */
                    if (field == 0)
                        table.node_count = values[i];
                    else if (field == 1)
                        table.leaf_count = values[i];
                    else if (field == 2)
                        table.tree_count = values[i];
                    else
                        table.feature_count = values[i];
                    td_rows_against("count", &table);
                }
            }

            /* A root out of range at tree 0, at a middle tree and at the last. */
            for (unsigned b = 0; b < sizeof(bad_roots) / sizeof(bad_roots[0]); ++b)
                for (int32_t at = 0; at < trees; at += (trees > 2 ? trees / 2 : 1))
                {
                    const uint32_t saved = td_roots[at];
                    td_roots[at] = bad_roots[b] == 0u ? (uint32_t)shape.node_count : bad_roots[b];
                    td_rows_against("root", &good);
                    td_roots[at] = saved;
                }

            /* A feature, a child or a leaf out of range at a split on the path (the root of
             * tree 0, of the last tree, and a split drawn at random), each side in turn. */
            for (int site = 0; site < 3; ++site)
            {
                const int32_t index = site == 0 ? (int32_t)td_roots[0]
                    : site == 1                 ? (int32_t)td_roots[trees - 1]
                                                : (int32_t)(td_next(&state) % (uint32_t)shape.node_count);
                const cmsis_nn_tree_node_f32 saved = td_nodes[index];
                const uint16_t features[] = {4u, (uint16_t)(TD_MISSING_LEFT | 4u), 0x7fffu, 0xffffu};
                const int16_t children[] = {
                    (int16_t)shape.node_count, INT16_MAX, (int16_t)(-shape.leaf_count - 1), INT16_MIN, (int16_t)index};
                for (unsigned i = 0; i < sizeof(features) / sizeof(features[0]); ++i)
                {
                    td_nodes[index].feature = features[i];
                    td_rows_against("feature", &good);
                    td_nodes[index] = saved;
                }
                for (unsigned i = 0; i < sizeof(children) / sizeof(children[0]); ++i)
                    for (int side = 0; side < 2; ++side)
                    {
                        if (side == 0)
                            td_nodes[index].left = children[i];
                        else
                            td_nodes[index].right = children[i];
                        td_rows_against(i == 4 ? "cycle" : "child or leaf", &good);
                        td_nodes[index] = saved;
                    }
                /* A cycle back to the tree's root from a split under it. */
                if (td_nodes[index].left >= 0)
                {
                    const int32_t child = td_nodes[index].left;
                    const cmsis_nn_tree_node_f32 child_saved = td_nodes[child];
                    td_nodes[child].right = (int16_t)index;
                    td_rows_against("cycle through a child", &good);
                    td_nodes[child] = child_saved;
                }
            }
        }
    td_report("refusals", before_cases, before_differences);
}

/* tree_count at its most, 65535 trees over three shared stumps, with a root out of range at
 * the first and at the last tree, and one past the most. Its buffers are the largest here. */
static void td_most_trees(void)
{
    static uint32_t roots[TD_MOST_TREES] TD_LARGE_SECTION;
    static int32_t got_leaves[TD_MOST_TREES + 2 * TD_GUARD] TD_LARGE_SECTION;
    static int32_t want_leaves[TD_MOST_TREES + 2 * TD_GUARD] TD_LARGE_SECTION;
    const unsigned long before_cases = td_cases;
    const unsigned long before_differences = td_differences;
    uint32_t state = UINT32_C(0x85ebca6b);
    const td_shape shape = td_build(TD_STUMP, 3, 2, 1, 0, &state);
    cmsis_nn_tree_table_f32 table = td_table(&shape);
    td_make_rows(2, &state);
    for (int32_t t = 0; t < TD_MOST_TREES; ++t)
        roots[t] = (uint32_t)(t % 3);
    table.roots = roots;
    for (int variant = 0; variant < 4; ++variant)
    {
        table.tree_count = variant == 3 ? TD_MOST_TREES + 1 : TD_MOST_TREES;
        if (variant == 1)
            roots[0] = 3u;
        if (variant == 2)
            roots[TD_MOST_TREES - 1] = 3u;
        for (int32_t r = 0; r < td_row_count; r += 4)
        {
            for (int32_t i = 0; i < TD_MOST_TREES + 2 * TD_GUARD; ++i)
            {
                got_leaves[i] = TD_SENTINEL;
                want_leaves[i] = TD_SENTINEL;
            }
            const arm_cmsis_nn_status got = arm_nn_tree_walk_f32(td_rows[r], &table, got_leaves + TD_GUARD);
            const arm_cmsis_nn_status want = twp_tree_walk_f32(td_rows[r], &table, want_leaves + TD_GUARD);
            ++td_cases;
            if (got != want || memcmp(got_leaves, want_leaves, sizeof(got_leaves)) != 0)
            {
                TEST_MESSAGE("difference: the most trees");
                ++td_differences;
            }
        }
        roots[0] = 0u;
        roots[TD_MOST_TREES - 1] = (uint32_t)((TD_MOST_TREES - 1) % 3);
    }
    td_report("most trees", before_cases, before_differences);
}
