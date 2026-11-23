/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_nn_tree_walk_values_f32 against its portable reference (tree_walk_values_reference.h), and against
 * scikit-learn's tree regressors on the golden set (tree_walk_values_f32_data.h).
 *
 * The entry copies words, so every comparison is of words: the status, then the whole output buffer, guard words
 * before and after values included, byte for byte, a NaN's sign, payload and quiet bit with it. No tolerance applies
 * on any build. The sweep builds its tables at run time from a seeded generator: tree shapes (single-leaf trees,
 * stumps, chains to depth 21, random trees to depth 21, roots shared among trees and placed at inner splits),
 * T in {1, 2, 3, 5, 16, 100}, F in {1, 4, 9}, the NaN route of every split random, all left and all right; rows
 * finite, then a NaN of each kind (quiet and signalling, both signs), each infinity and each zero in each feature in
 * turn; leaf values of three kinds (tv_kind_names), each at leaf_value_count = leaf_count and above it. */

#include <arm_nnfunctions.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "tree_walk_values_f32_data.h"
#include "tree_walk_values_reference.h"

#if defined(USING_FVP_CORSTONE_300)
    #define TV_LARGE_SECTION __attribute__((section(".bss.NoInit")))
#else
    #define TV_LARGE_SECTION
#endif

#define TV_TREES 100
#define TV_NODES 8192
#define TV_FEATURES 9
#define TV_ROWS 96
#define TV_LEAVES (TV_NODES + TV_TREES)
#define TV_EXTRA_VALUES 3
#define TV_GUARD 8
#define TV_SENTINEL UINT32_C(0x5a5a5a5a)
#define TV_REPORTED 8
#define TV_MISSING_LEFT UINT16_C(0x8000)

/* The limits, spelled here rather than read from the header, so moving one fails a case. */
#define TV_MOST_NODES 32767
#define TV_MOST_LEAVES 32768
#define TV_MOST_TREES 65535
#define TV_MOST_FEATURES 32767

static cmsis_nn_tree_node_f32 tv_nodes[TV_NODES];
static uint32_t tv_roots[TV_TREES];
static float32_t tv_rows[TV_ROWS][TV_FEATURES];
static int32_t tv_row_count;
static float32_t tv_leaf_values[TV_LEAVES + TV_EXTRA_VALUES];
static char tv_message[256];

static unsigned long tv_cases;
static unsigned long tv_differences;
static unsigned long tv_statuses[3];

static uint32_t tv_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static float32_t tv_float(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t tv_bits(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* The words of `count` values, each pair equal. */
static void tv_assert_words(const float32_t *want, const float32_t *got, int32_t count, const char *what)
{
    for (int32_t i = 0; i < count; ++i)
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(tv_bits(want[i]), tv_bits(got[i]), what);
}

static uint32_t tv_hash(uint32_t hash, const void *data, size_t size)
{
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < size; ++i)
        hash = (hash ^ bytes[i]) * UINT32_C(16777619);
    return hash;
}

/* Leaf words no float operation may change: NaNs of each sign, quiet and signalling, with payloads; the infinities;
 * both zeros; subnormals of each sign; the extremes of the normals. */
static const uint32_t tv_words[] = {
    UINT32_C(0x7fc00000), UINT32_C(0xffc00000), UINT32_C(0x7fc12345), UINT32_C(0xffffffff), UINT32_C(0x7f800001),
    UINT32_C(0xff800001), UINT32_C(0x7fbfffff), UINT32_C(0xffa5a5a5), UINT32_C(0x7f800000), UINT32_C(0xff800000),
    UINT32_C(0x00000000), UINT32_C(0x80000000), UINT32_C(0x00000001), UINT32_C(0x80000001), UINT32_C(0x007fffff),
    UINT32_C(0x807fffff), UINT32_C(0x00800000), UINT32_C(0x80800000), UINT32_C(0x7f7fffff), UINT32_C(0xff7fffff),
    UINT32_C(0x3f800000), UINT32_C(0xbf800000), UINT32_C(0x3eaaaaab), UINT32_C(0x4b000001)};
#define TV_WORDS ((uint32_t)(sizeof(tv_words) / sizeof(tv_words[0])))

/* ---- Tables and rows (the walk sweep's shapes) ---- */

static const uint32_t tv_thresholds[] = {UINT32_C(0xc0000000),
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
#define TV_THRESHOLDS ((uint32_t)(sizeof(tv_thresholds) / sizeof(tv_thresholds[0])))

static const uint32_t tv_specials[] = {UINT32_C(0x7fc00000),
                                       UINT32_C(0xffc00000),
                                       UINT32_C(0x7f800001),
                                       UINT32_C(0xff800001),
                                       UINT32_C(0x7fbfffff),
                                       UINT32_C(0x7f800000),
                                       UINT32_C(0xff800000),
                                       UINT32_C(0x00000000),
                                       UINT32_C(0x80000000)};
#define TV_SPECIALS ((int32_t)(sizeof(tv_specials) / sizeof(tv_specials[0])))

static float32_t tv_value(uint32_t *state)
{
    const uint32_t r = tv_next(state);
    const uint32_t t = tv_thresholds[(r >> 4) % TV_THRESHOLDS];
    switch (r & 3u)
    {
    case 0:
        return tv_float(t);
    case 1:
        return (t & 0x7fffffffu) == 0x7f7fffffu || (t & 0x7fffffffu) == 0u ? tv_float(t) : tv_float(t + 1u);
    case 2:
        return (t & 0x7fffffffu) == 0u ? tv_float(UINT32_C(0x80000001)) : tv_float(t - 1u);
    default:
        return (float32_t)(int32_t)((tv_next(state) >> 8) & 0xffffu) * (1.0f / 8192.0f) - 4.0f;
    }
}

typedef struct
{
    int32_t node_count;
    int32_t leaf_count;
    int32_t tree_count;
    int32_t feature_count;
} tv_shape;

enum
{
    TV_SINGLE_LEAF,
    TV_STUMP,
    TV_CHAIN,
    TV_RANDOM,
    TV_SHARED,
    TV_SHAPE_KINDS
};

static uint16_t tv_feature(uint32_t *state, int32_t features, int missing)
{
    const uint16_t feature = (uint16_t)(tv_next(state) % (uint32_t)features);
    const int left = missing == 0 ? (int)(tv_next(state) & 1u) : missing == 1;
    return (uint16_t)(feature | (left ? TV_MISSING_LEFT : 0u));
}

static void tv_split(int32_t index, uint32_t *state, int32_t features, int missing)
{
    memset(&tv_nodes[index], 0, sizeof(tv_nodes[index]));
    tv_nodes[index].threshold = tv_float(tv_thresholds[tv_next(state) % TV_THRESHOLDS]);
    tv_nodes[index].feature = tv_feature(state, features, missing);
}

static int16_t tv_leaf(tv_shape *shape) { return (int16_t)(-(shape->leaf_count++) - 1); }

static void tv_random_tree(tv_shape *shape, uint32_t *state, int32_t depth, int32_t budget, int missing)
{
    static int32_t queue[TV_NODES];
    static int32_t level[TV_NODES];
    int32_t head = 0;
    int32_t tail = 0;
    const int32_t first = shape->node_count;
    tv_split(shape->node_count++, state, shape->feature_count, missing);
    queue[tail] = first;
    level[tail++] = 1;
    while (head < tail)
    {
        const int32_t index = queue[head];
        const int32_t at = level[head++];
        for (int side = 0; side < 2; ++side)
        {
            int16_t child;
            if (at < depth && shape->node_count - first < budget && (tv_next(state) & 3u) != 0u)
            {
                child = (int16_t)shape->node_count;
                tv_split(shape->node_count++, state, shape->feature_count, missing);
                queue[tail] = child;
                level[tail++] = at + 1;
            }
            else
            {
                child = tv_leaf(shape);
            }
            if (side == 0)
                tv_nodes[index].left = child;
            else
                tv_nodes[index].right = child;
        }
    }
}

static tv_shape tv_build(int kind, int32_t trees, int32_t features, int32_t depth, int missing, uint32_t *state)
{
    tv_shape shape = {0, 0, trees, features};
    const int32_t budget = 60;
    for (int32_t t = 0; t < trees; ++t)
    {
        const int32_t root = shape.node_count;
        switch (kind)
        {
        case TV_SINGLE_LEAF: {
            const int16_t leaf = tv_leaf(&shape);
            tv_split(shape.node_count++, state, features, missing);
            tv_nodes[root].left = leaf;
            tv_nodes[root].right = leaf;
            break;
        }
        case TV_STUMP:
            tv_split(shape.node_count++, state, features, missing);
            tv_nodes[root].left = tv_leaf(&shape);
            tv_nodes[root].right = tv_leaf(&shape);
            break;
        case TV_CHAIN:
            for (int32_t d = 0; d < depth; ++d)
            {
                const int32_t index = shape.node_count;
                const int16_t next = d + 1 < depth ? (int16_t)(index + 1) : tv_leaf(&shape);
                const int16_t leaf = tv_leaf(&shape);
                tv_split(shape.node_count++, state, features, missing);
                if (tv_next(state) & 1u)
                {
                    tv_nodes[index].left = next;
                    tv_nodes[index].right = leaf;
                }
                else
                {
                    tv_nodes[index].left = leaf;
                    tv_nodes[index].right = next;
                }
            }
            break;
        default:
            tv_random_tree(&shape, state, depth, budget, missing);
            break;
        }
        tv_roots[t] = (uint32_t)root;
    }
    if (kind == TV_SHARED)
    {
        for (int32_t t = 0; t < trees; ++t)
            tv_roots[t] = tv_next(state) % (uint32_t)shape.node_count;
    }
    return shape;
}

static void tv_make_rows(int32_t features, uint32_t *state)
{
    tv_row_count = 0;
    for (int32_t r = 0; r < 6; ++r, ++tv_row_count)
        for (int32_t f = 0; f < features; ++f)
            tv_rows[tv_row_count][f] = tv_value(state);
    for (int32_t f = 0; f < features; ++f)
        for (int32_t s = 0; s < TV_SPECIALS; ++s, ++tv_row_count)
            for (int32_t g = 0; g < features; ++g)
                tv_rows[tv_row_count][g] = g == f ? tv_float(tv_specials[s]) : tv_value(state);
    for (int32_t f = 0; f < features; ++f)
        tv_rows[tv_row_count][f] = tv_float(tv_specials[f % 5]);
    ++tv_row_count;
}

static cmsis_nn_tree_table_f32 tv_table(const tv_shape *shape)
{
    cmsis_nn_tree_table_f32 table;
    memset(&table, 0, sizeof(table));
    table.nodes = tv_nodes;
    table.roots = tv_roots;
    table.node_count = shape->node_count;
    table.leaf_count = shape->leaf_count;
    table.tree_count = shape->tree_count;
    table.feature_count = shape->feature_count;
    return table;
}

/* ---- Leaf values ---- */

enum
{
    TV_ANY_WORD, /* any 32-bit pattern */
    TV_SPECIAL,  /* the words of tv_words only */
    TV_FINITE,   /* finite values of a regressor's range, both signs */
    TV_LEAF_KINDS
};
static const char *const tv_kind_names[TV_LEAF_KINDS] = {"any word", "special words", "finite"};
static unsigned long tv_kind_cases[TV_LEAF_KINDS];

/* `leaves` values of one kind, then TV_EXTRA_VALUES more that a correct entry never reads. */
static void tv_make_leaf_values(int kind, int32_t leaves, uint32_t *state)
{
    for (int32_t l = 0; l < leaves + TV_EXTRA_VALUES; ++l)
    {
        uint32_t word;
        if (kind == TV_ANY_WORD)
            word = tv_next(state);
        else if (kind == TV_SPECIAL)
            word = tv_words[tv_next(state) % TV_WORDS];
        else
            word = tv_bits((float32_t)((int32_t)(tv_next(state) & 0xfffffu) - 0x80000) * (1.0f / 1024.0f));
        tv_leaf_values[l] = tv_float(word);
    }
}

/* ---- One case ---- */

typedef struct
{
    float32_t values[TV_GUARD + TV_TREES + TV_GUARD];
} tv_out;

static tv_out tv_got;
static tv_out tv_want;

static void tv_fill(tv_out *out)
{
    for (size_t i = 0; i < sizeof(out->values) / sizeof(out->values[0]); ++i)
        out->values[i] = tv_float(TV_SENTINEL);
}

static int tv_untouched(const tv_out *out)
{
    for (size_t i = 0; i < sizeof(out->values) / sizeof(out->values[0]); ++i)
        if (tv_bits(out->values[i]) != TV_SENTINEL)
            return 0;
    return 1;
}

static int tv_guards_intact(const tv_out *out, int32_t trees)
{
    for (int32_t i = 0; i < TV_GUARD; ++i)
        if (tv_bits(out->values[i]) != TV_SENTINEL || tv_bits(out->values[TV_GUARD + trees + i]) != TV_SENTINEL)
            return 0;
    return 1;
}

/* Both bodies on one row and table; `pass` names the pointers passed (bit 0 input, 1 table, 2 leaf values, 3
 * values). A case differs when the statuses differ, when the two output buffers differ in any byte (guards and, on a
 * refusal, the prefix included), or when a guard word changed. Returns the entry's status. */
static arm_cmsis_nn_status tv_case(const char *what,
                                   int kind,
                                   const float32_t *row,
                                   const cmsis_nn_tree_table_f32 *table,
                                   const float32_t *leaf_values,
                                   int32_t leaf_value_count,
                                   int pass)
{
    tv_fill(&tv_got);
    tv_fill(&tv_want);
    const int32_t trees =
        table != NULL && table->tree_count >= 1 && table->tree_count <= TV_TREES ? table->tree_count : 0;
    const float32_t *input = (pass & 1) ? row : NULL;
    const cmsis_nn_tree_table_f32 *tab = (pass & 2) ? table : NULL;
    const float32_t *from = (pass & 4) ? leaf_values : NULL;
    const arm_cmsis_nn_status got =
        arm_nn_tree_walk_values_f32(input, tab, from, leaf_value_count, (pass & 8) ? &tv_got.values[TV_GUARD] : NULL);
    const arm_cmsis_nn_status want =
        tvr_reference(input, tab, from, leaf_value_count, (pass & 8) ? &tv_want.values[TV_GUARD] : NULL);
    int32_t at = -1;
    for (size_t i = 0; i < sizeof(tv_got.values) / sizeof(tv_got.values[0]) && at < 0; ++i)
        if (tv_bits(tv_got.values[i]) != tv_bits(tv_want.values[i]))
            at = (int32_t)i - TV_GUARD;
    const int differs = got != want || at != -1 || !tv_guards_intact(&tv_got, trees);
    ++tv_cases;
    if (kind >= 0)
        ++tv_kind_cases[kind];
    tv_statuses[got == ARM_CMSIS_NN_SUCCESS ? 0 : got == ARM_CMSIS_NN_ARG_ERROR ? 1 : 2] += 1;
    if (differs && ++tv_differences <= TV_REPORTED)
    {
        snprintf(tv_message,
                 sizeof(tv_message),
                 "difference: %s, kind %s, %ld trees, status %d against %d, first word %ld 0x%08lx against 0x%08lx",
                 what,
                 kind >= 0 ? tv_kind_names[kind] : "-",
                 table ? (long)table->tree_count : -1L,
                 (int)got,
                 (int)want,
                 (long)at,
                 at != -1 ? (unsigned long)tv_bits(tv_got.values[TV_GUARD + at]) : 0ul,
                 at != -1 ? (unsigned long)tv_bits(tv_want.values[TV_GUARD + at]) : 0ul);
        TEST_MESSAGE(tv_message);
    }
    return got;
}

static void tv_report(const char *what, unsigned long cases_before, unsigned long differences_before)
{
    snprintf(tv_message,
             sizeof(tv_message),
             "%s: %lu cases, %lu differences (all so far: %lu success, %lu argument error, %lu failure)",
             what,
             tv_cases - cases_before,
             tv_differences - differences_before,
             tv_statuses[0],
             tv_statuses[1],
             tv_statuses[2]);
    TEST_MESSAGE(tv_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(tv_differences - differences_before));
}

static uint32_t tv_inputs_hash(const tv_shape *shape)
{
    uint32_t hash = UINT32_C(2166136261);
    hash = tv_hash(hash, tv_nodes, (size_t)shape->node_count * sizeof(tv_nodes[0]));
    hash = tv_hash(hash, tv_roots, (size_t)shape->tree_count * sizeof(tv_roots[0]));
    hash = tv_hash(hash, tv_leaf_values, (size_t)(shape->leaf_count + TV_EXTRA_VALUES) * sizeof(float32_t));
    return tv_hash(hash, tv_rows, sizeof(tv_rows));
}

static const int32_t tv_tree_counts[] = {1, 2, 3, 5, 16, 100};
#define TV_TREE_COUNTS ((int)(sizeof(tv_tree_counts) / sizeof(tv_tree_counts[0])))
static const int32_t tv_feature_counts[] = {1, 4, 9};
#define TV_FEATURE_COUNTS ((int)(sizeof(tv_feature_counts) / sizeof(tv_feature_counts[0])))
static const int32_t tv_chain_depths[] = {1, 2, 7, 14, 21};
#define TV_CHAIN_DEPTHS ((int)(sizeof(tv_chain_depths) / sizeof(tv_chain_depths[0])))

static cmsis_nn_tree_node_f32 tv_node(float32_t threshold, uint16_t feature, int16_t left, int16_t right)
{
    cmsis_nn_tree_node_f32 node;
    memset(&node, 0, sizeof(node));
    node.threshold = threshold;
    node.feature = feature;
    node.left = left;
    node.right = right;
    return node;
}

static cmsis_nn_tree_table_f32 tv_small_table(const cmsis_nn_tree_node_f32 *nodes,
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

/* ---- The cases ---- */

static void tv_capability(void)
{
    TEST_ASSERT_EQUAL_INT(1, ARM_NN_HAS_TREE_WALK_VALUES_F32);
    TEST_ASSERT_EQUAL_INT(1, ARM_NN_HAS_TREE_WALK_F32);
}

/* Each word of tv_words through the entry by hand, alone and all at once: one tree per word, tree t's single split
 * sending both sides to leaf t, and a stump whose row goes left to one word and right to another. */
static void tv_words_pass_through(void)
{
    static cmsis_nn_tree_node_f32 nodes[TV_WORDS];
    static uint32_t roots[TV_WORDS];
    static float32_t leaf_values[TV_WORDS];
    float32_t values[TV_GUARD + TV_WORDS + TV_GUARD];
    const float32_t x = 0.5f;
    for (uint32_t i = 0; i < TV_WORDS; ++i)
    {
        nodes[i] = tv_node(0.0f, 0u, (int16_t)(-(int32_t)i - 1), (int16_t)(-(int32_t)i - 1));
        roots[i] = i;
        leaf_values[i] = tv_float(tv_words[i]);
    }
    for (uint32_t i = 0; i < TV_WORDS; ++i)
    {
        const uint32_t root = i;
        const cmsis_nn_tree_table_f32 one = tv_small_table(nodes, &root, (int32_t)TV_WORDS, (int32_t)TV_WORDS, 1, 1);
        float32_t value = tv_float(TV_SENTINEL);
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                              arm_nn_tree_walk_values_f32(&x, &one, leaf_values, (int32_t)TV_WORDS, &value));
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(tv_words[i], tv_bits(value), "one word");
    }
    const cmsis_nn_tree_table_f32 all =
        tv_small_table(nodes, roots, (int32_t)TV_WORDS, (int32_t)TV_WORDS, (int32_t)TV_WORDS, 1);
    for (uint32_t i = 0; i < TV_GUARD + TV_WORDS + TV_GUARD; ++i)
        values[i] = tv_float(TV_SENTINEL);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_tree_walk_values_f32(&x, &all, leaf_values, (int32_t)TV_WORDS, &values[TV_GUARD]));
    for (uint32_t i = 0; i < TV_GUARD; ++i)
    {
        TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(values[i]));
        TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(values[TV_GUARD + TV_WORDS + i]));
    }
    for (uint32_t i = 0; i < TV_WORDS; ++i)
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(tv_words[i], tv_bits(values[TV_GUARD + i]), "all words");

    /* A stump over -0.0f and a signalling NaN: the row picks each side in turn, a NaN row by the route bit. */
    {
        const cmsis_nn_tree_node_f32 stump[2] = {tv_node(0.0f, 0u, -1, -2), tv_node(0.0f, TV_MISSING_LEFT, -1, -2)};
        const float32_t sides[2] = {tv_float(UINT32_C(0x80000000)), tv_float(UINT32_C(0x7f800001))};
        const struct
        {
            uint32_t root;
            uint32_t x;
            uint32_t want;
        } cases[] = {
            {0u, UINT32_C(0x80000000), UINT32_C(0x80000000)}, /* -0 <= 0: left */
            {0u, UINT32_C(0x00000001), UINT32_C(0x7f800001)}, /* the smallest subnormal: right */
            {0u, UINT32_C(0xff800000), UINT32_C(0x80000000)}, /* -inf: left */
            {0u, UINT32_C(0x7f800000), UINT32_C(0x7f800001)}, /* +inf: right */
            {0u, UINT32_C(0xffc00001), UINT32_C(0x7f800001)}, /* a NaN, route bit clear: right */
            {1u, UINT32_C(0x7f800001), UINT32_C(0x80000000)}, /* a NaN, route bit set: left */
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
        {
            const float32_t row = tv_float(cases[i].x);
            const cmsis_nn_tree_table_f32 table = tv_small_table(stump, &cases[i].root, 2, 2, 1, 1);
            float32_t value = tv_float(TV_SENTINEL);
            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_values_f32(&row, &table, sides, 2, &value));
            TEST_ASSERT_EQUAL_HEX32_MESSAGE(cases[i].want, tv_bits(value), "stump side");
        }
    }
}

/* The precedence the declaration states, where several defects are present at once. */
static void tv_precedence(void)
{
    /* Tree 0 cycles at its root; tree 1's root is out of range; tree 2 reads a feature out of range. */
    const cmsis_nn_tree_node_f32 nodes[3] = {
        tv_node(0.0f, 0u, 0, 0),
        tv_node(0.0f, 0u, -1, -2),
        tv_node(0.0f, 7u, -1, -2),
    };
    const float32_t leaf_values[2] = {0.25f, -0.0f};
    const float32_t x = 1.0f;
    float32_t values[3] = {tv_float(TV_SENTINEL), tv_float(TV_SENTINEL), tv_float(TV_SENTINEL)};
    uint32_t roots[3] = {0u, 9u, 2u};
    cmsis_nn_tree_table_f32 table = tv_small_table(nodes, roots, 3, 2, 3, 1);

    /* An argument defect comes first, whatever the trees hold. */
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 2, NULL));
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_values_f32(&x, &table, NULL, 2, values));
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 1, values));
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_values_f32(NULL, &table, leaf_values, 2, values));
    for (int i = 0; i < 3; ++i)
        TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(values[i]));
    /* Then the trees in order: tree 0's cycle before tree 1's root, with nothing written. */
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE, arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 2, values));
    for (int i = 0; i < 3; ++i)
        TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(values[i]));
    /* Tree 0 good: tree 1's root, before tree 2's feature; tree 0's value written, the rest untouched. */
    roots[0] = 1u;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 2, values));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x80000000), tv_bits(values[0])); /* 1 > 0: right, leaf 1, -0.0f */
    TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(values[1]));
    TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(values[2]));
    /* Trees 0 and 1 good: tree 2's feature. */
    roots[1] = 1u;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 2, values));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x80000000), tv_bits(values[1]));
    TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(values[2]));
    /* Tree 2 cycles where it was out of range: a FAILURE after two good trees. */
    roots[2] = 0u;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE, arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 2, values));
    TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(values[2]));
    /* A defect off the row's path is not seen: node 0's cycle and node 2's feature are reached by no root. */
    roots[2] = 1u;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 2, values));
    for (int i = 0; i < 3; ++i)
        TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x80000000), tv_bits(values[i]));
    /* A longer leaf_values than the table's leaves is accepted. */
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_tree_walk_values_f32(&x, &table, leaf_values, INT32_MAX, values));
}

/* Every row of every table, at both leaf_value_counts, the leaf-value kind moving on with each table. */
static void tv_sweep(void)
{
    const unsigned long cases_before = tv_cases;
    const unsigned long differences_before = tv_differences;
    uint32_t state = UINT32_C(0x2545f491);
    int32_t table_number = 0;
    for (int ti = 0; ti < TV_TREE_COUNTS; ++ti)
        for (int fi = 0; fi < TV_FEATURE_COUNTS; ++fi)
            for (int shape_kind = 0; shape_kind < TV_SHAPE_KINDS; ++shape_kind)
                for (int di = 0; di < (shape_kind == TV_CHAIN ? TV_CHAIN_DEPTHS : 2); ++di)
                    for (int missing = 0; missing < 3; ++missing, ++table_number)
                    {
                        const int32_t depth = shape_kind == TV_CHAIN ? tv_chain_depths[di] : (di == 0 ? 5 : 21);
                        const tv_shape shape =
                            tv_build(shape_kind, tv_tree_counts[ti], tv_feature_counts[fi], depth, missing, &state);
                        const cmsis_nn_tree_table_f32 table = tv_table(&shape);
                        tv_make_rows(shape.feature_count, &state);
                        for (int k = 0; k < TV_LEAF_KINDS; ++k)
                        {
                            const int kind = (table_number + k) % TV_LEAF_KINDS;
                            tv_make_leaf_values(kind, shape.leaf_count, &state);
                            const uint32_t hash = tv_inputs_hash(&shape);
                            for (int32_t r = 0; r < tv_row_count; ++r)
                            {
                                tv_case("sweep", kind, tv_rows[r], &table, tv_leaf_values, shape.leaf_count, 15);
                                tv_case("sweep, more values",
                                        kind,
                                        tv_rows[r],
                                        &table,
                                        tv_leaf_values,
                                        shape.leaf_count + TV_EXTRA_VALUES,
                                        15);
                            }
                            TEST_ASSERT_EQUAL_HEX32_MESSAGE(hash, tv_inputs_hash(&shape), "inputs changed");
                        }
                    }
    tv_report("sweep", cases_before, differences_before);
    snprintf(tv_message,
             sizeof(tv_message),
             "sweep: %ld tables; cases by leaf values: %s %lu, %s %lu, %s %lu",
             (long)table_number,
             tv_kind_names[0],
             tv_kind_cases[0],
             tv_kind_names[1],
             tv_kind_cases[1],
             tv_kind_names[2],
             tv_kind_cases[2]);
    TEST_MESSAGE(tv_message);
}

static void tv_rows_against(const char *what, const cmsis_nn_tree_table_f32 *table, int32_t count, int must_refuse)
{
    for (int32_t r = 0; r < tv_row_count; ++r)
    {
        const arm_cmsis_nn_status status = tv_case(what, -1, tv_rows[r], table, tv_leaf_values, count, 15);
        if (must_refuse)
        {
            TEST_ASSERT_EQUAL_INT_MESSAGE(ARM_CMSIS_NN_ARG_ERROR, status, what);
            TEST_ASSERT_TRUE_MESSAGE(tv_untouched(&tv_got), what);
        }
    }
}

/* Each refusal, each from a good table with one thing changed; the reference states the expected status and the
 * prefix, and the case checks the guard words. The argument refusals also leave values untouched. */
static void tv_refusals(void)
{
    static const int32_t odd_counts[] = {INT32_MIN, -1, 0, 1};
    static const int32_t most[4] = {TV_MOST_NODES, TV_MOST_LEAVES, TV_MOST_TREES, TV_MOST_FEATURES};
    static const uint32_t bad_roots[] = {0u, UINT32_C(0x80000000), UINT32_MAX};
    const unsigned long cases_before = tv_cases;
    const unsigned long differences_before = tv_differences;
    uint32_t state = UINT32_C(0x68e31da4);
    for (int ti = 0; ti < TV_TREE_COUNTS; ++ti)
        for (int shape_kind = TV_STUMP; shape_kind < TV_SHAPE_KINDS; ++shape_kind)
        {
            const int32_t trees = tv_tree_counts[ti];
            const tv_shape shape = tv_build(shape_kind, trees, 4, shape_kind == TV_CHAIN ? 7 : 6, 0, &state);
            const int32_t leaves = shape.leaf_count;
            const cmsis_nn_tree_table_f32 good = tv_table(&shape);
            cmsis_nn_tree_table_f32 table;
            tv_make_rows(shape.feature_count, &state);
            tv_make_leaf_values(TV_SPECIAL, leaves, &state);

            /* Each pointer, in every combination, and the table's two. */
            for (int pass = 0; pass < 15; ++pass)
            {
                TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                                      tv_case("pointer", -1, tv_rows[0], &good, tv_leaf_values, leaves, pass));
                TEST_ASSERT_TRUE(tv_untouched(&tv_got));
            }
            table = good;
            table.nodes = NULL;
            tv_rows_against("NULL nodes", &table, leaves, 1);
            table = good;
            table.roots = NULL;
            tv_rows_against("NULL roots", &table, leaves, 1);

            /* Each count of the table either side of its range and at each end of it. */
            for (int field = 0; field < 4; ++field)
            {
                int32_t counts[8];
                int count = 0;
                for (unsigned i = 0; i < sizeof(odd_counts) / sizeof(odd_counts[0]); ++i)
                    counts[count++] = odd_counts[i];
                if (field != 2)
                    counts[count++] = most[field];
                counts[count++] = most[field] + 1;
                counts[count++] = INT32_MAX;
                for (int i = 0; i < count; ++i)
                {
                    table = good;
                    if (field == 2 && counts[i] > 0 && counts[i] <= trees)
                        continue;
                    if (field == 0)
                        table.node_count = counts[i];
                    else if (field == 1)
                        table.leaf_count = counts[i];
                    else if (field == 2)
                        table.tree_count = counts[i];
                    else
                        table.feature_count = counts[i];
                    /* A leaf_count raised past the values held passes leaf_value_count along with it, so the count
                     * under test, not the value count, decides. */
                    const int32_t values = field == 1 && counts[i] > leaves ? counts[i] : leaves;
                    tv_rows_against("count", &table, values, counts[i] < 1 || counts[i] > most[field]);
                }
            }

            /* leaf_value_count below the table's leaf count, at it, and above it. */
            {
                const int32_t refused[] = {INT32_MIN, -1, 0, leaves - 1};
                const int32_t accepted[] = {leaves, leaves + 1, INT32_MAX};
                for (unsigned i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i)
                    tv_rows_against("leaf_value_count refused", &good, refused[i], 1);
                for (unsigned i = 0; i < sizeof(accepted) / sizeof(accepted[0]); ++i)
                    for (int32_t r = 0; r < tv_row_count; ++r)
                        TEST_ASSERT_EQUAL_INT(
                            ARM_CMSIS_NN_SUCCESS,
                            tv_case(
                                "leaf_value_count accepted", -1, tv_rows[r], &good, tv_leaf_values, accepted[i], 15));
            }

            /* A root out of range at tree 0, at a middle tree and at the last. */
            for (unsigned b = 0; b < sizeof(bad_roots) / sizeof(bad_roots[0]); ++b)
                for (int32_t at = 0; at < trees; at += (trees > 2 ? trees / 2 : 1))
                {
                    const uint32_t saved = tv_roots[at];
                    tv_roots[at] = bad_roots[b] == 0u ? (uint32_t)shape.node_count : bad_roots[b];
                    for (int32_t r = 0; r < tv_row_count; ++r)
                        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                                              tv_case("root", -1, tv_rows[r], &good, tv_leaf_values, leaves, 15));
                    tv_roots[at] = saved;
                }
            {
                const uint32_t saved = tv_roots[trees - 1];
                tv_roots[trees - 1] = (uint32_t)shape.node_count;
                tv_rows_against("root, last tree", &good, leaves, 0);
                tv_roots[trees - 1] = saved;
            }

            /* A feature, a child or a leaf out of range at a split on the path, each side in turn, and cycles. */
            for (int site = 0; site < 3; ++site)
            {
                const int32_t index = site == 0 ? (int32_t)tv_roots[0]
                    : site == 1                 ? (int32_t)tv_roots[trees - 1]
                                                : (int32_t)(tv_next(&state) % (uint32_t)shape.node_count);
                const cmsis_nn_tree_node_f32 saved = tv_nodes[index];
                const uint16_t features[] = {4u, (uint16_t)(TV_MISSING_LEFT | 4u), 0x7fffu, 0xffffu};
                const int16_t children[] = {
                    (int16_t)shape.node_count, INT16_MAX, (int16_t)(-leaves - 1), INT16_MIN, (int16_t)index};
                for (unsigned i = 0; i < sizeof(features) / sizeof(features[0]); ++i)
                {
                    tv_nodes[index].feature = features[i];
                    tv_rows_against("feature", &good, leaves, 0);
                    tv_nodes[index] = saved;
                }
                for (unsigned i = 0; i < sizeof(children) / sizeof(children[0]); ++i)
                    for (int side = 0; side < 2; ++side)
                    {
                        if (side == 0)
                            tv_nodes[index].left = children[i];
                        else
                            tv_nodes[index].right = children[i];
                        tv_rows_against(i == 4 ? "cycle" : "child or leaf", &good, leaves, 0);
                        tv_nodes[index] = saved;
                    }
                if (tv_nodes[index].left >= 0)
                {
                    const int32_t child = tv_nodes[index].left;
                    const cmsis_nn_tree_node_f32 child_saved = tv_nodes[child];
                    tv_nodes[child].right = (int16_t)index;
                    tv_rows_against("cycle through a child", &good, leaves, 0);
                    tv_nodes[child] = child_saved;
                }
            }
        }
    tv_report("refusals", cases_before, differences_before);
}

/* The limits, generated here: 32767 splits in one chain reaching 32768 leaves over 32767 features; 65535 trees over
 * three shared stumps, then a bad root at the last tree, then one tree past the most. */
static void tv_limits(void)
{
    static cmsis_nn_tree_node_f32 nodes[TV_MOST_NODES] TV_LARGE_SECTION;
    static float32_t row[TV_MOST_FEATURES] TV_LARGE_SECTION;
    static float32_t leaf_values[TV_MOST_LEAVES] TV_LARGE_SECTION;
    static uint32_t roots[TV_MOST_TREES] TV_LARGE_SECTION;
    static float32_t got[TV_GUARD + TV_MOST_TREES + TV_GUARD] TV_LARGE_SECTION;
    static float32_t want[TV_GUARD + TV_MOST_TREES + TV_GUARD] TV_LARGE_SECTION;
    const uint16_t feature = TV_MOST_FEATURES - 1;
    uint32_t root = 0;
    float32_t value;
    cmsis_nn_tree_table_f32 table;

    /* The chain: split i sends a value at or below i to leaf i, else to split i + 1; the last split's right is the
     * code -32768. Leaf l's value is the signalling NaN 0x7f800000 + l + 1, so the word names the leaf. */
    for (int32_t i = 0; i < TV_MOST_NODES; ++i)
    {
        const int16_t right = (i == TV_MOST_NODES - 1) ? INT16_MIN : (int16_t)(i + 1);
        nodes[i] = tv_node((float32_t)i, feature, (int16_t)(-i - 1), right);
    }
    for (int32_t i = 0; i < TV_MOST_FEATURES; ++i)
        row[i] = -1.0f;
    for (int32_t l = 0; l < TV_MOST_LEAVES; ++l)
        leaf_values[l] = tv_float(UINT32_C(0x7f800000) + (uint32_t)l + 1u);
    table = tv_small_table(nodes, &root, TV_MOST_NODES, TV_MOST_LEAVES, 1, TV_MOST_FEATURES);
    row[feature] = 40000.0f;
    value = tv_float(TV_SENTINEL);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_tree_walk_values_f32(row, &table, leaf_values, TV_MOST_LEAVES, &value));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x7f800000) + 32768u, tv_bits(value));
    row[feature] = 1000.5f;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                          arm_nn_tree_walk_values_f32(row, &table, leaf_values, TV_MOST_LEAVES, &value));
    TEST_ASSERT_EQUAL_HEX32(UINT32_C(0x7f800000) + 1002u, tv_bits(value));
    /* One value fewer than the leaves, whatever leaf the row reaches. */
    value = tv_float(TV_SENTINEL);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          arm_nn_tree_walk_values_f32(row, &table, leaf_values, TV_MOST_LEAVES - 1, &value));
    TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(value));
    /* One leaf fewer: the code -32768 is out of range on the path. */
    row[feature] = 40000.0f;
    table.leaf_count = TV_MOST_LEAVES - 1;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          arm_nn_tree_walk_values_f32(row, &table, leaf_values, TV_MOST_LEAVES, &value));
    TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(value));

    /* 65535 trees over three shared stumps; then a root out of range at the last tree, then one tree past the most. */
    {
        const cmsis_nn_tree_node_f32 stumps[3] = {
            tv_node(0.0f, 0u, -1, -2),
            tv_node(5.0f, 0u, -3, -4),
            tv_node(-5.0f, 0u, -5, -6),
        };
        const float32_t x = 1.0f;
        for (int32_t t = 0; t < TV_MOST_TREES; ++t)
            roots[t] = (uint32_t)(t % 3);
        for (int32_t l = 0; l < 6; ++l)
            leaf_values[l] = tv_float(tv_words[(uint32_t)l * 5u % TV_WORDS]);
        table = tv_small_table(stumps, roots, 3, 6, TV_MOST_TREES, 1);
        for (int pass = 0; pass < 2; ++pass)
        {
            for (int32_t i = 0; i < TV_GUARD + TV_MOST_TREES + TV_GUARD; ++i)
                got[i] = want[i] = tv_float(TV_SENTINEL);
            const arm_cmsis_nn_status expected = pass == 0 ? ARM_CMSIS_NN_SUCCESS : ARM_CMSIS_NN_ARG_ERROR;
            if (pass == 1)
                roots[TV_MOST_TREES - 1] = 3u;
            TEST_ASSERT_EQUAL_INT(expected, arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 6, &got[TV_GUARD]));
            TEST_ASSERT_EQUAL_INT(expected, tvr_reference(&x, &table, leaf_values, 6, &want[TV_GUARD]));
            tv_assert_words(want, got, TV_GUARD + TV_MOST_TREES + TV_GUARD, "most trees");
            TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(got[TV_GUARD + TV_MOST_TREES]));
        }
        /* The prefix of the refusal at the last tree: every tree before it written. Tree 65533's root is stump 1,
         * where 1 <= 5 goes left to leaf 2, whose word is tv_words[10], +0.0f. */
        TEST_ASSERT_EQUAL_HEX32(tv_words[10], tv_bits(got[TV_GUARD + TV_MOST_TREES - 2]));
        TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(got[TV_GUARD + TV_MOST_TREES - 1]));
        roots[TV_MOST_TREES - 1] = (uint32_t)((TV_MOST_TREES - 1) % 3);
        table.tree_count = TV_MOST_TREES + 1;
        got[TV_GUARD] = tv_float(TV_SENTINEL);
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                              arm_nn_tree_walk_values_f32(&x, &table, leaf_values, 6, &got[TV_GUARD]));
        TEST_ASSERT_EQUAL_HEX32(TV_SENTINEL, tv_bits(got[TV_GUARD]));
    }
}

/* The golden set. Each fit's rows through the entry twice: with the fit's leaf values, each tree's word against
 * float32(tree_.value[apply(row)]); and with leaf l's value the word l, so the entry's word is the leaf it reached,
 * against apply()'s. Both calls are also held to the reference. The (row, tree) pairs whose leaf differs from apply()
 * must be exactly those the generator listed (tvg_differing), and a value may differ only at such a pair. */
static void tv_golden(void)
{
    static cmsis_nn_tree_node_f32 nodes[TVG_MOST_NODES];
    static uint32_t roots[TVG_MOST_TREES];
    static float32_t leaf_values[TVG_MOST_LEAVES];
    static float32_t leaf_ids[TVG_MOST_LEAVES];
    float32_t row[TVG_MOST_FEATURES];
    float32_t got[TVG_MOST_TREES];
    float32_t want[TVG_MOST_TREES];
    long total_rows = 0;
    long total_pairs = 0;
    long total_leaf_rows = 0;
    long total_value_pairs = 0;
    for (int32_t l = 0; l < TVG_MOST_LEAVES; ++l)
        leaf_ids[l] = tv_float((uint32_t)l);
    for (int f = 0; f < TVG_FITS; ++f)
    {
        const tvg_fit *const fit = &tvg_fits[f];
        long leaf_pairs = 0;
        long leaf_rows = 0;
        long value_pairs = 0;
        long listed = 0;
        for (int32_t i = 0; i < fit->node_count; ++i)
        {
            const int32_t at = fit->node_start + i;
            nodes[i] = tv_node(tv_float(tvg_threshold[at]), tvg_feature[at], tvg_left[at], tvg_right[at]);
        }
        for (int32_t t = 0; t < fit->tree_count; ++t)
            roots[t] = tvg_roots[fit->root_start + t];
        for (int32_t l = 0; l < fit->leaf_count; ++l)
            leaf_values[l] = tv_float(tvg_leaf_values[fit->leaf_value_start + l]);
        const cmsis_nn_tree_table_f32 table =
            tv_small_table(nodes, roots, fit->node_count, fit->leaf_count, fit->tree_count, fit->feature_count);
        for (int32_t r = 0; r < fit->row_count; ++r)
        {
            for (int32_t j = 0; j < fit->feature_count; ++j)
                row[j] = tv_float(tvg_rows[fit->row_value_start + r * fit->feature_count + j]);
            const int32_t first = fit->expect_start + r * fit->tree_count;
            int row_differs = 0;

            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                                  arm_nn_tree_walk_values_f32(row, &table, leaf_values, fit->leaf_count, got));
            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, tvr_reference(row, &table, leaf_values, fit->leaf_count, want));
            tv_assert_words(want, got, fit->tree_count, fit->name);
            for (int32_t t = 0; t < fit->tree_count; ++t)
                value_pairs += tv_bits(got[t]) != tvg_expected[first + t];

            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS,
                                  arm_nn_tree_walk_values_f32(row, &table, leaf_ids, fit->leaf_count, got));
            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, tvr_reference(row, &table, leaf_ids, fit->leaf_count, want));
            tv_assert_words(want, got, fit->tree_count, fit->name);
            for (int32_t t = 0; t < fit->tree_count; ++t)
            {
                const uint32_t leaf = tv_bits(got[t]);
                if (leaf == tvg_apply_leaf[first + t])
                    continue;
                ++leaf_pairs;
                row_differs = 1;
                for (int32_t d = 0; d < fit->differ_count; ++d)
                {
                    const tvg_differ *const pair = &tvg_differing[fit->differ_start + d];
                    if (pair->row == r && pair->tree == t && (uint32_t)pair->walk_leaf == leaf &&
                        (uint32_t)pair->apply_leaf == tvg_apply_leaf[first + t])
                        ++listed;
                }
            }
            leaf_rows += row_differs;
        }
        snprintf(tv_message,
                 sizeof(tv_message),
                 "golden %s: %ld trees, %ld rows, %ld (row, tree) pairs; leaf differs from apply: %ld rows, %ld "
                 "pairs (%ld listed); value differs from float32(tree_.value[apply]): %ld pairs",
                 fit->name,
                 (long)fit->tree_count,
                 (long)fit->row_count,
                 (long)fit->row_count * fit->tree_count,
                 leaf_rows,
                 leaf_pairs,
                 listed,
                 value_pairs);
        TEST_MESSAGE(tv_message);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(fit->differ_count, (int32_t)leaf_pairs, fit->name);
        TEST_ASSERT_EQUAL_INT32_MESSAGE(fit->differ_count, (int32_t)listed, fit->name);
        TEST_ASSERT_TRUE_MESSAGE(value_pairs <= leaf_pairs, fit->name);
        total_rows += fit->row_count;
        total_pairs += (long)fit->row_count * fit->tree_count;
        total_leaf_rows += leaf_rows;
        total_value_pairs += value_pairs;
    }
    snprintf(tv_message,
             sizeof(tv_message),
             "golden: %d fits, %ld rows, %ld (row, tree) pairs; leaf differs from apply: %ld rows (%d pairs listed); "
             "value differs: %ld pairs",
             TVG_FITS,
             total_rows,
             total_pairs,
             total_leaf_rows,
             TVG_DIFFERING,
             total_value_pairs);
    TEST_MESSAGE(tv_message);
}
