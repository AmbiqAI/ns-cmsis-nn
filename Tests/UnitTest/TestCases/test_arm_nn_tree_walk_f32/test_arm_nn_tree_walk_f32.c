/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include <arm_nnfunctions.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unity.h>

#include "tree_walk_f32_data.h"

#define TW_SENTINEL INT32_C(0x5a5a5a5a)
#define TW_GUARD 4
#define TW_MISSING_LEFT UINT16_C(0x8000)

/* The limits, spelled here rather than read from the header, so moving one fails a case. */
#define TW_MOST_NODES 32767
#define TW_MOST_LEAVES 32768
#define TW_MOST_TREES 65535
#define TW_MOST_FEATURES 32767

#define TW_POSITIVE_INF UINT32_C(0x7F800000)
#define TW_NEGATIVE_INF UINT32_C(0xFF800000)

/*
 * The step budget, observed: the body reads each visited node's feature through one
 * memcpy, so a second compile of the same source in this file, with memcpy counted,
 * reports how many nodes a walk visited. The linked library entry is what every other
 * case calls.
 */
static int32_t tw_steps;
void *tw_counted_memcpy(void *destination, const void *source, size_t size);
void *tw_counted_memcpy(void *destination, const void *source, size_t size)
{
    ++tw_steps;
    return memcpy(destination, source, size);
}
arm_cmsis_nn_status tw_counted_walk(const float32_t *input, const cmsis_nn_tree_table_f32 *table, int32_t *leaf_ids);
#undef memcpy
#define memcpy(destination, source, size) tw_counted_memcpy(destination, source, size)
#define arm_nn_tree_walk_f32 tw_counted_walk
#include "../../../../Source/BasicMathFunctions/arm_nn_tree_walk_f32.c"
#undef arm_nn_tree_walk_f32
#undef memcpy

static float32_t tw_float(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static cmsis_nn_tree_node_f32 tw_node(float32_t threshold, uint16_t feature, int16_t left, int16_t right)
{
    cmsis_nn_tree_node_f32 node;
    memset(&node, 0, sizeof(node));
    node.threshold = threshold;
    node.feature = feature;
    node.left = left;
    node.right = right;
    return node;
}

static cmsis_nn_tree_table_f32 tw_table(const cmsis_nn_tree_node_f32 *nodes,
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

static void tw_fill(int32_t *leaf_ids, int32_t count)
{
    for (int32_t i = 0; i < count; ++i)
    {
        leaf_ids[i] = TW_SENTINEL;
    }
}

static void tw_assert_untouched(const int32_t *leaf_ids, int32_t count, const char *label)
{
    for (int32_t i = 0; i < count; ++i)
    {
        TEST_ASSERT_EQUAL_INT32_MESSAGE(TW_SENTINEL, leaf_ids[i], label);
    }
}

/* The one tree's leaf a one-feature row reaches, through the library entry. */
static int32_t tw_walk_one(const cmsis_nn_tree_node_f32 *nodes, int32_t node_count, int32_t leaf_count, float32_t x)
{
    const uint32_t root = 0;
    const cmsis_nn_tree_table_f32 table = tw_table(nodes, &root, node_count, leaf_count, 1, 1);
    int32_t leaf = TW_SENTINEL;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(&x, &table, &leaf));
    return leaf;
}

/* A writer packs into this record: its layout is the ABI, so the layout is a case. */
static void tw_record_layout(void)
{
    TEST_ASSERT_EQUAL_size_t(12, sizeof(cmsis_nn_tree_node_f32));
    TEST_ASSERT_EQUAL_size_t(0, offsetof(cmsis_nn_tree_node_f32, threshold));
    TEST_ASSERT_EQUAL_size_t(4, offsetof(cmsis_nn_tree_node_f32, feature));
    TEST_ASSERT_EQUAL_size_t(6, offsetof(cmsis_nn_tree_node_f32, left));
    TEST_ASSERT_EQUAL_size_t(8, offsetof(cmsis_nn_tree_node_f32, right));
}

/* Case (a): every leaf of every row of every table equals the numpy walk's. */
static void tw_random_tables(void)
{
    static cmsis_nn_tree_node_f32 nodes[TW_RANDOM_NODES];
    int32_t rows_walked = 0;
    int32_t leaves_compared = 0;

    TEST_ASSERT_EQUAL_INT32(64, TW_RANDOM_TABLES);
    TEST_ASSERT_EQUAL_INT32(32, TW_RANDOM_ROWS);
    for (int32_t i = 0; i < TW_RANDOM_NODES; ++i)
    {
        nodes[i] = tw_node(tw_float(tw_random_threshold[i]), tw_random_feature[i], tw_random_left[i], tw_random_right[i]);
    }
    for (int32_t t = 0; t < TW_RANDOM_TABLES; ++t)
    {
        const tw_random_record *const record = &tw_random_records[t];
        const cmsis_nn_tree_table_f32 table = tw_table(&nodes[record->node_start],
                                                       &tw_random_roots[record->root_start],
                                                       record->node_count,
                                                       record->leaf_count,
                                                       record->tree_count,
                                                       record->feature_count);
        const uint32_t *row_bits = &tw_random_rows[record->row_start];
        const int32_t *expected = &tw_random_expected[record->leaf_start];

        for (int32_t r = 0; r < TW_RANDOM_ROWS; ++r)
        {
            float32_t row[8];
            int32_t leaf_ids[8 + TW_GUARD];

            memcpy(row, row_bits, (size_t)record->feature_count * sizeof(float32_t));
            tw_fill(leaf_ids, 8 + TW_GUARD);
            TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &table, leaf_ids));
            TEST_ASSERT_EQUAL_INT32_ARRAY(expected, leaf_ids, record->tree_count);
            tw_assert_untouched(&leaf_ids[record->tree_count], 8 + TW_GUARD - record->tree_count, "past tree_count");
            row_bits += record->feature_count;
            expected += record->tree_count;
            leaves_compared += record->tree_count;
            ++rows_walked;
        }
    }
    TEST_ASSERT_EQUAL_INT32(2048, rows_walked);
    TEST_ASSERT_EQUAL_INT32(TW_RANDOM_LEAVES, leaves_compared);
}

/* Case (b): a NaN of any kind goes by bit 15, and bit 15 moves nothing else. */
static void tw_nan_routes(void)
{
    static const uint32_t nans[] = {
        UINT32_C(0x7FC00000),
        UINT32_C(0xFFC00000),
        UINT32_C(0x7F800001),
        UINT32_C(0x7FFFFFFF),
        UINT32_C(0xFF800001),
    };
    /* Both read feature 1, so a NaN in feature 0 must not reach them. */
    const cmsis_nn_tree_node_f32 nodes[] = {
        tw_node(1.0f, (uint16_t)(TW_MISSING_LEFT | 1u), -1, -2),
        tw_node(1.0f, 1u, -3, -4),
    };
    const uint32_t roots[] = {0u, 1u};
    const cmsis_nn_tree_table_f32 table = tw_table(nodes, roots, 2, 4, 2, 2);
    int32_t leaf_ids[2 + TW_GUARD];

    for (size_t i = 0; i < sizeof(nans) / sizeof(nans[0]); ++i)
    {
        const float32_t row[2] = {0.0f, tw_float(nans[i])};
        tw_fill(leaf_ids, 2 + TW_GUARD);
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &table, leaf_ids));
        TEST_ASSERT_EQUAL_INT32(0, leaf_ids[0]); /* bit 15 set: left */
        TEST_ASSERT_EQUAL_INT32(3, leaf_ids[1]); /* bit 15 clear: right */
    }
    {
        const float32_t below[2] = {tw_float(nans[0]), 0.0f};
        const float32_t above[2] = {tw_float(nans[0]), 2.0f};
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(below, &table, leaf_ids));
        TEST_ASSERT_EQUAL_INT32(0, leaf_ids[0]);
        TEST_ASSERT_EQUAL_INT32(2, leaf_ids[1]);
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(above, &table, leaf_ids));
        TEST_ASSERT_EQUAL_INT32(1, leaf_ids[0]);
        TEST_ASSERT_EQUAL_INT32(3, leaf_ids[1]);
    }
}

/* Case (c): +inf right of every finite threshold, -inf left, whatever bit 15 says. */
static void tw_infinities(void)
{
    const float32_t thresholds[] = {-FLT_MAX, -1.0f, 0.0f, 1.0f, FLT_MAX};

    for (size_t i = 0; i < sizeof(thresholds) / sizeof(thresholds[0]); ++i)
    {
        const cmsis_nn_tree_node_f32 clear = tw_node(thresholds[i], 0u, -1, -2);
        const cmsis_nn_tree_node_f32 set = tw_node(thresholds[i], TW_MISSING_LEFT, -1, -2);
        TEST_ASSERT_EQUAL_INT32(1, tw_walk_one(&clear, 1, 2, tw_float(TW_POSITIVE_INF)));
        TEST_ASSERT_EQUAL_INT32(1, tw_walk_one(&set, 1, 2, tw_float(TW_POSITIVE_INF)));
        TEST_ASSERT_EQUAL_INT32(0, tw_walk_one(&clear, 1, 2, tw_float(TW_NEGATIVE_INF)));
        TEST_ASSERT_EQUAL_INT32(0, tw_walk_one(&set, 1, 2, tw_float(TW_NEGATIVE_INF)));
    }
    {
        /* Compared as floats, so an infinity equals itself and <= sends it left. */
        const cmsis_nn_tree_node_f32 top = tw_node(tw_float(TW_POSITIVE_INF), 0u, -1, -2);
        TEST_ASSERT_EQUAL_INT32(0, tw_walk_one(&top, 1, 2, tw_float(TW_POSITIVE_INF)));
    }
}

/*
 * Case (d): at every split of the 7-split table, x == t and the float below it go left and
 * the float above it goes right. The leaf must equal the numpy walk's and lie under the
 * side this file's own table of leaves names for the split.
 */
static void tw_boundary_rows(void)
{
    static const int32_t left_first[TW_BOUNDARY_SPLITS] = {0, 0, 4, 0, 2, 4, 6};
    static const int32_t left_last[TW_BOUNDARY_SPLITS] = {3, 1, 5, 0, 2, 4, 6};
    static const int32_t right_first[TW_BOUNDARY_SPLITS] = {4, 2, 6, 1, 3, 5, 7};
    static const int32_t right_last[TW_BOUNDARY_SPLITS] = {7, 3, 7, 1, 3, 5, 7};
    static cmsis_nn_tree_node_f32 nodes[TW_BOUNDARY_SPLITS];
    const uint32_t root = 0;
    cmsis_nn_tree_table_f32 table;

    TEST_ASSERT_EQUAL_INT32(7, TW_BOUNDARY_SPLITS);
    TEST_ASSERT_EQUAL_INT32(21, TW_BOUNDARY_ROWS);
    for (int32_t i = 0; i < TW_BOUNDARY_SPLITS; ++i)
    {
        nodes[i] = tw_node(
            tw_float(tw_boundary_threshold[i]), tw_boundary_feature[i], tw_boundary_left[i], tw_boundary_right[i]);
    }
    table = tw_table(nodes, &root, TW_BOUNDARY_SPLITS, TW_BOUNDARY_SPLITS + 1, 1, TW_BOUNDARY_SPLITS);
    for (int32_t r = 0; r < TW_BOUNDARY_ROWS; ++r)
    {
        float32_t row[TW_BOUNDARY_SPLITS];
        int32_t leaf = TW_SENTINEL;
        const int32_t split = tw_boundary_split[r];
        const int32_t goes_left = tw_boundary_kind[r] != 1;

        memcpy(row, &tw_boundary_row_bits[r * TW_BOUNDARY_SPLITS], sizeof(row));
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &table, &leaf));
        TEST_ASSERT_EQUAL_INT32(tw_boundary_expected[r], leaf);
        TEST_ASSERT_GREATER_OR_EQUAL_INT32(goes_left ? left_first[split] : right_first[split], leaf);
        TEST_ASSERT_LESS_OR_EQUAL_INT32(goes_left ? left_last[split] : right_last[split], leaf);
    }
}

/* A float compare, not a compare of bits: the two zeros are equal both ways round. */
static void tw_signed_zeros(void)
{
    const cmsis_nn_tree_node_f32 positive = tw_node(0.0f, 0u, -1, -2);
    const cmsis_nn_tree_node_f32 negative = tw_node(tw_float(UINT32_C(0x80000000)), 0u, -1, -2);

    TEST_ASSERT_EQUAL_INT32(0, tw_walk_one(&positive, 1, 2, tw_float(UINT32_C(0x80000000))));
    TEST_ASSERT_EQUAL_INT32(0, tw_walk_one(&negative, 1, 2, 0.0f));
}

/*
 * Case (e), the record's three limits at once: 32767 splits in one chain, each sending a
 * row at or below its index left to the leaf of that index, 32768 leaves with the last at
 * the code -32768, and every split reading the last of 32767 features.
 */
static void tw_most_nodes_leaves_features(void)
{
    static cmsis_nn_tree_node_f32 nodes[TW_MOST_NODES];
    static float32_t row[TW_MOST_FEATURES];
    const uint32_t root = 0;
    const uint16_t feature = TW_MOST_FEATURES - 1;
    cmsis_nn_tree_table_f32 table;
    int32_t leaf = TW_SENTINEL;

    for (int32_t i = 0; i < TW_MOST_NODES; ++i)
    {
        const int16_t right = (i == TW_MOST_NODES - 1) ? INT16_MIN : (int16_t)(i + 1);
        nodes[i] = tw_node((float32_t)i, feature, (int16_t)(-i - 1), right);
    }
    for (int32_t i = 0; i < TW_MOST_FEATURES; ++i)
    {
        row[i] = -1.0f; /* A read of any other feature goes left at the first split. */
    }
    table = tw_table(nodes, &root, TW_MOST_NODES, TW_MOST_LEAVES, 1, TW_MOST_FEATURES);

    row[feature] = 1000.5f;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(1001, leaf);

    row[feature] = 32766.0f;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(32766, leaf);

    /* The whole chain, every node once: the budget holds a tree as deep as the table. */
    row[feature] = 40000.0f;
    tw_steps = 0;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, tw_counted_walk(row, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(TW_MOST_LEAVES - 1, leaf);
    TEST_ASSERT_EQUAL_INT32(TW_MOST_NODES, tw_steps);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(TW_MOST_LEAVES - 1, leaf);

    /* One leaf fewer, and the code -32768 is out of range. */
    table.leaf_count = TW_MOST_LEAVES - 1;
    leaf = TW_SENTINEL;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_f32(row, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(TW_SENTINEL, leaf);
}

/*
 * Case (e), the tree limit. 65535 distinct one-split trees would need more splits than the
 * record's 32767, so these share three: nothing in the kernel requires trees to be disjoint
 * and a root is only checked against node_count.
 */
static void tw_most_trees(void)
{
    static uint32_t roots[TW_MOST_TREES];
    static int32_t leaf_ids[TW_MOST_TREES + TW_GUARD];
    const cmsis_nn_tree_node_f32 nodes[3] = {
        tw_node(0.0f, 0u, -1, -2),
        tw_node(5.0f, 0u, -3, -4),
        tw_node(-5.0f, 0u, -5, -6),
    };
    const int32_t expected[3] = {1, 2, 5};
    const float32_t x = 1.0f;
    cmsis_nn_tree_table_f32 table;

    for (int32_t t = 0; t < TW_MOST_TREES; ++t)
    {
        roots[t] = (uint32_t)(t % 3);
    }
    table = tw_table(nodes, roots, 3, 6, TW_MOST_TREES, 1);
    tw_fill(leaf_ids, TW_MOST_TREES + TW_GUARD);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(&x, &table, leaf_ids));
    for (int32_t t = 0; t < TW_MOST_TREES; ++t)
    {
        TEST_ASSERT_EQUAL_INT32(expected[t % 3], leaf_ids[t]);
    }
    tw_assert_untouched(&leaf_ids[TW_MOST_TREES], TW_GUARD, "past the most trees");
}

/*
 * Case (f). A chain of N splits whose last child returns to the root visits every split in
 * N steps and the root again on step N + 1: FAILURE, after exactly that many. The same
 * chain ending at a leaf succeeds in N steps, so the budget is not short by one either.
 */
static void tw_cycle_guard(void)
{
    enum
    {
        TW_CHAIN = 5
    };
    cmsis_nn_tree_node_f32 nodes[TW_CHAIN];
    const uint32_t root = 0;
    const float32_t x = 1.0f;
    cmsis_nn_tree_table_f32 table;
    int32_t leaf = TW_SENTINEL;

    for (int32_t i = 0; i < TW_CHAIN; ++i)
    {
        const int16_t next = (i == TW_CHAIN - 1) ? 0 : (int16_t)(i + 1);
        nodes[i] = tw_node(2.0f, 0u, next, next);
    }
    table = tw_table(nodes, &root, TW_CHAIN, 1, 1, 1);

    tw_steps = 0;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE, tw_counted_walk(&x, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(TW_CHAIN + 1, tw_steps);
    TEST_ASSERT_EQUAL_INT32(TW_SENTINEL, leaf);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE, arm_nn_tree_walk_f32(&x, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(TW_SENTINEL, leaf);

    /* The smallest cycle: one split, its own child, two steps. */
    nodes[0] = tw_node(2.0f, 0u, 0, 0);
    table.node_count = 1;
    tw_steps = 0;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE, tw_counted_walk(&x, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(2, tw_steps);

    /* The last child a leaf instead: every split visited once, and the walk ends. */
    for (int32_t i = 0; i < TW_CHAIN; ++i)
    {
        const int16_t next = (i == TW_CHAIN - 1) ? -1 : (int16_t)(i + 1);
        nodes[i] = tw_node(2.0f, 0u, next, next);
    }
    table.node_count = TW_CHAIN;
    tw_steps = 0;
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, tw_counted_walk(&x, &table, &leaf));
    TEST_ASSERT_EQUAL_INT32(0, leaf);
    TEST_ASSERT_EQUAL_INT32(TW_CHAIN, tw_steps);
}

/* The table every refusal below is one change away from: two trees, three splits. */
static const uint32_t tw_good_roots[2] = {0u, 2u};

static void tw_good_nodes(cmsis_nn_tree_node_f32 *nodes)
{
    nodes[0] = tw_node(0.0f, 0u, 1, -1);
    nodes[1] = tw_node(-1.0f, 1u, -2, -3);
    nodes[2] = tw_node(1.0f, 1u, -4, -5);
}

static void tw_expect_refused(const float32_t *input, const cmsis_nn_tree_table_f32 *table, const char *label)
{
    int32_t leaf_ids[2 + TW_GUARD];
    tw_fill(leaf_ids, 2 + TW_GUARD);
    TEST_ASSERT_EQUAL_INT_MESSAGE(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_f32(input, table, leaf_ids), label);
    tw_assert_untouched(leaf_ids, 2 + TW_GUARD, label);
}

/* Case (g), the checks made before the walk: the sentinel survives each one. */
static void tw_refused_before_the_walk(void)
{
    cmsis_nn_tree_node_f32 nodes[3];
    const float32_t row[2] = {-5.0f, -5.0f};
    const cmsis_nn_tree_table_f32 good = tw_table(nodes, tw_good_roots, 3, 5, 2, 2);
    cmsis_nn_tree_table_f32 table;
    uint32_t roots[2];
    int32_t leaf_ids[2 + TW_GUARD];
    static const int32_t bad_nodes[] = {0, -1, INT32_MIN, TW_MOST_NODES + 1, INT32_MAX};
    static const int32_t bad_leaves[] = {0, -1, INT32_MIN, TW_MOST_LEAVES + 1, INT32_MAX};
    static const int32_t bad_trees[] = {0, -1, INT32_MIN, TW_MOST_TREES + 1, INT32_MAX};
    static const int32_t bad_features[] = {0, -1, INT32_MIN, TW_MOST_FEATURES + 1, INT32_MAX};
    static const uint32_t bad_roots[] = {3u, UINT32_C(0x80000000), UINT32_MAX};

    tw_good_nodes(nodes);

    /* The table is good: what refuses below is the one change, not the table. */
    tw_fill(leaf_ids, 2 + TW_GUARD);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &good, leaf_ids));
    TEST_ASSERT_EQUAL_INT32(1, leaf_ids[0]);
    TEST_ASSERT_EQUAL_INT32(3, leaf_ids[1]);

    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_f32(row, &good, NULL));
    tw_expect_refused(NULL, &good, "NULL input");
    tw_expect_refused(row, NULL, "NULL table");
    table = good;
    table.nodes = NULL;
    tw_expect_refused(row, &table, "NULL nodes");
    table = good;
    table.roots = NULL;
    tw_expect_refused(row, &table, "NULL roots");

    for (size_t i = 0; i < sizeof(bad_nodes) / sizeof(bad_nodes[0]); ++i)
    {
        table = good;
        table.node_count = bad_nodes[i];
        tw_expect_refused(row, &table, "node_count");
        table = good;
        table.leaf_count = bad_leaves[i];
        tw_expect_refused(row, &table, "leaf_count");
        table = good;
        table.tree_count = bad_trees[i];
        tw_expect_refused(row, &table, "tree_count");
        table = good;
        table.feature_count = bad_features[i];
        tw_expect_refused(row, &table, "feature_count");
    }

    /* In the last tree: the first tree is walkable, and still nothing is written. */
    for (size_t i = 0; i < sizeof(bad_roots) / sizeof(bad_roots[0]); ++i)
    {
        table = good;
        roots[0] = tw_good_roots[0];
        roots[1] = bad_roots[i];
        table.roots = roots;
        tw_expect_refused(row, &table, "root");
    }
}

/*
 * Case (g), the checks the walk makes, on the path of the first tree, where they too come
 * before any write. tw_later_tree_keeps_earlier_leaves is the other position.
 */
static void tw_refused_in_the_first_tree(void)
{
    cmsis_nn_tree_node_f32 nodes[3];
    const float32_t row[2] = {-5.0f, -5.0f};
    const cmsis_nn_tree_table_f32 table = tw_table(nodes, tw_good_roots, 3, 5, 2, 2);

    int32_t leaf_ids[2];

    /* The row goes left at split 0, then left at split 1 to leaf 1. */
    tw_good_nodes(nodes);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &table, leaf_ids));
    TEST_ASSERT_EQUAL_INT32(1, leaf_ids[0]);

    nodes[0].feature = 2u;
    tw_expect_refused(row, &table, "feature == feature_count");
    tw_good_nodes(nodes);
    nodes[0].feature = (uint16_t)(TW_MISSING_LEFT | 2u);
    tw_expect_refused(row, &table, "feature == feature_count, bit 15 set");
    tw_good_nodes(nodes);
    nodes[1].feature = INT16_MAX;
    tw_expect_refused(row, &table, "the largest feature");

    tw_good_nodes(nodes);
    nodes[0].left = 3;
    tw_expect_refused(row, &table, "left child == node_count");
    tw_good_nodes(nodes);
    nodes[0].left = INT16_MAX;
    tw_expect_refused(row, &table, "the largest left child");

    tw_good_nodes(nodes);
    nodes[1].left = -6;
    tw_expect_refused(row, &table, "left leaf == leaf_count");
    tw_good_nodes(nodes);
    nodes[1].left = INT16_MIN;
    tw_expect_refused(row, &table, "the largest left leaf");

    {
        /* The same two checks on the right child: this row goes right at split 0. */
        const float32_t above[2] = {5.0f, 5.0f};
        cmsis_nn_tree_node_f32 right_nodes[3];
        const cmsis_nn_tree_table_f32 right_table = tw_table(right_nodes, tw_good_roots, 3, 5, 2, 2);

        tw_good_nodes(right_nodes);
        right_nodes[0].right = 3;
        tw_expect_refused(above, &right_table, "right child == node_count");
        tw_good_nodes(right_nodes);
        right_nodes[0].right = -6;
        tw_expect_refused(above, &right_table, "right leaf == leaf_count");
    }
}

/*
 * The walk's checks read the path a row takes, so a defect in a later tree is found after
 * the earlier trees' leaves are written. The header states this; the case holds it: the
 * finished leaves are right and nothing from the failing tree on is touched.
 */
static void tw_later_tree_keeps_earlier_leaves(void)
{
    cmsis_nn_tree_node_f32 nodes[3];
    const float32_t row[2] = {-5.0f, -5.0f};
    const cmsis_nn_tree_table_f32 table = tw_table(nodes, tw_good_roots, 3, 5, 2, 2);
    int32_t leaf_ids[2 + TW_GUARD];

    tw_good_nodes(nodes);
    nodes[2].feature = 2u;
    tw_fill(leaf_ids, 2 + TW_GUARD);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, arm_nn_tree_walk_f32(row, &table, leaf_ids));
    TEST_ASSERT_EQUAL_INT32(1, leaf_ids[0]);
    tw_assert_untouched(&leaf_ids[1], 1 + TW_GUARD, "from the failing tree on");

    tw_good_nodes(nodes);
    nodes[2].left = 2;
    nodes[2].right = 2;
    tw_fill(leaf_ids, 2 + TW_GUARD);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_FAILURE, arm_nn_tree_walk_f32(row, &table, leaf_ids));
    TEST_ASSERT_EQUAL_INT32(1, leaf_ids[0]);
    tw_assert_untouched(&leaf_ids[1], 1 + TW_GUARD, "from the cycling tree on");
}

/* The walk checks the path it takes and no other: the header says so and a validator's job is the rest. */
static void tw_unreached_defect(void)
{
    cmsis_nn_tree_node_f32 nodes[3];
    const float32_t row[2] = {-5.0f, -5.0f};
    const cmsis_nn_tree_table_f32 table = tw_table(nodes, tw_good_roots, 3, 5, 2, 2);
    int32_t leaf_ids[2];

    tw_good_nodes(nodes);
    nodes[0].right = 3; /* This row goes left at split 0. */
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(row, &table, leaf_ids));
    TEST_ASSERT_EQUAL_INT32(1, leaf_ids[0]);
    TEST_ASSERT_EQUAL_INT32(3, leaf_ids[1]);
}

/* Case (h): T values in the order of roots, not of the nodes, and nothing past T. */
static void tw_tree_order(void)
{
    const cmsis_nn_tree_node_f32 nodes[3] = {
        tw_node(0.0f, 0u, -1, -2),
        tw_node(0.0f, 0u, -3, -4),
        tw_node(0.0f, 0u, -5, -6),
    };
    const uint32_t roots[3] = {2u, 0u, 1u};
    const cmsis_nn_tree_table_f32 table = tw_table(nodes, roots, 3, 6, 3, 1);
    const float32_t x = -1.0f;
    const int32_t expected[3] = {4, 0, 2};
    int32_t leaf_ids[3 + TW_GUARD];

    tw_fill(leaf_ids, 3 + TW_GUARD);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, arm_nn_tree_walk_f32(&x, &table, leaf_ids));
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, leaf_ids, 3);
    tw_assert_untouched(&leaf_ids[3], TW_GUARD, "past tree_count");
}
