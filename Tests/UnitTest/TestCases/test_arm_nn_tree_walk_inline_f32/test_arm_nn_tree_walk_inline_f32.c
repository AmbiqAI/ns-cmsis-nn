/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* The bodies of the fused tree entries called as generated code calls them: this translation unit includes
 * Internal/arm_nn_tree_walk_flt.h and calls arm_nn_tree_walk_mean_argmax_body_f32 and arm_nn_tree_walk_values_body_f32
 * from functions of its own, each over tables of its own and with every count a literal, as a module of one fitted
 * model would. The tables are those of golden fits whose counts are the literals (each load checks it):
 *
 *   ti_m1    dt_iris            1 tree,    3 classes, 4 features,  5 splits,   6 leaves  (one tree writing a class)
 *   ti_m5    rf5_wine           5 trees,   3 classes, 13 features, 49 splits,  54 leaves
 *   ti_m16   ada16_wine         16 trees,  3 classes, 13 features, 48 splits,  64 leaves
 *   ti_m100  rf100_iris         100 trees, 3 classes, 4 features,  589 splits, 689 leaves
 *   ti_v1    dt_diabetes        1 tree,    10 features, 301 splits, 302 leaves  (values)
 *   ti_v16   rf16_friedman_nan  16 trees,  7 features,  240 splits, 256 leaves  (values)
 *   ti_v100  rf100_diabetes     100 trees, 10 features, 700 splits, 800 leaves  (values)
 *
 * Each mean module has two forms: _scores writes the caller's scores and class, and _class keeps its scores in a local
 * array and writes the class alone. Every case runs the public entry, each form of the module and the portable
 * reference of the entry's suite on one row and table. A case differs when any status differs, when a word of the
 * entry's and the module's outputs differs (the guard words around them and, on a refusal, what the body left written
 * included), when the class of either form differs from the entry's, or, on success, when a word of the reference's
 * scores or values or its class differs from the entry's. Nothing is tolerated: the leaf rows here hold no subnormal
 * and no NaN, so no build may answer otherwise.
 *
 * The cases: the golden rows; the rows with a NaN of each kind in each feature, with every split's NaN route set
 * left and then right; leaf rows that tie two classes at the top, so the first maximum decides; leaf rows whose mean
 * by the reciprocal differs from the mean by a divide in every class (where the tree count is not a power of two);
 * the malformed tables of the entry's refusals (a root, a child, a leaf and a feature out of range, and cycles), each
 * through the module as through the entry; and the body's own count checks, with the literal count out of range.
 * Each table, row and output is an array of exactly its size, so a read or write past one is seen by
 * AddressSanitizer. */

#include <arm_nnfunctions.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "../../../../Include/Internal/arm_nn_tree_walk_flt.h"
#include "../test_arm_nn_tree_walk_mean_argmax_f32/tree_mean_argmax_f32_data.h"
#include "../test_arm_nn_tree_walk_mean_argmax_f32/tree_mean_argmax_reference.h"
#include "../test_arm_nn_tree_walk_values_f32/tree_walk_values_f32_data.h"
#include "../test_arm_nn_tree_walk_values_f32/tree_walk_values_reference.h"

#define TI_CLASSES 3
#define TI_TREES 100
#define TI_MOST_NODES 700
#define TI_GUARD 8
#define TI_SCORE_SENTINEL UINT32_C(0x5a5a5a5a)
#define TI_INDEX_SENTINEL INT32_C(0x3c3c3c3c)
#define TI_MISSING_LEFT UINT16_C(0x8000)
#define TI_REPORTED 8

/* ---- The modules: tables of their own and literal counts ---- */

static cmsis_nn_tree_node_f32 ti_m1_nodes[5];
static uint32_t ti_m1_roots[1];
static float32_t ti_m1_leaf_rows[6 * 3];
static float32_t ti_m1_row[4];

static cmsis_nn_tree_node_f32 ti_m5_nodes[49];
static uint32_t ti_m5_roots[5];
static float32_t ti_m5_leaf_rows[54 * 3];
static float32_t ti_m5_row[13];

static cmsis_nn_tree_node_f32 ti_m16_nodes[48];
static uint32_t ti_m16_roots[16];
static float32_t ti_m16_leaf_rows[64 * 3];
static float32_t ti_m16_row[13];

static cmsis_nn_tree_node_f32 ti_m100_nodes[589];
static uint32_t ti_m100_roots[100];
static float32_t ti_m100_leaf_rows[689 * 3];
static float32_t ti_m100_row[4];

static cmsis_nn_tree_node_f32 ti_v1_nodes[301];
static uint32_t ti_v1_roots[1];
static float32_t ti_v1_leaf_values[302];
static float32_t ti_v1_row[10];

static cmsis_nn_tree_node_f32 ti_v16_nodes[240];
static uint32_t ti_v16_roots[16];
static float32_t ti_v16_leaf_values[256];
static float32_t ti_v16_row[7];

static cmsis_nn_tree_node_f32 ti_v100_nodes[700];
static uint32_t ti_v100_roots[100];
static float32_t ti_v100_leaf_values[800];
static float32_t ti_v100_short_values[799];
static float32_t ti_v100_row[10];

arm_cmsis_nn_status ti_m1_scores(const float32_t *input, float32_t *scores, int32_t *class_index);
arm_cmsis_nn_status ti_m1_class(const float32_t *input, int32_t *class_index);
arm_cmsis_nn_status ti_m5_scores(const float32_t *input, float32_t *scores, int32_t *class_index);
arm_cmsis_nn_status ti_m5_class(const float32_t *input, int32_t *class_index);
arm_cmsis_nn_status ti_m16_scores(const float32_t *input, float32_t *scores, int32_t *class_index);
arm_cmsis_nn_status ti_m16_class(const float32_t *input, int32_t *class_index);
arm_cmsis_nn_status ti_m100_scores(const float32_t *input, float32_t *scores, int32_t *class_index);
arm_cmsis_nn_status ti_m100_class(const float32_t *input, int32_t *class_index);
arm_cmsis_nn_status ti_m100_no_classes(const float32_t *input, float32_t *scores, int32_t *class_index);
arm_cmsis_nn_status ti_m100_too_many_classes(const float32_t *input, float32_t *scores, int32_t *class_index);
arm_cmsis_nn_status ti_v1_values(const float32_t *input, float32_t *values);
arm_cmsis_nn_status ti_v16_values(const float32_t *input, float32_t *values);
arm_cmsis_nn_status ti_v100_values(const float32_t *input, float32_t *values);
arm_cmsis_nn_status ti_v100_short(const float32_t *input, float32_t *values);

__attribute__((noinline)) arm_cmsis_nn_status ti_m1_scores(const float32_t *input,
                                                           float32_t *scores,
                                                           int32_t *class_index)
{
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m1_nodes, ti_m1_roots, 5, 6, 1, 4, ti_m1_leaf_rows, 3, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_m1_class(const float32_t *input, int32_t *class_index)
{
    float32_t scores[3];
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m1_nodes, ti_m1_roots, 5, 6, 1, 4, ti_m1_leaf_rows, 3, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_m5_scores(const float32_t *input,
                                                           float32_t *scores,
                                                           int32_t *class_index)
{
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m5_nodes, ti_m5_roots, 49, 54, 5, 13, ti_m5_leaf_rows, 3, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_m5_class(const float32_t *input, int32_t *class_index)
{
    float32_t scores[3];
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m5_nodes, ti_m5_roots, 49, 54, 5, 13, ti_m5_leaf_rows, 3, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_m16_scores(const float32_t *input,
                                                            float32_t *scores,
                                                            int32_t *class_index)
{
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m16_nodes, ti_m16_roots, 48, 64, 16, 13, ti_m16_leaf_rows, 3, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_m16_class(const float32_t *input, int32_t *class_index)
{
    float32_t scores[3];
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m16_nodes, ti_m16_roots, 48, 64, 16, 13, ti_m16_leaf_rows, 3, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_m100_scores(const float32_t *input,
                                                             float32_t *scores,
                                                             int32_t *class_index)
{
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m100_nodes, ti_m100_roots, 589, 689, 100, 4, ti_m100_leaf_rows, 3, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_m100_class(const float32_t *input, int32_t *class_index)
{
    float32_t scores[3];
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m100_nodes, ti_m100_roots, 589, 689, 100, 4, ti_m100_leaf_rows, 3, scores, class_index);
}

/* The body's class_count check with the literal out of range: below 1, and one past INT32_MAX / (4 x 689). */
__attribute__((noinline)) arm_cmsis_nn_status ti_m100_no_classes(const float32_t *input,
                                                                 float32_t *scores,
                                                                 int32_t *class_index)
{
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m100_nodes, ti_m100_roots, 589, 689, 100, 4, ti_m100_leaf_rows, 0, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_m100_too_many_classes(const float32_t *input,
                                                                       float32_t *scores,
                                                                       int32_t *class_index)
{
    return arm_nn_tree_walk_mean_argmax_body_f32(
        input, ti_m100_nodes, ti_m100_roots, 589, 689, 100, 4, ti_m100_leaf_rows, 779204, scores, class_index);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_v1_values(const float32_t *input, float32_t *values)
{
    return arm_nn_tree_walk_values_body_f32(
        input, ti_v1_nodes, ti_v1_roots, 301, 302, 1, 10, ti_v1_leaf_values, 302, values);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_v16_values(const float32_t *input, float32_t *values)
{
    return arm_nn_tree_walk_values_body_f32(
        input, ti_v16_nodes, ti_v16_roots, 240, 256, 16, 7, ti_v16_leaf_values, 256, values);
}

__attribute__((noinline)) arm_cmsis_nn_status ti_v100_values(const float32_t *input, float32_t *values)
{
    return arm_nn_tree_walk_values_body_f32(
        input, ti_v100_nodes, ti_v100_roots, 700, 800, 100, 10, ti_v100_leaf_values, 800, values);
}

/* The body's leaf_value_count check with the literal one below the leaf count, over an array of that many values. */
__attribute__((noinline)) arm_cmsis_nn_status ti_v100_short(const float32_t *input, float32_t *values)
{
    return arm_nn_tree_walk_values_body_f32(
        input, ti_v100_nodes, ti_v100_roots, 700, 800, 100, 10, ti_v100_short_values, 799, values);
}

/* ---- What a case knows of a module ---- */

typedef arm_cmsis_nn_status (*ti_scores_form)(const float32_t *input, float32_t *scores, int32_t *class_index);
typedef arm_cmsis_nn_status (*ti_class_form)(const float32_t *input, int32_t *class_index);
typedef arm_cmsis_nn_status (*ti_values_form)(const float32_t *input, float32_t *values);

/* A module's tables, the counts its call passes as literals, and its golden fit. */
typedef struct
{
    const char *name;
    const char *fit;
    cmsis_nn_tree_node_f32 *nodes;
    uint32_t *roots;
    float32_t *leaves; /* leaf rows, or leaf values */
    float32_t *row;
    int32_t node_count;
    int32_t leaf_count;
    int32_t tree_count;
    int32_t feature_count;
    int32_t class_count; /* 0 for the values form */
    ti_scores_form scores;
    ti_class_form class_only;
    ti_values_form values;
} ti_module;

static const ti_module ti_means[] = {
    {"ti_m1",
     "dt_iris",
     ti_m1_nodes,
     ti_m1_roots,
     ti_m1_leaf_rows,
     ti_m1_row,
     5,
     6,
     1,
     4,
     3,
     ti_m1_scores,
     ti_m1_class,
     NULL},
    {"ti_m5",
     "rf5_wine",
     ti_m5_nodes,
     ti_m5_roots,
     ti_m5_leaf_rows,
     ti_m5_row,
     49,
     54,
     5,
     13,
     3,
     ti_m5_scores,
     ti_m5_class,
     NULL},
    {"ti_m16",
     "ada16_wine",
     ti_m16_nodes,
     ti_m16_roots,
     ti_m16_leaf_rows,
     ti_m16_row,
     48,
     64,
     16,
     13,
     3,
     ti_m16_scores,
     ti_m16_class,
     NULL},
    {"ti_m100",
     "rf100_iris",
     ti_m100_nodes,
     ti_m100_roots,
     ti_m100_leaf_rows,
     ti_m100_row,
     589,
     689,
     100,
     4,
     3,
     ti_m100_scores,
     ti_m100_class,
     NULL},
};
#define TI_MEANS ((int)(sizeof(ti_means) / sizeof(ti_means[0])))

static const ti_module ti_values[] = {
    {"ti_v1",
     "dt_diabetes",
     ti_v1_nodes,
     ti_v1_roots,
     ti_v1_leaf_values,
     ti_v1_row,
     301,
     302,
     1,
     10,
     0,
     NULL,
     NULL,
     ti_v1_values},
    {"ti_v16",
     "rf16_friedman_nan",
     ti_v16_nodes,
     ti_v16_roots,
     ti_v16_leaf_values,
     ti_v16_row,
     240,
     256,
     16,
     7,
     0,
     NULL,
     NULL,
     ti_v16_values},
    {"ti_v100",
     "rf100_diabetes",
     ti_v100_nodes,
     ti_v100_roots,
     ti_v100_leaf_values,
     ti_v100_row,
     700,
     800,
     100,
     10,
     0,
     NULL,
     NULL,
     ti_v100_values},
};
#define TI_VALUES ((int)(sizeof(ti_values) / sizeof(ti_values[0])))

/* ---- Helpers ---- */

static char ti_message[256];
static unsigned long ti_cases;
static unsigned long ti_differences;
static unsigned long ti_statuses[3];
static int32_t ti_ref_leaves[TI_TREES];

static float32_t ti_float(uint32_t bits)
{
    float32_t value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t ti_bits(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static cmsis_nn_tree_node_f32 ti_node(uint32_t threshold, uint16_t feature, int16_t left, int16_t right)
{
    cmsis_nn_tree_node_f32 node;
    memset(&node, 0, sizeof(node));
    node.threshold = ti_float(threshold);
    node.feature = feature;
    node.left = left;
    node.right = right;
    return node;
}

static cmsis_nn_tree_table_f32 ti_table(const ti_module *m)
{
    cmsis_nn_tree_table_f32 table;
    memset(&table, 0, sizeof(table));
    table.nodes = m->nodes;
    table.roots = m->roots;
    table.node_count = m->node_count;
    table.leaf_count = m->leaf_count;
    table.tree_count = m->tree_count;
    table.feature_count = m->feature_count;
    return table;
}

static const tg_fit *ti_mean_fit(const ti_module *m)
{
    for (int f = 0; f < TG_FITS; ++f)
        if (strcmp(tg_fits[f].name, m->fit) == 0)
            return &tg_fits[f];
    TEST_FAIL_MESSAGE("no such golden fit");
    return NULL;
}

static const tvg_fit *ti_values_fit(const ti_module *m)
{
    for (int f = 0; f < TVG_FITS; ++f)
        if (strcmp(tvg_fits[f].name, m->fit) == 0)
            return &tvg_fits[f];
    TEST_FAIL_MESSAGE("no such golden fit");
    return NULL;
}

/* The fit's table into the module's arrays, its counts checked against the module's literals. */
static void ti_load_mean(const ti_module *m)
{
    const tg_fit *const fit = ti_mean_fit(m);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->node_count, fit->node_count, m->name);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->leaf_count, fit->leaf_count, m->name);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->tree_count, fit->tree_count, m->name);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->feature_count, fit->feature_count, m->name);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->class_count, fit->class_count, m->name);
    for (int32_t i = 0; i < fit->node_count; ++i)
    {
        const int32_t at = fit->node_start + i;
        m->nodes[i] = ti_node(tg_threshold[at], tg_feature[at], tg_left[at], tg_right[at]);
    }
    for (int32_t t = 0; t < fit->tree_count; ++t)
        m->roots[t] = tg_roots[fit->root_start + t];
    for (int32_t i = 0; i < fit->leaf_count * fit->class_count; ++i)
        m->leaves[i] = ti_float(tg_leaf_rows[fit->leaf_row_start + i]);
}

static void ti_load_values(const ti_module *m)
{
    const tvg_fit *const fit = ti_values_fit(m);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->node_count, fit->node_count, m->name);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->leaf_count, fit->leaf_count, m->name);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->tree_count, fit->tree_count, m->name);
    TEST_ASSERT_EQUAL_INT32_MESSAGE(m->feature_count, fit->feature_count, m->name);
    for (int32_t i = 0; i < fit->node_count; ++i)
    {
        const int32_t at = fit->node_start + i;
        m->nodes[i] = ti_node(tvg_threshold[at], tvg_feature[at], tvg_left[at], tvg_right[at]);
    }
    for (int32_t t = 0; t < fit->tree_count; ++t)
        m->roots[t] = tvg_roots[fit->root_start + t];
    for (int32_t l = 0; l < fit->leaf_count; ++l)
        m->leaves[l] = ti_float(tvg_leaf_values[fit->leaf_value_start + l]);
}

/* Golden row r of the module's fit into its row array. */
static int32_t ti_rows_of(const ti_module *m)
{
    return m->class_count > 0 ? ti_mean_fit(m)->row_count : ti_values_fit(m)->row_count;
}

static void ti_golden_row(const ti_module *m, int32_t r)
{
    if (m->class_count > 0)
    {
        const tg_fit *const fit = ti_mean_fit(m);
        for (int32_t j = 0; j < m->feature_count; ++j)
            m->row[j] = ti_float(tg_rows[fit->row_value_start + r * fit->feature_count + j]);
    }
    else
    {
        const tvg_fit *const fit = ti_values_fit(m);
        for (int32_t j = 0; j < m->feature_count; ++j)
            m->row[j] = ti_float(tvg_rows[fit->row_value_start + r * fit->feature_count + j]);
    }
}

/* ---- One case ---- */

typedef struct
{
    float32_t words[TI_GUARD + TI_TREES + TI_GUARD]; /* scores (TI_CLASSES of them) or values, between guards */
    int32_t index[3];
} ti_out;

static ti_out ti_entry_out;
static ti_out ti_module_out;
static ti_out ti_reference_out;

static void ti_fill(ti_out *out)
{
    for (size_t i = 0; i < sizeof(out->words) / sizeof(out->words[0]); ++i)
        out->words[i] = ti_float(TI_SCORE_SENTINEL);
    out->index[0] = out->index[1] = out->index[2] = TI_INDEX_SENTINEL;
}

static int ti_guards_intact(const ti_out *out, int32_t written)
{
    for (int32_t i = 0; i < TI_GUARD; ++i)
        if (ti_bits(out->words[i]) != TI_SCORE_SENTINEL ||
            ti_bits(out->words[TI_GUARD + written + i]) != TI_SCORE_SENTINEL)
            return 0;
    for (size_t i = TI_GUARD + (size_t)written + TI_GUARD; i < sizeof(out->words) / sizeof(out->words[0]); ++i)
        if (ti_bits(out->words[i]) != TI_SCORE_SENTINEL)
            return 0;
    return out->index[0] == TI_INDEX_SENTINEL && out->index[2] == TI_INDEX_SENTINEL;
}

static void ti_tally(const char *what, const ti_module *m, arm_cmsis_nn_status got, int differs)
{
    ++ti_cases;
    ti_statuses[got == ARM_CMSIS_NN_SUCCESS ? 0 : got == ARM_CMSIS_NN_ARG_ERROR ? 1 : 2] += 1;
    if (differs && ++ti_differences <= TI_REPORTED)
    {
        snprintf(ti_message, sizeof(ti_message), "difference: %s, %s, entry status %d", what, m->name, (int)got);
        TEST_MESSAGE(ti_message);
    }
}

/* The entry, both forms of the mean module and the reference on the module's row and tables. */
static arm_cmsis_nn_status ti_mean_case(const char *what, const ti_module *m)
{
    const cmsis_nn_tree_table_f32 table = ti_table(m);
    int32_t class_only = TI_INDEX_SENTINEL;
    ti_fill(&ti_entry_out);
    ti_fill(&ti_module_out);
    ti_fill(&ti_reference_out);
    const arm_cmsis_nn_status got = arm_nn_tree_walk_mean_argmax_f32(
        m->row, &table, m->leaves, m->class_count, &ti_entry_out.words[TI_GUARD], &ti_entry_out.index[1]);
    const arm_cmsis_nn_status module = m->scores(m->row, &ti_module_out.words[TI_GUARD], &ti_module_out.index[1]);
    const arm_cmsis_nn_status class_only_status = m->class_only(m->row, &class_only);
    const arm_cmsis_nn_status want = tr_reference(m->row,
                                                  &table,
                                                  m->leaves,
                                                  m->class_count,
                                                  &ti_reference_out.words[TI_GUARD],
                                                  &ti_reference_out.index[1],
                                                  ti_ref_leaves);
    int differs = got != module || got != class_only_status || got != want;
    differs |= memcmp(&ti_entry_out, &ti_module_out, sizeof(ti_out)) != 0;
    differs |= class_only != ti_entry_out.index[1];
    differs |= !ti_guards_intact(&ti_entry_out, m->class_count);
    if (got == ARM_CMSIS_NN_SUCCESS)
        differs |= memcmp(&ti_reference_out.words[TI_GUARD],
                          &ti_entry_out.words[TI_GUARD],
                          (size_t)m->class_count * sizeof(float32_t)) != 0 ||
            ti_reference_out.index[1] != ti_entry_out.index[1];
    else
        differs |= ti_entry_out.index[1] != TI_INDEX_SENTINEL;
    ti_tally(what, m, got, differs);
    return got;
}

/* The entry, the values module and the reference on the module's row and tables: every word, prefix included. */
static arm_cmsis_nn_status ti_values_case(const char *what, const ti_module *m)
{
    const cmsis_nn_tree_table_f32 table = ti_table(m);
    ti_fill(&ti_entry_out);
    ti_fill(&ti_module_out);
    ti_fill(&ti_reference_out);
    const arm_cmsis_nn_status got =
        arm_nn_tree_walk_values_f32(m->row, &table, m->leaves, m->leaf_count, &ti_entry_out.words[TI_GUARD]);
    const arm_cmsis_nn_status module = m->values(m->row, &ti_module_out.words[TI_GUARD]);
    const arm_cmsis_nn_status want =
        tvr_reference(m->row, &table, m->leaves, m->leaf_count, &ti_reference_out.words[TI_GUARD]);
    int differs = got != module || got != want;
    differs |= memcmp(&ti_entry_out, &ti_module_out, sizeof(ti_out)) != 0;
    differs |= memcmp(&ti_entry_out, &ti_reference_out, sizeof(ti_out)) != 0;
    differs |= !ti_guards_intact(&ti_entry_out, m->tree_count);
    ti_tally(what, m, got, differs);
    return got;
}

static arm_cmsis_nn_status ti_case(const char *what, const ti_module *m)
{
    return m->class_count > 0 ? ti_mean_case(what, m) : ti_values_case(what, m);
}

static void ti_load(const ti_module *m)
{
    if (m->class_count > 0)
        ti_load_mean(m);
    else
        ti_load_values(m);
}

static unsigned long ti_cases_before;
static unsigned long ti_differences_before;
static unsigned long ti_statuses_before[3];

static void ti_start(void)
{
    ti_cases_before = ti_cases;
    ti_differences_before = ti_differences;
    memcpy(ti_statuses_before, ti_statuses, sizeof(ti_statuses));
}

static void ti_report(const char *what)
{
    snprintf(ti_message,
             sizeof(ti_message),
             "inline %s: %lu cases, %lu differences (%lu success, %lu argument error, %lu failure)",
             what,
             ti_cases - ti_cases_before,
             ti_differences - ti_differences_before,
             ti_statuses[0] - ti_statuses_before[0],
             ti_statuses[1] - ti_statuses_before[1],
             ti_statuses[2] - ti_statuses_before[2]);
    TEST_MESSAGE(ti_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(ti_differences - ti_differences_before));
    TEST_ASSERT_TRUE(ti_cases > ti_cases_before);
}

static const ti_module *ti_module_at(int i) { return i < TI_MEANS ? &ti_means[i] : &ti_values[i - TI_MEANS]; }
#define TI_MODULES (TI_MEANS + TI_VALUES)

/* ---- The cases ---- */

static void ti_capability(void)
{
    TEST_ASSERT_EQUAL_INT(1, ARM_NN_HAS_TREE_WALK_INLINE_BODIES_F32);
    TEST_ASSERT_EQUAL_INT(1, ARM_NN_HAS_TREE_WALK_MEAN_ARGMAX_F32);
    TEST_ASSERT_EQUAL_INT(1, ARM_NN_HAS_TREE_WALK_VALUES_F32);
    /* The literal of ti_m100_too_many_classes is one past the bound at 689 leaves. */
    TEST_ASSERT_EQUAL_INT32(779203, INT32_MAX / (4 * 689));
}

/* Every golden row of each module's fit. */
static void ti_golden(void)
{
    ti_start();
    for (int i = 0; i < TI_MODULES; ++i)
    {
        const ti_module *const m = ti_module_at(i);
        ti_load(m);
        for (int32_t r = 0; r < ti_rows_of(m); ++r)
        {
            ti_golden_row(m, r);
            TEST_ASSERT_EQUAL_INT_MESSAGE(ARM_CMSIS_NN_SUCCESS, ti_case("golden", m), m->name);
        }
    }
    ti_report("golden rows");
}

/* A NaN of each kind in each feature of the first golden rows, and a row of NaNs, with every split's NaN route set
 * left and then right; then the fit's own routes. */
static void ti_nan_routes(void)
{
    static const uint32_t nans[] = {
        UINT32_C(0x7fc00000), UINT32_C(0xffc00000), UINT32_C(0x7f800001), UINT32_C(0xff800001), UINT32_C(0x7fc12345)};
    static uint16_t features[TI_MOST_NODES];
    unsigned long left_successes = 0;
    ti_start();
    for (int i = 0; i < TI_MODULES; ++i)
    {
        const ti_module *const m = ti_module_at(i);
        ti_load(m);
        for (int32_t n = 0; n < m->node_count; ++n)
            features[n] = m->nodes[n].feature;
        for (int route = 0; route < 3; ++route)
        {
            for (int32_t n = 0; n < m->node_count; ++n)
                m->nodes[n].feature = route == 0 ? (uint16_t)(features[n] | TI_MISSING_LEFT)
                    : route == 1                 ? (uint16_t)(features[n] & (uint16_t)~TI_MISSING_LEFT)
                                                 : features[n];
            const int32_t rows = ti_rows_of(m) < 8 ? ti_rows_of(m) : 8;
            for (int32_t r = 0; r < rows; ++r)
                for (int32_t f = 0; f < m->feature_count; ++f)
                    for (unsigned k = 0; k < sizeof(nans) / sizeof(nans[0]); ++k)
                    {
                        ti_golden_row(m, r);
                        m->row[f] = ti_float(nans[k]);
                        const arm_cmsis_nn_status status = ti_case("NaN route", m);
                        TEST_ASSERT_EQUAL_INT_MESSAGE(ARM_CMSIS_NN_SUCCESS, status, m->name);
                        left_successes += route == 0;
                    }
            for (int32_t f = 0; f < m->feature_count; ++f)
                m->row[f] = ti_float(nans[f % 5]);
            TEST_ASSERT_EQUAL_INT_MESSAGE(ARM_CMSIS_NN_SUCCESS, ti_case("NaN row", m), m->name);
        }
        for (int32_t n = 0; n < m->node_count; ++n)
            m->nodes[n].feature = features[n];
    }
    TEST_ASSERT_TRUE(left_successes > 0);
    ti_report("NaN routes, all left, all right and the fit's");
}

/* Leaf rows that tie classes a and b at the top in every leaf, so the two means are one value and the first of the
 * two wins; the third class is lower. */
static void ti_ties(void)
{
    static const int32_t pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    unsigned long ties = 0;
    ti_start();
    for (int i = 0; i < TI_MEANS; ++i)
    {
        const ti_module *const m = &ti_means[i];
        ti_load(m);
        for (int p = 0; p < 3; ++p)
        {
            const int32_t a = pairs[p][0];
            const int32_t b = pairs[p][1];
            const int32_t other = 3 - a - b;
            for (int32_t l = 0; l < m->leaf_count; ++l)
            {
                float32_t *const row = &m->leaves[l * 3];
                row[a] = row[b] = 0.5f + 0.0625f * (float32_t)(l % 7);
                row[other] = 0.125f * (float32_t)(l % 3);
            }
            for (int32_t r = 0; r < ti_rows_of(m); ++r)
            {
                ti_golden_row(m, r);
                TEST_ASSERT_EQUAL_INT_MESSAGE(ARM_CMSIS_NN_SUCCESS, ti_case("ties", m), m->name);
                TEST_ASSERT_EQUAL_HEX32(ti_bits(ti_entry_out.words[TI_GUARD + a]),
                                        ti_bits(ti_entry_out.words[TI_GUARD + b]));
                TEST_ASSERT_EQUAL_INT32_MESSAGE(a, ti_module_out.index[1], m->name);
                ++ties;
            }
        }
    }
    snprintf(ti_message, sizeof(ti_message), "inline ties: %lu rows with two classes equal at the top", ties);
    TEST_MESSAGE(ti_message);
    ti_report("ties");
}

static float32_t ti_add(float32_t a, float32_t b)
{
    volatile float32_t x = a;
    volatile float32_t y = b;
    return x + y;
}

static float32_t ti_multiply(float32_t a, float32_t b)
{
    volatile float32_t x = a;
    volatile float32_t y = b;
    return x * y;
}

static float32_t ti_divide(float32_t a, float32_t b)
{
    volatile float32_t x = a;
    volatile float32_t y = b;
    return x / y;
}

/* Every leaf row the same, (v0, v1, v2), each v the first candidate for which the sum of tree_count copies in tree
 * order times float32(1 / T) differs from that sum divided by T: every row's means are known, and the multiply is told
 * from a divide in every class. Where T is a power of two the two are one value and no row can tell them apart; the
 * case still runs. */
static void ti_reciprocal(void)
{
    unsigned long distinguished = 0;
    ti_start();
    for (int i = 0; i < TI_MEANS; ++i)
    {
        const ti_module *const m = &ti_means[i];
        const float32_t count = (float32_t)m->tree_count;
        const float32_t scale = ti_divide(1.0f, count);
        const int power_of_two = (m->tree_count & (m->tree_count - 1)) == 0;
        float32_t v[3];
        float32_t mean[3];
        ti_load(m);
        for (int32_t c = 0; c < 3; ++c)
        {
            int found = 0;
            for (int32_t k = 1 + 97 * c; k < 4096 && !found; ++k)
            {
                /* A value in [0.5, 1) with a mantissa spread over all its bits: sums of a few such words are rarely
                 * exact quotients, so their two roundings tell the forms apart. */
                const float32_t value =
                    ti_float(UINT32_C(0x3f000000) + ((uint32_t)k * UINT32_C(2654435)) % UINT32_C(0x800000));
                float32_t sum = value;
                for (int32_t t = 1; t < m->tree_count; ++t)
                    sum = ti_add(sum, value);
                v[c] = value;
                mean[c] = ti_multiply(sum, scale);
                found = power_of_two || ti_bits(mean[c]) != ti_bits(ti_divide(sum, count));
            }
            TEST_ASSERT_TRUE_MESSAGE(found, m->name);
            distinguished += !power_of_two;
        }
        for (int32_t l = 0; l < m->leaf_count; ++l)
            for (int32_t c = 0; c < 3; ++c)
                m->leaves[l * 3 + c] = v[c];
        for (int32_t r = 0; r < ti_rows_of(m); ++r)
        {
            ti_golden_row(m, r);
            TEST_ASSERT_EQUAL_INT_MESSAGE(ARM_CMSIS_NN_SUCCESS, ti_case("reciprocal", m), m->name);
            for (int32_t c = 0; c < 3; ++c)
                TEST_ASSERT_EQUAL_HEX32_MESSAGE(ti_bits(mean[c]), ti_bits(ti_module_out.words[TI_GUARD + c]), m->name);
        }
    }
    snprintf(ti_message,
             sizeof(ti_message),
             "inline reciprocal: %lu (module, class) pairs where the multiply differs from a divide",
             distinguished);
    TEST_MESSAGE(ti_message);
    TEST_ASSERT_TRUE(distinguished >= 6); /* ti_m5 and ti_m100, three classes each */
    ti_report("reciprocal");
}

/* The first golden rows of the module against its tables with one thing changed. */
static void ti_rows_against(const char *what, const ti_module *m)
{
    const int32_t rows = ti_rows_of(m) < 24 ? ti_rows_of(m) : 24;
    for (int32_t r = 0; r < rows; ++r)
    {
        ti_golden_row(m, r);
        (void)ti_case(what, m);
    }
}

/* The malformed tables of the entry's refusals, through the module and through the entry: a root out of range at the
 * first, a middle and the last tree; at the root of the first and of the last tree and at a middle split, a feature,
 * a child and a leaf out of range on either side and a split that names itself; and a child that names its parent. */
static void ti_malformed(void)
{
    static const uint32_t bad_roots[] = {0u, UINT32_C(0x80000000), UINT32_MAX};
    ti_start();
    for (int i = 0; i < TI_MODULES; ++i)
    {
        const ti_module *const m = ti_module_at(i);
        ti_load(m);
        for (unsigned b = 0; b < sizeof(bad_roots) / sizeof(bad_roots[0]); ++b)
        {
            const int32_t at[3] = {0, m->tree_count / 2, m->tree_count - 1};
            for (int k = 0; k < 3; ++k)
            {
                const uint32_t saved = m->roots[at[k]];
                m->roots[at[k]] = bad_roots[b] == 0u ? (uint32_t)m->node_count : bad_roots[b];
                ti_rows_against("root", m);
                m->roots[at[k]] = saved;
            }
        }
        for (int site = 0; site < 3; ++site)
        {
            const int32_t index = site == 0 ? (int32_t)m->roots[0]
                : site == 1                 ? (int32_t)m->roots[m->tree_count - 1]
                                            : m->node_count / 2;
            const cmsis_nn_tree_node_f32 saved = m->nodes[index];
            const uint16_t features[] = {(uint16_t)m->feature_count,
                                         (uint16_t)(TI_MISSING_LEFT | (uint16_t)m->feature_count),
                                         UINT16_C(0x7fff),
                                         UINT16_C(0xffff)};
            const int16_t children[] = {
                (int16_t)m->node_count, INT16_MAX, (int16_t)(-m->leaf_count - 1), INT16_MIN, (int16_t)index};
            for (unsigned f = 0; f < sizeof(features) / sizeof(features[0]); ++f)
            {
                m->nodes[index].feature = features[f];
                ti_rows_against("feature", m);
                m->nodes[index] = saved;
            }
            for (unsigned c = 0; c < sizeof(children) / sizeof(children[0]); ++c)
                for (int side = 0; side < 2; ++side)
                {
                    if (side == 0)
                        m->nodes[index].left = children[c];
                    else
                        m->nodes[index].right = children[c];
                    ti_rows_against(c == 4 ? "cycle" : "child or leaf", m);
                    m->nodes[index] = saved;
                }
            if (m->nodes[index].left >= 0)
            {
                const int32_t child = m->nodes[index].left;
                const cmsis_nn_tree_node_f32 child_saved = m->nodes[child];
                m->nodes[child].left = (int16_t)index;
                m->nodes[child].right = (int16_t)index;
                ti_rows_against("cycle through a child", m);
                m->nodes[child] = child_saved;
            }
        }
    }
    TEST_ASSERT_TRUE(ti_statuses[1] > ti_statuses_before[1]);
    TEST_ASSERT_TRUE(ti_statuses[2] > ti_statuses_before[2]);
    ti_report("malformed tables");
}

/* The body's own count checks with the literal out of range: refused before any read of the tables or write. */
static void ti_count_checks(void)
{
    const ti_module *const m = &ti_means[3];
    const ti_module *const v = &ti_values[2];
    ti_load(m);
    ti_load(v);
    ti_golden_row(m, 0);
    ti_golden_row(v, 0);
    for (int32_t l = 0; l < 799; ++l)
        ti_v100_short_values[l] = v->leaves[l];

    ti_fill(&ti_module_out);
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          ti_m100_no_classes(m->row, &ti_module_out.words[TI_GUARD], &ti_module_out.index[1]));
    TEST_ASSERT_TRUE(ti_guards_intact(&ti_module_out, 0));
    TEST_ASSERT_EQUAL_INT32(TI_INDEX_SENTINEL, ti_module_out.index[1]);
    TEST_ASSERT_EQUAL_HEX32(TI_SCORE_SENTINEL, ti_bits(ti_module_out.words[TI_GUARD]));
    TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR,
                          ti_m100_too_many_classes(m->row, &ti_module_out.words[TI_GUARD], &ti_module_out.index[1]));
    TEST_ASSERT_TRUE(ti_guards_intact(&ti_module_out, 0));
    TEST_ASSERT_EQUAL_INT32(TI_INDEX_SENTINEL, ti_module_out.index[1]);
    TEST_ASSERT_EQUAL_HEX32(TI_SCORE_SENTINEL, ti_bits(ti_module_out.words[TI_GUARD]));

    /* One leaf value short of the leaf count: refused for every row, whichever leaves its trees reach. */
    for (int32_t r = 0; r < ti_rows_of(v); ++r)
    {
        ti_golden_row(v, r);
        ti_fill(&ti_module_out);
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_ARG_ERROR, ti_v100_short(v->row, &ti_module_out.words[TI_GUARD]));
        TEST_ASSERT_TRUE(ti_guards_intact(&ti_module_out, 0));
        TEST_ASSERT_EQUAL_HEX32(TI_SCORE_SENTINEL, ti_bits(ti_module_out.words[TI_GUARD]));
    }
    /* The same table through the entry, with leaf_value_count 799: refused the same way. */
    {
        const cmsis_nn_tree_table_f32 table = ti_table(v);
        ti_fill(&ti_entry_out);
        TEST_ASSERT_EQUAL_INT(
            ARM_CMSIS_NN_ARG_ERROR,
            arm_nn_tree_walk_values_f32(v->row, &table, ti_v100_short_values, 799, &ti_entry_out.words[TI_GUARD]));
        TEST_ASSERT_TRUE(ti_guards_intact(&ti_entry_out, 0));
        TEST_ASSERT_EQUAL_HEX32(TI_SCORE_SENTINEL, ti_bits(ti_entry_out.words[TI_GUARD]));
    }
    snprintf(ti_message,
             sizeof(ti_message),
             "inline count checks: class_count 0 and 779204 at 689 leaves, leaf_value_count 799 at 800 leaves over %ld "
             "rows: refused, nothing written",
             (long)ti_rows_of(v));
    TEST_MESSAGE(ti_message);
}
