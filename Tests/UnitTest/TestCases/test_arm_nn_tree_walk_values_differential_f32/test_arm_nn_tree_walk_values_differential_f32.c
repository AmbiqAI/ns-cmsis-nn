/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_nn_tree_walk_values_f32 against the body it had at 61293a84 (tree_walk_values_previous.h), on every vector of
 * its own suite and on further randomised tables.
 *
 * The entry's suite is compiled here unchanged, with every call it makes to the entry routed through tvd_entry. That
 * call runs the previous body and then the entry on the same arguments, each from the same contents of values, and
 * hands the entry's answer back to the suite, which then checks it as it always does. A call differs when the two
 * statuses differ or when any word of values either body may write differs; nothing is tolerated on any build. The
 * words compared are tree_count where the arguments pass the entry's argument checks, and none where they do not (no
 * body writes then).
 *
 * The randomised sweep, tvd_random, builds tables with the suite's generator from seeds of its own, at tree counts the
 * suite's sweep does not use, and holds the two bodies to each other the same way. */

#include <arm_nnfunctions.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "tree_walk_values_previous.h"

#if defined(USING_FVP_CORSTONE_300)
    #define TVD_LARGE_SECTION __attribute__((section(".bss.NoInit")))
#else
    #define TVD_LARGE_SECTION
#endif

/* The most values a call writes: tree_count is at most 65535. */
#define TVD_SHADOW 65535
#define TVD_REPORTED 8

static float32_t tvd_before[TVD_SHADOW] TVD_LARGE_SECTION;
static float32_t tvd_after[TVD_SHADOW] TVD_LARGE_SECTION;
static unsigned long tvd_calls;
static unsigned long tvd_differences;
static unsigned long tvd_words;
static unsigned long tvd_answers[3]; /* the entry's SUCCESS, ARG_ERROR and FAILURE */
static char tvd_message[256];

/* Whether the entry's argument checks, as the declaration states them, accept the call: only then may a body write. */
static int tvd_accepted(const float32_t *input,
                        const cmsis_nn_tree_table_f32 *table,
                        const float32_t *leaf_values,
                        int32_t leaf_value_count,
                        const float32_t *values)
{
    return input != NULL && leaf_values != NULL && values != NULL && tpv_table_check(table) == ARM_CMSIS_NN_SUCCESS &&
        leaf_value_count >= table->leaf_count;
}

static arm_cmsis_nn_status tvd_entry(const float32_t *input,
                                     const cmsis_nn_tree_table_f32 *table,
                                     const float32_t *leaf_values,
                                     int32_t leaf_value_count,
                                     float32_t *values)
{
    const int32_t words = tvd_accepted(input, table, leaf_values, leaf_value_count, values) ? table->tree_count : 0;
    if (words > 0)
        memcpy(tvd_before, values, (size_t)words * sizeof(float32_t));

    /* The previous body first, into the caller's buffer; its answer is kept and the buffer restored. */
    const arm_cmsis_nn_status want = tpv_values_f32(input, table, leaf_values, leaf_value_count, values);
    if (words > 0)
    {
        memcpy(tvd_after, values, (size_t)words * sizeof(float32_t));
        memcpy(values, tvd_before, (size_t)words * sizeof(float32_t));
    }

    /* Then the entry, from the same state; its answer is the one the suite checks. */
    const arm_cmsis_nn_status got = arm_nn_tree_walk_values_f32(input, table, leaf_values, leaf_value_count, values);
    int32_t first = -1;
    for (int32_t t = 0; t < words && first < 0; ++t)
        if (memcmp(&values[t], &tvd_after[t], sizeof(float32_t)) != 0)
            first = t;
    const int differs = got != want || first >= 0;
    ++tvd_calls;
    tvd_words += (unsigned long)words;
    tvd_answers[got == ARM_CMSIS_NN_SUCCESS ? 0 : got == ARM_CMSIS_NN_ARG_ERROR ? 1 : 2] += 1;
    if (differs && ++tvd_differences <= TVD_REPORTED)
    {
        uint32_t g = 0;
        uint32_t w = 0;
        if (first >= 0)
        {
            memcpy(&g, &values[first], sizeof(g));
            memcpy(&w, &tvd_after[first], sizeof(w));
        }
        snprintf(tvd_message,
                 sizeof(tvd_message),
                 "difference: %ld trees, status %d against %d, value %ld 0x%08lx against 0x%08lx",
                 (long)words,
                 (int)got,
                 (int)want,
                 (long)first,
                 (unsigned long)g,
                 (unsigned long)w);
        TEST_MESSAGE(tvd_message);
    }
    return got;
}

/* The entry's suite, every call to the entry routed through tvd_entry. */
#define arm_nn_tree_walk_values_f32 tvd_entry
#include "../test_arm_nn_tree_walk_values_f32/test_arm_nn_tree_walk_values_f32.c"
#undef arm_nn_tree_walk_values_f32

static unsigned long tvd_calls_before;
static unsigned long tvd_differences_before;
static unsigned long tvd_words_before;

static void tvd_start(void)
{
    tvd_calls_before = tvd_calls;
    tvd_differences_before = tvd_differences;
    tvd_words_before = tvd_words;
}

/* One test's differential tally, which must hold no difference. */
static void tvd_check(const char *what)
{
    snprintf(tvd_message,
             sizeof(tvd_message),
             "differential %s: %lu calls, %lu value words compared, %lu differences (all calls so far: %lu success, "
             "%lu argument error, %lu failure)",
             what,
             tvd_calls - tvd_calls_before,
             tvd_words - tvd_words_before,
             tvd_differences - tvd_differences_before,
             tvd_answers[0],
             tvd_answers[1],
             tvd_answers[2]);
    TEST_MESSAGE(tvd_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(tvd_differences - tvd_differences_before));
    TEST_ASSERT_TRUE(tvd_calls > tvd_calls_before);
}

static void tvd_suite_capability(void)
{
    tvd_start();
    tv_capability();
    /* The capability case calls no entry: one call here, so the tally is not empty. */
    {
        const cmsis_nn_tree_node_f32 node = tv_node(0.0f, 0u, -1, -1);
        const uint32_t root = 0;
        const cmsis_nn_tree_table_f32 table = tv_small_table(&node, &root, 1, 1, 1, 1);
        const float32_t x = 0.0f;
        const float32_t leaf_values[1] = {0.5f};
        float32_t value;
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, tvd_entry(&x, &table, leaf_values, 1, &value));
    }
    tvd_check("capability");
}

static void tvd_suite_words(void)
{
    tvd_start();
    tv_words_pass_through();
    tvd_check("words");
}

static void tvd_suite_precedence(void)
{
    tvd_start();
    tv_precedence();
    tvd_check("precedence");
}

static void tvd_suite_sweep(void)
{
    tvd_start();
    tv_sweep();
    tvd_check("sweep");
}

static void tvd_suite_refusals(void)
{
    tvd_start();
    tv_refusals();
    tvd_check("refusals");
}

static void tvd_suite_limits(void)
{
    tvd_start();
    tv_limits();
    tvd_check("limits");
}

static void tvd_suite_golden(void)
{
    tvd_start();
    tv_golden();
    tvd_check("golden");
}

/* Randomised tables from seeds of their own: every shape at T in {4, 7, 64, 100} and F in {2, 7}, each leaf-value
 * kind, at leaf_value_count = leaf_count and above it. */
static void tvd_random(void)
{
    static const int32_t trees[] = {4, 7, 64, 100};
    static const int32_t features[] = {2, 7};
    uint32_t state = UINT32_C(0x9e3779b9);
    tvd_start();
    for (unsigned ti = 0; ti < sizeof(trees) / sizeof(trees[0]); ++ti)
        for (unsigned fi = 0; fi < sizeof(features) / sizeof(features[0]); ++fi)
            for (int shape_kind = 0; shape_kind < TV_SHAPE_KINDS; ++shape_kind)
            {
                const int32_t depth = shape_kind == TV_CHAIN ? 1 + (int32_t)(tv_next(&state) % 21u) : 21;
                const int missing = (int)(tv_next(&state) % 3u);
                const tv_shape shape = tv_build(shape_kind, trees[ti], features[fi], depth, missing, &state);
                const cmsis_nn_tree_table_f32 table = tv_table(&shape);
                tv_make_rows(shape.feature_count, &state);
                for (int kind = 0; kind < TV_LEAF_KINDS; ++kind)
                {
                    tv_make_leaf_values(kind, shape.leaf_count, &state);
                    for (int32_t r = 0; r < tv_row_count; ++r)
                        for (int32_t extra = 0; extra <= TV_EXTRA_VALUES; extra += TV_EXTRA_VALUES)
                        {
                            tv_fill(&tv_got);
                            (void)tvd_entry(
                                tv_rows[r], &table, tv_leaf_values, shape.leaf_count + extra, &tv_got.values[TV_GUARD]);
                            TEST_ASSERT_TRUE(tv_guards_intact(&tv_got, shape.tree_count));
                        }
                }
            }
    tvd_check("randomised tables");
}
