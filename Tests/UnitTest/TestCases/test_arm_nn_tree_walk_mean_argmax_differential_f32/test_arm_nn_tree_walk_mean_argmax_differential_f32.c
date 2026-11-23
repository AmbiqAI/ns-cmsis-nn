/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* arm_nn_tree_walk_mean_argmax_f32 against the body it had at 61293a84 (tree_mean_argmax_previous.h), on every vector
 * of its own suite and on further randomised tables.
 *
 * The entry's suite is compiled here unchanged, with every call it makes to the entry routed through tmd_entry. That
 * call runs the previous body and then the entry on the same arguments, each from the same contents of scores and
 * class_index, and hands the entry's answer back to the suite, which then checks it as it always does. A call differs
 * when the two statuses differ, when any word of scores either body may write differs (a NaN by its bits, as every
 * other word), or when class_index differs; nothing is tolerated on any build. The words compared are class_count
 * where the arguments pass the entry's argument checks, and none where they do not (no body writes then). One case
 * of the suite passes a class_count of INT32_MAX / 8 that the checks accept and whose first tree cycles, so neither
 * body writes, with a scores buffer shorter than that: a call whose class_count passes TMD_SHADOW is compared on its
 * status and class_index alone, and counted apart.
 *
 * The randomised sweep, tmd_random, builds tables with the suite's generator from seeds of its own, at tree counts
 * and class counts the suite's sweep does not use, and holds the two bodies to each other the same way. */

#include <arm_nnfunctions.h>
#include <stdio.h>
#include <string.h>
#include <unity.h>

#include "tree_mean_argmax_previous.h"

#if defined(USING_FVP_CORSTONE_300)
    #define TMD_LARGE_SECTION __attribute__((section(".bss.NoInit")))
#else
    #define TMD_LARGE_SECTION
#endif

/* The most scores a call of the suite writes: 16383, the class_count bound at the most leaves. */
#define TMD_SHADOW 16384
#define TMD_REPORTED 8

static float32_t tmd_before[TMD_SHADOW] TMD_LARGE_SECTION;
static float32_t tmd_after[TMD_SHADOW] TMD_LARGE_SECTION;
static unsigned long tmd_calls;
static unsigned long tmd_differences;
static unsigned long tmd_words;
static unsigned long tmd_beyond_shadow;
static unsigned long tmd_answers[3]; /* the entry's SUCCESS, ARG_ERROR and FAILURE */
static char tmd_message[256];

/* Whether the entry's argument checks, as the declaration states them, accept the call: only then may a body write. */
static int tmd_accepted(const float32_t *input,
                        const cmsis_nn_tree_table_f32 *table,
                        const float32_t *leaf_rows,
                        int32_t class_count,
                        const float32_t *scores,
                        const int32_t *class_index)
{
    return input != NULL && leaf_rows != NULL && scores != NULL && class_index != NULL &&
        tpm_table_check(table) == ARM_CMSIS_NN_SUCCESS && class_count >= 1 &&
        class_count <= INT32_MAX / (4 * table->leaf_count);
}

static arm_cmsis_nn_status tmd_entry(const float32_t *input,
                                     const cmsis_nn_tree_table_f32 *table,
                                     const float32_t *leaf_rows,
                                     int32_t class_count,
                                     float32_t *scores,
                                     int32_t *class_index)
{
    int32_t words = tmd_accepted(input, table, leaf_rows, class_count, scores, class_index) ? class_count : 0;
    if (words > TMD_SHADOW)
    {
        /* The caller's scores may be shorter than class_count here (the suite's cycle at INT32_MAX / 8 classes): no
         * word of it is read, and the call is compared on its status and class_index. */
        words = 0;
        ++tmd_beyond_shadow;
    }
    const int32_t index_before = class_index != NULL ? *class_index : 0;
    if (words > 0)
        memcpy(tmd_before, scores, (size_t)words * sizeof(float32_t));

    /* The previous body first, into the caller's buffers; its answer is kept and the buffers restored. */
    const arm_cmsis_nn_status want = tpm_mean_argmax_f32(input, table, leaf_rows, class_count, scores, class_index);
    const int32_t index_want = class_index != NULL ? *class_index : 0;
    if (words > 0)
    {
        memcpy(tmd_after, scores, (size_t)words * sizeof(float32_t));
        memcpy(scores, tmd_before, (size_t)words * sizeof(float32_t));
    }
    if (class_index != NULL)
        *class_index = index_before;

    /* Then the entry, from the same state; its answer is the one the suite checks. */
    const arm_cmsis_nn_status got =
        arm_nn_tree_walk_mean_argmax_f32(input, table, leaf_rows, class_count, scores, class_index);
    int32_t first = -1;
    for (int32_t c = 0; c < words && first < 0; ++c)
        if (memcmp(&scores[c], &tmd_after[c], sizeof(float32_t)) != 0)
            first = c;
    const int differs = got != want || first >= 0 || (class_index != NULL && *class_index != index_want);
    ++tmd_calls;
    tmd_words += (unsigned long)words;
    tmd_answers[got == ARM_CMSIS_NN_SUCCESS ? 0 : got == ARM_CMSIS_NN_ARG_ERROR ? 1 : 2] += 1;
    if (differs && ++tmd_differences <= TMD_REPORTED)
    {
        uint32_t g = 0;
        uint32_t w = 0;
        if (first >= 0)
        {
            memcpy(&g, &scores[first], sizeof(g));
            memcpy(&w, &tmd_after[first], sizeof(w));
        }
        snprintf(tmd_message,
                 sizeof(tmd_message),
                 "difference: %ld classes, status %d against %d, class %ld against %ld, score %ld 0x%08lx against "
                 "0x%08lx",
                 (long)class_count,
                 (int)got,
                 (int)want,
                 class_index != NULL ? (long)*class_index : -1L,
                 (long)index_want,
                 (long)first,
                 (unsigned long)g,
                 (unsigned long)w);
        TEST_MESSAGE(tmd_message);
    }
    return got;
}

/* The entry's suite, every call to the entry routed through tmd_entry. */
#define arm_nn_tree_walk_mean_argmax_f32 tmd_entry
#include "../test_arm_nn_tree_walk_mean_argmax_f32/test_arm_nn_tree_walk_mean_argmax_f32.c"
#undef arm_nn_tree_walk_mean_argmax_f32

static unsigned long tmd_calls_before;
static unsigned long tmd_differences_before;
static unsigned long tmd_words_before;

static void tmd_start(void)
{
    tmd_calls_before = tmd_calls;
    tmd_differences_before = tmd_differences;
    tmd_words_before = tmd_words;
}

/* One test's differential tally, which must hold no difference. */
static void tmd_check(const char *what)
{
    snprintf(tmd_message,
             sizeof(tmd_message),
             "differential %s: %lu calls, %lu score words compared, %lu differences (all calls so far: %lu success, "
             "%lu argument error, %lu failure; %lu compared on status and class only)",
             what,
             tmd_calls - tmd_calls_before,
             tmd_words - tmd_words_before,
             tmd_differences - tmd_differences_before,
             tmd_answers[0],
             tmd_answers[1],
             tmd_answers[2],
             tmd_beyond_shadow);
    TEST_MESSAGE(tmd_message);
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)(tmd_differences - tmd_differences_before));
    TEST_ASSERT_TRUE(tmd_calls > tmd_calls_before);
}

static void tmd_suite_capability(void)
{
    tmd_start();
    tm_capability();
    /* The capability case calls no entry: one call here, so the tally is not empty. */
    {
        const cmsis_nn_tree_node_f32 node = tm_node(0.0f, 0u, -1, -1);
        const uint32_t root = 0;
        const cmsis_nn_tree_table_f32 table = tm_small_table(&node, &root, 1, 1, 1, 1);
        const float32_t x = 0.0f;
        const float32_t row[1] = {0.5f};
        float32_t score;
        int32_t index = TM_INDEX_SENTINEL;
        TEST_ASSERT_EQUAL_INT(ARM_CMSIS_NN_SUCCESS, tmd_entry(&x, &table, row, 1, &score, &index));
    }
    tmd_check("capability");
}

static void tmd_suite_class_rules(void)
{
    tmd_start();
    tm_class_rules();
    tmd_check("class rules");
}

static void tmd_suite_precedence(void)
{
    tmd_start();
    tm_precedence();
    tmd_check("precedence");
}

static void tmd_suite_sweep(void)
{
    tmd_start();
    tm_sweep();
    tmd_check("sweep");
}

static void tmd_suite_refusals(void)
{
    tmd_start();
    tm_refusals();
    tmd_check("refusals");
}

static void tmd_suite_limits(void)
{
    tmd_start();
    tm_limits();
    tmd_check("limits");
}

static void tmd_suite_golden(void)
{
    tmd_start();
    tm_golden();
    tmd_check("golden");
}

/* Randomised tables from seeds of their own: every shape at T in {4, 7, 64, 100} and F in {2, 7}, every class count
 * from 1 to TM_CLASSES, the leaf-row kind drawn per table and class count. */
static void tmd_random(void)
{
    static const int32_t trees[] = {4, 7, 64, 100};
    static const int32_t features[] = {2, 7};
    uint32_t state = UINT32_C(0x9e3779b9);
    tmd_start();
    for (unsigned ti = 0; ti < sizeof(trees) / sizeof(trees[0]); ++ti)
        for (unsigned fi = 0; fi < sizeof(features) / sizeof(features[0]); ++fi)
            for (int shape_kind = 0; shape_kind < TM_SHAPE_KINDS; ++shape_kind)
            {
                const int32_t depth = shape_kind == TM_CHAIN ? 1 + (int32_t)(tm_next(&state) % 21u) : 21;
                const int missing = (int)(tm_next(&state) % 3u);
                const tm_shape shape = tm_build(shape_kind, trees[ti], features[fi], depth, missing, &state);
                const cmsis_nn_tree_table_f32 table = tm_table(&shape);
                tm_make_rows(shape.feature_count, &state);
                for (int32_t classes = 1; classes <= TM_CLASSES; ++classes)
                {
                    const int kind = (int)(tm_next(&state) % (uint32_t)TM_LEAF_KINDS);
                    tm_make_leaf_rows(kind, shape.leaf_count, classes, &state);
                    for (int32_t r = 0; r < tm_row_count; ++r)
                    {
                        tm_fill(&tm_got);
                        (void)tmd_entry(
                            tm_rows[r], &table, tm_leaf_rows, classes, &tm_got.scores[TM_GUARD], &tm_got.index[1]);
                        TEST_ASSERT_TRUE(tm_guards_intact(&tm_got, classes));
                    }
                }
            }
    tmd_check("randomised tables");
}
