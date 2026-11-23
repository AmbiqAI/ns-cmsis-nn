/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The portable reference of arm_nn_tree_walk_mean_argmax_f32: what its declaration states, in plain C, with no
 * intrinsics, no library header beyond the types and no code shared with the kernel. It checks the arguments in the
 * declaration's precedence, walks every tree into its own leaf list (testing a NaN with isnan: the suite is built with
 * -fno-finite-math-only, and spelling each limit), then forms each class's sum on its own, tree by tree in tree order
 * from tree 0's value, and multiplies it by the float32 reciprocal of the tree count. Every add, the reciprocal and
 * every multiply go through volatile operands, so a -ffast-math build of the suite can neither reassociate, contract
 * nor vectorise them: the reference is IEEE float32 arithmetic in that order under the caller's FPSCR. The class is
 * the first maximum by float compares, skipping NaNs, which makes +0 and -0 one value. */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdint.h>

#define TR_MOST_NODES 32767
#define TR_MOST_LEAVES 32768
#define TR_MOST_TREES 65535
#define TR_MOST_FEATURES 32767

static float32_t tr_add(float32_t a, float32_t b)
{
    volatile float32_t x = a;
    volatile float32_t y = b;
    return x + y;
}

static float32_t tr_multiply(float32_t a, float32_t b)
{
    volatile float32_t x = a;
    volatile float32_t y = b;
    return x * y;
}

static float32_t tr_reciprocal(int32_t count)
{
    volatile float32_t one = 1.0f;
    volatile float32_t divisor = (float32_t)count;
    return one / divisor;
}

/* The leaf one tree reaches from `root`, or the status that ends its walk. */
static arm_cmsis_nn_status
tr_walk(const float32_t *input, const cmsis_nn_tree_table_f32 *table, uint32_t root, int32_t *leaf)
{
    const int64_t nodes = table->node_count;
    int64_t at = root;
    if (at >= nodes)
        return ARM_CMSIS_NN_ARG_ERROR;
    for (int64_t visits = 0;; ++visits)
    {
        if (visits == nodes + 1)
            return ARM_CMSIS_NN_FAILURE;
        const cmsis_nn_tree_node_f32 split = table->nodes[at];
        const int64_t f = split.feature & 0x7fff;
        if (f >= table->feature_count)
            return ARM_CMSIS_NN_ARG_ERROR;
        const float32_t x = input[f];
        const int left = isnan(x) ? (split.feature & 0x8000) != 0 : x <= split.threshold;
        const int64_t child = left ? split.left : split.right;
        if (child < 0)
        {
            if (-child - 1 >= table->leaf_count)
                return ARM_CMSIS_NN_ARG_ERROR;
            *leaf = (int32_t)(-child - 1);
            return ARM_CMSIS_NN_SUCCESS;
        }
        if (child >= nodes)
            return ARM_CMSIS_NN_ARG_ERROR;
        at = child;
    }
}

/* `leaves` is the caller's, at least table->tree_count values. */
static arm_cmsis_nn_status tr_reference(const float32_t *input,
                                        const cmsis_nn_tree_table_f32 *table,
                                        const float32_t *leaf_rows,
                                        int32_t class_count,
                                        float32_t *scores,
                                        int32_t *class_index,
                                        int32_t *leaves)
{
    if (input == NULL || leaf_rows == NULL || scores == NULL || class_index == NULL || table == NULL ||
        table->nodes == NULL || table->roots == NULL)
        return ARM_CMSIS_NN_ARG_ERROR;
    const int64_t trees = table->tree_count;
    const int64_t leaf_count = table->leaf_count;
    if (table->node_count < 1 || table->node_count > TR_MOST_NODES || leaf_count < 1 || leaf_count > TR_MOST_LEAVES ||
        trees < 1 || trees > TR_MOST_TREES || table->feature_count < 1 || table->feature_count > TR_MOST_FEATURES)
        return ARM_CMSIS_NN_ARG_ERROR;
    if (class_count < 1 || (int64_t)class_count * leaf_count * 4 > INT32_MAX)
        return ARM_CMSIS_NN_ARG_ERROR;

    for (int64_t t = 0; t < trees; ++t)
    {
        const arm_cmsis_nn_status status = tr_walk(input, table, table->roots[t], &leaves[t]);
        if (status != ARM_CMSIS_NN_SUCCESS)
            return status;
    }

    const float32_t scale = tr_reciprocal((int32_t)trees);
    for (int64_t c = 0; c < class_count; ++c)
    {
        float32_t sum = leaf_rows[(int64_t)leaves[0] * class_count + c];
        for (int64_t t = 1; t < trees; ++t)
            sum = tr_add(sum, leaf_rows[(int64_t)leaves[t] * class_count + c]);
        scores[c] = tr_multiply(sum, scale);
    }

    int32_t best = 0;
    int have = 0;
    float32_t best_value = 0.0f;
    for (int32_t c = 0; c < class_count; ++c)
    {
        const float32_t v = scores[c];
        if (isnan(v))
            continue;
        if (!have || v > best_value)
        {
            best = c;
            best_value = v;
            have = 1;
        }
    }
    *class_index = best;
    return ARM_CMSIS_NN_SUCCESS;
}
