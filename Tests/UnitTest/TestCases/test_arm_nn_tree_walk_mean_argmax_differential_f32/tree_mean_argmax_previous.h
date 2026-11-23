/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* arm_nn_tree_walk_mean_argmax_f32 as it was at 61293a84, with every helper it called, copied so that no later change
 * to the library reaches it: from Include/Internal/arm_nn_tree_walk_flt.h, arm_nn_tree_walk_one_f32 (lines 33-81) and
 * arm_nn_tree_table_check_f32 (lines 85-99); from Include/Internal/arm_nn_arg_extrema_flt.h, the 32-bit alias type
 * (line 24), arm_nn_arg_key (lines 34-42) and arm_nn_arg_line_32 (lines 50-65); and from
 * Source/BasicMathFunctions/arm_nn_tree_walk_mean_argmax_f32.c, arm_nn_tree_reach_f32 (lines 28-41) and the entry
 * (lines 44-105). Each is renamed with the prefix tpm_ and the entry made static and noinline, so that, as the entry
 * is, it is compiled as a function of its own and not into the cases that call it; nothing else changed. It is compiled
 * only by this suite, as the body the differential cases hold the entry to, word for word. */

#include "arm_nnfunctions.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef uint32_t tpm_bits32_t __attribute__((__may_alias__));

static inline arm_cmsis_nn_status tpm_walk_one(const float32_t *input,
                                               const cmsis_nn_tree_node_f32 *nodes,
                                               int32_t node_count,
                                               int32_t leaf_count,
                                               int32_t feature_count,
                                               int32_t index,
                                               int32_t *leaf_id)
{
    for (int32_t step = 0; step <= node_count; ++step)
    {
        const cmsis_nn_tree_node_f32 *const node = &nodes[index];
        const int32_t feature = node->feature & ARM_NN_TREE_NODE_FEATURE_MASK;
        uint32_t bits;
        int32_t go_left;
        int32_t next;

        if (feature >= feature_count)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        memcpy(&bits, &input[feature], sizeof(bits));
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
        {
            go_left = (node->feature & ARM_NN_TREE_NODE_MISSING_LEFT) != 0u;
        }
        else
        {
            go_left = input[feature] <= node->threshold;
        }
        next = go_left ? node->left : node->right;
        if (next < 0)
        {
            const int32_t leaf = -next - 1;

            if (leaf >= leaf_count)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            *leaf_id = leaf;
            return ARM_CMSIS_NN_SUCCESS;
        }
        if (next >= node_count)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        index = next;
    }
    return ARM_CMSIS_NN_FAILURE;
}

static inline arm_cmsis_nn_status tpm_table_check(const cmsis_nn_tree_table_f32 *table)
{
    if (table == NULL || table->nodes == NULL || table->roots == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (table->node_count < 1 || table->node_count > ARM_NN_TREE_MAX_NODES || table->leaf_count < 1 ||
        table->leaf_count > ARM_NN_TREE_MAX_LEAVES || table->tree_count < 1 ||
        table->tree_count > ARM_NN_TREE_MAX_TREES || table->feature_count < 1 ||
        table->feature_count > ARM_NN_TREE_MAX_FEATURES)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    return ARM_CMSIS_NN_SUCCESS;
}

static inline uint32_t tpm_arg_key(uint32_t bits, uint32_t sign, uint32_t lose)
{
    const uint32_t magnitude_mask = sign - 1;
    const uint32_t infinity = sign == UINT32_C(0x8000) ? UINT32_C(0x7c00) : UINT32_C(0x7f800000);
    const uint32_t magnitude = bits & magnitude_mask;
    const uint32_t flipped = (bits & sign) ? (~bits & (sign | magnitude_mask)) : (bits | sign);
    const uint32_t key = magnitude == 0 ? sign : flipped;
    return magnitude > infinity ? lose : key;
}

static inline int32_t tpm_arg_line_32(const uint8_t *line, size_t stride, size_t reduction, bool maximum)
{
    const uint32_t sign = UINT32_C(0x80000000);
    const uint32_t lose = maximum ? 0 : UINT32_MAX;
    const tpm_bits32_t *element = (const tpm_bits32_t *)(const void *)line;
    uint32_t best_key = lose;
    int32_t best_index = 0;
    for (size_t k = 0; k < reduction; ++k)
    {
        const uint32_t key = tpm_arg_key(element[k * stride], sign, lose);
        const bool better = maximum ? key > best_key : key < best_key;
        best_key = better ? key : best_key;
        best_index = better ? (int32_t)k : best_index;
    }
    return best_index;
}

static inline arm_cmsis_nn_status tpm_reach(const float32_t *input,
                                            const cmsis_nn_tree_node_f32 *nodes,
                                            int32_t node_count,
                                            int32_t leaf_count,
                                            int32_t feature_count,
                                            uint32_t root,
                                            int32_t *leaf)
{
    if (root >= (uint32_t)node_count)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    return tpm_walk_one(input, nodes, node_count, leaf_count, feature_count, (int32_t)root, leaf);
}

__attribute__((noinline)) static arm_cmsis_nn_status tpm_mean_argmax_f32(const float32_t *input,
                                                                         const cmsis_nn_tree_table_f32 *table,
                                                                         const float32_t *leaf_rows,
                                                                         int32_t class_count,
                                                                         float32_t *scores,
                                                                         int32_t *class_index)
{
    if (input == NULL || leaf_rows == NULL || scores == NULL || class_index == NULL ||
        tpm_table_check(table) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    /* leaf_count x class_count values, at most INT32_MAX bytes: every row offset below fits an int32_t. */
    if (class_count < 1 || class_count > INT32_MAX / (4 * table->leaf_count))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* The table read once, into locals: a store to scores cannot alias them. */
    const cmsis_nn_tree_node_f32 *const nodes = table->nodes;
    const uint32_t *const roots = table->roots;
    const int32_t node_count = table->node_count;
    const int32_t leaf_count = table->leaf_count;
    const int32_t tree_count = table->tree_count;
    const int32_t feature_count = table->feature_count;
    int32_t leaf;

    /* Tree 0's row is the first value of each sum; each later tree's row is added to it. */
    arm_cmsis_nn_status status = tpm_reach(input, nodes, node_count, leaf_count, feature_count, roots[0], &leaf);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return status;
    }
    const float32_t *row = &leaf_rows[leaf * class_count];
    for (int32_t c = 0; c < class_count; ++c)
    {
        scores[c] = row[c];
    }
    for (int32_t tree = 1; tree < tree_count; ++tree)
    {
        status = tpm_reach(input, nodes, node_count, leaf_count, feature_count, roots[tree], &leaf);
        if (status != ARM_CMSIS_NN_SUCCESS)
        {
            return status;
        }
        row = &leaf_rows[leaf * class_count];
        for (int32_t c = 0; c < class_count; ++c)
        {
            scores[c] = scores[c] + row[c];
        }
    }

    /* The mean as each sum times one rounded reciprocal of the tree count. */
    const float32_t scale = 1.0f / (float32_t)tree_count;
    for (int32_t c = 0; c < class_count; ++c)
    {
        scores[c] = scores[c] * scale;
    }
    *class_index = tpm_arg_line_32((const uint8_t *)(const void *)scores, 1, (size_t)class_count, true);
    return ARM_CMSIS_NN_SUCCESS;
}
