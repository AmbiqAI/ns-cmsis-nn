/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* arm_nn_tree_walk_values_f32 as it was at 61293a84, with every helper it called, copied so that no later change to
 * the library reaches it: from Include/Internal/arm_nn_tree_walk_flt.h, arm_nn_tree_walk_one_f32 (lines 33-81) and
 * arm_nn_tree_table_check_f32 (lines 85-99); and from Source/BasicMathFunctions/arm_nn_tree_walk_values_f32.c, the
 * entry (lines 27-73). Each is renamed with the prefix tpv_ and the entry made static and noinline, so that, as the
 * entry is, it is compiled as a function of its own and not into the cases that call it; nothing else changed. It is
 * compiled only by this suite, as the body the differential cases hold the entry to, word for word. */

#include "arm_nnfunctions.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline arm_cmsis_nn_status tpv_walk_one(const float32_t *input,
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

static inline arm_cmsis_nn_status tpv_table_check(const cmsis_nn_tree_table_f32 *table)
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

__attribute__((noinline)) static arm_cmsis_nn_status tpv_values_f32(const float32_t *input,
                                                                    const cmsis_nn_tree_table_f32 *table,
                                                                    const float32_t *leaf_values,
                                                                    int32_t leaf_value_count,
                                                                    float32_t *values)
{
    if (input == NULL || leaf_values == NULL || values == NULL || tpv_table_check(table) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    /* Every leaf the walk accepts is below leaf_count, so below leaf_value_count: no read past leaf_values. */
    if (leaf_value_count < table->leaf_count)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* The table read once, into locals: a store to values cannot alias them. */
    const cmsis_nn_tree_node_f32 *const nodes = table->nodes;
    const uint32_t *const roots = table->roots;
    const int32_t node_count = table->node_count;
    const int32_t leaf_count = table->leaf_count;
    const int32_t tree_count = table->tree_count;
    const int32_t feature_count = table->feature_count;

    for (int32_t tree = 0; tree < tree_count; ++tree)
    {
        const uint32_t root = roots[tree];
        int32_t leaf;
        uint32_t word;

        if (root >= (uint32_t)node_count)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        const arm_cmsis_nn_status status =
            tpv_walk_one(input, nodes, node_count, leaf_count, feature_count, (int32_t)root, &leaf);
        if (status != ARM_CMSIS_NN_SUCCESS)
        {
            return status;
        }
        /* The leaf's value as a 32-bit word, through an integer: no float operation touches it. */
        memcpy(&word, &leaf_values[leaf], sizeof(word));
        memcpy(&values[tree], &word, sizeof(word));
    }
    return ARM_CMSIS_NN_SUCCESS;
}
