/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The float32 tree walk as it was at 38c6b99e (Source/BasicMathFunctions/arm_nn_tree_walk_f32.c,
 * lines 34-112), tree_walk_one renamed twp_walk_one and the entry renamed twp_tree_walk_f32 and made
 * static, copied so that no later change to the library reaches it. It is compiled only by this
 * suite, as the reference the differential cases hold arm_nn_tree_walk_f32 to: status and every
 * leaf written. It reads its features with the C library's memcpy, so it is included before the
 * suite's counted copy of the library source. */

#include "arm_nnfunctions.h"
#include <stddef.h>
#include <string.h>

/* Walk one tree. The checks and the step budget are the port's; see the header for what each returns. */
static arm_cmsis_nn_status
twp_walk_one(const float32_t *input, const cmsis_nn_tree_table_f32 *table, int32_t index, int32_t *leaf_id)
{
    for (int32_t step = 0; step <= table->node_count; ++step)
    {
        const cmsis_nn_tree_node_f32 *const node = &table->nodes[index];
        const int32_t feature = node->feature & ARM_NN_TREE_NODE_FEATURE_MASK;
        uint32_t bits;
        int32_t go_left;
        int32_t next;

        if (feature >= table->feature_count)
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

            if (leaf >= table->leaf_count)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            *leaf_id = leaf;
            return ARM_CMSIS_NN_SUCCESS;
        }
        if (next >= table->node_count)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
        index = next;
    }
    return ARM_CMSIS_NN_FAILURE;
}

/* The entry, made static. */
static arm_cmsis_nn_status
twp_tree_walk_f32(const float32_t *input, const cmsis_nn_tree_table_f32 *table, int32_t *leaf_ids)
{
    if (input == NULL || table == NULL || leaf_ids == NULL || table->nodes == NULL || table->roots == NULL)
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
    for (int32_t tree = 0; tree < table->tree_count; ++tree)
    {
        if (table->roots[tree] >= (uint32_t)table->node_count)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    for (int32_t tree = 0; tree < table->tree_count; ++tree)
    {
        const arm_cmsis_nn_status status = twp_walk_one(input, table, (int32_t)table->roots[tree], &leaf_ids[tree]);

        if (status != ARM_CMSIS_NN_SUCCESS)
        {
            return status;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
