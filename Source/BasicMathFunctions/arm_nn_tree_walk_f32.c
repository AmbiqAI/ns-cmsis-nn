/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_tree_walk_f32
 * Description:  The table walk of a packed float32 tree table
 *
 * $Date:        4 November 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

    #include <stddef.h>
    #include <string.h>

/* A writer packs into this record, so its layout is the ABI the header states. */
ARM_NN_STATIC_ASSERT(sizeof(cmsis_nn_tree_node_f32) == 12, tree_node_is_12_bytes);
ARM_NN_STATIC_ASSERT(offsetof(cmsis_nn_tree_node_f32, feature) == 4, tree_node_feature_at_4);
ARM_NN_STATIC_ASSERT(offsetof(cmsis_nn_tree_node_f32, left) == 6, tree_node_left_at_6);
ARM_NN_STATIC_ASSERT(offsetof(cmsis_nn_tree_node_f32, right) == 8, tree_node_right_at_8);

/* Walk one tree. The checks and the step budget are the port's; see the header for what each returns. */
static arm_cmsis_nn_status
tree_walk_one(const float32_t *input, const cmsis_nn_tree_table_f32 *table, int32_t index, int32_t *leaf_id)
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

/** @ingroup Reduction */
arm_cmsis_nn_status
arm_nn_tree_walk_f32(const float32_t *input, const cmsis_nn_tree_table_f32 *table, int32_t *leaf_ids)
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
        const arm_cmsis_nn_status status = tree_walk_one(input, table, (int32_t)table->roots[tree], &leaf_ids[tree]);

        if (status != ARM_CMSIS_NN_SUCCESS)
        {
            return status;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif
