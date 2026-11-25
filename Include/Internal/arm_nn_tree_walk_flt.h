/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_TREE_WALK_FLT_H
#define ARM_NN_TREE_WALK_FLT_H

#include "Internal/arm_nn_arg_extrema_flt.h"
#include "arm_nnfunctions.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if ARM_NN_ENABLE_F32

/* A writer packs into this record, so its layout is the ABI the header states. */
ARM_NN_STATIC_ASSERT(sizeof(cmsis_nn_tree_node_f32) == 12, tree_node_is_12_bytes);
ARM_NN_STATIC_ASSERT(offsetof(cmsis_nn_tree_node_f32, feature) == 4, tree_node_feature_at_4);
ARM_NN_STATIC_ASSERT(offsetof(cmsis_nn_tree_node_f32, left) == 6, tree_node_left_at_6);
ARM_NN_STATIC_ASSERT(offsetof(cmsis_nn_tree_node_f32, right) == 8, tree_node_right_at_8);

/* Walk one tree of a packed table from split `index`, which the caller has checked is below
 * node_count; the leaf it reaches goes to *leaf_id, which is written only on success. The checks
 * and the step budget are the port's, and arm_nn_tree_walk_f32's header states what each returns.
 * The counts come by value, so a caller that holds them in locals keeps them in registers across
 * the store of a leaf, which may otherwise alias the table's int32_t fields and force their
 * reload. Each visited split reads its feature through one memcpy: the tree-walk suite counts
 * those calls to observe the step budget. */
static inline arm_cmsis_nn_status arm_nn_tree_walk_one_f32(const float32_t *input,
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

/* The count checks every entry over a packed table makes before it reads the table: a NULL
 * table, nodes or roots, or a count outside its range, is ARM_CMSIS_NN_ARG_ERROR. */
static inline arm_cmsis_nn_status arm_nn_tree_table_check_f32(const cmsis_nn_tree_table_f32 *table)
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

/* The leaf tree `root` reaches, after the root check the walk itself leaves to its caller. */
static inline arm_cmsis_nn_status arm_nn_tree_reach_f32(const float32_t *input,
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
    return arm_nn_tree_walk_one_f32(input, nodes, node_count, leaf_count, feature_count, (int32_t)root, leaf);
}

/* The bodies of the two fused entries, arm_nn_tree_walk_mean_argmax_f32 and arm_nn_tree_walk_values_f32, which call
 * them after their own checks. Each takes the table as its arrays and its counts by value, so generated code that
 * includes this header can call it from its own translation unit with its counts as literals, and the compiler then
 * specialises the loops to them. Each computes what its entry's declaration states, in the order it states: the
 * entry and the body are one code.
 *
 * Checks. The body makes every check of its entry that depends on the table's contents or on its own count argument:
 * first its count against leaf_count (class_count < 1 or class_count > INT32_MAX / (4 * leaf_count) for the mean;
 * leaf_value_count < leaf_count for the values), ARM_CMSIS_NN_ARG_ERROR before any read or write; then, for each tree
 * ascending, its root against node_count and the walk's checks on the path the row takes (a feature >= feature_count,
 * a split child >= node_count, a leaf >= leaf_count: ARM_CMSIS_NN_ARG_ERROR; a walk past the step budget, a cycle:
 * ARM_CMSIS_NN_FAILURE). Statuses, their precedence and what each leaves written are the entry's.
 *
 * The caller's part. The entry checks these before it calls the body, and a caller of the body holds them itself:
 * no pointer is NULL; node_count, leaf_count, tree_count and feature_count are in arm_nn_tree_walk_f32's ranges (what
 * arm_nn_tree_table_check_f32 checks: a leaf_count of at least 1 keeps the bound above from dividing by zero, and a
 * tree_count of at least 1 lets the mean read roots[0]); each array holds the values its counts state (input
 * feature_count, nodes node_count, roots tree_count, leaf_rows leaf_count x class_count, scores class_count,
 * leaf_values leaf_value_count, values tree_count); and the outputs overlap nothing, as the declarations state.
 * Held, these leave no table a way to make the body read or write outside those arrays: a root, a feature, a child or
 * a leaf out of range on the path, and a cycle, are refused before the read they would cause. */

/* The mean of the leaf rows the trees reach and its first maximum, as arm_nn_tree_walk_mean_argmax_f32 states. */
static inline arm_cmsis_nn_status arm_nn_tree_walk_mean_argmax_body_f32(const float32_t *input,
                                                                        const cmsis_nn_tree_node_f32 *nodes,
                                                                        const uint32_t *roots,
                                                                        int32_t node_count,
                                                                        int32_t leaf_count,
                                                                        int32_t tree_count,
                                                                        int32_t feature_count,
                                                                        const float32_t *leaf_rows,
                                                                        int32_t class_count,
                                                                        float32_t *scores,
                                                                        int32_t *class_index)
{
    /* leaf_count x class_count values, at most INT32_MAX bytes: every row offset below fits an int32_t. */
    if (class_count < 1 || class_count > INT32_MAX / (4 * leaf_count))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    int32_t leaf;

    /* Tree 0's row is the first value of each sum; each later tree's row is added to it. */
    arm_cmsis_nn_status status =
        arm_nn_tree_reach_f32(input, nodes, node_count, leaf_count, feature_count, roots[0], &leaf);
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
        status = arm_nn_tree_reach_f32(input, nodes, node_count, leaf_count, feature_count, roots[tree], &leaf);
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
    *class_index = arm_nn_arg_line_32((const uint8_t *)(const void *)scores, 1, (size_t)class_count, true);
    return ARM_CMSIS_NN_SUCCESS;
}

/* The value of the leaf each tree reaches, as arm_nn_tree_walk_values_f32 states. */
static inline arm_cmsis_nn_status arm_nn_tree_walk_values_body_f32(const float32_t *input,
                                                                   const cmsis_nn_tree_node_f32 *nodes,
                                                                   const uint32_t *roots,
                                                                   int32_t node_count,
                                                                   int32_t leaf_count,
                                                                   int32_t tree_count,
                                                                   int32_t feature_count,
                                                                   const float32_t *leaf_values,
                                                                   int32_t leaf_value_count,
                                                                   float32_t *values)
{
    /* Every leaf the walk accepts is below leaf_count, so below leaf_value_count: no read past leaf_values. */
    if (leaf_value_count < leaf_count)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

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
            arm_nn_tree_walk_one_f32(input, nodes, node_count, leaf_count, feature_count, (int32_t)root, &leaf);
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

#endif /* ARM_NN_ENABLE_F32 */
#endif /* ARM_NN_TREE_WALK_FLT_H */
