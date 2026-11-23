/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

/* The portable reference of arm_nn_tree_walk_values_f32: what its declaration states, in plain C, with no
 * intrinsics, no library header beyond the types and no code shared with the kernel. It checks the arguments in the
 * declaration's precedence (each limit spelled here), then walks the trees in order, each from its own root check,
 * over 64-bit indices, testing a NaN with isnan (the suite is built with -fno-finite-math-only), and copies each
 * reached leaf's value as bytes. It never forms a float from a leaf value. */

#include <arm_nnfunctions.h>
#include <math.h>
#include <stdint.h>

#define TVR_MOST_NODES 32767
#define TVR_MOST_LEAVES 32768
#define TVR_MOST_TREES 65535
#define TVR_MOST_FEATURES 32767

/* The leaf one tree reaches from `root`, or the status that ends its walk. */
static arm_cmsis_nn_status
tvr_walk(const float32_t *input, const cmsis_nn_tree_table_f32 *table, uint32_t root, int64_t *leaf)
{
    const int64_t splits = table->node_count;
    int64_t at = root;
    if (at >= splits)
        return ARM_CMSIS_NN_ARG_ERROR;
    for (int64_t visited = 0; visited <= splits; ++visited)
    {
        const cmsis_nn_tree_node_f32 split = table->nodes[at];
        const int64_t f = split.feature & 0x7fff;
        if (f >= table->feature_count)
            return ARM_CMSIS_NN_ARG_ERROR;
        const float32_t x = input[f];
        const int64_t child =
            (isnan(x) ? (split.feature & 0x8000) != 0 : x <= split.threshold) ? split.left : split.right;
        if (child < 0)
        {
            if (-child - 1 >= table->leaf_count)
                return ARM_CMSIS_NN_ARG_ERROR;
            *leaf = -child - 1;
            return ARM_CMSIS_NN_SUCCESS;
        }
        if (child >= splits)
            return ARM_CMSIS_NN_ARG_ERROR;
        at = child;
    }
    return ARM_CMSIS_NN_FAILURE;
}

static arm_cmsis_nn_status tvr_reference(const float32_t *input,
                                         const cmsis_nn_tree_table_f32 *table,
                                         const float32_t *leaf_values,
                                         int32_t leaf_value_count,
                                         float32_t *values)
{
    if (input == NULL || leaf_values == NULL || values == NULL || table == NULL || table->nodes == NULL ||
        table->roots == NULL)
        return ARM_CMSIS_NN_ARG_ERROR;
    if (table->node_count < 1 || table->node_count > TVR_MOST_NODES || table->leaf_count < 1 ||
        table->leaf_count > TVR_MOST_LEAVES || table->tree_count < 1 || table->tree_count > TVR_MOST_TREES ||
        table->feature_count < 1 || table->feature_count > TVR_MOST_FEATURES)
        return ARM_CMSIS_NN_ARG_ERROR;
    if ((int64_t)leaf_value_count < (int64_t)table->leaf_count)
        return ARM_CMSIS_NN_ARG_ERROR;

    const uint8_t *const from = (const uint8_t *)(const void *)leaf_values;
    uint8_t *const to = (uint8_t *)(void *)values;
    for (int64_t t = 0; t < table->tree_count; ++t)
    {
        int64_t leaf = -1;
        const arm_cmsis_nn_status status = tvr_walk(input, table, table->roots[t], &leaf);
        if (status != ARM_CMSIS_NN_SUCCESS)
            return status;
        for (int64_t b = 0; b < 4; ++b)
            to[4 * t + b] = from[4 * leaf + b];
    }
    return ARM_CMSIS_NN_SUCCESS;
}
