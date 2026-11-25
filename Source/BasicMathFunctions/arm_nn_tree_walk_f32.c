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
 * $Date:        11 November 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_tree_walk_flt.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

/** @ingroup Reduction */
arm_cmsis_nn_status
arm_nn_tree_walk_f32(const float32_t *input, const cmsis_nn_tree_table_f32 *table, int32_t *leaf_ids)
{
    if (input == NULL || leaf_ids == NULL || arm_nn_tree_table_check_f32(table) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* The table read once, into locals: a leaf store cannot alias them. */
    const cmsis_nn_tree_node_f32 *const nodes = table->nodes;
    const uint32_t *const roots = table->roots;
    const int32_t node_count = table->node_count;
    const int32_t leaf_count = table->leaf_count;
    const int32_t tree_count = table->tree_count;
    const int32_t feature_count = table->feature_count;

    for (int32_t tree = 0; tree < tree_count; ++tree)
    {
        if (roots[tree] >= (uint32_t)node_count)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }
    }
    for (int32_t tree = 0; tree < tree_count; ++tree)
    {
        const arm_cmsis_nn_status status = arm_nn_tree_walk_one_f32(
            input, nodes, node_count, leaf_count, feature_count, (int32_t)roots[tree], &leaf_ids[tree]);

        if (status != ARM_CMSIS_NN_SUCCESS)
        {
            return status;
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif
