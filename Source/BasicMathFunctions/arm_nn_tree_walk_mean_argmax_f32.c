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
 * Title:        arm_nn_tree_walk_mean_argmax_f32
 * Description:  The table walk of a packed float32 tree table with the mean of its leaf rows and their first maximum
 *
 * $Date:        12 November 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_tree_walk_flt.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

/** @ingroup Reduction */
arm_cmsis_nn_status arm_nn_tree_walk_mean_argmax_f32(const float32_t *input,
                                                     const cmsis_nn_tree_table_f32 *table,
                                                     const float32_t *leaf_rows,
                                                     int32_t class_count,
                                                     float32_t *scores,
                                                     int32_t *class_index)
{
    if (input == NULL || leaf_rows == NULL || scores == NULL || class_index == NULL ||
        arm_nn_tree_table_check_f32(table) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    /* The table's fields pass by value: a store to scores cannot alias them. */
    return arm_nn_tree_walk_mean_argmax_body_f32(input,
                                                 table->nodes,
                                                 table->roots,
                                                 table->node_count,
                                                 table->leaf_count,
                                                 table->tree_count,
                                                 table->feature_count,
                                                 leaf_rows,
                                                 class_count,
                                                 scores,
                                                 class_index);
}

#endif
