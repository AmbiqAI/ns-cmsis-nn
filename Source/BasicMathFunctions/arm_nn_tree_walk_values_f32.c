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
 * Title:        arm_nn_tree_walk_values_f32
 * Description:  The table walk of a packed float32 tree table with the leaf value each tree reaches
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
arm_cmsis_nn_status arm_nn_tree_walk_values_f32(const float32_t *input,
                                                const cmsis_nn_tree_table_f32 *table,
                                                const float32_t *leaf_values,
                                                int32_t leaf_value_count,
                                                float32_t *values)
{
    if (input == NULL || leaf_values == NULL || values == NULL ||
        arm_nn_tree_table_check_f32(table) != ARM_CMSIS_NN_SUCCESS)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    /* The table's fields pass by value: a store to values cannot alias them. */
    return arm_nn_tree_walk_values_body_f32(input,
                                            table->nodes,
                                            table->roots,
                                            table->node_count,
                                            table->leaf_count,
                                            table->tree_count,
                                            table->feature_count,
                                            leaf_values,
                                            leaf_value_count,
                                            values);
}

#endif
