/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32

/** @ingroup Reduction */
arm_cmsis_nn_status
arm_nn_tree_walk_f32(const float32_t *input, const cmsis_nn_tree_table_f32 *table, int32_t *leaf_ids)
{
    (void)input;
    (void)table;
    (void)leaf_ids;
    return ARM_CMSIS_NN_ARG_ERROR;
}

#endif
