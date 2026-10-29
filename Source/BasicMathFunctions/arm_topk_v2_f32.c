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
arm_cmsis_nn_status arm_topk_v2_f32(const float32_t *input_data,
                                    const cmsis_nn_dims *input_dims,
                                    int32_t k,
                                    float32_t *output_values,
                                    int32_t *output_indices)
{
    (void)input_data;
    (void)input_dims;
    (void)k;
    (void)output_values;
    (void)output_indices;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
}

#endif
