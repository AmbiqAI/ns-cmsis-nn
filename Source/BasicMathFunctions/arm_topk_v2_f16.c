/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#include "Internal/arm_nn_topk_flt.h"

#if ARM_NN_ENABLE_F16

/** @ingroup Reduction */
arm_cmsis_nn_status arm_topk_v2_f16(const float16_t *input_data,
                                    const cmsis_nn_dims *input_dims,
                                    int32_t k,
                                    float16_t *output_values,
                                    int32_t *output_indices)
{
    return arm_nn_topk(input_data, input_dims, k, output_values, output_indices, 2);
}

#endif
