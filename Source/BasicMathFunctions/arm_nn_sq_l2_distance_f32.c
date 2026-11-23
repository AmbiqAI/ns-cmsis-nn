/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#include "Internal/arm_nn_distance_f32.h"

#if ARM_NN_ENABLE_F32
arm_cmsis_nn_status arm_nn_sq_l2_distance_f32(const float32_t *input,
                                              const float32_t *references,
                                              int32_t rows,
                                              int32_t features,
                                              float32_t *output)
{
    return distance_f32(input, references, rows, features, output, true);
}
#endif
