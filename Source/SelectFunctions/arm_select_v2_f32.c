/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 */

#include "Internal/arm_nn_select_word.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F32
arm_cmsis_nn_status arm_select_v2_f32(const bool *condition,
                                      const float32_t *x,
                                      const float32_t *y,
                                      const cmsis_nn_select_v2_params *params,
                                      float32_t *output)
{
    return arm_nn_select_word(
        condition, (const unsigned char *)x, (const unsigned char *)y, params, (unsigned char *)output);
}
#endif
