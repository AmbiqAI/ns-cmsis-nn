/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_ACTIVATION_LUT_S16_H
#define ARM_NN_ACTIVATION_LUT_S16_H

#include "arm_nnsupportfunctions.h"

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
/* Each active lane reads table[index] and table[index + 1]. */
__STATIC_FORCEINLINE mve_pred16_t arm_nn_activation_lut_predicate_s16(uint32x4_t index, mve_pred16_t active)
{
    return vcmphiq_m_u32(vdupq_n_u32(255), index, active);
}
#endif

#endif /* ARM_NN_ACTIVATION_LUT_S16_H */
