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
/* Each active lane reads table[index] and table[index + 1]. Keep 255 as
 * the inactive sentinel, including for indices too large to fit in 16 bits. */
__STATIC_FORCEINLINE mve_pred16_t arm_nn_activation_lut_indices_s16(const uint32x4_t even,
                                                                    const uint32x4_t odd,
                                                                    uint16x8_t *indices)
{
    const uint32x4_t last = vdupq_n_u32(255);
    uint16x8_t packed = vmovnbq_u32(vdupq_n_u16(0), vminq_u32(even, last));
    *indices = vmovntq_u32(packed, vminq_u32(odd, last));
    return vcmphiq_u16(vdupq_n_u16(255), *indices);
}
#endif

#endif /* ARM_NN_ACTIVATION_LUT_S16_H */
