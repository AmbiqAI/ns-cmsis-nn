/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 *
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */

#ifndef ARM_NN_S4_DECODE_H
#define ARM_NN_S4_DECODE_H

#include <stdint.h>

#include "arm_nn_compiler.h"

/* Mask before sign extension: shifting a negative packed byte left is undefined in C. */
__STATIC_FORCEINLINE int8_t arm_nn_s4_low_nibble(int8_t packed)
{
    const int32_t low = (uint8_t)packed & 0x0f;
    return (int8_t)((low ^ 8) - 8);
}

#endif /* ARM_NN_S4_DECODE_H */
