/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

#include "arm_nnfunctions.h"
#if ARM_NN_ENABLE_F32
    #include <stdint.h>

/* Compare byte distances, never add lengths to addresses that could wrap. */
static inline bool distance_overlaps(const void *a, size_t a_size, const void *b, size_t b_size)
{
    const uintptr_t left = (uintptr_t)a, right = (uintptr_t)b;
    return left < right ? right - left < a_size : left - right < b_size;
}
#endif
