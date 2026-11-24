/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 * SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
 * Licensed under the Ambiq Apollo SDK License.
 * See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
 */
#pragma once

#include "arm_nnfunctions.h"
#if ARM_NN_ENABLE_F32
    #include <math.h>
    #include <stdint.h>
    #include <string.h>

/* Floating comparisons against constant FLT_MAX disappear under -Ofast.
 * Only infinities exceed the finite bounds; classify them in the integer domain
 * so arithmetic overflow still saturates and NaNs retain their representation. */
static inline float32_t distance_finite_clamp(float32_t value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* A volatile integer read also breaks fast-math range inference from the
     * preceding arithmetic. A volatile float read still carries finite-only
     * assumptions on Clang, which can erase the infinity test entirely. */
    volatile uint32_t materialized_bits = bits;
    bits = materialized_bits;
    if ((bits & 0x7fffffffU) == 0x7f800000U)
    {
        bits = (bits & 0x80000000U) | 0x7f7fffffU;
        memcpy(&value, &bits, sizeof(value));
    }
    return value;
}

/* Compare byte distances, never add lengths to addresses that could wrap. */
static inline bool distance_overlaps(const void *a, size_t a_size, const void *b, size_t b_size)
{
    const uintptr_t left = (uintptr_t)a, right = (uintptr_t)b;
    return left < right ? right - left < a_size : left - right < b_size;
}

static inline arm_cmsis_nn_status
distance_f32(const float32_t *input, const float32_t *references, int32_t rows, int32_t features, float32_t *output)
{
    if (!input || !references || !output || rows <= 0 || features <= 0 ||
        rows > (INT32_MAX / (int32_t)sizeof(float32_t)) / features)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const size_t x_bytes = (size_t)features * sizeof(float32_t);
    const size_t out_bytes = (size_t)rows * sizeof(float32_t);
    if (distance_overlaps(input, x_bytes, output, out_bytes) ||
        distance_overlaps(references, (size_t)rows * x_bytes, output, out_bytes))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    for (int32_t row = 0; row < rows; ++row)
    {
        /* Preserve feature order even under the library's -Ofast builds. The
         * two finite clamps reproduce the SUB and MUL node boundaries. */
        volatile float32_t sum = 0.0f;
        for (int32_t col = 0; col < features; ++col)
        {
            const float32_t delta = distance_finite_clamp(input[col] - references[col]);
            const float32_t term = distance_finite_clamp(delta * delta);
            sum += term;
        }
        output[row] = sum;
        references += features;
    }
    return ARM_CMSIS_NN_SUCCESS;
}
#endif
